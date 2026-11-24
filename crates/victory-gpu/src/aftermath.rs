//! NVIDIA Nsight Aftermath GPU crash dumps (diagnostics builds,
//! `VICTORY_AFTERMATH`): when the host GPU hangs or faults, the driver's
//! dump names the shaders that were running, their warps and where each
//! stood, which device-removed data cannot.
//!
//! The SDK's library is loaded at run time from the path the variable
//! names (the DLL or its folder); the declarations here follow the SDK's
//! headers (`GFSDK_Aftermath*.h`, API version 2.27). Dumps, the shader
//! debug information the driver hands over and the decoded dumps go to
//! `VICTORY_AFTERMATH_DIR` (`aftermath` under the current folder).
//!
//! Crash dumps are enabled before the host device is made ([`enable`]),
//! the device's features after ([`initialize`]). Host pipelines record
//! their shaders by the SDK's hash ([`shaders`]), so the decoder gets their
//! DXIL and a dump's shaders are named by our hashes. On device removal
//! [`report`] waits for the dump, decodes it and sums it up.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_SHADER_BYTECODE, ID3D12Device, ID3D12GraphicsCommandList,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{HSTRING, Interface, PCSTR};

use crate::pipeline::Shader;

/// `GFSDK_Aftermath_Version_API` of the headers followed here; the library
/// refuses any other.
const VERSION: u32 = 0x21b;
const NOT_AVAILABLE: u32 = 2;

/// `GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_DX`.
const WATCH_DX: u32 = 1;
/// `GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks`: the
/// library keeps every shader's debug information and hands over only
/// those of a dump's shaders, when it is made.
const DEFER_DEBUG_INFO: u32 = 1;
/// `GFSDK_Aftermath_FeatureFlags`: markers, resource tracking, call stacks
/// (only with the Aftermath monitor running), shader debug information,
/// shader error reporting.
const FEATURES: u32 = 0x1 | 0x2 | 0x4000_0000 | 0x8 | SHADER_ERRORS;
/// `GFSDK_Aftermath_FeatureFlags_EnableShaderErrorReporting`.
const SHADER_ERRORS: u32 = 0x10;
/// `GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO`.
const ALL_INFO: u32 = 0x3fff;
/// `GFSDK_Aftermath_GpuCrashDumpFormatterFlags_UTF8_OUTPUT`.
const UTF8: u32 = 2;
/// `GFSDK_Aftermath_GpuCrashDumpDescriptionKey`s.
const APPLICATION_NAME: u32 = 1;
const APPLICATION_VERSION: u32 = 2;

/// `GFSDK_Aftermath_CrashDump_Status`, by value.
const DUMP_STATUS: [&str; 6] = [
    "not started",
    "collecting data",
    "collecting data failed",
    "invoking callback",
    "finished",
    "unknown",
];
const DUMP_FAILED: u32 = 2;
const DUMP_FINISHED: u32 = 4;
/// How long a removed device waits for the dump.
const DUMP_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Whether a result is not a failure (`GFSDK_Aftermath_SUCCEED`).
fn succeeded(result: u32) -> bool {
    result & 0xfff0_0000 != 0xbad0_0000
}

/// A result for the log: its name in the headers and its value.
fn result_name(result: u32) -> String {
    const FAILURES: [&str; 24] = [
        "Fail",
        "VersionMismatch",
        "NotInitialized",
        "InvalidAdapter",
        "InvalidParameter",
        "Unknown",
        "ApiError",
        "NvApiIncompatible",
        "GettingContextDataWithNewCommandList",
        "AlreadyInitialized",
        "D3DDebugLayerNotCompatible",
        "DriverInitFailed",
        "DriverVersionNotSupported",
        "OutOfMemory",
        "GetDataOnBundle",
        "GetDataOnDeferredContext",
        "FeatureNotEnabled",
        "NoResourcesRegistered",
        "ThisResourceNeverRegistered",
        "NotSupportedInUWP",
        "D3dDllNotSupported",
        "D3dDllInterceptionNotSupported",
        "Disabled",
        "NotSupportedOnContext",
    ];
    let name = match result {
        1 => Some("Success"),
        NOT_AVAILABLE => Some("NotAvailable"),
        _ if !succeeded(result) => FAILURES.get((result & 0xf_ffff) as usize).copied(),
        _ => None,
    };
    format!("{} ({result:#x})", name.unwrap_or("?"))
}

/// A context or decoder handle.
type Handle = *mut c_void;
/// The crash dump and shader debug information callbacks: data, bytes,
/// user data.
type DataCb = unsafe extern "C" fn(*const c_void, u32, *mut c_void);
type AddDescription = unsafe extern "C" fn(u32, *const c_char);
type DescriptionCb = unsafe extern "C" fn(AddDescription, *mut c_void);
type ResolveMarkerCb = unsafe extern "C" fn(*const c_void, u32, *mut c_void, *const c_void);
/// How a lookup hands the decoder its data.
type SetData = unsafe extern "C" fn(*const c_void, u32);
type DebugInfoLookupCb = unsafe extern "C" fn(*const [u64; 2], SetData, *mut c_void);
type ShaderLookupCb = unsafe extern "C" fn(*const u64, SetData, *mut c_void);
type SourceLookupCb = unsafe extern "C" fn(*const c_void, SetData, *mut c_void);

/// `GFSDK_Aftermath_GpuCrashDump_DeviceInfo` (booleans in 32-bit slots).
#[repr(C)]
#[derive(Default)]
struct DeviceInfo {
    status: u32,
    adapter_reset: u32,
    engine_reset: u32,
}

/// `GFSDK_Aftermath_GpuCrashDump_PageFaultInfo`.
#[repr(C)]
#[derive(Default)]
struct PageFaultInfo {
    va: u64,
    fault: u32,
    access: u32,
    engine: u32,
    client: u32,
    resources: u32,
}

/// `GFSDK_Aftermath_GpuCrashDump_ShaderInfo`.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct ShaderInfo {
    hash: u64,
    debug_info: u64,
    internal: u32,
    kind: u32,
}

macro_rules! api {
    ($($field:ident: $name:literal fn($($arg:ty),*);)*) => {
        /// The library's entry points used here.
        struct Api {
            $($field: unsafe extern "C" fn($($arg),*) -> u32,)*
        }

        impl Api {
            /// The entry points of the loaded library, or the name of the
            /// first one it lacks.
            fn find(module: HMODULE) -> Result<Api, &'static str> {
                Ok(Api {
                    $($field: {
                        // SAFETY: a plain lookup by a NUL-terminated name.
                        let found = unsafe {
                            GetProcAddress(module, PCSTR(concat!($name, "\0").as_ptr()))
                        }
                        .ok_or($name)?;
                        // SAFETY: the export has the signature the SDK's
                        // headers declare (on x64, the C convention).
                        unsafe {
                            std::mem::transmute::<
                                unsafe extern "system" fn() -> isize,
                                unsafe extern "C" fn($($arg),*) -> u32,
                            >(found)
                        }
                    },)*
                })
            }
        }
    };
}

api! {
    enable_crash_dumps: "GFSDK_Aftermath_EnableGpuCrashDumps" fn(
        u32, u32, u32, Option<DataCb>, Option<DataCb>, Option<DescriptionCb>,
        Option<ResolveMarkerCb>, *mut c_void
    );
    crash_dump_status: "GFSDK_Aftermath_GetCrashDumpStatus" fn(*mut u32);
    dx12_initialize: "GFSDK_Aftermath_DX12_Initialize" fn(u32, u32, *mut c_void);
    create_context: "GFSDK_Aftermath_DX12_CreateContextHandle" fn(*mut c_void, *mut Handle);
    release_context: "GFSDK_Aftermath_ReleaseContextHandle" fn(Handle);
    set_event_marker: "GFSDK_Aftermath_SetEventMarker" fn(Handle, *const c_void, u32);
    shader_hash: "GFSDK_Aftermath_GetShaderHash" fn(u32, *const D3D12_SHADER_BYTECODE, *mut u64);
    debug_info_id: "GFSDK_Aftermath_GetShaderDebugInfoIdentifier" fn(
        u32, *const c_void, u32, *mut [u64; 2]
    );
    create_decoder: "GFSDK_Aftermath_GpuCrashDump_CreateDecoder" fn(
        u32, *const c_void, u32, *mut Handle
    );
    destroy_decoder: "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder" fn(Handle);
    device_info: "GFSDK_Aftermath_GpuCrashDump_GetDeviceInfo" fn(Handle, *mut DeviceInfo);
    page_fault_info: "GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo" fn(Handle, *mut PageFaultInfo);
    active_shader_count: "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount" fn(Handle, *mut u32);
    active_shaders: "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo" fn(
        Handle, u32, *mut ShaderInfo
    );
    shader_info_hash: "GFSDK_Aftermath_GetShaderHashForShaderInfo" fn(
        Handle, *const ShaderInfo, *mut u64
    );
    generate_json: "GFSDK_Aftermath_GpuCrashDump_GenerateJSON" fn(
        Handle, u32, u32, Option<DebugInfoLookupCb>, Option<ShaderLookupCb>,
        Option<SourceLookupCb>, *mut c_void, *mut u32
    );
    json: "GFSDK_Aftermath_GpuCrashDump_GetJSON" fn(Handle, u32, *mut c_char);
}

struct Aftermath {
    api: Api,
    /// Where dumps and shader debug information go.
    dir: PathBuf,
    /// Whether the host device's features are on.
    device: AtomicBool,
    /// The last crash dump and its file.
    dump: Mutex<Option<(PathBuf, Vec<u8>)>>,
    /// Shader debug information the driver handed over, by identifier.
    debug_info: Mutex<HashMap<[u64; 2], Vec<u8>>>,
    shaders: Mutex<Shaders>,
}

/// Host pipeline shaders by the SDK's hash of their code, and those hashed
/// already (by address; recorded shaders live as long as the map).
#[derive(Default)]
struct Shaders {
    by_hash: HashMap<u64, Arc<Shader>>,
    seen: HashSet<usize>,
}

fn get() -> Option<&'static Aftermath> {
    static AFTERMATH: OnceLock<Option<Aftermath>> = OnceLock::new();
    AFTERMATH.get_or_init(load).as_ref()
}

/// The library `VICTORY_AFTERMATH` names, with crash dumps enabled.
fn load() -> Option<Aftermath> {
    let named = PathBuf::from(std::env::var_os("VICTORY_AFTERMATH")?);
    let dll = if named.is_dir() {
        named.join("GFSDK_Aftermath_Lib.x64.dll")
    } else {
        named
    };
    let dir =
        std::env::var_os("VICTORY_AFTERMATH_DIR").map_or_else(|| "aftermath".into(), PathBuf::from);
    let dir = std::path::absolute(&dir).unwrap_or(dir);
    // SAFETY: loading the SDK's library runs only its own initialisation.
    let module = match unsafe { LoadLibraryW(&HSTRING::from(dll.as_os_str())) } {
        Ok(module) => module,
        Err(e) => {
            tracing::warn!(dll = %dll.display(), error = %e, "no Aftermath library; GPU crash dumps off");
            return None;
        }
    };
    let api = match Api::find(module) {
        Ok(api) => api,
        Err(name) => {
            tracing::warn!(dll = %dll.display(), "the Aftermath library lacks {name}; GPU crash dumps off");
            return None;
        }
    };
    // SAFETY: the callbacks are thread-safe and live for the whole run.
    let result = unsafe {
        (api.enable_crash_dumps)(
            VERSION,
            WATCH_DX,
            DEFER_DEBUG_INFO,
            Some(on_dump),
            Some(on_debug_info),
            Some(on_description),
            None,
            std::ptr::null_mut(),
        )
    };
    if !succeeded(result) {
        tracing::warn!(result = %result_name(result), "Aftermath refused GPU crash dumps");
        return None;
    }
    tracing::info!(dll = %dll.display(), dir = %dir.display(), "Aftermath GPU crash dumps on");
    Some(Aftermath {
        api,
        dir,
        device: AtomicBool::new(false),
        dump: Mutex::new(None),
        debug_info: Mutex::default(),
        shaders: Mutex::default(),
    })
}

/// With `VICTORY_AFTERMATH`: enables GPU crash dumps. Before the host
/// device is made (the driver watches only devices made after).
pub(crate) fn enable() {
    get();
}

/// Turns Aftermath's features on for the host device, just made. The SDK
/// supports the D3D12 debug layer alongside (its old result for a debug
/// layer, `FAIL_D3DDebugLayerNotCompatible`, is no longer returned).
pub(crate) fn initialize(device: &ID3D12Device, debug_layer: bool) {
    let Some(a) = get() else { return };
    if debug_layer {
        tracing::info!("Aftermath with the D3D12 debug layer (the SDK supports both together)");
    }
    // SAFETY: a live device.
    // `VICTORY_AFTERMATH_ERRORS=0`: without shader error reporting, so only
    // real hangs and page faults end the run (the reporting also stops it at
    // shader errors the driver otherwise lets pass).
    let features = if std::env::var("VICTORY_AFTERMATH_ERRORS").is_ok_and(|v| v == "0") {
        FEATURES & !SHADER_ERRORS
    } else {
        FEATURES
    };
    let result = unsafe { (a.api.dx12_initialize)(VERSION, features, device.as_raw()) };
    if succeeded(result) {
        a.device.store(true, Ordering::Relaxed);
        tracing::info!(
            "Aftermath on the host device: markers, resource tracking, call stacks (with the \
             Aftermath monitor running), shader debug information, shader error reporting"
        );
    } else {
        tracing::warn!(
            result = %result_name(result),
            "Aftermath refused the host device; dumps keep only basic information"
        );
    }
}

/// Whether Aftermath's features are on for the host device.
pub(crate) fn on() -> bool {
    get().is_some_and(|a| a.device.load(Ordering::Relaxed))
}

/// Records the shaders of a host pipeline by the SDK's hash of their code,
/// once each, so a dump's shaders are named by our hashes and the decoder
/// gets their DXIL.
pub(crate) fn shaders<'a>(shaders: impl IntoIterator<Item = &'a Arc<Shader>>) {
    let Some(a) = get() else { return };
    let mut known = a.shaders.lock().unwrap_or_else(|e| e.into_inner());
    for shader in shaders {
        if !known.seen.insert(Arc::as_ptr(shader) as usize) {
            continue;
        }
        if let Some(hash) = shader.translated().and_then(|t| a.hash(&t.host.dxil)) {
            known.by_hash.entry(hash).or_insert_with(|| shader.clone());
        }
    }
}

impl Aftermath {
    /// The SDK's hash of DXIL or DXBC code.
    fn hash(&self, code: &[u8]) -> Option<u64> {
        let code = D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        };
        let mut hash = 0;
        // SAFETY: the code lives through the call; the output is ours.
        let result = unsafe { (self.api.shader_hash)(VERSION, &code, &mut hash) };
        succeeded(result).then_some(hash)
    }
}

/// A command list's Aftermath context, for event markers; released with
/// the list.
pub(crate) struct Marks(Handle);

// SAFETY: the SDK allows markers from any thread (D3D12's own rules for the
// list apply, which its owner keeps).
unsafe impl Send for Marks {}
// SAFETY: as above.
unsafe impl Sync for Marks {}

impl Marks {
    /// For an open list, when Aftermath is on the host device.
    pub(crate) fn new(list: &ID3D12GraphicsCommandList) -> Option<Marks> {
        let a = get().filter(|a| a.device.load(Ordering::Relaxed))?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: an open list; the output is ours.
        let result = unsafe { (a.api.create_context)(list.as_raw(), &mut handle) };
        if !succeeded(result) || handle.is_null() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(result = %result_name(result), "no Aftermath markers (logged once)")
            });
            return None;
        }
        Some(Marks(handle))
    }

    /// Marks the command about to be recorded (the library copies the text).
    pub(crate) fn mark(&self, text: &str) {
        let Some(a) = get() else { return };
        let text: Vec<u8> = text.bytes().chain([0]).collect();
        // SAFETY: our context on an open list; the text lives through the call.
        unsafe { (a.api.set_event_marker)(self.0, text.as_ptr().cast(), text.len() as u32) };
    }
}

impl Drop for Marks {
    fn drop(&mut self) {
        if let Some(a) = get() {
            // SAFETY: our context, released once.
            unsafe { (a.api.release_context)(self.0) };
        }
    }
}

/// The driver's crash dump: written at once, kept for [`report`].
unsafe extern "C" fn on_dump(data: *const c_void, size: u32, _: *mut c_void) {
    let Some(a) = get() else { return };
    // SAFETY: the library passes `size` bytes, valid during the call.
    let dump = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) }.to_vec();
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let file = a.dir.join(format!(
        "victory-{seconds}-{}-{}.nv-gpudmp",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::Relaxed) + 1
    ));
    match std::fs::create_dir_all(&a.dir).and_then(|()| std::fs::write(&file, &dump)) {
        Ok(()) => {
            tracing::info!(file = %file.display(), bytes = dump.len(), "Aftermath GPU crash dump")
        }
        Err(e) => {
            tracing::warn!(file = %file.display(), error = %e, "Aftermath GPU crash dump not written")
        }
    }
    *a.dump.lock().unwrap_or_else(|e| e.into_inner()) = Some((file, dump));
}

/// Debug information of a dump's shader (line tables from DXIL to the
/// GPU's code): written beside the dump as Nsight Graphics names it, kept
/// for decoding.
unsafe extern "C" fn on_debug_info(data: *const c_void, size: u32, _: *mut c_void) {
    let Some(a) = get() else { return };
    let mut id = [0u64; 2];
    // SAFETY: the library's own data, valid during the call; the output is ours.
    let result = unsafe { (a.api.debug_info_id)(VERSION, data, size, &mut id) };
    if !succeeded(result) {
        tracing::warn!(result = %result_name(result), "Aftermath shader debug information without an identifier");
        return;
    }
    // SAFETY: as above.
    let info = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) }.to_vec();
    let file = a
        .dir
        .join(format!("shader-{:016x}-{:016x}.nvdbg", id[0], id[1]));
    if let Err(e) = std::fs::create_dir_all(&a.dir).and_then(|()| std::fs::write(&file, &info)) {
        tracing::warn!(file = %file.display(), error = %e, "Aftermath shader debug information not written");
    }
    a.debug_info
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, info);
}

/// Names the application in the dump.
unsafe extern "C" fn on_description(add: AddDescription, _: *mut c_void) {
    // SAFETY: NUL-terminated texts; the library copies them.
    unsafe {
        add(APPLICATION_NAME, c"victory".as_ptr());
        add(
            APPLICATION_VERSION,
            concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast(),
        );
    }
}

/// On device removal: waits (up to 10 s) for the driver's crash dump,
/// decodes it to JSON beside the file and logs a summary, which it returns
/// for the failure report. Once per run; `None` with Aftermath off.
pub(crate) fn report() -> Option<String> {
    static REPORT: OnceLock<Option<String>> = OnceLock::new();
    REPORT.get_or_init(|| Some(get()?.report())).clone()
}

impl Aftermath {
    fn report(&self) -> String {
        let status = self.wait();
        let status = DUMP_STATUS.get(status as usize).copied().unwrap_or("?");
        let dump = self.dump.lock().unwrap_or_else(|e| e.into_inner()).take();
        let text = match dump {
            None => format!("no GPU crash dump (status {status})"),
            Some((file, dump)) => {
                let debug_info = self.debug_info.lock().unwrap_or_else(|e| e.into_inner());
                let shaders = self.shaders.lock().unwrap_or_else(|e| e.into_inner());
                match decode(&self.api, &dump, &debug_info, &shaders.by_hash) {
                    Ok((json, summary)) => {
                        let decoded = file.with_extension("nv-gpudmp.json");
                        if let Err(e) = std::fs::write(&decoded, json) {
                            tracing::warn!(file = %decoded.display(), error = %e, "decoded Aftermath dump not written");
                        }
                        format!("{summary}; dump {}", file.display())
                    }
                    Err(e) => format!("dump {} not decoded: {e}", file.display()),
                }
            }
        };
        tracing::error!("Aftermath: {text}");
        text
    }

    /// Polls the dump's status until it is finished or failed, or for
    /// [`DUMP_WAIT`]: the driver reports a removed device before it has
    /// collected the dump.
    fn wait(&self) -> u32 {
        let started = std::time::Instant::now();
        loop {
            let mut status = 5;
            // SAFETY: the output is ours.
            unsafe { (self.api.crash_dump_status)(&mut status) };
            if matches!(status, DUMP_FINISHED | DUMP_FAILED) || started.elapsed() > DUMP_WAIT {
                return status;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

/// What the decoder's lookups serve: shader debug information by
/// identifier and our shaders' DXIL by the SDK's hash.
struct Lookup<'a> {
    debug_info: &'a HashMap<[u64; 2], Vec<u8>>,
    shaders: &'a HashMap<u64, Arc<Shader>>,
}

unsafe extern "C" fn find_debug_info(id: *const [u64; 2], set: SetData, user: *mut c_void) {
    // SAFETY: `user` is the `Lookup` `decode` passes; `id` the library's.
    let (lookup, id) = unsafe { (&*user.cast::<Lookup>(), &*id) };
    if let Some(info) = lookup.debug_info.get(id) {
        // SAFETY: our bytes, alive through the call; the library copies them.
        unsafe { set(info.as_ptr().cast(), info.len() as u32) };
    }
}

unsafe extern "C" fn find_shader(hash: *const u64, set: SetData, user: *mut c_void) {
    // SAFETY: as in `find_debug_info`.
    let (lookup, hash) = unsafe { (&*user.cast::<Lookup>(), *hash) };
    if let Some(t) = lookup.shaders.get(&hash).and_then(|s| s.translated()) {
        // SAFETY: as in `find_debug_info`.
        unsafe { set(t.host.dxil.as_ptr().cast(), t.host.dxil.len() as u32) };
    }
}

/// A dump as JSON (everything the decoder knows, with shader addresses
/// mapped to DXIL where debug information allows), and its summary.
fn decode(
    api: &Api,
    dump: &[u8],
    debug_info: &HashMap<[u64; 2], Vec<u8>>,
    shaders: &HashMap<u64, Arc<Shader>>,
) -> Result<(Vec<u8>, String), String> {
    let check = |result: u32, what: &str| {
        if succeeded(result) {
            Ok(result)
        } else {
            Err(format!("{what}: {}", result_name(result)))
        }
    };
    let mut decoder = std::ptr::null_mut();
    // SAFETY: the dump lives through the call; the output is ours.
    let created = unsafe {
        (api.create_decoder)(
            VERSION,
            dump.as_ptr().cast(),
            dump.len() as u32,
            &mut decoder,
        )
    };
    check(created, "decoder")?;
    let decoded = (|| {
        let mut lookup = Lookup {
            debug_info,
            shaders,
        };
        let mut size = 0;
        // SAFETY: our decoder; `lookup` lives through the call, which is
        // where the lookups run.
        let generated = unsafe {
            (api.generate_json)(
                decoder,
                ALL_INFO,
                UTF8,
                Some(find_debug_info),
                Some(find_shader),
                None,
                (&raw mut lookup).cast(),
                &mut size,
            )
        };
        check(generated, "JSON")?;
        let mut json = vec![0u8; size as usize];
        // SAFETY: a buffer of the size the decoder asked for.
        let copied = unsafe { (api.json)(decoder, size, json.as_mut_ptr().cast()) };
        check(copied, "JSON")?;
        if let Some(end) = json.iter().position(|&b| b == 0) {
            json.truncate(end);
        }
        let summary = summary(api, decoder, &json, shaders);
        Ok((json, summary))
    })();
    // SAFETY: our decoder, destroyed once.
    unsafe { (api.destroy_decoder)(decoder) };
    decoded
}

/// The device's state, the page fault if any (with the hardware unit that
/// faulted), and each active shader by our stage and hash, with its warps
/// and where they stood (the shader's offset and, with debug information,
/// its DXIL line), from the decoder and the JSON it made.
fn summary(api: &Api, decoder: Handle, json: &[u8], shaders: &HashMap<u64, Arc<Shader>>) -> String {
    const DEVICE: [&str; 9] = [
        "active",
        "timeout",
        "out of memory",
        "page fault",
        "stopped",
        "reset",
        "unknown",
        "DMA fault",
        "removed without a GPU fault",
    ];
    const FAULT: [&str; 3] = ["unknown", "address translation", "illegal access"];
    const ACCESS: [&str; 4] = ["unknown", "read", "write", "atomic"];
    const ENGINE: [&str; 8] = [
        "unknown",
        "graphics",
        "graphics compute",
        "display",
        "copy",
        "video decoder",
        "video encoder",
        "other",
    ];
    const UNIT: [&str; 14] = [
        "unknown",
        "host interface",
        "front end",
        "primitive distributor",
        "graphics processing cluster",
        "polymorph engine",
        "raster engine",
        "2D rasterizer",
        "render output unit",
        "texture processing cluster",
        "copy engine",
        "video decoder",
        "video encoder",
        "other",
    ];
    let name = |names: &[&'static str], i: u32| names.get(i as usize).copied().unwrap_or("?");
    let mut text = String::new();
    let mut device = DeviceInfo::default();
    // SAFETY: our decoder; the output is ours.
    if succeeded(unsafe { (api.device_info)(decoder, &mut device) }) {
        text.push_str(&format!(
            "device {}{}{}",
            name(&DEVICE, device.status),
            if device.adapter_reset & 0xff != 0 {
                ", adapter reset"
            } else {
                ""
            },
            if device.engine_reset & 0xff != 0 {
                ", engine reset"
            } else {
                ""
            },
        ));
    }
    let mut fault = PageFaultInfo::default();
    // SAFETY: as above.
    let result = unsafe { (api.page_fault_info)(decoder, &mut fault) };
    if succeeded(result) && result != NOT_AVAILABLE {
        text.push_str(&format!(
            "; page fault at {:#x} ({}, {}) by the {} engine's {}",
            fault.va,
            name(&FAULT, fault.fault),
            name(&ACCESS, fault.access),
            name(&ENGINE, fault.engine),
            name(&UNIT, fault.client),
        ));
        // Which host copy of title memory the address is next to.
        text.push_str(&crate::shadow::near(fault.va));
        text.push_str(&crate::descriptor::host::near_heaps(fault.va));
    } else {
        text.push_str("; no page fault");
    }
    // The JSON names each shader the driver's way ("Shader name", with
    // the decoder's own "Shader hash") and gives warps as "<name> @
    // <offset>" under "Active Warps" (or a faulted warps list).
    let parsed = std::str::from_utf8(json).ok().and_then(Json::parse);
    let mut objects = Vec::new();
    if let Some(json) = &parsed {
        json.objects("", &mut objects);
    }
    let names: HashMap<&str, &str> = objects
        .iter()
        .filter_map(|(_, o)| Some((field(o, "Shader hash")?, field(o, "Shader name")?)))
        .collect();
    let mut warps: Vec<(&str, &[(String, Json)])> = objects
        .iter()
        .filter(|(_, o)| field(o, "GPU PC Address").is_some())
        .copied()
        .collect();
    let mut count = 0;
    // SAFETY: as above.
    let mut result = unsafe { (api.active_shader_count)(decoder, &mut count) };
    let mut active = vec![ShaderInfo::default(); count as usize];
    if succeeded(result) && count > 0 {
        // SAFETY: a buffer of `count` entries.
        result = unsafe { (api.active_shaders)(decoder, count, active.as_mut_ptr()) };
    }
    if !succeeded(result) || count == 0 {
        active.clear();
        text.push_str("; no active shaders");
    }
    for info in &active {
        let mut hash = 0;
        // SAFETY: our decoder and one of its shader infos.
        let ours = succeeded(unsafe { (api.shader_info_hash)(decoder, info, &mut hash) })
            .then(|| shaders.get(&hash))
            .flatten();
        text.push_str(&match ours {
            Some(s) => format!("; {} {}", s.stage, s.hash),
            None if info.internal & 0xff != 0 => format!("; internal shader {hash:#018x}"),
            None => format!("; unknown shader {hash:#018x} (type {})", info.kind),
        });
        let Some(driver_name) = names.get(info.hash.to_string().as_str()) else {
            continue;
        };
        let prefix = format!("{driver_name} @");
        let (its, rest): (Vec<_>, Vec<_>) = warps.into_iter().partition(|(_, o)| {
            field(o, "GPU PC Address").is_some_and(|pc| pc.starts_with(&prefix))
        });
        warps = rest;
        text.push_str(&describe_warps(&its));
    }
    if !warps.is_empty() {
        text.push_str("; warps of other shaders");
        text.push_str(&describe_warps(&warps));
    }
    text
}

/// Warps for the summary: how many, then each place (the shader's offset,
/// how many warps stood there, the DXIL line if mapped, and what else the
/// decoder says of it), four at most.
fn describe_warps(warps: &[(&str, &[(String, Json)])]) -> String {
    const SHOWN: usize = 4;
    const SAID: [&str; 3] = ["GPU PC Address", "Warp count", "Shader mapping"];
    let total: u64 = warps
        .iter()
        .map(|(_, o)| {
            field(o, "Warp count")
                .and_then(|n| n.parse::<u64>().ok())
                .unwrap_or(1)
        })
        .sum();
    let mut text = format!(": {total} warps");
    for (section, o) in warps.iter().take(SHOWN) {
        text.push_str(", ");
        if *section != "Active Warps" {
            text.push_str(&format!("{section}: "));
        }
        text.push_str(field(o, "GPU PC Address").unwrap_or("?"));
        if let Some(n) = field(o, "Warp count") {
            text.push_str(&format!(" x{n}"));
        }
        if let Some(line) = field(o, "Shader mapping") {
            text.push_str(&format!(" ({line})"));
        }
        for (key, value) in o.iter() {
            if let Json::Scalar(value) = value
                && !SAID.contains(&key.as_str())
            {
                text.push_str(&format!(" {key}={value}"));
            }
        }
    }
    if warps.len() > SHOWN {
        text.push_str(&format!(", {} more places", warps.len() - SHOWN));
    }
    text
}

/// A JSON value as the decoder writes it: objects keep their keys in
/// order, repeated ones too; numbers stay text (shader hashes pass 2^53).
enum Json {
    Scalar(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Option<Json> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        parser.value()
    }

    /// Every object within, with the key it is under (that of the list it
    /// is in, for a list's objects).
    fn objects<'a>(&'a self, under: &'a str, out: &mut Vec<(&'a str, &'a [(String, Json)])>) {
        match self {
            Json::Object(fields) => {
                out.push((under, fields));
                for (key, value) in fields {
                    value.objects(key, out);
                }
            }
            Json::List(items) => items.iter().for_each(|v| v.objects(under, out)),
            Json::Scalar(_) => {}
        }
    }
}

/// An object's scalar field.
fn field<'a>(object: &'a [(String, Json)], key: &str) -> Option<&'a str> {
    object.iter().find_map(|(k, v)| match v {
        Json::Scalar(s) if k == key => Some(s.as_str()),
        _ => None,
    })
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn eat(&mut self, byte: u8) -> bool {
        while self.bytes.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
        let found = self.bytes.get(self.at) == Some(&byte);
        self.at += usize::from(found);
        found
    }

    fn value(&mut self) -> Option<Json> {
        if self.eat(b'{') {
            let mut fields = Vec::new();
            if self.eat(b'}') {
                return Some(Json::Object(fields));
            }
            loop {
                let key = self.string()?;
                if !self.eat(b':') {
                    return None;
                }
                fields.push((key, self.value()?));
                if self.eat(b'}') {
                    return Some(Json::Object(fields));
                }
                if !self.eat(b',') {
                    return None;
                }
            }
        }
        if self.eat(b'[') {
            let mut items = Vec::new();
            if self.eat(b']') {
                return Some(Json::List(items));
            }
            loop {
                items.push(self.value()?);
                if self.eat(b']') {
                    return Some(Json::List(items));
                }
                if !self.eat(b',') {
                    return None;
                }
            }
        }
        if self.bytes.get(self.at) == Some(&b'"') {
            return self.string().map(Json::Scalar);
        }
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| !b",]} \t\r\n".contains(b))
        {
            self.at += 1;
        }
        (self.at > start).then(|| {
            Json::Scalar(String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned())
        })
    }

    /// A string, unescaped.
    fn string(&mut self) -> Option<String> {
        if !self.eat(b'"') {
            return None;
        }
        let mut out = Vec::new();
        loop {
            let byte = *self.bytes.get(self.at)?;
            self.at += 1;
            match byte {
                b'"' => return Some(String::from_utf8_lossy(&out).into_owned()),
                b'\\' => {
                    let escaped = *self.bytes.get(self.at)?;
                    self.at += 1;
                    match escaped {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let hex =
                                std::str::from_utf8(self.bytes.get(self.at..self.at + 4)?).ok()?;
                            self.at += 4;
                            let c = char::from_u32(u32::from_str_radix(hex, 16).ok()?)
                                .unwrap_or('\u{fffd}');
                            out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                _ => out.push(byte),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the library (`VICTORY_AFTERMATH`): crash dumps enabled before
    /// the host device, Aftermath on the device, and a host compute
    /// pipeline's shader recorded under the SDK's hash of its code.
    #[test]
    fn host_pipelines_record_their_shaders() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let Some(a) = get() else {
            return; // no library
        };
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        assert!(on(), "Aftermath refused the host device (see the log)");
        // No bindings: the host root signature declares what the shader's
        // translation reads, and this one reads nothing.
        const SOURCE: &str = "[numthreads(64, 1, 1)] void main() {}";
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the output is ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                windows::core::s!("aftermath"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
        }
        .unwrap();
        let blob = blob.unwrap();
        // SAFETY: the blob's own bytes.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        }
        .to_vec();
        let shader = Shader::with_code("compute", "a5a5a5a5a5a5a5a5", code.clone());
        let pipeline =
            crate::pipeline::host::HostPipeline::new(std::slice::from_ref(&shader), None);
        assert!(
            pipeline.is_some_and(|p| p.compute),
            "no host compute pipeline (see the log)"
        );
        let hash = a.hash(&code).unwrap();
        let known = a.shaders.lock().unwrap();
        assert!(
            known
                .by_hash
                .get(&hash)
                .is_some_and(|s| Arc::ptr_eq(s, &shader))
        );
        drop(known);
        // Event markers: a host list's notes become Aftermath markers.
        let list = host.new_list().unwrap();
        let marks = Marks::new(&list.list).expect("an Aftermath context for an open list");
        marks.mark("aftermath test marker");
        drop(marks);
        list.note_op(|| "aftermath test note".to_owned());
        list.close();
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Decodes a dump from the SDK's samples (`VICTORY_AFTERMATH_SAMPLE`:
    /// the `GpuCrashDump` folder of NVIDIA/nsight-aftermath-samples) with
    /// its shader debug information and shader objects.
    #[test]
    fn decodes_a_sample_dump() {
        let (Some(a), Some(sample)) = (get(), std::env::var_os("VICTORY_AFTERMATH_SAMPLE")) else {
            return; // no library or no sample
        };
        let sample = PathBuf::from(sample);
        let mut dump = None;
        let mut debug_info = HashMap::new();
        for entry in std::fs::read_dir(&sample).unwrap().flatten() {
            let path = entry.path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("nv-gpudmp") => dump = Some(std::fs::read(&path).unwrap()),
                Some("nvdbg") => {
                    let bytes = std::fs::read(&path).unwrap();
                    let mut id = [0u64; 2];
                    // SAFETY: the file's bytes; the output is ours.
                    let result = unsafe {
                        (a.api.debug_info_id)(
                            VERSION,
                            bytes.as_ptr().cast(),
                            bytes.len() as u32,
                            &mut id,
                        )
                    };
                    assert!(succeeded(result), "{}", result_name(result));
                    debug_info.insert(id, bytes);
                }
                _ => {}
            }
        }
        let mut shaders = HashMap::new();
        for (name, stage) in [("VertexShader", "vertex"), ("PixelShader", "pixel")] {
            let code =
                std::fs::read(sample.join("Extras/Shaders").join(format!("{name}.cso"))).unwrap();
            let hash = a.hash(&code).unwrap();
            shaders.insert(hash, Shader::with_code(stage, name, code));
        }
        let (json, summary) = decode(
            &a.api,
            &dump.expect("a .nv-gpudmp file"),
            &debug_info,
            &shaders,
        )
        .unwrap();
        eprintln!("{summary}");
        assert!(Json::parse(std::str::from_utf8(&json).unwrap()).is_some());
        // A vertex shader hung in a long loop: one warp, mapped to its line.
        assert!(summary.starts_with("device timeout"), "{summary}");
        assert!(
            summary.contains(
                "; vertex VertexShader: 1 warps, vertex_01 @ 0x00000330 x1 (VertexShader.hlsl:26)"
            ),
            "{summary}"
        );
    }
}
