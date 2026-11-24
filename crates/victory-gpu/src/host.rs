//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// Resources heap slots stopped viewing (see [`Host::retire`]).
    retired: Mutex<Vec<ID3D12Resource>>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds) and in unit
    /// tests: the debug layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            #[cfg(feature = "diagnostics")]
            if let Some(dump) = crate::aftermath::report() {
                text.push_str(&format!("; Aftermath: {dump}"));
            }
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics, and in unit tests
        // (which count its errors: [`errors_of`]).
        let debug = cfg!(test)
            || cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        // Diagnostics: GPU crash dumps (`VICTORY_AFTERMATH`), enabled before
        // the device is made.
        #[cfg(feature = "diagnostics")]
        crate::aftermath::enable();
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        #[cfg(feature = "diagnostics")]
        crate::aftermath::initialize(&device, debug);
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            retired: Mutex::default(),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// The errors the debug layer reports for what `f` does on this thread
/// (unit tests run in parallel on the one device); `None` without the
/// debug layer.
#[cfg(test)]
pub(crate) fn errors_of(f: impl FnOnce()) -> Option<Vec<String>> {
    use std::cell::RefCell;
    thread_local! {
        static ERRORS: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    }
    // The layer calls it on the thread that made the call it reports.
    unsafe extern "system" fn note(
        _category: D3D12_MESSAGE_CATEGORY,
        severity: D3D12_MESSAGE_SEVERITY,
        _id: D3D12_MESSAGE_ID,
        description: windows::core::PCSTR,
        _context: *mut std::ffi::c_void,
    ) {
        if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
            // SAFETY: the layer passes a NUL-terminated string.
            let text = unsafe { description.to_string() }.unwrap_or_default();
            ERRORS.with_borrow_mut(|errors| errors.iter_mut().for_each(|e| e.push(text.clone())));
        }
    }
    static COUNTING: OnceLock<bool> = OnceLock::new();
    let counting = *COUNTING.get_or_init(|| {
        let queue = host()
            .and_then(|h| h.messages.as_ref())
            .and_then(|q| q.cast::<ID3D12InfoQueue1>().ok());
        let mut cookie = 0;
        queue.is_some_and(|q| {
            // SAFETY: a function that lives as long as the process.
            unsafe {
                q.RegisterMessageCallback(
                    Some(note),
                    D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                    std::ptr::null_mut(),
                    &mut cookie,
                )
            }
            .is_ok()
        })
    });
    ERRORS.set(counting.then(Vec::new));
    f();
    ERRORS.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "diagnostics")]
    #[test]
    fn timed_commands_are_labelled_by_kind_and_pipeline() {
        for (note, label) in [
            (
                "indirect kind 10 max 4096 at 0xdc799b400 [mesh 1fa93fcea4869fa8, pixel d603540bd3056c4b] | 1fa93fcea4869fa8: u4=0x10",
                "indirect kind 10 [mesh 1fa93fcea4869fa8, pixel d603540bd3056c4b]",
            ),
            (
                "dispatch 257x1x1 [compute cc6c9f45e334afb7] | cc6c9f45e334afb7: u4=0x1",
                "dispatch [compute cc6c9f45e334afb7]",
            ),
            ("clamp kind 2 records 1", "clamp kind 2"),
            ("copy resource 0x1 to 0x2", "copy resource"),
            (
                "texture upload 0x25bf47bcfd0 (128x128x128 f83 mips 8) sub 0 of 8",
                "texture upload",
            ),
            ("resolve 0x1 to 0x2 format 28", "resolve"),
        ] {
            assert_eq!(timing_label(note), label);
        }
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Its indirect executions' clamps (`indirect`).
    clamps: Mutex<crate::indirect::Clamps>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: GPU timestamps at its start, before each
    /// noted command, at its end and around its prelude (see `STAMPS`), and
    /// where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// Whether its prelude last recorded was timed (diagnostics).
    #[cfg(feature = "diagnostics")]
    prelude_timed: std::sync::atomic::AtomicBool,
    /// With Aftermath on: the list's context for its event markers, which
    /// carry the same texts.
    #[cfg(feature = "diagnostics")]
    marks: Option<crate::aftermath::Marks>,
}

/// Timestamps per list (diagnostics): its start, one before each noted
/// command (commands past `NOTED` are not timed), its end; the last two
/// are its prelude's start and end.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;
#[cfg(feature = "diagnostics")]
const NOTED: usize = STAMPS as usize - 4;
#[cfg(feature = "diagnostics")]
const PRELUDE: u32 = STAMPS - 2;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// GPU time of the commands lists noted, since the last frame-rate line
/// (diagnostics, timing on): ms, count and the longest by command kind and
/// pipeline; lists run, ms waited for them, ms between their first and last
/// timestamps, ms in timed commands, commands timed and not.
#[cfg(feature = "diagnostics")]
#[derive(Default)]
struct GpuTimes {
    by_label: std::collections::HashMap<String, (f64, u64, f64)>,
    lists: u64,
    waited: f64,
    span: f64,
    timed: f64,
    commands: u64,
    untimed: u64,
    /// Of the waits, ms, as in [`Split`]; and waiting for the tile
    /// mappings queued before a submission's lists.
    submit: f64,
    launch: f64,
    prelude: f64,
    lead: f64,
    wake: f64,
    tiles: f64,
}

/// A timed list's submission on the CPU (QueryPerformanceCounter ticks):
/// before and after ExecuteCommandLists, when the wait ended, and the
/// queue's clock calibration then (GPU ticks, CPU ticks).
#[cfg(feature = "diagnostics")]
struct Submitted {
    before: i64,
    after: i64,
    woke: i64,
    calibration: Option<(u64, u64)>,
}

/// A timed list's wait, ms (diagnostics): in ExecuteCommandLists, from
/// calling it to the GPU's first timestamp (zero without the queue's clock
/// calibration), in its prelude, from its start to its first noted command,
/// from its start to its end, from its end to the wait's end (zero without
/// the calibration).
#[cfg(feature = "diagnostics")]
#[derive(Debug, Default)]
struct Split {
    submit: f64,
    launch: f64,
    prelude: f64,
    lead: f64,
    span: f64,
    wake: f64,
}

/// The CPU's performance counter now (diagnostics).
#[cfg(feature = "diagnostics")]
fn qpc() -> i64 {
    let mut now = 0;
    // SAFETY: writes the counter.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut now) };
    now
}

/// Ticks of [`qpc`] per second (diagnostics).
#[cfg(feature = "diagnostics")]
fn qpc_frequency() -> i64 {
    let mut f = 0;
    // SAFETY: writes the frequency.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut f) };
    f.max(1)
}

#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<GpuTimes>> = Mutex::new(None);

/// A noted command's kind and pipeline: "indirect kind 10 [mesh …]",
/// "dispatch [compute …]", "clamp kind 2", "copy resource".
#[cfg(feature = "diagnostics")]
fn timing_label(note: &str) -> String {
    let head = note.split(" | ").next().unwrap_or(note);
    let mut words = head.split(' ');
    let kind = match (words.next(), words.next(), words.next()) {
        (Some(what @ ("indirect" | "clamp")), Some("kind"), Some(k)) => format!("{what} kind {k}"),
        (Some(what @ ("dispatch" | "mesh" | "draw" | "indexed")), ..) => what.to_owned(),
        (Some(a), Some(b), _) if b.starts_with(|c: char| c.is_ascii_alphabetic()) => {
            format!("{a} {b}")
        }
        (Some(a), ..) => a.to_owned(),
        _ => String::new(),
    };
    let pipeline = head
        .find('[')
        .and_then(|s| head[s..].find(']').map(|e| &head[s..=s + e]));
    match pipeline {
        Some(p) => format!("{kind} {p}"),
        None => kind,
    }
}

/// Shader names by hash from `VICTORY_SHADER_NAMES` (a hash list of
/// `HASH <hash> @<offset> <name>` lines), for the GPU time lines.
#[cfg(feature = "diagnostics")]
fn shader_names() -> &'static std::collections::HashMap<String, String> {
    static NAMES: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let Some(path) = std::env::var_os("VICTORY_SHADER_NAMES") else {
            return Default::default();
        };
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            tracing::warn!(error = %e, ?path, "VICTORY_SHADER_NAMES unreadable");
            String::new()
        });
        text.lines()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                (words.next()? == "HASH").then_some(())?;
                let hash = words.next()?.to_ascii_lowercase();
                words.next()?;
                let name = words.collect::<Vec<_>>().join(" ");
                Some((hash, name.trim_end_matches("_Wrapped").to_owned()))
            })
            .collect()
    })
}

/// Logs the GPU time per frame by command kind and the pipelines that
/// took most, over `frames` frames (diagnostics, timing on).
#[cfg(feature = "diagnostics")]
fn log_gpu_times(frames: f64) {
    let Some(t) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let frames = frames.max(1.0);
    tracing::info!(
        "GPU timing per frame: {:.1} lists, {:.1} ms waited for them, {:.1} ms from their first \
         timestamp to their last, {:.1} ms in {:.0} timed commands ({:.0} more not timed); of the \
         waits: {:.1} ms in ExecuteCommandLists, {:.1} ms from calling it to the GPU's first \
         timestamp, {:.1} ms in preludes, {:.1} ms from lists' start to their first noted command, \
         {:.1} ms from lists' end to the wait's end; {:.1} ms waiting for tile mappings",
        t.lists as f64 / frames,
        t.waited / frames,
        t.span / frames,
        t.timed / frames,
        t.commands as f64 / frames,
        t.untimed as f64 / frames,
        t.submit / frames,
        t.launch / frames,
        t.prelude / frames,
        t.lead / frames,
        t.wake / frames,
        t.tiles / frames,
    );
    let mut kinds: std::collections::HashMap<&str, (f64, u64)> = Default::default();
    for (label, (ms, n, _)) in &t.by_label {
        let kind = label.split(" [").next().unwrap_or(label);
        let e = kinds.entry(kind).or_default();
        e.0 += ms;
        e.1 += n;
    }
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    let kinds: Vec<String> = kinds
        .iter()
        .map(|(kind, (ms, n))| format!("{kind} {:.2} ms {:.1}x", ms / frames, *n as f64 / frames))
        .collect();
    tracing::info!("GPU time per frame by command: {}", kinds.join("; "));
    let names = shader_names();
    let mut labels: Vec<_> = t.by_label.iter().collect();
    labels.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (label, (ms, n, longest)) in labels.into_iter().take(40) {
        let named: Vec<&str> = label
            .split(|c: char| !c.is_ascii_hexdigit())
            .filter(|w| w.len() == 16)
            .filter_map(|hash| names.get(hash).map(String::as_str))
            .collect();
        tracing::info!(
            "GPU time per frame: {:.2} ms in {:.1} commands (longest {longest:.2} ms): {label} {named:?}",
            ms / frames,
            *n as f64 / frames,
        );
    }
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Their staging buffers and host copies: what the prelude copies.
    staged: Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            #[cfg(feature = "diagnostics")]
            let marks = crate::aftermath::Marks::new(&list);
            let list = HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                clamps: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                prelude_timed: Default::default(),
                #[cfg(feature = "diagnostics")]
                marks,
            };
            #[cfg(feature = "diagnostics")]
            list.stamp_start();
            Ok(list)
        }
    }

    /// Runs closed lists and waits for them.
    /// Keeps `resource`, which a heap slot stopped viewing, until the
    /// submission running now (if one is) has finished: it may still read
    /// the slot's old view. Shaders that load slot indices from memory read
    /// slots no list names, so no list holds what they read.
    pub(crate) fn retire(&self, resource: ID3D12Resource) {
        self.retired
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(resource);
    }

    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let retired = std::mem::take(&mut *self.retired.lock().unwrap_or_else(|e| e.into_inner()));
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        crate::watch::forget_asked();
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        // Loose-memory windows copied for the first time get their tiles.
        crate::shadow::map_windows(&self.queue);
        let prepared = std::time::Instant::now();
        // Diagnostics: the tile mappings just queued, waited for on their own.
        #[cfg(feature = "diagnostics")]
        if timing() {
            *submitted += 1;
            // SAFETY: the fence and event are ours.
            let waited = unsafe {
                self.queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event))
                    .map(|()| WaitForSingleObject(self.event, INFINITE))
            };
            if waited.is_ok() {
                let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
                times.get_or_insert_with(GpuTimes::default).tiles +=
                    prepared.elapsed().as_secs_f64() * 1e3;
            }
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let before = qpc();
            // SAFETY: closed lists.
            unsafe { self.queue.ExecuteCommandLists(group) };
            #[cfg(feature = "diagnostics")]
            let after = qpc();
            // SAFETY: the fence and event are ours.
            unsafe {
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        WaitForSingleObject(self.event, INFINITE);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            #[cfg(feature = "diagnostics")]
            if timing() {
                let woke = qpc();
                let waited = started.elapsed();
                let (mut gpu, mut cpu) = (0, 0);
                // SAFETY: plain queries.
                let (frequency, calibrated) = unsafe {
                    (
                        self.queue.GetTimestampFrequency().unwrap_or(1),
                        self.queue.GetClockCalibration(&mut gpu, &mut cpu).is_ok(),
                    )
                };
                let calibration = calibrated.then_some((gpu, cpu));
                let split = runs[i].tally(
                    frequency,
                    waited,
                    &Submitted {
                        before,
                        after,
                        woke,
                        calibration,
                    },
                );
                if waited > std::time::Duration::from_millis(50) {
                    let commands = runs[i]
                        .notes
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len();
                    tracing::warn!(ms = waited.as_millis(), commands, ?split, "slow host list");
                    for (ms, op) in runs[i].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
            }
            let _ = (i, started);
        }
        // Retired before this submission: no submission still runs that
        // read them.
        drop(retired);
        let _ = &firsts;
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // One list's lock at a time: a submission may name a list twice.
        type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);
        let mut back: Vec<Declared> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                let Step::GpuWrite(shadow, address, len) = step else {
                    continue;
                };
                let i = match back
                    .iter()
                    .position(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
                {
                    Some(i) => i,
                    None => {
                        back.push((shadow.clone(), Vec::new()));
                        back.len() - 1
                    }
                };
                if !back[i].1.contains(&(*address, *len)) {
                    back[i].1.push((*address, *len));
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        let _charged = crate::copystat::charge(crate::copystat::Caller::CopyBack);
        for (shadow, declared) in &back {
            crate::watch::forget_asked();
            let written = shadow.copy_back(declared);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        crate::watch::disarm();
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded;
    /// without one, a null view (`desc` required) that drops what draws
    /// write to its slot.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The list's timestamps, read back after it ran (diagnostics, timing
    /// on): its start, one before each of its first `NOTED` noted
    /// commands, its end; and its prelude's start and end if a prelude ran.
    #[cfg(feature = "diagnostics")]
    fn timestamps(&self, notes: usize) -> (Vec<u64>, Option<[u64; 2]>) {
        let (Some((_, back)), count) = (self.stamps(), notes.min(NOTED)) else {
            return (Vec::new(), None);
        };
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the finished lists resolved into.
        unsafe {
            if back.Map(0, None, Some(&mut data)).is_err() {
                return (Vec::new(), None);
            }
            let all = data.cast::<u64>();
            let s = std::slice::from_raw_parts(all, count + 2).to_vec();
            let prelude = self
                .prelude_timed
                .load(Ordering::Relaxed)
                .then(|| [*all.add(PRELUDE as usize), *all.add(PRELUDE as usize + 1)]);
            back.Unmap(0, Some(&D3D12_RANGE::default()));
            (s, prelude)
        }
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (stamps, _) = self.timestamps(notes.len());
        let mut times: Vec<(f64, String)> = stamps
            .get(1..)
            .unwrap_or_default()
            .windows(2)
            .zip(notes.iter())
            .map(|(s, (note, _))| {
                let ticks = s[1].saturating_sub(s[0]);
                (
                    ticks as f64 * 1e3 / frequency.max(1) as f64,
                    note.chars().take(300).collect(),
                )
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Adds the list's command times to the frame's GPU time by command
    /// and pipeline, and splits the wait for it (diagnostics, timing on,
    /// after the list ran and was waited for `waited`).
    #[cfg(feature = "diagnostics")]
    fn tally(&self, frequency: u64, waited: std::time::Duration, submitted: &Submitted) -> Split {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (stamps, prelude) = self.timestamps(notes.len());
        let ms = |ticks: u64| ticks as f64 * 1e3 / frequency.max(1) as f64;
        // CPU times and GPU timestamps as ms since ExecuteCommandLists was
        // called.
        let since = |qpc: i64| (qpc - submitted.before) as f64 * 1e3 / qpc_frequency() as f64;
        let on_cpu = submitted.calibration.map(|(gpu, cpu)| {
            move |stamp: u64| {
                since(cpu as i64)
                    + stamp.wrapping_sub(gpu) as i64 as f64 * 1e3 / frequency.max(1) as f64
            }
        });
        let mut split = Split {
            submit: since(submitted.after),
            ..Split::default()
        };
        if let (Some(&start), Some(&end)) = (stamps.first(), stamps.last()) {
            split.prelude = prelude.map_or(0.0, |[from, to]| ms(to.saturating_sub(from)));
            split.lead = ms(stamps[1].saturating_sub(start));
            split.span = ms(end.saturating_sub(start));
            if let Some(on_cpu) = on_cpu {
                split.launch = on_cpu(prelude.map_or(start, |p| p[0]));
                split.wake = since(submitted.woke) - on_cpu(end);
            }
        }
        let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
        let t = times.get_or_insert_with(GpuTimes::default);
        t.lists += 1;
        t.waited += waited.as_secs_f64() * 1e3;
        t.submit += split.submit;
        t.launch += split.launch;
        t.prelude += split.prelude;
        t.lead += split.lead;
        t.span += split.span;
        t.wake += split.wake;
        t.untimed += notes.len().saturating_sub(NOTED) as u64;
        for (s, (note, _)) in stamps
            .get(1..)
            .unwrap_or_default()
            .windows(2)
            .zip(notes.iter())
        {
            let took = ms(s[1].saturating_sub(s[0]));
            t.timed += took;
            t.commands += 1;
            let e = t.by_label.entry(timing_label(note)).or_default();
            e.0 += took;
            e.1 += 1;
            e.2 = e.2.max(took);
        }
        split
    }

    /// Diagnostics, timing on: the timestamp of the list's start (just
    /// opened).
    #[cfg(feature = "diagnostics")]
    fn stamp_start(&self) {
        if let Some((heap, _)) = self.stamps() {
            // SAFETY: an open list and our query heap.
            unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0) };
        }
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = self
                .notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
                .min(NOTED) as u32;
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list
                    .EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count + 1);
                self.list
                    .ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 2, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        *self.deferred() = Deferred::default();
        self.clamps().reset();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        #[cfg(feature = "diagnostics")]
        self.stamp_start();
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// At submission, after the list's staging buffers were refreshed: the
    /// list that moves each texture this list uses from its real state to
    /// the one the list starts with, copying in staging buffers that hold
    /// contents their host textures have not had yet on the way; the
    /// textures' real states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let deferred = self.deferred();
        let due: Vec<_> = deferred
            .staged
            .iter()
            .filter(|(staging, _)| staging.due())
            .collect();
        let uploads: std::collections::HashSet<usize> = due
            .iter()
            .map(|(_, copy)| std::sync::Arc::as_ptr(copy) as usize)
            .collect();
        let move_to = |copy: &crate::resource::HostTexture, from: u32, to: u32| {
            (from != to).then(|| {
                transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(from as i32),
                    D3D12_RESOURCE_STATES(to as i32),
                )
            })
        };
        let (mut before, mut after) = (Vec::new(), Vec::new());
        for (key, (first, last, copy)) in states.iter() {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            if uploads.contains(key) {
                before.extend(move_to(copy, real, COPY_DEST));
                after.extend(move_to(copy, COPY_DEST, *first));
            } else {
                before.extend(move_to(copy, real, *first));
            }
        }
        // Uploaded but not otherwise used in the list: left for the copy.
        for (_, copy) in &due {
            if !states.contains_key(&(std::sync::Arc::as_ptr(copy) as usize)) {
                let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
                before.extend(move_to(copy, real, COPY_DEST));
            }
        }
        #[cfg(feature = "diagnostics")]
        let stamps = self
            .stamps()
            .filter(|_| !(before.is_empty() && due.is_empty()));
        #[cfg(feature = "diagnostics")]
        self.prelude_timed
            .store(stamps.is_some(), Ordering::Relaxed);
        if before.is_empty() && due.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                #[cfg(feature = "diagnostics")]
                if let Some((heap, _)) = stamps {
                    list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, PRELUDE);
                }
                if !before.is_empty() {
                    list.ResourceBarrier(&before);
                }
                for (staging, copy) in &due {
                    staging.copy_into(list, &copy.texture);
                }
                if !after.is_empty() {
                    list.ResourceBarrier(&after);
                }
                #[cfg(feature = "diagnostics")]
                if let Some((heap, back)) = stamps {
                    list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, PRELUDE + 1);
                    list.ResolveQueryData(
                        heap,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        PRELUDE,
                        2,
                        back,
                        u64::from(PRELUDE) * 8,
                    );
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among. With
    /// Aftermath on, also as an Aftermath event marker.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        // Timing needs the texts, not the markers: breadcrumbs cost GPU
        // time of their own.
        if dred_on() || self.stamps().is_some() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len();
            if let Some((heap, _)) = self.stamps()
                && at < NOTED
            {
                // SAFETY: an open list and our query heap.
                unsafe {
                    self.list
                        .EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at as u32 + 1)
                };
            }
            let what = what();
            if let Some(marks) = &self.marks {
                marks.mark(&what);
            }
            if dred_on() {
                let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
                const UNICODE: u32 = 0;
                // SAFETY: an open list; the marker is copied when recorded.
                unsafe {
                    self.list
                        .SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
                };
            }
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        } else if let Some(marks) = &self.marks {
            marks.mark(&what());
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    pub(crate) fn clamps(&self) -> std::sync::MutexGuard<'_, crate::indirect::Clamps> {
        self.clamps.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: the list's own mapped word.
                let used = unsafe { (at as *const u32).read_volatile() };
                let spent = match used {
                    0 => "none",
                    1..=0xff_ffff => "under 2^24",
                    0x100_0000..=0x7ff_ffff => "2^24-2^27",
                    _ => "all (records ran empty)",
                };
                crate::pipeline::note_why(format!("(not a drop) indirect budget at list reuse: {spent}"));
                for (k, seen) in crate::indirect::SEEN.iter().enumerate() {
                    // SAFETY: the list's own mapped words.
                    let n = unsafe { (at as *const u32).add(k + 1).read_volatile() };
                    seen.fetch_add(u64::from(n), Ordering::Relaxed);
                }
            }
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe {
                    for k in 0..8 {
                        (at as *mut u32).add(k).write_volatile(0);
                    }
                })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        copy: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Texture(staging.clone()));
        deferred.staged.push((staging, copy));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    batch.title_copied(dst, len);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                        add(C::CopyUs, started.elapsed().as_micros() as u64);
                    }
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data.
                    batch.flush();
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    let _charged = crate::copystat::charge(crate::copystat::Caller::Snapshot);
                    shadow.writing(*address, *len);
                    #[cfg(feature = "diagnostics")]
                    add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    continue;
                }
                Step::Texture(staging) => {
                    let (from, to) = staging.memory();
                    if batch.texture_due(staging, from, to) {
                        #[cfg(feature = "diagnostics")]
                        let _charged = crate::copystat::charge(crate::copystat::Caller::Texture);
                        staging.refresh();
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// How far apart two ranges of a loose-memory mirror slab may be and still
/// be refreshed as one ([`Batch::add`]).
const NEAR: usize = 4 << 20;

/// A submission's refreshes not done yet, merged per buffer copy (per
/// region of a loose-memory mirror slab) into the range that covers them:
/// done before a step that writes title memory or gives a range to shaders,
/// and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory,
/// or a copy into its memory (into any physical-page memory, which other
/// views may show).
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Ranges brought up to date since the last title write, with the
    /// asking before: (copy, asking, from, to). They change only where we
    /// wrote since (`watch::wrote`). Each flush asked about the world's
    /// 174 MB heap whole again (~0.2 s per frame).
    done: Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write,
    /// and the title memory each detiles.
    textures: std::collections::HashMap<usize, (usize, usize)>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        // A slab of the loose-memory mirror spans a GiB of title memory: its
        // ranges merge only with those near them (as in the 4 MiB windows
        // it replaces), not into one range over everything between.
        let near =
            |from: usize, to: usize| !shadow.mirror() || address <= to + NEAR && from <= end + NEAR;
        match self
            .refreshes
            .iter_mut()
            .find(|(s, from, to)| std::sync::Arc::ptr_eq(s, shadow) && near(*from, *to))
        {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging`, which detiles title memory `[from, to)`, is still
    /// to be refreshed in this submission (and notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>, from: usize, to: usize) -> bool {
        let key = std::sync::Arc::as_ptr(staging) as *const () as usize;
        self.textures.insert(key, (from, to)).is_none()
    }

    /// After a copy into title memory `[dst, dst + len)`: its pages count
    /// as written (ranges already asked about in this submission take them
    /// again), and textures in them are checked again; all are after a copy
    /// into physical-page memory, whose pages other views may show (the
    /// title's pool and tiled heap). Pending refreshes wait: the world's
    /// ~7,000 small copies per frame each flushed them and made every range
    /// be asked about again (~0.3 s per frame), and checking every texture
    /// again after each hashed ~180 MB a frame for ~15 MB changed.
    fn title_copied(&mut self, dst: usize, len: usize) {
        let end = dst + len;
        if crate::watch::view(dst) || crate::watch::view(end.saturating_sub(1).max(dst)) {
            self.textures.clear();
        } else {
            self.textures
                .retain(|_, &mut (from, to)| to <= dst || end <= from);
        }
        crate::watch::wrote(dst, end);
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        self.done.clear();
        crate::watch::forget_asked();
    }

    fn flush(&mut self) {
        for (shadow, from, to) in std::mem::take(&mut self.refreshes) {
            #[cfg(feature = "diagnostics")]
            let _charged = crate::copystat::charge(shadow.caller());
            let started = std::time::Instant::now();
            let asking = crate::watch::asking();
            let mut copied = 0;
            for (a, b) in self.stale(&shadow, from, to) {
                copied += shadow.update(a, b - a);
            }
            self.done.push((shadow, asking, from, to));
            let us = started.elapsed().as_micros() as u64;
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, us, 0);
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }

    /// The parts of `[from, to)` of `shadow` not brought up to date since
    /// the last title write, or written by us since.
    fn stale(&self, shadow: &std::sync::Arc<crate::shadow::Shadow>, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut current: Vec<(usize, usize)> = self
            .done
            .iter()
            .filter(|(s, asking, a, b)| {
                std::sync::Arc::ptr_eq(s, shadow)
                    && *a < to
                    && *b > from
                    && !crate::watch::wrote_since(*asking, *a, *b)
            })
            .map(|&(_, _, a, b)| (a, b))
            .collect();
        current.sort_unstable();
        let (mut parts, mut at) = (Vec::new(), from);
        for (a, b) in current {
            if a > at {
                parts.push((at, a.min(to)));
            }
            at = at.max(b);
            if at >= to {
                break;
            }
        }
        if at < to {
            parts.push((at, to));
        }
        parts
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
        // Through memory of ours, the kernel checking both sides: unmapped
        // source pages read as zeros, unmapped destination pages are left
        // alone (asking `VirtualQuery` first cost as much as the copies).
        let mut bytes = vec![0u8; len];
        copy_mapped(bytes.as_mut_ptr(), src, len, false);
        write_mapped(dst, bytes.as_ptr(), len);
        return;
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` through the kernel, which checks
/// both: false (and some bytes copied) if a source page is not readable or
/// a destination page not writable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the kernel probes both ranges and reports a fault instead of
    // raising it.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory (what shaders wrote, title copies
/// at submission), leaving pages that are not mapped alone. Guarded blocks
/// must have been opened (`watch::writing`): the kernel does not write
/// read-only pages. `ReadProcessMemory` into the destination, not
/// `WriteProcessMemory`, which asks every call for the destination's
/// protection (as slow as `VirtualQuery` in the title's big pools: ~20 us
/// a copy, ~0.15 s per world frame).
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        // The kernel reads our host memory and checks the destination.
        read(from, at as *mut u8, n)
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        #[cfg(feature = "diagnostics")]
        crate::pipeline::dump_drops(&drops, frames);
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            #[cfg(feature = "diagnostics")]
            {
                let span = std::mem::take(&mut self.copy_stats_frames) as f64;
                crate::copystat::log(span);
                crate::texture::seen::log(span);
                crate::shadow::log_roots(span);
                log_gpu_times(span);
                let uploaded = crate::texture::UPLOADED
                    .each_ref()
                    .map(|n| n.swap(0, Ordering::Relaxed) as f64 / span.max(1.0));
                tracing::info!(
                    "texture uploads per frame: {:.0} subresources, {:.1} MB",
                    uploaded[0],
                    uploaded[1] / 1e6
                );
                let seen = crate::indirect::SEEN.each_ref().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames);
                tracing::info!(
                    "indirect records run per frame: mesh {:.0} with groups, {:.0} empty; draws {:.0}, {:.0} empty; dispatches {:.0}, {:.0} empty",
                    seen[0], seen[1], seen[2], seen[3], seen[4], seen[5]
                );
                let heap = crate::draw::HEAP_SLOTS
                    .each_ref()
                    .map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames);
                tracing::info!(
                    "heap slots per frame: {:.0} made through user data; {:.0} stages bound with {:.0} heap accesses whose slot only the shader knows",
                    heap[0],
                    heap[1],
                    heap[2]
                );
                crate::descriptor::audit();
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
                if drops.len() > 40 {
                    let rest: u64 = drops[40..].iter().map(|(_, n)| n).sum();
                    tracing::info!(
                        "dropped {:.1}/frame for {} more reasons (VICTORY_DROPS=<file> lists them)",
                        rest as f64 / frames,
                        drops.len() - 40
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = victory_core::thread::spawn_host("victory-caption", || {
                loop {
                    let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                    match next {
                        // SAFETY: a window handle and a NUL-terminated text.
                        Some((hwnd, text)) => unsafe {
                            let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                        },
                        None => std::thread::sleep(std::time::Duration::from_millis(100)),
                    }
                }
            });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    victory_core::thread::spawn_host("victory-window", move || {
        let made = make_window(width, height);
        let ok = made.is_ok();
        let _ = sender.send(made.map(|h| h.0 as usize));
        if ok {
            pump();
        }
    })
    .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a, 0, 0));
        assert!(batch.texture_due(&b, 0, 0));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a, 0, 0));
        assert!(!batch.texture_due(&a.clone(), 0, 0));
        assert!(!batch.texture_due(&b, 0, 0));
    }

    /// Ranges of a loose-memory mirror slab merge only with ranges near
    /// them: a slab spans a GiB, and one range over all between two far
    /// ranges would have the watch check all of it.
    #[test]
    fn mirror_slab_ranges_merge_only_when_near() {
        use crate::shadow::tests::{gpu, release, title_memory};
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let p = title_memory(900, 512 << 20, 128 << 20, 1 << 20);
        assert_ne!(p, 0, "no free address for test memory");
        let slab = crate::shadow::find(p as u64, 1).unwrap();
        let mut batch = Batch::default();
        batch.add(&slab, p, 4096);
        batch.add(&slab, p + (2 << 20), 4096);
        batch.add(&slab, p + (100 << 20), 4096);
        let ranges: Vec<(usize, usize)> = batch.refreshes.iter().map(|&(_, a, b)| (a, b)).collect();
        let far = p + (100 << 20);
        assert_eq!(ranges, [(p, p + (2 << 20) + 4096), (far, far + 4096)]);
        batch.refreshes.clear();
        drop(slab);
        release(p);
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let _serial = crate::watch::serial();
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a, 0, 0));
        assert!(!batch.texture_due(&a, 0, 0));
        // Work that may write title memory may have changed the texture's
        // memory.
        batch.title_written();
        assert!(batch.texture_due(&a, 0, 0));
        assert!(!batch.texture_due(&a, 0, 0));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a, 0, 0));
    }

    /// A copy into write-watched title memory makes only the textures in
    /// it due again; one into physical-page memory, which other views may
    /// show, makes them all due again.
    #[test]
    fn title_copies_make_the_textures_they_write_due_again() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
            VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let _serial = crate::watch::serial();
        // SAFETY: write-watched memory and a view of a section, ours.
        let (p, v) = unsafe {
            let p = VirtualAlloc(
                std::ptr::null(),
                1 << 20,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                1 << 20,
                std::ptr::null(),
            );
            (
                p,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 1 << 20).Value as usize,
            )
        };
        assert!(p != 0 && v != 0);
        let mut batch = Batch::default();
        let (a, b, c) = (Arc::new(1u32), Arc::new(2u32), Arc::new(3u32));
        assert!(batch.texture_due(&a, p, p + 4096));
        assert!(batch.texture_due(&b, p + 65536, p + 131072));
        assert!(batch.texture_due(&c, v, v + 4096));
        batch.title_copied(p + 70000, 16);
        assert!(!batch.texture_due(&a, p, p + 4096));
        assert!(batch.texture_due(&b, p + 65536, p + 131072));
        assert!(!batch.texture_due(&c, v, v + 4096));
        batch.title_copied(v + 8192, 16);
        assert!(batch.texture_due(&a, p, p + 4096));
        assert!(batch.texture_due(&b, p + 65536, p + 131072));
        assert!(batch.texture_due(&c, v, v + 4096));
        // SAFETY: ours.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ });
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t, 0, 0) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}
