//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! Grand Theft Auto VI creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    container: Vec<u8>,
    /// Its result gets the least indirect light ([`crate::ambient`]).
    #[cfg(windows)]
    pub(crate) resolves_light: bool,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// `VICTORY_SHADER_SWAP=<folder>`: a container whose hash (16 digits, as
/// logged) names a file `<hash>.dxbc` there is replaced by that file's
/// container, of the same stage and root signature.
/// shortcut: for the distant trees' wind variants (`VS_Octa*Wind*`), whose
/// wind volumes nothing fills yet, so that they threw the trees' quads
/// across the screen; their variants without wind stand in for them.
fn swapped(hash: &str) -> Option<&'static [u8]> {
    static SWAPS: OnceLock<HashMap<String, Vec<u8>>> = OnceLock::new();
    SWAPS
        .get_or_init(|| {
            let mut swaps = HashMap::new();
            let Some(folder) = std::env::var_os("VICTORY_SHADER_SWAP") else { return swaps };
            for entry in std::fs::read_dir(folder).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "dxbc")
                    && let (Some(name), Ok(bytes)) = (path.file_stem().and_then(|n| n.to_str()), std::fs::read(&path))
                {
                    swaps.insert(name.to_owned(), bytes);
                }
            }
            tracing::info!(count = swaps.len(), "shader containers replaced");
            swaps
        })
        .get(hash)
        .map(Vec::as_slice)
}

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let (bytes, full) = match swapped(&full.to_hex()[..16]) {
        Some(other) => (other, blake3::hash(other)),
        None => (bytes, full),
    };
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                container: bytes.to_vec(),
                #[cfg(windows)]
                resolves_light: bytes
                    .windows(crate::ambient::PROGRAM.len())
                    .any(|w| w == crate::ambient::PROGRAM),
                #[cfg(windows)]
                translated: OnceLock::new(),
            })
        })
        .clone()
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    /// A compute program's group size.
    pub(crate) compute_threads: Option<[u32; 3]>,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    /// Whether the program intersects rays (`image_bvh*_intersect_ray`),
    /// against the title's acceleration structures in title memory.
    pub(crate) rays: bool,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let done = translate(&self.container).and_then(|t| {
                    match page_table_refuses(&t, &self.hash) {
                        Some(why) => Err(why.to_owned()),
                        None => Ok(t),
                    }
                });
                match &done {
                    Ok(t) => {
                        tracing::info!(
                            stage = self.stage,
                            hash = %self.hash,
                            user_data = t.host.user_data.len(),
                            samplers = t.host.samplers.len(),
                            "shader translated"
                        );
                        #[cfg(feature = "diagnostics")]
                        if t.rays {
                            tracing::info!(
                                stage = self.stage,
                                hash = %self.hash,
                                "ray intersections: the title's acceleration structures are read \
                                 through the title-memory page table (logged once per program)"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

#[cfg(all(windows, test, feature = "diagnostics"))]
impl Shader {
    /// A shader whose translation is `code` as it stands (tests that need
    /// host code without a console container).
    pub(crate) fn with_code(stage: &'static str, hash: &str, code: Vec<u8>) -> Arc<Shader> {
        use victory_shader::HwStage;
        let hw = match stage {
            "compute" => HwStage::Cs,
            "pixel" => HwStage::Ps,
            "mesh" => HwStage::Ms,
            _ => HwStage::Vs,
        };
        Arc::new(Shader {
            stage,
            hash: hash.to_owned(),
            container: Vec::new(),
            resolves_light: false,
            translated: OnceLock::from(Some(Translated {
                host: victory_shader::emit::HostShader {
                    stage: hw,
                    dxil: code,
                    user_data: Vec::new(),
                    samplers: Vec::new(),
                    static_samplers: Vec::new(),
                    root_cbvs: Vec::new(),
                    root_srvs: Vec::new(),
                    view_strides: false,
                    heap_indices: Vec::new(),
                    heap_unknown: 0,
                    root_uavs: Vec::new(),
                    memory: false,
                    memory_writes: false,
                    view_addresses: false,
                },
                compute_threads: None,
                used: Vec::new(),
                rays: false,
            })),
        })
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    let host = crate::shader_cache::host_shader(container)?;
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let native = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
                )
            }),
        None => None,
    };
    let rays = native.as_ref().is_some_and(|p| {
        victory_shader::gfx10::decode(&p.code).is_ok_and(|insts| {
            insts
                .iter()
                .any(|i| i.name().is_some_and(|n| n.starts_with("image_bvh")))
        })
    });
    let compute_threads = native.as_ref().and_then(|p| match p.io {
        victory_shader::native::StageIo::Cs(cs) => Some(cs.threads),
        _ => None,
    });
    Ok(Translated {
        host,
        compute_threads,
        used: native.map(|p| p.used).unwrap_or_default(),
        rays,
    })
}

/// `VICTORY_PAGE_TABLE`: which programs reach title memory at addresses
/// they compute, through the page table (`crate::memory`). The others
/// do not translate, as before the page table: their draws and dispatches
/// are dropped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PageTable {
    /// None.
    Off,
    /// All but ray programs and the BVH builders (the default).
    NonRt,
    /// All but ray programs: the BVH builders run.
    NoRays,
    All,
}

fn parse_page_table(value: &str) -> PageTable {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" => PageTable::Off,
        "" | "nonrt" => PageTable::NonRt,
        "norays" => PageTable::NoRays,
        "all" => PageTable::All,
        other => {
            tracing::warn!(
                value = other,
                "VICTORY_PAGE_TABLE expects off, nonrt, norays or all: nonrt"
            );
            PageTable::NonRt
        }
    }
}

#[cfg(windows)]
fn page_table() -> PageTable {
    static MODE: OnceLock<PageTable> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = parse_page_table(&std::env::var("VICTORY_PAGE_TABLE").unwrap_or_default());
        tracing::info!(?mode, "VICTORY_PAGE_TABLE");
        mode
    })
}

/// XboxBvhBuilder.dll's 52 programs (Grand Theft Auto VI 1.0.2.1), by
/// container hash.
const BVH_BUILDERS: [&str; 52] = [
    "0dd520d405264736",
    "0ddab4b63b1394e0",
    "14394a322f6dd5c2",
    "162216b9cfd91aaf",
    "190fbf0ebe040169",
    "198f6dfd2b81bcea",
    "28918b7150a7aad5",
    "35e51bb27a463294",
    "39321a95223001ed",
    "3b90165a77d95cdf",
    "3cea4534bbc15632",
    "3df629ffc3d0a2a6",
    "4b5aedc0e96b0770",
    "4ddd6f9702fab2a7",
    "4f3227f524514cd8",
    "5062bc7a48e842f8",
    "5279ae36796c31fc",
    "5326ef52f27f5a41",
    "54b3e5465215e92e",
    "5552dac6a2b7a1fd",
    "5c09eb42b7a3c362",
    "5e19e7c728d956df",
    "5e68767498c57eb6",
    "6b7cb8a6572efff1",
    "6f6a359d1166bb27",
    "77360df21e12ea43",
    "7bb5b1b91d283a28",
    "894c91654720595f",
    "8bb4592836034757",
    "9205d2655bf8cc88",
    "93be497edd442293",
    "9fc111d3be79364f",
    "a197f50cd8481541",
    "a63e2dfc25ba3155",
    "b0fc38d36f25b863",
    "bb55c842c391589f",
    "be6862bb8ce8d9e9",
    "c42a31951201f9a0",
    "c85822de2e41cfaa",
    "caa78b8ca62bc3d6",
    "cb9647e77864cb3a",
    "ce2fc10ab6471ab8",
    "db36ed4ffef76056",
    "e63d22c4def9ba6a",
    "e95858837a9aa22d",
    "eaaf3a4f6c87b080",
    "eb89347fa4f5dc34",
    "f28a33a89eaa2caa",
    "f6e47efa50226beb",
    "f6eb6498409ca974",
    "f94e5d8bf4e1cf05",
    "ffd93c8a029ff016",
];

/// Why `VICTORY_PAGE_TABLE` refuses translation `t` of the program
/// `hash`, if it does.
#[cfg(windows)]
fn page_table_refuses(t: &Translated, hash: &str) -> Option<&'static str> {
    if !t.host.memory && !t.rays {
        return None;
    }
    let builder = BVH_BUILDERS.contains(&hash);
    match page_table() {
        PageTable::Off => Some("title memory at computed addresses (VICTORY_PAGE_TABLE=off)"),
        PageTable::NonRt | PageTable::NoRays if t.rays => {
            Some("a ray program (VICTORY_PAGE_TABLE is not all)")
        }
        PageTable::NonRt if builder => Some("a BVH builder (VICTORY_PAGE_TABLE=nonrt)"),
        _ => None,
    }
}

#[cfg(test)]
mod page_table_tests {
    use super::{BVH_BUILDERS, PageTable, parse_page_table};

    #[test]
    fn the_page_table_switch_parses_or_falls_back_to_nonrt() {
        assert_eq!(parse_page_table(""), PageTable::NonRt);
        assert_eq!(parse_page_table(" OFF "), PageTable::Off);
        assert_eq!(parse_page_table("norays"), PageTable::NoRays);
        assert_eq!(parse_page_table("all"), PageTable::All);
        assert_eq!(parse_page_table("rt"), PageTable::NonRt);
    }

    #[test]
    fn builder_hashes_are_distinct_16_digit_hashes() {
        let mut sorted = BVH_BUILDERS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), BVH_BUILDERS.len());
        assert!(
            BVH_BUILDERS
                .iter()
                .all(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        );
    }
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Programs that have hung the host GPU: their
/// dispatches are dropped with `VICTORY_DROP_HANGING=1`.
/// XboxBvhBuilder.dll@0x53fd0's BVH refit climbed parent links that never
/// reached the root (runs 127 and 128, 1 warp and then 125 spinning) while
/// its thread counts ran as group counts ([`crate::draw::State::dispatch_x`]).
#[cfg(windows)]
const HANGING: &[&str] = &["5552dac6a2b7a1fd"];

/// Whether `this` holds a program of [`HANGING`] that is not to run.
#[cfg(windows)]
pub(crate) fn hangs(this: This) -> bool {
    static DROP: OnceLock<bool> = OnceLock::new();
    let drop =
        *DROP.get_or_init(|| std::env::var_os("VICTORY_DROP_HANGING").is_some_and(|v| v == "1"));
    drop && !this.is_null()
        && pipeline(this)
            .shaders
            .iter()
            .any(|shader| HANGING.contains(&shader.hash.as_str()))
}

/// Whether `this` holds one of XboxBvhBuilder.dll's programs.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null()
        && pipeline(this)
            .shaders
            .iter()
            .any(|shader| BVH_BUILDERS.contains(&shader.hash.as_str()))
}

/// `VICTORY_WAIT_BUILDER`: the BVH builders' host pipelines are started when
/// the title creates them, and a dispatch whose pipeline is not made yet
/// waits for it however long that takes instead of being dropped
/// ([`make::ready`]). Off unless set.
#[cfg(windows)]
fn wait_builder() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_BUILDER").is_some())
}

/// Warns, once per pipeline, of a BVH builder dispatch dropped and why: the
/// builder's passes after it then work on a broken tree.
#[cfg(windows)]
pub(crate) fn builder_dropped(this: This, why: &str) {
    static TOLD: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    if !is_builder(this) {
        return;
    }
    let mut told = TOLD.lock().unwrap_or_else(|e| e.into_inner());
    if told.contains(&(this as usize)) {
        return;
    }
    told.push(this as usize);
    let why = match why {
        "" => make::state(&pipeline(this).host),
        why => why,
    };
    tracing::warn!(
        frame = crate::frame::current(),
        shaders = ?pipeline(this).shaders,
        why,
        "a BVH builder dispatch dropped"
    );
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Time title threads waited for host pipelines since the last call.
#[cfg(windows)]
pub(crate) fn take_waited() -> std::time::Duration {
    make::take_waited()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        match value.strip_prefix('@') {
            Some(path) => parse_skip_hashes(&std::fs::read_to_string(path).unwrap_or_default()),
            None => parse_skip_hashes(&value),
        }
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// With `VICTORY_DROPS=<file>` (diagnostics): every reason of the drops
/// taken over `frames` frames, per frame, appended to the file (emptied at
/// the first call) under the frame presented last. Every meter window is
/// written; the log shows the first 40 reasons of one window in five.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn dump_drops(drops: &[(String, u64)], frames: f64) {
    use std::io::Write as _;
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let Some(file) = FILE.get_or_init(|| {
        let path = std::env::var_os("VICTORY_DROPS")?;
        match std::fs::File::create(&path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(e) => {
                tracing::warn!(?path, error = %e, "VICTORY_DROPS: cannot create the file");
                None
            }
        }
    }) else {
        return;
    };
    let mut text = format!("frame {} ({frames} frames)\n", crate::queue::presented());
    for (why, n) in drops {
        text += &format!("{:.1}\t{why}\n", *n as f64 / frames);
    }
    let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = file.write_all(text.as_bytes()) {
        tracing::warn!(error = %e, "VICTORY_DROPS: cannot write the file");
    }
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let must = wait_builder()
        && shaders
            .iter()
            .any(|shader| BVH_BUILDERS.contains(&shader.hash.as_str()));
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics, must));
    #[cfg(windows)]
    make::early(&host);
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// Device methods 0x1c8 and 0x218 (Grand Theft Auto VI calls each once at
/// startup, title+0x2477578 and title+0x24800d5): a plain compute or
/// graphics description, both the root signature then a shader, told apart
/// by that shader's DXIL kind. The first pipeline of each is logged.
fn create_plain(
    slot: usize,
    this: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(head) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let code = &head.stage;
    let kind = (!code.code.is_null())
        .then(|| {
            // SAFETY: the title passes `size` bytes of shader code.
            unsafe { std::slice::from_raw_parts(code.code, code.size) }
        })
        .and_then(|bytes| {
            let container = victory_shader::Container::parse(bytes).ok()?;
            let version = container
                .part(victory_shader::FourCC::DXIL)?
                .data
                .get(..4)?;
            Some(u32::from_le_bytes(version.try_into().ok()?) >> 16)
        });
    let (stage, hr) = match kind {
        Some(5) => ("compute", create_compute(this, desc, iid, out)),
        Some(1) => ("vertex", create_graphics(this, desc.cast(), iid, out)),
        _ => {
            tracing::warn!(slot = %format!("{slot:#x}"), ?kind, "plain pipeline description without a compute or vertex shader");
            return E_INVALIDARG;
        }
    };
    static LOGGED: [std::sync::atomic::AtomicBool; 2] =
        [const { std::sync::atomic::AtomicBool::new(false) }; 2];
    if !LOGGED[usize::from(slot == 0x218)].swap(true, std::sync::atomic::Ordering::Relaxed) {
        let hash = copy(stage, code)
            .map(|s| s.hash.clone())
            .unwrap_or_default();
        #[cfg(all(windows, feature = "diagnostics"))]
        let name = crate::host::shader_names().get(&hash).cloned();
        #[cfg(not(all(windows, feature = "diagnostics")))]
        let name: Option<String> = None;
        tracing::info!(slot = %format!("{slot:#x}"), stage, hash, name = name.as_deref().unwrap_or("?"), hr, "pipeline from a plain description");
    }
    hr
}

/// Device method 0x1c8: see [`create_plain`].
pub(crate) extern "system" fn create_plain_1c8(
    this: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_plain(0x1c8, this, desc, iid, out)
}

/// Device method 0x218: see [`create_plain`].
pub(crate) extern "system" fn create_plain_218(
    this: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_plain(0x218, this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    #[test]
    fn plain_descriptions_route_on_the_shader_kind() {
        let container = |kind: u32| {
            let words = |w: &[u32]| w.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
            let mut c = b"DXBC".to_vec();
            c.extend([0; 16]);
            c.extend(words(&[1, 52, 1, 36]));
            c.extend(b"DXIL");
            c.extend(words(&[8, (kind << 16) | 0x66, 2]));
            c
        };
        let code = |c: &[u8]| Bytecode {
            code: c.as_ptr(),
            size: c.len(),
        };
        let none = || code(&[]);
        let (cs, vs, ps) = (container(5), container(1), container(0));
        let mut p = std::ptr::null();
        let compute = ComputeHead {
            root: std::ptr::null(),
            stage: code(&cs),
        };
        let hr = create_plain_1c8(std::ptr::null(), &compute, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("compute", cs.clone())]);
        let graphics = GraphicsHead {
            root: std::ptr::null(),
            stages: [code(&vs), code(&ps), none(), none(), none()],
        };
        let hr = create_plain_218(
            std::ptr::null(),
            (&raw const graphics).cast(),
            &PIPELINE_IIDS[0],
            &mut p,
        );
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("vertex", vs.clone()), ("pixel", ps.clone())]);
        let pixel_first = ComputeHead {
            root: std::ptr::null(),
            stage: code(&ps),
        };
        let hr = create_plain_1c8(std::ptr::null(), &pixel_first, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, E_INVALIDARG);
    }

    /// The stream layout Grand Theft Auto VI builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}
