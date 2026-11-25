//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host. One
//!   of two immediate samplers chosen by uniform control flow takes two
//!   entries; the sampling runs with both and the result is chosen.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//! * **Records of the view heap** (added 2026-11-24): an indexed access
//!   (`idxen` dword loads, stores, atomics) goes to the raw view the host
//!   makes of every untyped buffer view (structured ones included) at byte
//!   index * stride + offset, plus the view's lead for a heap descriptor.
//!   Dwords `4i`, `4i + 1` and `4i + 2` of a raw buffer `t0` in
//!   [`VIEW_STRIDE_SPACE`] ([`HostShader::view_strides`], records of
//!   [`VIEW_RECORD_BYTES`]) hold the view at heap index `i`'s element size, which is what the console descriptor's
//!   stride field holds (a structured view's `StructureByteStride`, a typed
//!   view's format size, 4 for a raw view, 0 where no buffer view is), and
//!   its lead: the bytes its host raw view starts before the first record
//!   (raw views start on 16 bytes, records on any dword). The stride is the
//!   used-resource entry's where one names the descriptor, else the one read
//!   at run time (a descriptor the program reads from the heap itself,
//!   `ResourceDescriptorHeap[i]` in HLSL). Grand Theft Auto VI's instance
//!   culling and ray-tracing material resolves read their instance and
//!   geometry records this way. Dword `4i + 2` is the kind of view the host
//!   made there ([`view_kind`]: none, CBV, raw or typed buffer SRV or UAV,
//!   texture SRV or UAV; structured views count as raw, as the host makes
//!   them).
//! * **Heap slots by kind** (added 2026-11-24): a buffer access
//!   (`s_buffer_load_dword*`, `buffer_*`) through a heap descriptor no
//!   used-resource entry names (an index in a root constant or in memory)
//!   reads its slot's kind first and branches to a handle of that kind: a
//!   constant buffer, a raw buffer (with the view's lead) SRV or UAV, or a
//!   typed one for typed accesses. A slot of another kind reads zeros and
//!   drops writes. A handle of another kind than the slot's view is
//!   undefined (a constant buffer read of a buffer SRV hangs NVIDIA GPUs:
//!   Grand Theft Auto VI's meshlet culling). Accesses a used-resource entry
//!   names keep the handle its kind gives.
//! * **Heap slots through user data** (added 2026-11-24,
//!   [`HostShader::heap_indices`]): the slots a shader reaches through its
//!   user data without a used-resource entry, as each slot's user-data
//!   dwords, masks, scales and offset; the host makes them, with their
//!   data, for each draw or dispatch. Slots whose index each lane loads, or
//!   the shader loads from memory, are counted ([`HostShader::heap_unknown`]).
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view. Global loads
//!   from a root SRV read it raw. 64-bit texture atomics (two dwords per
//!   value; added 2026-11-24) are SM 6.6's on typed heap resources: the
//!   host's view must be `R32G32_UINT` (low dword in R) and the device must
//!   report 64-bit atomics on typed resources (options 9) and on descriptor
//!   heap resources (options 11), else the pipeline is refused.
//!
//! * **Root parameters past the user SGPRs** (added 2026-11-24): a
//!   compute program whose root signature does not fit its 16 user SGPRs
//!   loads the rest from memory whose 40-bit address is in user data 0-1;
//!   the table starts with user-data dword 16. They are user data as any
//!   other: the host fills dword `index` as above.
//! * **Root UAVs** (raw buffers) as root descriptors `u<i>` in
//!   [`ROOT_CBV_SPACE`] ([`HostShader::root_uavs`]); what shaders write there
//!   is the title's to see.
//! * **Title memory** (added 2026-11-24, decision 0005's unified-memory
//!   item): accesses at addresses the program computes (pointers in root
//!   constants or loaded from memory, BVH nodes) go through a page table,
//!   the root SRV `t0` in [`MEMORY_SPACE`] ([`HostShader::memory`]), dwords:
//!   4096 level-1 entries for address bits 28-39 (a level-2 block number,
//!   0 for none), then blocks of 4096 entries for bits 16-27 (64 KiB pages),
//!   two dwords each: the resource-heap index of a raw SRV of the host
//!   memory holding the page, whose raw UAV is the next index, and the low
//!   dword of the title address that memory starts at (byte offset =
//!   address low dword - it). Block 0 names null views (its entry 0 is
//!   how the shader tells a miss); addresses at or above 2^40 use it. The
//!   root UAV `u0` in [`MEMORY_SPACE`] ([`HostShader::memory_writes`]) takes
//!   bit `block × 4096 + page` for each page a shader writes (pages of an
//!   access's first and last byte: what the host copies back) and for each
//!   page with no copy an access reached, or for an address whose 256 MiB
//!   region has no block, bit `level-1 index` (block 0's bits). Reads there
//!   give 0, writes are dropped. In such a program (and one with root
//!   parameters past its user SGPRs) root SRVs and UAVs are title memory
//!   too: no root views; the shader reads their addresses as user data
//!   (root type [`RootType::Descriptor`] in [`HostShader::user_data`],
//!   dword 0 the address's low dword, 1 the high), and the host must have
//!   a copy holding each such address.
//! * **View addresses** ([`HostShader::view_addresses`]): a buffer
//!   descriptor's dwords 0-1 read as numbers (the BVH SRV's base address)
//!   come from the root SRV `t1` in [`MEMORY_SPACE`]: per resource-heap
//!   index, two dwords, the buffer view's title address (high dword bits
//!   0-15; stride bits 0), zeros for no buffer view.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

#[cfg(all(test, windows))]
mod atomic64_tests;
mod bitcode;
#[cfg(all(test, windows))]
mod bvh_tests;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
mod halves;
#[cfg(all(test, windows))]
mod halves_tests;
mod heap;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
#[cfg(all(test, windows))]
mod join_tests;
#[cfg(all(test, windows))]
mod kind_tests;
mod lower;
mod rectangles;
#[cfg(all(test, windows))]
mod memory_tests;
mod sampler;
mod shader;
#[cfg(all(test, windows))]
mod stride_tests;
#[cfg(all(test, windows))]
mod typed_tests;
#[cfg(all(test, windows))]
mod synthetic_tests;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;
#[cfg(all(test, windows))]
mod wave32_tests;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use heap::{HeapIndices, heap_indices};
pub use rectangles::rectangles;
pub use sampler::StaticSampler;
pub use shader::{
    MEMORY_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE, VIEW_RECORD_BYTES,
    VIEW_STRIDE_SPACE, view_kind,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// Heap slots whose view addresses shaders may read: a slot index at or
/// past it reads the last one's. The host's view addresses (`t1` in
/// [`shader::MEMORY_SPACE`], a root view, which nothing bounds) hold at
/// least this many: an index read from memory the host has not mirrored
/// or holds stale reached past them (run 129, a device page fault).
pub const VIEW_ADDRESS_SLOTS: u32 = 1 << 20;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// Whether the change `name` ported from Belobug's fork is on: each is
/// unless `VICTORY_NO_BELOBUG` lists it (names separated by commas) or is
/// `1` or `all`, which translates as before them. The value is part of
/// the translation, so the shader cache must key on it (`victory-gpu`'s
/// does).
pub fn belobug(name: &str) -> bool {
    !belobug_off(no_belobug(), name)
}

/// `VICTORY_NO_BELOBUG`'s value, empty when it is unset.
pub fn no_belobug() -> &'static str {
    static OFF: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    OFF.get_or_init(|| std::env::var("VICTORY_NO_BELOBUG").unwrap_or_default())
}

/// Whether `value` (`VICTORY_NO_BELOBUG`'s) turns `name` off. Pure, for
/// testing.
fn belobug_off(value: &str, name: &str) -> bool {
    value
        .split(',')
        .map(str::trim)
        .any(|v| v == name || v == "1" || v == "all")
}

/// Whether a wave64 compute group of fewer than 32 threads whose program
/// reads other lanes runs on 32-lane halves: one host wave holds the whole
/// group, as the lower half, and the upper half has no lanes. Otherwise it
/// keeps the console's 64-lane waves, which hosts with only 32-lane waves
/// refuse (subsurfacescatter:CS_SSSTileClassification, 4x4 threads). On
/// unless `VICTORY_SMALL_GROUPS_WAVE64` is set. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn small_groups_on_halves() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SMALL_GROUPS_WAVE64").is_none())
}

/// Whether buffer loads of a byte or short (`buffer_load_ubyte`,
/// `buffer_load_ushort` and their signed forms) are translated, as the
/// dword holding them shifted and extended; refused otherwise. On unless
/// `VICTORY_NO_SUBDWORD_LOADS` is set. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn subdword_buffer_loads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_SUBDWORD_LOADS").is_none())
}

/// Whether a scalar a waterfall loop reads from a descriptor, which code
/// after the loop reads, is each lane's own pass's where it cannot be the
/// last pass's as on the console (a pixel program, a loop on halves or one
/// code skips around); refused otherwise. On unless `VICTORY_NO_OWN_PASS`
/// is set. The value is part of the translation, so the shader cache must
/// key on it (`victory-gpu`'s does).
pub fn own_pass_scalars() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_OWN_PASS").is_none())
}

/// Whether an image descriptor (eight dwords) the code reads only as data,
/// with no access using it, is taken as a 2D texture's, whose size the
/// code reads; refused otherwise. On unless `VICTORY_NO_SIZE_ONLY` is set.
/// The value is part of the translation, so the shader cache must key on it
/// (`victory-gpu`'s does).
pub fn size_only_textures() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_SIZE_ONLY").is_none())
}

/// Whether a descriptor control flow joins from one constant (an immediate
/// sampler) on every edge that lanes using it take, and nothing on the
/// others, is that constant; refused otherwise. On unless
/// `VICTORY_NO_JOINED_CONSTANTS` is set. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn joined_constants() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_JOINED_CONSTANTS").is_none())
}

/// Whether dwords of an image descriptor whose size the code reads pass
/// through instructions that do not read them as data: a waterfall loop's
/// compares of a descriptor each lane holds in VGPRs (true in every lane),
/// and an unguarded image load or LDS read into VGPRs that held them
/// (overwritten); refused otherwise. On unless `VICTORY_NO_UNREAD_FIELDS`
/// is set. The value is part of the translation, so the shader cache must
/// key on it (`victory-gpu`'s does).
pub fn unread_fields() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_UNREAD_FIELDS").is_none())
}

/// How mesh programs take the console's tests of whether a lane mask has a
/// lane set and its wave minimums, maximums, ANDs and ORs
/// (`VICTORY_MESH_WAVE_OPS`). The value is part of the translation, so the
/// shader cache must key on it (`victory-gpu`'s does).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshWaveOps {
    /// `0` or `off`: refused, as before they were taken (the tree imposters'
    /// MS_VS_OctaSLODWindFire and the other `*Fire` mesh programs drop).
    Off,
    /// `1`: over the host's waves. Run 136 hung the GPU in
    /// MS_VS_OctaSLODWindFire's fire-list waterfall with this.
    Wave,
    /// Unset, `lane` or any other value: a reduction is the lane's own value,
    /// so a waterfall over the distinct values of a lane value walks each
    /// lane's values alone.
    Lane,
}

/// `VICTORY_MESH_WAVE_OPS`'s setting.
pub fn mesh_wave_ops() -> MeshWaveOps {
    static OPS: std::sync::OnceLock<MeshWaveOps> = std::sync::OnceLock::new();
    *OPS.get_or_init(|| parse_mesh_wave_ops(std::env::var("VICTORY_MESH_WAVE_OPS").ok().as_deref()))
}

/// `VICTORY_MESH_WAVE_OPS`'s value read. Pure, for testing.
fn parse_mesh_wave_ops(value: Option<&str>) -> MeshWaveOps {
    match value.map(str::trim) {
        Some("0" | "off") => MeshWaveOps::Off,
        Some("1") => MeshWaveOps::Wave,
        _ => MeshWaveOps::Lane,
    }
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Whether the shader reads the view heap's records: a raw buffer `t0`
    /// in [`VIEW_STRIDE_SPACE`] over two dwords (stride and lead) per slot
    /// of the view heap it indexes (see the module notes). A heap index is
    /// whatever the program computes, so the host binds a view of them,
    /// which reads zeros past their end, not a root SRV, which reads on.
    pub view_strides: bool,
    /// The view-heap slots the shader reaches through its user data that no
    /// used-resource entry names (see [`HeapIndex`]): the host makes them,
    /// with their data, for each draw or dispatch.
    pub heap_indices: Vec<HeapIndex>,
    /// Heap descriptor accesses whose slot the host cannot know (each lane
    /// loads its descriptor, or its index from memory).
    pub heap_unknown: u32,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
    /// Whether the shader reads title memory at addresses it computes: the
    /// host binds the page table `t0` in [`MEMORY_SPACE`] (module notes,
    /// "Title memory").
    pub memory: bool,
    /// Whether the host binds the page bitmap `u0` in [`MEMORY_SPACE`]
    /// (pages written and pages missed): whenever [`HostShader::memory`].
    pub memory_writes: bool,
    /// Whether it reads buffer views' title addresses from their
    /// descriptors: the host binds the view addresses `t1` in
    /// [`MEMORY_SPACE`] (module notes, "Title memory").
    pub view_addresses: bool,
}

/// A view-heap slot as the user data gives it: `(offset + the sum of the
/// terms) >> 5`, modulo 2^32 before the shift (descriptors are 32 bytes on
/// the console). A heap index in root constant `u` is one term of `u`
/// times 32, the second entry of a table at `t` one term of `t` times 1
/// with offset 32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeapIndex {
    pub terms: Vec<HeapTerm>,
    pub offset: u32,
    /// Whether the shader writes through it (stores, atomics).
    pub write: bool,
}

/// A user-data dword (as [`UserData::index`]) AND `mask`, times `scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapTerm {
    pub user_data: u32,
    pub mask: u32,
    pub scale: u32,
}

impl HeapIndex {
    /// The slot, from the user data the shader reads (by dword).
    pub fn slot(&self, user_data: &[u32]) -> u32 {
        let at = self.terms.iter().fold(self.offset, |at, t| {
            let value = user_data.get(t.user_data as usize).copied().unwrap_or(0);
            at.wrapping_add((value & t.mask).wrapping_mul(t.scale))
        });
        at >> 5
    }
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants`, `Table` or, in a program using title memory,
    /// `Descriptor` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// How the console's 64-lane waves of a wave64 compute program whose
/// lanes interact (or which shares LDS within one wave) run on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waves {
    /// On host waves of any size: no such program.
    Any,
    /// Each as two 32-lane host waves ([`halves`]).
    Halves,
    /// The group's one console wave as two 32-lane host waves that follow
    /// its own branches ([`halves`]); several console waves that never
    /// meet in turn on such a pair, others each on a pair of its own.
    Lockstep,
    /// As 64-lane host waves (the program declares `WaveSize(64)`, which
    /// hosts with only 32-lane waves refuse), for this reason the halves
    /// cannot run it.
    Wave64(&'static str),
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit(&c, &native, &p, &st, &io, &res)
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with_waves(c, native, p, st, io, res).map(|e| e.0)
}

/// [`emit`], and how the program's waves run.
pub fn emit_with_waves(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<(HostShader, Waves), ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = default_interp(e);
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let heaps = heap_indices(p, native, res);
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
    })?;
    let shader = HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        view_strides: l.view_strides,
        heap_indices: heaps.known,
        heap_unknown: heaps.unknown.len() as u32,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
    };
    Ok((shader, l.waves))
}

/// The interpolation DXC declares by default for a value passed between
/// stages.
fn default_interp(e: &container::SigElement) -> u8 {
    match (e.kind, e.comp) {
        (3, _) => 4,
        // Constant, as DXC declares it (the AMD compiler crashed on a
        // front face without a mode).
        (13, _) => 1,
        // Until a read says which mode (`lower`).
        (28, _) => 2,
        (0 | 6 | 7, 9) => 2,
        (0, _) => 1,
        _ => 0,
    }
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_belobug_switch_turns_off_what_it_lists() {
        assert!(!super::belobug_off("", "quad"));
        assert!(super::belobug_off("readfirstlane, quad", "quad"));
        assert!(!super::belobug_off("quadruple", "quad"));
        assert!(super::belobug_off("1", "quad"));
        assert!(super::belobug_off("all", "readfirstlane"));
    }

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }

    #[test]
    fn the_mesh_wave_ops_switch_parses_or_takes_each_lane() {
        use super::{MeshWaveOps, parse_mesh_wave_ops};
        assert_eq!(parse_mesh_wave_ops(None), MeshWaveOps::Lane);
        assert_eq!(parse_mesh_wave_ops(Some("1")), MeshWaveOps::Wave);
        assert_eq!(parse_mesh_wave_ops(Some("wave")), MeshWaveOps::Lane);
        assert_eq!(parse_mesh_wave_ops(Some(" 0 ")), MeshWaveOps::Off);
        assert_eq!(parse_mesh_wave_ops(Some("off")), MeshWaveOps::Off);
        assert_eq!(parse_mesh_wave_ops(Some("lane")), MeshWaveOps::Lane);
    }

    /// A heap index's slot is its offset plus each term's masked user-data
    /// dword times its scale, in bytes of 32 per slot, modulo 2^32; user
    /// data past the stage's reads as 0.
    #[test]
    fn heap_indices_give_the_slot_their_user_data_names() {
        use super::{HeapIndex, HeapTerm};
        let h = HeapIndex {
            terms: vec![
                HeapTerm {
                    user_data: 1,
                    mask: 0x3ff,
                    scale: 32,
                },
                HeapTerm {
                    user_data: 0,
                    mask: u32::MAX,
                    scale: 1,
                },
            ],
            offset: 32,
            write: false,
        };
        assert_eq!(h.slot(&[0, 7]), 8);
        assert_eq!(h.slot(&[64, 0xABC0_0007]), 10);
        assert_eq!(h.slot(&[0u32.wrapping_sub(64), 7]), 6);
        assert_eq!(h.slot(&[96]), 4);
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
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
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
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
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader whose 64-bit shift takes J (the second perspective
    /// barycentric, v1) as the high dword of a pair and keeps only the low
    /// dword, as the console compiler does with a register whose value it
    /// leaves undefined. The high dword is shifted out, so every pixel is 40
    /// whether the read takes 0 or the barycentric.
    #[test]
    #[cfg(windows)]
    fn barycentric_registers_reused_for_undefined_data_read_as_zero() {
        let words = [
            0x7E00_0285u32, // v_mov_b32 v0, 5
            0xD6FF_0002,
            0x0002_0083, // v_lshlrev_b64 v[2:3], 3, v[0:1]
            0x7E08_0D02, // v_cvt_f32_u32 v4, v2
            0xF800_180F,
            0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
            0xBF81_0000,
        ];
        let el = |name: &str, kind, cols, usage| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols,
            usage,
            half: false,
        };
        let emit_with = |inputs: &[SigElement], system_value: Option<u32>| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 5,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![],
                    // PERSP_CENTER: I and J in v0 and v1.
                    input_addr: 1 << 1,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let mut isg1 = super::container::signature(inputs, true);
            if let Some(sv) = system_value {
                // The first element's system value.
                let base = u32::from_le_bytes(isg1[4..8].try_into().unwrap()) as usize;
                isg1[base + 12..base + 16].copy_from_slice(&sv.to_le_bytes());
            }
            let osg1 = super::container::signature(&[el("SV_Target", 16, 4, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::emit(&c, &native, &p, &st, &io, &res)
        };
        // D3D_NAME_BARYCENTRICS: the console's own row is parsed and reused
        // (Belobug 416ac79), no longer refused.
        let reused = emit_with(&[el("SV_Barycentrics", 0, 3, 0)], Some(23)).map(|_| ());
        assert!(reused.is_ok(), "{reused:?}");
        let ps = emit_with(&[], None).unwrap();
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut ps.dxil.clone()).unwrap();
        }
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let vs = super::host::position_vs();
        let (root, pso) = host.pipeline(&vs, &ps).unwrap();
        let vertices = [
            -1.0f32, -1.0, 0.5, 1.0, //
            -1.0, 3.0, 0.5, 1.0, //
            3.0, -1.0, 0.5, 1.0,
        ];
        let texels = host
            .draw(
                &root,
                &pso,
                &super::host::Draw {
                    roots: vec![],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (4, 4),
                    mesh_groups: 0,
                },
            )
            .unwrap();
        assert!(texels.iter().all(|t| *t == [40.0; 4]), "{texels:?}");
    }

    /// A pixel shader that spills the lane mask `x < 2` to lane 1 of a VGPR
    /// in a branch only pixels with y < 2 take, restores it with
    /// `v_readlane` there, joins it with the unspilled mask the other path
    /// keeps and sets v4 to 2 under it (the textured, fog and deferred
    /// programs spill their masks so). The restored mask is the spilled one
    /// bit for bit, lanes outside the branch included: pixels with x < 2 are
    /// 2, the others 1.
    #[test]
    #[cfg(windows)]
    fn a_lane_mask_restored_from_a_vgpr_lane_is_the_spilled_mask() {
        let words = [
            0xD404_0004u32,
            0x0002_00F4, // v_cmp_gt_f32 s4, 2.0, v0
            0x7E08_02F2, // v_mov_b32 v4, 1.0
            0x7C08_02F4, // v_cmp_gt_f32 vcc_lo, 2.0, v1
            0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
            0xBF88_0005, // s_cbranch_execz join
            0xD761_0005,
            0x0001_0204, // v_writelane_b32 v5, s4, 1
            0xBE84_0380, // s_mov_b32 s4, 0
            0xD760_0004,
            0x0001_0305, // v_readlane_b32 s4, v5, 1
            0x887E_0A7E, // join: s_or_b32 exec_lo, exec_lo, s10
            0xBE8C_3C04, // s_and_saveexec_b32 s12, s4
            0x7E08_02F4, // v_mov_b32 v4, 2.0
            0xBEFE_030C, // s_mov_b32 exec_lo, s12
            0xF800_180F,
            0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
            0xBF81_0000,
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 6,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![],
                // POS_X_FLOAT and POS_Y_FLOAT: the pixel's x and y in v0 and v1.
                input_addr: 1 << 8 | 1 << 9,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, interp, usage| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(0),
            col: 0,
            cols: 4,
            usage,
            half: false,
        };
        let isg1 = super::container::signature(&[el("SV_Position", 3, 4, 3)], true);
        let osg1 = super::container::signature(&[el("SV_Target", 16, 0, 15)], false);
        let bytes =
            crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let ps = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut ps.dxil.clone()).unwrap();
        }
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let vs = super::host::position_vs();
        let (root, pso) = host.pipeline(&vs, &ps).unwrap();
        let vertices = [
            -1.0f32, -1.0, 0.5, 1.0, //
            -1.0, 3.0, 0.5, 1.0, //
            3.0, -1.0, 0.5, 1.0,
        ];
        let texels = host
            .draw(
                &root,
                &pso,
                &super::host::Draw {
                    roots: vec![],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (4, 4),
                    mesh_groups: 0,
                },
            )
            .unwrap();
        let want: Vec<[f32; 4]> = (0..16)
            .map(|i| [if i % 4 < 2 { 2.0 } else { 1.0 }; 4])
            .collect();
        assert_eq!(texels, want);
    }

    /// A pixel shader that spills 2.0 and 1.0 to lanes 1 and 0 of a VGPR
    /// (`v_writelane`), spills 4.0 to its lane 3 in a branch only pixels
    /// with y < 2 take, then restores lanes 1 and 0 after the join
    /// (`v_readlane`) and exports them: every pixel reads the spilled
    /// scalars, as the textured and ped decal programs do with descriptor
    /// offsets they spill.
    #[test]
    #[cfg(windows)]
    fn scalars_restored_from_vgpr_lanes_in_a_pixel_shader_are_the_spilled_ones() {
        let words = [
            0xBE84_03F4u32, // s_mov_b32 s4, 2.0
            0xD761_0005,
            0x0001_0204, // v_writelane_b32 v5, s4, 1
            0xBE84_03F2, // s_mov_b32 s4, 1.0
            0xD761_0005,
            0x0001_0004, // v_writelane_b32 v5, s4, 0
            0xBE84_0380, // s_mov_b32 s4, 0
            0x7C08_02F4, // v_cmp_gt_f32 vcc_lo, 2.0, v1
            0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
            0xBF88_0003, // s_cbranch_execz join
            0xBE85_03F6, // s_mov_b32 s5, 4.0
            0xD761_0005,
            0x0001_0605, // v_writelane_b32 v5, s5, 3
            0x887E_0A7E, // join: s_or_b32 exec_lo, exec_lo, s10
            0xD760_0006,
            0x0001_0305, // v_readlane_b32 s6, v5, 1
            0xD760_0007,
            0x0001_0105, // v_readlane_b32 s7, v5, 0
            0x7E08_0206, // v_mov_b32 v4, s6
            0x7E0C_0207, // v_mov_b32 v6, s7
            0xF800_180F,
            0x0604_0604, // exp mrt0, v4, v6, v4, v6 done vm
            0xBF81_0000,
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 7,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![],
                // POS_X_FLOAT and POS_Y_FLOAT: the pixel's x and y in v0 and v1.
                input_addr: 1 << 8 | 1 << 9,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, interp, usage| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(0),
            col: 0,
            cols: 4,
            usage,
            half: false,
        };
        let isg1 = super::container::signature(&[el("SV_Position", 3, 4, 3)], true);
        let osg1 = super::container::signature(&[el("SV_Target", 16, 0, 15)], false);
        let bytes =
            crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let ps = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut ps.dxil.clone()).unwrap();
        }
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let vs = super::host::position_vs();
        let (root, pso) = host.pipeline(&vs, &ps).unwrap();
        let vertices = [
            -1.0f32, -1.0, 0.5, 1.0, //
            -1.0, 3.0, 0.5, 1.0, //
            3.0, -1.0, 0.5, 1.0,
        ];
        let texels = host
            .draw(
                &root,
                &pso,
                &super::host::Draw {
                    roots: vec![],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (4, 4),
                    mesh_groups: 0,
                },
            )
            .unwrap();
        assert_eq!(texels, vec![[2.0, 1.0, 2.0, 1.0]; 16]);
    }

    /// A pixel shader reading 16-bit inputs packed two per dword, laid out
    /// as in the ped programs. TEXCOORD0 (3 channels) and TEXCOORD1 (1)
    /// share a row and TEXCOORD2 is packed after them; TEXCOORD3 and
    /// TEXCOORD4 (2 each) share the second parameter, whose stage data names
    /// no second row, as the ped decal programs' first one does. TEXCOORD1
    /// would start in a high half, so it takes a whole dword: the third of
    /// the second parameter, its own row's two being used up. Every pixel
    /// exports TEXCOORD1.x, TEXCOORD4.x, TEXCOORD2.w and TEXCOORD4.y.
    #[test]
    #[cfg(windows)]
    fn packed_16_bit_inputs_come_from_the_halves_the_console_puts_them_in() {
        let words = [
            0x7E06_0280u32, // v_mov_b32 v3, 0
            0xD742_0002,
            0x0002_0081, // v_interp_p1ll_f16 v2, v0, attr1.z
            0xD75A_0003,
            0x040A_0281, // v_interp_p2_f16 v3, v1, attr1.z, v2
            0x7E08_1703, // v_cvt_f32_f16 v4, v3
            0xD742_0002,
            0x0002_0041, // v_interp_p1ll_f16 v2, v0, attr1.y
            0xD75A_0003,
            0x040A_0241, // v_interp_p2_f16 v3, v1, attr1.y, v2
            0x7E0A_1703, // v_cvt_f32_f16 v5, v3
            0xD742_0002,
            0x0002_01C0, // v_interp_p1ll_f16 v2, v0, attr0.w high
            0xD75A_0003,
            0x040A_03C0, // v_interp_p2_f16 v3, v1, attr0.w, v2 high
            0x7E0C_1703, // v_cvt_f32_f16 v6, v3
            0xD742_0002,
            0x0002_0141, // v_interp_p1ll_f16 v2, v0, attr1.y high
            0xD75A_0003,
            0x040A_0341, // v_interp_p2_f16 v3, v1, attr1.y, v2 high
            0x7E0E_1703, // v_cvt_f32_f16 v7, v3
            0xF800_180F,
            0x0706_0504, // exp mrt0, v4, v5, v6, v7 done vm
            0xBF81_0000,
        ];
        let input = |row, attr, row2| crate::native::PsInput {
            usage: crate::native::Usage::Generic,
            row,
            mask: 3,
            attr,
            flat: false,
            fp16: true,
            pack2: true,
            row2,
        };
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![input(0, 0, 1), input(2, 1, 2)],
                // PERSP_CENTER: i and j in v0 and v1.
                input_addr: 1 << 1,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, index, kind, row, col, cols| SigElement {
            name: name.into(),
            index,
            kind,
            comp: 9,
            interp: if kind == 3 { 4 } else { 2 },
            row: Some(row),
            col,
            cols,
            usage: (1u8 << cols) - 1,
            half: false,
        };
        // (index, row, first channel, channels)
        let texcoords = [
            (0, 0, 0, 3),
            (1, 0, 3, 1),
            (2, 1, 0, 4),
            (3, 2, 0, 2),
            (4, 3, 0, 2),
        ];
        let mut elements: Vec<SigElement> = texcoords
            .iter()
            .map(|&(index, row, col, cols)| el("TEXCOORD", index, 0, row, col, cols))
            .collect();
        elements.push(el("SV_Position", 0, 3, 4, 0, 4));
        // The console's texture coordinates are 16-bit floats (type 6).
        let mut isg1 = super::container::signature(&elements, true);
        for i in 0..texcoords.len() {
            isg1[8 + 32 * i + 16] = 6;
        }
        let osg1 = super::container::signature(&[el("SV_Target", 0, 16, 0, 0, 4)], false);
        let bytes =
            crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let ps = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut ps.dxil.clone()).unwrap();
        }
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        // A vertex shader passing POSITION on as the position and writing
        // 1.0, 2.0, ... to the texture coordinates' channels in order.
        let mut s = super::shader::Shader::new(false);
        let b = s.entry;
        for col in 0..4 {
            let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            let x = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
            let args = [s.m.i32(5), s.m.i32(0), s.m.i8(col), x];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        let mut value = 1.0f32;
        for (id, &(_, _, _, cols)) in texcoords.iter().enumerate() {
            for col in 0..cols {
                let args = [
                    s.m.i32(id as u32),
                    s.m.i32(0),
                    s.m.i8(col),
                    s.m.f32(value.to_bits()),
                ];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
                value += 1.0;
            }
        }
        s.inputs.push(SigElement {
            interp: 0,
            ..el("POSITION", 0, 0, 0, 0, 4)
        });
        s.outputs = elements;
        s.m.terminate(b, super::bitcode::Term::Ret);
        let vs = super::HostShader {
            stage: super::HwStage::Vs,
            dxil: s.finish(),
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
        };
        let (root, pso) = host.pipeline(&vs, &ps).unwrap();
        let vertices = [
            -1.0f32, -1.0, 0.5, 1.0, //
            -1.0, 3.0, 0.5, 1.0, //
            3.0, -1.0, 0.5, 1.0,
        ];
        let texels = host
            .draw(
                &root,
                &pso,
                &super::host::Draw {
                    roots: vec![],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (4, 4),
                    mesh_groups: 0,
                },
            )
            .unwrap();
        assert_eq!(texels, vec![[4.0, 11.0, 8.0, 12.0]; 16]);
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    /// The console packs an integer beside a float in one signature row
    /// (TEXCOORD0 in x and y of row 1, TEXCOORD1 in z and w); DXIL wants one
    /// interpolation mode per row. A mesh program writes both, its position's
    /// x and y, then 7 and 9. One pixel program reads the integers flat and
    /// never reads the floats, another reads only the floats, interpolated.
    /// Each pair validates and links where the console put the values: the
    /// first sees the integers, the second the floats interpolated across
    /// the triangle.
    #[test]
    #[cfg(windows)]
    fn a_row_with_an_integer_beside_a_float_passes_both_through_a_linked_pair() {
        use crate::native::{PsInput, Usage, VsOutput};
        let el = |name: &str, index, kind, comp, row, col, cols| SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half: false,
        };
        let floats = el("TEXCOORD", 0, 0, 9, 1, 0, 2);
        let integers = el("TEXCOORD", 1, 0, 5, 1, 2, 2);
        let translate = |native: NativeProgram, parts: &[(FourCC, &[u8])]| {
            let bytes = crate::dxbc::build(parts);
            let c = Container::parse(&bytes).unwrap();
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            out
        };
        let ms_words = [
            0xD765_000Bu32,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v11, -1, 0
            0xD766_000B,
            0x0002_16C1, // v_mbcnt_hi_u32_b32 v11, -1, v11
            0x7E28_02F6, // v_mov_b32 v20, 4.0
            0x7D84_1682, // v_cmp_eq_u32 vcc, 2, v11
            0x0206_28F3, // v_cndmask_b32 v3, -1.0, v20, vcc
            0x7D84_1681, // v_cmp_eq_u32 vcc, 1, v11
            0x0208_28F3, // v_cndmask_b32 v4, -1.0, v20, vcc
            0x7E0A_02F0, // v_mov_b32 v5, 0.5
            0x7E0C_02F2, // v_mov_b32 v6, 1.0
            0x7E0E_02FF,
            0x0020_0400, // v_mov_b32 v7, 0x200400 (vertices 0, 1, 2)
            0x7E18_0287, // v_mov_b32 v12, 7
            0x7E1A_0289, // v_mov_b32 v13, 9
            0x7D88_1683, // v_cmp_gt_u32 vcc, 3, v11
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0xF800_020F,
            0x0D0C_0403, // exp param0, v3, v4, v12, v13
            0xF800_08CF,
            0x0605_0403, // exp pos0, v3, v4, v5, v6 done
            0x7D84_1680, // v_cmp_eq_u32 vcc, 0, v11
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0xF800_0941,
            0x0000_0007, // exp prim, v7, off, off, off done
            0xBF81_0000, // s_endpgm
        ];
        let ms_native = NativeProgram {
            stage: HwStage::Ms,
            wave64: true,
            vgprs: 24,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: vec![VsOutput {
                    usage: Usage::Generic,
                    row: 1,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
                clip_cntl: 0,
                threads: 64,
                max_vertices: 3,
                max_primitives: 1,
                topology: 2,
                lds_bytes: 0,
            }),
            code: ms_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = el("SV_Position", 0, 3, 9, 0, 0, 4);
        let osg1 = container::signature(&[position, floats.clone(), integers.clone()], false);
        let ms = translate(ms_native, &[(FourCC::OSG1, &osg1)]);
        let pixel = |words: &[u32], inputs: &[SigElement]| {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![PsInput {
                        usage: Usage::Generic,
                        row: 1,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    // PERSP_CENTER: I and J in v0 and v1.
                    input_addr: 1 << 1,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let isg1 = container::signature(inputs, true);
            let osg1 = container::signature(&[el("SV_Target", 0, 16, 9, 0, 0, 4)], false);
            translate(native, &[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)])
        };
        let flat = pixel(
            &[
                0xC80A_0202, // v_interp_mov_f32 v2, p0, attr0.z
                0xC80E_0302, // v_interp_mov_f32 v3, p0, attr0.w
                0x7E04_0D02, // v_cvt_f32_u32 v2, v2
                0x7E06_0D03, // v_cvt_f32_u32 v3, v3
                0xF800_180F,
                0x0302_0302, // exp mrt0, v2, v3, v2, v3 done vm
                0xBF81_0000,
            ],
            &[floats.clone(), integers],
        );
        let interpolated = pixel(
            &[
                0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
                0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
                0xC80C_0100, // v_interp_p1_f32 v3, v0, attr0.y
                0xC80D_0101, // v_interp_p2_f32 v3, v1, attr0.y
                0xF800_180F,
                0x0302_0302, // exp mrt0, v2, v3, v2, v3 done vm
                0xBF81_0000,
            ],
            &[floats],
        );
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let draw = |ps: &HostShader| {
            let (root, pso) = host.mesh_pipeline(&ms, ps).unwrap();
            host.draw(
                &root,
                &pso,
                &super::host::Draw {
                    roots: vec![],
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (4, 4),
                    mesh_groups: 1,
                },
            )
            .unwrap()
        };
        let texels = draw(&flat);
        assert!(
            texels.iter().all(|t| *t == [7.0, 9.0, 7.0, 9.0]),
            "{texels:?}"
        );
        // The pixel centre's x and y on the target, as the position gives them.
        for (n, t) in draw(&interpolated).into_iter().enumerate() {
            let x = (2 * (n % 4) + 1) as f32 / 4.0 - 1.0;
            let y = 1.0 - (2 * (n / 4) + 1) as f32 / 4.0;
            assert!(
                (t[0] - x).abs() < 1e-5 && (t[1] - y).abs() < 1e-5 && t[2] == t[0],
                "texel {n}: {t:?}"
            );
        }
    }

    /// A mesh program may test whether a lane mask has a lane set, as a test
    /// that a value is uniform does, but not read how many lanes it has: its
    /// lanes run on host waves of another size.
    #[test]
    fn a_mesh_program_reads_whether_a_mask_has_a_lane_set_but_not_its_count() {
        use crate::native::{MeshIo, Usage, VsOutput};
        let el = |name: &str, index, kind, comp, row, col, cols| SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half: false,
        };
        let osg1 = container::signature(
            &[
                el("SV_Position", 0, 3, 9, 0, 0, 4),
                el("TEXCOORD", 0, 0, 9, 1, 0, 2),
                el("TEXCOORD", 1, 0, 5, 1, 2, 2),
            ],
            false,
        );
        let lower = |w: &[u32]| {
            let words: Vec<u32> = [
                0xD765_000Bu32,
                0x0001_00C1, // v_mbcnt_lo_u32_b32 v11, -1, 0
                0xD766_000B,
                0x0002_16C1, // v_mbcnt_hi_u32_b32 v11, -1, v11
                0x7E06_02F2, // v_mov_b32 v3, 1.0
                0x7E08_02F2, // v_mov_b32 v4, 1.0
                0x7E0A_02F0, // v_mov_b32 v5, 0.5
                0xBE80_107E, // s_bcnt1_i32_b64 s0, exec
            ]
            .iter()
            .chain(w)
            .chain(&[
                0x7E18_0287, // v_mov_b32 v12, 7
                0x7E1A_0289, // v_mov_b32 v13, 9
                0x7E0E_02FF,
                0x0020_0400, // v_mov_b32 v7, 0x200400 (vertices 0, 1, 2)
                0x7D88_1683, // v_cmp_gt_u32 vcc, 3, v11
                0x87FE_6A7E, // s_and_b64 exec, exec, vcc
                0xF800_020F,
                0x0D0C_0403, // exp param0, v3, v4, v12, v13
                0xF800_08CF,
                0x0605_0403, // exp pos0, v3, v4, v5, v6 done
                0x7D84_1680, // v_cmp_eq_u32 vcc, 0, v11
                0x87FE_6A7E, // s_and_b64 exec, exec, vcc
                0xF800_0941,
                0x0000_0007, // exp prim, v7, off, off, off done
                0xBF81_0000, // s_endpgm
            ])
            .copied()
            .collect();
            let native = NativeProgram {
                stage: HwStage::Ms,
                wave64: true,
                vgprs: 24,
                layout: vec![],
                used: vec![],
                io: StageIo::Ms(MeshIo {
                    outputs: vec![VsOutput {
                        usage: Usage::Generic,
                        row: 1,
                        param: 0,
                        fp16: false,
                        pack2: false,
                    }],
                    clip_cntl: 0,
                    threads: 64,
                    max_vertices: 3,
                    max_primitives: 1,
                    topology: 2,
                    lds_bytes: 0,
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
            let c = Container::parse(&bytes).unwrap();
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            emit(&c, &native, &p, &st, &io, &res)
        };
        let any_lane = lower(&[
            0x8501_F0F2, // s_cselect_b32 s1, 1.0, 0.5
            0x7E0C_0201, // v_mov_b32 v6, s1
        ]);
        if let Err(e) = any_lane {
            panic!("{e:?}");
        }
        match lower(&[0x7E0C_0200]) {
            // v_mov_b32 v6, s0
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(what, "a lane mask read as data");
            }
            other => panic!("{:?}", other.map(|_| ())),
        }
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(&[element("TEXCOORD", 0, 0)], true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }
}
