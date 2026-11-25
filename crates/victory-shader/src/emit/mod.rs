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
//!   a heap sampler it uses must be a comparison sampler on the host.
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
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
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
mod sampler;
mod shader;
#[cfg(all(test, windows))]
mod stride_tests;
#[cfg(all(test, windows))]
mod typed_tests;
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
    ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE, VIEW_RECORD_BYTES, VIEW_STRIDE_SPACE, view_kind,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

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
    /// `Constants` or `Table` (see the module notes for the value).
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
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
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
    /// leaves undefined. The read takes 0 and every pixel is 40. That holds
    /// only without SV_Barycentrics in the signature, which is refused.
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
        // D3D_NAME_BARYCENTRICS.
        let refused = emit_with(&[el("SV_Barycentrics", 0, 3, 0)], Some(23)).map(|_| ());
        assert!(
            refused
                .as_ref()
                .is_err_and(|e| e.to_string().contains("a system value in the signature")),
            "{refused:?}"
        );
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
}
