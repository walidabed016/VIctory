//! Unordered twins of depth-stencil textures, one a plane. Grand Theft Auto
//! VI's material classification (CS_MaterialTilesInitial and
//! CS_GenerateMaterialMasks) writes a D16_UNORM_S8_UINT texture's depth
//! plane as R16_UNORM and its stencil plane as R8_UINT, and its tiled
//! lighting draws each material's tiles at that material's depth with an
//! equal test. The host keeps the texture as D24S8, which cannot be
//! unordered, so those views write an R16 and an R8 twin
//! ([`crate::resource::unordered_twin`]), and the planes move between the
//! texture and its twins through buffers: 16-bit depth through a compute
//! pass that converts it from and to 24 bits. With
//! `VICTORY_NO_STENCIL_TWIN=1` there are no such twins and the host refuses
//! those views, as before.

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, transition};
use crate::object::This;

const D16_UNORM_S8_UINT: u32 = 118;
const R32G8X24_TYPELESS: u32 = 19;
const D32_FLOAT_S8X24_UINT: u32 = 20;

/// Whether depth-stencil textures get a twin a plane (not
/// `VICTORY_NO_STENCIL_TWIN=1`).
pub(crate) fn stencil_twins() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let off = std::env::var("VICTORY_NO_STENCIL_TWIN").is_ok_and(|v| v == "1");
        if off {
            tracing::info!(
                "no unordered twins of depth-stencil textures (VICTORY_NO_STENCIL_TWIN)"
            );
        }
        !off
    })
}

/// The format of the twin that unordered views of `plane` of a texture of
/// title format `format` write: the texture's own for depth textures, R16
/// or R32 for a depth-stencil texture's depth and R8 for its stencil (with
/// `stencil_twins`); `None` for planes a texture lacks.
pub(crate) fn twin_format(format: u32, plane: u32, stencil_twins: bool) -> Option<u32> {
    const R32_TYPELESS: u32 = 39;
    const R16_TYPELESS: u32 = 53;
    const R8_TYPELESS: u32 = 60;
    match (format, plane) {
        (D16_UNORM_S8_UINT | R32G8X24_TYPELESS | D32_FLOAT_S8X24_UINT, _) if !stencil_twins => None,
        (D16_UNORM_S8_UINT, 0) => Some(R16_TYPELESS),
        (R32G8X24_TYPELESS | D32_FLOAT_S8X24_UINT, 0) => Some(R32_TYPELESS),
        (D16_UNORM_S8_UINT | R32G8X24_TYPELESS | D32_FLOAT_S8X24_UINT, 1) => Some(R8_TYPELESS),
        (_, 0) => Some(format),
        _ => None,
    }
}

/// Whether `depth`'s twins hold a plane each ([`exchange`]) rather than
/// the whole texture: whether its host texture has a stencil plane.
pub(crate) fn split(depth: This) -> bool {
    crate::resource::host_resource(depth).is_some_and(|t| {
        // SAFETY: a live resource.
        let format = unsafe { t.GetDesc() }.Format.0 as u32;
        matches!(format, 19 | 20 | 44 | 45)
    })
}

/// The plane a texture's unordered view names: a 2D view's fourth word.
pub(crate) fn plane(words: Option<&[u32; 10]>) -> u32 {
    const TEXTURE_2D: u32 = 4;
    words.filter(|w| w[1] == TEXTURE_2D).map_or(0, |w| w[3])
}

struct Convert {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Convert {}
// SAFETY: as above.
unsafe impl Sync for Convert {}

/// Converts `height` rows of `width` depths between 24-bit (in 32-bit
/// texels) and 16-bit unsigned normalized ones, two texels a thread: to 24
/// bits as the host rasterizes a 16-bit value's depth, back to the nearest
/// 16-bit value.
const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=5, b0), SRV(t0), UAV(u0)"
cbuffer Rows : register(b0) { uint width; uint height; uint from_pitch; uint to_pitch; uint to_24; };
ByteAddressBuffer from : register(t0);
RWByteAddressBuffer to : register(u0);
uint d24(uint v) { return (v << 8) + (v + 128) / 257; }
uint d16(uint v) { return (uint)((v & 0xffffff) / 16777215.0 * 65535.0 + 0.5); }
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x * 2;
    if (x >= width || id.y >= height) {
        return;
    }
    bool pair = x + 1 < width;
    if (to_24 != 0) {
        uint both = from.Load(id.y * from_pitch + x * 2);
        to.Store(id.y * to_pitch + x * 4, d24(both & 0xffff));
        if (pair) {
            to.Store(id.y * to_pitch + x * 4 + 4, d24(both >> 16));
        }
    } else {
        uint first = d16(from.Load(id.y * from_pitch + x * 4));
        uint second = pair ? d16(from.Load(id.y * from_pitch + x * 4 + 4)) : 0;
        to.Store(id.y * to_pitch + x * 2, first | second << 16);
    }
}
"#;

fn convert() -> Option<&'static Convert> {
    static CONVERT: OnceLock<Option<Convert>> = OnceLock::new();
    CONVERT
        .get_or_init(|| {
            make()
                .inspect_err(|e| {
                    tracing::warn!(error = %e, "no depth conversion: 16-bit depth twins dropped");
                })
                .ok()
        })
        .as_ref()
}

fn make() -> windows::core::Result<Convert> {
    let device = &crate::host::host()
        .ok_or_else(windows::core::Error::empty)?
        .device;
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("planes"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )?
    };
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Convert { root, pso })
    }
}

struct Buffers(Vec<(bool, u64, ID3D12Resource)>);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Buffers {}

/// A buffer of `size` bytes the planes move through, one per size for the
/// texture's planes (`rows` false) and one for 16-bit rows. Each move
/// leaves it in the common state, and moves run in submission order.
fn buffer(rows: bool, size: u64) -> Option<ID3D12Resource> {
    static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers(Vec::new()));
    let mut buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, _, b)) = buffers.0.iter().find(|b| (b.0, b.1) == (rows, size)) {
        return Some(b.clone());
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain descriptions; the out pointer ours.
    unsafe {
        crate::host::host()?
            .device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .ok()?
    };
    let buffer = buffer?;
    buffers.0.push((rows, size, buffer.clone()));
    Some(buffer)
}

/// Copies plane `plane` of `depth` into its twin `twin`, or with `back`
/// the twin into that plane. Returns whether it recorded a compute pass,
/// which replaces the list's compute root signature and pipeline.
pub(crate) fn exchange(list: &HostList, depth: This, twin: This, plane: u32, back: bool) -> bool {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    const COMMON: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COMMON;
    let (Some(host), Some(texture), Some(other)) = (
        crate::host::host(),
        crate::resource::host_resource(depth),
        crate::resource::host_resource(twin),
    ) else {
        return false;
    };
    // SAFETY: live resources.
    let (desc, twin_desc) = unsafe { (texture.GetDesc(), other.GetDesc()) };
    if desc.MipLevels != 1 || desc.DepthOrArraySize != 1 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                ?desc,
                "plane twin of a texture with mips or layers: dropped"
            )
        });
        return false;
    }
    let mut planes = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
    let mut rows = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut size, mut rows_size) = (0, 0);
    // SAFETY: descriptions of live resources; out pointers ours.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            2,
            0,
            Some(planes.as_mut_ptr()),
            None,
            None,
            Some(&mut size),
        );
        host.device.GetCopyableFootprints(
            &twin_desc,
            0,
            1,
            0,
            Some(&mut rows),
            None,
            None,
            Some(&mut rows_size),
        );
    }
    let converts = twin_desc.Format.0 == 53;
    let (Some(at), Some(planes_buffer)) =
        (planes.get(plane as usize).copied(), buffer(false, size))
    else {
        return false;
    };
    let (from, to) = if back {
        (other, texture)
    } else {
        (texture, other)
    };
    let (from_index, to_index) = if back { (0, plane) } else { (plane, 0) };
    let location = |r: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer; the location never releases it.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |b: &ID3D12Resource, footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(b) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        }
    };
    let l = &list.list;
    crate::list::ensure_state(list, if back { twin } else { depth }, COPY_SOURCE);
    if !converts {
        let mut at = at;
        at.Footprint.Format = twin_desc.Format;
        // SAFETY: an open list; live resources; whole subresources of one
        // texel size.
        unsafe {
            l.ResourceBarrier(&[transition(
                &planes_buffer,
                COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            l.CopyTextureRegion(
                &placed(&planes_buffer, at),
                0,
                0,
                0,
                &location(from, from_index),
                None,
            );
            l.ResourceBarrier(&[transition(
                &planes_buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        crate::list::ensure_state(list, if back { depth } else { twin }, COPY_DEST);
        // SAFETY: as above.
        unsafe {
            l.CopyTextureRegion(
                &location(to, to_index),
                0,
                0,
                0,
                &placed(&planes_buffer, at),
                None,
            );
            l.ResourceBarrier(&[transition(
                &planes_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                COMMON,
            )]);
        }
        return false;
    }
    let (Some(c), Some(rows_buffer)) = (convert(), buffer(true, rows_size)) else {
        return false;
    };
    let (first, first_at, second, second_at) = if back {
        (&rows_buffer, rows, &planes_buffer, at)
    } else {
        (&planes_buffer, at, &rows_buffer, rows)
    };
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    let unordered = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let constants = [
        desc.Width as u32,
        desc.Height,
        first_at.Footprint.RowPitch,
        second_at.Footprint.RowPitch,
        u32::from(back),
    ];
    // SAFETY: an open list; live resources; the pass reads and writes
    // within the footprints.
    unsafe {
        l.ResourceBarrier(&[transition(first, COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        l.CopyTextureRegion(
            &placed(first, first_at),
            0,
            0,
            0,
            &location(from, from_index),
            None,
        );
        l.ResourceBarrier(&[
            transition(first, D3D12_RESOURCE_STATE_COPY_DEST, read),
            transition(second, COMMON, unordered),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, first.GetGPUVirtualAddress() + first_at.Offset);
        l.SetComputeRootUnorderedAccessView(2, second.GetGPUVirtualAddress() + second_at.Offset);
        l.Dispatch((desc.Width as u32).div_ceil(16), desc.Height.div_ceil(8), 1);
        l.ResourceBarrier(&[transition(
            second,
            unordered,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
    }
    crate::list::ensure_state(list, if back { depth } else { twin }, COPY_DEST);
    // SAFETY: as above.
    unsafe {
        l.CopyTextureRegion(
            &location(to, to_index),
            0,
            0,
            0,
            &placed(second, second_at),
            None,
        );
        l.ResourceBarrier(&[
            transition(first, read, COMMON),
            transition(second, D3D12_RESOURCE_STATE_COPY_SOURCE, COMMON),
        ]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Depth-stencil textures get a twin a plane, and none with
    /// `VICTORY_NO_STENCIL_TWIN=1`; depth textures one of their own format.
    #[test]
    fn depth_stencil_textures_get_a_twin_a_plane() {
        assert_eq!(twin_format(D16_UNORM_S8_UINT, 0, true), Some(53));
        assert_eq!(twin_format(D16_UNORM_S8_UINT, 1, true), Some(60));
        assert_eq!(twin_format(R32G8X24_TYPELESS, 0, true), Some(39));
        assert_eq!(twin_format(D32_FLOAT_S8X24_UINT, 1, true), Some(60));
        assert_eq!(twin_format(D16_UNORM_S8_UINT, 1, false), None);
        assert_eq!(twin_format(D16_UNORM_S8_UINT, 0, false), None);
        assert_eq!(twin_format(53, 0, false), Some(53));
        assert_eq!(twin_format(53, 1, true), None);
        assert_eq!(plane(Some(&[62, 4, 0, 1, 0, 0, 0, 0, 0, 0])), 1);
        assert_eq!(plane(Some(&[62, 5, 0, 0, 1, 1, 0, 0, 0, 0])), 0);
        assert_eq!(plane(None), 0);
    }
}
