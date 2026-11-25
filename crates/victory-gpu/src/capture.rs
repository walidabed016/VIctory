//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated), and the
/// world frames of `VICTORY_CAPTURE_WORLD`: saved too, and their draws
/// traced at debug level.
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    });
    list.contains(&n)
        || world().is_some_and(|(_, every)| {
            world_frame(
                FIRST_WORLD.load(std::sync::atomic::Ordering::Relaxed),
                every,
                n,
            )
        })
}

/// `VICTORY_CAPTURE_WORLD=N[,M]`: the draws a frame runs on the host that
/// mark the world's start (it moves between runs by thousands of frames),
/// and every how many frames the world is captured again.
fn world() -> Option<(u64, u64)> {
    static PARSED: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *PARSED.get_or_init(|| {
        let value = std::env::var("VICTORY_CAPTURE_WORLD").ok()?;
        let mut parts = value.split(',').map(|p| p.trim().parse::<u64>());
        let draws = parts.next()?.ok()?;
        let every = parts.next().and_then(Result::ok).unwrap_or(0);
        Some((draws, every))
    })
}

/// The first world frame, 0 until a frame has run `VICTORY_CAPTURE_WORLD`'s
/// draws.
static FIRST_WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// World captures after the first, `every` frames apart.
const WORLD_AGAIN: u64 = 4;

/// Whether frame `n` is a world frame: `start`, then every `every` frames,
/// [`WORLD_AGAIN`] times.
fn world_frame(start: u64, every: u64, n: u64) -> bool {
    if start == 0 || n < start {
        return false;
    }
    let after = n - start;
    after == 0 || every != 0 && after.is_multiple_of(every) && after / every <= WORLD_AGAIN
}

/// At the present of frame `frame`: the first frame that ran
/// `VICTORY_CAPTURE_WORLD`'s draws on the host makes the next one the first
/// world frame (this one's targets were not noted while it was built).
pub(crate) fn note_present(frame: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static DRAWN: AtomicU64 = AtomicU64::new(0);
    let Some((draws, every)) = world() else {
        return;
    };
    let drawn = crate::draw::drawn();
    let ran = drawn - DRAWN.swap(drawn, Ordering::Relaxed);
    if frame > 1 && ran >= draws && FIRST_WORLD.load(Ordering::Relaxed) == 0 {
        FIRST_WORLD.store(frame + 1, Ordering::Relaxed);
        tracing::info!(
            frame,
            draws = ran,
            first = frame + 1,
            every,
            again = WORLD_AGAIN,
            "VICTORY_CAPTURE_WORLD: the world started; capturing from the next frame"
        );
    }
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n))
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::write(&path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture`'s first subresource (in the copy-source
/// state) into a readback buffer: of a 3D texture, every slice.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut footprint), None, None, None);
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: rows_bytes(&footprint.Footprint),
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
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// The bytes of a footprint's rows: `Height` rows `RowPitch` apart per
/// slice, `Depth` slices one after the other, the last row's padding
/// included (GetCopyableFootprints' total leaves it out).
fn rows_bytes(f: &D3D12_SUBRESOURCE_FOOTPRINT) -> u64 {
    u64::from(f.RowPitch) * u64::from(f.Height) * u64::from(f.Depth)
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`,
/// `<w>x<h>x<d>` for 3D textures, whose `d` slices show one under the other
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

/// With `VICTORY_DUMP_TARGETS`: the title resources the pipelines that run
/// in the frame being built (a listed one) read and write, as (resource,
/// written, pipeline name), to name at the present who wrote and who read
/// each dumped target.
static ACCESSES: std::sync::Mutex<std::collections::BTreeSet<(usize, bool, String)>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// Whether pipeline `name` holds a program `VICTORY_DUMP_DATA` names
/// (comma-separated hashes): its constant and buffer data is logged, and
/// the dumped targets it reads or writes are saved raw too.
fn data_wanted(name: &str) -> bool {
    static HASHES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    HASHES
        .get_or_init(|| {
            let list = std::env::var("VICTORY_DUMP_DATA").unwrap_or_default();
            list.split(',')
                .map(|h| h.trim().to_owned())
                .filter(|h| !h.is_empty())
                .collect()
        })
        .iter()
        .any(|h| name.contains(h.as_str()))
}

/// Logs the words of a constant or buffer view (at most 2 KB) that the
/// pipeline being bound reads from table `slot` at byte `offset`, once a
/// view per frame, when it holds a program `VICTORY_DUMP_DATA` names and
/// the frame being built is listed. The words are title memory's when the
/// draw is recorded; data the GPU writes later in the frame is not in them.
pub(crate) fn note_data(slot: u8, offset: u32, address: u64, len: u64) {
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<(u64, u64, String)>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let frame = crate::queue::presented() + 1;
    if address == 0 || len == 0 || !dump_on() || !listed(frame) {
        return;
    }
    let pipeline = crate::descriptor::host::BINDING.get() as crate::object::This;
    let Some(name) = crate::pipeline::host_pipeline(pipeline).map(|p| p.name.clone()) else {
        return;
    };
    if !data_wanted(&name)
        || !SEEN
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((frame, address, name.clone()))
    {
        return;
    }
    // SAFETY: a view's range in title memory, which the host copies from.
    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, len.min(2048) as usize) };
    let words: Vec<String> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| format!("{:08x}", u32::from_le_bytes(*w)))
        .collect();
    tracing::info!(
        frame,
        pipeline = %name,
        slot,
        offset = format_args!("{offset:#x}"),
        address = format_args!("{address:#x}"),
        len,
        words = %words.join(" "),
        "shader data"
    );
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() || !dump_on() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    note_gpu_write(resource);
    if !listed(crate::queue::presented() + 1) {
        return;
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// Notes that the pipeline being bound reads or writes (`write`)
/// `resource` in the frame being built.
pub(crate) fn note_access(resource: crate::object::This, write: bool) {
    if resource.is_null() || !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let pipeline = crate::descriptor::host::BINDING.get() as crate::object::This;
    let name = crate::pipeline::host_pipeline(pipeline)
        .map_or_else(|| format!("{pipeline:?}"), |p| p.name.clone());
    let mut accesses = ACCESSES.lock().unwrap_or_else(|e| e.into_inner());
    accesses.insert((resource as usize, write, name));
}

/// With `VICTORY_DUMP_TARGETS`, from the start: the frame the host GPU last
/// wrote each title texture (render target, depth, unordered view, a copy
/// or resolve), by resource, and the frame each host texture last took
/// title memory (an upload), by host texture; for the slot lines.
#[cfg(feature = "diagnostics")]
static GPU_WROTE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> =
    std::sync::Mutex::new(None);
#[cfg(feature = "diagnostics")]
static UPLOADS: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> =
    std::sync::Mutex::new(None);

/// The host GPU writes the title texture `resource` in the frame being
/// built (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn note_gpu_write(resource: crate::object::This) {
    if resource.is_null() || !dump_on() {
        return;
    }
    let mut map = GPU_WROTE.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(Default::default)
        .insert(resource as usize, crate::queue::presented() + 1);
}

/// Title memory is copied into the host texture `texture` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn note_upload(texture: &ID3D12Resource) {
    use windows::core::Interface;
    if !dump_on() {
        return;
    }
    let mut map = UPLOADS.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(Default::default)
        .insert(texture.as_raw() as usize, crate::queue::presented() + 1);
}

/// A texture slot a lighting program reads in a listed frame: its view
/// format and the programs that read it, with the host copy held.
#[cfg(feature = "diagnostics")]
struct SlotSeen {
    format: Option<u32>,
    programs: std::collections::BTreeSet<String>,
    copy: Option<std::sync::Arc<crate::resource::HostTexture>>,
}

/// (program family, register, root parameter, table offset in dwords,
/// resource: 0 without a texture view).
#[cfg(feature = "diagnostics")]
type SlotKey = (&'static str, u32, u8, u32, usize);

#[cfg(feature = "diagnostics")]
static SLOTS: std::sync::Mutex<std::collections::BTreeMap<SlotKey, SlotSeen>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The family of a pipeline named with its shaders' names
/// ([`crate::host::named`]): TL_PS, SkyMain, the material tiles' OR, the
/// first V-buffer resolve, or a deferred textured pixel shader.
#[cfg(feature = "diagnostics")]
fn lighting_family(named: &str) -> Option<&'static str> {
    if named.contains("\"TL_PS") {
        Some("TL_PS")
    } else if named.contains("CS_SkyMain") {
        Some("SkyMain")
    } else if named.contains("CS_MaterialTiles_") {
        Some("MaterialTiles")
    } else if named.contains("CS_ResolveToGBufferTiled_0_") {
        Some("Resolve")
    } else if named.contains("PS_DeferredTextured") {
        Some("DeferredTextured")
    } else {
        None
    }
}

/// Families whose textures are read back after the lighting ones.
#[cfg(feature = "diagnostics")]
fn material_family(family: &str) -> bool {
    matches!(family, "Resolve" | "DeferredTextured")
}

/// Notes the texture slot `u` a table gives the pipeline being bound
/// (`view` `None`: an entry without a view), if the pipeline is TL_PS or
/// SkyMain by `VICTORY_SHADER_NAMES` and the frame is listed.
#[cfg(feature = "diagnostics")]
pub(crate) fn note_slot(view: Option<&crate::descriptor::View>, u: &victory_shader::UsedResource) {
    use crate::descriptor::View;
    if u.kind != victory_shader::ResourceType::Srv
        || u.dwords != 8
        || !dump_on()
        || !listed(crate::queue::presented() + 1)
    {
        return;
    }
    type Families = std::collections::HashMap<usize, Option<(&'static str, String)>>;
    static FAMILIES: std::sync::Mutex<Option<Families>> = std::sync::Mutex::new(None);
    let pipeline = crate::descriptor::host::BINDING.get();
    let mut families = FAMILIES.lock().unwrap_or_else(|e| e.into_inner());
    let families = families.get_or_insert_with(Default::default);
    if !families.contains_key(&pipeline) {
        let Some(p) = crate::pipeline::host_pipeline(pipeline as crate::object::This) else {
            return;
        };
        let family = lighting_family(&crate::host::named(&p.name));
        let taken = family == Some("DeferredTextured")
            && families
                .values()
                .any(|f| matches!(f, Some(("DeferredTextured", _))));
        families.insert(
            pipeline,
            family.filter(|_| !taken).map(|f| (f, p.name.clone())),
        );
    }
    let Some(Some((family, name))) = families.get(&pipeline).cloned() else {
        return;
    };
    let placed = view.and_then(crate::descriptor::host::placed_texture);
    let (resource, format) = match placed.as_ref().or(view) {
        Some(View::Shader(r, words)) => (*r, words.map(|w| w[0])),
        _ => (0, None),
    };
    let key = (
        family,
        u.lower_bound,
        u.slot,
        u.table_offset_dwords,
        resource,
    );
    let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    slots
        .entry(key)
        .or_insert_with(|| SlotSeen {
            format,
            programs: Default::default(),
            copy: crate::resource::host_copy(resource as crate::object::This),
        })
        .programs
        .insert(name);
}

/// (program, depth texture, whether it has an unordered twin, description)
/// of the unordered views of depth textures in a listed frame.
#[cfg(feature = "diagnostics")]
static DEPTH_UAVS: std::sync::Mutex<std::collections::BTreeSet<(String, usize, bool, String)>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Notes the unordered view of a depth texture the bound draw or dispatch
/// writes in a listed frame, and whether the texture has an unordered twin
/// (without one, the host view goes to the depth texture itself).
#[cfg(feature = "diagnostics")]
pub(crate) fn note_depth_unordered(view: &crate::descriptor::View) {
    let crate::descriptor::View::Unordered(resource, _, words) = *view else {
        return;
    };
    if !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let resource = resource as crate::object::This;
    const ALLOW_DEPTH_STENCIL: i32 = 0x2;
    let Some(texture) = crate::resource::host_resource(resource) else {
        return;
    };
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Flags.0 & ALLOW_DEPTH_STENCIL == 0 {
        return;
    }
    let twin = crate::resource::unordered_twin(resource).is_some();
    let pipeline = crate::descriptor::host::BINDING.get();
    let program = crate::pipeline::host_pipeline(pipeline as crate::object::This)
        .map(|p| crate::host::named(&p.name).to_string())
        .unwrap_or_default();
    let description = format!(
        "{}x{} f{} flags {:#x}, view {:?}",
        d.Width,
        d.Height,
        d.Format.0,
        d.Flags.0,
        words.map(|w| [w[0], w[1], w[2], w[3]])
    );
    DEPTH_UAVS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((program, resource as usize, twin, description));
}

/// Who last wrote a texture: the host GPU in `frame` (`this_frame`, or
/// `gpu` there), the GPU or an upload of title memory in an earlier
/// frame, or nothing.
#[cfg(feature = "diagnostics")]
fn last_writer(frame: u64, this_frame: bool, gpu: Option<u64>, cpu: Option<u64>) -> String {
    if this_frame || gpu == Some(frame) {
        return "gpu this frame".into();
    }
    match (gpu, cpu) {
        (Some(g), Some(c)) if c > g => format!("cpu upload, frame {c} (gpu frame {g})"),
        (Some(g), Some(c)) => format!("gpu, frame {g} (cpu upload frame {c})"),
        (Some(g), None) => format!("gpu, frame {g}"),
        (None, Some(c)) => format!("cpu upload, frame {c}"),
        (None, None) => "never".into(),
    }
}

/// Bytes per texel, channels, and the decoder of a texel's channels.
#[cfg(feature = "diagnostics")]
type TexelDecoder = (usize, usize, fn(&[u8]) -> [f32; 4]);

/// A texel's channels as floats, NaN and infinity kept; `None` for formats
/// not handled (block compressed).
#[cfg(feature = "diagnostics")]
fn texel_decoder(format: u32) -> Option<TexelDecoder> {
    fn u16_at(b: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
    }
    fn u32_at(b: &[u8], i: usize) -> u32 {
        u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    }
    fn f16(h: u16) -> f32 {
        let sign = u32::from(h >> 15) << 31;
        let (exp, man) = (u32::from(h >> 10 & 0x1f), u32::from(h & 0x3ff));
        match exp {
            0 => f32::from_bits(sign | (man as f32 / 16_777_216.0).to_bits()),
            31 => f32::from_bits(sign | 0x7f80_0000 | man << 13),
            e => f32::from_bits(sign | (e + 112) << 23 | man << 13),
        }
    }
    fn small(bits: u32, man: u32) -> f32 {
        let (exp, m) = (bits >> man & 0x1f, bits & ((1 << man) - 1));
        match exp {
            0 => m as f32 / (1u32 << (14 + man)) as f32,
            31 => f32::from_bits(0x7f80_0000 | m << (23 - man)),
            e => f32::from_bits((e + 112) << 23 | m << (23 - man)),
        }
    }
    Some(match format {
        2 => (16, 4, |b| {
            [0, 1, 2, 3].map(|i| f32::from_bits(u32_at(b, i)))
        }),
        6 => (12, 3, |b| {
            [0, 1, 2, 0].map(|i| f32::from_bits(u32_at(b, i)))
        }),
        9 | 10 => (8, 4, |b| [0, 1, 2, 3].map(|i| f16(u16_at(b, i)))),
        11..=14 => (8, 4, |b| [0, 1, 2, 3].map(|i| f32::from(u16_at(b, i)))),
        15 | 16 => (8, 2, |b| {
            [
                f32::from_bits(u32_at(b, 0)),
                f32::from_bits(u32_at(b, 1)),
                0.0,
                0.0,
            ]
        }),
        19..=22 => (8, 1, |b| [f32::from_bits(u32_at(b, 0)), 0.0, 0.0, 0.0]),
        23..=25 | 89 => (4, 4, |b| {
            let v = u32_at(b, 0);
            [v & 0x3ff, v >> 10 & 0x3ff, v >> 20 & 0x3ff, v >> 30].map(|c| c as f32)
        }),
        26 => (4, 3, |b| {
            let v = u32_at(b, 0);
            [
                small(v & 0x7ff, 6),
                small(v >> 11 & 0x7ff, 6),
                small(v >> 22, 5),
                0.0,
            ]
        }),
        27..=32 | 87 | 88 | 90..=93 => (4, 4, |b| [b[0], b[1], b[2], b[3]].map(f32::from)),
        33 | 34 => (4, 2, |b| [f16(u16_at(b, 0)), f16(u16_at(b, 1)), 0.0, 0.0]),
        35..=38 => (4, 2, |b| {
            [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0, 0.0]
        }),
        39..=41 => (4, 1, |b| [f32::from_bits(u32_at(b, 0)), 0.0, 0.0, 0.0]),
        42 | 43 => (4, 1, |b| [u32_at(b, 0) as f32, 0.0, 0.0, 0.0]),
        44..=47 => (4, 1, |b| [(u32_at(b, 0) & 0xff_ffff) as f32, 0.0, 0.0, 0.0]),
        48..=52 => (2, 2, |b| [f32::from(b[0]), f32::from(b[1]), 0.0, 0.0]),
        53 | 54 => (2, 1, |b| [f16(u16_at(b, 0)), 0.0, 0.0, 0.0]),
        55..=59 => (2, 1, |b| [f32::from(u16_at(b, 0)), 0.0, 0.0, 0.0]),
        60..=65 => (1, 1, |b| [f32::from(b[0]), 0.0, 0.0, 0.0]),
        67 => (4, 3, |b| {
            let v = u32_at(b, 0);
            let scale = f32::from_bits(((v >> 27) + 103) << 23);
            [v & 0x1ff, v >> 9 & 0x1ff, v >> 18 & 0x1ff, 0].map(|c| c as f32 * scale)
        }),
        _ => return None,
    })
}

/// A tile's texels, those with every channel zero, and the sum of each
/// finite texel's largest channel.
#[cfg(feature = "diagnostics")]
type TileStats = (u64, u64, f64);

/// Of `w` x `h` texels `pitch` bytes a row apart in `format`: how many,
/// and how many have every channel zero, a NaN channel, an infinite one;
/// with `tile` above 0, also [`TileStats`] per `tile` x `tile` square (rows
/// of `w` / `tile` rounded up, row-major).
#[cfg(feature = "diagnostics")]
fn texel_stats(
    format: u32,
    w: usize,
    h: usize,
    pitch: usize,
    rows: &[u8],
    tile: usize,
) -> Option<([u64; 4], Vec<TileStats>)> {
    let (n, channels, decode) = texel_decoder(format)?;
    let side = std::num::NonZeroUsize::new(tile);
    let across = side.map_or(0, |s| w.div_ceil(s.get()));
    let mut tiles = vec![(0, 0, 0.0); across * side.map_or(0, |s| h.div_ceil(s.get()))];
    let mut counts = [0u64; 4];
    for y in 0..h {
        let Some(row) = rows.get(y * pitch..y * pitch + w * n) else {
            break;
        };
        for (x, texel) in row.chunks_exact(n).enumerate() {
            let v = decode(texel);
            let v = &v[..channels];
            let zero = v.iter().all(|&c| c == 0.0);
            counts[0] += 1;
            counts[1] += u64::from(zero);
            counts[2] += u64::from(v.iter().any(|c| c.is_nan()));
            counts[3] += u64::from(v.iter().any(|c| c.is_infinite()));
            if let Some(side) = side {
                let t = &mut tiles[y / side * across + x / side];
                t.0 += 1;
                t.1 += u64::from(zero);
                let top = v
                    .iter()
                    .copied()
                    .filter(|c| c.is_finite())
                    .fold(0.0f32, f32::max);
                t.2 += f64::from(top);
            }
        }
    }
    Some((counts, tiles))
}

/// The tile side [`dump_slots`] grids a texture by: a twentieth of its
/// width (128 for 2560x1440, the tiled lighting's tile) for screen-shaped
/// textures, else none.
#[cfg(feature = "diagnostics")]
fn screen_tile(w: usize, h: usize) -> usize {
    if w >= 320 && w.is_multiple_of(20) && (w * 9).abs_diff(h * 16) <= w / 8 {
        w / 20
    } else {
        0
    }
}

/// Per-tile stats as two grids, rows split by " / ": the percent of zero
/// texels, and the mean of the largest channel.
#[cfg(feature = "diagnostics")]
fn tile_grids(tiles: &[TileStats], across: usize) -> (String, String) {
    let rows = |cell: &dyn Fn(&TileStats) -> String| {
        tiles
            .chunks(across.max(1))
            .map(|row| row.iter().map(cell).collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join(" / ")
    };
    let zero = rows(&|t| format!("{}", (100 * t.1).checked_div(t.0).unwrap_or(0)));
    let mean = rows(&|t| format!("{:.1e}", if t.0 > 0 { t.2 / t.0 as f64 } else { 0.0 }));
    (zero, mean)
}

/// Each texel's bytes in hex (rows split by " / "), for textures of at most
/// 256 texels.
#[cfg(feature = "diagnostics")]
fn texel_hex(w: usize, h: usize, pitch: usize, n: usize, rows: &[u8]) -> Option<String> {
    if w * h > 256 || n == 0 {
        return None;
    }
    let mut out = Vec::new();
    for y in 0..h {
        let row = rows.get(y * pitch..y * pitch + w * n)?;
        let texels = row.chunks_exact(n).map(|t| {
            t.iter()
                .rev()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        });
        out.push(texels.collect::<Vec<_>>().join(" "));
    }
    Some(out.join(" / "))
}

/// Per `tile` x `tile` square of a texture of 4-byte RGBA8 texels, its
/// three commonest alpha bytes with their percent ("a:p,a:p", "-" for an
/// empty square), rows split by " / ".
#[cfg(feature = "diagnostics")]
fn alpha_grid(w: usize, h: usize, pitch: usize, rows: &[u8], tile: usize) -> String {
    let across = w.div_ceil(tile);
    let mut counts = vec![[0u32; 256]; across * h.div_ceil(tile)];
    for y in 0..h {
        let Some(row) = rows.get(y * pitch..y * pitch + w * 4) else {
            break;
        };
        for (x, texel) in row.as_chunks::<4>().0.iter().enumerate() {
            counts[y / tile * across + x / tile][usize::from(texel[3])] += 1;
        }
    }
    let cell = |c: &[u32; 256]| {
        let n: u32 = c.iter().sum();
        let mut top: Vec<(usize, u32)> =
            c.iter().copied().enumerate().filter(|e| e.1 > 0).collect();
        top.sort_by_key(|e| std::cmp::Reverse(e.1));
        if top.is_empty() {
            return "-".to_owned();
        }
        top.iter()
            .take(3)
            .map(|(a, k)| format!("{a}:{}", 100 * k / n))
            .collect::<Vec<_>>()
            .join(",")
    };
    counts
        .chunks(across.max(1))
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join(" / ")
}

/// For a texture of `n`-byte unsigned texels: each bit set in any texel,
/// with where it is set as one hex string per row (the first digit's top
/// bit is x 0), rows split by " / ".
#[cfg(feature = "diagnostics")]
fn bit_rows(w: usize, h: usize, pitch: usize, n: usize, rows: &[u8]) -> Vec<(u32, String)> {
    let texel = |y: usize, x: usize| {
        rows.get(y * pitch + x * n..y * pitch + x * n + n)
            .map_or(0, |b| {
                b.iter().rev().fold(0u32, |v, &b| v << 8 | u32::from(b))
            })
    };
    let any = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .fold(0, |a, (y, x)| a | texel(y, x));
    (0..32)
        .filter(|b| any >> b & 1 == 1)
        .map(|b| {
            let row = |y: usize| {
                (0..w)
                    .step_by(4)
                    .map(|x| {
                        let digit = (0..4).fold(0u32, |d, i| {
                            d << 1 | u32::from(x + i < w && texel(y, x + i) >> b & 1 == 1)
                        });
                        char::from_digit(digit, 16).unwrap_or('?')
                    })
                    .collect::<String>()
            };
            (b, (0..h).map(row).collect::<Vec<_>>().join(" / "))
        })
        .collect()
}

/// Of a block-compressed texture of `w` x `h` texels: its 4x4 blocks and
/// how many are all zero bytes; `None` for other formats.
#[cfg(feature = "diagnostics")]
fn zero_blocks(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<(u64, u64)> {
    let size = match format {
        70..=72 | 79..=81 => 8,
        73..=78 | 82..=84 | 94..=99 => 16,
        _ => return None,
    };
    let across = w.div_ceil(4);
    let mut counts = (0, 0);
    for y in 0..h.div_ceil(4) {
        let Some(row) = rows.get(y * pitch..y * pitch + across * size) else {
            break;
        };
        for block in row.chunks_exact(size) {
            counts.0 += 1;
            counts.1 += u64::from(block.iter().all(|&b| b == 0));
        }
    }
    Some(counts)
}

/// At the present of a listed frame: logs every texture slot TL_PS and
/// SkyMain read (one line per slot and resource), who last wrote each
/// texture, and the shares of its texels that are zero, NaN and infinite.
/// The values come from one readback of at most 64 textures and 128 MiB,
/// textures the frame did not write first, decoded until 750 ms in.
#[cfg(feature = "diagnostics")]
fn dump_slots(
    host: &crate::host::Host,
    frame: u64,
    targets: &[(usize, std::sync::Arc<crate::resource::HostTexture>)],
    accesses: &std::collections::BTreeSet<(usize, bool, String)>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::core::Interface;
    const TEXTURES: usize = 64;
    const BYTES: u64 = 128 << 20;
    const DEADLINE: std::time::Duration = std::time::Duration::from_millis(750);
    let depth = std::mem::take(&mut *DEPTH_UAVS.lock().unwrap_or_else(|e| e.into_inner()));
    for (program, resource, twin, description) in depth {
        tracing::info!(
            frame,
            program,
            resource = format_args!("{resource:#x}"),
            description,
            twin,
            "unordered view of a depth texture"
        );
    }
    let slots = std::mem::take(&mut *SLOTS.lock().unwrap_or_else(|e| e.into_inner()));
    if slots.is_empty() {
        return;
    }
    let written = |r: usize| {
        targets.iter().any(|t| t.0 == r) || accesses.iter().any(|(x, w, _)| *x == r && *w)
    };
    let mut order: Vec<(usize, _, Option<u32>, bool)> = Vec::new();
    for (key, seen) in &slots {
        if let Some(copy) = &seen.copy
            && !order.iter().any(|o| o.0 == key.4)
        {
            order.push((key.4, copy, seen.format, material_family(key.0)));
        }
    }
    order.sort_by_key(|o| (o.3, written(o.0)));
    let started = std::time::Instant::now();
    let mut reads = Vec::new();
    let mut skipped = std::collections::HashMap::new();
    if let Ok(list) = host.new_list() {
        let mut bytes = 0;
        for &(r, copy, format, _) in &order {
            let texture = &copy.texture;
            let mut size = 0;
            // SAFETY: a live resource; the total of its first subresource.
            let d = unsafe {
                let d = texture.GetDesc();
                host.device
                    .GetCopyableFootprints(&d, 0, 1, 0, None, None, None, Some(&mut size));
                d
            };
            let why = if d.SampleDesc.Count > 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
                Some("multisampled or a buffer")
            } else if reads.len() >= TEXTURES {
                Some("over the 64 texture cap")
            } else if bytes + size > BYTES {
                Some("over the 128 MiB cap")
            } else {
                None
            };
            if let Some(why) = why {
                skipped.insert(r, why);
                continue;
            }
            let state = crate::host::legal_state(texture, copy.state.load(Relaxed));
            let state = D3D12_RESOURCE_STATES(state as i32);
            let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
            // SAFETY: an open list of ours and a live texture, moved back to
            // the state it is tracked in.
            let read = unsafe {
                if state != source {
                    list.list
                        .ResourceBarrier(&[crate::host::transition(texture, state, source)]);
                }
                let read = read_back(&host.device, &list.list, texture);
                if state != source {
                    list.list
                        .ResourceBarrier(&[crate::host::transition(texture, source, state)]);
                }
                read
            };
            match read {
                Some((buffer, footprint)) => {
                    bytes += size;
                    reads.push((r, buffer, footprint, format));
                }
                None => {
                    skipped.insert(r, "readback failed");
                }
            }
        }
        list.close();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
    }
    let mut values = std::collections::HashMap::new();
    let mut details = std::collections::HashMap::new();
    let mut extra = std::collections::HashMap::new();
    for (r, buffer, footprint, view_format) in reads {
        if started.elapsed() > DEADLINE {
            skipped.insert(r, "past the 750 ms deadline");
            continue;
        }
        let f = footprint.Footprint;
        let host_format = f.Format.0 as u32;
        let bytes = |format: u32| {
            crate::resource::element(format)
                .filter(|e| e.1 == 1)
                .map(|e| e.0)
        };
        let format = view_format
            .filter(|&v| bytes(v).is_some() && bytes(v) == bytes(host_format))
            .unwrap_or(host_format);
        let mut data = std::ptr::null_mut();
        // SAFETY: the copy finished; the mapping spans the footprint's rows.
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_err() {
                continue;
            }
            let rows = std::slice::from_raw_parts(data.cast::<u8>(), rows_bytes(&f) as usize);
            let (w, h, pitch) = (
                f.Width as usize,
                (f.Height * f.Depth) as usize,
                f.RowPitch as usize,
            );
            let tile = if f.Depth == 1 { screen_tile(w, h) } else { 0 };
            let stats = texel_stats(format, w, h, pitch, rows, tile);
            let grids = stats
                .as_ref()
                .filter(|_| tile > 0)
                .map(|s| (tile, tile_grids(&s.1, w.div_ceil(tile))));
            let hex = bytes(host_format).and_then(|n| texel_hex(w, h, pitch, n as usize, rows));
            let alpha = (tile > 0 && matches!(format, 27..=32 | 87..=93))
                .then(|| alpha_grid(w, h, pitch, rows, tile));
            let bits = match (format, bytes(host_format)) {
                (42 | 57 | 62, Some(n)) if w * h > 256 && w * h <= 16384 => {
                    bit_rows(w, h, pitch, n as usize, rows)
                }
                _ => Vec::new(),
            };
            let blocks = zero_blocks(host_format, w, h, pitch, rows);
            buffer.Unmap(0, None);
            extra.insert(r, (alpha, bits, blocks));
            values.insert(r, (format, stats.map(|s| s.0)));
            if grids.is_some() || hex.is_some() {
                details.insert(r, (grids, hex));
            }
        }
    }
    let gpu = GPU_WROTE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default();
    let cpu = UPLOADS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default();
    for ((family, register, root, offset, resource), seen) in &slots {
        let r = *resource as crate::object::This;
        let (address, bytes, description) = crate::resource::texture_span(r).unwrap_or_default();
        let uploaded = seen
            .copy
            .as_ref()
            .and_then(|c| cpu.get(&(c.texture.as_raw() as usize)).copied());
        let writer = if *resource == 0 {
            "no texture view".to_owned()
        } else {
            last_writer(
                frame,
                written(*resource),
                gpu.get(resource).copied(),
                uploaded,
            )
        };
        // SAFETY: a live resource.
        let mips = seen
            .copy
            .as_ref()
            .map(|c| unsafe { c.texture.GetDesc() }.MipLevels);
        let shares = match values.get(resource) {
            Some((format, Some([n, zero, nan, inf]))) if *n > 0 => {
                let pc = |c: u64| 100.0 * c as f64 / *n as f64;
                format!(
                    "f{format}: {n} texels, zero {:.3}%, NaN {:.3}%, Inf {:.3}%",
                    pc(*zero),
                    pc(*nan),
                    pc(*inf)
                )
            }
            Some((format, _)) => match extra.get(resource).and_then(|e| e.2) {
                Some((n, zero)) if n > 0 => format!(
                    "f{format}: {n} blocks, all-zero {:.3}%",
                    100.0 * zero as f64 / n as f64
                ),
                _ => format!("f{format}: format not decoded"),
            },
            None => match skipped.get(resource) {
                Some(why) => format!("not read: {why}"),
                None if seen.copy.is_none() => "no host texture".to_owned(),
                None => "not read".to_owned(),
            },
        };
        tracing::info!(
            frame,
            family,
            register = format_args!("t{register}"),
            root,
            table_dword = offset,
            programs = seen.programs.len(),
            resource = ?r,
            address = format_args!("{address:#x}+{bytes:#x}"),
            description,
            view_format = ?seen.format,
            mips = ?mips,
            writer,
            values = shares,
            "lighting program texture slot"
        );
        match details.get(resource) {
            Some((Some((tile, (zero, mean))), _)) => tracing::info!(
                frame,
                family,
                register = format_args!("t{register}"),
                resource = ?r,
                tile,
                zero_percent = zero.as_str(),
                mean_top = mean.as_str(),
                "lighting program texture slot tiles"
            ),
            Some((None, Some(hex))) => tracing::info!(
                frame,
                family,
                register = format_args!("t{register}"),
                resource = ?r,
                texels = hex.as_str(),
                "lighting program texture slot texels"
            ),
            _ => {}
        }
        let Some((alpha, bits, _)) = extra.get(resource) else {
            continue;
        };
        if let Some(alpha) = alpha {
            tracing::info!(
                frame,
                family,
                register = format_args!("t{register}"),
                resource = ?r,
                alpha = alpha.as_str(),
                "lighting program texture slot alpha"
            );
        }
        for (bit, rows) in bits {
            tracing::info!(
                frame,
                family,
                register = format_args!("t{register}"),
                resource = ?r,
                bit,
                rows = rows.as_str(),
                "lighting program texture slot bits"
            );
        }
    }
    tracing::info!(
        frame,
        slots = slots.len(),
        textures = order.len(),
        read = values.len(),
        skipped = skipped.len(),
        ms = started.elapsed().as_millis() as u64,
        "lighting program texture slots read back"
    );
}

/// At the present of frame `frame`: saves the noted targets, and logs the
/// pipelines that wrote and read each.
pub(crate) fn dump_targets(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let accesses = std::mem::take(&mut *ACCESSES.lock().unwrap_or_else(|e| e.into_inner()));
    for (i, (resource, _)) in targets.iter().enumerate() {
        let programs = |write: bool| -> Vec<&str> {
            accesses
                .iter()
                .filter(|(r, w, _)| r == resource && *w == write)
                .map(|(_, _, name)| name.as_str())
                .collect()
        };
        tracing::info!(
            frame,
            index = i,
            resource = ?(*resource as crate::object::This),
            writers = ?programs(true),
            readers = ?programs(false),
            "target programs"
        );
    }
    #[cfg(feature = "diagnostics")]
    {
        let written: Vec<(usize, u64, u64)> = targets
            .iter()
            .filter_map(|&(r, _)| {
                let (address, bytes, _) = crate::resource::texture_span(r as crate::object::This)?;
                Some((r, address, bytes))
            })
            .collect();
        let mut read: Vec<(usize, u64, u64)> = Vec::new();
        for &(r, write, _) in &accesses {
            if write || read.iter().any(|x| x.0 == r) || targets.iter().any(|t| t.0 == r) {
                continue;
            }
            if let Some((address, bytes, _)) =
                crate::resource::texture_span(r as crate::object::This)
            {
                read.push((r, address, bytes));
            }
        }
        for (n, (reader, target)) in aliased(&read, &written).into_iter().enumerate() {
            if n == 300 {
                tracing::info!(frame, "aliased reads: more left out");
                break;
            }
            let r = reader as crate::object::This;
            let (address, bytes, description) =
                crate::resource::texture_span(r).unwrap_or_default();
            let t = target as crate::object::This;
            let (at, target_bytes, target_description) =
                crate::resource::texture_span(t).unwrap_or_default();
            let names = |res: usize, w: bool| -> Vec<&str> {
                accesses
                    .iter()
                    .filter(|(x, y, _)| *x == res && *y == w)
                    .map(|(_, _, name)| name.as_str())
                    .collect()
            };
            tracing::info!(
                frame,
                read = ?r,
                address = format_args!("{address:#x}+{bytes:#x}"),
                description,
                readers = ?names(reader, false),
                index = targets.iter().position(|x| x.0 == target),
                target = format_args!("{at:#x}+{target_bytes:#x}"),
                target_description,
                writers = ?names(target, true),
                "a texture nothing wrote this frame read over a written target's memory"
            );
        }
    }
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        let raw = accesses
            .iter()
            .any(|(r, _, name)| r == resource && data_wanted(name));
        let resource = *resource as crate::object::This;
        let prefix = format!("frame-{frame}-{i:02}-{resource:?}");
        dump_one(host, texture, state, resource, &prefix, &d, raw);
    }
    #[cfg(feature = "diagnostics")]
    dump_slots(host, frame, &targets, &accesses);
}

/// Pairs (reader, target) whose memory overlaps, of textures read but not
/// written in a frame and targets written in it, each as (resource,
/// address, bytes): a read through another resource over memory the frame
/// wrote, which the host keeps as two textures.
#[cfg(feature = "diagnostics")]
fn aliased(read: &[(usize, u64, u64)], written: &[(usize, u64, u64)]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for &(r, address, bytes) in read {
        for &(t, at, len) in written {
            if r != t && address < at.saturating_add(len) && at < address.saturating_add(bytes) {
                pairs.push((r, t));
            }
        }
    }
    pairs
}

/// Saves one target as `<prefix>-<w>x<h>[x<d>]-f<format>.png`, and with
/// `raw` its rows as read back (`RowPitch` bytes apart, slices after each
/// other) in a `.bin` beside it.
fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    resource: crate::object::This,
    prefix: &str,
    d: &D3D12_RESOURCE_DESC,
    raw: bool,
) {
    let Ok(list) = host.new_list() else { return };
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    list.close();
    host.execute(&[&list]);
    crate::pipelined::wait_all();
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        // A 3D texture's slices follow each other, so they show one under
        // the other.
        let height = f.Height * f.Depth;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), rows_bytes(&f) as usize);
        let depth = if f.Depth > 1 {
            format!("x{}", f.Depth)
        } else {
            String::new()
        };
        let name = format!("{prefix}-{}x{}{depth}-f{}", d.Width, d.Height, d.Format.0);
        if let Some(rgba) = to_rgba(
            f.Format.0 as u32,
            f.Width as usize,
            height as usize,
            pitch,
            rows,
        ) {
            save_as(&name, f.Width, height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        if raw && let Some(folder) = folder() {
            let path = folder.join(format!("{name}.bin"));
            match std::fs::write(&path, rows) {
                Ok(()) => tracing::info!(path = %path.display(), pitch, "raw target saved"),
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "raw target not saved")
                }
            }
        }
        buffer.Unmap(0, None);
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, everything else
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    let (above_zero, [p50, p90, p99]) = spread(&values);
    tracing::info!(
        format,
        low,
        high,
        above_zero,
        p50,
        p90,
        p99,
        "target values"
    );
    if values.len() <= 8 {
        tracing::info!(?values, "target texels");
    }
    for v in values {
        let c = v.map(|c| (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8);
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// The share of pixels with a channel above zero, in percent, and the 50th,
/// 90th and 99th percentiles of each pixel's largest channel: what an 8-bit
/// picture scaled to the brightest pixel cannot show.
fn spread(values: &[[f32; 3]]) -> (f32, [f32; 3]) {
    let mut peaks: Vec<f32> = values
        .iter()
        .map(|v| {
            v.iter()
                .copied()
                .filter(|c| c.is_finite())
                .fold(f32::MIN, f32::max)
        })
        .collect();
    let above = peaks.iter().filter(|&&p| p > 0.0).count();
    let last = peaks.len().saturating_sub(1);
    let mut at = |q: usize| {
        *peaks
            .select_nth_unstable_by(last * q / 100, f32::total_cmp)
            .1
    };
    (
        100.0 * above as f32 / values.len() as f32,
        [at(50), at(90), at(99)],
    )
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
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
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn world_frames_are_the_start_then_every_mth_four_more_times() {
        let frames: Vec<u64> = (0..2000)
            .filter(|&n| super::world_frame(1000, 100, n))
            .collect();
        assert_eq!(frames, [1000, 1100, 1200, 1300, 1400]);
        let once: Vec<u64> = (0..2000)
            .filter(|&n| super::world_frame(1000, 0, n))
            .collect();
        assert_eq!(once, [1000]);
        assert!(!(0..2000).any(|n| super::world_frame(0, 100, n)));
    }

    /// A read-only texture over a written target's memory pairs with it;
    /// one beside it, and the target itself, do not.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn reads_over_a_written_targets_memory_pair_with_it() {
        let written = [(1, 0x1000, 0x100), (2, 0x4000, 0x100)];
        let read = [
            (3, 0x1080, 0x10),
            (4, 0x1100, 0x100),
            (1, 0x1000, 0x100),
            (5, 0x3f00, 0x200),
        ];
        assert_eq!(super::aliased(&read, &written), [(3, 1), (5, 2)]);
    }

    /// Half floats keep NaN apart from infinity, R11G11B10 too, and
    /// R9G9B9E5 scales its mantissas by the shared exponent: a NaN, an
    /// infinity, a zero and a value count once each.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn texel_counts_tell_zero_nan_and_infinity() {
        let halves: Vec<u8> = [[0u16; 4], [0x7e00, 0, 0, 0], [0, 0x7c00, 0, 0], [0x3c00; 4]]
            .iter()
            .flatten()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        assert_eq!(
            super::texel_stats(10, 4, 1, 32, &halves, 0).map(|s| s.0),
            Some([4, 1, 1, 1])
        );
        let nan_11 = (31u32 << 6) | 1;
        let small: Vec<u8> = [nan_11, 31 << 6, 0, 15 << 6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            super::texel_stats(26, 4, 1, 16, &small, 0).map(|s| s.0),
            Some([4, 1, 1, 1])
        );
        let values: Vec<u8> = [0x3c00u16, 0xc000, 1, 0x7bff]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let decode = super::texel_decoder(10).unwrap().2;
        assert_eq!(decode(&values), [1.0, -2.0, 2f32.powi(-24), 65504.0]);
        let decode = super::texel_decoder(26).unwrap().2;
        let packed = (15u32 << 6) | (16 << 17) | (14 << 27);
        assert_eq!(decode(&packed.to_le_bytes()), [1.0, 2.0, 0.5, 0.0]);
        let half = (15u32 << 27) | 256;
        let decode = super::texel_decoder(67).unwrap().2;
        assert_eq!(decode(&half.to_le_bytes())[0], 0.5);
        let shared: Vec<u8> = [0u32, half].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            super::texel_stats(67, 1, 2, 4, &shared, 0).map(|s| s.0),
            Some([2, 1, 0, 0])
        );
    }

    /// A 4x2 R32_FLOAT texture in 2x2 tiles: the left tile all zero, the
    /// right one 1 and 3 over 4 texels; small textures print each texel.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn tiles_count_their_zero_texels_and_average_the_largest_channel() {
        let texels: Vec<u8> = [0.0f32, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (counts, tiles) = super::texel_stats(41, 4, 2, 16, &texels, 2).unwrap();
        assert_eq!(counts, [8, 6, 0, 0]);
        assert_eq!(tiles, [(4, 4, 0.0), (4, 2, 4.0)]);
        let (zero, mean) = super::tile_grids(&tiles, 2);
        assert_eq!((zero.as_str(), mean.as_str()), ("100 50", "0.0e0 1.0e0"));
        assert_eq!(super::screen_tile(2560, 1440), 128);
        assert_eq!(super::screen_tile(1280, 720), 64);
        assert_eq!(super::screen_tile(64, 64), 0);
        let hex = super::texel_hex(
            2,
            2,
            8,
            4,
            &[1, 0, 0, 0, 0xcb, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0],
        );
        assert_eq!(
            hex.as_deref(),
            Some("00000001 000000cb / 00000000 0000ffff")
        );
    }

    /// A 5x2 R8 mask prints bit 0 (x 0 and 4 of row 0) and bit 2 (x 3 of
    /// row 1) as hex rows; of two BC1 blocks one is all zero; a 2x2 tile's
    /// alpha bytes are three 7s and a 9.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn masks_print_each_bit_and_block_textures_count_zero_blocks() {
        let mask = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0];
        assert_eq!(
            super::bit_rows(5, 2, 8, 1, &mask),
            [(0, "88 / 00".to_owned()), (2, "00 / 10".to_owned())]
        );
        let blocks = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(super::zero_blocks(71, 8, 4, 16, &blocks), Some((2, 1)));
        let rgba = [0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 0, 0, 0, 7];
        assert_eq!(super::alpha_grid(2, 2, 8, &rgba, 2), "7:75,9:25");
    }

    /// A write this frame wins; otherwise the later of the GPU's and an
    /// upload's frames, or nothing.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_last_writer_is_this_frames_gpu_or_the_later_of_gpu_and_upload() {
        use super::last_writer;
        assert_eq!(last_writer(9, true, None, Some(8)), "gpu this frame");
        assert_eq!(last_writer(9, false, Some(9), None), "gpu this frame");
        assert_eq!(
            last_writer(9, false, Some(3), Some(7)),
            "cpu upload, frame 7 (gpu frame 3)"
        );
        assert_eq!(
            last_writer(9, false, Some(7), Some(3)),
            "gpu, frame 7 (cpu upload frame 3)"
        );
        assert_eq!(last_writer(9, false, None, Some(2)), "cpu upload, frame 2");
        assert_eq!(last_writer(9, false, None, None), "never");
    }

    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }

    #[test]
    fn spread_counts_pixels_above_zero_and_takes_percentiles_of_their_peaks() {
        let mut values: Vec<[f32; 3]> = (0..99).map(|i| [-1.0, i as f32, 0.0]).collect();
        values.push([f32::NAN, f32::INFINITY, 1000.0]);
        let (above, peaks) = super::spread(&values);
        assert_eq!(above, 99.0);
        assert_eq!(peaks, [49.0, 89.0, 98.0]);
    }

    /// A 3D readback holds the slices one after the other, rows `RowPitch`
    /// apart: the buffer spans them all, and they decode as one picture,
    /// slice under slice.
    #[test]
    fn a_3d_footprint_reads_back_as_its_slices_one_under_the_other() {
        use windows::Win32::Graphics::Direct3D12::D3D12_SUBRESOURCE_FOOTPRINT;
        use windows::Win32::Graphics::Dxgi::Common::*;
        // A 65x65x65 half-float LUT: 520-byte rows, padded to 768.
        let lut = D3D12_SUBRESOURCE_FOOTPRINT {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            Width: 65,
            Height: 65,
            Depth: 65,
            RowPitch: 768,
        };
        assert_eq!(super::rows_bytes(&lut), 768 * 65 * 65);
        let f = D3D12_SUBRESOURCE_FOOTPRINT {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Width: 2,
            Height: 2,
            Depth: 3,
            RowPitch: 256,
        };
        let mut rows = vec![0u8; super::rows_bytes(&f) as usize];
        let mut picture = Vec::new();
        for z in 0..3u8 {
            for y in 0..2u8 {
                for x in 0..2u8 {
                    let at = usize::from(z * 2 + y) * 256 + usize::from(x) * 4;
                    rows[at..at + 4].copy_from_slice(&[x, y, z, 9]);
                    picture.extend([x, y, z, 255]);
                }
            }
        }
        assert_eq!(super::to_rgba(28, 2, 2 * 3, 256, &rows), Some(picture));
    }

    /// On the host GPU: a 3D texture reads back every slice and a 1D one
    /// its row (65 texels wide, so rows end in padding), with nothing for
    /// the debug layer to report.
    #[test]
    fn read_back_takes_every_slice_of_a_3d_texture_and_the_row_of_a_1d_one() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let _serial = crate::watch::serial();
        let Some(host) = crate::host::host() else {
            eprintln!("no host GPU: skipped");
            return;
        };
        for (dimension, height, depth) in [
            (D3D12_RESOURCE_DIMENSION_TEXTURE3D, 3, 4),
            (D3D12_RESOURCE_DIMENSION_TEXTURE1D, 1, 1),
        ] {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: dimension,
                Width: 65,
                Height: height,
                DepthOrArraySize: depth,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            let mut up = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            // SAFETY: plain creation; outputs for one subresource.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut texture,
                    )
                    .unwrap();
                host.device
                    .GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut up), None, None, None);
            }
            let texture = texture.unwrap();
            let pitch = up.Footprint.RowPitch as usize;
            let rows = (height * u32::from(depth)) as usize;
            let (upload, data) = host
                .upload_buffer(super::rows_bytes(&up.Footprint))
                .unwrap();
            let mut picture = Vec::new();
            for (r, x) in (0..rows).flat_map(|r| (0..65).map(move |x| (r, x))) {
                let texel = [x as u8, r as u8, 7, 255];
                // SAFETY: inside the mapped upload buffer.
                unsafe {
                    data.add(r * pitch + x * 4)
                        .copy_from_nonoverlapping(texel.as_ptr(), 4)
                };
                picture.extend(texel);
            }
            let list = host.new_list().unwrap();
            let mut read = None;
            let errors = crate::host::errors_of(|| {
                // SAFETY: an open list of ours; locations naming live resources.
                unsafe {
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&upload),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: up,
                        },
                    };
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    };
                    list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                    list.list.ResourceBarrier(&[crate::host::transition(
                        &texture,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                }
                read = super::read_back(&host.device, &list.list, &texture);
                list.close();
                host.execute(&[&list]);
                crate::pipelined::wait_all();
            });
            assert_eq!(errors.unwrap_or_default(), Vec::<String>::new());
            let (buffer, footprint) = read.unwrap();
            let f = footprint.Footprint;
            assert_eq!((f.Width, f.Height, f.Depth), (65, height, u32::from(depth)));
            let mut mapped = std::ptr::null_mut();
            // SAFETY: the copy finished; the mapping spans the footprint's rows.
            let back = unsafe {
                buffer.Map(0, None, Some(&mut mapped)).unwrap();
                let bytes =
                    std::slice::from_raw_parts(mapped.cast::<u8>(), super::rows_bytes(&f) as usize);
                let back = super::to_rgba(28, 65, rows, f.RowPitch as usize, bytes);
                buffer.Unmap(0, None);
                back
            };
            assert_eq!(back, Some(picture));
        }
    }
}
