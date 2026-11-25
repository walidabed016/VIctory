//! Alias copies. The host gives every resource and every placed view its
//! own host texture, so a texture read over title memory the GPU wrote
//! through another resource would never see those writes: Grand Theft
//! Auto VI's sky and GI computes write R32_UINT textures (format 42) that
//! its lighting samples as R9G9B9E5 (67) through other resources at the
//! same address and size (run 91). A draw or dispatch that writes a texture
//! (render target, depth or unordered view) notes the write at the
//! texture's address ([`gpu_wrote`]); one that reads a texture through a
//! shader view first copies in the newest write at its address if another
//! resource made it ([`before_read`]), through a buffer footprint, which
//! allows two formats whose every subresource holds the same rows of the
//! same bytes. `VICTORY_NO_ALIAS_COPY=1` turns it off.
//!
//! shortcut: in recording order, and only reads that start where the write
//! starts with the same layout; other overlaps are logged (diagnostics),
//! not copied.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

#[cfg(feature = "diagnostics")]
use super::footprint;
use super::{BUFFER, resource};
use crate::host::HostList;
use crate::object::This;

/// A texture the GPU writes, by the start of its memory.
struct Written {
    resource: usize,
    /// Where its memory ends (diagnostics).
    #[cfg(feature = "diagnostics")]
    end: u64,
    /// When, in the order writes are noted.
    seq: u64,
}

#[derive(Default)]
struct Aliases {
    written: BTreeMap<u64, Vec<Written>>,
    /// Writes noted.
    seq: u64,
    /// Textures read through shader views: the newest write at their
    /// address they have taken or passed over.
    read: HashMap<usize, u64>,
    /// The buffer copies go through, and its bytes. Host lists all run on
    /// one queue, and each copy leaves it in the state it found it in.
    buffer: Option<(ID3D12Resource, u64)>,
    /// Reader and writer pairs logged (diagnostics).
    #[cfg(feature = "diagnostics")]
    logged: std::collections::HashSet<(usize, usize)>,
}

/// Held while a copy is recorded, so that its writer is not let go of
/// meanwhile ([`forget`]).
static ALIASES: Mutex<Option<Aliases>> = Mutex::new(None);

/// Since the last frame line (diagnostics): alias copies, their bytes, and
/// newer writes at a read's address not copied (layouts that differ).
#[cfg(feature = "diagnostics")]
static COUNTS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let off = std::env::var_os("VICTORY_NO_ALIAS_COPY").is_some_and(|v| v == "1");
        if off {
            tracing::info!("alias copies are off (VICTORY_NO_ALIAS_COPY=1)");
        }
        !off
    })
}

fn aliases() -> std::sync::MutexGuard<'static, Option<Aliases>> {
    ALIASES.lock().unwrap_or_else(|e| e.into_inner())
}

/// A draw or dispatch writes texture `this` through a render target, depth
/// or unordered view.
pub(crate) fn gpu_wrote(this: This) {
    if this.is_null() || !on() {
        return;
    }
    let r = resource(this);
    if r.desc.dimension == BUFFER || r.host.is_none() {
        return;
    }
    let start = r.data() as u64;
    let mut guard = aliases();
    let a = guard.get_or_insert_with(Aliases::default);
    a.seq += 1;
    let seq = a.seq;
    let at = a.written.entry(start).or_default();
    if let Some(w) = at.iter_mut().find(|w| w.resource == this as usize) {
        w.seq = seq;
        return;
    }
    at.push(Written {
        resource: this as usize,
        #[cfg(feature = "diagnostics")]
        end: start + footprint(&r.desc).map_or(0, |f| f.size),
        seq,
    });
}

/// The title let go of `resource`, whose memory starts at `start`.
pub(crate) fn forget(resource: usize, start: u64) {
    let mut guard = aliases();
    let Some(a) = guard.as_mut() else {
        return;
    };
    a.read.remove(&resource);
    if let Some(at) = a.written.get_mut(&start) {
        at.retain(|w| w.resource != resource);
        if at.is_empty() {
            a.written.remove(&start);
        }
    }
}

/// Before a draw or dispatch reads texture `this` through a shader view:
/// if the newest write at its address went through another resource and
/// it has not taken it yet, records the copy of that resource's host copy
/// into its own, leaving it in the copy destination state. Whether it
/// copied.
pub(crate) fn before_read(list: &HostList, this: This) -> bool {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    if this.is_null() || !on() {
        return false;
    }
    let r = resource(this);
    let Some(to) = r.host.as_ref().map(|h| h.texture.clone()) else {
        return false;
    };
    let start = r.data() as u64;
    let mut guard = aliases();
    let a = guard.get_or_insert_with(Aliases::default);
    let newest = a
        .written
        .get(&start)
        .and_then(|at| at.iter().max_by_key(|w| w.seq))
        .map(|w| (w.resource as This, w.seq));
    let Some((writer, seq)) = newest else {
        #[cfg(feature = "diagnostics")]
        inside(a, this, start);
        return false;
    };
    let seen = a.read.entry(this as usize).or_default();
    if writer == this || seq <= *seen {
        *seen = (*seen).max(seq);
        return false;
    }
    *seen = seq;
    // Alive: letting go of it waits for the lock.
    let Some(from) = super::host_resource(writer).cloned() else {
        return false;
    };
    let Some((written, reads, total)) = matching(&from, &to) else {
        #[cfg(feature = "diagnostics")]
        {
            COUNTS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            log_pair(a, this, writer, "not copied: their layouts differ");
        }
        return false;
    };
    let buffer = match &a.buffer {
        Some((buffer, size)) if *size >= total => buffer.clone(),
        _ => {
            let Some(buffer) = copy_buffer(total) else {
                return false;
            };
            a.buffer = Some((buffer.clone(), total));
            buffer
        }
    };
    list.keep(&buffer);
    // The writer goes back to its state after, in case the draw also binds
    // it.
    let back = list
        .transition(writer, COPY_SOURCE)
        .map(|(texture, before)| {
            let after = crate::host::legal_state(texture, COPY_SOURCE);
            let barrier = crate::host::transition(
                texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(after as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
            before
        });
    crate::list::ensure_state(list, this, COPY_DEST);
    let state = |before, after| crate::host::transition(&buffer, before, after);
    // SAFETY: an open list; live resources in the copy states, the buffer
    // large enough for every footprint.
    unsafe {
        list.list.ResourceBarrier(&[state(
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        for (i, footprint) in written.iter().enumerate() {
            list.list.CopyTextureRegion(
                &location(&buffer, Some(*footprint), 0),
                0,
                0,
                0,
                &location(&from, None, i as u32),
                None,
            );
        }
        list.list.ResourceBarrier(&[state(
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for (i, footprint) in reads.iter().enumerate() {
            list.list.CopyTextureRegion(
                &location(&to, None, i as u32),
                0,
                0,
                0,
                &location(&buffer, Some(*footprint), 0),
                None,
            );
        }
        list.list.ResourceBarrier(&[state(
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
    if let Some(before) = back {
        crate::list::ensure_state(list, writer, before);
    }
    #[cfg(feature = "diagnostics")]
    {
        COUNTS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        COUNTS[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
        let what = "copy: a texture read takes the writes another resource made over its memory";
        log_pair(a, this, writer, what);
    }
    drop(guard);
    // Its contents are the host's now, not title memory's.
    crate::texture::host_written(this);
    true
}

/// A copy location: a footprint in `resource` (a buffer), or its
/// subresource `index`.
fn location(
    resource: &ID3D12Resource,
    footprint: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    index: u32,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Type: if footprint.is_some() {
            D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT
        } else {
            D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
        },
        Anonymous: match footprint {
            Some(f) => D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
            None => D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        },
    }
}

type Footprints = Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>;

/// Each subresource's footprint, rows and bytes per row, and the bytes of
/// them all; `None` for multisampled textures and those with a stencil
/// plane.
fn layouts(texture: &ID3D12Resource) -> Option<(Footprints, Vec<u32>, Vec<u64>, u64)> {
    const STENCIL: [i32; 8] = [19, 20, 21, 22, 44, 45, 46, 47];
    let host = crate::host::host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.SampleDesc.Count != 1 || STENCIL.contains(&d.Format.0) {
        return None;
    }
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let n = count as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let (mut rows, mut bytes, mut total) = (vec![0; n], vec![0; n], 0);
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(bytes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    Some((footprints, rows, bytes, total))
}

/// The footprints a copy from `from` into `to` goes through, and the
/// buffer bytes they take, if every subresource of the two holds as many
/// rows of as many bytes at the same place.
fn matching(from: &ID3D12Resource, to: &ID3D12Resource) -> Option<(Footprints, Footprints, u64)> {
    let (written, rows, bytes, total) = layouts(from)?;
    let (reads, to_rows, to_bytes, to_total) = layouts(to)?;
    let placed = |(a, b): (
        &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )| {
        a.Offset == b.Offset
            && a.Footprint.RowPitch == b.Footprint.RowPitch
            && a.Footprint.Depth == b.Footprint.Depth
    };
    let same = written.len() == reads.len()
        && total == to_total
        && rows == to_rows
        && bytes == to_bytes
        && written.iter().zip(&reads).all(placed);
    same.then_some((written, reads, total))
}

/// A buffer of `size` bytes in GPU memory, in the common state.
fn copy_buffer(size: u64) -> Option<ID3D12Resource> {
    let host = crate::host::host()?;
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
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, size, "no buffer for alias copies");
    }
    if let Some(b) = &buffer {
        crate::host::name(b, || format!("alias copies {size:#x}"));
    }
    buffer
}

/// A read that starts inside memory another resource was written through,
/// past its start: logged once per pair (diagnostics).
#[cfg(feature = "diagnostics")]
fn inside(a: &mut Aliases, this: This, start: u64) {
    let writer = a.written.range(..start).next_back().and_then(|(_, at)| {
        at.iter()
            .find(|w| w.end > start && w.resource != this as usize)
            .map(|w| w.resource as This)
    });
    if let Some(writer) = writer {
        log_pair(
            a,
            this,
            writer,
            "not copied: the read starts inside the written memory",
        );
    }
}

/// Logs a reader and writer pair the first time, up to 300 pairs
/// (diagnostics).
#[cfg(feature = "diagnostics")]
fn log_pair(a: &mut Aliases, reader: This, writer: This, what: &str) {
    if a.logged.len() >= 300 || !a.logged.insert((reader as usize, writer as usize)) {
        return;
    }
    let span = |r: This| {
        super::texture_span(r).map_or_else(
            || format!("{r:p}"),
            |(at, bytes, desc)| format!("{r:p} {desc} at {at:#x}+{bytes:#x}"),
        )
    };
    tracing::info!(reader = span(reader), writer = span(writer), "alias {what}");
}

/// The frame line of alias copies per frame (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn line(frames: f64) -> String {
    let [copies, bytes, other] = COUNTS
        .each_ref()
        .map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    if !on() {
        return "  host costs per frame, alias copies: off (VICTORY_NO_ALIAS_COPY=1)".into();
    }
    format!(
        "  host costs per frame, alias copies: {copies:.1} copies {:.2} MiB; {other:.1} newer \
         writes not copied (layouts differ)",
        bytes / f64::from(1 << 20)
    )
}

#[cfg(test)]
mod tests {
    use super::super::{
        HeapProperties, RESOURCE_IIDS, Resource, ResourceDesc, create_at_address, create_committed,
        host_resource, texture_memory,
    };
    use super::*;
    use crate::object::S_OK;

    fn desc(format: u32, flags: u32) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            alignment: 0,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags,
        }
    }

    /// A buffer of 16 KiB on a heap of `kind`.
    fn buffer(kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let host = crate::host::host().unwrap();
        let mut b: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; the out pointer is ours.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: kind,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 16 << 10,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut b,
                )
                .unwrap();
        }
        b.unwrap()
    }

    /// The 64 rows of 64 four-byte texels of a 16 KiB buffer, as `format`.
    fn rows(format: i32) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format),
                Width: 64,
                Height: 64,
                Depth: 1,
                RowPitch: 256,
            },
        }
    }

    /// An R9G9B9E5 texture over the memory a compute wrote as R32_UINT
    /// (run 91's pairs) reads what it wrote, once per write; one whose rows
    /// hold other bytes does not, and neither does the writer itself.
    #[test]
    fn a_read_over_memory_another_resource_wrote_takes_its_writes() {
        let _serial = crate::watch::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let properties = HeapProperties {
            kind: 1,
            cpu_pages: 0,
            pool: 0,
            creation_nodes: 0,
            visible_nodes: 0,
        };
        let mut writer = null;
        let made = create_committed(
            null,
            &properties,
            0,
            &desc(42, 4),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut writer,
        );
        assert_eq!(made, S_OK);
        let address = texture_memory(writer).unwrap().0 as u64;
        let at = |format: u32| {
            let mut out = null;
            let made = create_at_address(
                null,
                address,
                &desc(format, 0),
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                &mut out,
            );
            assert_eq!(made, S_OK);
            out
        };
        let (reader, narrow) = (at(67), at(54));
        let upload = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let readback = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let pattern: Vec<u8> = (0..16u32 << 10).map(|i| (i * 7 + 3) as u8).collect();
        let mut data = std::ptr::null_mut();
        // SAFETY: our upload buffer, 16 KiB.
        unsafe {
            upload.Map(0, None, Some(&mut data)).unwrap();
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), data.cast::<u8>(), pattern.len());
        }
        let run = |list: HostList| {
            list.close();
            host.execute(&[&list]);
            crate::pipelined::wait_all();
        };
        // The compute's write.
        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, writer, 0x400);
        // SAFETY: an open list; live resources.
        unsafe {
            list.list.CopyTextureRegion(
                &location(host_resource(writer).unwrap(), None, 0),
                0,
                0,
                0,
                &location(&upload, Some(rows(42)), 0),
                None,
            );
        }
        gpu_wrote(writer);
        run(list);
        // The reads.
        let list = host.new_list().unwrap();
        assert!(!before_read(&list, writer), "the writer has its writes");
        assert!(before_read(&list, reader), "the write is copied in");
        assert!(!before_read(&list, reader), "once");
        assert!(
            !before_read(&list, narrow),
            "rows of 128 bytes take nothing"
        );
        crate::list::ensure_state(&list, reader, 0x800);
        // SAFETY: an open list; live resources.
        unsafe {
            list.list.CopyTextureRegion(
                &location(&readback, Some(rows(67)), 0),
                0,
                0,
                0,
                &location(host_resource(reader).unwrap(), None, 0),
                None,
            );
        }
        run(list);
        // SAFETY: the finished copy's read-back buffer, 16 KiB.
        let read = unsafe {
            readback.Map(0, None, Some(&mut data)).unwrap();
            let read = std::slice::from_raw_parts(data.cast::<u8>(), pattern.len()).to_vec();
            readback.Unmap(0, None);
            read
        };
        assert!(read == pattern, "the reader holds the writer's bytes");
        gpu_wrote(writer);
        let list = host.new_list().unwrap();
        assert!(
            before_read(&list, reader),
            "a newer write is copied in again"
        );
        run(list);
        assert_eq!(host.log_messages(), None, "debug layer errors");
        for r in [narrow, reader, writer] {
            crate::object::release_boxed::<Resource>(r);
        }
    }
}
