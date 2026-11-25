//! Alias copies. The host gives every resource and every placed view its
//! own host texture, so a texture read over title memory the GPU wrote
//! through another resource would never see those writes: Grand Theft
//! Auto VI's sky and GI computes write R32_UINT textures (format 42) that
//! its lighting samples as R9G9B9E5 (67) through other resources at the
//! same address and size (run 91). A draw or dispatch that writes a texture
//! (render target, depth or unordered view) notes the write at the
//! texture's address ([`gpu_wrote`]). A read through a shader view of a
//! texture another resource writes over ([`before_read`]) takes, through a
//! buffer footprint (two formats whose every subresource holds the same
//! rows of the same bytes):
//!
//! - a write its own list made before it, copied there and then;
//! - otherwise the newest write submitted before its list, copied in the
//!   list's prelude at submission ([`submit`]), as the console's memory
//!   would hold it when the list runs (lists are recorded in another order
//!   than they are submitted: run 97 counted 154 reads a frame that took
//!   the previous frame's write).
//!
//! `VICTORY_NO_ALIAS_COPY=1` turns it off; `VICTORY_ALIAS_RECORD_ORDER=1`
//! copies writes of other lists at the read too, in recording order (as
//! 5bc69ea).
//!
//! shortcut: only reads that start where the write starts with the same
//! layout; other overlaps are logged (diagnostics), not copied.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

#[cfg(feature = "diagnostics")]
use super::footprint;
use super::{BUFFER, HostTexture, resource};
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

/// A write as submitted: when it was noted, the writer's host copy, and
/// the writer.
type Write = (u64, Arc<HostTexture>, usize);

#[derive(Default)]
struct Aliases {
    written: BTreeMap<u64, Vec<Written>>,
    /// Writes noted.
    seq: u64,
    /// Readers of memory other resources write: the newest write a copy
    /// recorded at the read gave them.
    read: HashMap<usize, u64>,
    /// Addresses read through another resource than wrote them: their
    /// writes are noted per list.
    aliased: std::collections::HashSet<u64>,
    /// The newest write submitted at each of them.
    submitted: HashMap<u64, Write>,
    /// The newest write each reader holds once the lists submitted so far
    /// have run.
    taken: HashMap<usize, u64>,
    /// The buffer copies go through, and its bytes. Host lists all run on
    /// one queue, each copy leaves it in the state it found it in, and the
    /// lists that copy through it keep it.
    buffer: Option<(ID3D12Resource, u64)>,
    /// Reader and writer pairs logged (diagnostics).
    #[cfg(feature = "diagnostics")]
    logged: std::collections::HashSet<(usize, usize)>,
}

/// A host list's reads and writes of aliased memory, until it is reset
/// ([`HostList::aliased`]).
#[derive(Default)]
pub(crate) struct Recorded {
    /// The newest write the list makes at each aliased address.
    writes: HashMap<u64, Write>,
    /// Reads that do not take a write of the list's own, once per reader.
    reads: Vec<Read>,
    /// Readers that took a write of the list's own, and that write.
    inline: Vec<(usize, u64)>,
    /// What its prelude's copies name, kept until the list has run.
    kept: Vec<Arc<HostTexture>>,
    buffers: Vec<ID3D12Resource>,
}

struct Read {
    address: u64,
    reader: usize,
    to: Arc<HostTexture>,
    /// The write a copy at the read gave it (in recording order); `None`:
    /// the prelude's.
    held: Option<u64>,
}

/// The copies a list's prelude records ([`submit`]).
pub(crate) struct Copies {
    buffer: ID3D12Resource,
    copies: Vec<Copy>,
}

struct Copy {
    from: Arc<HostTexture>,
    to: Arc<HostTexture>,
    /// Their real states at submission, before the prelude moves them.
    real: [D3D12_RESOURCE_STATES; 2],
    written: Footprints,
    reads: Footprints,
}

/// Held while a copy is recorded, so that its writer is not let go of
/// meanwhile ([`forget`]).
static ALIASES: Mutex<Option<Aliases>> = Mutex::new(None);

/// Since the last frame line (diagnostics): alias copies, their bytes,
/// newer writes at a read's address not copied (layouts that differ), and
/// reads of aliased memory that, when their list was submitted, held an
/// older write than one submitted ahead of them (late), or a write not
/// submitted yet (early).
#[cfg(feature = "diagnostics")]
static COUNTS: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

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

/// Whether writes of other lists are copied at the read, in recording
/// order (`VICTORY_ALIAS_RECORD_ORDER=1`).
fn record_order() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ALIAS_RECORD_ORDER").is_some_and(|v| v == "1");
        if on {
            tracing::info!("alias copies in recording order (VICTORY_ALIAS_RECORD_ORDER=1)");
        }
        on
    })
}

fn aliases() -> std::sync::MutexGuard<'static, Option<Aliases>> {
    ALIASES.lock().unwrap_or_else(|e| e.into_inner())
}

/// A draw or dispatch recorded on `list` writes texture `this` through a
/// render target, depth or unordered view.
pub(crate) fn gpu_wrote(list: &HostList, this: This) {
    if this.is_null() || !on() {
        return;
    }
    let r = resource(this);
    let Some(copy) = r.host.as_ref() else {
        return;
    };
    if r.desc.dimension == BUFFER {
        return;
    }
    let start = r.data() as u64;
    let mut guard = aliases();
    let a = guard.get_or_insert_with(Aliases::default);
    a.seq += 1;
    let seq = a.seq;
    if a.aliased.contains(&start) {
        let write = (seq, copy.clone(), this as usize);
        list.aliased(|recorded| recorded.writes.insert(start, write));
    }
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
    a.taken.remove(&resource);
    if a.submitted.get(&start).is_some_and(|w| w.2 == resource) {
        a.submitted.remove(&start);
    }
    if let Some(at) = a.written.get_mut(&start) {
        at.retain(|w| w.resource != resource);
        if at.is_empty() {
            a.written.remove(&start);
        }
    }
}

/// Before a draw or dispatch recorded on `list` reads texture `this`
/// through a shader view, ahead of bringing it up to date from title
/// memory: if other resources write its memory, it takes their writes
/// from now on, not title memory's. The newest one, if its list made it,
/// is copied into its host copy now, leaving it in the copy destination
/// state (and so are other lists' in recording order); otherwise the
/// list's prelude copies the newest one submitted before it. Whether it
/// copied now.
pub(crate) fn before_read(list: &HostList, this: This) -> bool {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    if this.is_null() || !on() {
        return false;
    }
    let r = resource(this);
    let Some(to) = r.host.clone() else {
        return false;
    };
    let start = r.data() as u64;
    let mut guard = aliases();
    let a = guard.get_or_insert_with(Aliases::default);
    let Some(at) = a.written.get(&start) else {
        #[cfg(feature = "diagnostics")]
        inside(a, this, start);
        return false;
    };
    if at.iter().all(|w| w.resource == this as usize) {
        return false;
    }
    let Some((writer, seq)) = at
        .iter()
        .max_by_key(|w| w.seq)
        .map(|w| (w.resource as This, w.seq))
    else {
        return false;
    };
    if a.aliased.insert(start) {
        // Its writes went unnoted per list so far: the newest counts as
        // submitted.
        if let Some(copy) = super::host_copy(writer) {
            a.submitted
                .entry(start)
                .or_insert((seq, copy, writer as usize));
        }
    }
    if let std::collections::hash_map::Entry::Vacant(v) = a.read.entry(this as usize) {
        v.insert(0);
        // Its contents are the host's from now on, not title memory's.
        crate::texture::host_written(this);
    }
    // A write of its own list comes after its prelude, so the read takes
    // it whatever other lists wrote.
    let own = list.aliased(|recorded| recorded.writes.get(&start).map(|w| (w.0, w.2)));
    let (writer, fresh) = match own {
        Some((seq, writer)) => {
            let taken = list.aliased(|recorded| {
                let taken = recorded
                    .inline
                    .iter()
                    .any(|&(reader, s)| reader == this as usize && s >= seq);
                if !taken {
                    recorded.inline.push((this as usize, seq));
                }
                taken
            });
            (writer as This, !taken)
        }
        None if record_order() => {
            let seen = a.read.entry(this as usize).or_default();
            let fresh = seq > *seen;
            *seen = (*seen).max(seq);
            let held = *seen;
            list.aliased(|recorded| {
                if !recorded
                    .reads
                    .iter()
                    .any(|read| read.reader == this as usize)
                {
                    recorded.reads.push(Read {
                        address: start,
                        reader: this as usize,
                        to: to.clone(),
                        held: Some(held),
                    });
                }
            });
            (writer, fresh)
        }
        None => {
            list.aliased(|recorded| {
                if !recorded
                    .reads
                    .iter()
                    .any(|read| read.reader == this as usize)
                {
                    recorded.reads.push(Read {
                        address: start,
                        reader: this as usize,
                        to,
                        held: None,
                    });
                }
            });
            return false;
        }
    };
    let alive = a
        .written
        .get(&start)
        .is_some_and(|at| at.iter().any(|w| w.resource == writer as usize));
    let fresh = fresh && writer != this && alive;
    if !fresh {
        return false;
    }
    // Alive: letting go of it waits for the lock.
    let Some(from) = super::host_resource(writer).cloned() else {
        return false;
    };
    let Some((written, reads, total)) = matching(&from, &to.texture) else {
        #[cfg(feature = "diagnostics")]
        {
            COUNTS[2].fetch_add(1, Relaxed);
            log_pair(
                a,
                start,
                (this as usize, writer as usize),
                &to.texture,
                &from,
                "not copied: their layouts differ",
            );
        }
        return false;
    };
    let Some(buffer) = buffer_of(a, total) else {
        return false;
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
    copy_through(&list.list, &buffer, &from, &to.texture, &written, &reads);
    if let Some(before) = back {
        crate::list::ensure_state(list, writer, before);
    }
    #[cfg(feature = "diagnostics")]
    {
        COUNTS[0].fetch_add(1, Relaxed);
        COUNTS[1].fetch_add(total, Relaxed);
        let what = "copy: a texture read takes the writes another resource made over its memory";
        log_pair(
            a,
            start,
            (this as usize, writer as usize),
            &to.texture,
            &from,
            what,
        );
    }
    true
}

/// When `list` is submitted, before its prelude is recorded: the copies the
/// prelude records, of the newest write submitted at each address the
/// list reads without a write of its own; then its own writes count as
/// submitted.
pub(crate) fn submit(list: &HostList) -> Option<Copies> {
    if !on() {
        return None;
    }
    let (reads, inline, writes) = list.aliased(|recorded| {
        (
            recorded
                .reads
                .iter()
                .map(|r| (r.address, r.reader, r.to.clone(), r.held))
                .collect::<Vec<_>>(),
            recorded.inline.clone(),
            recorded
                .writes
                .iter()
                .map(|(&address, write)| (address, write.clone()))
                .collect::<Vec<_>>(),
        )
    });
    if reads.is_empty() && inline.is_empty() && writes.is_empty() {
        return None;
    }
    let mut guard = aliases();
    let a = guard.get_or_insert_with(Aliases::default);
    let (mut copies, mut total) = (Vec::new(), 0);
    for (address, reader, to, held) in reads {
        #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
        let Some((ahead, from, writer)) = a.submitted.get(&address).cloned() else {
            continue;
        };
        let mut holds = held.unwrap_or_else(|| a.taken.get(&reader).copied().unwrap_or(0));
        if held.is_none() && ahead > holds && !Arc::ptr_eq(&from, &to) {
            let matched = matching(&from.texture, &to.texture);
            #[cfg(feature = "diagnostics")]
            {
                let (count, what) = match matched {
                    Some(_) => (
                        0,
                        "copy at submission: a texture read takes the newest write \
                                    another resource made over its memory before its list",
                    ),
                    None => (2, "not copied: their layouts differ"),
                };
                COUNTS[count].fetch_add(1, Relaxed);
                log_pair(
                    a,
                    address,
                    (reader, writer),
                    &to.texture,
                    &from.texture,
                    what,
                );
            }
            if let Some((written, reads, bytes)) = matched {
                holds = ahead;
                total = total.max(bytes);
                #[cfg(feature = "diagnostics")]
                COUNTS[1].fetch_add(bytes, Relaxed);
                let real =
                    [&from, &to].map(|t| D3D12_RESOURCE_STATES(t.state.load(Relaxed) as i32));
                copies.push(Copy {
                    from,
                    to,
                    real,
                    written,
                    reads,
                });
            }
        }
        #[cfg(feature = "diagnostics")]
        if ahead > holds {
            COUNTS[3].fetch_add(1, Relaxed);
        } else if holds > ahead {
            COUNTS[4].fetch_add(1, Relaxed);
        }
        let taken = a.taken.entry(reader).or_default();
        *taken = (*taken).max(holds);
    }
    for (reader, seq) in inline {
        let taken = a.taken.entry(reader).or_default();
        *taken = (*taken).max(seq);
    }
    for (address, write) in writes {
        if a.submitted.get(&address).is_none_or(|w| w.0 < write.0) {
            a.submitted.insert(address, write);
        }
    }
    if copies.is_empty() {
        return None;
    }
    let buffer = buffer_of(a, total)?;
    drop(guard);
    list.aliased(|recorded| {
        recorded.buffers.push(buffer.clone());
        for c in &copies {
            recorded.kept.push(c.from.clone());
        }
    });
    Some(Copies { buffer, copies })
}

impl Copies {
    /// Records the copies first in the prelude `list`: each texture from its
    /// real state at submission to the copy's and back.
    pub(crate) fn record(&self, list: &ID3D12GraphicsCommandList) {
        const STATES: [D3D12_RESOURCE_STATES; 2] = [
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        ];
        for c in &self.copies {
            let real = c.real;
            let moves = |back: bool| -> Vec<D3D12_RESOURCE_BARRIER> {
                [&c.from, &c.to]
                    .into_iter()
                    .zip(real.into_iter().zip(STATES))
                    .filter(|(_, (real, copy))| real != copy)
                    .map(|(t, (real, copy))| match back {
                        false => crate::host::transition(&t.texture, real, copy),
                        true => crate::host::transition(&t.texture, copy, real),
                    })
                    .collect()
            };
            let (there, back) = (moves(false), moves(true));
            // SAFETY: an open list; live textures, in their real states.
            unsafe {
                if !there.is_empty() {
                    list.ResourceBarrier(&there);
                }
            }
            copy_through(
                list,
                &self.buffer,
                &c.from.texture,
                &c.to.texture,
                &c.written,
                &c.reads,
            );
            // SAFETY: as above.
            unsafe {
                if !back.is_empty() {
                    list.ResourceBarrier(&back);
                }
            }
        }
    }
}

/// The buffer copies go through, at least `size` bytes.
fn buffer_of(a: &mut Aliases, size: u64) -> Option<ID3D12Resource> {
    if let Some((buffer, bytes)) = &a.buffer
        && *bytes >= size
    {
        return Some(buffer.clone());
    }
    let size = size.next_power_of_two();
    let buffer = copy_buffer(size)?;
    a.buffer = Some((buffer.clone(), size));
    Some(buffer)
}

/// Records the copy of `from` into `to`, in the copy states, through
/// `buffer` (in the common state, as it is left).
fn copy_through(
    list: &ID3D12GraphicsCommandList,
    buffer: &ID3D12Resource,
    from: &ID3D12Resource,
    to: &ID3D12Resource,
    written: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    reads: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
) {
    let state = |before, after| crate::host::transition(buffer, before, after);
    // SAFETY: an open list; live resources in the copy states, the buffer
    // large enough for every footprint.
    unsafe {
        list.ResourceBarrier(&[state(
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        for (i, footprint) in written.iter().enumerate() {
            list.CopyTextureRegion(
                &location(buffer, Some(*footprint), 0),
                0,
                0,
                0,
                &location(from, None, i as u32),
                None,
            );
        }
        list.ResourceBarrier(&[state(
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for (i, footprint) in reads.iter().enumerate() {
            list.CopyTextureRegion(
                &location(to, None, i as u32),
                0,
                0,
                0,
                &location(buffer, Some(*footprint), 0),
                None,
            );
        }
        list.ResourceBarrier(&[state(
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
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
    let Some(writer) = writer else {
        return;
    };
    // Alive: letting go of either waits for the lock.
    if let (Some(to), Some(from)) = (super::host_resource(this), super::host_resource(writer)) {
        let what = "not copied: the read starts inside the written memory";
        log_pair(a, start, (this as usize, writer as usize), to, from, what);
    }
}

/// Logs a reader and writer pair over `address` the first time, up to 300
/// pairs (diagnostics).
#[cfg(feature = "diagnostics")]
fn log_pair(
    a: &mut Aliases,
    address: u64,
    pair: (usize, usize),
    reader: &ID3D12Resource,
    writer: &ID3D12Resource,
    what: &str,
) {
    if a.logged.len() >= 300 || !a.logged.insert(pair) {
        return;
    }
    let text = |t: &ID3D12Resource| {
        // SAFETY: a live resource.
        let d = unsafe { t.GetDesc() };
        format!(
            "{}x{}x{} f{} mips {} flags {:#x}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.Flags.0
        )
    };
    tracing::info!(
        address = format_args!("{address:#x}"),
        reader = text(reader),
        writer = text(writer),
        "alias {what}"
    );
}

/// The frame line of alias copies per frame (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn line(frames: f64) -> String {
    let [copies, bytes, other, late, early] = COUNTS
        .each_ref()
        .map(|c| c.swap(0, Relaxed) as f64 / frames);
    if !on() {
        return "  host costs per frame, alias copies: off (VICTORY_NO_ALIAS_COPY=1)".into();
    }
    format!(
        "  host costs per frame, alias copies: {copies:.1} copies {:.2} MiB; {other:.1} newer \
         writes not copied (layouts differ); aliased reads at submission: {late:.1} late (took an \
         older write than one submitted ahead), {early:.1} early (took a write submitted after)",
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

    /// A buffer of `size` bytes on a heap of `kind`.
    fn buffer(kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
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
                    },
                    state,
                    None,
                    &mut b,
                )
                .unwrap();
        }
        b.unwrap()
    }

    /// The 64 rows of 64 four-byte texels at `offset` in a buffer, as
    /// `format`.
    fn rows(format: i32, offset: u64) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format),
                Width: 64,
                Height: 64,
                Depth: 1,
                RowPitch: 256,
            },
        }
    }

    /// An R9G9B9E5 texture over the memory a compute writes as R32_UINT
    /// (run 91's pairs) holds, when its list runs, the newest write
    /// submitted before that list, whatever order the lists were recorded
    /// in, or the write its own list made before the read; one whose rows
    /// hold other bytes takes nothing, and neither does the writer itself.
    #[test]
    fn a_read_over_memory_another_resource_writes_takes_writes_in_submission_order() {
        const KIB16: u64 = 16 << 10;
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
        let upload = buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            4 * KIB16,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let readback = buffer(
            D3D12_HEAP_TYPE_READBACK,
            KIB16,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let pattern = |n: u32| -> Vec<u8> {
            (0..16u32 << 10)
                .map(|i| (i * 7 + n * 31 + 3) as u8)
                .collect()
        };
        let mut data = std::ptr::null_mut();
        // SAFETY: our upload buffer, 64 KiB.
        unsafe {
            upload.Map(0, None, Some(&mut data)).unwrap();
            for n in 0..4 {
                let to = data.cast::<u8>().add(n as usize * KIB16 as usize);
                std::ptr::copy_nonoverlapping(pattern(n).as_ptr(), to, KIB16 as usize);
            }
        }
        // The compute writing pattern `n`.
        let write = |list: &HostList, n: u32| {
            crate::list::ensure_state(list, writer, 0x400);
            // SAFETY: an open list; live resources.
            unsafe {
                list.list.CopyTextureRegion(
                    &location(host_resource(writer).unwrap(), None, 0),
                    0,
                    0,
                    0,
                    &location(&upload, Some(rows(42, u64::from(n) * KIB16)), 0),
                    None,
                );
            }
            gpu_wrote(list, writer);
        };
        // A draw reading the reader, and what it read.
        let read = |list: &HostList| {
            before_read(list, reader);
            crate::list::ensure_state(list, reader, 0x800);
            // SAFETY: an open list; live resources.
            unsafe {
                list.list.CopyTextureRegion(
                    &location(&readback, Some(rows(67, 0)), 0),
                    0,
                    0,
                    0,
                    &location(host_resource(reader).unwrap(), None, 0),
                    None,
                );
            }
        };
        let run = |list: HostList| {
            list.close();
            host.execute(&[&list]);
            crate::pipelined::wait_all();
        };
        let held = || {
            let mut data = std::ptr::null_mut();
            // SAFETY: the finished copy's read-back buffer, 16 KiB.
            unsafe {
                readback.Map(0, None, Some(&mut data)).unwrap();
                let read = std::slice::from_raw_parts(data.cast::<u8>(), KIB16 as usize).to_vec();
                readback.Unmap(0, None);
                read
            }
        };
        #[cfg(feature = "diagnostics")]
        let counts = || [2, 3, 4].map(|i| COUNTS[i].load(Relaxed));
        #[cfg(feature = "diagnostics")]
        let before = counts();
        let list = host.new_list().unwrap();
        write(&list, 0);
        run(list);
        let list = host.new_list().unwrap();
        assert!(!before_read(&list, writer), "the writer has its writes");
        assert!(!before_read(&list, narrow), "another list's write waits");
        read(&list);
        run(list);
        assert!(held() == pattern(0), "the reader holds the write");
        // Recorded before the write's list, submitted after it.
        let (reading, writing) = (host.new_list().unwrap(), host.new_list().unwrap());
        read(&reading);
        write(&writing, 1);
        run(writing);
        run(reading);
        assert!(
            held() == pattern(1),
            "a read submitted after a write takes it"
        );
        // Recorded after the write's list, submitted before it.
        let (writing, reading) = (host.new_list().unwrap(), host.new_list().unwrap());
        write(&writing, 2);
        read(&reading);
        run(reading);
        assert!(
            held() == pattern(1),
            "a read submitted before a write does not"
        );
        run(writing);
        let list = host.new_list().unwrap();
        write(&list, 3);
        assert!(
            before_read(&list, reader),
            "the list's own write is copied at the read"
        );
        read(&list);
        run(list);
        assert!(held() == pattern(3), "the reader holds its list's write");
        #[cfg(feature = "diagnostics")]
        {
            let now = counts();
            let since: Vec<u64> = now.iter().zip(before).map(|(n, b)| n - b).collect();
            // The narrow reader: rows of other bytes, so it keeps no write.
            assert_eq!(since, [1, 1, 0], "not copied, late and early reads");
        }
        assert_eq!(host.log_messages(), None, "debug layer errors");
        for r in [narrow, reader, writer] {
            crate::object::release_boxed::<Resource>(r);
        }
    }
}
