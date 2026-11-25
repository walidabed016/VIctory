//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling,
//! as the write watch or the guard on views and pools tells
//! ([`crate::watch`]); in memory neither tells about, a content hash of the
//! whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
    /// Detilings so far, and the one the host texture last got.
    generation: std::sync::atomic::AtomicU64,
    uploaded: std::sync::atomic::AtomicU64,
    /// What it is, for the per-texture refresh lines (diagnostics).
    #[cfg(feature = "diagnostics")]
    name: String,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
    /// Times the watch found the memory unwritten (diagnostics).
    #[cfg(feature = "diagnostics")]
    unwritten: u64,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    #[cfg(feature = "diagnostics")]
    crate::capture::note_gpu_write(resource);
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Once per list: has `resource`'s staging buffer refreshed at submission
/// if its memory changed, and copied into its host copy before the list
/// runs (in the list's prelude) if it holds contents the host copy has not
/// had yet.
pub(crate) fn upload(list: &HostList, resource: This) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let (Some(copy), Some(texture)) = (
        crate::resource::host_copy(resource),
        crate::resource::host_resource(resource),
    ) else {
        return;
    };
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    // The staging buffer lives as long as `staging`, which the list holds
    // until it is reset.
    list.defer_texture(staging, copy);
}

/// Subresources and bytes copied from staging buffers into host textures
/// since the last frame-rate line (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static UPLOADED: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Staging {
    /// Whether the staging buffer holds contents its host texture has not
    /// had yet; they count as had from here on (the caller copies them).
    pub(crate) fn due(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let generation = self.generation.load(Relaxed);
        generation != 0 && self.uploaded.swap(generation, Relaxed) != generation
    }

    /// Records the copy of every subresource into `texture` (in the copy
    /// destination state) in `list`.
    pub(crate) fn copy_into(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            UPLOADED[0].fetch_add(self.footprints.len() as u64, Relaxed);
            UPLOADED[1].fetch_add(self.size, Relaxed);
            crate::capture::note_upload(texture);
        }
    }

    /// The title memory it detiles: start and end.
    pub(crate) fn memory(&self) -> (usize, usize) {
        let start = self.memory as usize;
        (start, start + self.layout.size as usize)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all); true if it did: the
    /// staging buffer holds new contents.
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, detiled) = self.refresh_now();
        let us = started.elapsed().as_micros() as u64;
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(us, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        #[cfg(feature = "diagnostics")]
        seen::note(self, None, us);
        detiled
    }

    /// Returns the bytes of title memory read, and whether it detiled them.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, true) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                #[cfg(feature = "diagnostics")]
                if self.verify(last) {
                    return (size, true);
                }
                #[cfg(feature = "diagnostics")]
                seen::note(self, Some(seen::Outcome::Watched), 0);
                return (0, false);
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            let first = last.read.len() != size;
            last.read.resize(size, 0);
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let changed = read(&mut last.read, start) || first;
            #[cfg(feature = "diagnostics")]
            crate::copystat::cost(crate::copystat::Op::Copy, started, start, size as u64);
            // Written as the same bytes, or elsewhere in the pages it shares:
            // the staging buffer, and so the host texture, stay as they are.
            if !changed {
                #[cfg(feature = "diagnostics")]
                seen::note(self, Some(seen::Outcome::WrittenSame), 0);
                return (size, false);
            }
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            if verify_every() != 0 {
                last.hash = Some(blake3::hash(&last.read));
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            #[cfg(feature = "diagnostics")]
            seen::note(self, Some(seen::Outcome::WatchedWritten), 0);
            return (size, true);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            #[cfg(feature = "diagnostics")]
            seen::note(self, Some(seen::Outcome::Uncommitted), 0);
            return (0, false);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(crate::copystat::Op::Hash, started, start, size as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            #[cfg(feature = "diagnostics")]
            seen::note(self, Some(seen::Outcome::Hashed), 0);
            return (size, false);
        }
        #[cfg(feature = "diagnostics")]
        seen::note(
            self,
            Some(match last.hash {
                None => seen::Outcome::HashedFirst,
                Some(_) => seen::Outcome::HashedChanged,
            }),
            0,
        );
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Diagnostics (`VICTORY_TEXTURE_VERIFY=N`): every Nth time the watch
    /// finds the memory unwritten, whether it changed all the same, by its
    /// hash (a write nothing saw: the texture would show stale contents).
    /// Logs it and detiles the memory; true if it did.
    #[cfg(feature = "diagnostics")]
    fn verify(&self, last: &mut Last) -> bool {
        let every = verify_every();
        last.unwritten += 1;
        if every == 0 || !last.unwritten.is_multiple_of(every) {
            return false;
        }
        let (start, end) = self.memory();
        let started = std::time::Instant::now();
        let mut now = vec![0; end - start];
        crate::host::copy_mapped(now.as_mut_ptr(), self.memory, end - start, false);
        let hash = blake3::hash(&now);
        crate::copystat::cost(crate::copystat::Op::Hash, started, start, now.len() as u64);
        crate::copystat::add(crate::copystat::C::TexHashed, now.len() as u64);
        // Written since it was asked about: its fault opened a block, and
        // the next asking takes it.
        if last.hash == Some(hash) || !crate::watch::unwritten(start, end) {
            return false;
        }
        static STALE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if STALE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100 {
            tracing::warn!(
                "texture stale: {start:#x}+{:#x} ({}): its memory changed though no write to it was seen",
                end - start,
                self.name
            );
        }
        seen::note(self, Some(seen::Outcome::Stale), 0);
        last.hash = Some(hash);
        last.read = now;
        self.detile(&last.read);
        true
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let _detiling = Detiling(std::time::Instant::now(), self.memory as usize, tiled.len());
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            let detiled = if f.Depth == 1 && len >= SPLIT {
                detile_split(&self.layout, tiled, mip, slice, linear, pitch)
            } else {
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            };
            if let Err(e) = detiled {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Charges a detiling's time when it ends (diagnostics).
#[cfg(feature = "diagnostics")]
struct Detiling(std::time::Instant, usize, usize);

#[cfg(feature = "diagnostics")]
impl Drop for Detiling {
    fn drop(&mut self) {
        crate::copystat::cost(crate::copystat::Op::Detile, self.0, self.1, self.2 as u64);
    }
}

/// Every how many unwritten findings a texture's memory is hashed to
/// verify them (`VICTORY_TEXTURE_VERIFY`; 0, the default: never).
#[cfg(feature = "diagnostics")]
fn verify_every() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_VERIFY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// A copy or a subresource this big is split between threads: on one, the
/// world's screen-sized texture that changes every frame (14.7 MB) took
/// ~2.4 ms a frame to detile, and as long again to copy.
const SPLIT: usize = 4 << 20;

/// Threads a big copy or detiling is split between.
fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
}

/// Reads title memory at `from` into `to` (its length) as
/// [`crate::host::copy_mapped`] does, unmapped pages as zeros, a piece at a
/// time: whether any piece differed from what `to` held. Split between
/// threads when big.
fn read(to: &mut [u8], from: usize) -> bool {
    /// L2-sized: each piece is compared while it is still in the cache.
    const PIECE: usize = 64 << 10;
    let pieces = |to: &mut [u8], from: usize| {
        let mut piece = vec![0u8; PIECE.min(to.len())];
        let mut changed = false;
        for (k, held) in to.chunks_mut(PIECE).enumerate() {
            let read = &mut piece[..held.len()];
            let at = (from + k * PIECE) as *const u8;
            crate::host::copy_mapped(read.as_mut_ptr(), at, read.len(), false);
            if read != held {
                held.copy_from_slice(read);
                changed = true;
            }
        }
        changed
    };
    if to.len() < SPLIT {
        return pieces(to, from);
    }
    let per = to.len().div_ceil(threads()).next_multiple_of(PIECE);
    let parts = to
        .chunks_mut(per)
        .enumerate()
        .map(|(k, chunk)| (chunk, from + k * per))
        .collect();
    split(parts, |(chunk, from)| pieces(chunk, from))
        .into_iter()
        .any(|changed| changed.unwrap_or(true))
}

/// [`victory_formats::texture::Layout::detile`] of a big subresource with
/// one depth slice, its rows split between threads.
fn detile_split(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    pitch: usize,
) -> Result<(), victory_formats::texture::Error> {
    let height = layout.mips[mip as usize].height;
    let per = height.div_ceil(threads() as u32).max(1);
    let mut rest = linear;
    let mut parts = Vec::new();
    for first in (0..height).step_by(per as usize) {
        let end = (first + per).min(height);
        let n = ((end - first) as usize * pitch).min(rest.len());
        let (rows, tail) = std::mem::take(&mut rest).split_at_mut(n);
        rest = tail;
        parts.push((first..end, rows));
    }
    split(parts, |(rows, linear)| {
        layout.detile_rows(tiled, mip, slice, rows, linear, pitch)
    })
    .into_iter()
    .try_for_each(|done| done.unwrap_or(Err(victory_formats::texture::Error::Invalid)))
}

/// Runs `f` on each of `items` on the split threads, the caller among
/// them: the results in order, a panic as its payload.
///
/// The threads start once and stay: a thread's start and end run every
/// title image's thread notices (their TLS callbacks and `DllMain`), which
/// made a split that started its own threads cost ~110 ms whatever its
/// size (run 68: 12.7 MB a frame copied in 360 ms and detiled in 320 ms).
pub(crate) fn split<I: Send, R: Send>(
    items: Vec<I>,
    f: impl Fn(I) -> R + Sync,
) -> Vec<std::thread::Result<R>> {
    use std::collections::VecDeque;
    use std::sync::{Condvar, Mutex, MutexGuard, Once};
    type Job = Box<dyn FnOnce() + Send>;
    static QUEUE: Mutex<VecDeque<Job>> = Mutex::new(VecDeque::new());
    static QUEUED: Condvar = Condvar::new();
    fn queue() -> MutexGuard<'static, VecDeque<Job>> {
        QUEUE.lock().unwrap_or_else(|e| e.into_inner())
    }
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        for _ in 1..threads() {
            let worker = victory_core::thread::spawn_host("victory texture split", || {
                loop {
                    let mut waiting = queue();
                    let job = loop {
                        match waiting.pop_front() {
                            Some(job) => break job,
                            None => {
                                waiting = QUEUED.wait(waiting).unwrap_or_else(|e| e.into_inner())
                            }
                        }
                    };
                    drop(waiting);
                    job();
                }
            });
            if worker.is_err() {
                break;
            }
        }
    });
    let results: Vec<Mutex<Option<std::thread::Result<R>>>> =
        items.iter().map(|_| Mutex::new(None)).collect();
    let left = (Mutex::new(items.len()), Condvar::new());
    {
        let mut waiting = queue();
        for (item, result) in items.into_iter().zip(&results) {
            let (f, left) = (&f, &left);
            let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
                let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(item)));
                *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(done);
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            });
            // SAFETY: the job borrows from this call, which returns only
            // once every job it queued has run; a job does not unwind.
            waiting.push_back(unsafe {
                std::mem::transmute::<Box<dyn FnOnce() + Send + '_>, Job>(job)
            });
        }
    }
    QUEUED.notify_all();
    // The caller runs jobs too: they run even if no thread could start.
    loop {
        let job = queue().pop_front();
        let Some(job) = job else { break };
        job();
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
    drop(n);
    results
        .into_iter()
        .map(|r| {
            r.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .expect("every job ran")
        })
        .collect()
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive): resident or view pages as the
/// working set tells, else as `VirtualQuery` does.
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION};
    let mut p = at as usize;
    let end = p + len;
    if crate::watch::resident(p, end) {
        return true;
    }
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = crate::watch::query(p, &mut info);
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
        generation: Default::default(),
        uploaded: Default::default(),
        #[cfg(feature = "diagnostics")]
        name: format!(
            "{}x{}x{} f{} mips {}, {}",
            d.Width,
            d.Height,
            d.DepthOrArraySize,
            d.Format.0,
            d.MipLevels,
            crate::resource::backing_text(resource)
        ),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

/// Diagnostics: what each texture's refreshes found, logged with the
/// frame-rate line: totals by memory kind, then the textures that cost most
/// and those that changed most (`VICTORY_TEXTURE_LINES` lines each, 12 by
/// default).
#[cfg(feature = "diagnostics")]
pub(crate) mod seen {
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Clone, Copy)]
    pub(crate) enum Outcome {
        /// Watched (or guarded), not written since it was detiled.
        Watched,
        /// Watched and written since: detiled.
        WatchedWritten,
        /// Watched and written since, its bytes the same: left as it was.
        WrittenSame,
        /// Not committed: left as it was.
        Uncommitted,
        /// Hashed, unchanged.
        Hashed,
        /// Hashed for the first time: detiled.
        HashedFirst,
        /// Hashed and changed: detiled.
        HashedChanged,
        /// Unwritten as the watch tells, changed as its hash tells: detiled
        /// (`VICTORY_TEXTURE_VERIFY`).
        Stale,
    }
    const OUTCOMES: usize = 8;
    const NAMES: [&str; OUTCOMES] = [
        "unwritten",
        "written",
        "written same",
        "uncommitted",
        "hashed same",
        "hashed first",
        "hashed changed",
        "STALE",
    ];

    struct Seen {
        name: String,
        size: usize,
        memory: &'static str,
        refreshes: u64,
        us: u64,
        outcomes: [u64; OUTCOMES],
        /// Hash changes while a copy of ours had written the memory since
        /// the submission began.
        ours: u64,
    }

    static SEEN: Mutex<Option<HashMap<usize, Seen>>> = Mutex::new(None);

    /// A refresh of `staging` found `outcome`; `None`: it ended, taking
    /// `us`.
    pub(crate) fn note(staging: &super::Staging, outcome: Option<Outcome>, us: u64) {
        let (start, end) = staging.memory();
        let ours = matches!(outcome, Some(Outcome::HashedChanged))
            && crate::watch::wrote_since(0, start, end);
        let mut map = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let s = map
            .get_or_insert_with(HashMap::new)
            .entry(start)
            .or_insert_with(|| Seen {
                name: staging.name.clone(),
                size: end - start,
                memory: memory(start),
                refreshes: 0,
                us: 0,
                outcomes: [0; OUTCOMES],
                ours: 0,
            });
        match outcome {
            Some(o) => s.outcomes[o as usize] += 1,
            None => {
                s.refreshes += 1;
                s.us += us;
            }
        }
        s.ours += u64::from(ours);
    }

    /// The kind of memory at `at`: a view (physical pages), write-watched,
    /// or neither.
    fn memory(at: usize) -> &'static str {
        use windows_sys::Win32::System::Memory::GetWriteWatch;
        if crate::watch::view(at) {
            return "view";
        }
        let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
        // SAFETY: room for one answer; without the reset flag nothing changes.
        let watched = unsafe {
            GetWriteWatch(
                0,
                (at & !0xfff) as *const _,
                0x1000,
                (&raw mut found).cast(),
                &mut count,
                &mut granularity,
            ) == 0
        };
        if watched { "watched" } else { "unwatched" }
    }

    /// Logs the textures refreshed per frame over `frames` frames.
    pub(crate) fn log(frames: f64) {
        let Some(map) = SEEN.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        let lines = std::env::var("VICTORY_TEXTURE_LINES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12);
        let mut seen: Vec<(usize, Seen)> = map.into_iter().collect();
        let hashes = |s: &Seen| s.outcomes[4] + s.outcomes[5] + s.outcomes[6];
        let changes = |s: &Seen| s.outcomes[1] + s.outcomes[5] + s.outcomes[6] + s.outcomes[7];
        for kind in ["view", "watched", "unwatched"] {
            let of: Vec<&Seen> = seen
                .iter()
                .map(|(_, s)| s)
                .filter(|s| s.memory == kind)
                .collect();
            if of.is_empty() {
                continue;
            }
            let mut outcomes = [0u64; OUTCOMES];
            for s in &of {
                for (t, n) in outcomes.iter_mut().zip(s.outcomes) {
                    *t += n;
                }
            }
            let sum =
                |f: &dyn Fn(&Seen) -> u64| of.iter().map(|s| f(s)).sum::<u64>() as f64 / frames;
            let text: Vec<String> = NAMES
                .iter()
                .zip(outcomes)
                .filter(|&(_, n)| n > 0)
                .map(|(name, n)| format!("{name} {:.1}", n as f64 / frames))
                .collect();
            tracing::info!(
                "textures refreshed per frame, {kind} memory: {} distinct, {:.1} refreshes {:.2} ms, {:.1} MB hashed, {:.1} MB detiled; {}; {:.1} hash changes after copies of ours",
                of.len(),
                sum(&|s| s.refreshes),
                sum(&|s| s.us) / 1e3,
                sum(&|s| hashes(s) * s.size as u64) / 1e6,
                sum(&|s| changes(s) * s.size as u64) / 1e6,
                text.join(", "),
                sum(&|s| s.ours),
            );
        }
        let line = |order: &str, at: usize, s: &Seen| {
            let text: Vec<String> = NAMES
                .iter()
                .zip(s.outcomes)
                .filter(|&(_, n)| n > 0)
                .map(|(name, n)| format!("{name} {:.1}", n as f64 / frames))
                .collect();
            tracing::info!(
                "  texture {order} {at:#x}+{:#x} ({}, {}): {:.1} refreshes {:.2} ms per frame; {}",
                s.size,
                s.memory,
                s.name,
                s.refreshes as f64 / frames,
                s.us as f64 / 1e3 / frames,
                text.join(", ")
            );
        };
        seen.sort_by_key(|(_, s)| std::cmp::Reverse(s.us));
        for (at, s) in seen.iter().take(lines) {
            line("costing", *at, s);
        }
        seen.sort_by_key(|(_, s)| std::cmp::Reverse(changes(s)));
        for (at, s) in seen.iter().take(lines).filter(|(_, s)| changes(s) > 0) {
            line("changing", *at, s);
            // Which title buffer's copy holds its memory: shaders writing it
            // change the memory through copy-back.
            let at = *at as u64;
            if let Some(b) = crate::shadow::find(at, s.size as u64) {
                let (into, left) = (b.offset(at), b.left(at));
                tracing::info!(
                    "    in the copy of title buffer {:#x}+{:#x} ({:?}), shaders write it: {}",
                    at - into,
                    into + left,
                    b.caller(),
                    b.shaders_write()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

    /// A big subresource detiled on several threads, and a big read of memory
    /// split between them, give what one thread does.
    #[test]
    fn big_copies_and_detilings_split_between_threads() {
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 2048,
            height: 1031,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        })
        .unwrap();
        let memory: Vec<u8> = (0..layout.size as usize)
            .map(|i| (i * 13 + i / 4093) as u8)
            .collect();
        assert!(memory.len() >= SPLIT);
        let mut tiled = vec![0; memory.len()];
        assert!(read(&mut tiled, memory.as_ptr() as usize));
        assert!(tiled == memory);
        assert!(
            !read(&mut tiled, memory.as_ptr() as usize),
            "the same bytes"
        );
        tiled[SPLIT - 5] ^= 1;
        assert!(read(&mut tiled, memory.as_ptr() as usize));
        assert!(tiled == memory);
        let pitch = 2048 * 4 + 256;
        let mut one = vec![0; pitch * 1031];
        layout.detile(&tiled, 0, 0, &mut one, pitch, 0).unwrap();
        let mut split = vec![0; one.len()];
        detile_split(&layout, &tiled, 0, 0, &mut split, pitch).unwrap();
        assert!(split == one);
    }

    /// Splits run on threads started once, give their results in order, and
    /// a part that panics as its payload.
    #[test]
    fn splits_run_on_the_same_threads() {
        let mut ran = std::collections::HashSet::new();
        for _ in 0..50 {
            for id in split((0..16).collect(), |_: u32| std::thread::current().id()) {
                ran.insert(id.unwrap());
            }
        }
        // The workers and this thread, and another test's that took a job.
        assert!(ran.len() <= threads() + 1, "{} threads", ran.len());
        let done = split(vec![1, 2, 3], |x| {
            assert!(x != 2);
            x
        });
        assert!(matches!(done[..], [Ok(1), Err(_), Ok(3)]));
    }
}
