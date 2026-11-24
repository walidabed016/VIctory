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
            last.read.resize(size, 0);
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            #[cfg(feature = "diagnostics")]
            crate::copystat::cost(crate::copystat::Op::Copy, started, start, size as u64);
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
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
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
    const OUTCOMES: usize = 7;
    const NAMES: [&str; OUTCOMES] = [
        "unwritten",
        "written",
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
        let hashes = |s: &Seen| s.outcomes[3] + s.outcomes[4] + s.outcomes[5];
        let changes = |s: &Seen| s.outcomes[1] + s.outcomes[4] + s.outcomes[5] + s.outcomes[6];
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
