//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer and constant views in tables and in heap slots
//! user data names); views in heap slots only shaders know (an index they
//! load from memory) keep what the first stage that may read them after
//! the title wrote them copied, and what shaders write there goes back to
//! title memory only after that stage's submission.
//!
//! A title buffer of at least [`IN_PLACE_MIN`] bytes in committed private
//! memory needs no copy: a heap opened on its pages
//! (`OpenExistingHeapFromAddress1`) holds a buffer that is title memory
//! itself, as on the console, so the GPU reads what the title wrote and
//! the title reads what shaders wrote, with nothing to refresh, watch or
//! copy back. The world's big buffers, copied, cost ~1 s per frame
//! (dozens of overlapping 174 MB ranges per submission).
//! `VICTORY_COPY_TITLE_MEMORY=1` turns this off.
//!
//! Title memory outside buffers (loose memory) has one host copy, the
//! mirror ([`mirror`]): reserved buffers ("slabs") of [`SLAB`] bytes of
//! title addresses each, [`OVERLAP`] more past their end, in which each
//! 4 MiB window of title memory copied so far is mapped (64 KiB tiles) onto
//! pooled host memory at the window's own offset. A title address has one
//! host address, contiguous with its neighbours across windows, so root
//! views, which have no size, read as far past their address as on the
//! console (Grand Theft Auto VI's skinning reads up to 36 MiB past one);
//! windows never copied read as zeros instead of faulting.
//! `VICTORY_LOOSE_WINDOWS=1` copies loose memory into separate windows, as
//! before the mirror.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Where a copy's bytes are on the CPU.
#[derive(Clone, Copy)]
enum Memory {
    /// One mapping of the whole copy.
    Linear(*mut u8),
    /// A slab of the mirror: each window's bytes where the mirror placed
    /// them when they were first copied ([`Mirror::window`]).
    Mirror,
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    memory: Memory,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// It copies title memory outside title buffers (a window of it, or a
    /// slab of the mirror).
    loose: bool,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Shaders have been given the copy to write ([`Shadow::gpu_writes`]).
    shaders_write: AtomicBool,
    /// Ranges shaders of the submission write, until it is copied back.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, by offset, covering the
    /// ranges its shaders write: what the GPU wrote differs. One snapshot
    /// for a buffer's copy; for a mirror slab, which spans a GiB of title
    /// memory, one per region written.
    before: Mutex<Vec<(usize, Vec<u8>)>>,
    /// In place of snapshots over view memory ([`Shadow::arm`]): the regions
    /// whose title blocks are armed for the submission's copy-back, as
    /// (from, to, asking) offsets, and the blocks of them not armed, kept as
    /// the copy held them (by title block).
    armed: Mutex<Vec<(usize, usize, u64)>>,
    kept: Mutex<BTreeMap<usize, Vec<u8>>>,
    /// The buffer is title memory itself ([`in_place`]): nothing to copy.
    in_place: bool,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

/// Runs `f` over `0..n` in parts, on several threads where the parts
/// cover `len` bytes of at least 16 MiB (one thread copies or compares
/// ~10 GB/s; the host's memory takes several times that). The results in
/// order.
fn split<T: Send>(n: usize, len: usize, f: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let threads = match len {
        0..0x100_0000 => 1,
        _ => std::thread::available_parallelism().map_or(1, |t| t.get().min(8)),
    };
    if threads == 1 {
        return vec![f(0..n)];
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        let f = &f;
        let parts: Vec<_> = (0..threads)
            .map(|k| s.spawn(move || f((k * per).min(n)..((k + 1) * per).min(n))))
            .collect();
        parts
            .into_iter()
            .map(|p| p.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// What a copy held before a submission, for its copy-back: snapshots
/// (offset, bytes), blocks kept (title block, bytes) and armed regions
/// (from, to, asking; [`Shadow::arm`]).
struct Was<'a> {
    snapshots: &'a [(usize, Vec<u8>)],
    kept: &'a BTreeMap<usize, Vec<u8>>,
    armed: &'a [(usize, usize, u64)],
    start: usize,
}

enum Before<'a> {
    Bytes(&'a [u8]),
    /// As the armed title block holds it, armed at this asking.
    Armed(u64),
}

impl Was<'_> {
    /// What the copy held at offsets `[at, at + n)` (in one title block),
    /// if anything tells.
    fn at(&self, at: usize, n: usize) -> Option<Before<'_>> {
        if let Some((taken, bytes)) = self
            .snapshots
            .iter()
            .find(|(t, b)| *t <= at && at + n <= t + b.len())
        {
            return Some(Before::Bytes(&bytes[at - taken..at - taken + n]));
        }
        let block = (self.start + at) / PAGE;
        if let Some(bytes) = self.kept.get(&block) {
            let first = (block * PAGE).max(self.start) - self.start;
            return Some(Before::Bytes(&bytes[at - first..at - first + n]));
        }
        self.armed
            .iter()
            .find(|&&(a, b, _)| a <= at && at + n <= b)
            .map(|&(.., asked)| Before::Armed(asked))
    }
}

/// Whether copies of view memory that shaders write arm its blocks instead
/// of taking snapshots ([`Shadow::arm`]); `VICTORY_EAGER_SNAPSHOTS` turns
/// it off.
fn armed_snapshots() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_EAGER_SNAPSHOTS").is_none();
        if !on {
            tracing::info!("copies shaders write are snapshotted whole (VICTORY_EAGER_SNAPSHOTS)");
        }
        on
    })
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; the memory a copy maps is
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Whether this is a slab of the mirror of loose memory.
    pub(crate) fn mirror(&self) -> bool {
        matches!(self.memory, Memory::Mirror)
    }

    /// Whom refreshes of this copy charge their costs to (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn caller(&self) -> crate::copystat::Caller {
        use crate::copystat::Caller;
        match (self.mirror(), self.loose) {
            (true, _) => Caller::Mirror,
            (false, true) => Caller::Window,
            (false, false) => Caller::Buffer,
        }
    }

    /// Whether shaders have been given this copy to write.
    pub(crate) fn shaders_write(&self) -> bool {
        self.shaders_write.load(Ordering::Relaxed)
    }

    /// The word the copy holds at a title address inside the buffer
    /// (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64) -> Option<u32> {
        let (from, to) = self.inside(address as usize, 4);
        let at = self.host_at(from, to - from);
        // SAFETY: inside the copy's memory.
        (to - from == 4 && !at.is_null()).then(|| unsafe { at.cast::<u32>().read_unaligned() })
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(
        buffer: ID3D12Resource,
        memory: Memory,
        gpu: u64,
        start: u64,
        len: u64,
        in_place: bool,
    ) -> Shadow {
        let pages = match in_place {
            true => 0,
            false => (start + len).div_ceil(PAGE as u64) - start / PAGE as u64,
        };
        Shadow {
            buffer,
            memory,
            gpu,
            start,
            len,
            loose: false,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            shaders_write: AtomicBool::new(false),
            pending: Mutex::default(),
            before: Mutex::default(),
            armed: Mutex::default(),
            kept: Mutex::default(),
            in_place,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// The copy's bytes at offsets `[from, to)` in runs contiguous on the
    /// CPU: offset, host pointer and length. The pointer is null for a
    /// mirror window that has no host memory yet (the GPU reads zeros
    /// there); `make` gives such windows memory first.
    fn pieces(&self, from: usize, to: usize, make: bool) -> Vec<(usize, *mut u8, usize)> {
        if from >= to {
            return Vec::new();
        }
        let Memory::Linear(data) = self.memory else {
            let (start, w) = (self.start as usize, WINDOW as usize);
            let mut mirror = mirror_state();
            let mut pieces = Vec::new();
            let mut at = from;
            while at < to {
                let title = start + at;
                let next = ((title / w + 1) * w - start).min(to);
                let base = mirror.window((title / w) as u64, make);
                let host = if base.is_null() {
                    base
                } else {
                    base.wrapping_add(title % w)
                };
                pieces.push((at, host, next - at));
                at = next;
            }
            return pieces;
        };
        vec![(from, data.wrapping_add(from), to - from)]
    }

    /// The host pointer of the copy's `n` bytes at `offset` (within one
    /// mirror window), null where the mirror has no memory for them.
    fn host_at(&self, offset: usize, n: usize) -> *mut u8 {
        match self.memory {
            Memory::Linear(data) => data.wrapping_add(offset),
            Memory::Mirror => match self.pieces(offset, offset + n.max(1), false).as_slice() {
                &[(_, at, _)] => at,
                _ => std::ptr::null_mut(),
            },
        }
    }

    /// Appends the copy's bytes at offsets `[from, to)` to `out` (zeros
    /// where the mirror has no memory).
    fn gather(&self, from: usize, to: usize, out: &mut Vec<u8>) {
        const CHUNK: usize = 4 << 20;
        let filled = out.len();
        out.reserve(to - from);
        // (where in `out`, host address or 0, bytes)
        let mut chunks = Vec::new();
        for (offset, at, n) in self.pieces(from, to, false) {
            for k in (0..n).step_by(CHUNK) {
                let host = if at.is_null() { 0 } else { at as usize + k };
                chunks.push((filled + offset - from + k, host, (n - k).min(CHUNK)));
            }
        }
        let base = out.as_mut_ptr() as usize;
        split(chunks.len(), to - from, |part| {
            for &(to, host, n) in &chunks[part] {
                let to = (base + to) as *mut u8;
                // SAFETY: `to` lies in the room reserved past `filled`, the
                // chunks apart; `host` inside the copy's memory.
                unsafe {
                    match host {
                        0 => to.write_bytes(0, n),
                        host => to.copy_from_nonoverlapping(host as *const u8, n),
                    }
                }
            }
        });
        // SAFETY: every byte up to there was written above.
        unsafe { out.set_len(filled + to - from) };
    }

    /// Fills `dst` with the copy's bytes from offset `from` on (zeros where
    /// the mirror has no memory).
    fn fill(&self, from: usize, dst: &mut [u8]) {
        let mut done = 0;
        for (_, at, n) in self.pieces(from, from + dst.len(), false) {
            let to = &mut dst[done..done + n];
            if at.is_null() {
                to.fill(0);
            } else {
                // SAFETY: inside the copy's memory.
                to.copy_from_slice(unsafe { std::slice::from_raw_parts(at, n) });
            }
            done += n;
        }
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.shaders_write.store(true, Ordering::Relaxed);
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per region of the copy and submission, grown to cover
    /// every range the submission declares there (the world declares
    /// overlapping views of its 174 MB heap dozens of times), joining
    /// ranges less than a window apart, in reused memory (fresh snapshots
    /// cost ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.in_place {
            return;
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        if from < to && armed_snapshots() && crate::watch::view(self.start as usize + from) {
            self.arm(from, to);
            return;
        }
        let w = WINDOW as usize;
        let joins = |at: usize, n: usize| at <= to + w && from <= at + n + w;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (mut start, mut end) = (from, to);
        let mut taken = Vec::new();
        let mut i = 0;
        while i < before.len() {
            let (at, n) = (before[i].0, before[i].1.len());
            if joins(at, n) {
                (start, end) = (start.min(at), end.max(at + n));
                taken.push(before.swap_remove(i));
            } else {
                i += 1;
            }
        }
        if let [(at, bytes)] = taken.as_slice()
            && (*at, at + bytes.len()) == (start, end)
        {
            before.append(&mut taken);
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken (the
        // snapshots joined lie apart, in order).
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        taken.sort_unstable_by_key(|t| t.0);
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken_at, bytes) in taken {
            self.gather(at, taken_at, &mut grown);
            grown.extend_from_slice(&bytes);
            at = taken_at + bytes.len();
            give_back(bytes);
        }
        self.gather(at, end, &mut grown);
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(
            crate::copystat::Op::Snapshot,
            started,
            self.start as usize + start,
            (end - start) as u64,
        );
        before.push((start, grown));
    }

    /// [`Shadow::writing`] over view memory: no snapshot. The title blocks
    /// of offsets `[from, to)` are armed instead ([`crate::watch::arm`]):
    /// title memory holds what the copy does until the title writes a
    /// block, whose bytes are saved first. Blocks that cannot be armed
    /// (written since the copy took them, at the range's ends, not mapped)
    /// are kept as the copy holds them.
    fn arm(&self, from: usize, to: usize) {
        let start = self.start as usize;
        let (asked, loose) = {
            let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            let base = start / PAGE;
            crate::watch::arm(start + from, start + to, |page| {
                copied.get(page.wrapping_sub(base)).copied().unwrap_or(0)
            })
        };
        self.armed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((from, to, asked));
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let (started, had) = (std::time::Instant::now(), kept.len());
        for block in loose.into_iter().flat_map(|(a, b)| a..b) {
            kept.entry(block).or_insert_with(|| self.block_bytes(block));
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(
            crate::copystat::Op::Snapshot,
            started,
            start + from,
            ((kept.len() - had) * PAGE) as u64,
        );
    }

    /// The copy's bytes of title block `block` (64 KiB), cut to the buffer.
    fn block_bytes(&self, block: usize) -> Vec<u8> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (a, b) = ((block * PAGE).max(start), ((block + 1) * PAGE).min(end));
        let mut bytes = Vec::with_capacity(b - a);
        self.gather(a - start, b - start, &mut bytes);
        bytes
    }

    /// After the submission ran: copies the bytes its shaders wrote in the
    /// ranges it declared (address, bytes; [`Shadow::writing`]) back into
    /// title memory (unmapped pages left alone), and only those: the title's
    /// threads write the same buffers while the GPU runs (its command memory
    /// shares the big GPU heap with what shaders write), and putting back the
    /// copy's older bytes there broke the title's own records (a fence list
    /// read as -1, title+0x4e9b54a). Each byte is compared once however many
    /// declared ranges hold it (the world's overlap dozens of times).
    /// Returns the bytes written.
    pub(crate) fn copy_back(&self, declared: &[(usize, usize)]) -> usize {
        let mut ranges: Vec<(usize, usize)> = declared
            .iter()
            .map(|&(address, len)| self.inside(address, len))
            .filter(|(from, to)| from < to)
            .collect();
        ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (from, to) in ranges {
            match merged.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => merged.push((from, to)),
            }
        }
        let start = self.start as usize;
        if self.in_place {
            // Shaders wrote title memory; copies and textures of it that
            // the kernel's watch cannot tell (no CPU wrote) take it again.
            for (from, to) in merged {
                crate::watch::wrote(start + from, start + to);
            }
            return 0;
        }
        let mut snapshots = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut armed = self.armed.lock().unwrap_or_else(|e| e.into_inner());
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let was = Was {
            snapshots: &snapshots,
            kept: &kept,
            armed: &armed,
            start,
        };
        let mut written = 0;
        for (from, to) in merged {
            written += self.write_back(from, to, &was);
        }
        // The submission is done with its armed blocks (its copy-back
        // disarms them all).
        kept.clear();
        armed.clear();
        // Every submitted step that declared the ranges is done with them
        // (one submission runs at a time), and the snapshots with the last.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&(a, b)| !declared.contains(&(a, b - a)));
        if pending.is_empty() {
            for (_, bytes) in snapshots.drain(..) {
                give_back(bytes);
            }
        }
        written
    }

    /// [`Shadow::copy_back`] of offsets `[from, to)`, whose bytes before
    /// the submission `was` tells: by 64 KiB piece, the copy compared with
    /// them (on several threads for big ranges; armed blocks as title memory
    /// holds them), title memory read only where the GPU wrote.
    fn write_back(&self, from: usize, to: usize, was: &Was) -> usize {
        const PIECE: usize = 64 << 10;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let start = self.start as usize;
        // (offset, host address, bytes); none where a mirror window was
        // never copied: shaders read zeros there and their writes went
        // nowhere.
        let mut pieces = Vec::with_capacity((to - from).div_ceil(PIECE) + 1);
        let mut at = from;
        while at < to {
            let n = (PIECE - (start + at) % PIECE).min(to - at);
            let host = self.host_at(at, n);
            if !host.is_null() {
                pieces.push((at, host as usize, n));
            }
            at += n;
        }
        let changed: Vec<usize> = split(pieces.len(), to - from, |part| {
            let mut seen = None;
            part.filter(|&i| {
                let (at, host, n) = pieces[i];
                // SAFETY: inside the copy's memory, which the GPU is done
                // with; only this thread writes it.
                let host = unsafe { std::slice::from_raw_parts(host as *const u8, n) };
                match was.at(at, n) {
                    Some(Before::Bytes(before)) => host != before,
                    // Blocks no longer armed tell what they held below.
                    Some(Before::Armed(_)) => {
                        crate::watch::in_place((start + at) / PAGE, &mut seen, || {
                            // SAFETY: an armed block, mapped while it is read.
                            host != unsafe {
                                std::slice::from_raw_parts((start + at) as *const u8, n)
                            }
                        })
                        .unwrap_or(true)
                    }
                    None => false,
                }
            })
            .collect::<Vec<_>>()
        })
        .concat();
        let mut title = vec![0u8; PIECE];
        let (mut written, mut synced) = (0, Vec::new());
        for i in changed {
            let (at, host, n) = pieces[i];
            let saved;
            let old: &[u8] = match was.at(at, n) {
                Some(Before::Bytes(before)) => before,
                Some(Before::Armed(asked)) => {
                    let block = (start + at) / PAGE;
                    let Some(bytes) = crate::watch::before(block, asked) else {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::cost(
                            crate::copystat::Op::Missed,
                            std::time::Instant::now(),
                            start + at,
                            n as u64,
                        );
                        continue;
                    };
                    saved = bytes;
                    let first = start + at - block * PAGE;
                    &saved[first..first + n]
                }
                None => continue,
            };
            // SAFETY: as above.
            let host = unsafe { std::slice::from_raw_parts_mut(host as *mut u8, n) };
            let title = &mut title[..n];
            if !crate::host::read_into((start + at) as *const u8, title) {
                continue;
            }
            // Title memory with the GPU's bytes put in; the copy takes the
            // title's bytes in turn.
            let mut changed = false;
            for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                if *h != *o && *t != *h {
                    *t = *h;
                    changed = true;
                }
                *h = *t;
            }
            if changed {
                #[cfg(feature = "diagnostics")]
                let started = std::time::Instant::now();
                crate::watch::writing(start + at, n);
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written += n;
                #[cfg(feature = "diagnostics")]
                crate::copystat::cost(
                    crate::copystat::Op::WriteBack,
                    started,
                    start + at,
                    n as u64,
                );
            }
            synced.push((start + at) / PAGE);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(
            crate::copystat::Op::Compare,
            started,
            start + from,
            (to - from) as u64,
        );
        // Pages made equal to title memory are current in the copy, so our
        // own write is not taken for the title's; others keep what they had
        // (a title write there shows at the next refresh).
        let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
        if first < last
            && !synced.is_empty()
            && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
        {
            let base = start / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                copied[page - base] = now;
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.in_place {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        // Ranges the submission's shaders write ([`Shadow::writing`]) are
        // brought up to date as well: the submission has not run, so the
        // title's bytes are what its shaders start from, and the snapshot
        // takes them too ([`Shadow::copy`]), so they do not count as the
        // shaders' writes. Leaving those ranges out left every read after
        // the first dispatch writing the world's 174 MB heap with bytes of
        // earlier frames: indirect arguments the title had cleared read as
        // millions of groups.
        if self.mirror() {
            return self.update_mirror(from, to);
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer).
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some(copied) = self.update_watched(from, to) {
                self.watched.store(1, Ordering::Relaxed);
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy_whole(from, to)
    }

    /// Copies the pages of `[from, to)` the watch says changed since they
    /// were copied; `None` if it cannot tell for the range.
    fn update_watched(&self, from: usize, to: usize) -> Option<usize> {
        let (now, written) = crate::watch::writes(from, to, true)?;
        let copied = self.copy_stale(from, to, now, &written);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        Some(copied)
    }

    /// [`Shadow::update`] in a mirror slab, which spans title memory of
    /// every kind (write-watched allocations, runs of views, free
    /// addresses) while the watch answers for one allocation or one run of
    /// views at a time: asked as a whole, then by window, then by the
    /// regions the system reports in a window it cannot answer for, where
    /// committed regions it cannot watch are copied whole and others are
    /// left as they are (zeros, unless the title freed memory copied
    /// there). Nothing is ever taken as unwatched for the whole slab.
    fn update_mirror(&self, from: usize, to: usize) -> usize {
        if let Some(copied) = self.update_watched(from, to) {
            return copied;
        }
        #[cfg(feature = "diagnostics")]
        ROOTS[Root::Split as usize].fetch_add(1, Ordering::Relaxed);
        let w = WINDOW as usize;
        let mut copied = 0;
        let mut at = from;
        while at < to {
            let next = ((at / w + 1) * w).min(to);
            if let Some(n) = self.update_watched(at, next) {
                copied += n;
                at = next;
                continue;
            }
            let mut a = at;
            while a < next {
                let (end, committed) = region(a);
                let end = end.clamp(a + 1, next);
                if committed {
                    copied += self
                        .update_watched(a, end)
                        .unwrap_or_else(|| self.copy_whole(a, end));
                }
                a = end;
            }
            at = next;
        }
        copied
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(crate::copystat::Op::Copy, started, from, bytes as u64);
        bytes
    }

    /// [`Shadow::copy`] of memory nothing can tell the changes of.
    fn copy_whole(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let n = self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(crate::copystat::Op::Whole, started, from, n as u64);
        n
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy
    /// (giving mirror windows host memory), and into the snapshots of
    /// ranges shaders write where they have them.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        for (at, host, n) in self.pieces(from - start, to - start, true) {
            if !host.is_null() {
                crate::host::copy_mapped(host, (start + at) as *const u8, n, false);
            }
        }
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        for (taken, bytes) in before.iter_mut() {
            let (a, b) = ((from - start).max(*taken), (to - start).min(*taken + bytes.len()));
            if a < b {
                self.fill(a, &mut bytes[a - *taken..b - *taken]);
            }
        }
        // Armed blocks refreshed since: the copy no longer holds what title
        // memory does when the title writes them next, so they are kept.
        let armed = self.armed.lock().unwrap_or_else(|e| e.into_inner());
        if !armed.is_empty() {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            for block in from / PAGE..to.div_ceil(PAGE) {
                let (a, b) = (
                    (block * PAGE).saturating_sub(start),
                    (block + 1) * PAGE - start,
                );
                if armed.iter().any(|&(x, y, _)| x < b && a < y) {
                    kept.insert(block, self.block_bytes(block));
                }
            }
        }
        to - from
    }
}

/// Where the region of memory holding `at` ends, and whether it is
/// committed (as the system reports it).
fn region(at: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if crate::watch::query(at, &mut info) == 0 {
        return (usize::MAX, false);
    }
    (
        info.BaseAddress as usize + info.RegionSize,
        info.State == MEM_COMMIT,
    )
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use; else of the loose memory there.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    match buffer(address, end) {
        Some(copy) => copy,
        None => loose(address, end),
    }
}

/// The copy of the innermost title buffer holding `[address, end)`, made
/// on first use (`Some(None)` if it cannot be); `None` if no title buffer
/// holds the range.
fn buffer(address: u64, end: u64) -> Option<Option<Arc<Shadow>>> {
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)?;
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none()
        && let Some((buffer, start, len)) = in_place(key.0, entry.end - key.0)
    {
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        crate::host::name(&buffer, || format!("title memory in place {start:#x}+{len:#x}"));
        if len >= 16 << 20 {
            tracing::info!(start = format_args!("{start:#x}"), len, "large buffer used in place");
        } else {
            tracing::debug!(start = format_args!("{start:#x}"), len, "buffer used in place");
        }
        let memory = Memory::Linear(start as *mut u8);
        #[cfg(feature = "diagnostics")]
        made(gpu, len as u64, start as u64, len as u64);
        let shadow = Shadow::new(buffer, memory, gpu, start as u64, len as u64, true);
        entry.shadow = Some(Arc::new(shadow));
    }
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let Some((buffer, data)) =
            host().and_then(|h| h.shared_buffer(len.next_multiple_of(1 << 16)))
        else {
            return Some(None);
        };
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        #[cfg(feature = "diagnostics")]
        made(gpu, len.next_multiple_of(1 << 16), key.0, len);
        let shadow = Shadow::new(buffer, Memory::Linear(data), gpu, key.0, len, false);
        entry.shadow = Some(Arc::new(shadow));
    }
    Some(entry.shadow.clone())
}

/// Diagnostics: the host range (start, bytes) and title range (start,
/// bytes) of every title buffer's host copy made, newest last, for
/// device-removed reports to name a copy of a buffer the title has
/// released ([`near`]).
#[cfg(feature = "diagnostics")]
static MADE: Mutex<Vec<[u64; 4]>> = Mutex::new(Vec::new());

/// Diagnostics: a title buffer's host copy was made (see [`MADE`]).
#[cfg(feature = "diagnostics")]
fn made(gpu: u64, bytes: u64, start: u64, len: u64) {
    MADE.lock()
        .unwrap_or_else(|e| e.into_inner())
        .push([gpu, bytes, start, len]);
}

/// The host copy a root view at `address` reads, and how many of the
/// `reach` bytes past it (root views have no size; the caller says how far
/// shaders may read) that copy holds. The copy of the innermost title
/// buffer holding them all; else loose memory's (the mirror, which holds
/// title memory as the console has it, contiguous over loose memory and
/// buffers alike), unless the view starts in a title buffer whose copy
/// shaders write: that copy then, cut at its end, since a second copy of
/// its bytes would miss what shaders of the same submission write there.
pub(crate) fn root(address: u64, reach: u64) -> Option<(Arc<Shadow>, u64)> {
    let end = address.checked_add(reach)?;
    let shadow = match buffer(address, end) {
        Some(copy) => copy?,
        None => match written_at(address) {
            Some(written) => written,
            None => loose(address, end)?,
        },
    };
    let held = shadow.left(address).min(reach);
    Some((shadow, held))
}

/// The copy of the innermost title buffer holding `address` whose copy
/// shaders write, if any.
fn written_at(address: u64) -> Option<Arc<Shadow>> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .filter(|(_, e)| e.end > address)
        .find_map(|(_, e)| e.shadow.clone().filter(|s| s.shaders_write()))
}

/// The smallest title buffer used in place: below it a copy is cheap, and
/// each heap opened locks its pages.
pub(crate) const IN_PLACE_MIN: u64 = 1 << 20;

/// A host buffer that is title memory `[start, start + len)` itself
/// (whole 64 KiB pages around it): the buffer, its first address and its
/// length. `None` where the memory is not all committed private
/// read-write memory (physical-page views, reservations) or the host cannot
/// open it as a heap; such buffers are copied.
///
/// shortcut: memory the title frees while a submission still holds the
/// buffer stays locked to it, and title buffers keep their heaps until
/// released; loose memory ([`loose`]) is always copied (its windows are
/// never released).
fn in_place(start: u64, len: u64) -> Option<(ID3D12Resource, usize, usize)> {
    use std::sync::OnceLock;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE,
    };
    static OFF: OnceLock<bool> = OnceLock::new();
    if *OFF.get_or_init(|| std::env::var_os("VICTORY_COPY_TITLE_MEMORY").is_some()) || len < IN_PLACE_MIN {
        return None;
    }
    let from = start as usize & !(PAGE - 1);
    let to = usize::try_from(start + len).ok()?.next_multiple_of(PAGE);
    let mut at = from;
    while at < to {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if crate::watch::query(at, &mut info) == 0
            || info.State != MEM_COMMIT
            || info.Type != MEM_PRIVATE && info.Type != MEM_MAPPED
            || info.Protect & 0xff != PAGE_READWRITE
        {
            if len >= 16 << 20 {
                tracing::info!(
                    at = format_args!("{at:#x}"),
                    state = format_args!("{:#x}", info.State),
                    kind = format_args!("{:#x}", info.Type),
                    protect = format_args!("{:#x}", info.Protect),
                    "a large title buffer is copied: its memory is not all committed read-write"
                );
            }
            return None;
        }
        at = info.BaseAddress as usize + info.RegionSize;
    }
    let host = host()?;
    static DEVICE: OnceLock<Option<ID3D12Device13>> = OnceLock::new();
    let device = DEVICE.get_or_init(|| match host.device.cast::<ID3D12Device13>() {
        Ok(d) => Some(d),
        Err(e) => {
            tracing::warn!(error = %e, "title buffers are copied: the host cannot open heaps on title memory");
            None
        }
    });
    let device = device.as_ref()?;
    // SAFETY: committed read-write memory of ours for the heap's life (the
    // title's buffer).
    let heap: ID3D12Heap = match unsafe { device.OpenExistingHeapFromAddress1(from as *const _, to - from) } {
        Ok(heap) => heap,
        Err(e) => {
            if len >= 16 << 20 {
                // How many host allocations (views, placeholders split) the
                // range spans: a heap wraps one.
                let (mut at, mut pieces, mut first) = (from, Vec::new(), 0);
                while at < to {
                    // SAFETY: plain data filled in by the query.
                    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                    if crate::watch::query(at, &mut info) == 0 {
                        break;
                    }
                    let base = info.AllocationBase as usize;
                    if base != first {
                        first = base;
                        pieces.push((base, info.Type, 0usize));
                    }
                    if let Some(p) = pieces.last_mut() {
                        p.2 += (info.BaseAddress as usize + info.RegionSize).min(to) - at;
                    }
                    at = info.BaseAddress as usize + info.RegionSize;
                }
                let sizes: Vec<String> = pieces.iter().take(8).map(|(b, t, n)| format!("{b:#x}:{t:#x}:{n:#x}")).collect();
                tracing::warn!(error = %e, from = format_args!("{from:#x}"), len = to - from, allocations = pieces.len(), first = ?sizes, "a large title buffer is copied: no heap on its memory");
            }
            return None;
        }
    };
    // SAFETY: a plain query of a live heap.
    let shared = unsafe { heap.GetDesc() }.Flags.contains(D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (to - from) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: match shared {
            true => D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
            false => D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        },
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer the size of the heap at its start.
    let made = unsafe {
        host.device
            .CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
    };
    if let Err(e) = made {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(error = %e, len = to - from, "a title buffer is copied: no buffer in its heap"));
        return None;
    }
    Some((buffer?, from, to - from))
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices and skinning data, from its own
/// allocator). The mirror ([`mirror`]) holds it in windows of this size;
/// without the mirror, or for a range too long for a slab, such memory
/// gets copies of the aligned windows it touches.
/// shortcut: neither windows nor their copies are ever freed; key them to
/// the title's allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length), where the
/// mirror does not hold it.
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if let Some(slab) = mirror(address, end) {
        return Some(slab);
    }
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, Memory::Linear(data), gpu, start, len, false);
    shadow.loose = true;
    let shadow = Arc::new(shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Title addresses per slab of the mirror.
const SLAB: u64 = 1 << 30;
/// How far past its own title addresses a slab also shows title memory
/// (the next slab's first windows, the same host memory), so that a range
/// starting in a slab and at most this long is contiguous in it.
const OVERLAP: u64 = 64 << 20;
/// Host memory per pool heap of the mirror (16 windows).
const POOL_HEAP: u64 = 64 << 20;
/// D3D12's tile.
const TILE: u64 = 64 << 10;

/// The mirror of loose memory: its slabs (reserved buffers), the host
/// memory of the windows copied so far, and the tile mappings still to be
/// made. Each window copied has host memory in a pool heap, mapped into
/// every slab that shows it; the rest of a slab maps nothing (reads as
/// zeros, writes are dropped).
struct Mirror {
    /// By slab number.
    slabs: BTreeMap<u64, Arc<Shadow>>,
    /// Window number → pool heap and slot in it.
    windows: BTreeMap<u64, (usize, u32)>,
    /// Pool heaps, each with a buffer over it mapped for the CPU (its
    /// address).
    heaps: Vec<(ID3D12Heap, ID3D12Resource, usize)>,
    /// Slots used in the last heap.
    used: u32,
    /// (slab, window) pairs whose tiles are not mapped yet.
    unmapped: Vec<(u64, u64)>,
}

// SAFETY: D3D12 objects are free-threaded; the pool memory is written only
// on the submitting thread.
unsafe impl Send for Mirror {}

static MIRROR: Mutex<Mirror> = Mutex::new(Mirror {
    slabs: BTreeMap::new(),
    windows: BTreeMap::new(),
    heaps: Vec::new(),
    used: 0,
    unmapped: Vec::new(),
});

fn mirror_state() -> std::sync::MutexGuard<'static, Mirror> {
    MIRROR.lock().unwrap_or_else(|e| e.into_inner())
}

/// A slab or pool heap could not be made: loose memory goes to windows
/// from then on.
static MIRROR_FAILED: AtomicBool = AtomicBool::new(false);

/// Whether loose memory goes to the mirror: the host maps reserved
/// resources' tiles with unmapped ones reading zeros (tier 2), and
/// `VICTORY_LOOSE_WINDOWS` does not turn it off.
fn mirror_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        if std::env::var_os("VICTORY_LOOSE_WINDOWS").is_some() {
            tracing::info!("loose memory copied in windows (VICTORY_LOOSE_WINDOWS)");
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: the query writes the structure of the given size.
        let tier = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .map_or(0, |()| options.TiledResourcesTier.0);
        if tier < D3D12_TILED_RESOURCES_TIER_2.0 {
            tracing::warn!(
                tier,
                "loose memory copied in windows: the host has no tiled resources for its mirror"
            );
        }
        tier >= D3D12_TILED_RESOURCES_TIER_2.0
    });
    on && !MIRROR_FAILED.load(Ordering::Relaxed)
}

/// The slab that holds title memory `[address, end)`, if one can.
fn slab_of(address: u64, end: u64) -> Option<u64> {
    let k = address / SLAB;
    (end <= (k + 1) * SLAB + OVERLAP).then_some(k)
}

/// The slabs that show window `w`: its own and, through its overlap, the
/// one before.
fn slabs_showing(w: u64) -> impl Iterator<Item = u64> {
    let at = w * WINDOW;
    let k = at / SLAB;
    std::iter::once(k).chain((k > 0 && at - k * SLAB < OVERLAP).then(|| k - 1))
}

/// A buffer description of `width` bytes shaders may write.
fn buffer_desc(width: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
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
    }
}

impl Mirror {
    /// The CPU address of window `w`'s host memory, which `make` gives it
    /// first if it has none; null if it has none.
    fn window(&mut self, w: u64, make: bool) -> *mut u8 {
        let slot = match self.windows.get(&w) {
            Some(&slot) => Some(slot),
            None if make => self.place(w),
            None => None,
        };
        slot.map_or(std::ptr::null_mut(), |(heap, slot)| {
            (self.heaps[heap].2 + slot as usize * WINDOW as usize) as *mut u8
        })
    }

    /// Gives window `w` host memory, to be mapped into the slabs that show
    /// it ([`map_windows`]).
    fn place(&mut self, w: u64) -> Option<(usize, u32)> {
        const SLOTS: u32 = (POOL_HEAP / WINDOW) as u32;
        if self.heaps.is_empty() || self.used == SLOTS {
            let Some(heap) = pool_heap() else {
                MIRROR_FAILED.store(true, Ordering::Relaxed);
                return None;
            };
            self.heaps.push(heap);
            self.used = 0;
        }
        let slot = (self.heaps.len() - 1, self.used);
        self.used += 1;
        self.windows.insert(w, slot);
        let shown: Vec<u64> = slabs_showing(w)
            .filter(|k| self.slabs.contains_key(k))
            .collect();
        self.unmapped.extend(shown.into_iter().map(|k| (k, w)));
        Some(slot)
    }
}

/// A pool heap of the mirror in system memory, with a buffer over it
/// mapped for the CPU (write-back: the copies read it too).
fn pool_heap() -> Option<(ID3D12Heap, ID3D12Resource, usize)> {
    let host = host()?;
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_HEAP,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let (mut heap, mut buffer): (Option<ID3D12Heap>, Option<ID3D12Resource>) = (None, None);
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation of a heap, a buffer over all of it, and its
    // mapping.
    let made = unsafe {
        host.device.CreateHeap(&desc, &mut heap).and_then(|()| {
            let h = heap.as_ref().ok_or_else(windows::core::Error::empty)?;
            let d = buffer_desc(POOL_HEAP);
            host.device.CreatePlacedResource(
                h,
                0,
                &d,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )?;
            let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
            b.Map(0, None, Some(&mut data))
        })
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, "loose memory goes to windows: no host memory for the mirror");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || "title memory mirror pool".into());
    Some((heap?, buffer, data as usize))
}

/// The mirror's slab holding loose memory `[address, end)`, made on first
/// use; `None` without the mirror or for a range too long for a slab.
fn mirror(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if !mirror_on() {
        return None;
    }
    let k = slab_of(address, end)?;
    let mut m = mirror_state();
    if let Some(slab) = m.slabs.get(&k) {
        return Some(slab.clone());
    }
    let host = host()?;
    let (start, len) = (k * SLAB, SLAB + OVERLAP);
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a reserved buffer.
    let made = unsafe {
        host.device.CreateReservedResource(
            &buffer_desc(len),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    let buffer = match (made, buffer) {
        (Ok(()), Some(buffer)) => buffer,
        (made, _) => {
            MIRROR_FAILED.store(true, Ordering::Relaxed);
            tracing::warn!(?made, "loose memory goes to windows: no mirror slab");
            return None;
        }
    };
    crate::host::name(&buffer, || {
        format!("title memory mirror {start:#x}+{len:#x}")
    });
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut slab = Shadow::new(buffer, Memory::Mirror, gpu, start, len, false);
    slab.loose = true;
    let slab = Arc::new(slab);
    let shown: Vec<u64> = m
        .windows
        .range(start / WINDOW..(start + len) / WINDOW)
        .map(|(&w, _)| w)
        .collect();
    m.unmapped.extend(shown.into_iter().map(|w| (k, w)));
    m.slabs.insert(k, slab.clone());
    tracing::info!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory mirrored"
    );
    Some(slab)
}

/// Before a submission runs: maps the tiles of the windows given host
/// memory since the last call into the slabs that show them (until then the
/// GPU reads zeros there).
pub(crate) fn map_windows(queue: &ID3D12CommandQueue) {
    let mut m = mirror_state();
    if m.unmapped.is_empty() {
        return;
    }
    let mut pairs = std::mem::take(&mut m.unmapped);
    pairs.sort_unstable();
    pairs.dedup();
    let tiles = (WINDOW / TILE) as u32;
    // One call per slab and pool heap: the windows' first tiles in each.
    type Call = (Vec<D3D12_TILED_RESOURCE_COORDINATE>, Vec<u32>);
    let mut calls: BTreeMap<(u64, usize), Call> = BTreeMap::new();
    for (k, w) in pairs {
        let Some(&(heap, slot)) = m.windows.get(&w) else {
            continue;
        };
        let call = calls.entry((k, heap)).or_default();
        call.0.push(D3D12_TILED_RESOURCE_COORDINATE {
            X: ((w * WINDOW - k * SLAB) / TILE) as u32,
            ..Default::default()
        });
        call.1.push(slot * tiles);
    }
    for ((k, heap), (at, from)) in calls {
        let n = at.len();
        let sizes = vec![
            D3D12_TILE_REGION_SIZE {
                NumTiles: tiles,
                ..Default::default()
            };
            n
        ];
        let (counts, flags) = (vec![tiles; n], vec![D3D12_TILE_RANGE_FLAG_NONE; n]);
        // SAFETY: a reserved buffer and a pool heap of ours; the regions lie
        // in the buffer, the ranges in the heap.
        unsafe {
            queue.UpdateTileMappings(
                &m.slabs[&k].buffer,
                n as u32,
                Some(at.as_ptr()),
                Some(sizes.as_ptr()),
                &m.heaps[heap].0,
                n as u32,
                Some(flags.as_ptr()),
                Some(from.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            )
        };
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let m = mirror_state();
    all.extend(m.slabs.values().map(|s| (s.gpu, s.start, s.len)));
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    // Diagnostics: outside every copy held now, the newest copy made there
    // is one of a buffer the title has released.
    #[cfg(feature = "diagnostics")]
    if !below.is_some_and(|&(gpu, _, len)| va < gpu + len)
        && let Some(&[gpu, bytes, start, len]) = MADE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .rev()
            .find(|c| (c[0]..c[0] + c[1]).contains(&va))
    {
        text.push_str(&format!(
            "; inside the host copy made for title {start:#x}+{len:#x} at host {gpu:#x}+{bytes:#x}, offset {:#x}, a buffer the title has released",
            va - gpu
        ));
    }
    if let Some(slab) = m
        .slabs
        .values()
        .find(|s| (s.gpu..s.gpu + s.len).contains(&va))
    {
        let title = slab.start + (va - slab.gpu);
        let copied = m.windows.contains_key(&(title / WINDOW));
        text.push_str(&format!(
            " (the loose-memory mirror: title {title:#x}, its window copied: {copied})"
        ));
    }
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Diagnostics: root views bound since the last frame line ([`root`]), by
/// the copy they read; the bytes past them those copies hold; those cut
/// short; and mirror refreshes the watch could not answer for at once.
#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy)]
enum Root {
    Bound,
    Buffer,
    Mirror,
    Window,
    Bytes,
    Short,
    Split,
}

#[cfg(feature = "diagnostics")]
static ROOTS: [std::sync::atomic::AtomicU64; 7] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Counts a root view ([`root`]) and warns, once per address, about one
/// whose copy ends before its reach, or whose reach in the copy of loose
/// memory holds a title buffer shaders write.
#[cfg(feature = "diagnostics")]
pub(crate) fn note_root(address: u64, reach: u64, shadow: &Shadow, held: u64) {
    let add = |r: Root, n: u64| ROOTS[r as usize].fetch_add(n, Ordering::Relaxed);
    add(Root::Bound, 1);
    add(Root::Bytes, held);
    add(
        match (shadow.mirror(), shadow.loose) {
            (true, _) => Root::Mirror,
            (false, true) => Root::Window,
            (false, false) => Root::Buffer,
        },
        1,
    );
    if held < reach {
        add(Root::Short, 1);
        if first_warning(address) {
            tracing::warn!(
                address = format_args!("{address:#x}"),
                reach,
                held,
                copy = format_args!("{:#x}+{:#x}", shadow.start, shadow.len),
                shaders_write = shadow.shaders_write(),
                "a root view's host copy ends before its reach: past it the view reads other host memory"
            );
        }
    } else if shadow.loose
        && let Some(buffer) = written_in(address, address + reach)
        && first_warning(address)
    {
        tracing::warn!(
            address = format_args!("{address:#x}"),
            buffer = format_args!("{buffer:#x}"),
            "a root view in the copy of loose memory reaches a title buffer shaders write: it reads that buffer as the submission found it"
        );
    }
}

/// Where a title buffer whose copy shaders write starts in `[address,
/// end)`, if one does (diagnostics).
#[cfg(feature = "diagnostics")]
fn written_in(address: u64, end: u64) -> Option<u64> {
    buffers()
        .range(..(end, 0))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address && e.shadow.as_ref().is_some_and(|s| s.shaders_write()))
        .map(|(k, _)| k.0)
}

/// Whether `address` is warned about for the first time (of at most 4,096).
#[cfg(feature = "diagnostics")]
fn first_warning(address: u64) -> bool {
    static WARNED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    let warned = warned.get_or_insert_with(Default::default);
    warned.len() < 4096 && warned.insert(address)
}

/// Logs the root views per frame over `frames` frames, and the mirror's
/// size (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_roots(frames: f64) {
    let r = ROOTS
        .each_ref()
        .map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    let (slabs, windows) = {
        let m = mirror_state();
        (m.slabs.len(), m.windows.len())
    };
    tracing::info!(
        "root views per frame: {:.0} bound ({:.0} in title buffer copies, {:.0} in the loose-memory mirror, \
         {:.0} in loose windows), their copies holding {:.1} MB past them, {:.1} cut short; mirror: {slabs} \
         slabs, {} MiB of windows copied, {:.1} refreshes split by memory kind",
        r[Root::Bound as usize],
        r[Root::Buffer as usize],
        r[Root::Mirror as usize],
        r[Root::Window as usize],
        r[Root::Bytes as usize] / 1e6,
        r[Root::Short as usize],
        windows as u64 * (WINDOW >> 20),
        r[Root::Split as usize],
    );
}

/// The host end of the copy holding host GPU address `va` (tests).
#[cfg(test)]
pub(crate) fn host_end(va: u64) -> Option<u64> {
    let mut all: Vec<(u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.gpu + s.len)))
        .collect();
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    all.extend(loose.values().map(|s| (s.gpu, s.gpu + s.len)));
    all.extend(
        mirror_state()
            .slabs
            .values()
            .map(|s| (s.gpu, s.gpu + s.len)),
    );
    all.into_iter()
        .filter(|(a, b)| (*a..*b).contains(&va))
        .map(|(_, b)| b)
        .max()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    const MIB: usize = 1 << 20;

    /// Whether the host GPU and the mirror are here (tests that need them
    /// pass without them, saying so).
    pub(crate) fn gpu() -> bool {
        let here = host().is_some() && mirror_on();
        if !here {
            eprintln!("no host GPU with the loose-memory mirror: skipped");
        }
        here
    }

    /// The word test title memory holds at `at`.
    pub(crate) fn word(at: usize) -> u32 {
        (at as u32 / 4).wrapping_mul(0x9e37_79b1) | 1
    }

    /// Write-watched title memory as the runtime reserves it: `len` bytes
    /// reserved at a free address `below` bytes under a slab boundary
    /// (tests pass their own `slab` so that none shares addresses with
    /// another, whose pages the watch's journal would hold), the first
    /// `commit` committed and filled with [`word`]s. 0 if no such address
    /// is free.
    pub(crate) fn title_memory(slab: u64, below: usize, len: usize, commit: usize) -> usize {
        for k in (slab..slab + 8).rev() {
            let at = (k * SLAB) as usize - below;
            // SAFETY: a reservation at a free address, or none.
            let p = unsafe {
                VirtualAlloc(
                    at as *const _,
                    len,
                    MEM_RESERVE | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            if p != at {
                if p != 0 {
                    // SAFETY: the reservation just made.
                    unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
                }
                continue;
            }
            // SAFETY: committing part of our reservation.
            let committed =
                unsafe { VirtualAlloc(p as *const _, commit, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, p);
            for at in (p..p + commit).step_by(4) {
                // SAFETY: committed memory of ours.
                unsafe { (at as *mut u32).write(word(at)) };
            }
            return p;
        }
        0
    }

    /// A device-removed report names the copy of a released title buffer
    /// that holds the faulting address (diagnostics).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn faults_in_copies_of_released_buffers_are_named() {
        if host().is_none() {
            return;
        }
        let memory = vec![0u8; 3 << 16];
        let start = (memory.as_ptr() as u64).next_multiple_of(1 << 16);
        register(start as usize, start, 0x1_0000);
        let gpu = find(start, 16).expect("a copy").gpu;
        unregister(start as usize, start);
        let text = near(gpu + 0x40);
        assert!(
            text.contains(&format!(
                "inside the host copy made for title {start:#x}+0x10000"
            )),
            "{text}"
        );
    }

    pub(crate) fn release(p: usize) {
        // SAFETY: a reservation [`title_memory`] made.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Views of fresh pagefile sections, one after another from `at`,
    /// filled with [`word`]s; empty if the addresses are not free.
    fn views(at: usize, lens: &[usize]) -> Vec<usize> {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, MapViewOfFile3};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut made = Vec::new();
        let mut next = at;
        for &len in lens {
            // SAFETY: a section of ours mapped at a free address, or none.
            let v = unsafe {
                let section = CreateFileMappingW(
                    -1isize as _,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    len as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    next as *const _,
                    0,
                    len,
                    0,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
                .Value as usize
            };
            if v != next {
                unmap(&made);
                return Vec::new();
            }
            for a in (v..v + len).step_by(4) {
                // SAFETY: our view.
                unsafe { (a as *mut u32).write(word(a)) };
            }
            made.push(v);
            next += len;
        }
        made
    }

    fn unmap(views: &[usize]) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &v in views {
            // SAFETY: a view [`views`] mapped.
            crate::watch::title_memory_unmapping(v, 64 * MIB, || unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ })
            });
        }
    }

    /// A buffer's copy keeps a snapshot per region shaders write, not one
    /// over everything between them, and copies back what the GPU wrote in
    /// overlapping declared ranges once (a range of 20 MiB on several
    /// threads).
    #[test]
    fn a_buffer_copy_keeps_a_snapshot_per_region_shaders_write() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        // Write-watched memory (view memory is armed instead), its last page
        // not committed: the buffer is copied.
        let len = (48 << 20) + (64 << 10);
        let at = title_memory(500, 256 * MIB, len, 48 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 7, at as u64, len as u64);
        let s = find(at as u64, len as u64).unwrap();
        assert!(!s.in_place);
        let declared = [
            (at + MIB, 64 << 10),
            (at + MIB + (32 << 10), 64 << 10),
            (at + 20 * MIB, 20 * MIB),
        ];
        crate::watch::forget_asked();
        for (a, n) in declared {
            s.update(a, n);
            s.writing(a, n);
        }
        let spans: Vec<(usize, usize)> = {
            let mut spans: Vec<_> = s
                .before
                .lock()
                .unwrap()
                .iter()
                .map(|(a, b)| (a + at, b.len()))
                .collect();
            spans.sort_unstable();
            spans
        };
        assert_eq!(spans, [(at + MIB, 96 << 10), (at + 20 * MIB, 20 * MIB)]);
        // The GPU wrote a word where two declared ranges overlap, and one
        // in the big range.
        let written_at = [at + MIB + (40 << 10), at + 30 * MIB + 12];
        for w in written_at {
            // SAFETY: inside the copy's memory.
            unsafe { s.host_at(w - at, 4).cast::<u32>().write(0xfeed) };
        }
        assert_eq!(s.copy_back(&declared), 2 * (64 << 10));
        for w in written_at {
            // SAFETY: our test memory.
            assert_eq!(unsafe { (w as *const u32).read() }, 0xfeed);
            // SAFETY: as above.
            assert_eq!(unsafe { ((w + 4) as *const u32).read() }, word(w + 4));
        }
        assert!(s.before.lock().unwrap().is_empty());
        drop(s);
        unregister(usize::MAX - 7, at as u64);
        release(at);
    }

    /// A copy of view memory that shaders write takes no snapshot: its
    /// title blocks are armed, and the title's first write to one during
    /// the submission saves the block first. Copy-back writes what the GPU
    /// wrote and leaves the title's own writes: in blocks written during
    /// the submission, and in one written after the refresh but before
    /// arming (kept as the copy held it).
    #[test]
    fn a_copy_of_view_memory_arms_its_blocks_instead_of_a_snapshot() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = (962 * SLAB) as usize - 256 * MIB;
        let made = views(at, &[24 * MIB, 24 * MIB]);
        assert_eq!(made.len(), 2, "no free address for test views");
        register(usize::MAX - 8, at as u64, 48 << 20);
        let s = find(at as u64, 48 << 20).unwrap();
        assert!(!s.in_place);
        let declared = [(at + MIB, 32 * MIB)];
        crate::watch::forget_asked();
        s.update(at + MIB, 32 * MIB);
        let early = at + 2 * MIB + 100;
        // SAFETY: our views (guarded: our handler opens the block).
        unsafe { (early as *mut u32).write(0xe) };
        s.writing(at + MIB, 32 * MIB);
        assert!(s.before.lock().unwrap().is_empty(), "no snapshot");
        let kept: Vec<usize> = s.kept.lock().unwrap().keys().copied().collect();
        assert_eq!(kept, [early / PAGE]);
        // The GPU writes a word in a block the title leaves alone, one in a
        // block the title writes elsewhere, one where the title writes the
        // same word (the GPU's wins), and one in the block written early.
        let gpu = [at + 4 * MIB, at + 8 * MIB + 64, at + 12 * MIB, early + 8];
        for w in gpu {
            // SAFETY: inside the copy's memory.
            unsafe { s.host_at(w - at, 4).cast::<u32>().write(0xfeed) };
        }
        let title = [at + 8 * MIB + 128, at + 12 * MIB, at + 20 * MIB];
        for w in title {
            // SAFETY: as above.
            unsafe { (w as *mut u32).write(7) };
        }
        assert_eq!(s.copy_back(&declared), 4 * (64 << 10));
        crate::watch::disarm();
        // SAFETY: our views.
        let read = |w: usize| unsafe { (w as *const u32).read() };
        for w in gpu {
            assert_eq!(read(w), 0xfeed);
        }
        assert_eq!(read(at + 8 * MIB + 128), 7);
        assert_eq!(read(at + 20 * MIB), 7);
        assert_eq!(read(early), 0xe);
        assert_eq!(read(at + 4 * MIB + 4), word(at + 4 * MIB + 4));
        // Where compared, the copy holds what title memory does.
        assert_eq!(held(&s, at + 8 * MIB + 128), Some(7));
        assert!(s.kept.lock().unwrap().is_empty() && s.armed.lock().unwrap().is_empty());
        drop(s);
        unregister(usize::MAX - 8, at as u64);
        unmap(&made);
    }

    /// Title memory shown at two addresses (xmem maps physical pages more
    /// than once, as the console does): the title's write through the other
    /// address does not fault where a copy armed its blocks, and copy-back
    /// keeps it rather than take it for the GPU's.
    #[test]
    fn a_write_through_another_address_survives_copy_back() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            MapViewOfFile3, UnmapViewOfFile,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let (at, len) = ((963 * SLAB) as usize - 256 * MIB, MIB);
        // SAFETY: a section of ours, mapped at a free address (checked) and
        // wherever the system puts it.
        let (x, y) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let x = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at as *const _,
                0,
                len,
                0,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
            .Value as usize;
            let y = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            CloseHandle(section);
            (x, y)
        };
        assert!(x == at && y != 0, "no free address for test views");
        // As xmem tells it: both show the section's pages.
        const PAGES: u64 = 0x7e57 << 32;
        crate::watch::title_memory_shows(x, len, PAGES);
        crate::watch::title_memory_shows(y, len, PAGES);
        for a in (x..x + len).step_by(4) {
            // SAFETY: our view.
            unsafe { (a as *mut u32).write(word(a)) };
        }
        register(usize::MAX - 11, x as u64, (len / 2) as u64);
        let s = find(x as u64, (len / 2) as u64).unwrap();
        assert!(!s.in_place);
        let declared = [(x, len / 2)];
        crate::watch::forget_asked();
        s.update(x, len / 2);
        s.writing(x, len / 2);
        // While the GPU runs, the title writes a word through the other
        // address, and the GPU another of the same block.
        let (title, gpu) = (x + PAGE + 8, x + PAGE + 64);
        // SAFETY: our view of the section, and inside the copy's memory.
        unsafe {
            ((title - x + y) as *mut u32).write(7);
            s.host_at(gpu - x, 4).cast::<u32>().write(0xfeed);
        }
        s.copy_back(&declared);
        crate::watch::disarm();
        // SAFETY: our view.
        let read = |w: usize| unsafe { (w as *const u32).read() };
        assert_eq!(read(title), 7, "the title's write was undone");
        assert_eq!(read(gpu), 0xfeed);
        drop(s);
        unregister(usize::MAX - 11, x as u64);
        unmap(&[x]);
        // SAFETY: our other view.
        crate::watch::title_memory_unmapping(y, len, || unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: y as *mut _ })
        });
    }

    /// Two copies arm the same block in one submission, the second after
    /// the title's write saved it and a refresh guarded it again (a step
    /// that writes title memory lies between them): the second copy took
    /// that write, so the block's saved bytes are not what it held, and its
    /// copy-back must not take the write for the GPU's and put it back over
    /// the title's next one (runs 57, 60 and 64, title+0x4e9b54a).
    #[test]
    fn a_block_saved_before_another_copy_arms_it_keeps_the_titles_later_write() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = (964 * SLAB) as usize - 256 * MIB;
        // Copies across views: not used in place.
        let made = views(at, &[6 * MIB; 4]);
        assert_eq!(made.len(), 4, "no free address for test views");
        register(usize::MAX - 12, at as u64, (8 * MIB) as u64);
        register(usize::MAX - 13, (at + 4 * MIB) as u64, (8 * MIB) as u64);
        let a = find(at as u64, (8 * MIB) as u64).unwrap();
        let b = find((at + 4 * MIB) as u64, (8 * MIB) as u64).unwrap();
        assert!(!Arc::ptr_eq(&a, &b) && !a.in_place && !b.in_place);
        let (da, db) = ([(at, 8 * MIB)], [(at + 4 * MIB, 8 * MIB)]);
        let x = at + 5 * MIB + 100;
        // SAFETY: our view (guarded: our handler opens the block).
        let write = |w: usize, v: u32| unsafe { (w as *mut u32).write(v) };
        crate::watch::forget_asked();
        a.update(at, 8 * MIB);
        a.writing(at, 8 * MIB);
        write(x, 1);
        crate::watch::forget_asked();
        b.update(at + 4 * MIB, 8 * MIB);
        b.writing(at + 4 * MIB, 8 * MIB);
        assert_eq!(held(&b, x), Some(1));
        // While the GPU runs, the title writes the word again, and the GPU
        // writes a word elsewhere in each copy.
        write(x, 2);
        let gpu = [(&a, at + MIB), (&b, at + 10 * MIB)];
        for (s, w) in gpu {
            // SAFETY: inside the copy's memory.
            unsafe {
                s.host_at(w - s.start as usize, 4)
                    .cast::<u32>()
                    .write(0xfeed)
            };
        }
        crate::watch::forget_asked();
        a.copy_back(&da);
        crate::watch::forget_asked();
        b.copy_back(&db);
        crate::watch::disarm();
        // SAFETY: our view.
        let read = |w: usize| unsafe { (w as *const u32).read() };
        assert_eq!(read(x), 2, "the title's write was undone");
        for (_, w) in gpu {
            assert_eq!(read(w), 0xfeed);
        }
        drop((a, b));
        unregister(usize::MAX - 12, at as u64);
        unregister(usize::MAX - 13, (at + 4 * MIB) as u64);
        unmap(&made);
    }

    /// The word a copy holds for title address `at` (0 where it has no
    /// memory).
    fn held(s: &Shadow, at: usize) -> Option<u32> {
        let host = s.host_at(at - s.start as usize, 4);
        // SAFETY: inside the copy's memory.
        (!host.is_null()).then(|| unsafe { host.cast::<u32>().read_unaligned() })
    }

    #[test]
    fn a_range_is_in_the_slab_it_starts_in_while_it_fits() {
        let s = SLAB;
        assert_eq!(slab_of(5 * s + 10, 5 * s + 20), Some(5));
        // Past the slab's end, in its overlap.
        assert_eq!(slab_of(6 * s - 1, 6 * s + OVERLAP), Some(5));
        // Longer than the overlap: a window of its own.
        assert_eq!(slab_of(6 * s - 1, 6 * s + OVERLAP + 1), None);
        assert_eq!(slab_of(6 * s, 6 * s + 1), Some(6));
        // A root view's reach (36 MiB) from anywhere in a slab fits it.
        assert_eq!(slab_of(6 * s - 1, 6 * s - 1 + (36 << 20)), Some(5));
    }

    #[test]
    fn windows_at_a_slab_start_are_shown_by_the_slab_before() {
        let per = SLAB / WINDOW;
        let shows = |w: u64| slabs_showing(w).collect::<Vec<_>>();
        assert_eq!(shows(0), [0]);
        assert_eq!(shows(3 * per), [3, 2]);
        assert_eq!(shows(3 * per + OVERLAP / WINDOW - 1), [3, 2]);
        assert_eq!(shows(3 * per + OVERLAP / WINDOW), [3]);
        assert_eq!(shows(4 * per - 1), [3]);
        // Every window a slab holds, its own and its overlap's, maps into it.
        for w in 3 * per..4 * per + OVERLAP / WINDOW {
            assert!(shows(w).contains(&3), "window {w}");
        }
    }

    /// Title memory across a slab boundary has one host copy: brought up
    /// to date through the slab before (its overlap), the window after the
    /// boundary holds the same bytes, at the same host memory, in the slab
    /// after. Windows never copied have no memory (the GPU reads zeros).
    #[test]
    fn loose_memory_has_one_host_copy_across_slabs() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let p = title_memory(400, 32 * MIB, 64 * MIB, 64 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        let boundary = p + 32 * MIB;
        let a = find((boundary - MIB) as u64, (8 * MIB) as u64).unwrap();
        let b = find((boundary + MIB) as u64, MIB as u64).unwrap();
        assert!(a.mirror() && b.mirror() && !Arc::ptr_eq(&a, &b));
        assert_eq!((a.start + SLAB) as usize, boundary);
        assert_eq!(b.start as usize, boundary);
        crate::watch::forget_asked();
        assert_eq!(a.update(boundary - MIB, 8 * MIB), 8 * MIB);
        let at = boundary + 2 * MIB + 12;
        assert_eq!(
            a.host_at(at - a.start as usize, 4),
            b.host_at(at - b.start as usize, 4)
        );
        assert_eq!(held(&a, at), Some(word(at)));
        assert_eq!(held(&b, at), Some(word(at)));
        assert_eq!(held(&a, boundary - 4), Some(word(boundary - 4)));
        assert_eq!(held(&b, boundary + 20 * MIB), None);
        // A title write shows in both at the next refresh through either.
        let at = boundary + 3 * MIB;
        // SAFETY: committed test memory.
        unsafe { (at as *mut u32).write(7) };
        crate::watch::forget_asked();
        b.update(boundary, 4 * MIB);
        assert_eq!((held(&a, at), held(&b, at)), (Some(7), Some(7)));
        drop((a, b));
        release(p);
    }

    /// A mirror slab keeps a snapshot per region shaders write, not one
    /// over the GiB between them, and copies back what they wrote.
    #[test]
    fn a_mirror_slab_keeps_a_snapshot_per_region_shaders_write() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let p = title_memory(300, 512 * MIB, 128 * MIB, 128 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        let s = find(p as u64, 1).unwrap();
        assert!(s.mirror());
        crate::watch::forget_asked();
        let ranges = [
            (p + MIB, 64 << 10),
            (p + 100 * MIB, 64 << 10),
            (p + MIB + (128 << 10), 64 << 10),
        ];
        for (at, n) in ranges {
            s.update(at, n);
            s.writing(at, n);
        }
        let spans: Vec<(usize, usize)> = {
            let before = s.before.lock().unwrap();
            let mut spans: Vec<_> = before
                .iter()
                .map(|(at, b)| (at + s.start as usize, b.len()))
                .collect();
            spans.sort_unstable();
            spans
        };
        assert_eq!(spans, [(p + MIB, 192 << 10), (p + 100 * MIB, 64 << 10)]);
        // The GPU wrote a word in the far region.
        let at = p + 100 * MIB + 8;
        let host = s.host_at(at - s.start as usize, 4);
        // SAFETY: a placed window of the copy.
        unsafe { host.cast::<u32>().write(0xfeed) };
        let written = s.copy_back(&ranges);
        assert_eq!(written, 64 << 10);
        // SAFETY: committed test memory.
        assert_eq!(unsafe { (at as *const u32).read() }, 0xfeed);
        // SAFETY: as above.
        assert_eq!(unsafe { ((at + 4) as *const u32).read() }, word(at + 4));
        assert!(s.before.lock().unwrap().is_empty());
        drop(s);
        release(p);
    }

    /// A root view reads the title buffer that holds its whole reach; one
    /// reaching past its buffer reads the mirror, all of its reach, unless
    /// shaders write that buffer's copy: then that copy, cut at its end.
    #[test]
    fn root_views_leave_their_buffer_for_the_mirror_unless_shaders_write_it() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let p = title_memory(200, 512 * MIB, 8 * MIB, 8 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        let (buffer, len) = (p + MIB, 512 << 10);
        register(0x7e57, buffer as u64, len as u64);
        let root_at = (buffer + 16) as u64;
        let (s, held) = root(root_at, 64 << 10).unwrap();
        assert!(!s.loose && held == 64 << 10);
        let (s, held) = root(root_at, 36 << 20).unwrap();
        assert!(s.mirror() && held == 36 << 20);
        let b = find(buffer as u64, len as u64).unwrap();
        b.shaders_write.store(true, Ordering::Relaxed);
        let (s, held) = root(root_at, 36 << 20).unwrap();
        assert!(Arc::ptr_eq(&s, &b));
        assert_eq!(held as usize, len - 16);
        unregister(0x7e57, buffer as u64);
        drop((s, b));
        release(p);
    }

    /// Refreshing a range that runs past the title's committed memory
    /// copies what is committed and leaves the rest zero, without taking
    /// the slab as unwatched.
    #[test]
    fn mirror_refreshes_past_committed_memory_leave_zeros() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        // 16 MiB committed of a 64 MiB reservation; free addresses after.
        let p = title_memory(100, 512 * MIB, 64 * MIB, 16 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        let s = find((p + 8 * MIB) as u64, 36 << 20).unwrap();
        crate::watch::forget_asked();
        s.update(p + 8 * MIB, 36 << 20);
        assert_eq!(held(&s, p + 12 * MIB), Some(word(p + 12 * MIB)));
        assert_eq!(held(&s, p + 30 * MIB), Some(0));
        // Past the reservation: memory of no allocation, or another's.
        let past = p + 70 * MIB;
        s.update(past, 4 * MIB);
        assert_ne!(s.watched.load(Ordering::Relaxed), NOT_WATCHED);
        // A title write is still taken page by page.
        let at = p + 9 * MIB;
        // SAFETY: committed test memory.
        unsafe { (at as *mut u32).write(9) };
        crate::watch::forget_asked();
        assert_eq!(s.update(p + 8 * MIB, 36 << 20), 64 << 10);
        assert_eq!(held(&s, at), Some(9));
        drop(s);
        release(p);
    }

    /// What a root view's reach costs per submission: the first copy of
    /// 36 MiB of loose memory, then the watch's check of it with nothing
    /// written and with 16 pages written, for write-watched memory and for
    /// views of physical pages (guarded). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture reach_refresh`.
    #[test]
    #[ignore = "benchmark"]
    fn reach_refresh_costs() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
            UnmapViewOfFile,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let reach = 36 * MIB;
        let measure = |what: &str, p: usize| {
            let s = find(p as u64, reach as u64).unwrap();
            crate::watch::forget_asked();
            let started = std::time::Instant::now();
            let copied = s.update(p, reach);
            let first = started.elapsed();
            const ROUNDS: u32 = 20;
            let mut clean = std::time::Duration::ZERO;
            for _ in 0..ROUNDS {
                crate::watch::forget_asked();
                let started = std::time::Instant::now();
                assert_eq!(s.update(p, reach), 0);
                clean += started.elapsed();
            }
            let mut dirty = std::time::Duration::ZERO;
            for round in 0..ROUNDS {
                for k in 0..16 {
                    // SAFETY: committed test memory.
                    unsafe { ((p + k * 2 * MIB + 64) as *mut u32).write(round) };
                }
                crate::watch::forget_asked();
                let started = std::time::Instant::now();
                s.update(p, reach);
                dirty += started.elapsed();
            }
            println!(
                "{what}: first copy {} MiB in {first:?}; check, nothing written {:?}; 16 pages written {:?}",
                copied >> 20,
                clean / ROUNDS,
                dirty / ROUNDS
            );
        };
        let p = title_memory(800, 512 * MIB, 48 * MIB, 48 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        measure("write-watched memory", p);
        release(p);
        // A view of a pagefile section where the runtime maps physical pages.
        let at = (850 * SLAB) as usize - 512 * MIB;
        // SAFETY: a section of ours mapped at a free address (checked).
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (48 * MIB) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at as *const _,
                0,
                48 * MIB,
                0,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
            .Value as usize
        };
        assert_eq!(view, at, "no free address for the view");
        let _ = FILE_MAP_WRITE;
        for a in (view..view + 48 * MIB).step_by(4) {
            // SAFETY: our view.
            unsafe { (a as *mut u32).write(word(a)) };
        }
        measure("a view (guarded)", view);
        // SAFETY: our view.
        crate::watch::title_memory_unmapping(view, 48 * MIB, || unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        });
    }
}
