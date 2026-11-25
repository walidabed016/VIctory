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
//! `VICTORY_COPY_TITLE_MEMORY=1` turns this off. With VRAM twins, each
//! gets one ([`in_place_twins_on`]).
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

/// Snapshots of a copy: offset and bytes.
type Snapshots = Vec<(usize, Vec<u8>)>;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    /// The copy in VRAM that shaders use instead when copies have twins
    /// ([`twins_on`]); `buffer` stays the CPU's view of it. A mirror
    /// slab's is a reserved buffer whose windows get VRAM as submissions
    /// use them ([`map_windows`]).
    twin: Option<ID3D12Resource>,
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
    /// Ranges last brought up to date wholly guarded and clean, with the
    /// number of the latest change to title memory before
    /// ([`crate::watch::changes`]): unchanged since, they are up to date.
    synced: Mutex<std::collections::HashMap<(usize, usize), u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Shaders have been given the copy to write ([`Shadow::gpu_writes`]).
    shaders_write: AtomicBool,
    /// What follows is kept per submission, by its slot ([`preparing`]):
    /// a submission's copy-back may run while the next is prepared.
    /// Ranges shaders of the submission write, until it is copied back.
    pending: [Mutex<Vec<(usize, usize)>>; SLOTS],
    /// The copy's bytes before the submission ran, by offset, covering the
    /// ranges its shaders write: what the GPU wrote differs. One snapshot
    /// for a buffer's copy; for a mirror slab, which spans a GiB of title
    /// memory, one per region written.
    before: [Mutex<Snapshots>; SLOTS],
    /// In place of snapshots over view memory ([`Shadow::arm`]): the regions
    /// whose title blocks are armed for the submission's copy-back, as
    /// (from, to, asking) offsets, and the blocks of them not armed, kept as
    /// the copy held them (by title block).
    armed: [Mutex<Vec<(usize, usize, u64)>>; SLOTS],
    kept: [Mutex<BTreeMap<usize, Vec<u8>>>; SLOTS],
    /// The buffer is title memory itself ([`in_place`]): nothing to copy.
    /// With a twin ([`in_place_twins_on`]), the twin takes the pages the
    /// title writes, and the compare stores the words shaders changed into
    /// title memory itself.
    in_place: bool,
    /// With [`audit_on`] (diagnostics): the copy's bytes of the ranges
    /// armed, by offset, as eager snapshots hold them ([`Shadow::audit`]).
    #[cfg(feature = "diagnostics")]
    audit: [Mutex<Snapshots>; SLOTS],
    /// Title ranges the CPU wrote into `buffer` since the twin took them
    /// ([`upload_twins`]).
    dirty: Mutex<Vec<(usize, usize)>>,
    /// Listed for the next upload ([`touched`]).
    listed: AtomicBool,
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

/// Runs `f` over `0..n` in parts, on the split threads
/// ([`crate::texture::split`], started once: a thread's start and end run
/// every title image's thread notices, ~110 ms) where the parts cover `len`
/// bytes of at least 16 MiB (one thread copies or compares ~10 GB/s; the
/// host's memory takes several times that). The results in order.
fn split<T: Send>(n: usize, len: usize, f: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let threads = match len {
        0..0x100_0000 => 1,
        _ => std::thread::available_parallelism().map_or(1, |t| t.get().min(8)),
    };
    if threads == 1 {
        return vec![f(0..n)];
    }
    let per = n.div_ceil(threads);
    let parts = (0..threads)
        .map(|k| (k * per).min(n)..((k + 1) * per).min(n))
        .collect();
    crate::texture::split(parts, f)
        .into_iter()
        .map(|done| done.unwrap_or_else(|e| std::panic::resume_unwind(e)))
        .collect()
}

/// The runs of words shaders wrote and the span of the words the copy takes
/// from title memory ([`merge_words`]).
type Merged = (Vec<(usize, usize)>, Option<(usize, usize)>);

/// Puts the bytes shaders wrote into `title`, title memory as read back,
/// and has the copy `host` take the result, by 4-byte word: a byte the GPU
/// changed (`host` against `old`, what the copy held before) that the title
/// does not already hold is the GPU's. Returns the runs of the words the GPU
/// changed and the span of the words in which the copy takes bytes of the
/// title's. 16 bytes at a time: word by word, with a slice compare and a
/// copy per word, it took most of copy-back's samples in runs 79 and 85.
fn merge_words(title: &mut [u8], host: &mut [u8], old: &[u8]) -> Merged {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi8, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_or_si128, _mm_storeu_si128,
    };
    assert!(host.len() == title.len() && old.len() == title.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut taken: Option<(usize, usize)> = None;
    let mut word = |a: usize, b: usize, gpu: bool, titles: bool| {
        if titles {
            taken = Some((taken.map_or(a, |s| s.0), b));
        }
        match runs.last_mut() {
            Some(run) if gpu && run.1 == a => run.1 = b,
            _ if gpu => runs.push((a, b)),
            _ => {}
        }
    };
    let lanes = title.len() / 16 * 16;
    for at in (0..lanes).step_by(16) {
        // SAFETY: SSE2 is part of x86_64; 16 bytes at `at` in each slice,
        // all as long as `title`.
        let (keep, same, unwritten) = unsafe {
            let t = _mm_loadu_si128(title.as_ptr().add(at).cast::<__m128i>());
            let h = _mm_loadu_si128(host.as_ptr().add(at).cast::<__m128i>());
            let o = _mm_loadu_si128(old.as_ptr().add(at).cast::<__m128i>());
            let (same, unwritten) = (_mm_cmpeq_epi8(t, h), _mm_cmpeq_epi8(h, o));
            // The title's bytes but where the GPU changed one the title
            // lacks.
            let keep = _mm_or_si128(same, unwritten);
            let merged = _mm_or_si128(_mm_and_si128(keep, t), _mm_andnot_si128(keep, h));
            _mm_storeu_si128(title.as_mut_ptr().add(at).cast::<__m128i>(), merged);
            _mm_storeu_si128(host.as_mut_ptr().add(at).cast::<__m128i>(), merged);
            (
                _mm_movemask_epi8(keep) as u32,
                _mm_movemask_epi8(same) as u32,
                _mm_movemask_epi8(unwritten) as u32,
            )
        };
        let (gpu, titles) = (!keep & 0xffff, !same & unwritten);
        if gpu | titles != 0 {
            for w in 0..4 {
                let (bits, a) = (0xf << (4 * w), at + 4 * w);
                word(a, a + 4, gpu & bits != 0, titles & bits != 0);
            }
        }
    }
    let words = title[lanes..]
        .chunks_mut(4)
        .zip(host[lanes..].chunks_mut(4))
        .zip(old[lanes..].chunks(4));
    for (k, ((t, h), o)) in words.enumerate() {
        let mut gpu = false;
        for ((t, h), o) in t.iter_mut().zip(h.iter()).zip(o) {
            if *h != *o && *t != *h {
                *t = *h;
                gpu = true;
            }
        }
        let a = lanes + 4 * k;
        let titles = *h != *t;
        h.copy_from_slice(t);
        word(a, a + t.len(), gpu, titles);
    }
    (runs, taken)
}

/// [`merge_words`] told the words the GPU changed by the compare's bits
/// (`bits`, from bit `first`, that of the word `skip` bytes into which the
/// slices start) instead of by what the copy held: those words' bytes are
/// the GPU's, and the copy takes the title's others. By word, as the
/// compare tells: a byte the title wrote meanwhile into a word the GPU
/// changed is lost. 16 bytes at a time where the slices hold 32 whole
/// words ([`lane_merges`]).
fn merge_bits(
    title: &mut [u8],
    host: &mut [u8],
    bits: &[u32],
    first: usize,
    skip: usize,
) -> Merged {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_andnot_si128, _mm_castsi128_ps, _mm_cmpeq_epi32,
        _mm_loadu_si128, _mm_movemask_ps, _mm_or_si128, _mm_storeu_si128,
    };
    /// By four bits of the compare's, the words they mark, as lanes.
    const LANES: [[u32; 4]; 16] = {
        let mut lanes = [[0; 4]; 16];
        let mut g = 0;
        while g < 16 {
            let mut w = 0;
            while w < 4 {
                if g >> w & 1 != 0 {
                    lanes[g][w] = u32::MAX;
                }
                w += 1;
            }
            g += 1;
        }
        lanes
    };
    assert!(host.len() == title.len() && skip < 4);
    let by_lanes = lane_merges();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut taken: Option<(usize, usize)> = None;
    let mut word = |a: usize, b: usize, gpu: bool| {
        if !gpu {
            taken = Some((taken.map_or(a, |s| s.0), b));
            return;
        }
        match runs.last_mut() {
            Some(run) if run.1 == a => run.1 = b,
            _ => runs.push((a, b)),
        }
    };
    let len = title.len();
    let words = (len + skip).div_ceil(4);
    let edge = |w: usize| (4 * w).saturating_sub(skip).min(len);
    for k in 0..words.div_ceil(32) {
        let mut mask = bits_at(bits, first + 32 * k);
        let held = (words - 32 * k).min(32);
        if held < 32 {
            mask &= (1 << held) - 1;
        }
        let (from, to) = (edge(32 * k), edge(32 * k + held));
        if mask == 0 && title[from..to] == host[from..to] {
            continue;
        }
        if by_lanes && to - from == 128 {
            for lane in 0..8 {
                let (at, g) = (from + 16 * lane, (mask >> (4 * lane) & 0xf) as usize);
                // SAFETY: SSE2 is part of x86_64; 16 bytes at `at` in both
                // slices, as long as each other.
                let differ = unsafe {
                    let t = _mm_loadu_si128(title.as_ptr().add(at).cast::<__m128i>());
                    let h = _mm_loadu_si128(host.as_ptr().add(at).cast::<__m128i>());
                    let differ = !_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(t, h))) & 0xf;
                    if differ != 0 {
                        let gpu = _mm_loadu_si128(LANES[g].as_ptr().cast::<__m128i>());
                        let merged = _mm_or_si128(_mm_and_si128(gpu, h), _mm_andnot_si128(gpu, t));
                        _mm_storeu_si128(title.as_mut_ptr().add(at).cast::<__m128i>(), merged);
                        _mm_storeu_si128(host.as_mut_ptr().add(at).cast::<__m128i>(), merged);
                    }
                    differ
                };
                let (gpu, titles) = (differ as usize & g, differ as usize & !g);
                if titles != 0 {
                    let (a, b) = (
                        titles.trailing_zeros(),
                        usize::BITS - titles.leading_zeros(),
                    );
                    word(at + 4 * a as usize, at + 4 * b as usize, false);
                }
                for w in (0..4).filter(|w| gpu >> w & 1 != 0) {
                    word(at + 4 * w, at + 4 * w + 4, true);
                }
            }
            continue;
        }
        for w in 0..held {
            let (a, b) = (edge(32 * k + w), edge(32 * k + w + 1));
            let (t, h) = (&mut title[a..b], &mut host[a..b]);
            if t == h {
                continue;
            }
            let gpu = mask >> w & 1 != 0;
            if gpu {
                t.copy_from_slice(h);
            } else {
                h.copy_from_slice(t);
            }
            word(a, b, gpu);
        }
    }
    (runs, taken)
}

/// Whether [`merge_bits`] compares and merges four words at a time where
/// the slices hold 32 whole words: word by word, with a slice compare and a
/// copy per word, it took most of copy-back's samples in run 100.
/// `VICTORY_NO_SIMD_MERGE` turns it off.
fn lane_merges() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_SIMD_MERGE").is_none();
        if !on {
            tracing::info!(
                "copy-back merges by the compare's bits word by word (VICTORY_NO_SIMD_MERGE)"
            );
        }
        on
    })
}

/// The 32 bits of `bits` from bit `at` (clear past its end).
fn bits_at(bits: &[u32], at: usize) -> u32 {
    let (i, s) = (at / 32, at % 32);
    let low = bits.get(i).map_or(0, |b| b >> s);
    match s {
        0 => low,
        _ => low | bits.get(i + 1).map_or(0, |b| b << (32 - s)),
    }
}

/// Whether any of `count` bits of `bits` from bit `first` is set: the
/// words between the first and the last read whole (copy-back reads a
/// world frame's ~6 MB of them).
fn any_bit(bits: &[u32], first: usize, count: usize) -> bool {
    if count == 0 {
        return false;
    }
    let end = first + count;
    let (a, z) = (first / 32, (end - 1) / 32);
    let edge = |i: usize| {
        let (lo, hi) = (first.max(32 * i) - 32 * i, end.min(32 * i + 32) - 32 * i);
        bits.get(i)
            .map_or(0, |b| b & (u32::MAX >> (32 - (hi - lo))) << lo)
    };
    edge(a) != 0
        || edge(z) != 0
        || bits
            .get(a + 1..z.min(bits.len()))
            .is_some_and(|m| m.iter().any(|&b| b != 0))
}

/// What tells [`Shadow::write_back`] the words shaders changed: what the
/// copy held before the submission, or the compare's bits ([`told`]) from
/// the bit of the word holding the range's first byte, with the title
/// blocks it found no word changed in ([`unchanged`]), whose bits go
/// unread.
enum How<'a> {
    Was(&'a Was<'a>),
    Bits {
        bits: &'a [u32],
        first: usize,
        unchanged: &'a std::collections::HashSet<usize>,
    },
}

/// What a copy held before a submission, for its copy-back: snapshots
/// (offset, bytes), blocks kept (title block, bytes) and armed regions
/// (from, to, asking; [`Shadow::arm`]).
struct Was<'a> {
    snapshots: &'a [(usize, Vec<u8>)],
    kept: &'a BTreeMap<usize, Vec<u8>>,
    armed: &'a [(usize, usize, u64)],
    start: usize,
    /// Title blocks in which shaders changed no word ([`unchanged`]).
    unchanged: &'a std::collections::HashSet<usize>,
    /// The submission's slot ([`preparing`]), for its audit.
    #[cfg(feature = "diagnostics")]
    slot: usize,
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

/// Diagnostics: `VICTORY_SNAPSHOT_AUDIT` takes eager snapshots beside the
/// copy-on-write ones and logs where copy-back would write other bytes
/// with them ([`Shadow::audit`]).
#[cfg(feature = "diagnostics")]
fn audit_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SNAPSHOT_AUDIT").is_some();
        if on {
            tracing::info!(
                "copy-on-write snapshots audited against eager ones (VICTORY_SNAPSHOT_AUDIT)"
            );
        }
        on
    })
}

/// Tests: a title write [`Shadow::write_back`] makes in its race window,
/// after the copy took title bytes and before it asks the guards.
#[cfg(test)]
static BEFORE_ASKING: Mutex<Option<Box<dyn FnOnce() + Send>>> = Mutex::new(None);

/// Tests: a title write while [`Shadow::write_back`] writes a piece back,
/// after it read the piece.
#[cfg(test)]
static BEFORE_WRITING: Mutex<Option<Box<dyn FnOnce() + Send>>> = Mutex::new(None);

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

/// Submissions whose copy-back state a copy keeps apart: those in flight
/// at once ([`crate::pipelined::DEPTH`]).
const SLOTS: usize = crate::pipelined::DEPTH;

thread_local! {
    /// The slot of the submission this thread prepares
    /// ([`crate::pipelined::room`]; 0 when not pipelined). Copies keep
    /// snapshots, armed and kept blocks and pending ranges per slot, and the
    /// twin compares their buffers, so that a submission's copy-back on the
    /// completion thread and the next one's preparation each touch their
    /// own. A submission is prepared on the thread that executes it, one at
    /// a time.
    static PREPARING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// This thread prepares a submission in slot `slot` next.
pub(crate) fn preparing(slot: usize) {
    PREPARING.with(|p| p.set(slot));
}

fn prep_slot() -> usize {
    PREPARING.with(std::cell::Cell::get)
}

const NOT_WATCHED: u8 = 2;

/// Whether refreshes of ranges last brought up to date wholly guarded and
/// clean are skipped while title memory there is unchanged since
/// ([`crate::watch::last_change`]): run 149 asked ~3,000 times a frame
/// (16.5 ms) about 13.5 GB, 99.4% of it unchanged, and walked every page it
/// checked. `VICTORY_REFRESH_ASKS` asks about them again.
fn skip_synced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_REFRESH_ASKS").is_none();
        if !on {
            tracing::info!(
                "refreshes of unchanged guarded ranges ask again (VICTORY_REFRESH_ASKS)"
            );
        }
        on
    })
}

// SAFETY: D3D12 resources are free-threaded; the memory a copy maps is
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.in_place {
            crate::watch::title_memory_held(self.start as usize, self.len as usize, false);
        }
    }
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The buffer shaders use: the twin, else the host copy.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        self.twin.as_ref().unwrap_or(&self.buffer)
    }

    /// Title memory `[from, to)` of a mirror slab is used by the submission
    /// being prepared: its windows' twins get VRAM and are brought up to
    /// date before it runs ([`map_windows`]).
    pub(crate) fn want(&self, from: usize, to: usize) {
        if self.twin.is_none() || !self.mirror() || from >= to {
            return;
        }
        let w = WINDOW as usize;
        mirror_state()
            .wanted
            .extend((from / w..to.div_ceil(w)).map(|w| w as u64));
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
        twin: Option<ID3D12Resource>,
        memory: Memory,
        start: u64,
        len: u64,
        in_place: bool,
    ) -> Shadow {
        let pages = match in_place && twin.is_none() {
            true => 0,
            false => (start + len).div_ceil(PAGE as u64) - start / PAGE as u64,
        };
        // SAFETY: live buffers.
        let gpu = unsafe { twin.as_ref().unwrap_or(&buffer).GetGPUVirtualAddress() };
        Shadow {
            buffer,
            twin,
            memory,
            gpu,
            start,
            len,
            loose: false,
            copied: Mutex::new(vec![0; pages as usize]),
            synced: Default::default(),
            watched: AtomicU8::new(0),
            shaders_write: AtomicBool::new(false),
            pending: Default::default(),
            before: Default::default(),
            armed: Default::default(),
            kept: Default::default(),
            in_place,
            #[cfg(feature = "diagnostics")]
            audit: Default::default(),
            dirty: Mutex::default(),
            listed: AtomicBool::new(false),
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
        #[cfg(feature = "diagnostics")]
        self.note_reached(address, len, true, false);
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// Diagnostics: counts `[address, address + len)` as given to the
    /// pipeline being bound to read, or to `write`, by where its shaders
    /// reach it ([`REACHED`]); a view that reads a copy of its `own`
    /// reaches system memory.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_reached(&self, address: u64, len: u64, write: bool, own: bool) {
        let len = len.min(self.left(address));
        let mut r = Reach {
            views: 1,
            ..Reach::default()
        };
        if own {
            r.system = len;
        } else if self.in_place && self.twin.is_none() {
            r.in_place = len;
        } else if !self.mirror() {
            if self.twin.is_some() {
                r.vram = len;
            } else {
                r.system = len;
            }
        } else {
            let m = mirror_state();
            let end = address + len;
            for n in address / WINDOW..end.div_ceil(WINDOW) {
                let bytes = end.min((n + 1) * WINDOW) - address.max(n * WINDOW);
                if self.twin.is_some() && m.twinned.contains_key(&n) {
                    r.vram += bytes;
                    r.vram_windows.insert(n);
                } else if m.windows.contains_key(&n) {
                    r.system += bytes;
                    r.system_windows.insert(n);
                }
            }
        }
        let pipeline = crate::descriptor::host::BINDING.get();
        let mut reached = REACHED.lock().unwrap_or_else(|e| e.into_inner());
        reached.entry(pipeline).or_default()[usize::from(write)].add(r);
    }

    /// The copy's offsets of the ranges `declared` (address, bytes),
    /// merged.
    fn merged(&self, declared: &[(usize, usize)]) -> Vec<(usize, usize)> {
        merge(
            declared
                .iter()
                .map(|&(address, len)| self.inside(address, len))
                .filter(|(from, to)| from < to)
                .collect(),
        )
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote
    /// ([`Shadow::snapshot`]), unless the compare tells it by word.
    pub(crate) fn writing(self: &Arc<Self>, address: usize, len: usize) {
        if self.in_place {
            return;
        }
        let slot = prep_slot();
        self.want(address, address + len);
        self.pending[slot]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        if !self.told_by_words() {
            self.snapshot_after(address, len, slot);
        }
    }

    /// [`Shadow::snapshot`] once no submission in flight writes the range:
    /// pipelined, deferred to the completion of the newest that does
    /// (`crate::pipelined::deferred_snapshots`), or after waiting for it.
    fn snapshot_after(self: &Arc<Self>, address: usize, len: usize, slot: usize) {
        use crate::pipelined::Writer;
        let s = self.clone();
        let (from, to) = (address, address + len);
        if !crate::pipelined::defer(from, to, Writer::Snapshot, move || {
            s.snapshot(address, len, slot)
        }) {
            crate::pipelined::wait_writes(from, to, Writer::Snapshot);
            self.snapshot(address, len, slot);
        }
    }

    /// Whether the compare tells copy-back by word what shaders changed in
    /// this copy ([`words_on`]): its GpuWrite steps take no snapshot, and
    /// ranges no compare tells take theirs at submission
    /// ([`snapshot_untold`]).
    pub(crate) fn told_by_words(&self) -> bool {
        words_on()
            && twin_compare_on()
            && self.twin.is_some()
            && !self.in_place
            && self.start.is_multiple_of(4)
    }

    /// The bytes of title addresses `[address, address + len)` kept for the
    /// submission in `slot`: one snapshot per region of the copy and
    /// submission, grown to cover every range the submission declares there
    /// (the world declares overlapping views of its 174 MB heap dozens of
    /// times), joining ranges less than a window apart, in reused memory
    /// (fresh snapshots cost ~2.3 million page faults per world frame).
    fn snapshot(&self, address: usize, len: usize, slot: usize) {
        let (from, to) = self.inside(address, len);
        if from < to && armed_snapshots() && crate::watch::view(self.start as usize + from) {
            #[cfg(feature = "diagnostics")]
            self.audit_snapshot(from, to, slot);
            self.arm(from, to, slot);
            return;
        }
        let w = WINDOW as usize;
        let joins = |at: usize, n: usize| at <= to + w && from <= at + n + w;
        let mut before = self.before[slot].lock().unwrap_or_else(|e| e.into_inner());
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
    /// are kept as the copy holds them. For the submission in `slot`.
    fn arm(&self, from: usize, to: usize, slot: usize) {
        let start = self.start as usize;
        let (asked, loose) = {
            let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            let base = start / PAGE;
            crate::watch::arm(start + from, start + to, slot, |page| {
                copied.get(page.wrapping_sub(base)).copied().unwrap_or(0)
            })
        };
        self.armed[slot]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((from, to, asked));
        let mut kept = self.kept[slot].lock().unwrap_or_else(|e| e.into_inner());
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

    /// Diagnostics, with [`audit_on`]: takes the copy's bytes of offsets
    /// `[from, to)` not taken yet, as an eager snapshot holds them.
    #[cfg(feature = "diagnostics")]
    fn audit_snapshot(&self, from: usize, to: usize, slot: usize) {
        if !audit_on() {
            return;
        }
        let mut audit = self.audit[slot].lock().unwrap_or_else(|e| e.into_inner());
        let mut taken: Vec<(usize, usize)> = audit
            .iter()
            .map(|(a, b)| (*a, a + b.len()))
            .filter(|&(a, b)| a < to && from < b)
            .collect();
        taken.sort_unstable();
        let mut at = from;
        let mut gaps = Vec::new();
        for (a, b) in taken {
            if at < a {
                gaps.push((at, a));
            }
            at = at.max(b);
        }
        if at < to {
            gaps.push((at, to));
        }
        for (a, b) in gaps {
            let mut bytes = pooled(b - a);
            self.gather(a, b, &mut bytes);
            audit.push((a, bytes));
        }
    }

    /// Diagnostics, with [`audit_on`]: logs where copy-back of the copy's
    /// offsets `[at, at + host.len())` would write other bytes than with an
    /// eager snapshot. `lazy` is what copy-back takes the copy to have held
    /// there (none: it writes nothing), `how` where that came from. Of the
    /// submission in `slot`.
    #[cfg(feature = "diagnostics")]
    fn audit(
        &self,
        slot: usize,
        at: usize,
        host: &[u8],
        lazy: Option<&[u8]>,
        how: impl FnOnce() -> String,
    ) {
        static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !audit_on() {
            return;
        }
        let n = host.len();
        let audit = self.audit[slot].lock().unwrap_or_else(|e| e.into_inner());
        let Some((taken, eager)) = audit
            .iter()
            .find(|(t, b)| *t <= at && at + n <= t + b.len())
        else {
            return;
        };
        let eager = &eager[at - taken..at - taken + n];
        let start = self.start as usize;
        let mut title = vec![0u8; n];
        if !crate::host::read_into((start + at) as *const u8, &mut title) {
            return;
        }
        let writes = |before: &[u8], i: usize| host[i] != before[i] && title[i] != host[i];
        let (mut differ, mut first, mut by_lazy, mut by_eager) = (0, None, 0, 0);
        for i in 0..n {
            let (l, e) = (lazy.is_some_and(|l| writes(l, i)), writes(eager, i));
            by_lazy += usize::from(l);
            by_eager += usize::from(e);
            if l != e {
                differ += 1;
                first.get_or_insert(i);
            }
        }
        let Some(first) = first else {
            return;
        };
        let k = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
        if k > 1000 && !k.is_multiple_of(1000) {
            return;
        }
        let i = first & !3;
        let word = |b: &[u8]| {
            b.get(i..i + 4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        };
        tracing::warn!(
            pieces = k,
            frame = crate::frame::current(),
            asking = crate::watch::asking(),
            copy = format_args!("{:#x}+{:#x}", self.start, self.len),
            at = format_args!("{:#x}", start + at + first),
            differ,
            by_lazy,
            by_eager,
            host = ?word(host),
            lazy = ?lazy.and_then(word),
            eager = ?word(eager),
            title = ?word(&title),
            how = %how(),
            "copy-back with the copy-on-write snapshot writes other bytes than with the eager one"
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
    /// declared ranges hold it (the world's overlap dozens of times), except
    /// in the title blocks a compare on the GPU found `unchanged`. Of the
    /// submission in `slot` ([`preparing`]), whose state alone it clears:
    /// the next may be prepared meanwhile. Returns the bytes written.
    pub(crate) fn copy_back(
        &self,
        declared: &[(usize, usize)],
        unchanged: &std::collections::HashSet<usize>,
        told: Option<ToldOf>,
        slot: usize,
    ) -> usize {
        let merged = self.merged(declared);
        #[cfg(feature = "diagnostics")]
        lux_copy_back(self, &merged, "before");
        let start = self.start as usize;
        if self.in_place {
            // Shaders wrote title memory (with a twin, the compare stored
            // what they changed there); copies and textures of it that the
            // kernel's watch cannot tell (no CPU wrote) take it again. The
            // twin holds it already: of the pages it had taken, it takes
            // again only those the title wrote before our write (`title`)
            // or after it (`since`); the others count as taken at our write.
            for (from, to) in merged {
                let (a, b) = (start + from, start + to);
                let title = self
                    .twin
                    .as_ref()
                    .and_then(|_| crate::watch::writes_now(a, b));
                let at = crate::watch::wrote(a, b);
                let Some((_, title)) = title else {
                    continue;
                };
                let Some((_, since)) = crate::watch::writes_now(a, b) else {
                    continue;
                };
                let (base, end) = (start / PAGE, start + self.len as usize);
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for &(page, last) in &since {
                    let wrote = title
                        .binary_search_by_key(&page, |w| w.0)
                        .map_or(0, |i| title[i].1);
                    let whole = (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= b;
                    let taken = &mut copied[page - base];
                    if whole && *taken != 0 && wrote <= *taken && last <= at {
                        *taken = at;
                    }
                }
            }
            return 0;
        }
        let mut snapshots = self.before[slot].lock().unwrap_or_else(|e| e.into_inner());
        let mut armed = self.armed[slot].lock().unwrap_or_else(|e| e.into_inner());
        let mut kept = self.kept[slot].lock().unwrap_or_else(|e| e.into_inner());
        let was = Was {
            snapshots: &snapshots,
            kept: &kept,
            armed: &armed,
            start,
            unchanged,
            #[cfg(feature = "diagnostics")]
            slot,
        };
        // The ranges the compare told by word ([`told`]), whole words, within
        // those declared, once each: offsets and the bit of the word holding
        // the first byte. What the copy held tells the rest.
        let mut by_words: Vec<(usize, usize, usize)> = Vec::new();
        for &(title, bytes, bit) in told.map_or(&[][..], |t| t.1) {
            let (a, b) = (title - start, title - start + bytes);
            let first = merged.partition_point(|&(_, to)| to <= a);
            for &(from, to) in merged[first..].iter().take_while(|r| r.0 < b) {
                let lo = from.max(a);
                by_words.push((lo, to.min(b), bit + lo / 4 - a / 4));
            }
        }
        by_words.sort_unstable();
        let mut end = 0;
        by_words.retain_mut(|r| {
            if r.0 < end {
                r.2 += end / 4 - r.0 / 4;
                r.0 = end;
            }
            end = end.max(r.1);
            r.0 < r.1
        });
        let mut written = 0;
        #[cfg(feature = "diagnostics")]
        let lux_merged = merged.clone();
        for (from, to) in merged {
            let mut at = from;
            for &(a, b, _) in &by_words {
                if b <= at {
                    continue;
                }
                if a >= to {
                    break;
                }
                if a > at {
                    written += self.write_back(at, a, &How::Was(&was));
                }
                at = at.max(b);
            }
            if at < to {
                written += self.write_back(at, to, &How::Was(&was));
            }
        }
        if let Some((bits, _)) = told {
            for &(from, to, first) in &by_words {
                let how = How::Bits {
                    bits,
                    first,
                    unchanged,
                };
                written += self.write_back(from, to, &how);
            }
        }
        #[cfg(feature = "diagnostics")]
        lux_copy_back(self, &lux_merged, "after");
        // The submission is done with its armed blocks (its copy-back
        // disarms them all).
        kept.clear();
        armed.clear();
        #[cfg(feature = "diagnostics")]
        for (_, bytes) in self.audit[slot]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            give_back(bytes);
        }
        // Every submitted step that declared the ranges is done with them,
        // and the snapshots with the last.
        let mut pending = self.pending[slot].lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&(a, b)| !declared.contains(&(a, b - a)));
        if pending.is_empty() {
            for (_, bytes) in snapshots.drain(..) {
                give_back(bytes);
            }
        }
        written
    }

    /// [`Shadow::copy_back`] of offsets `[from, to)`, `how` tells the words
    /// shaders changed: by 64 KiB piece, the copy compared with its bytes
    /// before the submission (on several threads for big ranges; armed
    /// blocks as title memory holds them) or the compare's bits read, title
    /// memory read only where the GPU wrote.
    fn write_back(&self, from: usize, to: usize, how: &How) -> usize {
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
        let changed: Vec<usize> = match how {
            How::Bits {
                bits,
                first,
                unchanged,
            } => (0..pieces.len())
                .filter(|&i| {
                    let (at, _, n) = pieces[i];
                    if unchanged.contains(&((start + at) / PAGE)) {
                        crate::host::TWINS[4].fetch_add(n as u64, Ordering::Relaxed);
                        return false;
                    }
                    let words = (at + n).div_ceil(4) - at / 4;
                    any_bit(bits, first + at / 4 - from / 4, words)
                })
                .collect(),
            How::Was(was) => split(pieces.len(), to - from, |part| {
                let mut seen = None;
                part.filter(|&i| {
                    let (at, host, n) = pieces[i];
                    if was.unchanged.contains(&((start + at) / PAGE)) {
                        crate::host::TWINS[4].fetch_add(n as u64, Ordering::Relaxed);
                        return false;
                    }
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
            .concat(),
        };
        let mut title = vec![0u8; PIECE];
        let (mut written, mut synced) = (0, Vec::new());
        for i in changed {
            let (at, host, n) = pieces[i];
            let saved;
            let old: Option<&[u8]> = match how {
                How::Bits { .. } => None,
                How::Was(was) => match was.at(at, n) {
                    Some(Before::Bytes(before)) => Some(before),
                    Some(Before::Armed(asked)) => {
                        let block = (start + at) / PAGE;
                        let Some(bytes) = crate::watch::before(block, asked) else {
                            #[cfg(feature = "diagnostics")]
                            self.audit(
                                was.slot,
                                at,
                                // SAFETY: as below.
                                unsafe { std::slice::from_raw_parts(host as *const u8, n) },
                                None,
                                || format!("missed: {}", crate::watch::describe(block, asked)),
                            );
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
                        Some(&saved[first..first + n])
                    }
                    None => continue,
                },
            };
            #[cfg(feature = "diagnostics")]
            if let (How::Was(was), Some(old)) = (how, old) {
                self.audit(
                    was.slot,
                    at,
                    // SAFETY: as below.
                    unsafe { std::slice::from_raw_parts(host as *const u8, n) },
                    Some(old),
                    || match was.at(at, n) {
                        Some(Before::Armed(asked)) => {
                            crate::watch::describe((start + at) / PAGE, asked)
                        }
                        _ if was.kept.contains_key(&((start + at) / PAGE)) => "kept".into(),
                        _ => "snapshot".into(),
                    },
                );
            }
            // SAFETY: as above.
            let host = unsafe { std::slice::from_raw_parts_mut(host as *mut u8, n) };
            let title = &mut title[..n];
            if !crate::host::read_into((start + at) as *const u8, title) {
                continue;
            }
            // Title memory with the GPU's bytes put in; the copy takes the
            // title's bytes in turn. Only the words the GPU changed go back:
            // the title writes the piece while we compare it, and writing it
            // whole put the bytes we read back over those writes.
            let (runs, taken) = match (old, how) {
                (Some(old), _) => merge_words(title, host, old),
                (None, How::Bits { bits, first, .. }) => {
                    merge_bits(title, host, bits, first + at / 4 - from / 4, at % 4)
                }
                (None, How::Was(_)) => unreachable!("a piece told by no bytes"),
            };
            #[cfg(test)]
            if let Some(race) = BEFORE_WRITING
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                race();
            }
            if !runs.is_empty() {
                #[cfg(feature = "diagnostics")]
                let started = std::time::Instant::now();
                let to = start + at;
                // A kernel copy per run costs ~0.7 us: blocks guarded until
                // now are written in place when the GPU's words are scattered.
                crate::watch::writing_in_place(to, n, |ours| {
                    for &(a, b) in &runs {
                        let (dst, src) = ((to + a) as *mut u8, title[a..b].as_ptr());
                        if ours && runs.len() > 16 {
                            // SAFETY: guarded blocks just opened: mapped
                            // read-write, and no unmapping runs meanwhile.
                            unsafe { std::ptr::copy_nonoverlapping(src, dst, b - a) };
                        } else {
                            crate::host::write_mapped(dst, src, b - a);
                        }
                    }
                });
                written += n;
                #[cfg(feature = "diagnostics")]
                crate::copystat::cost(
                    crate::copystat::Op::WriteBack,
                    started,
                    start + at,
                    n as u64,
                );
            }
            synced.push(i);
            if let Some((a, b)) = taken
                && self.twin.is_some()
            {
                self.dirty
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((start + at + a, start + at + b));
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(
            crate::copystat::Op::Compare,
            started,
            start + from,
            (to - from) as u64,
        );
        #[cfg(test)]
        if let Some(race) = BEFORE_ASKING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            race();
        }
        // Pages made equal to title memory are current in the copy, so our
        // own write is not taken for the title's; others keep what they had
        // (a title write there shows at the next refresh). Only those still
        // equal once guarded again: the title may have written one since the
        // copy took it (unguarded once we wrote it), and the asking takes that
        // write as well, which no copy holds yet (run 67's audit).
        let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
        if first < last
            && !synced.is_empty()
            && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, _)) = crate::watch::writes_now(first * PAGE, last * PAGE)
        {
            let current = split(synced.len(), synced.len() * PIECE, |part| {
                let mut title = vec![0u8; PIECE];
                part.filter_map(|k| {
                    let (at, host, n) = pieces[synced[k]];
                    let page = (start + at) / PAGE;
                    // SAFETY: inside the copy's memory, which only this
                    // thread writes.
                    let host = unsafe { std::slice::from_raw_parts(host as *const u8, n) };
                    let title = &mut title[..n];
                    let equal = (first..last).contains(&page)
                        && crate::host::read_into((start + at) as *const u8, title)
                        && *title == *host;
                    equal.then_some(page)
                })
                .collect::<Vec<_>>()
            })
            .concat();
            let base = start / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for page in current {
                copied[page - base] = now;
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched), waiting for the submissions in flight whose shaders
    /// write the pages it copies (`crate::pipelined::wait_writes`).
    /// Returns the bytes copied.
    pub(crate) fn update(self: &Arc<Self>, address: usize, len: usize) -> usize {
        if self.in_place && self.twin.is_none() {
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
            self.want(from, to);
        }
        let skip = skip_synced();
        let since = crate::watch::changes();
        if skip
            && let Some(last) = crate::watch::last_change(from, to)
            && let Some(&at) = self
                .synced
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&(from, to))
            && last <= at
        {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Skipped, 1);
                add(C::SkippedBytes, (to - from) as u64);
            }
            return 0;
        }
        let copied = if self.mirror() {
            self.update_mirror(from, to)
        } else {
            self.update_free(from, to)
        };
        if skip && crate::watch::unwritten(from, to) {
            let mut synced = self.synced.lock().unwrap_or_else(|e| e.into_inner());
            if synced.len() >= 4096 {
                synced.clear();
            }
            synced.insert((from, to), since);
        }
        copied
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer).
    fn update_free(self: &Arc<Self>, from: usize, to: usize) -> usize {
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
    fn update_watched(self: &Arc<Self>, from: usize, to: usize) -> Option<usize> {
        #[cfg(feature = "diagnostics")]
        let mut started = None;
        let stale = {
            // Before the journal, as arming takes them; let go before the
            // copies, which may wait for a copy-back that takes them.
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            crate::watch::writes_with(from, to, true, |now, written| {
                #[cfg(feature = "diagnostics")]
                {
                    started = Some(std::time::Instant::now());
                }
                self.stale(&mut copied, from, to, now, written)
            })?
        };
        let bytes = stale.into_iter().map(|(a, b)| self.copy(a, b)).sum();
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, Op, add, cost};
            if let Some(started) = started {
                cost(Op::Copy, started, from, bytes as u64);
            }
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(bytes) as u64);
        }
        Some(bytes)
    }

    /// [`Shadow::update`] in a mirror slab, which spans title memory of
    /// every kind (write-watched allocations, runs of views, free
    /// addresses) while the watch answers for one allocation or one run of
    /// views at a time: asked as a whole, then by window, then by the
    /// regions the system reports in a window it cannot answer for, where
    /// committed regions it cannot watch are copied whole and others are
    /// left as they are (zeros, unless the title freed memory copied
    /// there). Nothing is ever taken as unwatched for the whole slab.
    fn update_mirror(self: &Arc<Self>, from: usize, to: usize) -> usize {
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

    /// The pages of `[from, to)` written after their last copy (`written`:
    /// last writes by page, in page order; `now`: this asking), noted
    /// `copied` now: the runs to copy, whole pages within the buffer.
    fn stale(
        &self,
        copied: &mut [u64],
        from: usize,
        to: usize,
        now: u64,
        written: crate::watch::Written<'_>,
    ) -> Vec<(usize, usize)> {
        let base = self.start as usize / PAGE;
        let mut writes = written.peekable();
        let (mut runs, mut run) = (Vec::new(), None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&(p, w)) if p == page => {
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
                runs.push(((first * PAGE).max(from), (page * PAGE).min(to)));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            runs.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to)));
        }
        runs
    }

    /// [`Shadow::copy`] of memory nothing can tell the changes of.
    fn copy_whole(self: &Arc<Self>, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let n = self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::cost(crate::copystat::Op::Whole, started, from, n as u64);
        n
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy
    /// (giving mirror windows host memory), and into the snapshots of
    /// ranges shaders write where they have them. Pipelined, bytes a
    /// submission in flight has left to copy back are taken after its
    /// copy-back, on the completion thread, and the submission being
    /// prepared waits on the GPU for it
    /// (`crate::pipelined::deferred_refreshes`).
    fn copy(self: &Arc<Self>, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        #[cfg(feature = "diagnostics")]
        crate::pipelining::touch(crate::pipelining::Touch::Refresh, from, to - from);
        // Its twin takes it before the submission being prepared runs.
        if self.twin.is_some() {
            self.dirty
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((from, to));
        }
        let slot = prep_slot();
        if crate::pipelined::deferred_refreshes() {
            // Mirror windows get their memory now, to be mapped before the
            // submission runs.
            if !self.in_place {
                self.pieces(from - start, to - start, true);
            }
            let copy = self.clone();
            let refresh = crate::pipelined::Writer::Refresh;
            if crate::pipelined::defer(from, to, refresh, move || copy.take(from, to, slot)) {
                return to - from;
            }
        } else {
            // Pipelined: the copy and the snapshots would take bytes a
            // submission in flight writes, or be overwritten by what it
            // wrote, before its copy-back. Pages the title wrote elsewhere
            // in the range do not wait for it.
            crate::pipelined::wait_writes(from, to, crate::pipelined::Writer::Refresh);
        }
        self.take(from, to, slot);
        to - from
    }

    /// [`Shadow::copy`] of `[from, to)` (inside the buffer) for the
    /// submission in `slot`, which takes it, once no submission in flight
    /// has it left to write.
    fn take(&self, from: usize, to: usize, slot: usize) {
        let start = self.start as usize;
        // Title memory used in place is the copy: only its twin takes it.
        if !self.in_place {
            for (at, host, n) in self.pieces(from - start, to - start, true) {
                if !host.is_null() {
                    crate::host::copy_mapped(host, (start + at) as *const u8, n, false);
                }
            }
        }
        let mut before = self.before[slot].lock().unwrap_or_else(|e| e.into_inner());
        for (taken, bytes) in before.iter_mut() {
            let (a, b) = (
                (from - start).max(*taken),
                (to - start).min(*taken + bytes.len()),
            );
            if a < b {
                self.fill(a, &mut bytes[a - *taken..b - *taken]);
            }
        }
        #[cfg(feature = "diagnostics")]
        for (taken, bytes) in self.audit[slot]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
        {
            let (a, b) = (
                (from - start).max(*taken),
                (to - start).min(*taken + bytes.len()),
            );
            if a < b {
                self.fill(a, &mut bytes[a - *taken..b - *taken]);
            }
        }
        // Armed blocks refreshed since: the copy no longer holds what title
        // memory does when the title writes them next, so they are kept.
        let armed = self.armed[slot].lock().unwrap_or_else(|e| e.into_inner());
        if !armed.is_empty() {
            let mut kept = self.kept[slot].lock().unwrap_or_else(|e| e.into_inner());
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
        crate::host::name(&buffer, || {
            format!("title memory in place {start:#x}+{len:#x}")
        });
        crate::watch::title_memory_held(start, len, true);
        let twin = (twins_on()
            && twin_compare_on()
            && in_place_twins_on()
            && crate::watch::write_watched(start))
        .then(|| vram_buffer(len as u64))
        .flatten();
        if let Some(twin) = &twin {
            crate::host::name(twin, || {
                format!("title memory in place twin {start:#x}+{len:#x}")
            });
        }
        let twinned = twin.is_some();
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{start:#x}"),
                len,
                twinned,
                "large buffer used in place"
            );
        } else {
            tracing::debug!(
                start = format_args!("{start:#x}"),
                len,
                twinned,
                "buffer used in place"
            );
        }
        let memory = Memory::Linear(start as *mut u8);
        let shadow = Shadow::new(buffer, twin, memory, start as u64, len as u64, true);
        #[cfg(feature = "diagnostics")]
        made(shadow.gpu, len as u64, start as u64, len as u64);
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
        let twin = twins_on()
            .then(|| vram_buffer(len.next_multiple_of(1 << 16)))
            .flatten();
        if let Some(twin) = &twin {
            crate::host::name(twin, || format!("title buffer twin {:#x}+{len:#x}", key.0));
        }
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
        let shadow = Shadow::new(buffer, twin, Memory::Linear(data), key.0, len, false);
        #[cfg(feature = "diagnostics")]
        made(shadow.gpu, len.next_multiple_of(1 << 16), key.0, len);
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

/// Whether title buffers used in place get VRAM twins: shaders reach title
/// memory in place at system memory's speed (run 79: the hair strands'
/// ~90 ms of GPU a frame, most of it their stores; the light clusters'
/// bits, ~17 ms with their clear and the light proxies' atomic ORs;
/// hair_resolve's reads, 7 ms). Buffers shaders only read get one too,
/// since the first use cannot tell; only of write-watched memory, since
/// the kernel refuses to guard memory a heap holds and nothing else tells
/// the title's writes there ([`crate::watch::title_memory_held`]). The
/// twin takes the pages the title writes before each submission
/// ([`Shadow::update`]), and after it the compare stores the words shaders
/// changed into title memory itself ([`compare_twins`]), so copy-back has
/// nothing to write. Like a copy, a twin holds only what refreshes took:
/// ranges no draw is known to read (heap slots only shaders index) keep
/// what was last taken, where title memory in place showed the title's
/// latest. Only with the compare: copied down whole ([`download_twins`]),
/// the twin's ranges put back over title memory what the title wrote there
/// while the submission ran, as they still do in a submission whose ranges
/// pass [`KEEP_MAX`] or the VRAM the host gives.
/// `VICTORY_NO_IN_PLACE_TWINS` turns it off.
fn in_place_twins_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_IN_PLACE_TWINS").is_none();
        if !on {
            tracing::info!(
                "title buffers used in place have no VRAM twins (VICTORY_NO_IN_PLACE_TWINS)"
            );
        }
        on
    })
}

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
    let twin = twins_on().then(|| vram_buffer(len)).flatten();
    if let Some(twin) = &twin {
        crate::host::name(twin, || format!("title memory twin {start:#x}+{len:#x}"));
    }
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, twin, Memory::Linear(data), start, len, false);
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
    /// VRAM heaps for the slabs' twins, each with a buffer over it.
    vram: Vec<(ID3D12Heap, ID3D12Resource)>,
    /// Window number → VRAM heap and slot of its twin, and the submission
    /// that last used it.
    twinned: BTreeMap<u64, (usize, u32, u64)>,
    /// VRAM slots windows gone cold gave back.
    vram_free: Vec<VramSlot>,
    /// Slots used in the last VRAM heap.
    vram_used: u32,
    /// Windows the submission being prepared uses ([`Shadow::want`]).
    wanted: std::collections::BTreeSet<u64>,
    /// Windows given VRAM for this submission: their twins take them whole.
    fresh: Vec<u64>,
    /// Twin tiles to map onto a VRAM slot, or to unmap: slab, window, slot.
    retile: Vec<(u64, u64, Option<VramSlot>)>,
    /// Submissions prepared.
    now: u64,
    /// Windows submissions used since the last frame-rate line
    /// (diagnostics).
    #[cfg(feature = "diagnostics")]
    hot: std::collections::BTreeSet<u64>,
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
    vram: Vec::new(),
    twinned: BTreeMap::new(),
    vram_free: Vec::new(),
    vram_used: 0,
    wanted: std::collections::BTreeSet::new(),
    fresh: Vec::new(),
    retile: Vec::new(),
    now: 0,
    #[cfg(feature = "diagnostics")]
    hot: std::collections::BTreeSet::new(),
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

    /// Before a submission: gives the windows it uses VRAM for their
    /// twins (past the budget, windows unused longest give theirs back) and
    /// notes the tiles to map.
    fn twin_windows(&mut self) {
        self.now += 1;
        let now = self.now;
        let mut new = Vec::new();
        for w in std::mem::take(&mut self.wanted) {
            #[cfg(feature = "diagnostics")]
            self.hot.insert(w);
            match self.twinned.get_mut(&w) {
                Some(t) => t.2 = now,
                None if self.windows.contains_key(&w) => new.push(w),
                None => {}
            }
        }
        if new.is_empty() {
            return;
        }
        let budget = twin_budget();
        let over = (self.twinned.len() + new.len()).saturating_sub(budget);
        if over > 0 {
            let mut cold: Vec<(u64, u64)> = self
                .twinned
                .iter()
                .filter(|(_, t)| t.2 + COLD < now)
                .map(|(&w, t)| (t.2, w))
                .collect();
            cold.sort_unstable();
            for (_, w) in cold.into_iter().take(over) {
                if let Some((heap, slot, _)) = self.twinned.remove(&w) {
                    crate::host::TWINS[6].fetch_add(1, Ordering::Relaxed);
                    self.vram_free.push((heap, slot));
                    for k in self.twin_slabs(w) {
                        self.retile.push((k, w, None));
                    }
                }
            }
        }
        // Windows past the budget, or past the VRAM the host gives, stay
        // in system memory, which their twins show ([`map_windows`]).
        for w in new {
            if self.twinned.len() >= budget {
                break;
            }
            let Some(slot) = self.vram_slot() else {
                break;
            };
            self.twinned.insert(w, (slot.0, slot.1, now));
            for k in self.twin_slabs(w) {
                self.retile.push((k, w, Some(slot)));
            }
            self.fresh.push(w);
        }
    }

    /// A VRAM slot for a window's twin: one a cold window gave back, else
    /// the next of the last VRAM heap, else of a new one.
    fn vram_slot(&mut self) -> Option<VramSlot> {
        const SLOTS: u32 = (POOL_HEAP / WINDOW) as u32;
        if let Some(slot) = self.vram_free.pop() {
            return Some(slot);
        }
        if self.vram.is_empty() || self.vram_used == SLOTS {
            if NO_VRAM.load(Ordering::Relaxed) {
                return None;
            }
            let Some(heap) = vram_heap() else {
                NO_VRAM.store(true, Ordering::Relaxed);
                return None;
            };
            self.vram.push(heap);
            self.vram_used = 0;
        }
        self.vram_used += 1;
        Some((self.vram.len() - 1, self.vram_used - 1))
    }

    /// The slabs with twins that show window `w`.
    fn twin_slabs(&self, w: u64) -> Vec<u64> {
        slabs_showing(w)
            .filter(|k| self.slabs.get(k).is_some_and(|s| s.twin.is_some()))
            .collect()
    }
}

/// Whether host copies have VRAM twins that shaders use: the copies live
/// in system memory, which shaders read at 4.5–8 GB/s and write at ~0.1
/// GB/s over this host's PCIe link, against ~930 GB/s in VRAM (VICT-9's
/// probe). What the CPU writes into a copy goes to its twin before the
/// submission ([`upload_twins`]), and what shaders wrote comes back before
/// copy-back ([`download_twins`]). `VICTORY_NO_VRAM_TWINS` turns them off.
pub(crate) fn twins_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_VRAM_TWINS").is_none();
        if !on {
            tracing::info!("host copies have no VRAM twins (VICTORY_NO_VRAM_TWINS)");
        }
        on
    })
}

/// Mirror windows with VRAM twins past which windows unused for [`COLD`]
/// submissions give theirs back: `VICTORY_TWIN_MIB` MiB of them (4096).
fn twin_budget() -> usize {
    static WINDOWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WINDOWS.get_or_init(|| {
        let mib = std::env::var("VICTORY_TWIN_MIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096);
        mib / (WINDOW >> 20) as usize
    })
}

/// Submissions a window's twin stays unused before it may give its VRAM
/// back (~10 frames).
const COLD: u64 = 200;

/// A VRAM slot of the mirror's twins: its heap and place in it.
type VramSlot = (usize, u32);

/// The host gave no VRAM heap for the mirror's twins: windows not
/// twinned yet stay in system memory.
static NO_VRAM: AtomicBool = AtomicBool::new(false);

/// A VRAM heap for the mirror's twins, with a buffer over it.
fn vram_heap() -> Option<(ID3D12Heap, ID3D12Resource)> {
    let host = host()?;
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_HEAP,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let (mut heap, mut buffer): (Option<ID3D12Heap>, Option<ID3D12Resource>) = (None, None);
    // SAFETY: plain creation of a heap and a buffer over all of it.
    let made = unsafe {
        host.device.CreateHeap(&desc, &mut heap).and_then(|()| {
            let h = heap.as_ref().ok_or_else(windows::core::Error::empty)?;
            host.device.CreatePlacedResource(
                h,
                0,
                &buffer_desc(POOL_HEAP),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        })
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, "no VRAM for more mirror twins: further windows stay in system memory");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || "title memory mirror twin pool".into());
    Some((heap?, buffer))
}

/// A buffer of `len` bytes in VRAM that shaders write: the twin of a host
/// copy.
fn vram_buffer(len: u64) -> Option<ID3D12Resource> {
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer.
    let made = unsafe {
        host()?.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(len),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, len, "no VRAM twin: the host copy stays in system memory");
    }
    buffer
}

/// A reserved buffer of `len` bytes, none of it mapped.
fn reserved(len: u64) -> Option<ID3D12Resource> {
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a reserved buffer.
    let made = unsafe {
        host()?.device.CreateReservedResource(
            &buffer_desc(len),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, len, "no reserved buffer");
    }
    buffer
}

/// Copies whose host copy the CPU wrote, for the next [`upload_twins`].
static TOUCHED: Mutex<Vec<Arc<Shadow>>> = Mutex::new(Vec::new());

/// The CPU may have written `s`'s host copy (a refresh, a copy-back): its
/// twin takes what it wrote before the next submission runs.
pub(crate) fn touched(s: &Arc<Shadow>) {
    if s.twin.is_some() && !s.listed.swap(true, Ordering::Relaxed) {
        TOUCHED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(s.clone());
    }
}

/// Before a submission runs, after [`map_windows`]: records into `list`
/// the copies into twins of what the CPU wrote into host copies since the
/// last ([`touched`]), and of the mirror windows given VRAM, whole.
/// Returns the bytes copied.
pub(crate) fn upload_twins(list: &ID3D12GraphicsCommandList) -> u64 {
    let touched = std::mem::take(&mut *TOUCHED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut m = mirror_state();
    let w = WINDOW as usize;
    let fresh: std::collections::BTreeSet<u64> = std::mem::take(&mut m.fresh).into_iter().collect();
    let mut windows: Vec<(u64, usize, usize)> = fresh.iter().map(|&k| (k, 0, w)).collect();
    let mut bytes = 0;
    for s in &touched {
        s.listed.store(false, Ordering::Relaxed);
        let ranges = std::mem::take(&mut *s.dirty.lock().unwrap_or_else(|e| e.into_inner()));
        let Some(twin) = &s.twin else {
            continue;
        };
        // The copies read and write its buffers when the submission runs,
        // and the title may have released the copy: this may be its last
        // reference.
        if !ranges.is_empty()
            && let Some(host) = host()
        {
            host.retire(s.buffer.clone());
            host.retire(twin.clone());
        }
        let start = s.start as usize;
        for (from, to) in merge(ranges) {
            if s.mirror() {
                let mut at = from;
                while at < to {
                    let next = ((at / w + 1) * w).min(to);
                    let k = (at / w) as u64;
                    if !fresh.contains(&k) {
                        windows.push((k, at % w, next - at + at % w));
                    }
                    at = next;
                }
                continue;
            }
            // SAFETY: an open list; both buffers hold the range.
            unsafe {
                list.CopyBufferRegion(
                    twin,
                    (from - start) as u64,
                    &s.buffer,
                    (from - start) as u64,
                    (to - from) as u64,
                )
            };
            bytes += (to - from) as u64;
        }
    }
    windows.sort_unstable();
    let mut runs: Vec<(u64, usize, usize)> = Vec::with_capacity(windows.len());
    for (k, a, b) in windows {
        match runs.last_mut() {
            Some(r) if r.0 == k && a <= r.2 => r.2 = r.2.max(b),
            _ => runs.push((k, a, b)),
        }
    }
    let mut whole = 0;
    for (k, a, b) in runs {
        let (Some(&(vh, vs, _)), Some(&(h, s))) = (m.twinned.get(&k), m.windows.get(&k)) else {
            continue;
        };
        if fresh.contains(&k) {
            whole += (b - a) as u64;
        }
        // SAFETY: an open list; the pool buffers hold both windows' slots.
        unsafe {
            list.CopyBufferRegion(
                &m.vram[vh].1,
                (vs as usize * w + a) as u64,
                &m.heaps[h].1,
                (s as usize * w + a) as u64,
                (b - a) as u64,
            )
        };
        bytes += (b - a) as u64;
    }
    crate::host::TWINS[5].fetch_add(whole, Ordering::Relaxed);
    bytes
}

/// A copy and the ranges a submission's shaders declared writing in it:
/// title address and bytes.
pub(crate) type Declared = (Arc<Shadow>, Vec<(usize, usize)>);

/// After a submission ran: records into `list` the copies of the ranges
/// its shaders wrote (`back`: copies and the ranges declared, address and
/// bytes) from the twins to the host copies, where copy-back compares
/// them. Returns the bytes copied.
pub(crate) fn download_twins(list: &ID3D12GraphicsCommandList, back: &[Declared]) -> u64 {
    let m = mirror_state();
    let w = WINDOW as usize;
    let mut bytes = 0;
    for (s, declared) in back {
        let Some(twin) = &s.twin else {
            continue;
        };
        let start = s.start as usize;
        for (from, to) in s.merged(declared) {
            if !s.mirror() {
                // SAFETY: an open list; both buffers hold the range.
                unsafe {
                    list.CopyBufferRegion(
                        &s.buffer,
                        from as u64,
                        twin,
                        from as u64,
                        (to - from) as u64,
                    )
                };
                bytes += (to - from) as u64;
                continue;
            }
            let (mut at, end) = (start + from, start + to);
            while at < end {
                let next = ((at / w + 1) * w).min(end);
                let k = (at / w) as u64;
                if let (Some(&(vh, vs, _)), Some(&(h, slot))) =
                    (m.twinned.get(&k), m.windows.get(&k))
                {
                    // SAFETY: an open list; the pool buffers hold both
                    // windows' slots.
                    unsafe {
                        list.CopyBufferRegion(
                            &m.heaps[h].1,
                            (slot as usize * w + at % w) as u64,
                            &m.vram[vh].1,
                            (vs as usize * w + at % w) as u64,
                            (next - at) as u64,
                        )
                    };
                    bytes += (next - at) as u64;
                }
                at = next;
            }
        }
    }
    bytes
}

/// Whether what shaders wrote into twins is found on the GPU: before a
/// submission the twins' ranges its shaders write are kept in VRAM
/// ([`snapshot_twins`]), and after it a compute shader compares them and
/// stores only the words shaders changed into the host copies
/// ([`compare_twins`]), instead of copying the ranges down whole
/// ([`download_twins`]). `VICTORY_NO_TWIN_COMPARE` turns it off.
pub(crate) fn twin_compare_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_TWIN_COMPARE").is_none();
        if !on {
            tracing::info!("what shaders wrote comes down whole (VICTORY_NO_TWIN_COMPARE)");
        }
        on
    })
}

/// Whether copy-back skips the 64 KiB blocks of title memory in which
/// [`compare_twins`] found no word shaders changed: the compare notes each
/// block where one changed ([`unchanged`]). `VICTORY_NO_TWIN_SKIP` turns it
/// off.
pub(crate) fn twin_skip_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_TWIN_SKIP").is_none();
        if !on {
            tracing::info!(
                "copy-back compares every block shaders may write (VICTORY_NO_TWIN_SKIP)"
            );
        }
        on
    })
}

/// Whether [`compare_twins`] also marks, a bit per word, the words shaders
/// changed in twinned copies ([`CHANGED`]), for their copy-back: it writes
/// those words back with no snapshot taken and nothing compared on the CPU,
/// and their GpuWrite steps do not wait for the submissions in flight
/// ([`Shadow::told_by_words`]). The GPU waits instead, before a
/// submission's compares, for the copy-backs in flight that read what they
/// store ([`compared`]). `VICTORY_NO_TWIN_WORDS` turns it off.
pub(crate) fn words_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_TWIN_WORDS").is_none();
        if !on {
            tracing::info!(
                "copy-back compares twinned copies with snapshots (VICTORY_NO_TWIN_WORDS)"
            );
        }
        on
    })
}

/// A twin's range a submission's shaders write, kept before it runs
/// ([`snapshot_twins`]): the GPU addresses of the twin's bytes, of the
/// kept bytes and of the host copy's, and the length, a multiple of 4.
pub(crate) struct Compare {
    twin: u64,
    kept: u64,
    host: u64,
    bytes: u64,
    /// The title address of the range's first byte.
    title: usize,
    /// The copy of `back` the range is in.
    of: usize,
    /// Where the compare notes the range's blocks in [`NOTES`]: the GPU
    /// address and index of the first one's word.
    notes: Option<(u64, usize)>,
    /// Where the compare marks the range's words in [`CHANGED`]: the GPU
    /// address of the slot's bitmap and the bit of the range's first word;
    /// none where copy-back is not told by word or the slot has no bitmap.
    words: Option<(u64, usize)>,
}

/// Where the compare notes the 64 KiB blocks of title memory it found a
/// changed word in, a word per block: in VRAM while it runs (its groups'
/// stores into a word of system memory cost ~25 ms per million), then
/// copied down for copy-back.
struct Notes {
    /// The VRAM buffer and its GPU address.
    vram: ID3D12Resource,
    gpu: u64,
    /// The system memory buffer and its mapping.
    down: ID3D12Resource,
    data: usize,
    /// Words held, and words the submission being prepared uses.
    words: usize,
    used: usize,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is written when
// a submission of its slot is prepared and read at its copy-back.
unsafe impl Send for Notes {}

/// By submission slot ([`preparing`]): a submission's copy-back reads its
/// notes while the next one's compare writes its own.
static NOTES: [Mutex<Option<Notes>>; SLOTS] = [const { Mutex::new(None) }; SLOTS];

/// VRAM the twins' ranges are kept in before a submission: the buffer,
/// its GPU address and size.
struct Keep(Option<(ID3D12Resource, u64, u64)>);

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Keep {}

/// By submission slot, as [`NOTES`].
static KEEP: [Mutex<Keep>; SLOTS] = [const { Mutex::new(Keep(None)) }; SLOTS];

/// Where the compare marks the words shaders changed in twinned copies, a
/// bit per word of the ranges in the order [`KEEP`] holds them
/// ([`words_on`]), in system memory: it stores only the bitmap words not
/// zero, and copy-back clears them ([`Told::clear`]). A world frame
/// compares ~1.3 GB, mostly unchanged: copying its 40 MB of bits down
/// whole cost ~6 ms of GPU time.
struct Changed {
    /// The buffer (alive while mapped), its GPU address and its mapping.
    _buffer: ID3D12Resource,
    gpu: u64,
    data: usize,
    /// Bitmap words held, and those the submission being prepared uses.
    words: usize,
    used: usize,
}

// SAFETY: as [`Notes`].
unsafe impl Send for Changed {}

/// By submission slot, as [`NOTES`].
static CHANGED: [Mutex<Option<Changed>>; SLOTS] = [const { Mutex::new(None) }; SLOTS];

/// The GPU address of [`CHANGED`], made (clear) or grown first if needed
/// to hold `words` bitmap words; none if the host gives no memory for
/// them. Of slot `slot`.
fn changed(words: usize, slot: usize) -> Option<u64> {
    let mut changed = CHANGED[slot].lock().unwrap_or_else(|e| e.into_inner());
    if changed.as_ref().is_none_or(|c| c.words < words) {
        // The slot's last submission has completed: the smaller buffer can
        // go.
        *changed = None;
        let len = (4 * words as u64).next_multiple_of(64 << 10);
        let (buffer, data) = host()?.shared_buffer(len)?;
        crate::host::name(&buffer, || "words the twin compare found changed".into());
        // SAFETY: a live buffer, and its mapping of `len` bytes.
        let gpu = unsafe {
            std::ptr::write_bytes(data, 0, len as usize);
            buffer.GetGPUVirtualAddress()
        };
        *changed = Some(Changed {
            _buffer: buffer,
            gpu,
            data: data as usize,
            words: len as usize / 4,
            used: 0,
        });
    }
    let c = changed.as_mut()?;
    c.used = words;
    Some(c.gpu)
}

/// The host gave no VRAM to keep twins' ranges in: they come down whole.
static NO_KEEP: AtomicBool = AtomicBool::new(false);

/// The most VRAM kept for one submission: past it its ranges come down
/// whole.
const KEEP_MAX: u64 = 1 << 30;

/// Before a submission, after [`upload_twins`]: records into `list` copies
/// within VRAM of the twins' ranges its shaders write (`back`), for
/// [`compare_twins`] after it. None, recording nothing, where the ranges
/// must come down whole ([`download_twins`]): the compare is off or has no
/// pipeline, a range's last word passes its buffer's end, or the ranges
/// pass [`KEEP_MAX`] or the VRAM the host gives. For the submission in
/// `slot` ([`preparing`]).
pub(crate) fn snapshot_twins(
    list: &ID3D12GraphicsCommandList,
    back: &[Declared],
    slot: usize,
) -> Option<Vec<Compare>> {
    if !twin_compare_on() || NO_KEEP.load(Ordering::Relaxed) || compare_pipeline().is_none() {
        return None;
    }
    // The buffer and offset of each range's twin bytes, its twin's and host
    // copy's GPU addresses, its length, title address and copy.
    let mut ranges: Vec<(ID3D12Resource, u64, u64, u64, u64, usize, usize)> = Vec::new();
    let m = mirror_state();
    let w = WINDOW as usize;
    for (of, (s, declared)) in back.iter().enumerate() {
        let Some(twin) = &s.twin else {
            continue;
        };
        let start = s.start as usize;
        for (from, to) in s.merged(declared) {
            let (from, to) = (from & !3, (to + 3) & !3);
            if !s.mirror() {
                // SAFETY: live buffers.
                let (width, gpu, host) = unsafe {
                    (
                        twin.GetDesc().Width,
                        twin.GetGPUVirtualAddress(),
                        s.buffer.GetGPUVirtualAddress(),
                    )
                };
                if to as u64 > width {
                    return None;
                }
                let title = start + from;
                let (from, len) = (from as u64, (to - from) as u64);
                ranges.push((twin.clone(), from, gpu + from, host + from, len, title, of));
                continue;
            }
            let (mut at, end) = (start + from, start + to);
            while at < end {
                let next = ((at / w + 1) * w).min(end);
                let k = (at / w) as u64;
                if let (Some(&(vh, vs, _)), Some(&(h, slot))) =
                    (m.twinned.get(&k), m.windows.get(&k))
                {
                    let (vram, o) = (&m.vram[vh].1, (vs as usize * w + at % w) as u64);
                    let to = (slot as usize * w + at % w) as u64;
                    // SAFETY: live buffers.
                    let (gpu, host) = unsafe {
                        (
                            vram.GetGPUVirtualAddress(),
                            m.heaps[h].1.GetGPUVirtualAddress(),
                        )
                    };
                    let len = (next - at) as u64;
                    ranges.push((vram.clone(), o, gpu + o, host + to, len, at, of));
                }
                at = next;
            }
        }
    }
    drop(m);
    if ranges.is_empty() {
        return Some(Vec::new());
    }
    let need: u64 = ranges.iter().map(|r| r.4.next_multiple_of(256)).sum();
    if ranges.iter().any(|r| r.4 >= 1 << 31) || need > KEEP_MAX {
        return None;
    }
    let mut keep = KEEP[slot].lock().unwrap_or_else(|e| e.into_inner());
    if keep.0.as_ref().is_none_or(|k| k.2 < need) {
        // The slot's last submission has completed: the smaller buffer can
        // go.
        keep.0 = None;
        let size = need.next_multiple_of(64 << 20);
        let Some(buffer) = vram_buffer(size) else {
            NO_KEEP.store(true, Ordering::Relaxed);
            return None;
        };
        crate::host::name(&buffer, || "twins' ranges kept for the compare".into());
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        keep.0 = Some((buffer, gpu, size));
    }
    let (kept, gpu, _) = keep.0.as_ref()?;
    let blocks = |title: usize, bytes: u64| (title + bytes as usize - 1) / PAGE - title / PAGE + 1;
    let words: usize = ranges.iter().map(|r| blocks(r.5, r.4)).sum();
    let notes = twin_skip_on().then(|| notes(list, words, slot)).flatten();
    // A bit per kept word (those of copies not told by word go unused).
    let told = words_on()
        .then(|| changed(need as usize / 128, slot))
        .flatten();
    let (mut at, mut word) = (0, 0);
    let mut compares = Vec::with_capacity(ranges.len());
    for (source, offset, twin, host, bytes, title, of) in ranges {
        // SAFETY: an open list; both buffers hold the range.
        unsafe { list.CopyBufferRegion(kept, at, &source, offset, bytes) };
        compares.push(Compare {
            twin,
            kept: gpu + at,
            host,
            bytes,
            title,
            of,
            notes: notes.map(|gpu| (gpu + 4 * word as u64, word)),
            words: told
                .filter(|_| back[of].0.told_by_words())
                .map(|gpu| (gpu, at as usize / 4)),
        });
        at += bytes.next_multiple_of(256);
        word += blocks(title, bytes);
    }
    Some(compares)
}

/// The GPU address of [`NOTES`] in VRAM, made or grown first if needed,
/// with `words` words cleared by a copy recorded into `list`; none if the
/// host gives no memory for them. Of slot `slot`.
fn notes(list: &ID3D12GraphicsCommandList, words: usize, slot: usize) -> Option<u64> {
    let mut notes = NOTES[slot].lock().unwrap_or_else(|e| e.into_inner());
    if notes.as_ref().is_none_or(|n| n.words < words) {
        // The slot's last submission has completed: the smaller buffers
        // can go.
        *notes = None;
        let len = (4 * words as u64).next_multiple_of(64 << 10);
        let vram = vram_buffer(len)?;
        let (down, data) = host()?.shared_buffer(len)?;
        crate::host::name(&vram, || "blocks the twin compare found changed".into());
        // SAFETY: a live buffer.
        let gpu = unsafe { vram.GetGPUVirtualAddress() };
        *notes = Some(Notes {
            vram,
            gpu,
            down,
            data: data as usize,
            words: len as usize / 4,
            used: 0,
        });
    }
    let n = notes.as_mut()?;
    n.used = words;
    // SAFETY: the mapping holds the words; no submission of the slot is in
    // flight. An open list; both buffers hold them.
    unsafe {
        std::ptr::write_bytes(n.data as *mut u32, 0, words);
        list.CopyBufferRegion(&n.vram, 0, &n.down, 0, 4 * words as u64);
    }
    Some(n.gpu)
}

/// After the submission ran with its compare: by copy of `back` (indexed
/// as [`snapshot_twins`] took it), the 64 KiB blocks of title memory the
/// compare covered and found no word shaders changed in, which copy-back
/// skips. Of the submission in `slot`.
pub(crate) fn unchanged(
    compares: &[Compare],
    copies: usize,
    slot: usize,
) -> Vec<std::collections::HashSet<usize>> {
    let mut covered = vec![std::collections::HashSet::new(); copies];
    let mut changed = vec![std::collections::HashSet::new(); copies];
    let notes = NOTES[slot].lock().unwrap_or_else(|e| e.into_inner());
    let Some(data) = notes.as_ref().map(|n| n.data) else {
        return covered;
    };
    for c in compares {
        let Some((_, word)) = c.notes else {
            continue;
        };
        let first = c.title / PAGE;
        for (k, block) in (first..=(c.title + c.bytes as usize - 1) / PAGE).enumerate() {
            covered[c.of].insert(block);
            // SAFETY: the notes' mapping, which the finished compare wrote.
            if unsafe { (data as *const u32).add(word + k).read_volatile() } != 0 {
                changed[c.of].insert(block);
            }
        }
    }
    for (covered, changed) in covered.iter_mut().zip(&changed) {
        covered.retain(|b| !changed.contains(b));
    }
    covered
}

/// What the compares told copy-back by word of one copy ([`Told::of`]): the
/// bits, and the ranges told (title address, bytes, bit of the first word).
pub(crate) type ToldOf<'a> = (&'a [u32], &'a [(usize, usize, usize)]);

/// What a submission's compares told copy-back by word ([`words_on`]):
/// the slot's bits, held while copy-back reads them, and by copy of `back`
/// the ranges told (title address, bytes, bit of the first word).
pub(crate) struct Told {
    changed: std::sync::MutexGuard<'static, Option<Changed>>,
    ranges: Vec<Vec<(usize, usize, usize)>>,
}

impl Told {
    /// Copy `of` of `back`'s bits and ranges told, if any.
    pub(crate) fn of(&self, of: usize) -> Option<ToldOf<'_>> {
        let ranges = self.ranges.get(of).filter(|r| !r.is_empty())?;
        let c = self.changed.as_ref()?;
        // SAFETY: the mapping holds `used` words, which the finished compare
        // set; the slot's next submission is prepared after this one
        // completes.
        let bits = unsafe { std::slice::from_raw_parts(c.data as *const u32, c.used) };
        Some((bits, ranges))
    }

    /// After copy-back: clears the bits the compares may have set, those of
    /// the title blocks they found a changed word in (by copy, `unchanged`
    /// holds the others), for the slot's next compares.
    pub(crate) fn clear(&self, unchanged: &[std::collections::HashSet<usize>]) {
        let Some(c) = self.changed.as_ref() else {
            return;
        };
        for (of, ranges) in self.ranges.iter().enumerate() {
            for &(title, bytes, bit) in ranges {
                for block in title / PAGE..=(title + bytes - 1) / PAGE {
                    if unchanged.get(of).is_some_and(|u| u.contains(&block)) {
                        continue;
                    }
                    let from = (block * PAGE).max(title) - title;
                    let to = ((block + 1) * PAGE).min(title + bytes) - title;
                    let a = (bit + from / 4) / 32;
                    let b = (bit + to.div_ceil(4)).div_ceil(32).min(c.used);
                    if a < b {
                        // SAFETY: inside the mapping's `used` words; the
                        // compares that set them are done, and copy-back.
                        unsafe { std::ptr::write_bytes((c.data as *mut u32).add(a), 0, b - a) };
                    }
                }
            }
        }
    }
}

/// After the submission ran with its compares (`None`: none ran): what
/// they told copy-back by word, for `copies` copies. Of the submission in
/// `slot`.
pub(crate) fn told(compares: Option<&[Compare]>, copies: usize, slot: usize) -> Told {
    let mut ranges = vec![Vec::new(); copies];
    for c in compares.into_iter().flatten() {
        if let Some((_, bit)) = c.words {
            ranges[c.of].push((c.title, c.bytes as usize, bit));
        }
    }
    Told {
        changed: CHANGED[slot].lock().unwrap_or_else(|e| e.into_inner()),
        ranges,
    }
}

/// The title memory `compares` store into host copies (or title memory
/// used in place), `[from, to)`.
pub(crate) fn compared(compares: &[Compare]) -> Vec<(usize, usize)> {
    compares
        .iter()
        .map(|c| (c.title, c.title + c.bytes as usize))
        .collect()
}

/// At submission, after its compares are recorded: the ranges of copies
/// told by word ([`Shadow::told_by_words`]) that no compare tells (a
/// mirror window without a twin, no VRAM to keep them, no compare list)
/// take the snapshots their GpuWrite steps left, once no submission in
/// flight writes them.
pub(crate) fn snapshot_untold(back: &[Declared], compares: Option<&[Compare]>) {
    let slot = prep_slot();
    for (of, (s, declared)) in back.iter().enumerate() {
        if !s.told_by_words() {
            continue;
        }
        let mut told: Vec<(usize, usize)> = compares
            .into_iter()
            .flatten()
            .filter(|c| c.of == of && c.words.is_some())
            .map(|c| (c.title, c.title + c.bytes as usize))
            .collect();
        told.sort_unstable();
        let untold = |from: usize, to: usize| s.snapshot_after(from, to - from, slot);
        for &(address, len) in declared {
            let (mut at, end) = (address, address + len);
            for &(a, b) in &told {
                if b <= at {
                    continue;
                }
                if a >= end {
                    break;
                }
                if a > at {
                    untold(at, a);
                }
                at = at.max(b);
            }
            if at < end {
                untold(at, end);
            }
        }
    }
}

/// After a submission ran: records into `list` the compares of the twins'
/// ranges [`snapshot_twins`] kept with what the twins hold now, which store
/// the words shaders changed into the host copies, where copy-back compares
/// them. Returns the bytes compared. Of the submission in `slot`.
pub(crate) fn compare_twins(
    list: &ID3D12GraphicsCommandList,
    compares: &[Compare],
    slot: usize,
) -> u64 {
    let Some((root, pipeline)) = compare_pipeline() else {
        return 0;
    };
    if compares.is_empty() {
        return 0;
    }
    // SAFETY: an open list; our root signature and pipeline.
    unsafe {
        list.SetComputeRootSignature(root);
        list.SetPipelineState(pipeline);
    }
    let mut bytes = 0;
    for c in compares {
        let groups = c.bytes.div_ceil(1024).min(65535) as u32;
        let first = (c.title % PAGE) as u32;
        let constants = [
            c.bytes as u32,
            groups,
            first,
            c.notes.is_some().into(),
            c.words.map_or(0, |w| w.1) as u32,
            c.words.is_some().into(),
        ];
        // SAFETY: an open list; the addresses hold the range, the twin's and
        // the kept bytes in VRAM, its blocks' notes and its words' bits
        // (the host copy's, unused, without).
        unsafe {
            list.SetComputeRootShaderResourceView(0, c.twin);
            list.SetComputeRootShaderResourceView(1, c.kept);
            list.SetComputeRootUnorderedAccessView(2, c.host);
            list.SetComputeRootUnorderedAccessView(3, c.notes.map_or(c.host, |n| n.0));
            list.SetComputeRoot32BitConstants(4, 6, constants.as_ptr().cast(), 0);
            list.SetComputeRootUnorderedAccessView(5, c.words.map_or(c.host, |w| w.0));
            list.Dispatch(groups, 1, 1);
        }
        bytes += c.bytes;
    }
    // The notes come down for copy-back.
    if compares.iter().any(|c| c.notes.is_some())
        && let Some(n) = NOTES[slot]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
    {
        // SAFETY: an open list; both buffers hold the words.
        unsafe {
            list.ResourceBarrier(&[crate::host::transition(
                &n.vram,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&n.down, 0, &n.vram, 0, 4 * n.used as u64);
        }
    }
    bytes
}

/// The compare of [`compare_twins`]. A group takes 1 KiB at a time, its
/// threads a word each in turn, and stores the words that changed: the
/// group's stores land on consecutive words of system memory (stores 16
/// bytes apart ran at ~0.7 GB/s, 16-byte stores at ~0.08). With `noting`
/// it notes the 64 KiB blocks of title memory (`first`: the range's offset
/// in its first) the KiB has a changed word in, once per KiB and block.
/// With `wording` it marks each changed word's bit in `words` (`wordbase`:
/// the bit of the range's first word, a multiple of 64), storing the KiB's
/// eight bitmap words, those not zero (copy-back clears them). fxc's `&&`
/// evaluates both sides, so every lane reads `seen` and `bits` in the
/// conditions on them: at its index masked into them, since a lane past
/// them would read past the group's memory, which faults where the driver
/// reports shader errors.
const COMPARE: &str = r#"
#define RS "SRV(t0), SRV(t1), UAV(u0), UAV(u1), RootConstants(num32BitConstants=6, b0), UAV(u2)"
cbuffer Range : register(b0) {
    uint bytes; uint groups; uint first; uint noting; uint wordbase; uint wording;
};
ByteAddressBuffer twin : register(t0);
ByteAddressBuffer kept : register(t1);
RWByteAddressBuffer host : register(u0);
RWByteAddressBuffer notes : register(u1);
RWByteAddressBuffer words : register(u2);
groupshared uint seen[2];
groupshared uint bits[8];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    for (uint base = group.x * 1024; base < bytes; base += groups * 1024) {
        uint block = (first + base) >> 16;
        if (lane < 2) seen[lane] = 0;
        if (lane < 8) bits[lane] = 0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < 4; k++) {
            uint at = base + k * 256 + lane * 4;
            if (at < bytes) {
                uint now = twin.Load(at);
                if (now != kept.Load(at)) {
                    host.Store(at, now);
                    seen[((first + at) >> 16) - block] = 1;
                    InterlockedOr(bits[k * 2 + (lane >> 5)], 1u << (lane & 31));
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (noting != 0 && lane < 2 && seen[lane & 1] != 0) notes.Store((block + lane) * 4, 1);
        if (wording != 0 && lane < 8 && bits[lane & 7] != 0)
            words.Store(((wordbase + (base >> 2)) >> 5) * 4 + lane * 4, bits[lane & 7]);
    }
}
"#;

/// The root signature and pipeline of [`COMPARE`], made on first use.
fn compare_pipeline() -> Option<(&'static ID3D12RootSignature, &'static ID3D12PipelineState)> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    struct Made(ID3D12RootSignature, ID3D12PipelineState);
    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Made {}
    // SAFETY: as above.
    unsafe impl Sync for Made {}
    static MADE: std::sync::OnceLock<Option<Made>> = std::sync::OnceLock::new();
    let made = MADE.get_or_init(|| {
        let host = host()?;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names live through the call.
        let compiled = unsafe {
            D3DCompile(
                COMPARE.as_ptr().cast(),
                COMPARE.len(),
                windows::core::s!("twin compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            let text = errors.map(|b| {
                // SAFETY: the blob's own bytes.
                let bytes = unsafe {
                    std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
                };
                String::from_utf8_lossy(bytes).into_owned()
            });
            tracing::warn!(error = %e, ?text, "no twin compare shader: what shaders wrote comes down whole");
            return None;
        }
        let code = code?;
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a shader with its root signature inside.
        let made = unsafe {
            host.device
                .CreateRootSignature::<ID3D12RootSignature>(0, bytes)
                .and_then(|root| {
                    let pipeline: ID3D12PipelineState = host.device.CreateComputePipelineState(
                        &D3D12_COMPUTE_PIPELINE_STATE_DESC {
                            pRootSignature: std::mem::transmute_copy(&root),
                            CS: D3D12_SHADER_BYTECODE {
                                pShaderBytecode: bytes.as_ptr().cast(),
                                BytecodeLength: bytes.len(),
                            },
                            ..Default::default()
                        },
                    )?;
                    Ok(Made(root, pipeline))
                })
        };
        made.inspect_err(|e| {
            tracing::warn!(error = %e, "no twin compare pipeline: what shaders wrote comes down whole");
        })
        .ok()
    });
    made.as_ref().map(|m| (&m.0, &m.1))
}

/// Ranges sorted and joined where they overlap or meet.
fn merge(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    merged
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
    let (start, len) = (k * SLAB, SLAB + OVERLAP);
    let Some(buffer) = reserved(len) else {
        MIRROR_FAILED.store(true, Ordering::Relaxed);
        tracing::warn!("loose memory goes to windows: no mirror slab");
        return None;
    };
    crate::host::name(&buffer, || {
        format!("title memory mirror {start:#x}+{len:#x}")
    });
    let twin = twins_on().then(|| reserved(len)).flatten();
    if let Some(twin) = &twin {
        crate::host::name(twin, || {
            format!("title memory mirror twin {start:#x}+{len:#x}")
        });
    }
    let mut slab = Shadow::new(buffer, twin, Memory::Mirror, start, len, false);
    slab.loose = true;
    let slab = Arc::new(slab);
    let shown: Vec<u64> = m
        .windows
        .range(start / WINDOW..(start + len) / WINDOW)
        .map(|(&w, _)| w)
        .collect();
    m.unmapped.extend(shown.into_iter().map(|w| (k, w)));
    if slab.twin.is_some() {
        let twinned: Vec<(u64, (usize, u32))> = m
            .twinned
            .range(start / WINDOW..(start + len) / WINDOW)
            .map(|(&w, &(heap, slot, _))| (w, (heap, slot)))
            .collect();
        m.retile
            .extend(twinned.into_iter().map(|(w, slot)| (k, w, Some(slot))));
    }
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
    if twins_on() {
        m.twin_windows();
        map_twins(&mut m, queue);
    }
    if m.unmapped.is_empty() {
        return;
    }
    let mut pairs = std::mem::take(&mut m.unmapped);
    pairs.sort_unstable();
    pairs.dedup();
    let tiles = (WINDOW / TILE) as u32;
    // One call per slab (or its twin, which shows the windows without VRAM
    // of their own as well) and pool heap: the windows' first tiles in each.
    type Call = (Vec<D3D12_TILED_RESOURCE_COORDINATE>, Vec<u32>);
    let mut calls: BTreeMap<(u64, bool, usize), Call> = BTreeMap::new();
    for (k, w) in pairs {
        let Some(&(heap, slot)) = m.windows.get(&w) else {
            continue;
        };
        let twin = m.slabs.get(&k).is_some_and(|s| s.twin.is_some()) && !m.twinned.contains_key(&w);
        for onto in std::iter::once(false).chain(twin.then_some(true)) {
            let call = calls.entry((k, onto, heap)).or_default();
            call.0.push(D3D12_TILED_RESOURCE_COORDINATE {
                X: ((w * WINDOW - k * SLAB) / TILE) as u32,
                ..Default::default()
            });
            call.1.push(slot * tiles);
        }
    }
    for ((k, onto, heap), (at, from)) in calls {
        let slab = &m.slabs[&k];
        let resource = slab.twin.as_ref().filter(|_| onto).unwrap_or(&slab.buffer);
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
                resource,
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

/// The twins' tiles [`Mirror::twin_windows`] noted: each mapped onto its
/// window's VRAM slot, or back onto its host memory.
fn map_twins(m: &mut Mirror, queue: &ID3D12CommandQueue) {
    let tiles = (WINDOW / TILE) as u32;
    for (k, w, slot) in std::mem::take(&mut m.retile) {
        let Some(twin) = m.slabs.get(&k).and_then(|s| s.twin.as_ref()) else {
            continue;
        };
        let (heap, slot) = match (slot, m.windows.get(&w)) {
            (Some((heap, slot)), _) => (&m.vram[heap].0, slot),
            (None, Some(&(heap, slot))) => (&m.heaps[heap].0, slot),
            (None, None) => continue,
        };
        let at = D3D12_TILED_RESOURCE_COORDINATE {
            X: ((w * WINDOW - k * SLAB) / TILE) as u32,
            ..Default::default()
        };
        let size = D3D12_TILE_REGION_SIZE {
            NumTiles: tiles,
            ..Default::default()
        };
        // SAFETY: a reserved buffer and a heap of ours; the region lies in
        // the buffer, the range in the heap.
        unsafe {
            queue.UpdateTileMappings(
                twin,
                1,
                Some(&at),
                Some(&size),
                heap,
                1,
                Some(&D3D12_TILE_RANGE_FLAG_NONE),
                Some(&(slot * tiles)),
                Some(&tiles),
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

/// The programs that read or write the lux-compression buffers, `[L, 1/L,
/// M, 1/M]` (diagnostics, WAL-10): TL_PS_PBR_MATERIAL_STANDARD and its
/// NoDir twin, CS_SkyMain, ps_sun_scattering and CS_ApplyCombinedDeferred
/// read them; SetLuxCompressionInfo and ComputeLuxCompressionInfo write.
#[cfg(feature = "diagnostics")]
const LUX_PROGRAMS: [&str; 7] = [
    "1cd9bc634417a981",
    "48e74d16cbfa45e3",
    "348e1e8c414c7eca",
    "758649b5598455c8",
    "577248728544d74b",
    "9502f5c0e56974af",
    "441d01cd75dc5d5d",
];

/// Title addresses of the buffer views [`LUX_PROGRAMS`] bound on capture
/// frames, and what was logged about them (diagnostics).
#[cfg(feature = "diagnostics")]
#[derive(Default)]
struct Lux {
    watched: std::collections::BTreeSet<u64>,
    logged: std::collections::HashSet<(u64, String, u64, u64)>,
    lines: u32,
}

#[cfg(feature = "diagnostics")]
static LUX: Mutex<Option<Lux>> = Mutex::new(None);

#[cfg(feature = "diagnostics")]
static LUX_ON: AtomicBool = AtomicBool::new(false);

/// Lines the lux probe writes at most.
#[cfg(feature = "diagnostics")]
const LUX_LINES: u32 = 3000;

/// The watched addresses in `[from, to)`.
#[cfg(feature = "diagnostics")]
fn lux_in(watched: &std::collections::BTreeSet<u64>, from: u64, to: u64) -> Vec<u64> {
    watched.range(from..to).copied().collect()
}

/// The four words at a title address as title memory holds them, as floats
/// (diagnostics).
#[cfg(feature = "diagnostics")]
fn title_words(shadow: &Shadow, address: u64) -> Option<[f32; 4]> {
    (shadow.left(address) >= 16).then(|| {
        // SAFETY: title memory inside the buffer the copy is of.
        unsafe { std::ptr::read_unaligned(address as *const [u32; 4]) }.map(f32::from_bits)
    })
}

/// The four words at a title address as the copy holds them (diagnostics).
#[cfg(feature = "diagnostics")]
fn copy_words(shadow: &Shadow, address: u64) -> [Option<f32>; 4] {
    [0, 4, 8, 12].map(|o| shadow.peek(address + o).map(f32::from_bits))
}

/// Diagnostics (WAL-10): on capture frames, logs each buffer view a lux
/// program binds, with the first four words title memory and the copy hold
/// there, and watches its address; on any frame, logs once each unordered
/// view whose range covers a watched address, by program.
#[cfg(feature = "diagnostics")]
pub(crate) fn probe_lux(write: bool, address: u64, len: u64, shadow: &Shadow, own: bool) {
    let frame = crate::queue::presented() + 1;
    let listed = crate::capture::listed(frame);
    if !listed && !(write && LUX_ON.load(Ordering::Relaxed)) {
        return;
    }
    let program = crate::pipeline::host_pipeline(
        crate::descriptor::host::BINDING.get() as crate::object::This
    )
    .map_or("none", |p| p.name.as_str());
    let capture = listed && LUX_PROGRAMS.iter().any(|h| program.contains(h));
    if !capture && !write {
        return;
    }
    let mut lux = LUX.lock().unwrap_or_else(|e| e.into_inner());
    let lux = lux.get_or_insert_with(Lux::default);
    if capture {
        lux.watched.insert(address);
        LUX_ON.store(true, Ordering::Relaxed);
        if lux.lines < LUX_LINES && lux.logged.insert((frame, program.to_owned(), address, len)) {
            lux.lines += 1;
            tracing::info!(
                frame,
                program,
                write,
                address = format_args!("{address:#x}"),
                len,
                copy = format_args!("{:#x}+{:#x}", shadow.start, shadow.len),
                in_place = shadow.in_place,
                twin = shadow.twin.is_some(),
                mirror = shadow.mirror(),
                loose = shadow.loose,
                own,
                title = ?title_words(shadow, address),
                copied = ?copy_words(shadow, address),
                "lux probe: a view"
            );
        }
    }
    if write {
        for at in lux_in(&lux.watched, address, address + len) {
            if lux.lines < LUX_LINES && lux.logged.insert((0, program.to_owned(), address, len)) {
                lux.lines += 1;
                tracing::info!(
                    frame,
                    program,
                    range = format_args!("{address:#x}+{len:#x}"),
                    watched = format_args!("{at:#x}"),
                    "lux probe: shaders are given a range that covers a lux view to write"
                );
            }
        }
    }
}

/// Diagnostics (WAL-10): logs the watched lux addresses a copy-back's
/// declared ranges cover on capture frames, with the words title memory
/// holds there `when` (before or after it), and before it the copy's (after
/// it, the copy's locks are held, so the copy is not read).
#[cfg(feature = "diagnostics")]
fn lux_copy_back(shadow: &Shadow, merged: &[(usize, usize)], when: &str) {
    lux_reads_done();
    if !LUX_ON.load(Ordering::Relaxed) {
        return;
    }
    let frame = crate::queue::presented();
    if !crate::capture::listed(frame) && !crate::capture::listed(frame + 1) {
        return;
    }
    let mut lux = LUX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lux) = lux.as_mut() else {
        return;
    };
    let start = shadow.start;
    for &(from, to) in merged {
        let (a, b) = (start + from as u64, start + to as u64);
        for at in lux_in(&lux.watched, a, b) {
            if lux.lines >= LUX_LINES {
                return;
            }
            lux.lines += 1;
            tracing::info!(
                frame,
                when,
                watched = format_args!("{at:#x}"),
                declared = format_args!("{a:#x}+{:#x}", b - a),
                copy = format_args!("{start:#x}+{:#x}", shadow.len),
                in_place = shadow.in_place,
                twin = shadow.twin.is_some(),
                title = ?title_words(shadow, at),
                copied = ?(when == "before").then(|| copy_words(shadow, at)),
                "lux probe: a copy-back covers a lux view"
            );
        }
    }
}

/// Bytes of a lux program's view read back as the GPU reads it, at most.
#[cfg(feature = "diagnostics")]
const LUX_READ: usize = 4096;

/// Views of lux programs read back at most: per program and capture frame,
/// and in all.
#[cfg(feature = "diagnostics")]
const LUX_READS: (usize, usize) = (12, 1000);

/// A lux program's view read back as the GPU reads it at its draw or
/// dispatch (diagnostics, WAL-10): title memory's words when the list was
/// submitted, the read-back slab's slot the list copied the view's bytes
/// into, and a word set once the list has run.
#[cfg(feature = "diagnostics")]
struct LuxRead {
    frame: u64,
    program: String,
    constant: bool,
    address: u64,
    len: usize,
    path: &'static str,
    bound: u64,
    title: Arc<Mutex<Vec<u32>>>,
    slot: usize,
    ran: Box<std::sync::atomic::AtomicU64>,
    _source: ID3D12Resource,
}

#[cfg(feature = "diagnostics")]
#[derive(Default)]
struct LuxReads {
    /// The read-back slab ([`LUX_READ`] bytes per slot) and its mapping.
    slab: Option<(ID3D12Resource, usize)>,
    reads: Vec<LuxRead>,
    taken: usize,
    per_program: std::collections::HashMap<(u64, String), usize>,
}

#[cfg(feature = "diagnostics")]
static LUX_READS_STATE: Mutex<Option<LuxReads>> = Mutex::new(None);

#[cfg(feature = "diagnostics")]
static LUX_READS_PENDING: AtomicBool = AtomicBool::new(false);

/// Diagnostics (WAL-10): on capture frames, has the GPU copy the bytes a
/// lux program's view reads into a read-back slab where its draw or
/// dispatch reads them, from the memory the view is made on (`own`: the
/// length [`crate::descriptor::host::own_copy_of`] gave its own host copy;
/// else the copy's twin, the copy, or title memory in place), and keeps
/// title memory's words at submission, for [`lux_reads_done`] to log.
#[cfg(feature = "diagnostics")]
pub(crate) fn lux_read(
    list: &HostList,
    constant: bool,
    address: u64,
    len: u64,
    shadow: &Shadow,
    own: Option<u64>,
) {
    let frame = crate::queue::presented() + 1;
    if !crate::capture::listed(frame) {
        return;
    }
    let program = crate::pipeline::host_pipeline(
        crate::descriptor::host::BINDING.get() as crate::object::This
    )
    .map_or("none", |p| p.name.as_str());
    if !LUX_PROGRAMS.iter().any(|h| program.contains(h)) {
        return;
    }
    let (source, offset, bound, path) = match own {
        Some(key) => {
            let Some((buffer, offset, gpu)) = crate::descriptor::host::own_place(address, key)
            else {
                return;
            };
            (buffer, offset, gpu, "own")
        }
        None if shadow.mirror() => return,
        None => (
            shadow.resource().clone(),
            shadow.offset(address),
            shadow.gpu(address),
            match (shadow.twin.is_some(), shadow.in_place) {
                (true, _) => "twin",
                (false, true) => "in_place",
                (false, false) => "copy",
            },
        ),
    };
    let n = (len.min(shadow.left(address)) as usize).min(LUX_READ) & !3;
    if n == 0 {
        return;
    }
    let mut state = LUX_READS_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = state.get_or_insert_with(LuxReads::default);
    if s.taken == LUX_READS.1 {
        return;
    }
    let count = s
        .per_program
        .entry((frame, program.to_owned()))
        .or_default();
    if *count == LUX_READS.0 {
        return;
    }
    if s.slab.is_none() {
        s.slab = host()
            .and_then(|h| h.shared_buffer((LUX_READ * LUX_READS.1) as u64))
            .map(|(buffer, data)| (buffer, data as usize));
    }
    let Some((slab, _)) = &s.slab else {
        return;
    };
    *count += 1;
    let slot = s.taken;
    s.taken += 1;
    let title = Arc::new(Mutex::new(Vec::new()));
    let keep = title.clone();
    list.defer_host(move || {
        // SAFETY: title memory inside the buffer the copy is of.
        let words = (0..n / 4)
            .map(|k| unsafe { ((address + 4 * k as u64) as *const u32).read_unaligned() })
            .collect();
        *keep.lock().unwrap_or_else(|e| e.into_inner()) = words;
    });
    // SAFETY: an open list; the slot lies in the slab, the bytes in the
    // view's memory.
    unsafe {
        list.list
            .CopyBufferRegion(slab, (slot * LUX_READ) as u64, &source, offset, n as u64)
    };
    let ran = Box::new(std::sync::atomic::AtomicU64::new(0));
    list.defer_write(ran.as_ptr().cast(), crate::queue::Written::Value(1), 8);
    s.reads.push(LuxRead {
        frame,
        program: program.to_owned(),
        constant,
        address,
        len: n,
        path,
        bound,
        title,
        slot,
        ran,
        _source: source,
    });
    LUX_READS_PENDING.store(true, Ordering::Relaxed);
}

/// What the GPU read through a lux program's view, against title memory at
/// submission (diagnostics): words of each that are not zero, words that
/// differ, the first differences (byte offset, title's word, GPU's word)
/// and the first four words of each as floats.
#[cfg(feature = "diagnostics")]
type LuxCompare = (
    usize,
    usize,
    usize,
    Vec<(usize, u32, u32)>,
    [f32; 4],
    [f32; 4],
);

#[cfg(feature = "diagnostics")]
fn lux_compare(title: &[u32], gpu: &[u32]) -> LuxCompare {
    let nonzero = |w: &[u32]| w.iter().filter(|&&w| w != 0).count();
    let differ: Vec<(usize, u32, u32)> = title
        .iter()
        .zip(gpu)
        .enumerate()
        .filter(|(_, (t, g))| t != g)
        .map(|(k, (&t, &g))| (4 * k, t, g))
        .collect();
    let head =
        |w: &[u32]| std::array::from_fn(|k| w.get(k).map_or(f32::NAN, |&w| f32::from_bits(w)));
    (
        nonzero(title),
        nonzero(gpu),
        differ.len(),
        differ.into_iter().take(8).collect(),
        head(title),
        head(gpu),
    )
}

/// Diagnostics (WAL-10): logs the lux programs' views read back by lists
/// that have run ([`lux_read`]).
#[cfg(feature = "diagnostics")]
fn lux_reads_done() {
    if !LUX_READS_PENDING.load(Ordering::Relaxed) {
        return;
    }
    let mut state = LUX_READS_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = state.as_mut() else {
        return;
    };
    let Some(data) = s.slab.as_ref().map(|s| s.1) else {
        return;
    };
    let (done, waiting): (Vec<LuxRead>, Vec<LuxRead>) = std::mem::take(&mut s.reads)
        .into_iter()
        .partition(|r| r.ran.load(Ordering::Acquire) != 0);
    s.reads = waiting;
    LUX_READS_PENDING.store(!s.reads.is_empty(), Ordering::Relaxed);
    for r in done {
        // SAFETY: the slot's bytes, which the list that has run copied.
        let gpu = unsafe {
            std::slice::from_raw_parts((data + r.slot * LUX_READ) as *const u32, r.len / 4)
        };
        let title = r.title.lock().unwrap_or_else(|e| e.into_inner());
        let (title_nonzero, gpu_nonzero, differ, first, title_head, gpu_head) =
            lux_compare(&title, gpu);
        let first: Vec<String> = first
            .iter()
            .map(|(at, t, g)| format!("+{at:#x}: {t:#010x}/{g:#010x}"))
            .collect();
        tracing::info!(
            frame = r.frame,
            program = r.program,
            constant = r.constant,
            address = format_args!("{:#x}", r.address),
            bytes = r.len,
            path = r.path,
            bound = format_args!("{:#x}", r.bound),
            submitted = !title.is_empty(),
            title_nonzero,
            gpu_nonzero,
            differ,
            ?first,
            ?title_head,
            ?gpu_head,
            "lux probe: what the GPU read"
        );
    }
}

/// Constant views of lux programs' fresh tables logged at most: per
/// program and capture frame, and in all (diagnostics).
#[cfg(feature = "diagnostics")]
const LUX_TABLE_CONSTANTS: (usize, usize) = (16, 400);

#[cfg(feature = "diagnostics")]
static LUX_TABLE_LOGGED: Mutex<Option<std::collections::HashMap<(u64, String), usize>>> =
    Mutex::new(None);

/// Diagnostics (WAL-10): on capture frames, logs at submission each
/// constant view of a lux program's fresh table, which the list copies
/// into its own memory at `to` from title memory at `address` just
/// before (the view shaders read is made there): its register, the words
/// title memory and the copy hold, and the copy's words, up to
/// [`LUX_READ`] bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn lux_table_constant(
    list: &HostList,
    register: Option<u32>,
    address: u64,
    size: u32,
    to: *mut u8,
) {
    let frame = crate::queue::presented() + 1;
    if !crate::capture::listed(frame) {
        return;
    }
    let program = crate::pipeline::host_pipeline(
        crate::descriptor::host::BINDING.get() as crate::object::This
    )
    .map_or("none", |p| p.name.as_str());
    if !LUX_PROGRAMS.iter().any(|h| program.contains(h)) {
        return;
    }
    {
        let mut logged = LUX_TABLE_LOGGED.lock().unwrap_or_else(|e| e.into_inner());
        let logged = logged.get_or_insert_with(Default::default);
        if logged.values().sum::<usize>() >= LUX_TABLE_CONSTANTS.1 {
            return;
        }
        let count = logged.entry((frame, program.to_owned())).or_default();
        if *count >= LUX_TABLE_CONSTANTS.0 {
            return;
        }
        *count += 1;
    }
    let n = (size as usize).min(LUX_READ) / 4;
    let (program, to) = (program.to_owned(), to as usize);
    list.defer_host(move || {
        // SAFETY: title memory the view covers, and the list's copy of it,
        // filled just before.
        let read = |at: usize| unsafe { (at as *const u32).read_unaligned() };
        let title: Vec<u32> = (0..n).map(|k| read(address as usize + 4 * k)).collect();
        let copy: Vec<u32> = (0..n).map(|k| read(to + 4 * k)).collect();
        let (title_nonzero, copy_nonzero, differ, first, ..) = lux_compare(&title, &copy);
        let words: Vec<String> = copy.iter().map(|w| format!("{w:08x}")).collect();
        tracing::info!(
            frame,
            program,
            register,
            address = format_args!("{address:#x}"),
            size,
            title_nonzero,
            copy_nonzero,
            differ,
            ?first,
            words = words.join(" "),
            "lux probe: a table's constant view at submission"
        );
    });
}

/// Diagnostics (WAL-10): on capture frames, logs each constant view a lux
/// program has made: its title address and size and where the host view
/// is made (`own`: on its own host copy).
#[cfg(feature = "diagnostics")]
pub(crate) fn lux_constant_made(address: u64, size: u32, location: u64, own: bool) {
    let frame = crate::queue::presented() + 1;
    if !crate::capture::listed(frame) {
        return;
    }
    let program = crate::pipeline::host_pipeline(
        crate::descriptor::host::BINDING.get() as crate::object::This
    )
    .map_or("none", |p| p.name.as_str());
    if !LUX_PROGRAMS.iter().any(|h| program.contains(h)) {
        return;
    }
    let mut lux = LUX.lock().unwrap_or_else(|e| e.into_inner());
    let lux = lux.get_or_insert_with(Lux::default);
    if lux.lines < LUX_LINES
        && lux
            .logged
            .insert((frame, format!("made {program}"), address, u64::from(size)))
    {
        lux.lines += 1;
        tracing::info!(
            frame,
            program,
            address = format_args!("{address:#x}"),
            size,
            location = format_args!("{location:#x}"),
            own,
            "lux probe: a constant view is made"
        );
    }
}

/// What shaders were given to read or to write (diagnostics): views,
/// bytes they reach in VRAM (twins) and in system memory (copies, views'
/// own copies and mirror windows without a VRAM twin), bytes of title
/// memory used in place (system memory, never twinned), and the mirror
/// windows reached in VRAM and in system memory.
#[cfg(feature = "diagnostics")]
#[derive(Default)]
pub(crate) struct Reach {
    views: u64,
    vram: u64,
    system: u64,
    in_place: u64,
    vram_windows: std::collections::BTreeSet<u64>,
    system_windows: std::collections::BTreeSet<u64>,
}

#[cfg(feature = "diagnostics")]
impl Reach {
    fn add(&mut self, r: Reach) {
        self.views += r.views;
        self.vram += r.vram;
        self.system += r.system;
        self.in_place += r.in_place;
        self.vram_windows.extend(r.vram_windows);
        self.system_windows.extend(r.system_windows);
    }

    /// Per frame over `frames` frames, with the first mirror windows it
    /// reaches in system memory.
    pub(crate) fn per_frame(&self, frames: f64) -> String {
        if self.views == 0 {
            return "nothing".to_owned();
        }
        let windows: Vec<String> = self
            .system_windows
            .iter()
            .take(8)
            .map(|n| format!("{:#x}", n * WINDOW))
            .collect();
        format!(
            "{:.1} views: {:.1} MB in VRAM, {:.1} MB in system memory, {:.1} MB in title memory used in \
             place; {} windows in system memory: {}",
            self.views as f64 / frames,
            self.vram as f64 / frames / 1e6,
            self.system as f64 / frames / 1e6,
            self.in_place as f64 / frames / 1e6,
            self.system_windows.len(),
            windows.join(" "),
        )
    }
}

/// What shaders were given to read (first) and to write since the GPU time
/// lines, by title pipeline (diagnostics).
#[cfg(feature = "diagnostics")]
static REACHED: Mutex<BTreeMap<usize, [Reach; 2]>> = Mutex::new(BTreeMap::new());

/// Takes what shaders were given to read (first) and to write, by host
/// pipeline name as the GPU time lines show it (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_reached() -> std::collections::HashMap<String, [Reach; 2]> {
    let reached = std::mem::take(&mut *REACHED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut named: std::collections::HashMap<String, [Reach; 2]> = Default::default();
    for (pipeline, [reads, writes]) in reached {
        let name = crate::pipeline::host_pipeline(pipeline as crate::object::This)
            .map_or_else(|| format!("{pipeline:#x}"), |p| p.name.clone());
        let e = named.entry(name).or_default();
        e[0].add(reads);
        e[1].add(writes);
    }
    named
}

/// Logs the mirror's twins against the windows submissions used and the
/// windows shaders were given to write (diagnostics).
#[cfg(feature = "diagnostics")]
fn log_written() {
    let (placed, twinned, heaps, used, used_system) = {
        let mut m = mirror_state();
        let used = std::mem::take(&mut m.hot);
        let system = used
            .iter()
            .filter(|w| !m.twinned.contains_key(w) && m.windows.contains_key(w))
            .count();
        (
            m.windows.len(),
            m.twinned.len(),
            m.vram.len(),
            used.len(),
            system,
        )
    };
    let (mut vram_windows, mut system_windows) = (
        std::collections::BTreeSet::<u64>::new(),
        std::collections::BTreeSet::<u64>::new(),
    );
    for [_, w] in REACHED.lock().unwrap_or_else(|e| e.into_inner()).values() {
        vram_windows.extend(&w.vram_windows);
        system_windows.extend(&w.system_windows);
    }
    let mib = WINDOW >> 20;
    tracing::info!(
        "mirror twins: {placed} windows have host memory, {twinned} VRAM ({} MiB in {heaps} heaps; budget \
         {} MiB); submissions used {used} windows ({} MiB) since the last line, {used_system} of them \
         in system memory; shaders were given {} windows in VRAM and {} in system memory to write",
        twinned as u64 * mib,
        twin_budget() as u64 * mib,
        used as u64 * mib,
        vram_windows.len(),
        system_windows.len(),
    );
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
    #[cfg(feature = "diagnostics")]
    log_written();
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

    /// [`merge_words`] by lanes gives what merging word by word gave (title and
    /// copy bytes, the GPU's runs, the span taken) on random pieces of every
    /// length up to a few lanes, the GPU and the title writing little, much
    /// or everything.
    #[test]
    fn merging_by_lanes_matches_merging_by_words() {
        fn by_words(title: &mut [u8], host: &mut [u8], old: &[u8]) -> Merged {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut taken: Option<(usize, usize)> = None;
            let words = title
                .chunks_mut(4)
                .zip(host.chunks_mut(4))
                .zip(old.chunks(4));
            for (k, ((t, h), o)) in words.enumerate() {
                let mut gpu = false;
                for ((t, h), o) in t.iter_mut().zip(h.iter()).zip(o) {
                    if *h != *o && *t != *h {
                        *t = *h;
                        gpu = true;
                    }
                }
                let (a, b) = (4 * k, 4 * k + t.len());
                if *h != *t {
                    taken = Some((taken.map_or(a, |s| s.0), b));
                }
                h.copy_from_slice(t);
                match runs.last_mut() {
                    Some(run) if gpu && run.1 == a => run.1 = b,
                    _ if gpu => runs.push((a, b)),
                    _ => {}
                }
            }
            (runs, taken)
        }
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // One byte in `k` (never for 0).
        let mut one_in = |k: u64| k != 0 && next() % k == 0;
        let odds = [0, 1, 2, 7, 64];
        for round in 0..20_000 {
            let n = round % 80 + if round % 3 == 0 { 0 } else { 80 };
            // How often the GPU writes a byte, the title already holds the
            // GPU's byte, the title writes one of its own.
            let (gpu, holds, writes) = (odds[round % 5], odds[round / 5 % 5], odds[round / 25 % 5]);
            let old: Vec<u8> = (0..n).map(|i| (i * 31 + round) as u8).collect();
            let host: Vec<u8> = old
                .iter()
                .map(|&o| {
                    if one_in(gpu) {
                        o ^ (1 + (round % 255) as u8)
                    } else {
                        o
                    }
                })
                .collect();
            let title: Vec<u8> = (0..n)
                .map(|i| {
                    if one_in(holds) {
                        host[i]
                    } else if one_in(writes) {
                        old[i].wrapping_add(3)
                    } else {
                        old[i]
                    }
                })
                .collect();
            let (mut t1, mut h1) = (title.clone(), host.clone());
            let (mut t2, mut h2) = (title, host);
            let lanes = merge_words(&mut t1, &mut h1, &old);
            let words = by_words(&mut t2, &mut h2, &old);
            assert_eq!(lanes, words, "round {round}");
            assert_eq!((t1, h1), (t2, h2), "round {round}");
        }
    }

    #[test]
    fn merging_bits_by_lanes_matches_merging_them_by_words() {
        fn by_words(
            title: &mut [u8],
            host: &mut [u8],
            bits: &[u32],
            first: usize,
            skip: usize,
        ) -> Merged {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut taken: Option<(usize, usize)> = None;
            let len = title.len();
            let edge = |w: usize| (4 * w).saturating_sub(skip).min(len);
            for w in 0..(len + skip).div_ceil(4) {
                let (a, b) = (edge(w), edge(w + 1));
                let (t, h) = (&mut title[a..b], &mut host[a..b]);
                if t == h {
                    continue;
                }
                if bits_at(bits, first + w) & 1 != 0 {
                    t.copy_from_slice(h);
                    match runs.last_mut() {
                        Some(run) if run.1 == a => run.1 = b,
                        _ => runs.push((a, b)),
                    }
                } else {
                    h.copy_from_slice(t);
                    taken = Some((taken.map_or(a, |s| s.0), b));
                }
            }
            (runs, taken)
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let odds = [0, 1, 2, 7, 64];
        for round in 0..20_000 {
            let n = [0, 3, 127, 128, 129, 300, 512, 1021][round % 8];
            let skip = round / 8 % 4;
            // How often a word's bit is set, the host copy differs, the
            // title holds the host copy's word.
            let (set, differs, holds) = (
                odds[round / 32 % 5],
                odds[round / 160 % 5],
                odds[round / 800 % 5],
            );
            let mut one_in = |k: u64| k != 0 && next() % k == 0;
            let first = round % 37;
            let bits: Vec<u32> = (0..(first + n / 4 + 2).div_ceil(32))
                .map(|_| (0..32).fold(0, |m, b| m | u32::from(one_in(set)) << b))
                .collect();
            let title: Vec<u8> = (0..n).map(|i| (i * 31 + round) as u8).collect();
            let host: Vec<u8> = title
                .iter()
                .map(|&t| if one_in(differs) { t ^ 0x5a } else { t })
                .collect();
            let title: Vec<u8> = title
                .iter()
                .zip(&host)
                .map(|(&t, &h)| if one_in(holds) { h } else { t })
                .collect();
            let (mut t1, mut h1) = (title.clone(), host.clone());
            let (mut t2, mut h2) = (title, host);
            let lanes = merge_bits(&mut t1, &mut h1, &bits, first, skip);
            let words = by_words(&mut t2, &mut h2, &bits, first, skip);
            assert_eq!(lanes, words, "round {round}");
            assert_eq!((t1, h1), (t2, h2), "round {round}");
        }
    }

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

    /// What a pipeline's views give its shaders to read and to write counts
    /// by where they reach it.
    #[test]
    #[cfg(feature = "diagnostics")]
    fn shader_reads_and_writes_count_where_shaders_reach_them_by_pipeline() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        // Write-watched memory: the buffer is copied, with a VRAM twin if
        // the host gave one.
        let len = 8 * MIB;
        let at = title_memory(510, 256 * MIB, len, 4 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 11, at as u64, len as u64);
        let s = find(at as u64, len as u64).unwrap();
        assert!(!s.in_place && !s.mirror());
        let pipeline = 0x5eed_0009;
        crate::descriptor::host::BINDING.set(pipeline);
        s.note_reached(at as u64, 2 * MIB as u64, true, false);
        s.note_reached(at as u64 + MIB as u64, MIB as u64, true, false);
        // A read past the buffer's end counts to its end; a view's own copy
        // is system memory.
        s.note_reached(at as u64 + 7 * MIB as u64, 2 * MIB as u64, false, false);
        s.note_reached(at as u64, 256, false, true);
        crate::descriptor::host::BINDING.set(0);
        let [reads, writes] = REACHED.lock().unwrap().remove(&pipeline).unwrap();
        let twin = s.twin.is_some();
        let (reached, other) = if twin {
            (writes.vram, writes.system)
        } else {
            (writes.system, writes.vram)
        };
        assert_eq!(
            (writes.views, reached, other, writes.in_place),
            (2, 3 * MIB as u64, 0, 0)
        );
        let own = if twin { 0 } else { MIB as u64 };
        assert_eq!(
            (
                reads.views,
                reads.vram + reads.system,
                reads.system,
                reads.in_place
            ),
            (2, MIB as u64 + 256, own + 256, 0)
        );
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
            writing(&s, a, n);
        }
        let spans: Vec<(usize, usize)> = {
            let mut spans: Vec<_> = s.before[0]
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
        assert_eq!(
            s.copy_back(&declared, &Default::default(), None, 0),
            2 * (64 << 10)
        );
        for w in written_at {
            // SAFETY: our test memory.
            assert_eq!(unsafe { (w as *const u32).read() }, 0xfeed);
            // SAFETY: as above.
            assert_eq!(unsafe { ((w + 4) as *const u32).read() }, word(w + 4));
        }
        assert!(s.before[0].lock().unwrap().is_empty());
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
        writing(&s, at + MIB, 32 * MIB);
        assert!(s.before[0].lock().unwrap().is_empty(), "no snapshot");
        let kept: Vec<usize> = s.kept[0].lock().unwrap().keys().copied().collect();
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
        assert_eq!(
            s.copy_back(&declared, &Default::default(), None, 0),
            4 * (64 << 10)
        );
        crate::watch::disarm(0);
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
        assert!(s.kept[0].lock().unwrap().is_empty() && s.armed[0].lock().unwrap().is_empty());
        drop(s);
        unregister(usize::MAX - 8, at as u64);
        unmap(&made);
    }

    /// A refresh of a range brought up to date wholly guarded and clean
    /// asks nothing while title memory there is unchanged; a title write in
    /// it, our write to it, or a remapping is taken at the next refresh.
    #[test]
    fn unchanged_guarded_ranges_are_not_asked_about_again() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = (968 * SLAB) as usize - 256 * MIB;
        // Two views: a copy, not title memory used in place.
        let made = views(at, &[4 * MIB, 4 * MIB]);
        assert_eq!(made.len(), 2, "no free address for test views");
        register(usize::MAX - 10, at as u64, 8 << 20);
        let s = find(at as u64, 8 << 20).unwrap();
        assert!(!s.in_place);
        // SAFETY: inside the copy's memory.
        let host = |w: usize| unsafe { s.host_at(w - at, 4).cast::<u32>().read() };
        crate::watch::forget_asked();
        s.update(at, 8 * MIB);
        let asked = crate::watch::asking();
        crate::watch::forget_asked();
        assert_eq!(s.update(at, 8 * MIB), 0);
        assert_eq!(
            crate::watch::asking(),
            asked,
            "asked about an unchanged range"
        );
        let w = at + 3 * MIB + 8;
        // SAFETY: our view (guarded: our handler opens the block).
        unsafe { (w as *mut u32).write(0x5eed) };
        crate::watch::forget_asked();
        assert_eq!(s.update(at, 8 * MIB), PAGE);
        assert_eq!(host(w), 0x5eed);
        crate::watch::forget_asked();
        assert_eq!(s.update(at, 8 * MIB), 0);
        let asked = crate::watch::asking();
        // Our write (shaders in place, a copy into title memory).
        crate::watch::wrote(at + 5 * MIB, at + 5 * MIB + 4);
        crate::watch::forget_asked();
        s.update(at, 8 * MIB);
        assert!(
            crate::watch::asking() > asked + 1,
            "not asked again after our write"
        );
        drop(s);
        unregister(usize::MAX - 10, at as u64);
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
        writing(&s, x, len / 2);
        // While the GPU runs, the title writes a word through the other
        // address, and the GPU another of the same block.
        let (title, gpu) = (x + PAGE + 8, x + PAGE + 64);
        // SAFETY: our view of the section, and inside the copy's memory.
        unsafe {
            ((title - x + y) as *mut u32).write(7);
            s.host_at(gpu - x, 4).cast::<u32>().write(0xfeed);
        }
        s.copy_back(&declared, &Default::default(), None, 0);
        crate::watch::disarm(0);
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
        writing(&a, at, 8 * MIB);
        write(x, 1);
        crate::watch::forget_asked();
        b.update(at + 4 * MIB, 8 * MIB);
        writing(&b, at + 4 * MIB, 8 * MIB);
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
        a.copy_back(&da, &Default::default(), None, 0);
        crate::watch::forget_asked();
        b.copy_back(&db, &Default::default(), None, 0);
        crate::watch::disarm(0);
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

    /// The title writes a block while copy-back runs, after the copy took
    /// the block's bytes and before it asks the guards: the copy does not
    /// count as current there, so the next refresh takes the write, and the
    /// next copy-back keeps it rather than put the copy's older bytes over
    /// it (run 67's audit: blocks armed clean, unsaved, with title memory
    /// unlike the copy).
    #[test]
    fn a_title_write_during_copy_back_reaches_the_copy() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = (965 * SLAB) as usize - 256 * MIB;
        // Copies across views: not used in place.
        let made = views(at, &[6 * MIB; 4]);
        assert_eq!(made.len(), 4, "no free address for test views");
        register(usize::MAX - 15, at as u64, (16 * MIB) as u64);
        let s = find(at as u64, (16 * MIB) as u64).unwrap();
        assert!(!s.in_place);
        let declared = [(at, 16 * MIB)];
        let (x, y) = (at + 5 * MIB + 64, at + 9 * MIB);
        // SAFETY: our views (guarded: our handler opens the block).
        let read = |w: usize| unsafe { (w as *const u32).read() };
        // The GPU writes x; the title writes next to it while copy-back runs.
        crate::watch::forget_asked();
        s.update(at, 16 * MIB);
        writing(&s, at, 16 * MIB);
        // SAFETY: inside the copy's memory.
        unsafe { s.host_at(x - at, 4).cast::<u32>().write(0xfeed) };
        // SAFETY: as above.
        *BEFORE_ASKING.lock().unwrap() =
            Some(Box::new(move || unsafe { ((x + 8) as *mut u32).write(7) }));
        crate::watch::forget_asked();
        s.copy_back(&declared, &Default::default(), None, 0);
        crate::watch::disarm(0);
        // The next submission: the GPU writes elsewhere.
        crate::watch::forget_asked();
        s.update(at, 16 * MIB);
        assert_eq!(
            held(&s, x + 8),
            Some(7),
            "the copy missed the title's write"
        );
        writing(&s, at, 16 * MIB);
        // SAFETY: inside the copy's memory.
        unsafe { s.host_at(y - at, 4).cast::<u32>().write(0xbeef) };
        crate::watch::forget_asked();
        s.copy_back(&declared, &Default::default(), None, 0);
        crate::watch::disarm(0);
        assert_eq!(read(x + 8), 7, "the title's write was undone");
        assert_eq!((read(x), read(y)), (0xfeed, 0xbeef));
        drop(s);
        unregister(usize::MAX - 15, at as u64);
        unmap(&made);
    }

    /// Pipelined, a submission is copied back while the next is prepared:
    /// each keeps its own armed blocks, so the first one's copy-back leaves
    /// the next one's to its own, which puts back what its shaders wrote and
    /// keeps the title's write made beside it while both were in flight.
    #[test]
    fn copy_back_leaves_the_next_submissions_state() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = (967 * SLAB) as usize - 256 * MIB;
        // Copies across views: not used in place.
        let made = views(at, &[6 * MIB; 4]);
        assert_eq!(made.len(), 4, "no free address for test views");
        register(usize::MAX - 24, at as u64, (16 * MIB) as u64);
        let s = find(at as u64, (16 * MIB) as u64).unwrap();
        assert!(!s.in_place);
        let (x, y) = (at + MIB + 64, at + 12 * MIB + 64);
        // SAFETY: our views (guarded: our handler opens the block).
        let read = |w: usize| unsafe { (w as *const u32).read() };
        crate::watch::forget_asked();
        preparing(0);
        s.update(at, 8 * MIB);
        writing(&s, at, 8 * MIB);
        // SAFETY: inside the copy's memory.
        unsafe { s.host_at(x - at, 4).cast::<u32>().write(0xfeed) };
        crate::watch::forget_asked();
        preparing(1);
        s.update(at + 8 * MIB, 8 * MIB);
        writing(&s, at + 8 * MIB, 8 * MIB);
        // SAFETY: as above.
        unsafe { s.host_at(y - at, 4).cast::<u32>().write(0xbeef) };
        // SAFETY: our views.
        unsafe { ((y + 8) as *mut u32).write(7) };
        s.copy_back(&[(at, 8 * MIB)], &Default::default(), None, 0);
        crate::watch::disarm(0);
        s.copy_back(&[(at + 8 * MIB, 8 * MIB)], &Default::default(), None, 1);
        crate::watch::disarm(1);
        preparing(0);
        assert_eq!((read(x), read(y), read(y + 8)), (0xfeed, 0xbeef, 7));
        drop(s);
        unregister(usize::MAX - 24, at as u64);
        unmap(&made);
    }

    /// Pipelined: a refresh waits for a submission in flight only to copy
    /// pages its shaders write, not for every range that overlaps them;
    /// deferred (`crate::pipelined::deferred_refreshes`), it does not wait,
    /// and those pages are copied once the submission's copy-back is done.
    #[test]
    fn a_refresh_waits_only_to_copy_pages_shaders_in_flight_write() {
        use std::time::Duration;
        /// Lets the held submission complete, also when the test fails.
        struct Go(std::sync::mpsc::Sender<()>);
        impl Drop for Go {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let _serial = crate::watch::serial();
        if !gpu() || !crate::pipelined::on() {
            return;
        }
        // Write-watched memory, its last page not committed: copied.
        let len = 4 * MIB + (64 << 10);
        let at = title_memory(1200, 256 * MIB, len, 4 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 25, at as u64, len as u64);
        let s = find(at as u64, len as u64).unwrap();
        assert!(!s.in_place);
        crate::watch::forget_asked();
        s.update(at, 4 * MIB);
        // A submission in flight whose shaders write the second MiB.
        let (go, wait) = std::sync::mpsc::channel::<()>();
        crate::pipelined::launch(
            vec![(at + MIB, at + 2 * MIB)],
            Vec::new(),
            Vec::new(),
            Box::new(move |_| {
                let _ = wait.recv();
            }),
        );
        let go = Go(go);
        let refresh = || {
            let (s, (tx, rx)) = (s.clone(), std::sync::mpsc::channel());
            std::thread::spawn(move || {
                crate::watch::forget_asked();
                let _ = tx.send(s.update(at, 4 * MIB));
            });
            rx
        };
        let (x, y) = (at + 3 * MIB + 8, at + MIB + 8);
        // SAFETY: committed title memory of ours.
        unsafe { (x as *mut u32).write(0xfeed) };
        let copied = refresh()
            .recv_timeout(Duration::from_secs(5))
            .expect("waited for the submission in flight");
        assert_eq!(copied, PAGE);
        assert_eq!(held(&s, x), Some(0xfeed));
        // SAFETY: as above.
        unsafe { (y as *mut u32).write(0xbeef) };
        let done = refresh();
        if crate::pipelined::deferred_refreshes() {
            let copied = done
                .recv_timeout(Duration::from_secs(5))
                .expect("waited for the submission in flight");
            assert_eq!(copied, PAGE);
            std::thread::sleep(Duration::from_millis(300));
            assert_ne!(
                held(&s, y),
                Some(0xbeef),
                "copied a page the submission in flight writes"
            );
            drop(go);
            let started = std::time::Instant::now();
            while held(&s, y) != Some(0xbeef) {
                assert!(
                    started.elapsed() < Duration::from_secs(5),
                    "not copied after the submission completed"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        } else {
            assert!(
                done.recv_timeout(Duration::from_millis(300)).is_err(),
                "copied a page the submission in flight writes"
            );
            drop(go);
            let copied = done
                .recv_timeout(Duration::from_secs(5))
                .expect("still waiting after the submission completed");
            assert_eq!(copied, PAGE);
            assert_eq!(held(&s, y), Some(0xbeef));
        }
        drop(s);
        unregister(usize::MAX - 25, at as u64);
        release(at);
    }

    /// Big splits run on the split threads, started once, not on threads
    /// of their own each time (a thread's start and end run the title
    /// images' thread notices, ~110 ms in the game).
    /// The lux probe watches a view's address and names the ranges that
    /// cover it: the start, not the end (diagnostics, WAL-10).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn lux_watch_names_the_addresses_a_range_covers() {
        let watched: std::collections::BTreeSet<u64> = [0x1000, 0x1010, 0x2000].into();
        assert_eq!(lux_in(&watched, 0x1000, 0x1010), [0x1000]);
        assert_eq!(lux_in(&watched, 0x0f00, 0x2001), [0x1000, 0x1010, 0x2000]);
        assert!(lux_in(&watched, 0x1011, 0x2000).is_empty());
    }

    /// The lux probe tells what the GPU read through a view from what title
    /// memory held: words not zero in each, the words that differ by byte
    /// offset, and the first four of each (diagnostics, WAL-10).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn lux_reads_name_the_words_the_gpu_read_differently() {
        let one = 1.0f32.to_bits();
        let title = [one, 0, 7, 9, 3];
        let gpu = [one, 0, 0, 9, 0];
        let (title_nonzero, gpu_nonzero, differ, first, title_head, gpu_head) =
            lux_compare(&title, &gpu);
        assert_eq!((title_nonzero, gpu_nonzero, differ), (4, 2, 2));
        assert_eq!(first, [(8, 7, 0), (16, 3, 0)]);
        assert_eq!(title_head[0], 1.0);
        assert_eq!(gpu_head[2].to_bits(), 0);
        let (.., short) = lux_compare(&[], &[one]);
        assert_eq!(short[0], 1.0);
        assert!(short[1].is_nan());
    }

    #[test]
    fn big_splits_run_on_threads_started_once() {
        let mut ran = std::collections::HashSet::new();
        for _ in 0..20 {
            ran.extend(split(64, 64 << 20, |_| std::thread::current().id()));
        }
        // The split threads and this one, and other tests' that took a part.
        assert!(ran.len() <= 10, "{} threads", ran.len());
    }

    /// The title writes a piece of a buffer while copy-back writes the
    /// GPU's words in it back: only those words go back, so the title's
    /// write next to them stays (run 68).
    #[test]
    fn a_title_write_while_copy_back_writes_a_piece_stays() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = title_memory(502, 256 * MIB, MIB, MIB);
        assert_ne!(at, 0, "no free address for test memory");
        // Small enough not to be used in place.
        register(usize::MAX - 16, at as u64, 512 << 10);
        let s = find(at as u64, 512 << 10).unwrap();
        assert!(!s.in_place);
        let declared = [(at, 512 << 10)];
        let x = at + (128 << 10) + 32;
        crate::watch::forget_asked();
        s.update(at, 512 << 10);
        writing(&s, at, 512 << 10);
        // SAFETY: inside the copy's memory.
        unsafe { s.host_at(x - at, 4).cast::<u32>().write(0xfeed) };
        // SAFETY: our test memory.
        *BEFORE_WRITING.lock().unwrap() =
            Some(Box::new(move || unsafe { ((x + 64) as *mut u32).write(7) }));
        crate::watch::forget_asked();
        assert_eq!(
            s.copy_back(&declared, &Default::default(), None, 0),
            64 << 10
        );
        // SAFETY: as above.
        let read = |w: usize| unsafe { (w as *const u32).read() };
        assert_eq!(read(x + 64), 7, "the title's write was undone");
        assert_eq!((read(x), read(x + 4)), (0xfeed, word(x + 4)));
        drop(s);
        unregister(usize::MAX - 16, at as u64);
        release(at);
    }

    /// The GPU's words scattered over a guarded piece go back in place,
    /// and only they.
    #[test]
    fn scattered_gpu_words_go_back_in_place() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = (966 * SLAB) as usize - 256 * MIB;
        // Copies across views: not used in place.
        let made = views(at, &[MIB; 4]);
        assert_eq!(made.len(), 4, "no free address for test views");
        register(usize::MAX - 17, at as u64, (4 * MIB) as u64);
        let s = find(at as u64, (4 * MIB) as u64).unwrap();
        assert!(!s.in_place);
        let declared = [(at, 4 * MIB)];
        let piece = at + MIB + (64 << 10);
        crate::watch::forget_asked();
        s.update(at, 4 * MIB);
        writing(&s, at, 4 * MIB);
        let gpu: Vec<usize> = (0..32).map(|k| piece + k * 1024 + 8).collect();
        for &w in &gpu {
            // SAFETY: inside the copy's memory.
            unsafe { s.host_at(w - at, 4).cast::<u32>().write(0xfeed) };
        }
        crate::watch::forget_asked();
        assert_eq!(
            s.copy_back(&declared, &Default::default(), None, 0),
            64 << 10
        );
        crate::watch::disarm(0);
        // SAFETY: our views.
        let read = |w: usize| unsafe { (w as *const u32).read() };
        for &w in &gpu {
            assert_eq!((read(w), read(w + 4)), (0xfeed, word(w + 4)));
        }
        drop(s);
        unregister(usize::MAX - 17, at as u64);
        unmap(&made);
    }

    /// With VRAM twins, shaders use the twins of a buffer's copy and of a
    /// mirror slab: what the CPU wrote into the copies reaches the twins
    /// before the submission runs, and what the GPU wrote there comes back
    /// to title memory after.
    #[test]
    fn a_submission_runs_on_the_twins_of_copies() {
        let _serial = crate::watch::serial();
        if !gpu() || !twins_on() {
            return;
        }
        let host = host().unwrap();
        let at = title_memory(501, 256 * MIB, 32 * MIB, 32 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        // Small enough not to be used in place.
        register(usize::MAX - 14, at as u64, 512 << 10);
        let buffer = find(at as u64, 512 << 10).unwrap();
        let slab = find((at + 20 * MIB) as u64, 1).unwrap();
        assert!(buffer.twin.is_some() && !buffer.mirror());
        assert!(slab.twin.is_some() && slab.mirror());
        let scratch = vram_buffer(64 << 10).unwrap();
        let list = host.new_list().unwrap();
        // In each, the GPU copies 4 KiB and a word of the copy over the next
        // (the compare's last 16 bytes hold a word).
        const N: usize = 4100;
        let places = [(&buffer, at + (64 << 10)), (&slab, at + 20 * MIB)];
        for (s, a) in places {
            s.refresh(&list, a as u64, 4096 + N as u64);
            s.gpu_writes(&list, (a + 4096) as u64, N as u64);
            let (twin, o) = (s.resource(), s.offset(a as u64));
            // SAFETY: an open list; our buffers, the ranges inside them.
            unsafe {
                list.list.CopyBufferRegion(&scratch, 0, twin, o, N as u64);
                list.list.ResourceBarrier(&[
                    crate::host::transition(
                        twin,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ),
                    crate::host::transition(
                        &scratch,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                list.list
                    .CopyBufferRegion(twin, o + 4096, &scratch, 0, N as u64);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &scratch,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
        }
        list.close();
        let counted = || {
            crate::host::TWINS
                .each_ref()
                .map(|n| n.load(Ordering::Relaxed))
        };
        let before = counted();
        host.execute(&[&list]);
        list.reset();
        let after = counted();
        // Both ranges came down, through the compare where it is on.
        let k = if twin_compare_on() { 2 } else { 3 };
        assert!(
            after[k] - before[k] >= 2 * N as u64,
            "{before:?} to {after:?}"
        );
        for (_, a) in places {
            for w in (a..a + N).step_by(4) {
                // SAFETY: our test memory.
                let got = unsafe { ((w + 4096) as *const u32).read() };
                assert_eq!(got, word(w), "at {:#x}", w + 4096);
            }
        }
        drop((buffer, slab));
        unregister(usize::MAX - 14, at as u64);
        release(at);
    }

    /// Told by the compare's bits, copy-back puts back the words the GPU
    /// changed where the title does not hold them already, gives the copy
    /// the title's other words, and keeps to the bytes it is given of a word
    /// cut at either end.
    #[test]
    fn bits_tell_copy_back_the_words_the_gpu_changed() {
        let mut title: Vec<u8> = (0..28u8).collect();
        let mut host = title.clone();
        let mut want = title.clone();
        // Byte i is in word (i + 2) / 4: word 0 ends at byte 2, and word 7,
        // cut at the end, starts at byte 26.
        for (from, to, gpu) in [(0, 2, 0xbb), (2, 6, 0xaa), (10, 14, 0x33), (26, 28, 0xdd)] {
            host[from..to].fill(gpu);
            want[from..to].fill(gpu);
        }
        // Word 3 the title holds already; word 5 is the title's.
        title[10..14].fill(0x33);
        title[18..22].fill(0xcc);
        want[18..22].fill(0xcc);
        let first = 35;
        let mut bits = [0u32; 3];
        for w in [0, 1, 3, 7, 8] {
            bits[(first + w) / 32] |= 1 << ((first + w) % 32);
        }
        assert!(any_bit(&bits, first + 3, 1) && any_bit(&bits, 0, first + 1));
        assert!(!any_bit(&bits, first + 4, 3) && !any_bit(&bits, first + 9, 60));
        assert!(!any_bit(&bits, 0, first) && !any_bit(&bits, first, 0));
        assert!(any_bit(&bits, 1, 95) && !any_bit(&bits, 96, 64));
        let (runs, taken) = merge_bits(&mut title, &mut host, &bits, first, 2);
        assert_eq!(runs, [(0, 6), (26, 28)]);
        assert_eq!(taken, Some((18, 22)));
        assert_eq!(title, want);
        assert_eq!(host, want);
    }

    /// Copy-back told by the compare's bits writes back the bytes declared
    /// of the words they tell, and only those: a title byte outside them in
    /// a word the GPU changed stays, also between two ranges whose compares
    /// (whole words) overlap.
    #[test]
    fn copy_back_told_by_bits_keeps_to_the_bytes_declared() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let at = title_memory(1300, 256 * MIB, MIB, MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 26, at as u64, 512 << 10);
        let s = find(at as u64, 512 << 10).unwrap();
        assert!(!s.in_place);
        let declared = [(at + 6, 999), (at + 1007, 20)];
        crate::watch::forget_asked();
        s.update(at, 512 << 10);
        for &(address, len) in &declared {
            s.writing(address, len);
        }
        let put = |to: *mut u8, bytes: &[u8]| {
            // SAFETY: inside our test memory or the copy's.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to, bytes.len()) }
        };
        // The GPU's bytes, and the title's since.
        let gpu = [
            (at + 6, &[0xa6, 0xa7][..]),
            (at + 500, &0xfeed_u32.to_le_bytes()[..]),
            (at + 1004, &[0xe4][..]),
            (at + 1012, &0xbeef_u32.to_le_bytes()[..]),
        ];
        for (a, bytes) in gpu {
            put(s.host_at(a - at, bytes.len()), bytes);
        }
        let title = [
            (at + 4, &[0x11][..]),
            (at + 600, &0x600d_u32.to_le_bytes()[..]),
            (at + 1006, &[0x22][..]),
        ];
        for (a, bytes) in title {
            put(a as *mut u8, bytes);
        }
        // The compares' ranges as kept, and the bits of the words they found
        // changed: those at 4, 500, 1004 (in both) and 1012.
        let ranges = [(at + 4, 1004, 0), (at + 1004, 24, 256)];
        let mut bits = [0u32; 16];
        for b in [0, 124, 250, 256, 258] {
            bits[b / 32] |= 1 << (b % 32);
        }
        crate::watch::forget_asked();
        let written = s.copy_back(&declared, &Default::default(), Some((&bits, &ranges)), 0);
        assert_eq!(written, 999 + 19);
        // SAFETY: our test memory.
        let byte = |a: usize| unsafe { (a as *const u8).read() };
        let original = |a: usize| word(a & !3).to_le_bytes()[a % 4];
        assert_eq!(
            byte(at + 4),
            0x11,
            "the title's byte before the first range"
        );
        assert_eq!(byte(at + 5), original(at + 5));
        assert_eq!((byte(at + 6), byte(at + 7)), (0xa6, 0xa7));
        assert_eq!(held(&s, at + 500), Some(0xfeed));
        // SAFETY: as above.
        let read = |a: usize| unsafe { (a as *const u32).read() };
        assert_eq!(
            (read(at + 500), read(at + 600), read(at + 1012)),
            (0xfeed, 0x600d, 0xbeef)
        );
        assert_eq!(held(&s, at + 600), Some(0x600d));
        assert_eq!(byte(at + 1004), 0xe4);
        assert_eq!(byte(at + 1005), original(at + 1005));
        assert_eq!(byte(at + 1006), 0x22, "the title's byte between the ranges");
        assert_eq!(byte(at + 1007), original(at + 1007));
        drop(s);
        unregister(usize::MAX - 26, at as u64);
        release(at);
    }

    /// Pipelined, copy-back told by bits: the compares after a submission
    /// wait on the GPU, not the CPU, for the copy-back in flight of one
    /// whose shaders write what they compare, which reads the host copy
    /// they store into.
    #[test]
    fn compares_wait_on_the_gpu_for_copy_backs_in_flight() {
        use std::time::{Duration, Instant};
        /// Lets the held copy-back go on, also when the test fails.
        struct Go(std::sync::mpsc::Sender<()>);
        impl Drop for Go {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let _serial = crate::watch::serial();
        if !gpu() || !twins_on() || !crate::pipelined::on() {
            return;
        }
        let host = host().unwrap();
        let at = title_memory(1310, 256 * MIB, 32 * MIB, 32 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 27, at as u64, 512 << 10);
        let s = find(at as u64, 512 << 10).unwrap();
        if !s.told_by_words() || compare_pipeline().is_none() {
            drop(s);
            unregister(usize::MAX - 27, at as u64);
            release(at);
            return;
        }
        let twin = s.resource().clone();
        let (upload, data) = host.upload_buffer(4096).unwrap();
        // SAFETY: our mapped upload buffer.
        unsafe { data.cast::<[u32; 2]>().write([0xfeed, 0xbeef]) };
        crate::pipelined::wait_all();
        crate::watch::forget_asked();
        let (x, y) = (at + 64, at + 128);
        // Lists apart: the pipeline tells lists in flight by address.
        let (one, two) = (host.new_list().unwrap(), host.new_list().unwrap());
        let submit = |list: &crate::host::HostList, to: usize, from: u64, refresh: bool| {
            if refresh {
                s.refresh(list, at as u64, 4096);
            }
            s.gpu_writes(list, at as u64, 4096);
            // SAFETY: an open list; live buffers holding the ranges.
            unsafe {
                list.list
                    .CopyBufferRegion(&twin, s.offset(to as u64), &upload, from, 4)
            };
            list.close();
            host.execute(&[list]);
        };
        let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
        let (go, wait) = std::sync::mpsc::channel::<()>();
        *BEFORE_WRITING.lock().unwrap() = Some(Box::new(move || {
            let _ = held_tx.send(());
            let _ = wait.recv_timeout(Duration::from_secs(10));
        }));
        let go = Go(go);
        submit(&one, x, 0, true);
        held_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the first copy-back wrote back nothing");
        let started = Instant::now();
        submit(&two, y, 4, false);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the second submission waited on the CPU"
        );
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            held(&s, y),
            Some(word(y)),
            "the second's compare stored during the first's copy-back"
        );
        drop(go);
        crate::pipelined::wait_all();
        // SAFETY: our test memory.
        let read = |a: usize| unsafe { (a as *const u32).read() };
        assert_eq!((read(x), read(y)), (0xfeed, 0xbeef));
        drop((one, two));
        drop(s);
        unregister(usize::MAX - 27, at as u64);
        release(at);
    }

    /// The bits a compare sets are clear by the next compare of its slot:
    /// a title write since, in a word the first changed, stays when the
    /// next changes a word beside it.
    #[test]
    fn a_compare_tells_only_its_own_submissions_words() {
        let _serial = crate::watch::serial();
        if !gpu() || !twins_on() {
            return;
        }
        let host = host().unwrap();
        let at = title_memory(1320, 256 * MIB, 32 * MIB, 32 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 28, at as u64, 512 << 10);
        let s = find(at as u64, 512 << 10).unwrap();
        if !s.told_by_words() || compare_pipeline().is_none() {
            drop(s);
            unregister(usize::MAX - 28, at as u64);
            release(at);
            return;
        }
        let twin = s.resource().clone();
        let (upload, data) = host.upload_buffer(4096).unwrap();
        // SAFETY: our mapped upload buffer.
        unsafe { data.cast::<[u32; 2]>().write([0xfeed, 0xbeef]) };
        crate::pipelined::wait_all();
        crate::watch::forget_asked();
        let (x, y) = (at + 64, at + 128);
        let depth = crate::pipelined::DEPTH;
        let lists: Vec<_> = (0..=depth).map(|_| host.new_list().unwrap()).collect();
        let gpu_writes = |list: &crate::host::HostList, to: usize, from: u64| {
            s.gpu_writes(list, at as u64, 4096);
            // SAFETY: an open list; live buffers holding the ranges.
            unsafe {
                list.list
                    .CopyBufferRegion(&twin, s.offset(to as u64), &upload, from, 4)
            };
            list.close();
            host.execute(&[list]);
            crate::pipelined::wait_all();
        };
        // SAFETY: our test memory.
        let read = |a: usize| unsafe { (a as *const u32).read() };
        s.refresh(&lists[0], at as u64, 4096);
        gpu_writes(&lists[0], x, 0);
        assert_eq!(read(x), 0xfeed);
        // SAFETY: as above.
        unsafe { (x as *mut u32).write(7) };
        // Submissions in the other slots, then one in the first's.
        for list in &lists[1..depth] {
            list.close();
            host.execute(&[list]);
        }
        gpu_writes(&lists[depth], y, 4);
        assert_eq!(read(y), 0xbeef);
        assert_eq!(
            read(x),
            7,
            "the title's write undone by the first compare's bit"
        );
        drop(s);
        unregister(usize::MAX - 28, at as u64);
        release(at);
    }

    /// Of one compare list, a range told by word marks the bits of the words
    /// shaders changed, and a range with no bitmap (its copy or window not
    /// told by word, or its slot given no memory for one) stores its changed
    /// words and no bits: none in the bitmap, none through the host copy
    /// bound in its place. Each range's first KiB changes whole, so every
    /// lane of a group passes the compare's conditions on its bits.
    #[test]
    fn a_range_with_no_bitmap_stores_no_bits() {
        let _serial = crate::watch::serial();
        if !gpu() || compare_pipeline().is_none() {
            return;
        }
        let host = host().unwrap();
        const LEN: usize = 64 << 10;
        let (twin, kept) = (
            vram_buffer(2 * LEN as u64).unwrap(),
            vram_buffer(2 * LEN as u64).unwrap(),
        );
        let (out, got) = host.shared_buffer(2 * LEN as u64).unwrap();
        let (bitmap, marks) = host.shared_buffer(LEN as u64).unwrap();
        let (source, data) = host.shared_buffer(2 * LEN as u64).unwrap();
        let changed = |i: usize| i % (LEN / 4) < 256 || i.is_multiple_of(61);
        let fill = |to: &ID3D12Resource, f: &dyn Fn(usize) -> u32| {
            for i in 0..2 * LEN / 4 {
                // SAFETY: inside the mapped buffer.
                unsafe { data.cast::<u32>().add(i).write(f(i)) };
            }
            let list = host.new_list().unwrap();
            // SAFETY: an open list; live buffers holding the range.
            unsafe {
                list.list
                    .CopyBufferRegion(to, 0, &source, 0, 2 * LEN as u64)
            };
            list.close();
            host.execute(&[&list]);
            crate::pipelined::wait_all();
        };
        fill(&kept, &|i| i as u32);
        fill(&twin, &|i| if changed(i) { !(i as u32) } else { i as u32 });
        // SAFETY: the mapped buffers.
        unsafe {
            std::ptr::write_bytes(got, 0, 2 * LEN);
            std::ptr::write_bytes(marks, 0, LEN);
        }
        // SAFETY: live buffers.
        let gpu = |b: &ID3D12Resource| unsafe { b.GetGPUVirtualAddress() };
        let range = |at: usize, words| Compare {
            twin: gpu(&twin) + at as u64,
            kept: gpu(&kept) + at as u64,
            host: gpu(&out) + at as u64,
            bytes: LEN as u64,
            title: at,
            of: 0,
            notes: None,
            words,
        };
        let compares = [range(0, Some((gpu(&bitmap), 0))), range(LEN, None)];
        let list = host.new_list().unwrap();
        assert_eq!(compare_twins(&list.list, &compares, 0), 2 * LEN as u64);
        list.close();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        // SAFETY: the mapped buffers, which the finished compare wrote.
        let (words, bits) = unsafe {
            (
                std::slice::from_raw_parts(got.cast::<u32>(), 2 * LEN / 4),
                std::slice::from_raw_parts(marks.cast::<u32>(), LEN / 4),
            )
        };
        for (i, &word) in words.iter().enumerate() {
            assert_eq!(word, if changed(i) { !(i as u32) } else { 0 }, "word {i}");
        }
        for i in 0..LEN / 4 {
            assert_eq!(bits_at(bits, i) & 1 != 0, changed(i), "word {i}'s bit");
        }
        assert!(
            bits[LEN / 128..].iter().all(|&b| b == 0),
            "bits past the range told by word"
        );
    }

    /// The title releases a copy whose twin has yet to take what copy-back
    /// gave its host copy: the next submission's upload keeps both buffers
    /// until it has run (they went with the copy's last reference, and the
    /// device was removed).
    #[test]
    fn an_upload_keeps_a_released_copys_buffers_until_it_runs() {
        let _serial = crate::watch::serial();
        if !gpu() || !twins_on() {
            return;
        }
        let host = host().unwrap();
        let at = title_memory(1330, 256 * MIB, 32 * MIB, 32 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 29, at as u64, 512 << 10);
        let s = find(at as u64, 512 << 10).unwrap();
        assert!(s.twin.is_some() && !s.in_place);
        let (upload, data) = host.upload_buffer(4096).unwrap();
        // SAFETY: our mapped upload buffer.
        unsafe { data.cast::<u32>().write(0xfeed) };
        crate::pipelined::wait_all();
        crate::watch::forget_asked();
        let (first, list) = (host.new_list().unwrap(), host.new_list().unwrap());
        s.refresh(&first, at as u64, 4096);
        first.close();
        host.execute(&[&first]);
        crate::pipelined::wait_all();
        // The title writes x after the copy and its twin took it, and the
        // GPU writes y beside it: copy-back gives the copy x, for the twin
        // to take.
        let (x, y) = (at + 64, at + 128);
        // SAFETY: our test memory.
        unsafe { (x as *mut u32).write(7) };
        s.gpu_writes(&list, at as u64, 4096);
        // SAFETY: an open list; live buffers holding the ranges.
        unsafe {
            list.list
                .CopyBufferRegion(s.resource(), s.offset(y as u64), &upload, 0, 4)
        };
        list.close();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: as above.
        let read = |a: usize| unsafe { (a as *const u32).read() };
        assert_eq!((read(x), read(y)), (7, 0xfeed));
        drop(s);
        unregister(usize::MAX - 29, at as u64);
        let next = host.new_list().unwrap();
        next.close();
        host.execute(&[&next]);
        crate::pipelined::wait_all();
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        release(at);
    }

    /// With the compare's notes, copy-back skips the title blocks in which
    /// shaders changed no word, and not one whose only changed word lies in
    /// 16 bytes the compare read with the block before it.
    #[test]
    fn copy_back_skips_blocks_the_compare_found_unchanged() {
        let _serial = crate::watch::serial();
        if !gpu() || !twins_on() {
            return;
        }
        let host = host().unwrap();
        let at = title_memory(506, 256 * MIB, 32 * MIB, 32 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        register(usize::MAX - 19, at as u64, 512 << 10);
        let s = find(at as u64, 512 << 10).unwrap();
        assert!(s.twin.is_some() && !s.mirror() && at.is_multiple_of(PAGE));
        let scratch = vram_buffer(64 << 10).unwrap();
        let list = host.new_list().unwrap();
        // Shaders may write 256 KiB from 4 bytes into the first block: the
        // compare's 16-byte reads straddle the blocks' ends.
        s.refresh(&list, at as u64, (256 << 10) + 4);
        s.gpu_writes(&list, at as u64 + 4, 256 << 10);
        let (twin, o) = (s.resource(), s.offset(at as u64));
        // The GPU writes the first word of the second block, and 4 KiB of
        // the third, with the first block's words.
        let (one, two) = (at + PAGE, at + 2 * PAGE + 4096);
        // SAFETY: an open list; our buffers, the ranges inside them.
        unsafe {
            list.list.CopyBufferRegion(&scratch, 0, twin, o, 4096);
            list.list.ResourceBarrier(&[
                crate::host::transition(
                    twin,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
                crate::host::transition(
                    &scratch,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
            list.list
                .CopyBufferRegion(twin, o + PAGE as u64, &scratch, 0, 4);
            list.list
                .CopyBufferRegion(twin, s.offset(two as u64), &scratch, 0, 4096);
            list.list.ResourceBarrier(&[crate::host::transition(
                &scratch,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
        }
        list.close();
        let skipped = || crate::host::TWINS[4].load(Ordering::Relaxed);
        let before = skipped();
        host.execute(&[&list]);
        list.reset();
        let skipped = skipped() - before;
        // The fourth block, and the fifth's 4 bytes.
        if twin_compare_on() && twin_skip_on() {
            assert!(skipped >= PAGE as u64, "{skipped} bytes skipped");
        } else {
            assert_eq!(skipped, 0);
        }
        // SAFETY: our test memory.
        let read = |w: usize| unsafe { (w as *const u32).read() };
        assert_eq!(read(one), word(at), "the second block's first word");
        for w in (0..4096).step_by(4) {
            assert_eq!(read(two + w), word(at + w), "at {:#x}", two + w);
        }
        for w in (at + 3 * PAGE..at + 4 * PAGE).step_by(4) {
            assert_eq!(read(w), word(w), "at {w:#x}");
        }
        drop(s);
        unregister(usize::MAX - 19, at as u64);
        release(at);
    }

    /// A big title buffer used in place, in a pool, has a twin: what
    /// shaders write in it reaches title memory through the compare, and
    /// the twin takes the title's writes after (the pool's write watch
    /// tells them: the heap keeps it from being guarded), not the words it
    /// wrote itself.
    #[test]
    fn a_buffer_used_in_place_takes_what_shaders_write_through_its_twin() {
        let _serial = crate::watch::serial();
        if !gpu() || !twins_on() || !twin_compare_on() || !in_place_twins_on() {
            return;
        }
        let host = host().unwrap();
        let at = title_memory(1100, 256 * MIB, 32 * MIB, 32 * MIB);
        assert_ne!(at, 0, "no free address for test memory");
        crate::watch::title_memory_watched(at, 32 * MIB, true);
        register(usize::MAX - 23, at as u64, 32 << 20);
        let s = find(at as u64, 32 << 20).unwrap();
        if !s.in_place {
            eprintln!("no heap on title memory: skipped");
            drop(s);
            unregister(usize::MAX - 23, at as u64);
            crate::watch::title_memory_watched(at, 32 * MIB, false);
            release(at);
            return;
        }
        assert!(s.twin.is_some() && s.start == at as u64);
        let twin = s.resource().clone();
        let (upload, data) = host.upload_buffer(4096).unwrap();
        // SAFETY: our mapped upload buffer.
        unsafe { data.cast::<u32>().write(0xfeed_f00d) };
        let up = || crate::host::TWINS[1].load(Ordering::Relaxed);
        // What earlier tests left for twins goes up first.
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // The GPU writes a word of the second block.
        let list = host.new_list().unwrap();
        s.refresh(&list, at as u64, 32 << 20);
        s.gpu_writes(&list, at as u64, MIB as u64);
        // SAFETY: an open list; live buffers holding the ranges.
        unsafe {
            list.list
                .CopyBufferRegion(&twin, (PAGE + 8) as u64, &upload, 0, 4)
        };
        list.close();
        let first = up();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(
            up() - first,
            32 << 20,
            "the twin takes the buffer whole first"
        );
        // SAFETY: committed test memory.
        let read = |w: usize| unsafe { (w as *const u32).read_volatile() };
        assert_eq!(read(at + PAGE + 8), 0xfeed_f00d);
        assert_eq!(read(at + PAGE + 12), word(at + PAGE + 12));
        // The title writes a word of the fourth block: the twin takes that
        // block, and not the MiB shaders were given.
        // SAFETY: as above.
        unsafe { ((at + 3 * PAGE + 16) as *mut u32).write(0x1234_5678) };
        let (back, got) = host.shared_buffer(1 << 16).unwrap();
        let list = host.new_list().unwrap();
        s.refresh(&list, at as u64, 32 << 20);
        // SAFETY: an open list; live buffers holding the ranges.
        unsafe {
            list.list
                .CopyBufferRegion(&back, 0, &twin, (PAGE + 8) as u64, 4);
            list.list
                .CopyBufferRegion(&back, 4, &twin, (3 * PAGE + 16) as u64, 4);
        }
        list.close();
        let second = up();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(up() - second, PAGE as u64, "bytes copied into the twin");
        // SAFETY: the mapped buffer the finished copies wrote.
        let twin_words = unsafe {
            [
                got.cast::<u32>().read_volatile(),
                got.cast::<u32>().add(1).read_volatile(),
            ]
        };
        assert_eq!(twin_words, [0xfeed_f00d, 0x1234_5678]);
        drop((s, list));
        unregister(usize::MAX - 23, at as u64);
        crate::watch::title_memory_watched(at, 32 * MIB, false);
        release(at);
    }

    /// What the twins' compare costs on the GPU: 256 MiB with no word
    /// changed, one word in 64, and a tenth of the 64 KiB blocks whole,
    /// without and with notes. Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture twin_compare_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn twin_compare_costs() {
        let _serial = crate::watch::serial();
        if !gpu() || compare_pipeline().is_none() {
            return;
        }
        let host = host().unwrap();
        const LEN: usize = 256 * MIB;
        let (twin, kept) = (
            vram_buffer(LEN as u64).unwrap(),
            vram_buffer(LEN as u64).unwrap(),
        );
        let (out, got) = host.shared_buffer(LEN as u64).unwrap();
        let (source, data) = host.shared_buffer(LEN as u64).unwrap();
        let fill = |f: &dyn Fn(usize) -> u32| {
            for i in 0..LEN / 4 {
                // SAFETY: inside the mapped buffer.
                unsafe { data.cast::<u32>().add(i).write(f(i)) };
            }
        };
        let run = |record: &dyn Fn(&ID3D12GraphicsCommandList)| {
            let list = host.new_list().unwrap();
            record(&list.list);
            list.close();
            let started = std::time::Instant::now();
            host.execute(&[&list]);
            crate::pipelined::wait_all();
            started.elapsed()
        };
        let copy = |to: &ID3D12Resource| {
            run(&|l| {
                // SAFETY: an open list; live buffers holding the range.
                unsafe { l.CopyBufferRegion(to, 0, &source, 0, LEN as u64) }
            })
        };
        fill(&|i| i as u32);
        copy(&kept);
        // References: a submission's own cost, and a copy within VRAM.
        let best = |record: &dyn Fn(&ID3D12GraphicsCommandList)| {
            let took = (0..3).map(|_| run(record)).min().unwrap();
            took.as_secs_f64() * 1e3
        };
        let within = |len: u64| {
            best(&|l| {
                // SAFETY: an open list; live buffers holding the range.
                unsafe { l.CopyBufferRegion(&twin, 0, &kept, 0, len) }
            })
        };
        let (empty, within) = (within(4), within(LEN as u64));
        println!("a submission: {empty:.1} ms; 256 MiB copied within VRAM: {within:.1} ms");
        // SAFETY: live buffers.
        let gpu = |b: &ID3D12Resource| unsafe { b.GetGPUVirtualAddress() };
        let cases: [(&str, &dyn Fn(usize) -> bool); 3] = [
            ("no word changed", &|_| false),
            ("one word in 64 changed", &|i| i.is_multiple_of(64)),
            ("a tenth of the blocks changed", &|i| {
                (4 * i / PAGE).is_multiple_of(10)
            }),
        ];
        for (what, changed) in cases {
            fill(&|i| if changed(i) { !(i as u32) } else { i as u32 });
            copy(&twin);
            for (noting, wording) in [(false, false), (true, false), (true, true)] {
                let c = Compare {
                    twin: gpu(&twin),
                    kept: gpu(&kept),
                    host: gpu(&out),
                    bytes: LEN as u64,
                    title: 0,
                    of: 0,
                    notes: noting.then(|| {
                        let list = host.new_list().unwrap();
                        let gpu = notes(&list.list, LEN / PAGE, 0).unwrap();
                        list.close();
                        host.execute(&[&list]);
                        crate::pipelined::wait_all();
                        (gpu, 0)
                    }),
                    words: wording
                        .then(|| super::changed(LEN / 128, 0).map(|gpu| (gpu, 0)))
                        .flatten(),
                };
                let took = best(&|l| _ = compare_twins(l, std::slice::from_ref(&c), 0));
                println!("{what}, notes {noting}, words {wording}: {took:.1} ms for 256 MiB");
                if c.words.is_some() {
                    let told = told(Some(std::slice::from_ref(&c)), 1, 0);
                    let (bits, _) = told.of(0).unwrap();
                    for i in (0..LEN / 4).step_by(4099) {
                        assert_eq!(bits_at(bits, i) & 1 != 0, changed(i), "word {i}'s bit");
                    }
                    told.clear(&[]);
                }
            }
            for i in (0..LEN / 4).step_by(4099) {
                // SAFETY: inside the mapped buffer the compare wrote.
                let word = unsafe { got.cast::<u32>().add(i).read_volatile() };
                if changed(i) {
                    assert_eq!(word, !(i as u32), "word {i}");
                }
            }
        }
    }

    /// A GpuWrite step of the submission being prepared, then its snapshot
    /// at submission where no compare tells copy-back what shaders change
    /// ([`snapshot_untold`]).
    fn writing(s: &Arc<Shadow>, address: usize, len: usize) {
        s.writing(address, len);
        snapshot_untold(&[(s.clone(), vec![(address, len)])], None);
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
            writing(&s, at, n);
        }
        let spans: Vec<(usize, usize)> = {
            let before = s.before[0].lock().unwrap();
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
        let written = s.copy_back(&ranges, &Default::default(), None, 0);
        assert_eq!(written, 64 << 10);
        // SAFETY: committed test memory.
        assert_eq!(unsafe { (at as *const u32).read() }, 0xfeed);
        // SAFETY: as above.
        assert_eq!(unsafe { ((at + 4) as *const u32).read() }, word(at + 4));
        assert!(s.before[0].lock().unwrap().is_empty());
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
