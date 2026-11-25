//! Title memory at raw GPU addresses (decision 0005, the unified-memory
//! item; the contract is `victory_shader::emit`'s "Title memory").
//! Translated shaders that read or write memory at addresses they compute
//! look each address up in a page table: per 64 KiB page below 2^40, the
//! heap index of a pair of raw views (SRV, UAV) of the host copy holding
//! the page (`crate::shadow`) and the low dword of that copy's title
//! address. The copies are the ones draws and dispatches use through views,
//! so what one shader writes through a view another reads at its address,
//! within a submission too.
//!
//! Per list, at its first dispatch or draw with such a shader in a view
//! heap (and again when a root view's address is in no copy the table
//! names): every host copy made so far gets its pair of views in dynamic
//! slots of that heap (held until the list is reset), the page table goes
//! into the list's upload memory, and every copy is brought up to date at
//! each submission (page by page, as every refresh), and then armed
//! ([`crate::shadow::Shadow::arm_table`]): what each copy held before the
//! submission is known after it. Shaders mark the pages they write in the
//! table's bitmap; after the submission the bytes shaders changed in each
//! marked page of a title buffer's copy go back into title memory, and
//! only those ([`crate::shadow::Shadow::copy_back_page`]); the bitmap is
//! cleared. The title's threads write those buffers meanwhile: a whole page
//! from the copy put back what the title wrote there since the copy took it
//! (run 131: a child array of the title's scene nodes, read as a pointer,
//! crashed it). Pages of loose memory (outside title buffers) stay on the
//! host unless `VICTORY_LOOSE_COPY_BACK=1` has them go back the same way.
//! shortcut: by default the title never sees what shaders write in loose
//! memory.
//! Such shaders read their root views there too, so the host makes a copy
//! for each root view's address before the table.
//!
//! Cost per submission with such a shader: one write-watch check of every
//! host copy (the refresh), copies of the pages the title wrote, and a
//! compare of each 64 KiB page shaders wrote; per list, two views per copy
//! and a 16 KiB + 32 KiB-per-256-MiB-block table.
//!
//! Memory no copy holds when the table is made reads 0 and takes no
//! writes (null views, bounds-checked); shaders mark such pages (or 256
//! MiB regions with no block) in the bitmap and [`log_frame`] counts them.
//! shortcut: a page two copies share without either holding all of it is
//! the larger one's.

use std::sync::Arc;

use crate::host::HostList;
use crate::shadow::Shadow;

/// Pages per level-2 block: 4096 pages of 64 KiB, 256 MiB.
const BLOCK: usize = 4096;
/// Level-1 entries: 2^40 bytes in 256 MiB blocks.
const LEVEL1: usize = 4096;

/// Title blocks (64 KiB) page tables have written in copies that go back:
/// kept before each submission where they cannot be armed.
static WRITTEN: std::sync::Mutex<std::collections::BTreeSet<usize>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Whether page tables' writes in loose memory go back into it too, byte
/// for byte as in title buffers (`VICTORY_LOOSE_COPY_BACK=1`).
fn loose_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_LOOSE_COPY_BACK").is_ok_and(|v| v == "1");
        if on {
            tracing::info!(
                "what page tables' shaders write in loose memory goes back into it (VICTORY_LOOSE_COPY_BACK)"
            );
        }
        on
    })
}

/// Whether the page tables of all lists using a view heap share one run of
/// its dynamic slots with views of the host copies
/// (`VICTORY_SHARED_COPY_VIEWS=1`, DexNom b8ff22e's raw page views), instead
/// of each table taking two slots per copy until its list is reset: with
/// some 3,800 copies, run 150's tables took 7,612 slots each and the heap
/// ran out, dropping the BVH builder's dispatches ("its bindings did not
/// bind").
fn shared_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_SHARED_COPY_VIEWS").is_ok_and(|v| v == "1");
        if on {
            tracing::info!(
                "page tables share one run of copy views per view heap (VICTORY_SHARED_COPY_VIEWS)"
            );
        }
        on
    })
}

/// Whether page tables' bitmaps of written pages are in VRAM the CPU maps
/// (the GPU upload heap, `VICTORY_VRAM_BITMAP=1`) rather than in system
/// memory, where each shader store's atomic mark costs a PCIe round trip
/// (as the argument clamp's budget word did, VICT-9). The CPU reads each
/// bitmap once after its submission.
fn vram_bitmap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_VRAM_BITMAP").is_ok_and(|v| v == "1");
        if on {
            tracing::info!(
                "page tables' bitmaps of written pages are in VRAM (VICTORY_VRAM_BITMAP)"
            );
        }
        on
    })
}

/// A view heap's views of host copies for page tables
/// (`VICTORY_SHARED_COPY_VIEWS`): a run of its dynamic slots, the null pair
/// then a pair per piece of a copy (`build`). A piece keeps its pair while
/// the run lasts, and a new piece takes the next free one, which no table
/// names yet. A new run, twice the pieces, replaces a full one or one that
/// is mostly copies gone; a replaced run's slots go back once no list
/// whose table names it is left (its lease, held until the list is reset).
#[derive(Default)]
pub(crate) struct CopyViews(std::sync::Mutex<Views>);

#[derive(Default)]
struct Views {
    /// The run: its lease, first slot and pairs.
    run: Option<(Arc<()>, u32, u32)>,
    /// Pair by piece (copy, offset); the copies, held so that their
    /// addresses stay theirs.
    pairs: std::collections::HashMap<(usize, u64), u32>,
    held: Vec<Arc<Shadow>>,
    /// Replaced runs not yet given back: lease, first slot, slots.
    retired: Vec<(std::sync::Weak<()>, u32, u32)>,
}

impl CopyViews {
    /// The heap slot of each piece's pair (copy, offset, bytes) in
    /// `heap`'s run, made where missing, and the null pair's and the run's
    /// lease; None if no run fits.
    fn pairs(
        &self,
        heap: &crate::descriptor::host::HostHeap,
        pieces: &[(Arc<Shadow>, u64, u64)],
    ) -> Option<(u32, Vec<u32>, Arc<()>)> {
        let mut v = self.0.lock().unwrap_or_else(|e| e.into_inner());
        v.retired.retain(|(lease, first, n)| {
            let held = lease.strong_count() > 0;
            if !held {
                heap.release(*first, *n);
            }
            held
        });
        let key = |(s, offset, _): &(Arc<Shadow>, u64, u64)| (Arc::as_ptr(s) as usize, *offset);
        let found = pieces
            .iter()
            .filter(|p| v.pairs.contains_key(&key(p)))
            .count();
        let (missing, gone) = (pieces.len() - found, v.pairs.len().saturating_sub(found));
        let room = v.run.as_ref().map_or(0, |r| r.2 as usize - v.pairs.len());
        if v.run.is_none() || missing > room || gone > pieces.len().max(1024) {
            let live = pieces.len() as u32;
            let want = (2 * live).max(1024).next_power_of_two();
            let (first, pairs) = [want, live + 64]
                .into_iter()
                .filter(|&p| p >= live)
                .find_map(|p| heap.dynamic(2 * (p + 1)).map(|f| (f, p)))?;
            heap.raw_pair_at(first, None, 0, 0);
            let previous = v.run.take().map(|(lease, f, n)| {
                v.retired.push((Arc::downgrade(&lease), f, 2 * (n + 1)));
                n
            });
            tracing::info!(
                pairs,
                pieces = live,
                previous,
                "title memory copy views: a new shared run"
            );
            v.run = Some((Arc::new(()), first, pairs));
            v.pairs.clear();
            v.held.clear();
        }
        let (lease, first, _) = v.run.clone()?;
        let mut slots = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let pair = match v.pairs.get(&key(piece)) {
                Some(&pair) => pair,
                None => {
                    let (s, offset, len) = piece;
                    let pair = v.pairs.len() as u32;
                    heap.raw_pair_at(
                        first + 2 + 2 * pair,
                        Some(&s.buffer),
                        *offset,
                        len.next_multiple_of(4),
                    );
                    v.pairs.insert(key(piece), pair);
                    v.held.push(s.clone());
                    pair
                }
            };
            slots.push(first + 2 + 2 * pair);
        }
        Some((first, slots, lease))
    }
}

/// A list's page table.
pub(crate) struct Map {
    /// The page table's host GPU address (the root SRV).
    pub(crate) table: u64,
    /// The page bitmap's host GPU address (the root UAV).
    pub(crate) dirty: u64,
    /// The title ranges of the copies it names.
    ranges: Vec<(u64, u64)>,
    /// Ranges that go back byte for byte instead (root UAVs, see
    /// [`Map::precise`]).
    precise: Arc<std::sync::Mutex<Vec<(u64, u64)>>>,
}

impl Map {
    /// Whether `address` is in a copy the table names, or can never be
    /// (at or above 2^40).
    pub(crate) fn holds(&self, address: u64) -> bool {
        address >= 1 << 40 || self.ranges.iter().any(|&(s, l)| address >= s && address - s < l)
    }

    /// Shaders write `[address, address + len)` through a root UAV, which
    /// goes back by its bytes before and after the run
    /// ([`Shadow::gpu_writes`]): the marked pages it touches are left to
    /// that. A whole page from the copy would put back what it held at
    /// the submission over what the title (or DirectStorage, through
    /// physical pages no watch sees) wrote there since.
    /// shortcut: a marked page partly in such a range loses what shaders
    /// wrote in its other part.
    pub(crate) fn precise(&self, address: u64, len: u64) {
        self.precise
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, len));
    }
}

/// Since the last frame line: draws and dispatches with shaders using title
/// memory (by pipeline), page tables made, submissions of them in which a
/// lane its loop limit or budget stopped marked the bitmap
/// (`victory_shader::emit::LOOP_HIT_BIT`), pages copied back (and of them,
/// those whose earlier bytes were not known), pages and 256 MiB regions
/// missed (no copy), and the first missed addresses.
#[derive(Default)]
struct Stats {
    uses: std::collections::BTreeMap<String, u64>,
    tables: u64,
    stopped: u64,
    written: u64,
    unknown: u64,
    kept: u64,
    missed_pages: u64,
    missed_regions: u64,
    first_missed: Vec<u64>,
}

static STATS: std::sync::Mutex<Option<Stats>> = std::sync::Mutex::new(None);

fn stats<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    f(STATS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// A draw or dispatch of `pipeline` uses title memory.
pub(crate) fn note_use(pipeline: &str) {
    stats(|s| *s.uses.entry(pipeline.to_owned()).or_default() += 1);
}

/// Logs what title memory did since the last call, if anything: a line per
/// presented frame while shaders use it.
pub(crate) fn log_frame() {
    let s = stats(std::mem::take);
    if s.uses.is_empty() && s.written + s.kept + s.missed_pages + s.missed_regions + s.stopped == 0
    {
        return;
    }
    let mut uses: Vec<(&String, &u64)> = s.uses.iter().collect();
    uses.sort_by(|a, b| b.1.cmp(a.1));
    let uses: Vec<String> = uses.iter().take(32).map(|(p, n)| format!("{n} {p}")).collect();
    let missed: Vec<String> = s.first_missed.iter().map(|a| format!("{a:#x}")).collect();
    tracing::info!(
        "title memory this frame: {} draws/dispatches ({}), {} page tables ({} runs with \
         lanes stopped at their loop limit or budget), {} pages \
         copied back ({} left in the copy: not known before), {} of loose memory kept on \
         the host, missed {} pages and {} 256 MiB regions with no copy (first: {})",
        s.uses.values().sum::<u64>(),
        uses.join(", "),
        s.tables,
        s.stopped,
        s.written,
        s.unknown,
        s.kept,
        s.missed_pages,
        s.missed_regions,
        missed.join(" ")
    );
}

/// Our view heap for lists that bind none of the title's (dynamic slots
/// only).
pub(crate) fn heap() -> Option<&'static crate::descriptor::host::HostHeap> {
    static HEAP: std::sync::OnceLock<Option<crate::descriptor::host::HostHeap>> =
        std::sync::OnceLock::new();
    HEAP.get_or_init(|| crate::descriptor::host::HostHeap::new(false, 0))
        .as_ref()
}

/// The page table for host copies of title memory `copies` (start,
/// bytes), whose view pairs start at heap index `first + 2` (copy k at
/// `first + 2 + 2k`; `first` is the null pair): the table's dwords, the
/// level-1 index of each block, and the copy each entry names.
#[cfg(test)]
fn table(copies: &[(u64, u64)], first: u32) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let pairs: Vec<u32> = (0..copies.len() as u32)
        .map(|k| first + 2 + 2 * k)
        .collect();
    table_of(copies, &pairs, first)
}

/// As [`table`], with copy k's view pair at heap index `pairs[k]` and the
/// null pair at `null`.
fn table_of(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let mut order: Vec<usize> = (0..copies.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
    let mut level1 = vec![0u32; LEVEL1];
    // Block 0 is the null block (no copy).
    let mut blocks = vec![usize::MAX];
    let mut owner: Vec<Option<(usize, bool)>> = vec![None; BLOCK];
    for &k in &order {
        let (start, len) = copies[k];
        if len == 0 || start + len > 1 << 40 {
            continue;
        }
        for page in start >> 16..=(start + len - 1) >> 16 {
            let l1 = (page >> 12) as usize;
            if level1[l1] == 0 {
                level1[l1] = blocks.len() as u32;
                blocks.push(l1);
                owner.resize(owner.len() + BLOCK, None);
            }
            let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
            // Larger copies come first: a copy holding the whole page takes
            // it from them; one holding part of it only takes it if free.
            let whole = page << 16 >= start && (page + 1) << 16 <= start + len;
            if whole || owner[entry].is_none() {
                owner[entry] = Some((k, whole));
            }
        }
    }
    let mut dwords = level1;
    for o in &owner {
        let entry = match *o {
            Some((k, _)) => [pairs[k], copies[k].0 as u32],
            None => [null, 0],
        };
        dwords.extend(entry);
    }
    (dwords, blocks, owner.into_iter().map(|o| o.map(|o| o.0)).collect())
}

/// The page table of `list` in view heap `heap` (a host heap): views of
/// every host copy in dynamic slots, which `slots` gets to give back (or
/// in the heap's shared run, [`CopyViews`]);
/// `before` the list's earlier table, whose byte-for-byte ranges it keeps.
pub(crate) fn build(
    list: &HostList,
    heap: &crate::descriptor::host::HostHeap,
    slots: &mut Vec<u32>,
    before: Option<&Map>,
) -> Option<Map> {
    let copies: Vec<Arc<Shadow>> = crate::shadow::all();
    // A raw view holds at most 2^27 dwords: a longer copy (a slab of the
    // mirror) is viewed in pieces of at most that (copy, offset, bytes).
    const VIEW: u64 = 4 << 27;
    let pieces: Vec<(usize, u64, u64)> = copies
        .iter()
        .enumerate()
        .flat_map(|(k, s)| {
            let len = s.range().1;
            (0..len.div_ceil(VIEW).max(1)).map(move |p| (k, p * VIEW, (len - p * VIEW).min(VIEW)))
        })
        .collect();
    let ranges: Vec<(u64, u64)> = pieces
        .iter()
        .map(|&(k, offset, len)| (copies[k].range().0 + offset, len))
        .collect();
    let n = 2 * (pieces.len() as u32 + 1);
    let views = if shared_views() {
        let viewed: Vec<(Arc<Shadow>, u64, u64)> = pieces
            .iter()
            .map(|&(k, offset, len)| (copies[k].clone(), offset, len))
            .collect();
        heap.copy_views
            .pairs(heap, &viewed)
            .map(|(null, pairs, lease)| (null, pairs, Some(lease)))
    } else {
        heap.dynamic(n).map(|first| {
            slots.extend([first, n]);
            heap.raw_pair_at(first, None, 0, 0);
            for (p, &(k, offset, len)) in pieces.iter().enumerate() {
                let len = len.next_multiple_of(4);
                heap.raw_pair_at(
                    first + 2 + 2 * p as u32,
                    Some(&copies[k].buffer),
                    offset,
                    len,
                );
            }
            (
                first,
                (0..pieces.len() as u32)
                    .map(|p| first + 2 + 2 * p)
                    .collect(),
                None,
            )
        })
    };
    let Some((first, pairs, lease)) = views else {
        static FULL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let times = FULL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if times.is_power_of_two() {
            tracing::warn!(
                times,
                views = n,
                copies = copies.len(),
                "title memory page table: no dynamic slots for its copies' views (its draws and dispatches dropped)"
            );
        }
        return None;
    };
    let (dwords, blocks, owner) = table_of(&ranges, &pairs, first);
    let (table, at) = list.scratch(dwords.len() as u64 * 4)?;
    // SAFETY: list upload memory of that size.
    unsafe { std::ptr::copy_nonoverlapping(dwords.as_ptr(), at.cast::<u32>(), dwords.len()) };
    let bits = owner.len();
    let host = crate::host::host()?;
    let size = (bits / 8).next_multiple_of(256) as u64;
    let (bitmap, data) = match vram_bitmap() {
        true => host.vram_shared_buffer(size)?,
        false => host.shared_buffer(size)?,
    };
    crate::host::name(&bitmap, || "title memory written pages".into());
    let words = bits / 32;
    // SAFETY: our mapped buffer of at least `words` dwords.
    unsafe { std::ptr::write_bytes(data.cast::<u32>(), 0, words) };
    // SAFETY: a live buffer.
    let dirty = unsafe { bitmap.GetGPUVirtualAddress() };
    list.keep(&bitmap);
    // Every copy current at each submission, then armed: copy-back tells
    // what shaders changed from what it held.
    for s in &copies {
        let (start, len) = s.range();
        s.refresh(list, start, len);
    }
    let back: Vec<Arc<Shadow>> = copies
        .iter()
        .filter(|s| !s.is_loose() || loose_back())
        .cloned()
        .collect();
    list.defer_keep(move || {
        let written = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        for s in &back {
            s.arm_table(&written);
        }
    });
    let data = data as usize;
    tracing::debug!(copies = ranges.len(), blocks = blocks.len() - 1, "title memory page table");
    stats(|s| s.tables += 1);
    let precise = before.map(|m| m.precise.clone()).unwrap_or_default();
    let byte_for_byte = precise.clone();
    list.defer_after(move |slot| {
        // The shared run's views stay until the list is reset.
        let _ = &lease;
        let precise = byte_for_byte.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the bitmap is mapped and the list's run is over.
        let mapped = unsafe { std::slice::from_raw_parts_mut(data as *mut u32, words) };
        // Read whole (VRAM the CPU reads slowly word by word); the marked
        // words cleared for the list's next run.
        let mut marks = mapped.to_vec();
        for (word, &mark) in mapped.iter_mut().zip(&marks) {
            if mark != 0 {
                *word = 0;
            }
        }
        let (mut written, mut unknown, mut kept, mut pages, mut regions, mut missed) =
            (0, 0, 0, 0, 0, Vec::new());
        let mut stopped = 0;
        for (w, mark) in marks.iter_mut().enumerate() {
            let mut bits = std::mem::take(mark);
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if entry == victory_shader::emit::LOOP_HIT_BIT as usize {
                    stopped = 1;
                    continue;
                }
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                // Block 0's bits are 256 MiB regions with no block.
                let Some(&l1) = blocks.get(block).filter(|_| block != 0) else {
                    regions += 1;
                    missed.push((entry as u64) << 28);
                    continue;
                };
                let at = (l1 << 28 | page << 16) as u64;
                match owner.get(entry) {
                    Some(_) if precise.iter().any(|&(a, n)| a < at + (1 << 16) && at < a + n) => {}
                    Some(Some(k)) if copies[pieces[*k].0].is_loose() && !loose_back() => kept += 1,
                    Some(Some(k)) => {
                        WRITTEN
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert((at >> 16) as usize);
                        let (n, known) = copies[pieces[*k].0].copy_back_page(at as usize, slot);
                        tracing::debug!(
                            page = format_args!("{at:#x}"),
                            copy = format_args!("{:#x?}", copies[pieces[*k].0].range()),
                            bytes = n,
                            known,
                            "title memory page written back"
                        );
                        written += 1;
                        unknown += u64::from(!known);
                    }
                    _ => {
                        pages += 1;
                        missed.push(at);
                    }
                }
            }
        }
        if stopped != 0 {
            static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let times = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if times.is_power_of_two() {
                tracing::warn!(
                    times,
                    budget = victory_shader::emit::loop_budget_passes(),
                    limit = victory_shader::emit::loop_limit(),
                    "a page table's shaders had lanes stopped at their loop limit or budget"
                );
            }
        }
        stats(|s| {
            s.stopped += stopped;
            s.written += written;
            s.unknown += unknown;
            s.kept += kept;
            s.missed_pages += pages;
            s.missed_regions += regions;
            let room = 4usize.saturating_sub(s.first_missed.len());
            s.first_missed.extend(missed.into_iter().take(room));
        });
    });
    Some(Map {
        table,
        dirty,
        ranges,
        precise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's lookup (`victory_shader` lowering) over a table's
    /// dwords: the view index and the byte offset in that view.
    fn look_up(dwords: &[u32], address: u64) -> (u32, u32) {
        let (lo, hi) = (address as u32, (address >> 32) as u32);
        let inside = hi < 256;
        let l1 = if inside { hi << 4 | lo >> 28 } else { 0 };
        let block = if inside { dwords[l1 as usize] } else { 0 };
        let entry = block << 12 | (lo >> 16 & 0xFFF);
        let at = 4096 + 2 * entry as usize;
        (dwords[at], lo.wrapping_sub(dwords[at + 1]))
    }

    #[test]
    fn addresses_find_the_copy_holding_them() {
        // A 4 MiB window, a buffer inside it from 0x40_0000+0x1_8000
        // (pages in part), and a buffer above 4 GiB.
        let copies = [
            (0x40_0000, 0x40_0000),
            (0x41_8000, 0x3_0000),
            (0xd_1234_0000, 0x2_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, 100);
        // Window, whole pages of the inner buffer, the high buffer.
        assert_eq!(look_up(&dwords, 0x40_0010), (102, 0x10));
        assert_eq!(look_up(&dwords, 0x42_0004), (104, 0x8004));
        assert_eq!(look_up(&dwords, 0xd_1235_fffc), (106, 0x1_fffc));
        // Pages the inner buffer holds in part stay the window's.
        assert_eq!(look_up(&dwords, 0x41_8000), (102, 0x1_8000));
        assert_eq!(look_up(&dwords, 0x44_0000), (102, 0x4_0000));
        // No copy, and above 2^40: the null pair, which reads nothing.
        assert_eq!(look_up(&dwords, 0x9000_0000).0, 100);
        assert_eq!(look_up(&dwords, 0x200_0000_0000).0, 100);
        // Entries map back to their copy and page.
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK
            + (a >> 16 & 0xFFF) as usize;
        assert_eq!(owner[entry(0x42_0000)], Some(1));
        assert_eq!(owner[entry(0xd_1234_0000)], Some(2));
        assert_eq!(blocks.len(), 3, "null, low 256 MiB, the high buffer's");
    }

    /// Page tables of a heap share one run of copy views: a piece keeps
    /// its pair, a new one takes the next, a full run is replaced and goes
    /// back once its lists are done with it.
    #[test]
    fn page_tables_share_a_run_of_copy_views() {
        let Some(heap) = crate::descriptor::host::HostHeap::new(false, 4) else {
            return;
        };
        let memory = vec![0u8; 4 << 16];
        let base = (memory.as_ptr() as u64).next_multiple_of(1 << 16);
        let copies: Vec<Arc<Shadow>> = (0..2)
            .map(|i| {
                let start = base + i * 0x1_0000;
                crate::shadow::register(start as usize, start, 0x1_0000);
                crate::shadow::find(start, 16).expect("a copy")
            })
            .collect();
        let views = CopyViews::default();
        // Pieces of 16 bytes inside the 64 KiB copies: a view past its
        // buffer removes the device.
        let piece = |k: usize, offset: u64| (copies[k].clone(), offset, 0x10);
        let (null, a, first) = views.pairs(&heap, &[piece(0, 0), piece(1, 0)]).unwrap();
        assert_eq!(a, [null + 2, null + 4]);
        let (again, b, second) = views
            .pairs(&heap, &[piece(1, 0), piece(0, 0x100), piece(0, 0)])
            .unwrap();
        assert_eq!((again, b), (null, vec![null + 4, null + 6, null + 2]));
        assert!(Arc::ptr_eq(&first, &second), "the same run");
        // More pieces than the run holds: a new run, the old one kept
        // while a list holds it.
        let many: Vec<_> = (0..1100)
            .map(|i| piece(i % 2, (i / 2) as u64 * 0x10))
            .collect();
        let (null2, c, third) = views.pairs(&heap, &many).unwrap();
        assert_ne!(null2, null);
        assert_eq!(c[1], null2 + 4);
        assert_eq!(views.0.lock().unwrap().retired.len(), 1);
        drop((first, second));
        let (_, d, fourth) = views.pairs(&heap, &many).unwrap();
        assert_eq!(d, c);
        assert!(views.0.lock().unwrap().retired.is_empty(), "given back");
        // Mostly pieces gone: a new run too.
        let (null3, _, fifth) = views.pairs(&heap, &many[..3]).unwrap();
        assert_ne!(null3, null2);
        drop((third, fourth));
        views.pairs(&heap, &many[..3]).unwrap();
        assert!(views.0.lock().unwrap().retired.is_empty(), "given back");
        drop(fifth);
        for i in 0..2 {
            let start = base + i * 0x1_0000;
            crate::shadow::unregister(start as usize, start);
        }
    }
}
