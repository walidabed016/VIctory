//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`], [`title_memory_unmapping`]), which resets
//! their protection.
//!
//! A physical page can show at several addresses (the title's pool view and
//! its tiled heap, which the GPU maps the pool's pages into), and the
//! runtime writes pages through views of its own (DirectStorage's page
//! destinations). The runtime tells which page each block of its views
//! shows ([`title_memory_shows`]) and when it writes pages
//! ([`physical_pages_writing`]): a block guarded clean has every other
//! block showing its page guarded too, and a write through any of them, or
//! to the page, makes them all written.
//!
//! shortcut: a system call writing into a guarded block fails (none is
//! known to; the runtime opens the title's own reads first).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked about since [`forget_asked`] (in order, apart),
    /// and whether they are view memory (guarded, so only askers that guard
    /// may skip them). Asked again before the title can have written them
    /// for this submission, the kernel (or the guard) only finds writes
    /// later submissions read, and their own askings take those; the
    /// world's overlapping copies of its big buffers asked about the same
    /// pages ~8 GB a frame.
    fresh: Vec<(usize, usize, bool)>,
    /// Our own writes since [`forget_asked`]: (asking, start, end).
    ours: Vec<(u64, usize, usize)>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    fresh: Vec::new(),
    ours: Vec::new(),
});

/// The title may have written its memory since (a submission begins, its
/// copies into title memory ran, or its shaders' writes go back): every
/// range is asked about again.
pub(crate) fn forget_asked() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.fresh.clear();
    j.ours.clear();
}

/// The latest asking's number.
pub(crate) fn asking() -> u64 {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).asked
}

/// Whether we wrote title memory in `[start, end)` ([`wrote`]) after
/// asking `asked`, since [`forget_asked`].
pub(crate) fn wrote_since(asked: u64, start: usize, end: usize) -> bool {
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ours
        .iter()
        .rev()
        .take_while(|w| w.0 > asked)
        .any(|w| w.1 < end && w.2 > start)
}

/// We wrote `[start, end)` of title memory: shaders in place (no CPU write
/// the kernel sees) or a copy at submission (seen by the kernel only at the
/// next asking of a range not asked about since [`forget_asked`]). Its pages
/// count as written now.
pub(crate) fn wrote(start: usize, end: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.asked += 1;
    let now = j.asked;
    for page in start / PAGE..end.div_ceil(PAGE) {
        j.written.insert(page, now);
    }
    j.ours.push((now, start, end));
    // So are the blocks showing the same physical pages elsewhere.
    let mut g = guards();
    let Some(shown) = &g.shown else {
        return;
    };
    let aliases: Vec<usize> = (start / BLOCK..end.div_ceil(BLOCK))
        .flat_map(|b| shown.aliases(b))
        .collect();
    for a in aliases {
        g.clean.remove(a, a + 1);
        j.written.insert(a * BLOCK / PAGE, now);
        j.ours.push((now, a * BLOCK, (a + 1) * BLOCK));
    }
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// the latest asking's number, which no page in the list exceeds (pages
/// asked about since [`forget_asked`] are not asked about again). `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let mut gaps = Vec::new();
    let mut at = first;
    for &(a, b, view) in &j.fresh {
        if b <= at || view && !guarded {
            continue;
        }
        if a >= last {
            break;
        }
        if a > at {
            gaps.push((at, a));
        }
        at = b;
    }
    if at < last {
        gaps.push((at, last));
    }
    // Pools are guarded as views are, not asked about ([`guard_pools`]).
    let pools: Vec<(usize, usize)> = if guarded && guard_pools() {
        let g = guards();
        gaps.iter()
            .flat_map(|&(a, b)| g.pools.within(a, b))
            .collect()
    } else {
        Vec::new()
    };
    let mut pieces = Vec::new();
    for (from, to) in gaps {
        let mut at = from;
        for &(a, b) in pools.iter().filter(|&&(a, b)| from <= a && b <= to) {
            if at < a {
                pieces.push((at, a, false));
            }
            pieces.push((a, b, true));
            at = b;
        }
        if at < to {
            pieces.push((at, to, false));
        }
    }
    for (from, to, pool) in pieces {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let asked = if pool {
            None
        } else {
            ask(&mut j.found, from, to)
        };
        let (failed, count) = (asked.is_none(), asked.unwrap_or(0));
        let guarded = if failed {
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let written = if pool {
                Some(guard_pool(&mut j.found, from, to))
            } else {
                guard(from * PAGE, to * PAGE)
            };
            #[cfg(feature = "diagnostics")]
            crate::copystat::cost(
                crate::copystat::Op::Guard,
                started,
                from * PAGE,
                ((to - from) * PAGE) as u64,
            );
            written?
        } else {
            Vec::new()
        };
        let at = j.fresh.partition_point(|&(a, ..)| a < from);
        j.fresh.insert(at, (from, to, failed));
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        j.asked += 1;
        let now = j.asked;
        let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
        for page in pages.collect::<Vec<_>>() {
            j.written.insert(page, now);
        }
    }
    Some((
        j.asked,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// `VirtualQuery` of `at` into `info`: what the system reports of the
/// region holding it. It walks the page tables from `at` to the region's
/// end (~17 us per MiB of the title's big pools).
pub(crate) fn query(
    at: usize,
    info: &mut windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION,
) -> usize {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe {
        windows_sys::Win32::System::Memory::VirtualQuery(at as *const _, info, size_of_val(info))
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::cost(
        crate::copystat::Op::Query,
        started,
        at,
        (info.BaseAddress as usize + info.RegionSize).saturating_sub(at) as u64,
    );
    n
}

/// Whether every kernel page of `[start, end)` is in the working set or a
/// committed page of a view, as the working set tells (a fraction of a us
/// a page, many pages a call; `VirtualQuery` walks the page tables to the
/// region's end, ~10 ms from the middle of a 1 GiB view). False says
/// nothing: private pages not touched yet are committed too.
pub(crate) fn resident(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::ProcessStatus::{
        K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const VALID: usize = 1;
    const SHARED: usize = 1 << 15;
    let pages = start / KERNEL_PAGE..end.div_ceil(KERNEL_PAGE);
    let mut asked: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = Vec::new();
    for first in pages.clone().step_by(4096) {
        asked.clear();
        asked.extend((first..pages.end.min(first + 4096)).map(|page| {
            // SAFETY: plain data, the address filled in here.
            let mut entry: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
            entry.VirtualAddress = (page * KERNEL_PAGE) as *mut _;
            entry
        }));
        // SAFETY: `asked` holds entries of the size given; the system fills
        // in their flags.
        let told = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                asked.as_mut_ptr().cast(),
                size_of_val(asked.as_slice()) as u32,
            ) != 0
        };
        let answered = |e: &PSAPI_WORKING_SET_EX_INFORMATION| {
            // SAFETY: the union's flags are set for valid and invalid pages
            // alike.
            let flags = unsafe { e.VirtualAttributes.Flags };
            flags & (VALID | SHARED) != 0
        };
        if !told || !asked.iter().all(answered) {
            return false;
        }
    }
    true
}

/// Whether `at` is view memory. A committed page of a view is shared in
/// the working set's terms, which the system tells for the page alone
/// (~0.3 us); asking `VirtualQuery` for every guarded asking walked the
/// title's big views to their ends, ~0.3 s per world frame. Other pages are
/// told apart without it where they are write-watched (private memory).
pub(crate) fn view(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{GetWriteWatch, MEM_MAPPED, MEMORY_BASIC_INFORMATION};
    use windows_sys::Win32::System::ProcessStatus::{
        K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const SHARED: usize = 1 << 15;
    // SAFETY: plain data, the address filled in below.
    let mut page: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
    page.VirtualAddress = at as *mut _;
    const VALID: usize = 1;
    // SAFETY: one entry of the right size; the union's flags are set for
    // valid and invalid pages alike.
    let flags = unsafe {
        match K32QueryWorkingSetEx(
            GetCurrentProcess(),
            (&raw mut page).cast(),
            size_of_val(&page) as u32,
        ) {
            0 => 0,
            _ => page.VirtualAttributes.Flags,
        }
    };
    if flags & SHARED != 0 {
        return true;
    }
    // Resident and not shared: private (a title copy's destination, ~9,000
    // a frame, is told apart by this alone).
    if flags & VALID != 0 {
        return false;
    }
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; without the reset flag nothing changes.
    let watched = unsafe {
        GetWriteWatch(
            0,
            (at & !(KERNEL_PAGE - 1)) as *const _,
            KERNEL_PAGE,
            (&raw mut found).cast(),
            &mut count,
            &mut granularity,
        ) == 0
    };
    if watched {
        return false;
    }
    // Pages of a view not committed yet (sections whose pages the runtime
    // commits as the title maps them), placeholders, unwatched memory.
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    query(at, &mut info) != 0 && info.Type == MEM_MAPPED
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: Blocks,
    /// Of those, the ones not written since a copy took them; and blocks
    /// not mapped at all (placeholders in the title's tiled heaps), which
    /// read as zeros until the runtime maps them ([`title_memory_remapped`]).
    clean: Blocks,
    /// Blocks armed for the submission's copy-back ([`arm`]): the first
    /// write to one saves its bytes before it is opened.
    armed: Blocks,
    /// Those saved, by block.
    saved: BTreeMap<usize, Vec<u8>>,
    /// Write-watched title memory guarded instead ([`title_memory_watched`]).
    pools: Blocks,
    /// The physical pages the runtime's views show.
    shown: Option<Shown>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: Blocks(BTreeMap::new()),
    clean: Blocks(BTreeMap::new()),
    armed: Blocks(BTreeMap::new()),
    saved: BTreeMap::new(),
    pools: Blocks(BTreeMap::new()),
    shown: None,
});

/// The physical pages blocks of the runtime's views show
/// ([`title_memory_shows`]), by its numbering of them.
#[derive(Default)]
struct Shown {
    /// By block: the page it shows.
    page: HashMap<usize, u64>,
    /// By page: the blocks showing it.
    blocks: HashMap<u64, Vec<usize>>,
    /// Pages the runtime writes through views of its own, with how many
    /// such writes are under way ([`physical_pages_writing`]).
    writing: HashMap<u64, usize>,
}

impl Shown {
    /// The blocks other than `block` showing the page it shows.
    fn aliases(&self, block: usize) -> Vec<usize> {
        let Some(page) = self.page.get(&block) else {
            return Vec::new();
        };
        self.blocks[page]
            .iter()
            .copied()
            .filter(|&b| b != block)
            .collect()
    }

    /// Whether the runtime is writing the page `block` shows.
    fn being_written(&self, block: usize) -> bool {
        self.page
            .get(&block)
            .is_some_and(|page| self.writing.contains_key(page))
    }
}

/// Held to read an armed block in place ([`in_place`]) and to arm blocks
/// ([`arm`]); the runtime's unmappings wait for it, and it for them
/// ([`title_memory_unmapping`]).
static MAPPING: RwLock<()> = RwLock::new(());

/// Unmappings done: armed runs read before the latest may hold blocks it
/// unmapped.
static UNMAPPED: AtomicU64 = AtomicU64::new(0);

/// A set of blocks as runs (first, end), apart and in order, so that a
/// check of a range costs what it holds of runs, not its blocks (the
/// world's root views reach 36 MiB of guarded memory from each bind).
struct Blocks(BTreeMap<usize, usize>);

impl Blocks {
    fn contains(&self, block: usize) -> bool {
        self.0
            .range(..=block)
            .next_back()
            .is_some_and(|(_, &end)| block < end)
    }

    /// The runs that hold blocks of `[from, to)`, whole.
    fn touching(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let first = self
            .0
            .range(..from)
            .next_back()
            .filter(|(_, end)| **end > from);
        first
            .into_iter()
            .chain(self.0.range(from..to))
            .map(|(&a, &b)| (a, b))
            .collect()
    }

    /// The runs of `[from, to)` it holds, cut to it.
    fn within(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let runs = self.touching(from, to).into_iter();
        runs.map(|(a, b)| (a.max(from), b.min(to))).collect()
    }

    /// The runs of `[from, to)` it does not hold.
    fn gaps(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut gaps = Vec::new();
        let mut at = from;
        for (a, b) in self.within(from, to) {
            if at < a {
                gaps.push((at, a));
            }
            at = b;
        }
        if at < to {
            gaps.push((at, to));
        }
        gaps
    }

    fn insert(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let (mut from, mut to) = (from, to);
        // Runs it overlaps or meets join it.
        let met: Vec<(usize, usize)> = self.touching(from.saturating_sub(1), to + 1);
        for (a, b) in met {
            self.0.remove(&a);
            (from, to) = (from.min(a), to.max(b));
        }
        self.0.insert(from, to);
    }

    fn remove(&mut self, from: usize, to: usize) {
        for (a, b) in self.touching(from, to) {
            self.0.remove(&a);
            if a < from {
                self.0.insert(a, from);
            }
            if b > to {
                self.0.insert(to, b);
            }
        }
    }
}

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Asks the kernel which kernel pages of write-watched pages `[from, to)`
/// were written since it was last asked, and forgets them: their
/// addresses in `found`, and how many; `None` if the memory is not
/// write-watched.
fn ask(found: &mut Vec<usize>, from: usize, to: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (to - from) * PAGE / KERNEL_PAGE;
    found.resize(count.max(found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (from * PAGE) as *const _,
            (to - from) * PAGE,
            found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{Op, cost};
        cost(Op::Ask, started, from * PAGE, ((to - from) * PAGE) as u64);
        if !failed && count > 0 {
            cost(
                Op::Found,
                started,
                from * PAGE,
                (count * KERNEL_PAGE) as u64,
            );
        }
    }
    (!failed).then_some(count)
}

/// For view memory: [`reguard`]. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    view(start).then(|| reguard(start, end, false).0)
}

/// For pool pages `[from, to)`: [`reguard`]; the kernel answers for the
/// blocks that cannot be guarded (not wholly committed).
fn guard_pool(found: &mut Vec<usize>, from: usize, to: usize) -> Vec<usize> {
    let (mut written, left) = reguard(from * PAGE, to * PAGE, true);
    for (a, b) in left {
        let n = ask(found, a * BLOCK / PAGE, b * BLOCK / PAGE).unwrap_or(0);
        written.extend(found[..n].iter().map(|&at| at / PAGE));
    }
    written
}

/// The pages of `[start, end)`'s blocks written since they were last
/// guarded (all of a block's pages; all of those never guarded), guarding
/// them again before the caller copies them. Of a `pool`, the blocks that
/// cannot be guarded are left out, as runs.
fn reguard(start: usize, end: usize, pool: bool) -> (Vec<usize>, Vec<(usize, usize)>) {
    install();
    let mut g = guards();
    let (mut written, mut left) = (Vec::new(), Vec::new());
    for (from, to) in g.clean.gaps(start / BLOCK, end.div_ceil(BLOCK)) {
        // Blocks still read-only were written through the runtime: clean
        // again. The others are guarded again.
        let mut at = from;
        for (a, b) in g.protected.within(from, to) {
            if at < a {
                protect(&mut g, at, a, pool);
            }
            g.clean.insert(a, b);
            at = b;
        }
        if at < to {
            protect(&mut g, at, to, pool);
        }
        if !pool {
            guard_aliases(&mut g, from, to);
        }
        let unguarded = match pool {
            true => g.protected.gaps(from, to),
            false => Vec::new(),
        };
        written.extend(
            (from * BLOCK / PAGE..to * BLOCK / PAGE)
                .filter(|&p| !unguarded.iter().any(|&(a, b)| a <= p && p < b)),
        );
        left.extend(unguarded);
    }
    (written, left)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views. A block not committed (unmapped) is
/// noted clean, unguarded: copies took its zeros, and every asking used to
/// report it written again, so each refresh of the world's sparse tiled
/// heaps copied ~300 MB of unmapped pages a frame, 4 KiB at a time. Of a
/// `pool`, only a block wholly not committed is (the runtime tells when it
/// commits pool memory).
fn protect(g: &mut Guards, from: usize, to: usize, pool: bool) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION};
    let set = read_only;
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.insert(from, to);
        g.clean.insert(from, to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            continue;
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = query(block * BLOCK, &mut info);
        let whole = !pool || info.BaseAddress as usize + info.RegionSize >= (block + 1) * BLOCK;
        if n != 0 && info.State != MEM_COMMIT && whole {
            g.clean.insert(block, block + 1);
        }
    }
}

/// Makes blocks `[from, to)` read-only; false if they are not all
/// committed read-write or read-only memory of one view or allocation.
fn read_only(from: usize, to: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: title memory; only its protection changes.
    let mut set = unsafe {
        VirtualProtect(
            (from * BLOCK) as *const _,
            (to - from) * BLOCK,
            PAGE_READONLY,
            &mut old,
        ) != 0
    };
    // Pools take the title's protection: memory it can execute (or
    // otherwise not just read or write) is not guarded, since our
    // handler gives write access back as plain read-write. Read-only
    // is ours, guarded before and not opened since, or the title's.
    if set && old != PAGE_READWRITE && old != PAGE_READONLY {
        // SAFETY: as above, back as it was.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                old,
                &mut old,
            )
        };
        set = false;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::cost(
        crate::copystat::Op::Protect,
        started,
        from * BLOCK,
        ((to - from) * BLOCK) as u64,
    );
    set
}

/// Blocks `[from, to)` were just guarded (or found still guarded) and
/// noted clean: the other blocks showing their physical pages are guarded
/// too, not noted clean (what they show was not taken through them), so
/// that a write through any of them is seen. A block is not clean if one of
/// those cannot be guarded, or while the runtime writes its page.
fn guard_aliases(g: &mut Guards, from: usize, to: usize) {
    let Some(shown) = &g.shown else {
        return;
    };
    let aliased: Vec<(usize, Vec<usize>, bool)> = (from..to)
        .filter(|b| shown.page.contains_key(b))
        .map(|b| (b, shown.aliases(b), shown.being_written(b)))
        .collect();
    for (block, aliases, being_written) in aliased {
        if !g.clean.contains(block) || !g.protected.contains(block) {
            continue;
        }
        let mut guarded = !being_written;
        for a in aliases {
            if !guarded {
                break;
            }
            if g.protected.contains(a) {
                continue;
            }
            guarded = read_only(a, a + 1);
            if guarded {
                g.protected.insert(a, a + 1);
            }
        }
        if !guarded {
            g.clean.remove(block, block + 1);
        }
    }
}

/// Blocks `[from, to)` are about to be written: so are the other blocks
/// showing their physical pages, which are no longer clean, the armed ones
/// saved first.
fn aliases_written(g: &mut Guards, from: usize, to: usize) {
    let Some(shown) = &g.shown else {
        return;
    };
    let aliases: Vec<usize> = (from..to).flat_map(|b| shown.aliases(b)).collect();
    for a in aliases {
        save(g, a, a + 1);
        g.clean.remove(a, a + 1);
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let runs = g.protected.within(start / BLOCK, end.div_ceil(BLOCK));
    for &(from, to) in &runs {
        save(g, from, to);
        g.protected.remove(from, to);
        g.clean.remove(from, to);
        for block in from..to {
            let mut old = 0;
            // SAFETY: a block we made read-only.
            unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
        }
        aliases_written(g, from, to);
    }
    !runs.is_empty()
}

/// Saves the bytes of the armed blocks of guarded blocks `[from, to)` not
/// saved yet (guarded: mapped, and as they were armed).
fn save(g: &mut Guards, from: usize, to: usize) {
    for (a, b) in g.armed.within(from, to) {
        for block in a..b {
            if g.saved.contains_key(&block) {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            // SAFETY: a guarded block: mapped and readable.
            let bytes = unsafe { std::slice::from_raw_parts((block * BLOCK) as *const u8, BLOCK) };
            g.saved.insert(block, bytes.to_vec());
            #[cfg(feature = "diagnostics")]
            crate::copystat::cost(
                crate::copystat::Op::Saved,
                started,
                block * BLOCK,
                BLOCK as u64,
            );
        }
    }
}

/// Arms title memory `[start, end)` for a submission whose shaders write a
/// copy of it, in place of a snapshot of the copy: until [`disarm`], the
/// first write to an armed block saves its bytes first, so what the copy
/// held before the submission is the block's saved bytes or the block
/// itself ([`before`]). Armed are the whole blocks guarded and not written
/// since, nor since the copy took them (`copied`: the asking after which
/// the copy took a page, 0 if never). Returns the asking now and the runs
/// of blocks of the range not armed. Waits for the runtime's unmappings.
pub(crate) fn arm(
    start: usize,
    end: usize,
    copied: impl Fn(usize) -> u64,
) -> (u64, Vec<(usize, usize)>) {
    let _mapping = MAPPING.read().unwrap_or_else(|e| e.into_inner());
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut g = guards();
    let (first, last) = (start.div_ceil(BLOCK), end / BLOCK);
    let mut armed = Vec::new();
    for (a, b) in g.clean.within(first, last.max(first)) {
        for (a, b) in g.protected.within(a, b) {
            let mut run: Option<usize> = None;
            for block in a..=b {
                let current = block < b && {
                    let taken = copied(block * BLOCK / PAGE);
                    let written = j.written.get(&(block * BLOCK / PAGE)).copied().unwrap_or(0);
                    taken != 0 && written <= taken
                };
                match (current, run) {
                    (true, None) => run = Some(block),
                    (false, Some(from)) => {
                        armed.push((from, block));
                        run = None;
                    }
                    _ => {}
                }
            }
        }
    }
    for &(a, b) in &armed {
        g.armed.insert(a, b);
    }
    let (mut loose, mut at) = (Vec::new(), start / BLOCK);
    for (a, b) in armed {
        if at < a {
            loose.push((at, a));
        }
        at = b;
    }
    if at < end.div_ceil(BLOCK) {
        loose.push((at, end.div_ceil(BLOCK)));
    }
    (j.asked, loose)
}

/// `read` of armed block `block` in place, while no unmapping can run;
/// `None` if it is not armed. `seen` keeps the armed runs from one call
/// to the next until an unmapping. An unmapping waits for one block's
/// read, not a whole copy-back's.
pub(crate) fn in_place<T>(
    block: usize,
    seen: &mut Option<(u64, Vec<(usize, usize)>)>,
    read: impl FnOnce() -> T,
) -> Option<T> {
    let _mapping = MAPPING.read().unwrap_or_else(|e| e.into_inner());
    let unmapped = UNMAPPED.load(Ordering::Relaxed);
    if seen.as_ref().is_none_or(|(u, _)| *u != unmapped) {
        let runs = guards().armed.0.iter().map(|(&a, &b)| (a, b)).collect();
        *seen = Some((unmapped, runs));
    }
    let (_, runs) = seen.as_ref()?;
    runs.iter()
        .any(|&(a, b)| a <= block && block < b)
        .then(read)
}

/// What armed block `block` held when armed at asking `asked`: its saved
/// bytes, or the block's own if still guarded and not written since; `None`
/// if neither (written where no guard saw it: shaders in place, or a copy
/// of ours).
pub(crate) fn before(block: usize, asked: u64) -> Option<Vec<u8>> {
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let g = guards();
    if let Some(bytes) = g.saved.get(&block) {
        return Some(bytes.clone());
    }
    let page = block * BLOCK / PAGE;
    let kept = g.armed.contains(block)
        && g.protected.contains(block)
        && g.clean.contains(block)
        && j.written.get(&page).is_none_or(|&w| w <= asked);
    kept.then(|| {
        // SAFETY: an armed guarded block: mapped, and held still by the
        // lock (a write to it waits for it in our handler, a remapping too).
        unsafe { std::slice::from_raw_parts((block * BLOCK) as *const u8, BLOCK) }.to_vec()
    })
}

/// After a submission's copy-back: nothing is armed.
pub(crate) fn disarm() {
    let mut g = guards();
    g.armed = Blocks(BTreeMap::new());
    g.saved.clear();
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped title memory `[at, at + len)`: what it shows is new
/// and read-write, so it is no longer guarded. It unmaps through
/// [`title_memory_unmapping`].
pub fn title_memory_remapped(at: usize, len: usize) {
    unguard(
        &mut guards(),
        at / BLOCK,
        at.saturating_add(len).div_ceil(BLOCK),
        true,
    );
}

/// The runtime unmaps title memory `[at, at + len)` with `unmap`. Armed
/// blocks keep what they held, and nothing of the range is guarded or
/// armed after, though a copy may guard it again while `unmap` runs: a
/// block left guarded with nothing mapped would be armed and read in place.
/// Copies reading armed blocks in place and arming wait for it.
pub fn title_memory_unmapping<R>(at: usize, len: usize, unmap: impl FnOnce() -> R) -> R {
    let _mapping = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    let (from, to) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    unguard(&mut guards(), from, to, true);
    let unmapped = unmap();
    let mut g = guards();
    unguard(&mut g, from, to, false);
    forget_shown(&mut g, from, to);
    drop(g);
    UNMAPPED.fetch_add(1, Ordering::Relaxed);
    unmapped
}

/// The runtime mapped physical pages at title memory `[at, at + len)`,
/// after [`title_memory_remapped`]: block by block, pages `first`,
/// `first + 1`... by its numbering (consecutive pages of an allocation
/// numbered consecutively). The other blocks showing those pages are no
/// longer clean (the new ones are not guarded), the armed ones saved first.
pub fn title_memory_shows(at: usize, len: usize, first: u64) {
    let mut g = guards();
    let (from, to) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    forget_shown(&mut g, from, to);
    let shown = g.shown.get_or_insert_with(Shown::default);
    let mut others = Vec::new();
    for (block, page) in (from..to).zip(first..) {
        let blocks = shown.blocks.entry(page).or_default();
        others.extend_from_slice(blocks);
        blocks.push(block);
        shown.page.insert(block, page);
    }
    for a in others {
        save(&mut g, a, a + 1);
        g.clean.remove(a, a + 1);
    }
}

/// Blocks `[from, to)` show no physical pages any more.
fn forget_shown(g: &mut Guards, from: usize, to: usize) {
    let Some(shown) = &mut g.shown else {
        return;
    };
    // A whole reservation is not walked block by block.
    let gone: Vec<usize> = if to - from > shown.page.len() {
        let within = |b: &&usize| from <= **b && **b < to;
        shown.page.keys().filter(within).copied().collect()
    } else {
        (from..to).filter(|b| shown.page.contains_key(b)).collect()
    };
    for block in gone {
        if let Some(page) = shown.page.remove(&block)
            && let Some(blocks) = shown.blocks.get_mut(&page)
        {
            blocks.retain(|&b| b != block);
            if blocks.is_empty() {
                shown.blocks.remove(&page);
            }
        }
    }
}

/// The runtime is about to write physical pages `[first, first + n)`
/// (numbered as for [`title_memory_shows`]) through a view of its own,
/// until [`physical_pages_written`]: the blocks showing them are not
/// clean, the armed ones saved first, and are not guarded clean until then.
pub fn physical_pages_writing(first: u64, n: usize) {
    let mut g = guards();
    let shown = g.shown.get_or_insert_with(Shown::default);
    let mut blocks = Vec::new();
    for page in first..first + n as u64 {
        *shown.writing.entry(page).or_default() += 1;
        blocks.extend(shown.blocks.get(&page).into_iter().flatten().copied());
    }
    for b in blocks {
        save(&mut g, b, b + 1);
        g.clean.remove(b, b + 1);
    }
}

/// The runtime wrote physical pages `[first, first + n)` after
/// [`physical_pages_writing`]: the blocks showing them count as written.
pub fn physical_pages_written(first: u64, n: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut g = guards();
    let Some(shown) = &mut g.shown else {
        return;
    };
    let mut blocks = Vec::new();
    for page in first..first + n as u64 {
        if let Some(k) = shown.writing.get_mut(&page) {
            *k -= 1;
            if *k == 0 {
                shown.writing.remove(&page);
            }
        }
        blocks.extend(shown.blocks.get(&page).into_iter().flatten().copied());
    }
    if blocks.is_empty() {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    for b in blocks {
        g.clean.remove(b, b + 1);
        j.written.insert(b * BLOCK / PAGE, now);
    }
}

/// Whether every block of `[start, end)` is guarded and clean: not
/// written since a guarded asking took it (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn unwritten(start: usize, end: usize) -> bool {
    let g = guards();
    let (from, to) = (start / BLOCK, end.div_ceil(BLOCK));
    g.clean.gaps(from, to).is_empty() && g.protected.gaps(from, to).is_empty()
}

/// The runtime reserved (`watched`) or released write-watched title memory
/// `[at, at + len)`: xmem's pools, which copies guard as they do views
/// ([`guard_pools`]). It commits pool memory, and changes its protection,
/// through [`title_memory_remapped`]; it decommits and releases it through
/// [`title_memory_unmapping`].
pub fn title_memory_watched(at: usize, len: usize, watched: bool) {
    let mut g = guards();
    let (from, to) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    if watched {
        g.pools.insert(from, to);
    } else {
        g.pools.remove(from, to);
    }
}

/// Whether copies guard pools instead of asking the kernel which pages the
/// title wrote: asking costs ~1 us per MB asked about however few pages
/// were written, and each answer that finds some ~65 us more (its TLB
/// flush reaches every busy thread): ~43 ms a world frame. A guard costs a
/// fault in the title's first write to a block, and ~5 us to guard it
/// again. `VICTORY_WRITE_WATCH_POOLS` asks the kernel.
fn guard_pools() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_WRITE_WATCH_POOLS").is_none();
        if !on {
            tracing::info!(
                "pools are asked about through the write watch (VICTORY_WRITE_WATCH_POOLS)"
            );
        }
        on
    })
}

/// Blocks `[from, to)` are guarded and armed no more; armed blocks still
/// `mapped` keep what they held, saved first.
fn unguard(g: &mut Guards, from: usize, to: usize, mapped: bool) {
    if mapped {
        for (a, b) in g.protected.within(from, to) {
            save(g, a, b);
        }
    }
    g.armed.remove(from, to);
    g.protected.remove(from, to);
    // Unmapped blocks are clean without being guarded.
    g.clean.remove(from, to);
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

/// The journal and guards are the process's: tests that ask or remap (or
/// submit, which asks) run one at a time (one asker's `forget_asked`
/// changes another's answers).
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        let _serial = serial();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        forget_asked();
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        // Asked about already in this submission: the write shows in the
        // next one.
        let (_, same) = writes(start, end, false).expect("watched");
        assert!(same.is_empty());
        forget_asked();
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        forget_asked();
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        let _serial = serial();
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn block_runs_join_and_split() {
        let mut b = Blocks(BTreeMap::new());
        b.insert(10, 20);
        b.insert(20, 25);
        b.insert(30, 40);
        b.insert(5, 12);
        assert_eq!(b.within(0, 100), [(5, 25), (30, 40)]);
        assert_eq!(b.gaps(0, 50), [(0, 5), (25, 30), (40, 50)]);
        b.remove(8, 9);
        b.remove(24, 31);
        assert_eq!(b.within(0, 100), [(5, 8), (9, 24), (31, 40)]);
        assert_eq!(b.within(7, 33), [(7, 8), (9, 24), (31, 33)]);
        assert_eq!(b.gaps(7, 33), [(8, 9), (24, 31)]);
        b.insert(0, 50);
        assert_eq!(b.within(0, 100), [(0, 50)]);
    }

    /// Guards are kept as runs: a check over a big view reports the blocks
    /// written since (by the title, faulting, or through the runtime) and
    /// guards them again, the view staying one run.
    #[test]
    fn guarded_blocks_are_kept_as_runs() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        let _serial = serial();
        const N: usize = 1024;
        // SAFETY: a pagefile section of N blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (N * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, N * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let pages =
            |from: usize, to: usize| (view / PAGE + from..view / PAGE + to).collect::<Vec<_>>();
        assert_eq!(guard(view, view + N * BLOCK), Some(pages(0, N)));
        // SAFETY: our view; our handler opens the guarded blocks.
        unsafe {
            ((view + 5 * BLOCK + 8) as *mut u32).write(1);
            ((view + 6 * BLOCK) as *mut u32).write(2);
        }
        writing(view + 700 * BLOCK, 100);
        let mut written = pages(5, 7);
        written.extend(pages(700, 701));
        assert_eq!(guard(view, view + N * BLOCK), Some(written));
        assert_eq!(guard(view, view + N * BLOCK), Some(Vec::new()));
        let first = view / BLOCK;
        assert_eq!(
            guards().protected.within(first, first + N),
            [(first, first + N)]
        );
        assert_eq!(
            guards().clean.within(first, first + N),
            [(first, first + N)]
        );
        title_memory_remapped(view, N * BLOCK);
        assert!(guards().protected.within(first, first + N).is_empty());
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        };
    }

    /// View memory is told apart from write-watched memory and placeholders
    /// as `VirtualQuery` tells it, without walking regions: committed or not
    /// (sections whose pages are committed as the title maps them), guarded
    /// or not.
    #[test]
    fn view_memory_is_told_apart() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_RESERVE_PLACEHOLDER,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, SEC_RESERVE, UnmapViewOfFile,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        // SAFETY: a section of ours with its first block committed, mapped
        // read-write; a write-watched allocation; a placeholder.
        let (v, p, h) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            assert_eq!(
                VirtualAlloc(v as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) as usize,
                v
            );
            let p = VirtualAlloc(
                std::ptr::null(),
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            let h = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            (v, p, h)
        };
        assert!(view(v), "committed, never touched");
        // SAFETY: the committed block of our view.
        unsafe { (v as *mut u8).write(1) };
        assert!(view(v + 100), "committed, touched");
        assert!(guard(v, v + BLOCK).is_some());
        assert!(view(v), "guarded");
        assert!(view(v + BLOCK), "not committed");
        assert!(!view(p), "write-watched");
        assert!(!view(h), "a placeholder");
        title_memory_remapped(v, 2 * BLOCK);
        // SAFETY: ours.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ });
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
            VirtualFree(h as *mut _, 0, MEM_RELEASE);
        }
    }

    /// An armed block tells what it held when armed: itself while guarded,
    /// its saved bytes once the title writes it or the runtime remaps it,
    /// nothing once something no guard sees wrote it, or after disarming.
    #[test]
    fn armed_blocks_tell_what_they_held() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        let _serial = serial();
        // SAFETY: a pagefile section of four blocks, mapped read-write.
        let v = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4 * BLOCK).Value as usize
        };
        assert_ne!(v, 0);
        // SAFETY: our view, not guarded yet.
        unsafe { std::slice::from_raw_parts_mut(v as *mut u8, 4 * BLOCK).fill(1) };
        forget_asked();
        let (now, _) = writes(v, v + 4 * BLOCK, true).expect("view memory");
        let (asked, loose) = arm(v, v + 4 * BLOCK, |_| now);
        assert!(loose.is_empty());
        let b = v / BLOCK;
        let ones = vec![1u8; BLOCK];
        assert_eq!(before(b, asked), Some(ones.clone()));
        // SAFETY: our view (guarded: our handler saves, then opens).
        unsafe { ((v + BLOCK + 8) as *mut u8).write(2) };
        assert_eq!(before(b + 1, asked), Some(ones.clone()));
        title_memory_remapped(v + 2 * BLOCK, BLOCK);
        assert_eq!(before(b + 2, asked), Some(ones.clone()));
        wrote(v + 3 * BLOCK, v + 4 * BLOCK);
        assert_eq!(before(b + 3, asked), None);
        disarm();
        assert_eq!(before(b, asked), None);
        title_memory_remapped(v, 4 * BLOCK);
        // The view stays: the journal keeps its pages, which another test
        // given the same addresses would read as its own.
    }

    /// A block a copy guards again while the runtime unmaps it is left
    /// unguarded after, so no copy arms it and reads it in place.
    #[test]
    fn unmapped_blocks_are_left_unguarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_RESERVE, MEMORY_MAPPED_VIEW_ADDRESS,
            MapViewOfFile, PAGE_NOACCESS, UnmapViewOfFile, VirtualAlloc,
        };
        let _serial = serial();
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let v = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(v, 0);
        forget_asked();
        let (now, _) = writes(v, v + 2 * BLOCK, true).expect("view memory");
        let b = v / BLOCK;
        title_memory_unmapping(v, 2 * BLOCK, || {
            // A copy asks again before the view goes: guarded again.
            forget_asked();
            writes(v, v + 2 * BLOCK, true).expect("view memory");
            assert_eq!(guards().protected.within(b, b + 2), vec![(b, b + 2)]);
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ }) };
        });
        assert!(guards().protected.within(b, b + 2).is_empty());
        let (_, loose) = arm(v, v + 2 * BLOCK, |_| now);
        assert_eq!(loose, vec![(b, b + 2)]);
        assert!(guards().armed.within(b, b + 2).is_empty());
        // The addresses stay taken: the journal keeps their pages, which
        // another test given them would read as its own.
        // SAFETY: a reservation of the free range.
        let kept = unsafe { VirtualAlloc(v as *const _, 2 * BLOCK, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!kept.is_null());
    }

    /// An armed block is read in place until the runtime unmaps it; the
    /// armed runs a reader kept are read again after an unmapping.
    #[test]
    fn armed_blocks_are_read_in_place_until_unmapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_RESERVE, MEMORY_MAPPED_VIEW_ADDRESS,
            MapViewOfFile, PAGE_NOACCESS, UnmapViewOfFile, VirtualAlloc,
        };
        let _serial = serial();
        // SAFETY: a pagefile section of a block, mapped read-write.
        let v = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize
        };
        assert_ne!(v, 0);
        // SAFETY: our view, not guarded yet.
        unsafe { std::slice::from_raw_parts_mut(v as *mut u8, BLOCK).fill(3) };
        forget_asked();
        let (now, _) = writes(v, v + BLOCK, true).expect("view memory");
        let (_, loose) = arm(v, v + BLOCK, |_| now);
        assert!(loose.is_empty());
        let b = v / BLOCK;
        let mut seen = None;
        // SAFETY: an armed block, mapped while it is read.
        let read = || unsafe { (v as *const u8).read() };
        assert_eq!(in_place(b, &mut seen, read), Some(3));
        assert_eq!(in_place(b + 1, &mut seen, read), None);
        // SAFETY: our view.
        title_memory_unmapping(v, BLOCK, || unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ })
        });
        assert_eq!(in_place(b, &mut seen, || unreachable!()), None::<u8>);
        disarm();
        // The address stays taken: the journal keeps its pages, which
        // another test given it would read as its own.
        // SAFETY: a reservation of the free range.
        let kept = unsafe { VirtualAlloc(v as *const _, BLOCK, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!kept.is_null());
    }

    /// A pool is guarded as views are, without asking the kernel: a block
    /// counts as written when first guarded and after the title's write;
    /// a decommitted block is clean until it is committed again (read-write,
    /// then written and guarded again). A block partly committed is left to
    /// the kernel, as are askers that do not guard.
    #[test]
    fn pools_are_guarded_instead_of_asked_about() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RESERVE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let _serial = serial();
        // SAFETY: a write-watched reservation of five blocks, four of them
        // committed and the second half of the fifth.
        let p = unsafe {
            let p = VirtualAlloc(
                std::ptr::null(),
                5 * BLOCK,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            assert_ne!(p, 0);
            let half = (p + 4 * BLOCK + BLOCK / 2) as *const _;
            assert!(!VirtualAlloc(p as *const _, 4 * BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert!(!VirtualAlloc(half, BLOCK / 2, MEM_COMMIT, PAGE_READWRITE).is_null());
            p
        };
        title_memory_watched(p, 5 * BLOCK, true);
        let b = p / BLOCK;
        let pages = |w: &[(usize, u64)], at: u64| -> Vec<usize> {
            w.iter()
                .filter(|&&(_, a)| a == at)
                .map(|&(page, _)| page)
                .collect()
        };
        forget_asked();
        let (first, w) = writes(p, p + 4 * BLOCK, true).expect("a pool");
        assert_eq!(pages(&w, first), vec![b, b + 1, b + 2, b + 3]);
        assert_eq!(guards().protected.within(b, b + 5), vec![(b, b + 4)]);
        assert!(guards().clean.within(b + 4, b + 5).is_empty());
        // SAFETY: our pool (guarded: our handler opens the block; the
        // fifth is not guarded).
        unsafe {
            ((p + BLOCK + 8) as *mut u8).write(1);
            ((p + 4 * BLOCK + BLOCK / 2 + 8) as *mut u8).write(1);
        }
        forget_asked();
        let (now, w) = writes(p, p + 5 * BLOCK, true).expect("a pool");
        assert_eq!(pages(&w, now), vec![b + 1, b + 4]);
        assert!(writes(p, p + 4 * BLOCK, false).is_some());
        // SAFETY: our pool's third block, decommitted.
        title_memory_unmapping(p + 2 * BLOCK, BLOCK, || unsafe {
            VirtualFree((p + 2 * BLOCK) as *mut _, BLOCK, MEM_DECOMMIT)
        });
        forget_asked();
        writes(p, p + 4 * BLOCK, true).expect("a pool");
        forget_asked();
        let (now, w) = writes(p, p + 4 * BLOCK, true).expect("a pool");
        assert!(pages(&w, now).is_empty());
        assert_eq!(guards().clean.within(b + 2, b + 3), vec![(b + 2, b + 3)]);
        // SAFETY: committed again, as the runtime's hooks tell.
        let c = unsafe {
            VirtualAlloc(
                (p + 2 * BLOCK) as *const _,
                BLOCK,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_eq!(c, p + 2 * BLOCK);
        title_memory_remapped(c, BLOCK);
        // SAFETY: our pool, read-write and not guarded.
        unsafe { ((c + 8) as *mut u8).write(1) };
        forget_asked();
        let (now, w) = writes(p, p + 4 * BLOCK, true).expect("a pool");
        assert_eq!(pages(&w, now), vec![b + 2]);
        assert_eq!(guards().protected.within(b, b + 4), vec![(b, b + 4)]);
        // SAFETY: our pool, decommitted; the reservation keeps its addresses
        // from other tests, whose pages the journal would mix up with ours.
        title_memory_unmapping(p, 5 * BLOCK, || unsafe {
            VirtualFree(p as *mut _, 5 * BLOCK, MEM_DECOMMIT)
        });
        title_memory_watched(p, 5 * BLOCK, false);
    }

    /// Committed memory is told by the working set where it answers (view
    /// pages, touched pages), by `VirtualQuery` where it does not.
    #[test]
    fn committed_memory_is_told_by_the_working_set() {
        use crate::texture::committed;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };
        const LEN: usize = 256 << 20;
        // SAFETY: a section of ours with its first half committed, mapped
        // read-write; private memory committed and not touched.
        let (v, p) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize;
            assert_eq!(
                VirtualAlloc(v as *const _, LEN / 2, MEM_COMMIT, PAGE_READWRITE) as usize,
                v
            );
            let p = VirtualAlloc(
                std::ptr::null(),
                BLOCK,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            (v, p as usize)
        };
        let mid = v + LEN / 4;
        assert!(
            resident(mid, mid + (4 << 20)),
            "committed view pages, not touched"
        );
        assert!(committed(mid as *const u8, 4 << 20));
        assert!(!resident(v + LEN / 2 - 4096, v + LEN / 2 + 4096));
        assert!(!committed((v + LEN / 2 - 4096) as *const u8, 8192));
        assert!(!committed((v + LEN / 2) as *const u8, 4096));
        assert!(!resident(p, p + BLOCK), "private, not touched: no answer");
        assert!(committed(p as *const u8, BLOCK));
        // SAFETY: ours.
        unsafe { (p as *mut u8).write(1) };
        assert!(resident(p, p + 4096), "touched");
        assert!(!view(p), "private, touched");
        // SAFETY: ours.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ });
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let _serial = serial();
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// A block of view memory with nothing mapped (a placeholder in a tiled
    /// heap) is taken as written once (copies read its zeros), then clean
    /// until the runtime maps memory there.
    #[test]
    fn unmapped_view_blocks_are_reported_once() {
        let _serial = serial();
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let null = std::ptr::null_mut();
        // SAFETY: a placeholder of two blocks of ours; the first becomes a
        // view of a pagefile section, the second stays a placeholder.
        let p = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                null,
                0,
            ) as usize;
            assert_ne!(p, 0);
            VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                null,
                0,
            );
            assert_eq!(view.Value as usize, p);
            p
        };
        let (mapped, unmapped) = (p / PAGE, p / PAGE + 1);
        let last = |w: &[(usize, u64)], page: usize| w.iter().find(|e| e.0 == page).map(|e| e.1);
        forget_asked();
        let (first, w) = writes(p, p + 2 * BLOCK, true).expect("view memory");
        assert_eq!(last(&w, mapped), Some(first));
        assert_eq!(last(&w, unmapped), Some(first));
        // Asked about in this submission already: not asked again.
        let (_, w) = writes(p, p + 2 * BLOCK, true).expect("view memory");
        assert_eq!(last(&w, unmapped), Some(first));
        forget_asked();
        let (_, w) = writes(p, p + 2 * BLOCK, true).expect("view memory");
        assert_eq!(last(&w, unmapped), Some(first), "unmapped: clean");
        assert_eq!(last(&w, mapped), Some(first), "guarded, not written");
        // The runtime maps memory there: taken again.
        title_memory_remapped(p + BLOCK, BLOCK);
        forget_asked();
        let (third, w) = writes(p, p + 2 * BLOCK, true).expect("view memory");
        // Askers that do not guard get no answer for view memory, though
        // guarded askers asked about it in this submission.
        assert!(writes(p, p + 2 * BLOCK, false).is_none());
        assert_eq!(last(&w, unmapped), Some(third));
        assert_eq!(last(&w, mapped), Some(first));
        title_memory_remapped(p, 2 * BLOCK);
        // The memory stays: the journal keeps its pages, which another test
        // given the same addresses would read as its own.
    }

    /// A pagefile section of `blocks` blocks mapped twice, read-write.
    fn two_views(blocks: usize) -> (usize, usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a section of ours, mapped wherever the system puts it.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (blocks * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let x = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK).Value as usize;
            let y = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK).Value as usize;
            CloseHandle(section);
            assert!(x != 0 && y != 0);
            (x, y)
        }
    }

    /// Unmaps views of `blocks` blocks as the runtime does, keeping their
    /// addresses: the journal keeps their pages, which another test given
    /// them would read as its own.
    fn let_go(views: &[usize], blocks: usize) {
        use windows_sys::Win32::System::Memory::{
            MEMORY_MAPPED_VIEW_ADDRESS, PAGE_NOACCESS, UnmapViewOfFile,
        };
        for &v in views {
            // SAFETY: a view of ours.
            title_memory_unmapping(v, blocks * BLOCK, || unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ })
            });
            // SAFETY: a reservation of the range just freed.
            let kept =
                unsafe { VirtualAlloc(v as *const _, blocks * BLOCK, MEM_RESERVE, PAGE_NOACCESS) };
            assert!(!kept.is_null());
        }
    }

    /// The blocks of `[at, at + blocks)` a guarded asking in a new
    /// submission finds written, by index.
    fn written_blocks(at: usize, blocks: usize) -> Vec<usize> {
        forget_asked();
        let (now, w) = writes(at, at + blocks * BLOCK, true).expect("view memory");
        w.iter()
            .filter(|&&(_, a)| a == now)
            .map(|&(page, _)| page - at / PAGE)
            .collect()
    }

    /// Physical pages shown at two addresses: a write through either one,
    /// faulting, through the kernel or a copy of ours, is a write to the
    /// other, which a copy of it guards too.
    #[test]
    fn writes_through_another_address_are_seen() {
        let _serial = serial();
        let (x, y) = two_views(4);
        const PAGES: u64 = 0x7e58 << 32;
        title_memory_shows(x, 4 * BLOCK, PAGES);
        title_memory_shows(y, 4 * BLOCK, PAGES);
        assert_eq!(written_blocks(x, 4), [0, 1, 2, 3]);
        assert!(written_blocks(x, 4).is_empty());
        let yb = y / BLOCK;
        assert_eq!(
            guards().protected.within(yb, yb + 4),
            [(yb, yb + 4)],
            "y is guarded with x"
        );
        assert!(
            guards().clean.within(yb, yb + 4).is_empty(),
            "but not clean"
        );
        // SAFETY: our view (guarded: our handler opens the block).
        unsafe { ((y + BLOCK + 8) as *mut u32).write(1) };
        assert_eq!(written_blocks(x, 4), [1]);
        writing(y + 2 * BLOCK, 10);
        assert_eq!(written_blocks(x, 4), [2]);
        wrote(y + 3 * BLOCK, y + 3 * BLOCK + 1);
        assert_eq!(written_blocks(x, 4), [3]);
        assert!(written_blocks(x, 4).is_empty());
        // SAFETY: our view, guarded.
        unsafe { ((x + 8) as *mut u32).write(2) };
        assert_eq!(written_blocks(y, 4), [0, 1, 2, 3], "y's first asking");
        // SAFETY: our view, guarded.
        unsafe { ((x + 8) as *mut u32).write(3) };
        assert_eq!(written_blocks(y, 4), [0]);
        // SAFETY: the word written through x, read through y.
        assert_eq!(unsafe { ((y + 8) as *const u32).read() }, 3);
        let_go(&[x, y], 4);
        assert!(
            guards()
                .shown
                .as_ref()
                .is_none_or(|s| !s.page.contains_key(&(x / BLOCK)))
        );
    }

    /// While the runtime writes physical pages through a view of its own,
    /// the blocks showing them are written at every asking and none is
    /// armed; once it is done, they are written once more.
    #[test]
    fn physical_page_writes_are_seen() {
        let _serial = serial();
        // y stands for the runtime's own view.
        let (x, y) = two_views(2);
        const PAGES: u64 = 0x7e59 << 32;
        title_memory_shows(x, 2 * BLOCK, PAGES);
        assert_eq!(written_blocks(x, 2), [0, 1]);
        assert!(written_blocks(x, 2).is_empty());
        physical_pages_writing(PAGES + 1, 1);
        assert_eq!(written_blocks(x, 2), [1]);
        assert_eq!(written_blocks(x, 2), [1]);
        let (_, loose) = arm(x, x + 2 * BLOCK, |_| u64::MAX - 1);
        let b = x / BLOCK;
        assert_eq!(loose, [(b + 1, b + 2)]);
        disarm();
        // SAFETY: the runtime's view, not guarded.
        unsafe { ((y + BLOCK + 8) as *mut u32).write(5) };
        physical_pages_written(PAGES + 1, 1);
        assert_eq!(written_blocks(x, 2), [1]);
        assert!(written_blocks(x, 2).is_empty());
        let_go(&[x, y], 2);
    }

    /// A page mapped at a new address is no longer clean where it showed
    /// already (nothing guards the new address yet); unmapped, the address
    /// no longer counts.
    #[test]
    fn a_new_address_of_a_page_leaves_it_unclean() {
        let _serial = serial();
        let (x, y) = two_views(1);
        const PAGES: u64 = 0x7e5a << 32;
        title_memory_shows(x, BLOCK, PAGES);
        assert_eq!(written_blocks(x, 1), [0]);
        assert!(written_blocks(x, 1).is_empty());
        title_memory_shows(y, BLOCK, PAGES);
        assert_eq!(written_blocks(x, 1), [0]);
        // SAFETY: our view, guarded as x's alias.
        unsafe { (y as *mut u32).write(1) };
        assert_eq!(written_blocks(x, 1), [0]);
        let_go(&[y], 1);
        assert!(written_blocks(x, 1).is_empty());
        let x_block = x / BLOCK;
        assert!(guards().shown.as_ref().unwrap().aliases(x_block).is_empty());
        let_go(&[x], 1);
    }
}
