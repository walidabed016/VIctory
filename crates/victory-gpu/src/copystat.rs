//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Time of the title copies.
    CopyUs,
    /// Time keeping the bytes of ranges shaders write before a submission.
    SnapshotUs,
}
const COUNTERS: usize = 24;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, the host costs by
/// caller, and the ranges that took most bytes.
pub(crate) fn log(frames: f64) {
    for line in lines(frames) {
        tracing::info!("{line}");
    }
}

/// The lines [`log`] logs, taking the counters.
pub(crate) fn lines(frames: f64) -> Vec<String> {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    let mut lines = vec![format!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB {:.2} ms; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; snapshots {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        ms(c[C::CopyUs as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        ms(c[C::SnapshotUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
    )];
    cost_lines(frames, &mut lines);
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        lines.push(format!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        ));
    }
    lines
}

/// Who a host cost is charged to: the kind of copy a refresh brings up to
/// date, or the step that pays it ([`charge`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Caller {
    Other,
    /// Refreshes of title buffers' copies.
    Buffer,
    /// Refreshes of the loose-memory mirror (root views, mostly).
    Mirror,
    /// Refreshes of loose-memory windows (without the mirror).
    Window,
    /// Texture staging refreshes.
    Texture,
    /// Snapshots of ranges shaders write.
    Snapshot,
    /// Copying back what shaders wrote.
    CopyBack,
}
const CALLERS: usize = 7;
const CALLER_NAMES: [&str; CALLERS] = [
    "other",
    "buffers",
    "mirror",
    "windows",
    "textures",
    "snapshots",
    "copy back",
];

/// The system calls and passes over memory a host cost goes to.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Op {
    /// Write-watch askings (bytes: the ranges asked).
    Ask,
    /// Of those, askings that found written pages (bytes: those pages):
    /// each flushes the TLB of every thread of the process.
    Found,
    /// Guarded askings of view memory (bytes: the ranges).
    Guard,
    /// `VirtualQuery` (bytes: from the address to its region's end, which
    /// it walks).
    Query,
    /// `VirtualProtect` (bytes: the ranges).
    Protect,
    /// Title memory copied where a watch or guard said it changed.
    Copy,
    /// Title memory copied whole: nothing could tell what changed.
    Whole,
    /// Texture memory hashed.
    Hash,
    /// Texture memory detiled.
    Detile,
    /// Bytes taken into snapshots of ranges shaders write.
    Snapshot,
    /// Bytes of copies compared with their snapshots after a submission.
    Compare,
    /// Bytes written back into title memory.
    WriteBack,
}
const OPS: usize = 12;
const OP_NAMES: [&str; OPS] = [
    "asks",
    "asks finding writes",
    "guarded asks",
    "queries",
    "protects",
    "copies",
    "whole copies",
    "hashes",
    "detiles",
    "snapshots",
    "compares",
    "writes back",
];

/// Calls, microseconds and bytes per caller and operation, since the last
/// line.
static COSTS: [[[AtomicU64; 3]; OPS]; CALLERS] =
    [const { [const { [const { AtomicU64::new(0) }; 3] }; OPS] }; CALLERS];

thread_local! {
    static CALLER: std::cell::Cell<Caller> = const { std::cell::Cell::new(Caller::Other) };
}

/// Charges the costs of this thread to `caller` until dropped.
pub(crate) struct Charged(Caller);

impl Drop for Charged {
    fn drop(&mut self) {
        CALLER.with(|c| c.set(self.0));
    }
}

pub(crate) fn charge(caller: Caller) -> Charged {
    Charged(CALLER.with(|c| c.replace(caller)))
}

/// A slow operation: microseconds, caller, operation, address, bytes.
type Slow = (u64, Caller, Op, usize, u64);

/// The slowest operations since the last line.
static SLOWEST: Mutex<Vec<Slow>> = Mutex::new(Vec::new());

/// An operation of `bytes` at `at`, begun at `started`, charged to this
/// thread's caller.
pub(crate) fn cost(op: Op, started: std::time::Instant, at: usize, bytes: u64) {
    let us = started.elapsed().as_micros() as u64;
    let caller = CALLER.with(std::cell::Cell::get);
    let c = &COSTS[caller as usize][op as usize];
    c[0].fetch_add(1, Relaxed);
    c[1].fetch_add(us, Relaxed);
    c[2].fetch_add(bytes, Relaxed);
    if us >= 1000 {
        let mut slowest = SLOWEST.lock().unwrap_or_else(|e| e.into_inner());
        slowest.push((us, caller, op, at, bytes));
        slowest.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
        slowest.truncate(8);
    }
}

/// The host costs per frame by caller and operation, and the slowest
/// operations, as lines.
fn cost_lines(frames: f64, lines: &mut Vec<String>) {
    for (caller, ops) in COSTS.iter().enumerate() {
        let mut text = String::new();
        for (op, c) in ops.iter().enumerate() {
            let [n, us, bytes] = c.each_ref().map(|a| a.swap(0, Relaxed) as f64 / frames);
            if n > 0.0 {
                use std::fmt::Write;
                let sep = if text.is_empty() { "" } else { "; " };
                let _ = write!(
                    text,
                    "{sep}{} {n:.0} {:.1} ms {:.1} MB",
                    OP_NAMES[op],
                    us / 1e3,
                    bytes / 1e6
                );
            }
        }
        if !text.is_empty() {
            lines.push(format!(
                "  host costs per frame, {}: {text}",
                CALLER_NAMES[caller]
            ));
        }
    }
    let slowest = std::mem::take(&mut *SLOWEST.lock().unwrap_or_else(|e| e.into_inner()));
    for (us, caller, op, at, bytes) in slowest {
        lines.push(format!(
            "  slow host operation: {} for {}, {at:#x}+{bytes:#x}, {:.1} ms",
            OP_NAMES[op as usize],
            CALLER_NAMES[caller as usize],
            us as f64 / 1e3
        ));
    }
}
