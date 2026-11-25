//! Diagnostics (WAL-17): what submissions that return before the GPU is
//! done would wait for. Today a submission prepares, runs and copies back
//! before it returns; pipelined, the next one would be prepared while the
//! previous one still runs. Two things would make it wait:
//!
//! - its preparation reading or writing title memory the previous
//!   submission's shaders write (until that one's copy-back): refreshes of
//!   host copies, the title's buffer copies (done on the CPU), fills of
//!   upload memory; and CPU work that may write title memory, whose ranges
//!   are not known;
//! - the title asking for a fence value the latest submission signals.
//!   Fence values the title reads straight from memory (Grand Theft Auto VI
//!   does) are not seen.
//!
//! Counted per frame and logged with the frame line.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Submissions so far.
static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct State {
    /// The previous submission's declared write ranges (title addresses,
    /// sorted, merged).
    previous: Vec<(usize, usize)>,
    /// What the current submission's preparation touched of them: by
    /// refreshes, title copies, fills; bytes; CPU work run.
    touched: [bool; 3],
    bytes: u64,
    run: bool,
    counts: Counts,
}

/// Per frame: submissions; those that touched by each way and by any;
/// bytes; those that ran CPU work.
#[derive(Default)]
struct Counts {
    submissions: u64,
    by: [u64; 3],
    any: u64,
    all_bytes: u64,
    runs: u64,
    /// Title fence calls: event waits by how many submissions ago the
    /// value was signalled (0, 1, 2 or more, not by a queue), and reads of
    /// fences a queue signalled by the same.
    waits: [u64; 4],
    reads: [u64; 4],
    queue_signals: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    f(STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(State::default))
}

/// The ways preparation touches title memory.
#[derive(Clone, Copy)]
pub(crate) enum Touch {
    Refresh = 0,
    TitleCopy = 1,
    Fill = 2,
}

/// A submission starts preparing.
pub(crate) fn begin() {
    SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
    state(|s| {
        s.touched = [false; 3];
        s.bytes = 0;
        s.run = false;
    });
}

/// Preparation touches title memory `[from, from + len)`.
pub(crate) fn touch(how: Touch, from: usize, len: usize) {
    state(|s| {
        let overlap = overlap(&s.previous, from, from + len);
        if overlap > 0 {
            s.touched[how as usize] = true;
            s.bytes += overlap as u64;
        }
    });
}

/// Preparation runs CPU work that may write title memory.
pub(crate) fn run() {
    state(|s| s.run = true);
}

/// The submission's shaders write `writes` (title addresses, bytes).
pub(crate) fn end(writes: impl IntoIterator<Item = (usize, usize)>) {
    let mut ranges: Vec<(usize, usize)> = writes.into_iter().map(|(a, n)| (a, a + n)).collect();
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (a, b) in ranges {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    state(|s| {
        let c = &mut s.counts;
        c.submissions += 1;
        for (n, &t) in c.by.iter_mut().zip(&s.touched) {
            *n += u64::from(t);
        }
        c.any += u64::from(s.touched.iter().any(|&t| t));
        c.all_bytes += s.bytes;
        c.runs += u64::from(s.run);
        s.previous = merged;
    });
}

/// Bytes of `[from, to)` inside `ranges` (sorted, merged).
fn overlap(ranges: &[(usize, usize)], from: usize, to: usize) -> usize {
    let first = ranges.partition_point(|&(_, end)| end <= from);
    ranges[first..]
        .iter()
        .take_while(|&&(start, _)| start < to)
        .map(|&(start, end)| end.min(to) - start.max(from))
        .sum()
}

/// A queue signals a fence: the submission it follows.
pub(crate) fn queue_signal() -> u64 {
    state(|s| s.counts.queue_signals += 1);
    SUBMISSIONS.load(Ordering::Relaxed)
}

/// How many submissions ago `at` (a [`queue_signal`]) was; 3 for none.
fn bucket(at: Option<u64>) -> usize {
    at.map_or(3, |at| {
        SUBMISSIONS
            .load(Ordering::Relaxed)
            .saturating_sub(at)
            .min(2) as usize
    })
}

/// The title waits with an event for a value signalled after submission
/// `at` (`None`: not by a queue, or reached before).
pub(crate) fn fence_wait(at: Option<u64>) {
    state(|s| s.counts.waits[bucket(at)] += 1);
}

/// The title reads a fence's value; `at` as for [`fence_wait`].
pub(crate) fn fence_read(at: Option<u64>) {
    state(|s| s.counts.reads[bucket(at)] += 1);
}

/// Logs and resets the counts, over `frames` frames.
pub(crate) fn log(frames: f64) {
    let s = state(|s| std::mem::take(&mut s.counts));
    let f = frames.max(1.0);
    let per = |n: u64| n as f64 / f;
    tracing::info!(
        "pipelining probe per frame: {:.1} submissions, {:.1} of them prepared title memory the \
         previous one's shaders write ({:.1} by refreshes, {:.1} by title copies, {:.1} by \
         fills; {:.1} MB), {:.1} ran CPU work that may write title memory; {:.1} queue signals; \
         title event waits for a fence value signalled 0/1/2+ submissions ago or not by a queue: \
         {:.1}/{:.1}/{:.1}/{:.1}; fence value calls likewise: {:.1}/{:.1}/{:.1}/{:.1} (values \
         read straight from memory are not seen)",
        per(s.submissions),
        per(s.any),
        per(s.by[0]),
        per(s.by[1]),
        per(s.by[2]),
        s.all_bytes as f64 / 1e6 / f,
        per(s.runs),
        per(s.queue_signals),
        per(s.waits[0]),
        per(s.waits[1]),
        per(s.waits[2]),
        per(s.waits[3]),
        per(s.reads[0]),
        per(s.reads[1]),
        per(s.reads[2]),
        per(s.reads[3]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlaps_count_bytes_inside_the_ranges() {
        let ranges = [(100, 200), (300, 400)];
        assert_eq!(overlap(&ranges, 0, 100), 0);
        assert_eq!(overlap(&ranges, 150, 350), 50 + 50);
        assert_eq!(overlap(&ranges, 0, 1000), 200);
        assert_eq!(overlap(&ranges, 200, 300), 0);
        assert_eq!(overlap(&ranges, 399, 401), 1);
    }
}
