//! Pipelined submissions (WAL-17, design B). A submission returns once its
//! lists are queued: a completion thread waits for each in turn, copies
//! back what its shaders wrote, writes its lists' bottom-of-pipe values and
//! signals the title fences queues signalled after it. Meanwhile the title
//! thread prepares the next one, and waits only where a step would touch
//! memory a submission still in flight writes or reads, for one of its
//! lists still in flight, or for room ([`DEPTH`]).
//!
//! Off unless `VICTORY_PIPELINED=1` and the host copies keep their
//! copy-back state per submission ([`READY`]); never with
//! `VICTORY_GPU_TIMING`, which waits for each submission alone. Off,
//! every wait here returns at once and nothing is deferred.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock, mpsc};
use std::time::{Duration, Instant};

use crate::object::This;

/// Whether shadow.rs and watch.rs keep copy-back state per submission
/// (snapshots, armed blocks, pending ranges, the twin compares' buffers)
/// and leave copy-back's writes out of the title's. Until then, copying a
/// submission back while the next is prepared would undo title writes.
const READY: bool = false;

/// Submissions queued and not yet completed, at most: one copying back,
/// one on the GPU, one queued behind it.
pub(crate) const DEPTH: usize = 3;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINED").is_none_or(|v| v != "1") {
            return false;
        }
        #[cfg(all(windows, feature = "diagnostics"))]
        if crate::host::timing() {
            tracing::warn!("VICTORY_PIPELINED is off with VICTORY_GPU_TIMING");
            return false;
        }
        if !READY {
            tracing::warn!("VICTORY_PIPELINED needs copy-back state per submission; off");
            return false;
        }
        tracing::info!(depth = DEPTH, "pipelined submissions");
        true
    })
}

/// A submission queued and not yet completed.
struct Flight {
    seq: u64,
    /// Title memory its shaders write `[from, to)`, sorted and merged:
    /// copied back when it completes.
    writes: Vec<(usize, usize)>,
    /// Title memory its lists read through host copies and staging
    /// buffers, as `writes`.
    reads: Vec<(usize, usize)>,
    /// Its host lists (addresses).
    lists: Vec<usize>,
    /// Title fences queues signalled after it: (fence, value).
    signals: Vec<(usize, u64)>,
}

/// Why the title waited.
#[derive(Clone, Copy)]
enum Why {
    Depth,
    List,
    Writes,
    Reads,
    All,
    Signal,
}

struct State {
    flying: VecDeque<Flight>,
    /// The next submission's number, and the last completed (0: none).
    next: u64,
    done: u64,
    launched: u64,
    deferred: u64,
    /// Per [`Why`]: waits and time waited.
    waits: [(u64, Duration); 6],
}

/// Completes a submission, given when the one before it completed.
pub(crate) type Job = Box<dyn FnOnce(Instant) + Send>;

struct Pipeline {
    state: Mutex<State>,
    completed: Condvar,
    jobs: OnceLock<mpsc::Sender<Job>>,
    /// Keeps a fence alive until its deferred signal, and signals (and
    /// lets go of) it.
    hold: fn(usize),
    signal: fn(usize, u64),
}

impl Pipeline {
    const fn new(hold: fn(usize), signal: fn(usize, u64)) -> Self {
        Self {
            state: Mutex::new(State {
                flying: VecDeque::new(),
                next: 1,
                done: 0,
                launched: 0,
                deferred: 0,
                waits: [(0, Duration::ZERO); 6],
            }),
            completed: Condvar::new(),
            jobs: OnceLock::new(),
            hold,
            signal,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Waits until submission `seq` has completed.
    fn wait<'a>(&'a self, s: MutexGuard<'a, State>, seq: u64, why: Why) -> MutexGuard<'a, State> {
        if s.done >= seq {
            return s;
        }
        let started = Instant::now();
        let mut s = self
            .completed
            .wait_while(s, |s| s.done < seq)
            .unwrap_or_else(|e| e.into_inner());
        let w = &mut s.waits[why as usize];
        (w.0, w.1) = (w.0 + 1, w.1 + started.elapsed());
        s
    }

    /// Waits until no submission `hit` picks is in flight (completions
    /// come in order: the newest it picks).
    fn wait_for(&self, why: Why, hit: impl Fn(&Flight) -> bool) {
        let s = self.lock();
        if let Some(seq) = s.flying.iter().rev().find(|f| hit(f)).map(|f| f.seq) {
            drop(self.wait(s, seq, why));
        }
    }

    /// Before preparing a submission of `lists`: none of them in flight
    /// (they reset their allocators and preludes and refill their uploads)
    /// and room for one more. Its number.
    fn room(&self, lists: &[usize]) -> u64 {
        let mut s = self.lock();
        let listed = |f: &Flight| f.lists.iter().any(|l| lists.contains(l));
        if let Some(seq) = s.flying.iter().rev().find(|f| listed(f)).map(|f| f.seq) {
            s = self.wait(s, seq, Why::List);
        }
        if s.flying.len() >= DEPTH {
            let seq = s.flying[s.flying.len() - DEPTH].seq;
            s = self.wait(s, seq, Why::Depth);
        }
        s.next
    }

    /// Queues a submission's completion.
    fn launch(
        &'static self,
        writes: Vec<(usize, usize)>,
        reads: Vec<(usize, usize)>,
        lists: Vec<usize>,
        job: Job,
    ) {
        let jobs = self.jobs.get_or_init(|| self.start());
        let mut s = self.lock();
        let seq = s.next;
        s.next += 1;
        s.launched += 1;
        s.flying.push_back(Flight {
            seq,
            writes: merged(writes),
            reads: merged(reads),
            lists,
            signals: Vec::new(),
        });
        // In order with the numbers: under the lock.
        if jobs.send(job).is_err() {
            tracing::error!("the completion thread is gone");
        }
    }

    fn start(&'static self) -> mpsc::Sender<Job> {
        let (send, jobs) = mpsc::channel::<Job>();
        let spawned = std::thread::Builder::new()
            .name("victory completions".into())
            .spawn(move || {
                let mut last = Instant::now();
                for job in jobs {
                    job(last);
                    self.retire();
                    last = Instant::now();
                }
            });
        if let Err(e) = spawned {
            tracing::error!(error = %e, "no completion thread");
        }
        send
    }

    /// The oldest submission completed: its fence signals, then whoever
    /// waits for it. Signalled under the lock, so that a signal made at
    /// once (nothing in flight) never comes before a deferred one.
    fn retire(&self) {
        let mut s = self.lock();
        let Some(f) = s.flying.pop_front() else {
            return;
        };
        for &(fence, value) in &f.signals {
            (self.signal)(fence, value);
        }
        s.done = f.seq;
        drop(s);
        self.completed.notify_all();
    }

    /// A queue signals `fence`: after the newest submission in flight, if
    /// any (else the caller signals at once).
    fn signal_after(&self, fence: usize, value: u64) -> bool {
        let mut guard = self.lock();
        let s = &mut *guard;
        let Some(f) = s.flying.back_mut() else {
            return false;
        };
        (self.hold)(fence);
        f.signals.push((fence, value));
        s.deferred += 1;
        true
    }

    /// Waits for a deferred signal of `fence` to `value` or past it;
    /// whether there was one.
    fn wait_signal(&self, fence: usize, value: u64) -> bool {
        let s = self.lock();
        let reaches = |f: &&Flight| f.signals.iter().any(|&(x, v)| x == fence && v >= value);
        let Some(seq) = s.flying.iter().find(reaches).map(|f| f.seq) else {
            return false;
        };
        drop(self.wait(s, seq, Why::Signal));
        true
    }
}

/// `ranges` sorted, those that overlap or touch joined.
fn merged(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (a, b) in ranges {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Whether `[from, to)` overlaps `ranges` (sorted, merged).
fn overlaps(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    let first = ranges.partition_point(|&(_, end)| end <= from);
    ranges.get(first).is_some_and(|&(start, _)| start < to)
}

static PIPELINE: Pipeline = Pipeline::new(hold_fence, signal_fence);

fn hold_fence(fence: usize) {
    crate::object::add_ref(fence as This);
}

fn signal_fence(fence: usize, value: u64) {
    let fence = fence as This;
    crate::fence::signal(fence, value);
    crate::fence::release(fence);
}

/// Before preparing a submission of `lists` (host list addresses): waits
/// for room and for those lists. The submission's slot, below [`DEPTH`]:
/// its lists of its own (twin copies) are free again.
pub(crate) fn room(lists: impl IntoIterator<Item = usize>) -> usize {
    if !on() {
        return 0;
    }
    (PIPELINE.room(&lists.into_iter().collect::<Vec<_>>()) % DEPTH as u64) as usize
}

/// A submission is queued: `job` waits for it and completes it, after the
/// ones before it. `writes` and `reads` are title memory `[from, to)`.
pub(crate) fn launch(
    writes: Vec<(usize, usize)>,
    reads: Vec<(usize, usize)>,
    lists: Vec<usize>,
    job: Job,
) {
    PIPELINE.launch(writes, reads, lists, job);
}

/// Before preparation reads or writes title memory `[from, to)`: waits
/// until no submission in flight writes it (its copy-back is done).
pub(crate) fn wait_writes(from: usize, to: usize) {
    if on() {
        PIPELINE.wait_for(Why::Writes, |f| overlaps(&f.writes, from, to));
    }
}

/// Before preparation writes title memory `[from, to)` as the GPU would
/// later (a title copy): waits until no submission in flight reads or
/// writes it.
pub(crate) fn wait_reads(from: usize, to: usize) {
    if on() {
        PIPELINE.wait_for(Why::Reads, |f| {
            overlaps(&f.reads, from, to) || overlaps(&f.writes, from, to)
        });
    }
}

/// Waits until every submission has completed.
pub(crate) fn wait_all() {
    if on() {
        PIPELINE.wait_for(Why::All, |_| true);
    }
}

/// Before a host list is reset or let go: waits for its submissions.
pub(crate) fn wait_list(list: usize) {
    if on() {
        PIPELINE.wait_for(Why::List, |f| f.lists.contains(&list));
    }
}

/// A queue signals `fence`: deferred until the submissions queued before
/// it complete, if any are in flight (else the caller signals at once).
pub(crate) fn signal_after(fence: This, value: u64) -> bool {
    on() && PIPELINE.signal_after(fence as usize, value)
}

/// A blocking wait for `fence` to reach `value`: waits for a deferred
/// signal that reaches it; whether there was one.
pub(crate) fn wait_signal(fence: This, value: u64) -> bool {
    on() && PIPELINE.wait_signal(fence as usize, value)
}

/// Logs and resets the counts, over `frames` frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let (launched, deferred, waits) = {
        let mut s = PIPELINE.lock();
        let counts = (s.launched, s.deferred, s.waits);
        (s.launched, s.deferred, s.waits) = (0, 0, [(0, Duration::ZERO); 6]);
        counts
    };
    let f = frames.max(1.0);
    let w = waits.map(|(n, t)| (n as f64 / f, t.as_secs_f64() * 1e3 / f));
    tracing::info!(
        "pipelined submissions per frame: {:.1}, {:.1} fence signals deferred; title threads \
         waited (times, ms) for room {:.1} {:.1}, for their lists {:.1} {:.1}, for memory \
         shaders write {:.1} {:.1}, for memory read {:.1} {:.1}, for all {:.1} {:.1}, for \
         fences {:.1} {:.1}",
        launched as f64 / f,
        deferred as f64 / f,
        w[Why::Depth as usize].0,
        w[Why::Depth as usize].1,
        w[Why::List as usize].0,
        w[Why::List as usize].1,
        w[Why::Writes as usize].0,
        w[Why::Writes as usize].1,
        w[Why::Reads as usize].0,
        w[Why::Reads as usize].1,
        w[Why::All as usize].0,
        w[Why::All as usize].1,
        w[Why::Signal as usize].0,
        w[Why::Signal as usize].1,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Fences signalled by the test pipelines: (fence, value), in order.
    static SIGNALLED: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());

    fn record(fence: usize, value: u64) {
        SIGNALLED.lock().unwrap().push((fence, value));
    }

    fn signalled(fence: usize) -> Vec<u64> {
        let all = SIGNALLED.lock().unwrap();
        all.iter().filter(|s| s.0 == fence).map(|s| s.1).collect()
    }

    fn pipeline() -> &'static Pipeline {
        Box::leak(Box::new(Pipeline::new(|_| {}, record)))
    }

    /// A job that completes once its sender is used (or dropped).
    fn held() -> (mpsc::Sender<()>, Job) {
        let (go, gate) = mpsc::channel::<()>();
        (
            go,
            Box::new(move |_| {
                let _ = gate.recv();
            }),
        )
    }

    /// Runs `wait` on its own thread: done after `go`, not before.
    fn blocks_until(wait: impl FnOnce() + Send + 'static, go: mpsc::Sender<()>) {
        let (done, finished) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            wait();
            done.send(()).unwrap();
        });
        assert!(
            finished.recv_timeout(Duration::from_millis(50)).is_err(),
            "did not wait"
        );
        go.send(()).unwrap();
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("still waiting");
        waiter.join().unwrap();
    }

    #[test]
    fn ranges_merge_and_overlap() {
        let ranges = merged(vec![(300, 400), (100, 200), (150, 250), (250, 260)]);
        assert_eq!(ranges, [(100, 260), (300, 400)]);
        assert!(!overlaps(&ranges, 0, 100));
        assert!(overlaps(&ranges, 99, 101));
        assert!(!overlaps(&ranges, 260, 300));
        assert!(overlaps(&ranges, 399, 1000));
        assert!(!overlaps(&ranges, 400, 1000));
        assert!(!overlaps(&[], 0, usize::MAX));
    }

    #[test]
    fn completions_run_in_order_and_signal_after_their_submission() {
        const FENCE: usize = 1;
        let p = pipeline();
        let order = Arc::new(Mutex::new(Vec::new()));
        let (go, first) = held();
        let o = order.clone();
        p.launch(
            Vec::new(),
            Vec::new(),
            vec![1],
            Box::new(move |last| {
                first(last);
                o.lock().unwrap().push(1);
            }),
        );
        assert!(p.signal_after(FENCE, 5));
        let o = order.clone();
        p.launch(
            Vec::new(),
            Vec::new(),
            vec![2],
            Box::new(move |_| o.lock().unwrap().push(2)),
        );
        assert!(p.signal_after(FENCE, 6));
        assert!(
            signalled(FENCE).is_empty(),
            "signalled before its submission completed"
        );
        go.send(()).unwrap();
        p.wait_for(Why::All, |_| true);
        assert_eq!(*order.lock().unwrap(), [1, 2]);
        assert_eq!(signalled(FENCE), [5, 6]);
        assert!(
            !p.signal_after(FENCE, 7),
            "nothing in flight: signalled at once"
        );
    }

    #[test]
    fn waits_end_when_the_submission_touching_the_range_completes() {
        let p = pipeline();
        let (go, job) = held();
        p.launch(vec![(100, 200)], vec![(300, 400)], vec![7], job);
        let writes = |from, to| move |f: &Flight| overlaps(&f.writes, from, to);
        p.wait_for(Why::Writes, writes(0, 100));
        p.wait_for(Why::Writes, writes(300, 400));
        p.wait_for(Why::List, |f| f.lists.contains(&8));
        blocks_until(move || p.wait_for(Why::Writes, writes(150, 160)), go);
        p.wait_for(Why::Writes, writes(150, 160));
    }

    #[test]
    fn room_waits_for_depth_and_for_lists_in_flight() {
        let p = pipeline();
        let mut gos = Vec::new();
        for list in 0..DEPTH {
            let (go, job) = held();
            p.launch(Vec::new(), Vec::new(), vec![list], job);
            gos.push(go);
        }
        let mut gos = gos.into_iter();
        let (first, second) = (gos.next().unwrap(), gos.next().unwrap());
        blocks_until(move || assert_eq!(p.room(&[99]), DEPTH as u64 + 1), first);
        blocks_until(
            move || {
                p.room(&[1]);
            },
            second,
        );
        drop(gos);
        p.wait_for(Why::All, |_| true);
    }

    #[test]
    fn blocking_fence_waits_end_with_the_deferred_signal() {
        const FENCE: usize = 2;
        let p = pipeline();
        let (go, job) = held();
        p.launch(Vec::new(), Vec::new(), Vec::new(), job);
        assert!(p.signal_after(FENCE, 3));
        assert!(!p.wait_signal(FENCE, 4), "no deferred signal reaches 4");
        blocks_until(move || assert!(p.wait_signal(FENCE, 3)), go);
        assert_eq!(signalled(FENCE), [3]);
    }
}
