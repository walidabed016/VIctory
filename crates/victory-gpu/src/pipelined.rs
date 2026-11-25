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

#[cfg(feature = "diagnostics")]
mod trace;

/// Whether shadow.rs and watch.rs keep copy-back state per submission
/// (snapshots, armed blocks, pending ranges, the twin compares' buffers)
/// and leave copy-back's writes out of the title's. Until then, copying a
/// submission back while the next is prepared would undo title writes.
const READY: bool = true;

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

/// Pipelined, with `VICTORY_SUBMIT_THREAD=1`: the title's submissions are
/// prepared, executed and launched in order on a host submit thread, and
/// the title returns once one is queued for it ([`queue`]).
pub(crate) fn submit_thread() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = on() && std::env::var_os("VICTORY_SUBMIT_THREAD").is_some_and(|v| v == "1");
        if on {
            tracing::info!("host submit thread (VICTORY_SUBMIT_THREAD)");
        }
        on
    })
}

/// Submissions queued for the submit thread, at most.
const QUEUED: usize = 64;

/// Prepares, executes and launches a submission queued for the submit
/// thread.
pub(crate) type Submit = Box<dyn FnOnce() + Send>;

/// A submission queued for the submit thread and not launched yet.
struct Queued {
    ticket: u64,
    /// Its host lists (addresses).
    lists: Vec<usize>,
    /// Title fences queues signalled after it, as [`Flight::signals`].
    signals: Vec<(usize, u64)>,
}

thread_local! {
    /// Whether this is the submit thread.
    static SUBMITTER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// On the submit thread: the ticket of the submission it runs, until
    /// launched.
    static TICKET: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// A submission queued and not yet completed.
struct Flight {
    seq: u64,
    /// Title memory its shaders write `[from, to)`, sorted and merged:
    /// copied back when it completes.
    writes: Vec<(usize, usize)>,
    /// Of `writes`, those not copied back yet once its copy-back has begun
    /// ([`copied_back_but`]); none before.
    left: Option<Vec<(usize, usize)>>,
    /// Title memory its lists read through host copies and staging
    /// buffers, as `writes`.
    reads: Vec<(usize, usize)>,
    /// Its host lists (addresses).
    lists: Vec<usize>,
    /// Title fences queues signalled after it: (fence, value).
    signals: Vec<(usize, u64)>,
    /// Refreshes of title memory `[from, to)` it had left to copy back,
    /// done once it is copied back ([`defer`]).
    refreshes: Vec<(usize, usize, Refresh)>,
}

/// Why the title waited.
#[derive(Clone, Copy)]
enum Why {
    Depth,
    List,
    Writes(Writer),
    Reads,
    All,
    Signal,
}

impl Why {
    /// Its waits in [`State::waits`]: those for memory shaders write by
    /// the step that waited, after the others.
    fn index(self) -> usize {
        match self {
            Why::Depth => 0,
            Why::List => 1,
            Why::Reads => 2,
            Why::All => 3,
            Why::Signal => 4,
            Why::Writes(by) => 5 + by as usize,
        }
    }
}

/// The step of a submission's preparation that waits for memory shaders
/// write ([`wait_writes`]).
#[derive(Clone, Copy)]
pub(crate) enum Writer {
    /// A refresh of a host copy.
    Refresh,
    /// A snapshot of a host copy its shaders write.
    Snapshot,
    /// A fill of upload memory.
    Fill,
    /// A title copy's source.
    TitleCopy,
    /// A texture's detiling.
    Texture,
}

struct State {
    flying: VecDeque<Flight>,
    /// Title memory refreshes wait to be copied back ([`wanted`]).
    wanted: Vec<(usize, usize)>,
    queued: VecDeque<Queued>,
    /// The last ticket given to a queued submission, and the last launched.
    tickets: u64,
    dequeued: u64,
    /// The next submission's number, and the last completed (0: none).
    next: u64,
    done: u64,
    launched: u64,
    deferred: u64,
    /// Refreshes deferred to copy-backs ([`defer`]).
    refreshed: u64,
    /// The newest submission refreshes were deferred to since the GPU was
    /// last told to wait ([`gpu_wait`]).
    gpu_wait: Option<u64>,
    /// Per [`Why::index`]: waits and time waited.
    waits: [(u64, Duration); 10],
}

/// Completes a submission, given when the one before it completed.
pub(crate) type Job = Box<dyn FnOnce(Instant) + Send>;

/// A refresh deferred to a copy-back ([`defer`]).
type Refresh = Box<dyn FnOnce() + Send>;

struct Pipeline {
    state: Mutex<State>,
    completed: Condvar,
    jobs: OnceLock<mpsc::Sender<Job>>,
    submits: OnceLock<mpsc::Sender<(u64, Submit)>>,
    /// Keeps a fence alive until its deferred signal, and signals (and
    /// lets go of) it.
    hold: fn(usize),
    signal: fn(usize, u64),
    /// Tells the GPU that a submission (its number) has completed, its
    /// copy-back done ([`newest_writing`]).
    copied: fn(u64),
}

impl Pipeline {
    const fn new(hold: fn(usize), signal: fn(usize, u64), copied: fn(u64)) -> Self {
        Self {
            state: Mutex::new(State {
                flying: VecDeque::new(),
                wanted: Vec::new(),
                queued: VecDeque::new(),
                tickets: 0,
                dequeued: 0,
                next: 1,
                done: 0,
                launched: 0,
                deferred: 0,
                refreshed: 0,
                gpu_wait: None,
                waits: [(0, Duration::ZERO); 10],
            }),
            completed: Condvar::new(),
            jobs: OnceLock::new(),
            submits: OnceLock::new(),
            hold,
            signal,
            copied,
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
        let w = &mut s.waits[why.index()];
        (w.0, w.1) = (w.0 + 1, w.1 + started.elapsed());
        #[cfg(feature = "diagnostics")]
        trace::waited(why.index(), seq, started);
        s
    }

    /// Waits until queued submission `ticket` has launched.
    fn wait_launched<'a>(
        &'a self,
        s: MutexGuard<'a, State>,
        ticket: u64,
        why: Why,
    ) -> MutexGuard<'a, State> {
        if s.dequeued >= ticket {
            return s;
        }
        let started = Instant::now();
        let mut s = self
            .completed
            .wait_while(s, |s| s.dequeued < ticket)
            .unwrap_or_else(|e| e.into_inner());
        let w = &mut s.waits[why.index()];
        (w.0, w.1) = (w.0 + 1, w.1 + started.elapsed());
        #[cfg(feature = "diagnostics")]
        trace::waited(why.index(), 0, started);
        s
    }

    /// Off the submit thread: waits until the newest queued submission
    /// `hit` picks has launched.
    fn wait_queued<'a>(
        &'a self,
        s: MutexGuard<'a, State>,
        hit: impl Fn(&Queued) -> bool,
        why: Why,
    ) -> MutexGuard<'a, State> {
        if SUBMITTER.get() {
            return s;
        }
        match s.queued.iter().rev().find(|q| hit(q)).map(|q| q.ticket) {
            Some(ticket) => self.wait_launched(s, ticket, why),
            None => s,
        }
    }

    /// Queues a submission for the submit thread, after room for it.
    fn queue(&'static self, lists: Vec<usize>, submit: Submit) {
        let submits = self.submits.get_or_init(|| self.start_submitting());
        let mut s = self.lock();
        if s.queued.len() >= QUEUED {
            let ticket = s.queued[s.queued.len() - QUEUED].ticket;
            s = self.wait_launched(s, ticket, Why::Depth);
        }
        s.tickets += 1;
        let ticket = s.tickets;
        s.queued.push_back(Queued {
            ticket,
            lists,
            signals: Vec::new(),
        });
        // In order with the tickets: under the lock.
        if submits.send((ticket, submit)).is_err() {
            tracing::error!("the submit thread is gone");
        }
    }

    fn start_submitting(&'static self) -> mpsc::Sender<(u64, Submit)> {
        let (send, submits) = mpsc::channel::<(u64, Submit)>();
        let spawned = std::thread::Builder::new()
            .name("victory submissions".into())
            .spawn(move || {
                SUBMITTER.set(true);
                for (ticket, submit) in submits {
                    TICKET.set(Some(ticket));
                    submit();
                    if TICKET.take().is_some() {
                        self.unlaunched(ticket);
                    }
                }
            });
        if let Err(e) = spawned {
            tracing::error!(error = %e, "no submit thread");
        }
        send
    }

    /// Queued submission `ticket` launched or ran: out of the queue; the
    /// fence signals queues made after it.
    fn dequeue(s: &mut State, ticket: u64) -> Vec<(usize, u64)> {
        s.dequeued = s.dequeued.max(ticket);
        match s.queued.iter().position(|q| q.ticket == ticket) {
            Some(i) => s.queued.remove(i).map_or_else(Vec::new, |q| q.signals),
            None => Vec::new(),
        }
    }

    /// Queued submission `ticket` ran without launching: its fence signals
    /// follow the newest submission in flight, or are made now.
    fn unlaunched(&self, ticket: u64) {
        let mut guard = self.lock();
        let s = &mut *guard;
        let signals = Self::dequeue(s, ticket);
        match s.flying.back_mut() {
            Some(f) => f.signals.extend(signals),
            None => {
                for (fence, value) in signals {
                    (self.signal)(fence, value);
                }
            }
        }
        drop(guard);
        self.completed.notify_all();
    }

    /// Waits until no submission `hit` picks is in flight (completions
    /// come in order: the newest it picks).
    fn wait_for(&self, why: Why, hit: impl Fn(&Flight) -> bool) {
        let s = self.lock();
        if let Some(seq) = s.flying.iter().rev().find(|f| hit(f)).map(|f| f.seq) {
            drop(self.wait(s, seq, why));
        }
    }

    /// Waits until no submission in flight has `[from, to)` of its writes
    /// left to copy back, noting the range wanted meanwhile.
    fn wait_copied(&self, why: Why, from: usize, to: usize) {
        let left = |f: &Flight| overlaps(f.left.as_ref().unwrap_or(&f.writes), from, to);
        let mut s = self.lock();
        if !s.flying.iter().any(left) {
            return;
        }
        let started = Instant::now();
        s.wanted.push((from, to));
        let mut s = self
            .completed
            .wait_while(s, |s| s.flying.iter().any(left))
            .unwrap_or_else(|e| e.into_inner());
        if let Some(i) = s.wanted.iter().position(|&r| r == (from, to)) {
            s.wanted.swap_remove(i);
        }
        let w = &mut s.waits[why.index()];
        (w.0, w.1) = (w.0 + 1, w.1 + started.elapsed());
    }

    /// Defers `refresh` of `[from, to)` to the newest submission in
    /// flight that has some of it left to copy back; whether there was one.
    fn defer(&self, from: usize, to: usize, refresh: Refresh) -> bool {
        let mut guard = self.lock();
        let s = &mut *guard;
        let left = |f: &&mut Flight| overlaps(f.left.as_ref().unwrap_or(&f.writes), from, to);
        let Some(f) = s.flying.iter_mut().rev().find(left) else {
            return false;
        };
        f.refreshes.push((from, to, refresh));
        s.gpu_wait = s.gpu_wait.max(Some(f.seq));
        s.refreshed += 1;
        true
    }

    /// Runs the refreshes deferred to the oldest submission in flight that
    /// its copy-back has no more left to write (`all`: every one).
    fn run_refreshes(&self, all: bool) {
        let ready: Vec<(usize, usize, Refresh)> = {
            let mut s = self.lock();
            let Some(Flight {
                writes,
                left,
                refreshes,
                ..
            }) = s.flying.front_mut()
            else {
                return;
            };
            let left = left.as_ref().unwrap_or(writes);
            let (ready, kept) = std::mem::take(refreshes)
                .into_iter()
                .partition(|&(from, to, _)| all || !overlaps(left, from, to));
            *refreshes = kept;
            ready
        };
        for (_, _, refresh) in ready {
            refresh();
        }
    }

    /// The oldest submission in flight has only `left` of its writes left
    /// to copy back.
    fn copied_back_but(&self, left: Vec<(usize, usize)>) {
        let mut s = self.lock();
        let Some(f) = s.flying.front_mut() else {
            return;
        };
        f.left = Some(merged(left));
        drop(s);
        self.completed.notify_all();
        self.run_refreshes(false);
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
        #[cfg(feature = "diagnostics")]
        trace::preparing(s.next);
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
        #[cfg(feature = "diagnostics")]
        trace::launched(seq);
        let ticket = TICKET.take();
        let signals = ticket.map_or_else(Vec::new, |t| Self::dequeue(&mut s, t));
        s.flying.push_back(Flight {
            seq,
            writes: merged(writes),
            left: None,
            reads: merged(reads),
            lists,
            signals,
            refreshes: Vec::new(),
        });
        // In order with the numbers: under the lock.
        if jobs.send(job).is_err() {
            tracing::error!("the completion thread is gone");
        }
        drop(s);
        if ticket.is_some() {
            self.completed.notify_all();
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
                    self.run_refreshes(true);
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
        (self.copied)(f.seq);
        s.done = f.seq;
        #[cfg(feature = "diagnostics")]
        trace::retired(f.seq);
        drop(s);
        self.completed.notify_all();
    }

    /// A queue signals `fence`: after the newest submission in flight, if
    /// any (else the caller signals at once).
    fn signal_after(&self, fence: usize, value: u64) -> bool {
        let mut guard = self.lock();
        let s = &mut *guard;
        let signals = match (s.queued.back_mut(), s.flying.back_mut()) {
            (Some(q), _) => &mut q.signals,
            (None, Some(f)) => &mut f.signals,
            (None, None) => return false,
        };
        (self.hold)(fence);
        signals.push((fence, value));
        s.deferred += 1;
        true
    }

    /// Waits for a deferred signal of `fence` to `value` or past it;
    /// whether there was one.
    fn wait_signal(&self, fence: usize, value: u64) -> bool {
        let reaches =
            |signals: &[(usize, u64)]| signals.iter().any(|&(x, v)| x == fence && v >= value);
        let s = self.lock();
        let queued = s.queued.iter().any(|q| reaches(&q.signals));
        let s = self.wait_queued(s, |q| reaches(&q.signals), Why::Signal);
        let Some(seq) = s.flying.iter().find(|f| reaches(&f.signals)).map(|f| f.seq) else {
            return queued;
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

static PIPELINE: Pipeline = Pipeline::new(hold_fence, signal_fence, crate::host::copied_back);

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
/// until no submission in flight writes it (its copy-back is done). A
/// refresh waits only until the copy-backs of the range are done
/// ([`early_refreshes`]).
pub(crate) fn wait_writes(from: usize, to: usize, by: Writer) {
    if !on() {
        return;
    }
    if matches!(by, Writer::Refresh) && early_refreshes() {
        PIPELINE.wait_copied(Why::Writes(by), from, to);
        return;
    }
    PIPELINE.wait_for(Why::Writes(by), |f| overlaps(&f.writes, from, to));
}

/// Whether a refresh waits only until the copy-backs of the ranges it
/// copies are done, not those of everything the submissions in flight
/// write, and copy-back does first the copies refreshes wait for
/// ([`wanted`]), then the smallest: in run 98 refreshes waited ~46 ms a
/// frame for submissions in flight, whose copy-backs took ~45 ms a frame.
/// `VICTORY_NO_EARLY_REFRESH` turns it off.
pub(crate) fn early_refreshes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_NO_EARLY_REFRESH").is_none();
        if !on {
            tracing::info!("refreshes wait for whole copy-backs (VICTORY_NO_EARLY_REFRESH)");
        }
        on
    })
}

/// The copy-back of the oldest submission in flight has done all but
/// `left` (title memory `[from, to)`): refreshes of the rest go on.
pub(crate) fn copied_back_but(left: impl IntoIterator<Item = (usize, usize)>) {
    if on() {
        PIPELINE.copied_back_but(left.into_iter().collect());
    }
}

/// Title memory refreshes wait for copy-back to write
/// ([`early_refreshes`]).
pub(crate) fn wanted() -> Vec<(usize, usize)> {
    match on() {
        true => PIPELINE.lock().wanted.clone(),
        false => Vec::new(),
    }
}

/// Whether a refresh of title memory a submission in flight has left to
/// copy back waits for none of it: the completion thread does it once that
/// is copied back, and the next submission waits on the GPU for that
/// submission's copy-back ([`defer`], [`gpu_wait`]). In run 103 the title
/// thread making the submissions waited ~37 ms a frame for refreshes, 27 of
/// them for copy-backs. With [`early_refreshes`];
/// `VICTORY_NO_DEFERRED_REFRESH` turns it off.
pub(crate) fn deferred_refreshes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = early_refreshes() && std::env::var_os("VICTORY_NO_DEFERRED_REFRESH").is_none();
        if early_refreshes() && !on {
            tracing::info!(
                "refreshes wait on the CPU for copy-backs (VICTORY_NO_DEFERRED_REFRESH)"
            );
        }
        on
    })
}

/// A refresh of title memory `[from, to)` ([`deferred_refreshes`]): when a
/// submission in flight has some of it left to copy back, `refresh` runs on
/// the completion thread once that is copied back, and the submission being
/// prepared waits on the GPU for that submission ([`gpu_wait`]). Whether it
/// was deferred; if not, nothing in flight has the range left to write.
pub(crate) fn defer(from: usize, to: usize, refresh: impl FnOnce() + Send + 'static) -> bool {
    on() && deferred_refreshes() && PIPELINE.defer(from, to, Box::new(refresh))
}

/// The newest submission whose copy-back refreshes were deferred to
/// ([`defer`]) since the last call: the GPU waits for it (its number on the
/// copy-back fence, [`crate::host::copied_back`]) before the submission
/// being prepared runs.
pub(crate) fn gpu_wait() -> Option<u64> {
    match on() {
        true => PIPELINE.lock().gpu_wait.take(),
        false => None,
    }
}

/// Waits on the CPU until submission `seq` has completed, when the GPU
/// cannot wait for it ([`gpu_wait`]).
pub(crate) fn wait_completed(seq: u64) {
    drop(PIPELINE.wait(PIPELINE.lock(), seq, Why::Writes(Writer::Refresh)));
}

/// The newest submission in flight whose shaders write any of `ranges`
/// (title memory `[from, to)`): its number, which the GPU learns when its
/// copy-back is done ([`crate::host::copied_back`]). None when not
/// pipelined.
pub(crate) fn newest_writing(ranges: impl IntoIterator<Item = (usize, usize)>) -> Option<u64> {
    if !on() {
        return None;
    }
    let ranges: Vec<(usize, usize)> = ranges.into_iter().collect();
    let s = PIPELINE.lock();
    s.flying
        .iter()
        .rev()
        .find(|f| {
            ranges
                .iter()
                .any(|&(from, to)| overlaps(&f.writes, from, to))
        })
        .map(|f| f.seq)
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

/// The completion thread saw the oldest submission in flight done on the
/// GPU; `early`: done before it got to it (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn gpu_done(early: bool) {
    let seq = PIPELINE.lock().flying.front().map(|f| f.seq);
    if let Some(seq) = seq {
        trace::gpu_done(seq, early);
    }
}

/// Waits until every submission has completed.
pub(crate) fn wait_all() {
    if on() {
        drained();
        PIPELINE.wait_for(Why::All, |_| true);
    }
}

/// Before a host list is reset or let go: waits for its submissions.
pub(crate) fn wait_list(list: usize) {
    if on() {
        drop(PIPELINE.wait_queued(PIPELINE.lock(), |q| q.lists.contains(&list), Why::List));
        PIPELINE.wait_for(Why::List, |f| f.lists.contains(&list));
    }
}

/// Whether a submission queued or in flight runs host list `list`.
pub(crate) fn flying(list: usize) -> bool {
    on() && {
        let s = PIPELINE.lock();
        s.queued.iter().any(|q| q.lists.contains(&list))
            || s.flying.iter().any(|f| f.lists.contains(&list))
    }
}

/// With the submit thread, off it: queues `submit`, the title's
/// submission of host lists `lists`, for it; whether it did.
pub(crate) fn queue(lists: Vec<usize>, submit: Submit) -> bool {
    if !submit_thread() || SUBMITTER.get() {
        return false;
    }
    PIPELINE.queue(lists, submit);
    true
}

/// With the submit thread and `VICTORY_QUEUED_PRESENT=1`, off it: queues
/// `present` for the submit thread, after the submissions queued before,
/// so that the title does not wait for them; whether it did.
pub(crate) fn queue_present(present: Submit) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on =
            submit_thread() && std::env::var_os("VICTORY_QUEUED_PRESENT").is_some_and(|v| v == "1");
        if on {
            tracing::info!("presents queued for the submit thread (VICTORY_QUEUED_PRESENT)");
        }
        on
    });
    on && queue(Vec::new(), present)
}

/// Off the submit thread: waits until the submissions queued for it have
/// launched.
pub(crate) fn drained() {
    if submit_thread() {
        drop(PIPELINE.wait_queued(PIPELINE.lock(), |_| true, Why::All));
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
    let (launched, deferred, refreshed, waits) = {
        let mut s = PIPELINE.lock();
        let counts = (s.launched, s.deferred, s.refreshed, s.waits);
        (s.launched, s.deferred, s.refreshed, s.waits) = (0, 0, 0, [(0, Duration::ZERO); 10]);
        counts
    };
    let f = frames.max(1.0);
    let w = waits.map(|(n, t)| (n as f64 / f, t.as_secs_f64() * 1e3 / f));
    let by = |writer: Writer| w[Why::Writes(writer).index()];
    let writes = [
        Writer::Refresh,
        Writer::Snapshot,
        Writer::Fill,
        Writer::TitleCopy,
        Writer::Texture,
    ]
    .map(by);
    let all_writes = writes
        .iter()
        .fold((0.0, 0.0), |a, w| (a.0 + w.0, a.1 + w.1));
    tracing::info!(
        "pipelined submissions per frame: {:.1}, {:.1} fence signals deferred, {:.1} refreshes \
         deferred to copy-backs; title threads \
         waited (times, ms) for room {:.1} {:.1}, for their lists {:.1} {:.1}, for memory \
         shaders write {:.1} {:.1} (refreshes {:.1} {:.1}, snapshots {:.1} {:.1}, fills {:.1} \
         {:.1}, title copies {:.1} {:.1}, textures {:.1} {:.1}), for memory read {:.1} {:.1}, \
         for all {:.1} {:.1}, for fences {:.1} {:.1}",
        launched as f64 / f,
        deferred as f64 / f,
        refreshed as f64 / f,
        w[Why::Depth.index()].0,
        w[Why::Depth.index()].1,
        w[Why::List.index()].0,
        w[Why::List.index()].1,
        all_writes.0,
        all_writes.1,
        writes[0].0,
        writes[0].1,
        writes[1].0,
        writes[1].1,
        writes[2].0,
        writes[2].1,
        writes[3].0,
        writes[3].1,
        writes[4].0,
        writes[4].1,
        w[Why::Reads.index()].0,
        w[Why::Reads.index()].1,
        w[Why::All.index()].0,
        w[Why::All.index()].1,
        w[Why::Signal.index()].0,
        w[Why::Signal.index()].1,
    );
    let names = [
        "room",
        "lists",
        "memory read",
        "all",
        "fences",
        "refreshes",
        "snapshots",
        "fills",
        "title copies",
        "textures",
    ];
    trace::log(f, &names);
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
        Box::leak(Box::new(Pipeline::new(|_| {}, record, |_| {})))
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
    fn queued_submissions_launch_in_order_with_the_signals_made_after_them() {
        const FENCE: usize = 3;
        let p = pipeline();
        let (go, gate) = mpsc::channel::<()>();
        p.queue(
            vec![1],
            Box::new(move || {
                let _ = gate.recv();
                p.launch(Vec::new(), Vec::new(), vec![1], Box::new(|_| {}));
            }),
        );
        assert!(p.signal_after(FENCE, 5));
        p.queue(
            vec![2],
            Box::new(move || p.launch(Vec::new(), Vec::new(), vec![2], Box::new(|_| {}))),
        );
        assert!(p.signal_after(FENCE, 6));
        let waits_for_list_2 = move || {
            drop(p.wait_queued(p.lock(), |q| q.lists.contains(&2), Why::List));
            p.wait_for(Why::List, |f| f.lists.contains(&2));
        };
        blocks_until(waits_for_list_2, go);
        drop(p.wait_queued(p.lock(), |_| true, Why::All));
        p.wait_for(Why::All, |_| true);
        assert_eq!(signalled(FENCE), [5, 6]);
        // One that never launches: the signals after it are made at once.
        let (go, gate) = mpsc::channel::<()>();
        p.queue(
            vec![3],
            Box::new(move || {
                let _ = gate.recv();
            }),
        );
        assert!(p.signal_after(FENCE, 7));
        assert!(!p.wait_signal(FENCE, 8));
        blocks_until(move || assert!(p.wait_signal(FENCE, 7)), go);
        assert_eq!(signalled(FENCE), [5, 6, 7]);
    }

    #[test]
    fn waits_end_when_the_submission_touching_the_range_completes() {
        let p = pipeline();
        let (go, job) = held();
        p.launch(vec![(100, 200)], vec![(300, 400)], vec![7], job);
        let writes = |from, to| move |f: &Flight| overlaps(&f.writes, from, to);
        let why = Why::Writes(Writer::Refresh);
        p.wait_for(why, writes(0, 100));
        p.wait_for(why, writes(300, 400));
        p.wait_for(Why::List, |f| f.lists.contains(&8));
        blocks_until(move || p.wait_for(why, writes(150, 160)), go);
        p.wait_for(why, writes(150, 160));
    }

    /// A refresh waits until the copy-back of its range is done, noted
    /// wanted meanwhile; other steps wait for the whole submission.
    #[test]
    fn a_refresh_waits_only_for_the_copy_back_of_its_range() {
        let p = pipeline();
        let (part, parted) = mpsc::channel::<()>();
        let (go, gate) = mpsc::channel::<()>();
        p.launch(
            vec![(100, 200), (300, 400)],
            Vec::new(),
            vec![9],
            Box::new(move |_| {
                let _ = parted.recv();
                p.copied_back_but(vec![(300, 400)]);
                let _ = gate.recv();
            }),
        );
        let refresh = Why::Writes(Writer::Refresh);
        blocks_until(move || p.wait_copied(refresh, 150, 160), part);
        p.wait_copied(refresh, 150, 160);
        let waiter = std::thread::spawn(move || p.wait_copied(refresh, 300, 350));
        let started = Instant::now();
        while p.lock().wanted != [(300, 350)] {
            assert!(started.elapsed() < Duration::from_secs(10), "never wanted");
            std::thread::sleep(Duration::from_millis(1));
        }
        let snapshot = Why::Writes(Writer::Snapshot);
        blocks_until(
            move || p.wait_for(snapshot, |f| overlaps(&f.writes, 150, 160)),
            go,
        );
        waiter.join().unwrap();
        assert!(p.lock().wanted.is_empty());
    }

    /// A refresh of memory a submission in flight has left to copy back
    /// runs on the completion thread once that is copied back, and the GPU
    /// waits for that submission; one of memory nothing in flight has left
    /// to write is not deferred.
    #[test]
    fn a_deferred_refresh_runs_once_its_range_is_copied_back() {
        let p = pipeline();
        let (part, parted) = mpsc::channel::<()>();
        let (go, gate) = mpsc::channel::<()>();
        let (ran, runs) = mpsc::channel::<&'static str>();
        p.launch(
            vec![(100, 200), (300, 400)],
            Vec::new(),
            vec![9],
            Box::new(move |_| {
                let _ = parted.recv();
                p.copied_back_but(vec![(300, 400)]);
                let _ = gate.recv();
            }),
        );
        let seq = p.lock().flying[0].seq;
        let (first, second) = (ran.clone(), ran);
        assert!(p.defer(150, 160, Box::new(move || first.send("first").unwrap())));
        assert!(p.defer(300, 350, Box::new(move || second.send("second").unwrap())));
        assert!(!p.defer(500, 600, Box::new(|| {})));
        assert_eq!(p.lock().gpu_wait.take(), Some(seq));
        let quiet = Duration::from_millis(50);
        assert!(
            runs.recv_timeout(quiet).is_err(),
            "ran before its copy-back"
        );
        part.send(()).unwrap();
        assert_eq!(runs.recv_timeout(Duration::from_secs(10)), Ok("first"));
        assert!(
            runs.recv_timeout(quiet).is_err(),
            "ran before its copy-back"
        );
        assert!(!p.defer(150, 160, Box::new(|| {})));
        go.send(()).unwrap();
        assert_eq!(runs.recv_timeout(Duration::from_secs(10)), Ok("second"));
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
