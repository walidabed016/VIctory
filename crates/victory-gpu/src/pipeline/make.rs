//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline ahead of those started early and waits
//! for it within a budget of a few milliseconds per frame, so the cheapest
//! ones are there at once; past the budget a draw or dispatch whose
//! pipeline is not ready is dropped, as an untranslated one is. A longer
//! wait is a hitch in the world, and the half second this waited before
//! kept run 70's start screen at 1.9 fps for minutes.
//!
//! shortcut: a pass the title runs only once (a colour-grading bake) whose
//! pipeline is not made within the budget is lost in a run that has to
//! compile it. A warm run started it when the title created it and loads it
//! from the library, and a warm run is what is played.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    /// Queued (and counted as waiting), queued as a first use, and taken
    /// by a worker.
    queued: AtomicBool,
    first_use: AtomicBool,
    taken: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
    /// Made however long that takes, never dropped for want of time: a
    /// BVH builder's with `VICTORY_WAIT_BUILDER` (see [`ready`]).
    must: bool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        must: bool,
    ) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            first_use: AtomicBool::new(false),
            taken: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            must,
        }
    }

    fn make(&self) {
        let started = Instant::now();
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let ms = started.elapsed().as_millis() as u64;
        if ms >= 1000 {
            tracing::info!(ms, shaders = ?self.shaders, "slow host pipeline");
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Why `p` has no host pipeline, for a warning: still being made, or
/// refused (a shader that does not translate, or the host's refusal).
pub(crate) fn state(p: &Pending) -> &'static str {
    match p.made.get() {
        None => "no host pipeline: being made",
        Some(None) => "no host pipeline: refused",
        Some(Some(_)) => "its host pipeline did not bind",
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(4);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// Time title threads waited for pipelines, over all frames.
static WAITED_US: AtomicU64 = AtomicU64::new(0);

/// Time title threads waited for pipelines since the last call (for the
/// frame line).
pub(crate) fn take_waited() -> Duration {
    Duration::from_micros(WAITED_US.swap(0, Ordering::Relaxed))
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use. One that must be made is made by then, on this
/// thread unless a worker has it already: a BVH builder pass dropped leaves
/// the passes after it a broken tree, whose climbs ran past every frame
/// (runs 146 and 148, PrefixSum64 queued behind 2,100 pipelines).
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p, true);
        match p.must {
            true => make_here(p),
            false => wait(p, crate::frame::current()),
        }
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers once, and at its first use once more ahead
/// of those started early, if it was started early and no worker has taken
/// it yet.
fn start(p: &Arc<Pending>, first_use: bool) {
    let new = !p.queued.swap(true, Ordering::Relaxed);
    if new {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let ahead =
        first_use && !p.first_use.swap(true, Ordering::Relaxed) && !p.taken.load(Ordering::Relaxed);
    if new || ahead {
        queue(p.clone(), first_use);
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it or it must be made (during loading, so the world's
/// first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && (p.must || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static FRAME_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        FRAME_US.store(0, Ordering::Relaxed);
    }
    let left = BUDGET.saturating_sub(Duration::from_micros(FRAME_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    let waited = started.elapsed().as_micros() as u64;
    FRAME_US.fetch_add(waited, Ordering::Relaxed);
    WAITED_US.fetch_add(waited, Ordering::Relaxed);
}

/// Makes `p` on this thread unless a worker has taken it, else waits for
/// that worker, however long either takes.
fn make_here(p: &Pending) {
    let started = Instant::now();
    let here = !p.taken.swap(true, Ordering::Relaxed);
    if here {
        p.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        held = done.wait(held).unwrap_or_else(|e| e.into_inner());
    }
    drop(held);
    let waited = started.elapsed();
    WAITED_US.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    tracing::info!(
        ms = waited.as_millis() as u64,
        here,
        made = matches!(p.made.get(), Some(Some(_))),
        shaders = ?p.shaders,
        "a pipeline that must be made waited for at its first use"
    );
}

/// Pipelines for the workers: first uses, then those started early.
static QUEUE: Mutex<[VecDeque<Arc<Pending>>; 2]> = Mutex::new([VecDeque::new(), VecDeque::new()]);
/// Signalled whenever a pipeline is queued.
static WORK: Condvar = Condvar::new();

/// Hands `p` to the workers (three quarters of the logical processors, 2
/// to 24), started on first use. A few driver compiles take tens of seconds
/// (minutes with Aftermath's shader debug information): with 8 workers,
/// those held every worker and the queue stood still for minutes (run 70).
fn queue(p: Arc<Pending>, first_use: bool) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let workers =
            std::thread::available_parallelism().map_or(2, |n| (n.get() * 3 / 4).clamp(2, 24));
        (0..workers)
            .filter(|i| {
                let spawned =
                    victory_core::thread::spawn_host(&format!("victory-pipelines-{i}"), work);
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count()
    });
    if workers == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            p.make();
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    QUEUE.lock().unwrap_or_else(|e| e.into_inner())[usize::from(!first_use)].push_back(p);
    WORK.notify_one();
}

/// A worker: makes the queued pipelines, first uses first, each once.
fn work() {
    // Below the title's own threads (loading, recording).
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    loop {
        let p = {
            let mut queues = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(p) = queues.iter_mut().find_map(VecDeque::pop_front) {
                    break p;
                }
                queues = WORK.wait(queues).unwrap_or_else(|e| e.into_inner());
            }
        };
        if p.taken.swap(true, Ordering::Relaxed) {
            continue;
        }
        p.make();
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None, false));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline that must be made is made by the time its first use
    /// returns (here: no host pipeline for no shaders).
    #[test]
    fn a_pipeline_that_must_be_made_is_made_at_first_use() {
        let p = Arc::new(Pending::new(Vec::new(), None, true));
        assert!(ready(&p).is_none());
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A frame waits for pipelines no longer than its budget, over all its
    /// waits.
    #[test]
    fn frames_wait_at_most_their_budget() {
        let p = Pending::new(Vec::new(), None, false);
        let started = Instant::now();
        for _ in 0..10 {
            wait(&p, u64::MAX - 1);
        }
        let waited = started.elapsed();
        assert!(waited < Duration::from_millis(100), "waited {waited:?}");
        assert!(p.made.get().is_none());
    }
}
