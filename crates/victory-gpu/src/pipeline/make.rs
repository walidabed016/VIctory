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
//! for it within a budget per frame, so the cheap ones (menus, one-off
//! passes such as a colour-grading bake) are there at once as before; past
//! the budget, or when no worker takes it at once (all are busy, so it
//! would not be made in time anyway), a draw or dispatch whose pipeline is
//! not ready is dropped, as an untranslated one is. Waiting for pipelines
//! no worker had taken kept run 70's start screen at 1.9 fps for minutes.
//!
//! shortcut: the budget (half a second of waiting per title frame) is a
//! guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

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
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
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

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(500);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// How long a frame waits for a pipeline no worker has taken.
const TAKE: Duration = Duration::from_millis(10);

/// Time title threads waited for pipelines, over all frames.
static WAITED_US: AtomicU64 = AtomicU64::new(0);

/// Time title threads waited for pipelines since the last call (for the
/// frame line).
pub(crate) fn take_waited() -> Duration {
    Duration::from_micros(WAITED_US.swap(0, Ordering::Relaxed))
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
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
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left, and past `TAKE` only
/// while a worker is making it.
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
        let waited = started.elapsed();
        let taken = p.taken.load(Ordering::Relaxed);
        let rest = left.saturating_sub(waited);
        if rest.is_zero() || (!taken && waited >= TAKE) {
            break;
        }
        let rest = if taken { rest } else { rest.min(TAKE - waited) };
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    let waited = started.elapsed().as_micros() as u64;
    FRAME_US.fetch_add(waited, Ordering::Relaxed);
    WAITED_US.fetch_add(waited, Ordering::Relaxed);
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
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A frame does not spend its budget on a pipeline no worker takes (all
    /// are busy: it would not be made in time).
    #[test]
    fn frames_do_not_wait_for_pipelines_no_worker_takes() {
        let p = Pending::new(Vec::new(), None);
        let started = Instant::now();
        wait(&p, u64::MAX - 1);
        let waited = started.elapsed();
        assert!(waited < BUDGET / 2, "waited {waited:?}");
        assert!(p.made.get().is_none());
    }
}
