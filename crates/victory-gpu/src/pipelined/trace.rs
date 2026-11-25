//! Diagnostics: when each recent submission started being prepared, was
//! launched, was seen done on the GPU and was copied back, and what each
//! wait of the title's waited on, so the chain the frame is serialised on
//! shows. [`log`] sums them per frame; with `VICTORY_PIPE_TRACE=1` it also
//! lists the newest submissions and waits now and then.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Submission {
    seq: u64,
    thread: u32,
    /// When its preparation started (after waiting for room).
    prepared: Instant,
    launched: Option<Instant>,
    /// When the completion thread saw it done on the GPU, and whether it
    /// was done before the thread got to it (behind older copy-backs).
    gpu: Option<(Instant, bool)>,
    retired: Option<Instant>,
    /// When it started and ended on the GPU (`VICTORY_GPU_SPANS=1`).
    span: Option<(Instant, Instant)>,
}

#[derive(Clone, Copy)]
struct Wait {
    thread: u32,
    /// [`super::Why::index`].
    why: usize,
    /// The submission it waited for.
    seq: u64,
    from: Instant,
    to: Instant,
}

#[derive(Default)]
struct Trace {
    submissions: VecDeque<Submission>,
    waits: VecDeque<Wait>,
    /// When [`log`] last summed.
    since: Option<Instant>,
    /// Waits for every submission: where called, when they ended, ms.
    alls: VecDeque<(&'static std::panic::Location<'static>, Instant, f64)>,
}

static TRACE: Mutex<Option<Trace>> = Mutex::new(None);

/// Submissions kept, and waits.
const KEEP: usize = 1024;

/// A small number per thread.
fn thread_id() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    thread_local!(static ID: u32 = NEXT.fetch_add(1, Relaxed));
    ID.with(|id| *id)
}

fn with<R>(f: impl FnOnce(&mut Trace) -> R) -> R {
    let mut trace = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    f(trace.get_or_insert_with(Trace::default))
}

fn submission(t: &mut Trace, seq: u64) -> Option<&mut Submission> {
    t.submissions.iter_mut().rev().find(|s| s.seq == seq)
}

/// A thread's preparing and waiting times, added on first use.
fn thread(threads: &mut Vec<(u32, f64, f64)>, id: u32) -> &mut (u32, f64, f64) {
    let i = match threads.iter().position(|t| t.0 == id) {
        Some(i) => i,
        None => {
            threads.push((id, 0.0, 0.0));
            threads.len() - 1
        }
    };
    &mut threads[i]
}

/// Submission `seq` starts being prepared, on this thread.
pub(super) fn preparing(seq: u64) {
    let (thread, now) = (thread_id(), Instant::now());
    with(|t| {
        if submission(t, seq).is_some() {
            return;
        }
        t.submissions.push_back(Submission {
            seq,
            thread,
            prepared: now,
            launched: None,
            gpu: None,
            retired: None,
            span: None,
        });
        if t.submissions.len() > KEEP {
            t.submissions.pop_front();
        }
    });
}

pub(super) fn launched(seq: u64) {
    let now = Instant::now();
    with(|t| submission(t, seq).map(|s| s.launched = Some(now)));
}

pub(super) fn gpu_done(seq: u64, early: bool) {
    let now = Instant::now();
    with(|t| submission(t, seq).map(|s| s.gpu = Some((now, early))));
}

/// Submission `seq` ran on the GPU from `start` to `end`.
pub(super) fn span(seq: u64, start: Instant, end: Instant) {
    with(|t| submission(t, seq).map(|s| s.span = Some((start, end))));
}

/// A wait for every submission, called at `caller`, waited since `from`.
pub(super) fn waited_all(caller: &'static std::panic::Location<'static>, from: Instant) {
    let to = Instant::now();
    let ms = to.saturating_duration_since(from).as_secs_f64() * 1e3;
    with(|t| {
        t.alls.push_back((caller, to, ms));
        if t.alls.len() > KEEP {
            t.alls.pop_front();
        }
    });
}

pub(super) fn retired(seq: u64) {
    let now = Instant::now();
    with(|t| submission(t, seq).map(|s| s.retired = Some(now)));
}

/// This thread waited since `from` for submission `seq`, for `why`.
pub(super) fn waited(why: usize, seq: u64, from: Instant) {
    let (thread, to) = (thread_id(), Instant::now());
    with(|t| {
        t.waits.push_back(Wait {
            thread,
            why,
            seq,
            from,
            to,
        });
        if t.waits.len() > 4 * KEEP {
            t.waits.pop_front();
        }
    });
}

/// Sums the submissions retired and the waits ended since the last call,
/// per frame over `frames` frames; `names` names the waits' reasons.
pub(super) fn log(frames: f64, names: &[&str]) {
    static WINDOWS: AtomicU64 = AtomicU64::new(0);
    let now = Instant::now();
    let (submissions, waits, alls, since) = with(|t| {
        let since = t.since.replace(now);
        (
            t.submissions.iter().copied().collect::<Vec<_>>(),
            t.waits.iter().copied().collect::<Vec<_>>(),
            t.alls.iter().copied().collect::<Vec<_>>(),
            since,
        )
    });
    let Some(since) = since else {
        return;
    };
    let ms = |a: Instant, b: Instant| b.saturating_duration_since(a).as_secs_f64() * 1e3;
    let f = frames.max(1.0);
    let done: Vec<&Submission> = submissions
        .iter()
        .filter(|s| s.retired.is_some_and(|r| r > since))
        .collect();
    // Per submission: preparing, the GPU idle before it (nothing queued),
    // the completion thread waiting for it and copying it back.
    let (mut prep, mut idle, mut waiting, mut copying) = (0.0, 0.0, 0.0, 0.0);
    let mut threads: Vec<(u32, f64, f64)> = Vec::new();
    for s in &done {
        let (Some(launched), Some((gpu, early)), Some(retired)) = (s.launched, s.gpu, s.retired)
        else {
            continue;
        };
        let p = ms(s.prepared, launched);
        prep += p;
        thread(&mut threads, s.thread).1 += p;
        copying += ms(gpu, retired);
        let before = submissions.iter().find(|b| b.seq + 1 == s.seq);
        if let Some(before) = before
            && let (Some((gpu_before, _)), Some(retired_before)) = (before.gpu, before.retired)
        {
            idle += ms(gpu_before, launched);
            if !early {
                waiting += ms(retired_before.max(launched), gpu);
            }
        }
    }
    // Each wait by what the submission it waited for was doing: on the GPU
    // (or queued for it), behind older submissions' copy-backs, or being
    // copied back.
    let mut by = vec![[0.0f64; 4]; names.len()];
    for w in waits.iter().filter(|w| w.to > since) {
        let all = ms(w.from, w.to);
        let s = submissions.iter().find(|s| s.seq == w.seq);
        let parts = match s.and_then(|s| s.gpu.zip(s.retired)) {
            Some(((gpu, early), _)) => {
                let before = ms(w.from, gpu.min(w.to));
                let after = all - before;
                if early {
                    [0.0, before, after, 0.0]
                } else {
                    [before, 0.0, after, 0.0]
                }
            }
            None => [0.0, 0.0, 0.0, all],
        };
        if let Some(b) = by.get_mut(w.why) {
            for (sum, part) in b.iter_mut().zip(parts) {
                *sum += part;
            }
        }
        thread(&mut threads, w.thread).2 += all;
    }
    let waits_text: Vec<String> = names
        .iter()
        .zip(&by)
        .filter(|(_, b)| b.iter().sum::<f64>() > 0.0)
        .map(|(name, b)| {
            format!(
                "{name} {:.1}/{:.1}/{:.1}/{:.1}",
                b[0] / f,
                b[1] / f,
                b[2] / f,
                b[3] / f
            )
        })
        .collect();
    let threads_text: Vec<String> = threads
        .iter()
        .map(|(id, p, w)| format!("t{id} preparing {:.1} waiting {:.1}", p / f, w / f))
        .collect();
    tracing::info!(
        "pipe trace per frame over {} submissions: preparing {:.1} ms; GPU idle with nothing \
         queued {:.1} ms; completion thread waiting for the GPU {:.1} ms, copying back {:.1} ms; \
         title waits in ms (on the GPU or queued / behind older copy-backs / its copy-back / \
         unknown): {}; threads: {}",
        done.len(),
        prep / f,
        idle / f,
        waiting / f,
        copying / f,
        waits_text.join(", "),
        threads_text.join(", ")
    );
    // The GPU's time from the spans: running some submission's lists, idle
    // with the next not launched yet, and idle with it launched but held
    // (fence waits for copy-backs, tile mappings).
    let (mut busy, mut unlaunched, mut held, mut spans) = (0.0, 0.0, 0.0, 0);
    let mut ended: Option<Instant> = None;
    for s in &done {
        let Some((start, end)) = s.span else {
            continue;
        };
        spans += 1;
        match ended {
            Some(e) if start > e => {
                let gap = ms(e, start);
                let late = s.launched.map_or(0.0, |l| ms(e, l).min(gap));
                unlaunched += late;
                held += gap - late;
                busy += ms(start, end);
            }
            Some(e) => busy += ms(e, end),
            None => busy += ms(start, end),
        }
        ended = Some(ended.map_or(end, |e| e.max(end)));
    }
    if spans > 0 {
        tracing::info!(
            "GPU spans per frame over {spans} submissions: running lists {:.1} ms, idle with \
             the next submission not launched {:.1} ms, idle with it launched but held {:.1} ms",
            busy / f,
            unlaunched / f,
            held / f
        );
    }
    let mut callers: Vec<(&std::panic::Location, u64, f64)> = Vec::new();
    for &(caller, _, waited) in alls.iter().filter(|a| a.1 > since) {
        match callers.iter_mut().find(|c| std::ptr::eq(c.0, caller)) {
            Some(c) => (c.1, c.2) = (c.1 + 1, c.2 + waited),
            None => callers.push((caller, 1, waited)),
        }
    }
    if !callers.is_empty() {
        callers.sort_by(|a, b| b.2.total_cmp(&a.2));
        let text: Vec<String> = callers
            .iter()
            .take(6)
            .map(|(c, n, w)| {
                format!(
                    "{}:{} {:.1}x {:.1} ms",
                    c.file(),
                    c.line(),
                    *n as f64 / f,
                    w / f
                )
            })
            .collect();
        tracing::info!(
            "waits for every submission per frame, by caller: {}",
            text.join(", ")
        );
    }
    let listed = std::env::var_os("VICTORY_PIPE_TRACE").is_some_and(|v| v == "1");
    if !listed || !WINDOWS.fetch_add(1, Relaxed).is_multiple_of(6) {
        return;
    }
    let newest: Vec<&&Submission> = done.iter().rev().take(40).collect();
    let Some(base) = newest.last().map(|s| s.prepared) else {
        return;
    };
    let at = |t: Option<Instant>| t.map_or_else(|| "-".into(), |t| format!("{:.1}", ms(base, t)));
    for s in newest.iter().rev() {
        let gpu = s.gpu.map_or_else(
            || "-".into(),
            |(t, early)| format!("{:.1}{}", ms(base, t), if early { "*" } else { "" }),
        );
        let ran = s.span.map_or_else(String::new, |(a, b)| {
            format!(" ran {}-{}", at(Some(a)), at(Some(b)))
        });
        tracing::info!(
            "pipe trace: seq {} t{} prepared {} launched {}{ran} gpu done {} copied back {} (ms; * \
             done before the completion thread got to it)",
            s.seq,
            s.thread,
            at(Some(s.prepared)),
            at(s.launched),
            gpu,
            at(s.retired)
        );
    }
    for w in waits.iter().filter(|w| w.to > base) {
        tracing::info!(
            "pipe wait: t{} {} for seq {} from {:.1} to {:.1} ms",
            w.thread,
            names.get(w.why).copied().unwrap_or("?"),
            w.seq,
            ms(base, w.from),
            ms(base, w.to)
        );
    }
}
