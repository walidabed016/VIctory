//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given behind a wait not met yet, in order
    /// ([`waits`]).
    #[cfg(windows)]
    held: std::sync::Mutex<std::collections::VecDeque<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// Whether queues keep the order the title's waits ask for (on by default).
/// A queue told to wait for a fence, or given a list that waits for a
/// memory value, holds what it is given from there on until the fence or
/// the value is there, or until a submission that signals or writes it is
/// queued ahead of it: the host runs every queue's submissions on one
/// queue, in order ([`release_held`]). Lists are cut at their waits
/// (`crate::list::cut_at`), so that a queue stands at a wait with what came
/// before it run. The title runs its depth processing on the compute queue
/// behind a wait for a value the graphics queue writes after the depth pass;
/// run at once, it read the depth of the frame before (Lucia's hair black
/// in run 157). `VICTORY_QUEUE_WAITS=0`: every submission runs when it is
/// made and waits are ignored, as before.
#[cfg(windows)]
pub(crate) fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_QUEUE_WAITS").is_none_or(|v| v != "0");
        tracing::info!(
            on,
            "queues keep the order of the title's waits (VICTORY_QUEUE_WAITS)"
        );
        on
    })
}

/// One thing a queue holds.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence (held) to reach the value.
    Fence(usize, u64),
    /// The queue waits for a memory value.
    Value(crate::list::ValueWait),
    /// Host lists (addresses) to run in order, when they were given, and
    /// whether the queue was left holding them by the call that gave them.
    Run(Vec<usize>, std::time::Instant, bool),
    /// A signal of the fence (held), due after what the queue holds before.
    Signal(usize, u64),
}

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The queues that hold something (addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Held while a queue gives the host a submission or a signal, or holds
/// one: the host gets them in the order the queues' waits allow.
#[cfg(windows)]
static SUBMITTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The host lists queues hold (addresses), each as often as held.
#[cfg(windows)]
static HELD_LISTS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Bottom-of-pipe values submissions queued or in flight store once they
/// have run, pipelined: host list, address, value, bytes ([`value_met`]).
#[cfg(windows)]
static PROMISED: std::sync::Mutex<Vec<(usize, usize, u64, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// Since the last [`log_holds`]: submissions a queue held, microseconds
/// they were held and the longest hold; waits passed unmet (at presents,
/// and where queues were run on for a CPU wait for a fence or a page
/// remap), and those two runs.
#[cfg(windows)]
static HOLDS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Host lists a title list let go of (`crate::list`) while a queue held or
/// a submission ran some of them, with its upload memory and draw state
/// (dynamic descriptor slots) if a queue held them: kept until none is.
#[cfg(windows)]
pub(crate) struct Left {
    pub(crate) lists: Vec<Box<crate::host::HostList>>,
    pub(crate) _uploads: Vec<Vec<u8>>,
    pub(crate) _state: Option<crate::draw::State>,
}

// SAFETY: D3D12 objects are free-threaded; nothing records into a list
// left, and only the queues and the host submit it.
#[cfg(windows)]
unsafe impl Send for Left {}

#[cfg(windows)]
static LEFT: std::sync::Mutex<Vec<Left>> = std::sync::Mutex::new(Vec::new());

/// Keeps what a title list left until no queue holds and no submission
/// runs any of its host lists; those go back for spare use then.
#[cfg(windows)]
pub(crate) fn leave(left: Left) {
    lock(&LEFT).push(left);
    sweep();
}

#[cfg(windows)]
fn sweep() {
    let done: Vec<Left> = {
        let mut left = lock(&LEFT);
        if left.is_empty() {
            return;
        }
        let busy = |l: &Left| {
            l.lists.iter().any(|host| {
                let at = std::ptr::from_ref(&**host) as usize;
                holds(at) || crate::pipelined::flying(at)
            })
        };
        let (kept, done) = std::mem::take(&mut *left).into_iter().partition(busy);
        *left = kept;
        done
    };
    for left in done {
        left.lists.into_iter().for_each(crate::host::give_back);
    }
}

/// Whether a queue holds host list `list` (its address).
#[cfg(windows)]
pub(crate) fn holds(list: usize) -> bool {
    let none = lock(&WAITING).is_empty();
    !none && lock(&HELD_LISTS).contains(&list)
}

/// Whether a queue may pass the fence wait: the fence is there, or a
/// submission queued or in flight signals it.
#[cfg(windows)]
fn fence_met(fence: usize, value: u64) -> bool {
    crate::fence::value(fence as This) >= value
        || crate::pipelined::will_signal(fence as This, value)
}

/// Whether a queue may pass `wait`: memory holds the value, or, pipelined,
/// a submission queued or in flight stores it at the bottom of the pipe:
/// what the queue runs next goes to the host behind that one, which is the
/// order the wait asks for.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    if wait.met() {
        return true;
    }
    let (at, bytes) = wait.place();
    let promised = lock(&PROMISED);
    promised
        .iter()
        .rev()
        .filter(|p| p.1 == at && p.3 >= bytes)
        .find(|p| crate::pipelined::flying(p.0))
        .is_some_and(|p| wait.holds_for(p.2))
}

/// Notes the bottom-of-pipe values of host lists `runs`, just given to the
/// host ([`value_met`]).
#[cfg(windows)]
fn promise(runs: &[&crate::host::HostList]) {
    if !crate::pipelined::on() {
        return;
    }
    let lists: Vec<usize> = runs
        .iter()
        .map(|&l| std::ptr::from_ref(l) as usize)
        .collect();
    let mut promised = lock(&PROMISED);
    // A host list run again (or recorded again) promises anew.
    promised.retain(|p| !lists.contains(&p.0));
    if promised.len() > 256 {
        promised.retain(|p| crate::pipelined::flying(p.0));
    }
    for (&list, at) in runs.iter().zip(lists) {
        let writes = list.value_writes();
        promised.extend(
            writes
                .into_iter()
                .map(|(to, value, bytes)| (at, to, value, bytes)),
        );
    }
}

/// Gives the host host lists (addresses) as one submission, in order.
#[cfg(windows)]
fn run(host: &crate::host::Host, lists: &[usize]) {
    if lists.is_empty() {
        return;
    }
    // SAFETY: held lists are live: their title lists leave them to the
    // queues when reset or let go of (`crate::list`).
    let runs: Vec<&crate::host::HostList> = lists
        .iter()
        .map(|&l| unsafe { &*(l as *const _) })
        .collect();
    host.submit(&runs);
    promise(&runs);
}

/// Runs what queues hold as far as their waits are met (`all`: all of it,
/// waits met or not), each queue's in the order it was given. Called where
/// a wait may have been met: when a queue is given a submission or a
/// signal, when the title signals a fence, and at presents (`all`).
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    if lock(&WAITING).is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    let _one = lock(&SUBMITTING);
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            // SAFETY: a live queue: one let go of runs what it holds first.
            let held = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(held);
            // Runs ready, made one submission (and taken off HELD_LISTS once
            // given to the host, so that a reset finds them held or flying).
            let mut ready: Vec<usize> = Vec::new();
            let flush = |ready: &mut Vec<usize>| {
                run(host, ready);
                let mut lists = lock(&HELD_LISTS);
                for l in ready.drain(..) {
                    if let Some(i) = lists.iter().position(|&h| h == l) {
                        lists.swap_remove(i);
                    }
                }
            };
            loop {
                let met = match held.front() {
                    None => break,
                    Some(Held::Fence(fence, value)) => fence_met(*fence, *value),
                    Some(Held::Value(wait)) => value_met(wait),
                    Some(Held::Run(..) | Held::Signal(..)) => true,
                };
                if !met && !ready.is_empty() {
                    // What is ready may be what the queue waits for.
                    flush(&mut ready);
                    continue;
                }
                if !met && !all {
                    break;
                }
                if !met {
                    HOLDS[3].fetch_add(1, Ordering::Relaxed);
                    static SEEN: AtomicU64 = AtomicU64::new(0);
                    if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                        tracing::warn!(
                            kind = queue(q as This).kind,
                            "a queue's wait was not met when it was run on (at a present, a CPU \
                             fence wait or a page remap)"
                        );
                    }
                }
                moved = true;
                match held.pop_front() {
                    Some(Held::Fence(fence, _)) => {
                        crate::fence::release(fence as This);
                    }
                    Some(Held::Run(lists, since, waited)) => {
                        if waited {
                            let us = since.elapsed().as_micros() as u64;
                            HOLDS[0].fetch_add(1, Ordering::Relaxed);
                            HOLDS[1].fetch_add(us, Ordering::Relaxed);
                            HOLDS[2].fetch_max(us, Ordering::Relaxed);
                        }
                        ready.extend(lists);
                    }
                    Some(Held::Signal(fence, value)) => {
                        flush(&mut ready);
                        signal_now(fence as This, value);
                        crate::fence::release(fence as This);
                    }
                    Some(Held::Value(_)) | None => {}
                }
            }
            flush(&mut ready);
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            break;
        }
    }
    sweep();
}

/// After a call that gave queue `this` something: what it still holds was
/// left held by that call (counted when it runs: [`HOLDS`]).
#[cfg(windows)]
fn left_holding(this: This) {
    let mut held = lock(&queue(this).held);
    let mut first = true;
    for h in held.iter_mut() {
        if let Held::Run(_, _, waited) = h
            && !*waited
        {
            *waited = true;
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if first && SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    "a queue holds a submission for a wait"
                );
            }
            first = false;
        }
    }
}

/// Adds queue `this` to those that hold something.
#[cfg(windows)]
fn waiting(this: This) {
    let mut waiting = lock(&WAITING);
    if !waiting.contains(&(this as usize)) {
        waiting.push(this as usize);
    }
}

/// A submission to a queue that keeps the order of waits ([`waits`]): run
/// now if the queue holds nothing and no list waits for a memory value;
/// otherwise held, part by part, and run as far as the waits are met.
/// (Only a queue that holds something is in [`WAITING`], and it leaves it
/// once what it held was given to the host.)
#[cfg(windows)]
fn submit(this: This, host: &crate::host::Host, lists: &[This]) {
    let holding = lock(&WAITING).contains(&(this as usize));
    if !holding && !lists.iter().any(|&l| crate::list::is_cut(l)) {
        let runs: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.submit(&runs);
        promise(&runs);
    } else {
        let _one = lock(&SUBMITTING);
        let pieces = crate::list::pieces(lists);
        lock(&HELD_LISTS).extend(pieces.iter().flat_map(|(_, hosts)| hosts.iter().copied()));
        let now = std::time::Instant::now();
        {
            let mut held = lock(&queue(this).held);
            for (waits, hosts) in pieces {
                held.extend(waits.into_iter().map(Held::Value));
                held.push_back(Held::Run(hosts, now, false));
            }
        }
        waiting(this);
    }
    // What ran may be what another queue waits for.
    release_held(false);
    left_holding(this);
}

/// Before the title waits on the CPU for `fence` to reach `value`: if a
/// queue holds the signal that gets it there, what the queues hold runs,
/// past waits not met if need be. The title thread that waits may be the
/// one that would give a queue what it waits for.
#[cfg(windows)]
pub(crate) fn before_cpu_wait(fence: This, value: u64) {
    let none = lock(&WAITING).is_empty();
    if none || fence_met(fence as usize, value) {
        return;
    }
    let holds_signal = || {
        let queues = lock(&WAITING).clone();
        queues.into_iter().any(|q| {
            // SAFETY: a live queue, as in [`release_held`].
            let held = unsafe { &(*(q as *const Queue)).held };
            lock(held)
                .iter()
                .any(|h| matches!(*h, Held::Signal(f, v) if f == fence as usize && v >= value))
        })
    };
    if !holds_signal() {
        return;
    }
    release_held(false);
    if fence_met(fence as usize, value) || !holds_signal() {
        return;
    }
    HOLDS[4].fetch_add(1, Ordering::Relaxed);
    release_held(true);
}

/// Logs how often and how long queues held submissions for the title's
/// waits, per frame over `frames` frames, and resets the counts.
#[cfg(windows)]
pub(crate) fn log_holds(frames: f64) {
    if !waits() {
        return;
    }
    let f = frames.max(1.0);
    let [held, us, longest, unmet, fence, remap] =
        HOLDS.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    let [recorded, cuts] = crate::list::take_waits();
    let average = if held > 0 {
        us as f64 / 1e3 / held as f64
    } else {
        0.0
    };
    tracing::info!(
        "queue waits per frame: {:.1} waits for memory values ({:.1} lists cut at one), \
         {:.1} submissions held {average:.2} ms on average (longest {:.1} ms); {:.1} waits \
         passed unmet, at presents or when queues were run on for a CPU fence wait ({:.1}) or \
         a page remap ({:.1})",
        recorded as f64 / f,
        cuts as f64 / f,
        held as f64 / f,
        longest as f64 / 1e3,
        unmet as f64 / f,
        fence as f64 / f,
        remap as f64 / f,
    );
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    // A queue let go of runs what it holds first.
    #[cfg(windows)]
    if lock(&WAITING).contains(&(this as usize)) {
        release_held(true);
    }
    object::release_boxed::<Queue>(this)
}

/// Runs the lists' host lists and waits for them (decision 0007).
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        if waits() && !this.is_null() {
            submit(this, host, lists);
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.submit(&host_lists);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if waits() && !this.is_null() {
        let holding = lock(&WAITING).contains(&(this as usize));
        let signalled = if holding {
            let _one = lock(&SUBMITTING);
            let mut held = lock(&queue(this).held);
            if !held.is_empty() {
                // After what the queue holds.
                crate::object::add_ref(fence);
                held.push_back(Held::Signal(fence as usize, value));
                return S_OK;
            }
            drop(held);
            signal_now(fence, value)
        } else {
            signal_now(fence, value)
        };
        // Another queue may wait for the fence.
        release_held(false);
        return signalled;
    }
    let _ = this;
    signal_now(fence, value)
}

fn signal_now(fence: This, value: u64) -> HRESULT {
    #[cfg(feature = "diagnostics")]
    crate::fence::queue_signal(fence);
    if crate::pipelined::signal_after(fence, value) {
        return S_OK;
    }
    crate::fence::signal(fence, value)
}

/// From here on the queue runs nothing it is given until `fence` reaches
/// `value` ([`waits`]): what it is given is held until then, or until a
/// submission queued or in flight signals it.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if waits() && !this.is_null() {
        let holding = lock(&WAITING).contains(&(this as usize));
        if !holding && fence_met(fence as usize, value) {
            return S_OK;
        }
        {
            let _one = lock(&SUBMITTING);
            let mut held = lock(&queue(this).held);
            if held.is_empty() && fence_met(fence as usize, value) {
                return S_OK;
            }
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    value,
                    "a queue waits for a fence not reached yet"
                );
            }
            crate::object::add_ref(fence);
            held.push_back(Held::Fence(fence as usize, value));
            drop(held);
            waiting(this);
        }
        release_held(false);
        left_holding(this);
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // shortcut: what a queue still holds at a present runs now, waits met or
    // not: a wait is not kept across frames.
    #[cfg(windows)]
    release_held(true);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(windows)]
    {
        let plane = match planes {
            0 => None,
            // SAFETY: the title passes `planes` plane descriptions (null is
            // tolerated), read now: they live only for the call.
            _ => unsafe { plane_parameters.as_ref() },
        };
        let shown = Shown(
            plane
                .filter(|p| p.count > 0 && !p.resources.is_null())
                // SAFETY: the plane lists `count` resources.
                .map(|p| unsafe { *p.resources }),
        );
        // After the submissions queued before: they set the real states of
        // what they draw (`host_transition`).
        if !crate::pipelined::queue_present(Box::new(move || shown.show(frame))) {
            crate::pipelined::drained();
            shown.show(frame);
        }
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}

/// The resource a present shows, if any. The title keeps it until a fence
/// signalled after the present, which a queued present goes before.
#[cfg(windows)]
#[derive(Clone, Copy)]
struct Shown(Option<This>);

// SAFETY: our objects are shared between threads; see above for its life.
#[cfg(windows)]
unsafe impl Send for Shown {}

#[cfg(windows)]
impl Shown {
    fn show(self, frame: u64) {
        #[cfg(feature = "diagnostics")]
        crate::capture::note_present(frame);
        #[cfg(feature = "diagnostics")]
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
        if let Some(resource) = self.0
            && let Some(texture) = crate::resource::host_resource(resource)
            && let Some(host) = crate::host::host()
        {
            const COPY_SOURCE: u32 = 0x800;
            let state = crate::resource::host_transition(resource, COPY_SOURCE)
                .map_or(COPY_SOURCE, |(_, before)| before);
            let shown = host.present(texture, state, frame);
            crate::resource::host_transition(resource, state);
            #[cfg(feature = "diagnostics")]
            blinks(frame, resource, state, shown);
            #[cfg(not(feature = "diagnostics"))]
            let _ = shown;
        }
    }
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    // After what the queue holds: it runs past its waits if need be.
    #[cfg(windows)]
    if !this.is_null() && !lock(&queue(this).held).is_empty() {
        release_held(false);
        if !lock(&queue(this).held).is_empty() {
            HOLDS[5].fetch_add(1, Ordering::Relaxed);
            release_held(true);
        }
    }
    #[cfg(not(windows))]
    let _ = this;
    // Pipelined: the runtime remaps memory submissions in flight copy back
    // into; with `crate::pipelined::narrow_remaps`, only those wait.
    let narrow = crate::pipelined::narrow_remaps();
    if !narrow {
        crate::pipelined::wait_all();
    }
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    if narrow {
        let remapped: Vec<(usize, usize)> = batches
            .into_iter()
            .flatten()
            .filter(|b| !b.ranges.is_null())
            .map(|b| {
                // SAFETY: each batch holds `count` ranges.
                let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
                let len: u64 = ranges.iter().map(|r| u64::from(r.count) * PAGE).sum();
                (b.destination as usize, (b.destination + len) as usize)
            })
            .collect();
        crate::pipelined::wait_remapped(&remapped);
    }
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    /// A queue told to wait for a fence holds what it is given, signals
    /// too, until the title signals the fence; a present runs what a queue
    /// still holds.
    #[cfg(windows)]
    #[test]
    fn queues_hold_what_follows_a_wait_for_a_fence() {
        let _serial = crate::watch::serial();
        if crate::host::host().is_none() || !waits() {
            return;
        }
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 2,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        let fence = || {
            let mut f = null;
            let iid = &crate::fence::FENCE_IIDS[0];
            assert_eq!(crate::fence::create(null, 0, 0, iid, &mut f), S_OK);
            f
        };
        let (waited, signalled) = (fence(), fence());
        let value = |f: This| {
            crate::pipelined::wait_all();
            crate::fence::value(f)
        };
        assert_eq!(wait(q, waited, 1), S_OK);
        assert_eq!(signal(q, signalled, 5), S_OK);
        assert_eq!(value(signalled), 0, "held behind the wait");
        // The title's own signal (the fence's method at 0x50).
        let title_signal = slot::<extern "system" fn(This, u64) -> HRESULT>(waited, 0x50);
        assert_eq!(title_signal(waited, 1), S_OK);
        assert_eq!(value(signalled), 5, "run once the fence got there");
        assert_eq!(wait(q, waited, 2), S_OK);
        assert_eq!(signal(q, signalled, 6), S_OK);
        assert_eq!(value(signalled), 5);
        release_held(true);
        assert_eq!(value(signalled), 6, "run at a present");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for o in [waited, signalled, q] {
            assert_eq!(release(o), 0);
        }
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
