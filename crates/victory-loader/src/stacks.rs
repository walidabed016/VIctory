//! Where every thread is: `VICTORY_STACKS=<seconds>[,<seconds>...]` prints
//! each thread's unwound stack as `module+rva` that many seconds after
//! [`dump_stacks_on_schedule`], to find what a title waits on.
//! `VICTORY_HEARTBEAT=<rva>` (hex) names a title u64 that the title's main
//! loop keeps storing a QueryPerformanceCounter value in for a watchdog of
//! its own: [`dump_stacks_on_stall`] logs each long gap between those stores
//! and prints the stacks once a gap passes [`STALLED`] seconds.

use std::ffi::c_void;

use windows_sys::Wdk::System::Threading::{
    NtQueryInformationThread, ThreadBasicInformation, ThreadQuerySetWin32StartAddress,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription, OpenThread, ResumeThread,
    SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::{MAX_FRAMES, locate, unwind_into};

/// Reads `VICTORY_STACKS` and, if set, starts a thread that prints every
/// thread's stack at each listed time (seconds from now).
pub fn dump_stacks_on_schedule() {
    let Ok(list) = std::env::var("VICTORY_STACKS") else {
        return;
    };
    let mut times: Vec<f64> = list
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    times.sort_by(f64::total_cmp);
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        for t in times {
            let at = std::time::Duration::from_secs_f64(t.max(0.0));
            std::thread::sleep(at.saturating_sub(start.elapsed()));
            dump_stacks();
        }
    });
}

/// Heartbeat gaps at least this long (seconds) are logged when they end.
const LATE: f64 = 5.0;

/// A heartbeat gap this long (seconds) prints every thread's stack, once a
/// gap. The game's watchdog (title+0x2215a60) ends it after 30 s without a
/// heartbeat at title+0x71b6338.
const STALLED: f64 = 20.0;

/// Reads `VICTORY_HEARTBEAT` and, if it names an aligned u64 in the image
/// at `base` of `size` bytes, starts a thread that looks at it every 100 ms.
pub fn dump_stacks_on_stall(base: usize, size: usize) {
    let Some(rva) = std::env::var("VICTORY_HEARTBEAT")
        .ok()
        .and_then(|r| usize::from_str_radix(r.trim().trim_start_matches("0x"), 16).ok())
        .filter(|&rva| rva % 8 == 0 && rva + 8 <= size)
    else {
        return;
    };
    let mut frequency = 0;
    // SAFETY: writes one i64.
    unsafe { QueryPerformanceFrequency(&mut frequency) };
    tracing::info!("VICTORY_HEARTBEAT: watching title+{rva:#x}");
    let watching = victory_core::thread::spawn_host("victory heartbeat", move || {
        let mut gap = Gap::default();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            let mut now = 0;
            // SAFETY: writes one i64; reads an aligned u64 inside the mapped
            // image, checked above.
            let beat = unsafe {
                QueryPerformanceCounter(&mut now);
                ((base + rva) as *const i64).read_volatile()
            };
            match gap.see(beat, now, frequency) {
                Some(Heard::Late(seconds, longest)) => tracing::info!(
                    "title heartbeat after {seconds:.1} s (longest gap {longest:.1} s)"
                ),
                Some(Heard::Stalled(seconds)) => {
                    tracing::warn!("no title heartbeat for {seconds:.1} s; stacks follow");
                    dump_stacks();
                }
                None => {}
            }
        }
    });
    if let Err(e) = watching {
        tracing::warn!(error = %e, "VICTORY_HEARTBEAT: no thread");
    }
}

/// What [`Gap::see`] found worth saying.
#[derive(Debug, PartialEq)]
enum Heard {
    /// A gap of this many seconds ended; the longest so far.
    Late(f64, f64),
    /// No heartbeat for this many seconds, said once a gap.
    Stalled(f64),
}

/// The heartbeat last seen (0 before the first), the longest gap since, and
/// whether this gap was said to stall.
#[derive(Default)]
struct Gap {
    beat: i64,
    longest: f64,
    stalled: bool,
}

impl Gap {
    /// Sees heartbeat `beat` at counter `now`, which counts `frequency` a
    /// second.
    fn see(&mut self, beat: i64, now: i64, frequency: i64) -> Option<Heard> {
        let seconds = |ticks: i64| ticks as f64 / frequency as f64;
        if beat != self.beat {
            let first = self.beat == 0;
            let gap = seconds(beat - self.beat);
            (self.beat, self.stalled) = (beat, false);
            if first {
                return None;
            }
            self.longest = self.longest.max(gap);
            return (gap >= LATE).then_some(Heard::Late(gap, self.longest));
        }
        if beat == 0 || self.stalled || seconds(now - beat) < STALLED {
            return None;
        }
        self.stalled = true;
        Some(Heard::Stalled(seconds(now - beat)))
    }
}

/// `THREAD_BASIC_INFORMATION` (ntddk).
#[repr(C)]
struct BasicInfo {
    exit_status: i32,
    teb: *const usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// Prints every other thread's stack to stderr. Each thread is suspended
/// only while its context is read and unwound into a buffer allocated
/// beforehand (it may hold the heap lock).
pub fn dump_stacks() {
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return;
    }
    let mut ids = Vec::new();
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    eprintln!("VIctory: stacks of {} threads", ids.len());
    for id in ids {
        // SAFETY: opens a thread of our own process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        if thread.is_null() {
            continue;
        }
        let (frames, (priority, base)) = frames_of(thread);
        eprintln!(
            "  thread {id} \"{}\" start {} priority {priority} (base {base})",
            name_of(thread),
            locate(start_of(thread))
        );
        for (i, pc) in frames.into_iter().enumerate() {
            eprintln!("    #{i:<2} {}", locate(pc as usize));
        }
        // SAFETY: we opened it.
        unsafe { CloseHandle(thread) };
    }
}

/// The thread's frames and its current and base scheduling priority (0-31).
fn frames_of(thread: HANDLE) -> (Vec<u64>, (i32, i32)) {
    let mut frames = Vec::with_capacity(MAX_FRAMES);
    let mut info: BasicInfo = BasicInfo {
        exit_status: 0,
        teb: std::ptr::null(),
        client_id: [0; 2],
        affinity: 0,
        priority: 0,
        base_priority: 0,
    };
    // SAFETY: writes `info`, whose layout and size are the system's.
    let status = unsafe {
        NtQueryInformationThread(
            thread,
            ThreadBasicInformation,
            (&raw mut info).cast(),
            size_of::<BasicInfo>() as u32,
            std::ptr::null_mut(),
        )
    };
    if status != 0 || info.teb.is_null() {
        return (frames, (0, 0));
    }
    // SAFETY: the TEB of a live thread of this process: its NT_TIB holds
    // the stack base at +8 and the stack limit at +0x10.
    let (high, low) = unsafe { (info.teb.add(1).read(), info.teb.add(2).read()) };
    // SAFETY: zeroed is a valid CONTEXT.
    let mut context = Aligned(unsafe { std::mem::zeroed() });
    context.0.ContextFlags = CONTEXT_FULL_AMD64;
    // SAFETY: suspends, reads and resumes a thread that is not this one;
    // nothing between suspend and resume allocates or takes a lock the
    // thread could hold besides the unwinder's own.
    unsafe {
        if SuspendThread(thread) == u32::MAX {
            return (frames, (info.priority, info.base_priority));
        }
        if GetThreadContext(thread, &mut context.0) != 0 {
            unwind_into(context.0, low, high, &mut frames);
        }
        ResumeThread(thread);
    }
    (frames, (info.priority, info.base_priority))
}

fn start_of(thread: HANDLE) -> usize {
    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    unsafe {
        NtQueryInformationThread(
            thread,
            ThreadQuerySetWin32StartAddress,
            (&raw mut start).cast(),
            size_of::<usize>() as u32,
            std::ptr::null_mut(),
        )
    };
    start
}

fn name_of(thread: HANDLE) -> String {
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast::<c_void>());
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_threads_stack_unwinds_to_its_start() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let parked = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            stop_rx.recv().ok();
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        assert!(!thread.is_null());
        let (frames, _) = frames_of(thread);
        // Waiting in the kernel, several frames down from the thread start.
        assert!(frames.len() >= 3, "{frames:?}");
        assert!(start_of(thread) != 0);
        // SAFETY: opened above.
        unsafe { CloseHandle(thread) };
        dump_stacks();
        stop_tx.send(()).unwrap();
        parked.join().unwrap();
    }

    #[test]
    fn heartbeat_gaps_are_said_once() {
        let mut gap = Gap::default();
        // Ten counts a second.
        let mut see = |beat, now| gap.see(beat, now, 10);
        assert_eq!(see(0, 5), None, "no heartbeat yet");
        assert_eq!(see(10, 10), None, "the first");
        assert_eq!(see(20, 20), None, "a second later");
        assert_eq!(see(20, 219), None, "19.9 s");
        assert_eq!(see(20, 220), Some(Heard::Stalled(20.0)));
        assert_eq!(see(20, 300), None, "said once a gap");
        assert_eq!(see(330, 330), Some(Heard::Late(31.0, 31.0)));
        assert_eq!(see(340, 340), None, "a second");
        assert_eq!(see(400, 400), Some(Heard::Late(6.0, 31.0)));
        assert_eq!(
            see(400, 600),
            Some(Heard::Stalled(20.0)),
            "a new gap says it again"
        );
    }
}
