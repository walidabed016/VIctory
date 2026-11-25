//! Tests: the GPU side guarding title memory again between the runtime's
//! opening of it ([`super::files::opening`]) and the write that follows,
//! and a stand-in for the GPU side's handler of writes to guarded memory.

use std::sync::Mutex;

use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};

/// Destinations the GPU side guards again once they are opened.
static GUARD_AGAIN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Ranges whose guard [`stand_in_guard`] stands in for.
static GUARDED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// The next opening of `at` is followed by a guard on its page.
pub(crate) fn guard_again(at: usize) {
    GUARD_AGAIN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(at);
}

/// Whether the guard [`guard_again`] asked for at `at` is still to come.
pub(crate) fn still_to_guard(at: usize) -> bool {
    GUARD_AGAIN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&at)
}

/// What the GPU side's copies do between the opening of a destination and
/// its write: guard it again, read-only.
pub(crate) fn guarded_again(at: usize) {
    let mut again = GUARD_AGAIN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(i) = again.iter().position(|&d| d == at) else {
        return;
    };
    again.swap_remove(i);
    let mut old = 0;
    // SAFETY: a test's own allocation; only its protection changes.
    let guarded = unsafe { VirtualProtect(at as _, 1, PAGE_READONLY, &mut old) };
    assert_ne!(guarded, 0);
}

/// Stands in for the GPU side's handler for `[at, at + len)`: a write
/// that faults there gives its page write access back and goes on.
pub(crate) fn stand_in_guard(at: usize, len: usize) {
    use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn reopen(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        let at = r.ExceptionInformation[1];
        let ours = r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
            && r.NumberParameters >= 2
            && r.ExceptionInformation[0] == 1
            && GUARDED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .any(|&(a, b)| a <= at && at < b);
        if !ours {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let mut old = 0;
        // SAFETY: a page of a test's own allocation.
        unsafe { VirtualProtect(at as _, 1, PAGE_READWRITE, &mut old) };
        EXCEPTION_CONTINUE_EXECUTION
    }
    GUARDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((at, at + len));
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only changes the protection of the tests' ranges.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(reopen));
    });
}

/// A fresh read-write allocation of `len` bytes, never freed.
pub(crate) fn memory(len: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, VirtualAlloc};
    // SAFETY: a fresh allocation of our own.
    let at = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            len,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    assert!(!at.is_null());
    at as usize
}
