//! Host threads that never run title code.
//!
//! `victory-loader` passes every thread's start and end on to the title
//! images (their TLS callbacks and `DllMain`), which costs about 110 ms a
//! thread with the game's images. Threads started here are left out.

/// Starts a thread named `name` that runs `f`, which must never call
/// title code. The title images do not hear of the thread's start or end.
/// A panic in `f` ends the thread, as in [`std::thread::spawn`]. Nothing
/// joins the thread.
pub fn spawn_host(name: &str, f: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
    #[cfg(windows)]
    return windows::spawn(name, Box::new(f));
    #[cfg(not(windows))]
    std::thread::Builder::new()
        .name(name.into())
        .spawn(f)
        .map(drop)
}

/// Whether a thread whose start routine is at `start` came from
/// [`spawn_host`].
pub fn from_spawn_host(start: usize) -> bool {
    #[cfg(windows)]
    return start == windows::start as *const () as usize;
    #[cfg(not(windows))]
    {
        let _ = start;
        false
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        CreateThread, GetCurrentThread, STACK_SIZE_PARAM_IS_A_RESERVATION, SetThreadDescription,
    };

    type Job = (Vec<u16>, Box<dyn FnOnce() + Send>);

    /// Stack reserved per thread, as Rust's std reserves.
    const STACK: usize = 2 << 20;

    pub(super) fn spawn(name: &str, f: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        let job: *mut Job = Box::into_raw(Box::new((name.encode_utf16().chain([0]).collect(), f)));
        // SAFETY: `start` takes ownership of `job` once the thread runs.
        let thread = unsafe {
            CreateThread(
                std::ptr::null(),
                STACK,
                Some(start),
                job.cast(),
                STACK_SIZE_PARAM_IS_A_RESERVATION,
                std::ptr::null_mut(),
            )
        };
        if thread.is_null() {
            let e = std::io::Error::last_os_error();
            // SAFETY: no thread started, so `job` is still ours.
            drop(unsafe { Box::from_raw(job) });
            return Err(e);
        }
        // SAFETY: our handle to the thread, which runs on without it.
        unsafe { CloseHandle(thread) };
        Ok(())
    }

    pub(super) unsafe extern "system" fn start(job: *mut c_void) -> u32 {
        // SAFETY: the job `spawn` handed to this thread.
        let (name, f) = *unsafe { Box::from_raw(job.cast::<Job>()) };
        // SAFETY: names the calling thread with a NUL-terminated text.
        unsafe { SetThreadDescription(GetCurrentThread(), name.as_ptr()) };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        0
    }
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn host_threads_run_their_job_and_are_known_by_their_start() {
        let (sender, receiver) = std::sync::mpsc::channel();
        super::spawn_host("victory test", move || {
            use windows_sys::Wdk::System::Threading::{
                NtQueryInformationThread, ThreadQuerySetWin32StartAddress,
            };
            let mut start = 0usize;
            // SAFETY: writes one pointer-sized start address of this thread.
            unsafe {
                NtQueryInformationThread(
                    windows_sys::Win32::System::Threading::GetCurrentThread(),
                    ThreadQuerySetWin32StartAddress,
                    (&raw mut start).cast(),
                    size_of::<usize>() as u32,
                    std::ptr::null_mut(),
                )
            };
            sender.send(start).unwrap();
        })
        .unwrap();
        let start = receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert!(super::from_spawn_host(start));
        assert!(!super::from_spawn_host(0));
    }
}
