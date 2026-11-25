//! Diagnostics (`VICTORY_GPU_PROGRESS`): how far the GPU got into a
//! submission that does not finish. Around each dispatch the list writes
//! marks into read-back memory, which the CPU reads while the GPU runs:
//! before the dispatch, its number once everything before it has
//! completed; after it, the same number once it has started. When the
//! completion thread has waited [`LATE`] for a submission, it logs the
//! dispatch the marks name and, each time they move on, how long the one
//! before took (to a tenth of a second). Runs 146 and 148 stalled in the
//! first BLAS build with no device removal, so with no breadcrumbs and no
//! Aftermath dump to say which dispatch ran on.
//!
//! The mark before a dispatch waits for all earlier work, so dispatches no
//! longer overlap: slower, and only for finding a stall.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[cfg(test)]
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

/// How long a submission runs before its progress is logged.
const LATE: Duration = Duration::from_secs(2);

/// Dispatch texts kept for the marks to name.
const KEPT: usize = 1 << 15;

/// Whether dispatches are marked.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_GPU_PROGRESS").is_some();
        if on {
            tracing::info!("VICTORY_GPU_PROGRESS: dispatches marked, late submissions logged");
        }
        on
    })
}

/// The read-back buffer the marks land in, mapped: dword 0 is the dispatch
/// everything before which has completed, dword 1 the last one started.
struct Marks {
    buffer: ID3D12Resource,
    at: *const u32,
}

// SAFETY: the buffer is never released, so its mapping stays valid; the
// GPU writes it and the CPU only reads it, volatile.
unsafe impl Send for Marks {}
// SAFETY: as above.
unsafe impl Sync for Marks {}

impl Marks {
    /// (everything before it completed, started).
    fn read(&self) -> (u32, u32) {
        // SAFETY: two dwords of the mapped buffer.
        unsafe { (self.at.read_volatile(), self.at.add(1).read_volatile()) }
    }
}

fn marks() -> Option<&'static Marks> {
    static MARKS: OnceLock<Option<Marks>> = OnceLock::new();
    MARKS
        .get_or_init(|| {
            let device = &crate::host::host()?.device;
            let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut at = std::ptr::null_mut();
            // SAFETY: plain object creation and a mapping of our own buffer.
            let made = unsafe {
                device
                    .CreateCommittedResource(
                        &readback,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut buffer,
                    )
                    .and_then(|()| {
                        let buffer = buffer.as_ref().ok_or(windows::core::Error::empty())?;
                        buffer.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 8 }), Some(&mut at))
                    })
            };
            if let Err(e) = made {
                tracing::warn!(error = %e, "VICTORY_GPU_PROGRESS: no read-back buffer for the marks");
                return None;
            }
            Some(Marks {
                buffer: buffer?,
                at: at.cast(),
            })
        })
        .as_ref()
}

/// The texts of the dispatches marked, oldest first, by their numbers
/// (which rise), and the last number given.
static TEXTS: Mutex<(u32, VecDeque<(u32, String)>)> = Mutex::new((0, VecDeque::new()));

/// Before a dispatch recorded on `list`: notes `text` under a new number,
/// which the GPU writes once everything before the dispatch has completed.
/// The number, for [`after`]; none when off.
pub(crate) fn before(
    list: &ID3D12GraphicsCommandList,
    text: impl FnOnce() -> String,
) -> Option<u32> {
    if !on() {
        return None;
    }
    let m = marks()?;
    let n = {
        let mut texts = TEXTS.lock().unwrap_or_else(|e| e.into_inner());
        texts.0 += 1;
        let n = texts.0;
        texts.1.push_back((n, text()));
        if texts.1.len() > KEPT {
            texts.1.pop_front();
        }
        n
    };
    write(list, m, 0, n, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
    Some(n)
}

/// After the dispatch [`before`] numbered: the GPU writes the number once
/// it has started.
pub(crate) fn after(list: &ID3D12GraphicsCommandList, n: Option<u32>) {
    if let (Some(n), Some(m)) = (n, marks()) {
        write(list, m, 1, n, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
    }
}

fn write(
    list: &ID3D12GraphicsCommandList,
    m: &Marks,
    dword: u64,
    value: u32,
    mode: D3D12_WRITEBUFFERIMMEDIATE_MODE,
) {
    let Ok(list) = list.cast::<ID3D12GraphicsCommandList2>() else {
        return;
    };
    // SAFETY: our buffer, in the copy-destination state read-back buffers
    // keep; an open list.
    unsafe {
        let param = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: m.buffer.GetGPUVirtualAddress() + 4 * dword,
            Value: value,
        };
        list.WriteBufferImmediate(1, &param, Some(&mode));
    }
}

/// The text noted for dispatch `n`.
fn text(n: u32) -> String {
    let texts = TEXTS.lock().unwrap_or_else(|e| e.into_inner());
    match texts.1.binary_search_by_key(&n, |(k, _)| *k) {
        Ok(i) => format!("#{n} {}", texts.1[i].1),
        Err(_) if n == 0 => "none yet".to_owned(),
        Err(_) => format!("#{n} (text no longer kept)"),
    }
}

/// Where the marks are: the dispatch running and, if it differs, the last
/// one started.
fn describe((done, started): (u32, u32)) -> String {
    match done == started {
        true => format!("running {}", text(done)),
        false => format!(
            "everything before {} done; last started {}",
            text(done),
            text(started)
        ),
    }
}

/// On the completion thread: waits until `fence` reaches `value`. Past
/// [`LATE`], logs where the marks are, each time they move on (with how
/// long they stayed), and every 5 s while they stay.
pub(crate) fn wait(fence: &ID3D12Fence, value: u64) {
    if on() {
        watch(fence, value);
    }
}

/// [`wait`], marks or not: the lines it logged.
fn watch(fence: &ID3D12Fence, value: u64) -> usize {
    let Some(m) = marks() else {
        return 0;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return 0;
    };
    let started = Instant::now();
    let mut at: Option<((u32, u32), Instant)> = None;
    let mut told = started;
    let mut lines = 0;
    // SAFETY: our fence and event.
    if unsafe { fence.SetEventOnCompletion(value, event) }.is_ok() {
        loop {
            let ms = LATE
                .saturating_sub(started.elapsed())
                .max(Duration::from_millis(100));
            // SAFETY: our event.
            if unsafe { WaitForSingleObject(event, ms.as_millis() as u32) } == WAIT_OBJECT_0 {
                break;
            }
            let now = m.read();
            let late = started.elapsed().as_secs_f64();
            match at {
                Some((was, since)) if was == now => {
                    if told.elapsed() >= Duration::from_secs(5) {
                        told = Instant::now();
                        let secs = since.elapsed().as_secs_f64();
                        tracing::warn!(
                            "GPU progress: late {late:.1} s, still {} after {secs:.1} s",
                            describe(now)
                        );
                        lines += 1;
                    }
                }
                _ => {
                    if let Some((was, since)) = at {
                        let secs = since.elapsed().as_secs_f64();
                        tracing::warn!("GPU progress: {} for {secs:.1} s", describe(was));
                        lines += 1;
                    }
                    tracing::warn!("GPU progress: late {late:.1} s, {}", describe(now));
                    lines += 1;
                    at = Some((now, Instant::now()));
                    told = Instant::now();
                }
            }
        }
    }
    if let Some((was, since)) = at {
        let secs = since.elapsed().as_secs_f64();
        let late = started.elapsed().as_secs_f64();
        tracing::warn!(
            "GPU progress: {} for {secs:.1} s; the submission completed after {late:.1} s",
            describe(was)
        );
        lines += 1;
    }
    // SAFETY: our event.
    unsafe {
        let _ = CloseHandle(event);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The marks land in the read-back memory the CPU reads: the one that
    /// waits for earlier work, and the one written once a dispatch started.
    #[test]
    fn marks_land_where_the_cpu_reads_them() {
        let (Some(host), Some(m)) = (crate::host::host(), marks()) else {
            return;
        };
        // SAFETY: plain object creation, recording and a wait on our own
        // queue and fence.
        unsafe {
            let device = &host.device;
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            };
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&desc).unwrap();
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            write(&list, m, 0, 7, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
            write(&list, m, 1, 9, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, HANDLE::default()).unwrap();
        }
        assert_eq!(m.read(), (7, 9));
    }

    /// A submission past [`LATE`] is logged where the marks are, and again
    /// when it completes.
    #[test]
    fn late_submissions_are_logged_until_they_complete() {
        let Some(host) = crate::host::host() else {
            return;
        };
        // SAFETY: plain object creation.
        let fence: ID3D12Fence =
            unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let lines = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(LATE + Duration::from_millis(300));
                // SAFETY: our fence.
                unsafe { fence.Signal(1) }.unwrap();
            });
            watch(&fence, 1)
        });
        assert_eq!(lines, 2);
    }

    /// A mark names the dispatch noted under its number.
    #[test]
    fn marks_name_their_dispatches() {
        let n = {
            let mut texts = TEXTS.lock().unwrap();
            texts.0 += 1;
            let n = texts.0;
            texts.1.push_back((
                n,
                "frame 1 dispatch 2x1x1 [compute 0123456789abcdef]".into(),
            ));
            n
        };
        assert_eq!(
            describe((n, n)),
            format!("running #{n} frame 1 dispatch 2x1x1 [compute 0123456789abcdef]")
        );
        assert_eq!(text(0), "none yet");
    }
}
