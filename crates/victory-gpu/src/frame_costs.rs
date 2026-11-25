//! Benchmark of the host's costs of a world frame (diagnostics builds, by
//! hand): title memory of the kinds and sizes Grand Theft Auto VI's world
//! frames read, refreshed, written and copied back through the submission
//! path as run 50 measured late in an in-world cutscene. Per frame:
//!
//! - 18 submissions, each with ~150 root views reaching 36 MiB into loose
//!   memory (2,681 a frame, all in the mirror) and ~600 small refreshes of
//!   buffer copies;
//! - the world's 174 MB buffer, three views, refreshed in every submission
//!   and written by shaders in 4 of them;
//! - three title copies by the GPU (`CopyMemoryToMemoryX`) a submission;
//! - title writes between submissions: pages of write-watched pools,
//!   blocks of views, blocks of the world buffer;
//! - 24 busy threads, as the title has: the kernel's TLB flushes reach
//!   every thread of the process.
//!
//! `cargo test --release -p victory-gpu --features diagnostics -- --ignored
//! --nocapture world_frame`

use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile3, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;
const MEM_WRITE_WATCH: u32 = 0x20_0000;
/// Where the benchmark's title memory starts (mirror slab 940 on).
const BASE: usize = 940 * GIB;
const SUBMISSIONS: usize = 18;
const ROOTS: usize = 150;
const REACH: usize = 36 * MIB;

struct Rng(u64);

impl Rng {
    fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// Write-watched memory at `at`, committed and filled: a pool, as xmem
/// reserves them.
fn pool(at: usize, len: usize) -> usize {
    // SAFETY: a reservation at a free address (checked), committed whole.
    let p = unsafe {
        VirtualAlloc(
            at as *const _,
            len,
            MEM_RESERVE | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        ) as usize
    };
    assert_eq!(p, at, "no free address for a pool");
    // SAFETY: our reservation.
    let c = unsafe { VirtualAlloc(p as *const _, len, MEM_COMMIT, PAGE_READWRITE) as usize };
    assert_eq!(c, p);
    crate::watch::title_memory_watched(p, len, true);
    fill(p, len);
    p
}

/// A view of a fresh pagefile section at `at`, filled.
fn view(at: usize, len: usize) -> usize {
    // SAFETY: a section of ours mapped at a free address (checked).
    let v = unsafe {
        let section = CreateFileMappingW(
            -1isize as _,
            std::ptr::null(),
            PAGE_READWRITE,
            (len >> 32) as u32,
            len as u32,
            std::ptr::null(),
        );
        assert!(!section.is_null());
        MapViewOfFile3(
            section,
            GetCurrentProcess(),
            at as *const _,
            0,
            len,
            0,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
        .Value as usize
    };
    assert_eq!(v, at, "no free address for a view");
    fill(v, len);
    v
}

fn fill(at: usize, len: usize) {
    for a in (at..at + len).step_by(4096) {
        // SAFETY: committed memory of ours.
        unsafe { (a as *mut u64).write(a as u64) };
    }
}

fn touch(at: usize, round: usize) {
    // SAFETY: committed test memory (a guarded block opens on the write).
    unsafe { (at as *mut u64).write_volatile(round as u64) };
}

#[test]
#[ignore = "benchmark"]
fn world_frame() {
    use crate::copystat;
    let _serial = crate::watch::serial();
    if !crate::shadow::tests::gpu() {
        return;
    }
    let host = crate::host::host().unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let busy: Vec<_> = (0..24)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut v = vec![0u8; MIB];
                let mut i = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let n = v.len();
                    v[i % n] = v[i % n].wrapping_add(1);
                    i += 4099;
                }
            })
        })
        .collect();
    // Loose memory: two write-watched pools, four big views, 400 views of a
    // console page; then the world buffer's three views and 200 small
    // buffers in the second pool.
    let pools = [
        (pool(BASE, GIB), GIB),
        (pool(BASE + 5 * GIB / 4, GIB / 2), GIB / 2),
    ];
    let views: Vec<(usize, usize)> = (0..4)
        .map(|k| (view(BASE + 2 * GIB + k * 256 * MIB, 256 * MIB), 256 * MIB))
        .collect();
    let pages: Vec<usize> = (0..400)
        .map(|k| view(BASE + 13 * GIB / 4 + k * 64 * 1024, 64 * 1024))
        .collect();
    let world = BASE + 7 * GIB / 2;
    let mut at = world;
    for len in [0x1b0_0000, 0x2f0_0000, 0x640_0000] {
        view(at, len);
        at += len;
    }
    let world_len = at - world;
    crate::shadow::register(1, world as u64, world_len as u64);
    let small: Vec<usize> = (0..200).map(|k| pools[1].0 + 64 * MIB + k * MIB).collect();
    for (k, &b) in small.iter().enumerate() {
        crate::shadow::register(2 + k, b as u64, 256 << 10);
    }
    // Where root views point: 400 places in loose memory, 256-byte aligned.
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let loose: Vec<(usize, usize)> = pools.iter().chain(&views).copied().collect();
    let places: Vec<usize> = (0..400)
        .map(|_| {
            let (at, len) = loose[rng.next(loose.len())];
            at + rng.next(len - REACH) / 256 * 256
        })
        .collect();
    let (upload, data) = host.upload_buffer(64 << 10).unwrap();
    // SAFETY: our mapped upload buffer.
    unsafe { data.write_bytes(0x5a, 64 << 10) };
    let lists: Vec<_> = (0..SUBMISSIONS).map(|_| host.new_list().unwrap()).collect();
    let frame = |round: usize, rng: &mut Rng| {
        let started = std::time::Instant::now();
        for (s, list) in lists.iter().enumerate() {
            for _ in 0..ROOTS {
                let at = places[rng.next(places.len())] as u64;
                let (shadow, len) = crate::shadow::root(at, REACH as u64).unwrap();
                shadow.refresh(list, at, len);
            }
            for _ in 0..600 {
                let b = small[rng.next(small.len())] + rng.next(16) * 16 * 1024;
                let shadow = crate::shadow::find(b as u64, 16 * 1024).unwrap();
                shadow.refresh(list, b as u64, 16 * 1024);
            }
            let copy = crate::shadow::find(world as u64, world_len as u64).unwrap();
            copy.refresh(list, world as u64, world_len as u64);
            if s % 5 == 0 {
                copy.gpu_writes(list, world as u64, world_len as u64);
                for k in 0..4 {
                    let offset = (rng.next(world_len / (64 << 10)) * (64 << 10)) as u64;
                    let _ = k;
                    // SAFETY: an open list; live buffers, the range inside both.
                    unsafe {
                        list.list
                            .CopyBufferRegion(copy.resource(), offset, &upload, 0, 64 << 10)
                    };
                }
            }
            for _ in 0..3 {
                let (from, to) = (
                    pools[0].0 + rng.next(GIB / 4096) * 4096,
                    pools[0].0 + rng.next(GIB / 4096) * 4096,
                );
                list.defer(move || {
                    // SAFETY: 4 KiB of committed test memory each.
                    unsafe { std::ptr::copy(from as *const u8, to as *mut u8, 4096) };
                });
            }
            list.close();
            host.execute(&[list]);
            list.reset();
            // The title writes between submissions.
            for _ in 0..40 {
                let (at, len) = pools[rng.next(2)];
                touch(at + rng.next(len / 4096) * 4096, round);
            }
            for _ in 0..16 {
                let (at, len) = views[rng.next(views.len())];
                touch(at + rng.next(len / 65536) * 65536, round);
            }
            for _ in 0..4 {
                touch(pages[rng.next(pages.len())], round);
            }
            for _ in 0..10 {
                touch(world + rng.next(world_len / 65536) * 65536, round);
            }
        }
        started.elapsed()
    };
    for round in 0..2 {
        let took = frame(round, &mut rng);
        println!("warm-up frame {round}: {took:.2?}");
    }
    copystat::lines(1.0);
    let twins = || {
        crate::host::TWINS
            .each_ref()
            .map(|n| n.swap(0, std::sync::atomic::Ordering::Relaxed) as f64)
    };
    twins();
    const FRAMES: usize = 4;
    let mut took = std::time::Duration::ZERO;
    for round in 0..FRAMES {
        took += frame(2 + round, &mut rng);
    }
    println!(
        "world frame, host side: {:.1} ms",
        took.as_secs_f64() * 1e3 / FRAMES as f64
    );
    let [us, up, compared, down, skipped, whole, given] = twins().map(|n| n / FRAMES as f64);
    println!(
        "VRAM twins per frame: {:.1} ms recording; {:.1} MB copied up ({:.1} MB of windows \
         given VRAM, whole; {given:.1} windows gave theirs back), {:.1} MB compared on the \
         GPU, {:.1} MB copied down whole; copy-back skipped {:.1} MB unchanged",
        us / 1e3,
        up / 1e6,
        whole / 1e6,
        compared / 1e6,
        down / 1e6,
        skipped / 1e6
    );
    for line in copystat::lines(FRAMES as f64) {
        println!("{line}");
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for t in busy {
        t.join().unwrap();
    }
    crate::shadow::unregister(1, world as u64);
    for (k, &b) in small.iter().enumerate() {
        crate::shadow::unregister(2 + k, b as u64);
    }
    for (at, len) in pools {
        // SAFETY: our reservations.
        crate::watch::title_memory_unmapping(at, len, || unsafe {
            VirtualFree(at as *mut _, 0, MEM_RELEASE)
        });
        crate::watch::title_memory_watched(at, len, false);
    }
    let mut maps: Vec<(usize, usize)> = views.clone();
    maps.extend(pages.iter().map(|&p| (p, 64 * 1024)));
    let mut at = world;
    for len in [0x1b0_0000, 0x2f0_0000, 0x640_0000] {
        maps.push((at, len));
        at += len;
    }
    for (at, len) in maps {
        // SAFETY: our views.
        crate::watch::title_memory_unmapping(at, len, || unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: at as *mut _,
            })
        });
    }
}
