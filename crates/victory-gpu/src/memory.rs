//! Title memory at raw GPU addresses (decision 0005, the unified-memory
//! item; the contract is `victory_shader::emit`'s "Title memory").
//! Translated shaders that read or write memory at addresses they compute
//! look each address up in a page table: per 64 KiB page below 2^40, the
//! heap index of a pair of raw views (SRV, UAV) of the host copy holding
//! the page (`crate::shadow`) and the low dword of that copy's title
//! address. The copies are the ones draws and dispatches use through views,
//! so what one shader writes through a view another reads at its address,
//! within a submission too.
//!
//! Per list, at its first dispatch or draw with such a shader in a view
//! heap: every host copy made so far gets its pair of views in dynamic
//! slots of that heap (held until the list is reset), the page table goes
//! into the list's upload memory, and every copy is brought up to date at
//! each submission (page by page, as every refresh). Shaders that write
//! mark the pages they write in the list's bitmap; after the submission
//! each marked page goes back into title memory
//! ([`crate::shadow::Shadow::copy_back_page`]) and the bitmap is cleared.
//!
//! Cost per submission with such a shader: one write-watch check of every
//! host copy (the refresh), copies of the pages the title wrote, and a
//! compare of each 64 KiB page shaders wrote; per list, two views per copy
//! and a 16 KiB + 32 KiB-per-256-MiB-block table.
//!
//! shortcut: memory no copy holds when the table is made (or that is only
//! made later in the list's recording) reads 0 and takes no writes; a page
//! two copies share without either holding all of it is the larger one's.
//! Upgrade when a world run shows raw addresses outside every copy.

use std::sync::Arc;

use crate::host::HostList;
use crate::shadow::Shadow;

/// Pages per level-2 block: 4096 pages of 64 KiB, 256 MiB.
const BLOCK: usize = 4096;
/// Level-1 entries: 2^40 bytes in 256 MiB blocks.
const LEVEL1: usize = 4096;

/// A list's page table.
pub(crate) struct Map {
    /// The page table's host GPU address (the root SRV).
    pub(crate) table: u64,
    /// The written-pages bitmap's host GPU address (the root UAV).
    pub(crate) dirty: u64,
}

/// Our view heap for lists that bind none of the title's (dynamic slots
/// only).
pub(crate) fn heap() -> Option<&'static crate::descriptor::host::HostHeap> {
    static HEAP: std::sync::OnceLock<Option<crate::descriptor::host::HostHeap>> =
        std::sync::OnceLock::new();
    HEAP.get_or_init(|| crate::descriptor::host::HostHeap::new(false, 0))
        .as_ref()
}

/// The page table for host copies of title memory `copies` (start,
/// bytes), whose view pairs start at heap index `first + 2` (copy k at
/// `first + 2 + 2k`; `first` is the null pair): the table's dwords, the
/// level-1 index of each block, and the copy each entry names.
fn table(copies: &[(u64, u64)], first: u32) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let mut order: Vec<usize> = (0..copies.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
    let mut level1 = vec![0u32; LEVEL1];
    // Block 0 is the null block (no copy).
    let mut blocks = vec![usize::MAX];
    let mut owner: Vec<Option<(usize, bool)>> = vec![None; BLOCK];
    for &k in &order {
        let (start, len) = copies[k];
        if len == 0 || start + len > 1 << 40 {
            continue;
        }
        for page in start >> 16..=(start + len - 1) >> 16 {
            let l1 = (page >> 12) as usize;
            if level1[l1] == 0 {
                level1[l1] = blocks.len() as u32;
                blocks.push(l1);
                owner.resize(owner.len() + BLOCK, None);
            }
            let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
            // Larger copies come first: a copy holding the whole page takes
            // it from them; one holding part of it only takes it if free.
            let whole = page << 16 >= start && (page + 1) << 16 <= start + len;
            if whole || owner[entry].is_none() {
                owner[entry] = Some((k, whole));
            }
        }
    }
    let mut dwords = level1;
    for o in &owner {
        let entry = match *o {
            Some((k, _)) => [first + 2 + 2 * k as u32, copies[k].0 as u32],
            None => [first, 0],
        };
        dwords.extend(entry);
    }
    (dwords, blocks, owner.into_iter().map(|o| o.map(|o| o.0)).collect())
}

/// The page table of `list` in view heap `heap` (a host heap): views of
/// every host copy in dynamic slots, which `slots` gets to give back.
pub(crate) fn build(
    list: &HostList,
    heap: &crate::descriptor::host::HostHeap,
    slots: &mut Vec<u32>,
) -> Option<Map> {
    let copies: Vec<Arc<Shadow>> = crate::shadow::all();
    // A raw view holds at most 2^27 dwords: a longer copy (a slab of the
    // mirror) is viewed in pieces of at most that (copy, offset, bytes).
    const VIEW: u64 = 4 << 27;
    let pieces: Vec<(usize, u64, u64)> = copies
        .iter()
        .enumerate()
        .flat_map(|(k, s)| {
            let len = s.range().1;
            (0..len.div_ceil(VIEW).max(1)).map(move |p| (k, p * VIEW, (len - p * VIEW).min(VIEW)))
        })
        .collect();
    let ranges: Vec<(u64, u64)> = pieces
        .iter()
        .map(|&(k, offset, len)| (copies[k].range().0 + offset, len))
        .collect();
    let n = 2 * (pieces.len() as u32 + 1);
    let first = heap.dynamic(n)?;
    slots.extend([first, n]);
    heap.raw_pair_at(first, None, 0, 0);
    for (p, &(k, offset, len)) in pieces.iter().enumerate() {
        let len = len.next_multiple_of(4);
        heap.raw_pair_at(first + 2 + 2 * p as u32, Some(&copies[k].buffer), offset, len);
    }
    let (dwords, blocks, owner) = table(&ranges, first);
    let (table, at) = list.scratch(dwords.len() as u64 * 4)?;
    // SAFETY: list upload memory of that size.
    unsafe { std::ptr::copy_nonoverlapping(dwords.as_ptr(), at.cast::<u32>(), dwords.len()) };
    let bits = owner.len();
    let (bitmap, data) = crate::host::host()?.shared_buffer((bits / 8).next_multiple_of(256) as u64)?;
    crate::host::name(&bitmap, || "title memory written pages".into());
    let words = bits / 32;
    // SAFETY: our mapped buffer of at least `words` dwords.
    unsafe { std::ptr::write_bytes(data.cast::<u32>(), 0, words) };
    // SAFETY: a live buffer.
    let dirty = unsafe { bitmap.GetGPUVirtualAddress() };
    list.keep(&bitmap);
    // Every copy current at each submission.
    for s in &copies {
        let (start, len) = s.range();
        s.refresh(list, start, len);
    }
    let data = data as usize;
    tracing::debug!(copies = ranges.len(), blocks = blocks.len() - 1, "title memory page table");
    list.defer_after(move || {
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks = unsafe { std::slice::from_raw_parts_mut(data as *mut u32, words) };
        for (w, mark) in marks.iter_mut().enumerate() {
            let mut bits = std::mem::take(mark);
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                if let (Some(&l1), Some(Some(k))) = (blocks.get(block), owner.get(entry))
                    && block != 0
                {
                    copies[pieces[*k].0].copy_back_page(l1 << 28 | page << 16);
                }
            }
        }
    });
    Some(Map { table, dirty })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's lookup (`victory_shader` lowering) over a table's
    /// dwords: the view index and the byte offset in that view.
    fn look_up(dwords: &[u32], address: u64) -> (u32, u32) {
        let (lo, hi) = (address as u32, (address >> 32) as u32);
        let inside = hi < 256;
        let l1 = if inside { hi << 4 | lo >> 28 } else { 0 };
        let block = if inside { dwords[l1 as usize] } else { 0 };
        let entry = block << 12 | (lo >> 16 & 0xFFF);
        let at = 4096 + 2 * entry as usize;
        (dwords[at], lo.wrapping_sub(dwords[at + 1]))
    }

    #[test]
    fn addresses_find_the_copy_holding_them() {
        // A 4 MiB window, a buffer inside it from 0x40_0000+0x1_8000
        // (pages in part), and a buffer above 4 GiB.
        let copies = [
            (0x40_0000, 0x40_0000),
            (0x41_8000, 0x3_0000),
            (0xd_1234_0000, 0x2_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, 100);
        // Window, whole pages of the inner buffer, the high buffer.
        assert_eq!(look_up(&dwords, 0x40_0010), (102, 0x10));
        assert_eq!(look_up(&dwords, 0x42_0004), (104, 0x8004));
        assert_eq!(look_up(&dwords, 0xd_1235_fffc), (106, 0x1_fffc));
        // Pages the inner buffer holds in part stay the window's.
        assert_eq!(look_up(&dwords, 0x41_8000), (102, 0x1_8000));
        assert_eq!(look_up(&dwords, 0x44_0000), (102, 0x4_0000));
        // No copy, and above 2^40: the null pair, which reads nothing.
        assert_eq!(look_up(&dwords, 0x9000_0000).0, 100);
        assert_eq!(look_up(&dwords, 0x200_0000_0000).0, 100);
        // Entries map back to their copy and page.
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK
            + (a >> 16 & 0xFFF) as usize;
        assert_eq!(owner[entry(0x42_0000)], Some(1));
        assert_eq!(owner[entry(0xd_1234_0000)], Some(2));
        assert_eq!(blocks.len(), 3, "null, low 256 MiB, the high buffer's");
    }
}
