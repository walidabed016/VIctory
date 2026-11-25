//! GPU probes of XboxBvhBuilder's FindTreelets (compute ffd93c8a029ff016)
//! and ConstructBVH4 (compute 5552dac6a2b7a1fd), the passes before and at
//! run 141's and 144's hang, over random trees of the sizes the title
//! builds. They run only when `VICTORY_TREELETS` and `VICTORY_REFIT` name
//! the containers (cut from the title's XboxBvhBuilder.dll at 0xae100 and
//! 0x53fd0), on the host GPU (WARP with `VICTORY_PROBE_WARP`). With
//! `VICTORY_COHERENT` the results must be exact. Without it they report
//! what stale reads did, and run only under a `VICTORY_SHADER_LOOP_LIMIT`
//! of at most 1024: ConstructBVH4 then climbs broken trees until the limit.

use super::host::{Dispatch, Host, Root, View};
use crate::native::RootType;

const TITLE: u64 = 0x12_3450_0000;
const PAGE: usize = 1 << 16;
const DW: usize = PAGE / 4;
const SENTINEL: f32 = 5.0e5;
/// Internal bounds before a pass writes them: a box no right answer holds,
/// so a read of a node's bounds from before its lane wrote them shows.
const UNSET: [f32; 6] = [
    -SENTINEL, -SENTINEL, -SENTINEL, SENTINEL, SENTINEL, SENTINEL,
];

/// A random binary tree over n leaves as the builder lays it out: internal
/// nodes from the root 0 (a child numbered above its parent), leaf j at
/// n - 1 + j, the root its own parent.
struct Tree {
    parent: Vec<u32>,
    children: Vec<[u32; 2]>,
}

fn tree(n: usize, seed: u64) -> Tree {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as usize
    };
    let mut parent = vec![0u32; 2 * n - 1];
    let mut children = vec![[0u32; 2]; n - 1];
    let mut next = 1u32;
    let mut stack = vec![(0usize, n, 0u32)];
    while let Some((lo, hi, id)) = stack.pop() {
        let len = hi - lo;
        let split = (lo + len / 4 + rng() % (len / 2 + 1)).clamp(lo + 1, hi - 1);
        let mut pair = [0u32; 2];
        for (k, (a, b)) in [(lo, split), (split, hi)].into_iter().enumerate() {
            let child = if b - a == 1 {
                (n - 1 + a) as u32
            } else {
                next += 1;
                stack.push((a, b, next - 1));
                next - 1
            };
            parent[child as usize] = id;
            pair[k] = child;
        }
        children[id as usize] = pair;
    }
    assert_eq!(next as usize, n - 1);
    Tree { parent, children }
}

/// The union of two boxes (min x y z, max x y z).
fn union(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    std::array::from_fn(|d| {
        if d < 3 {
            a[d].min(b[d])
        } else {
            a[d].max(b[d])
        }
    })
}

/// The builder's buffers in title memory at TITLE, each on pages of its own.
struct Image {
    memory: Vec<u32>,
    pages: usize,
}

impl Image {
    /// Dword offset of a new buffer of this many dwords.
    fn alloc(&mut self, dwords: usize) -> usize {
        let at = self.pages * DW;
        self.pages += dwords.div_ceil(DW).max(1);
        self.memory.resize(self.pages * DW, 0);
        at
    }
}

fn address(at: usize) -> u64 {
    TITLE + 4 * at as u64
}

/// The page table (`emit` "Title memory") for one copy of `pages` pages
/// at TITLE whose views are at heap index `view`, null views at `null`.
fn page_table(view: u32, null: u32, pages: usize) -> Vec<u32> {
    let mut t = vec![0u32; 4096];
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    for p in 0..pages as u32 {
        let entry = 4096 + 2 * (4096 + ((lo >> 16 & 0xFFF) + p) as usize);
        t[entry..entry + 2].copy_from_slice(&[view, lo]);
    }
    t
}

/// Runs the container that env var `var` names once over `memory`, with
/// root descriptors at the title addresses `slots` and root constants
/// `consts`; the memory after it.
fn dispatch(
    var: &str,
    memory: &[u32],
    slots: &[u64],
    consts: &[u32],
    groups: u32,
) -> Option<Vec<u32>> {
    let container = std::fs::read(std::env::var_os(var)?).expect("container");
    let shader = super::translate(&container).expect("translate");
    let host = Host::new(std::env::var("VICTORY_PROBE_WARP").is_ok()).ok()?;
    let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
    let mut ud = vec![0u32; shader.user_data_dwords() as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Descriptor => (slots[u.slot as usize] >> (32 * u.dword)) as u32,
            RootType::Constants => consts.get(u.dword as usize).copied().unwrap_or(0),
            _ => 0,
        };
    }
    assert!(shader.memory && shader.memory_writes);
    assert!(!shader.view_addresses && !shader.view_strides);
    let at = 1 + (shader.root_cbvs.len() + shader.root_srvs.len() + shader.root_uavs.len()) as u32;
    let roots = vec![
        (0, Root::Constants(ud)),
        (
            at,
            Root::View {
                srv: true,
                data: page_table(0, 2, memory.len() / DW),
            },
        ),
        (at + 1, Root::Uav(2 * 4096 / 32)),
    ];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots,
                views: &[View::Memory(memory), View::Pair, View::NullPair, View::Pair],
                samplers: &[],
                groups: [groups, 1, 1],
            },
        )
        .expect("dispatch");
    out.into_iter().next()
}

/// Whether the probes may run: without coherent memory, only under a loop
/// limit that ends ConstructBVH4's climbs of a broken tree in milliseconds.
fn may_run() -> bool {
    let ok = super::coherent_memory() || super::loop_limit() <= 1024;
    if !ok {
        eprintln!(
            "without VICTORY_COHERENT, set VICTORY_SHADER_LOOP_LIMIT to at most 1024: skipped"
        );
    }
    ok
}

/// What is wrong with the tree in `parents` and `children` from the root
/// 0: nodes reached twice, nodes never reached, children whose parent is
/// not the node holding them.
fn topology(n: usize, parents: &[u32], children: &[u32]) -> (usize, usize, usize) {
    let nodes = 2 * n - 1;
    let mut seen = vec![0u32; nodes];
    let (mut twice, mut disagree) = (0, 0);
    let mut stack = vec![0u32];
    while let Some(x) = stack.pop() {
        let x = x as usize;
        if x >= nodes || seen[x] > 0 {
            twice += 1;
            continue;
        }
        seen[x] = 1;
        if x < n - 1 {
            for &c in &children[2 * x..2 * x + 2] {
                if (c as usize) < nodes && parents[c as usize] as usize != x {
                    disagree += 1;
                }
                stack.push(c);
            }
        }
    }
    (twice, seen.iter().filter(|&&s| s == 0).count(), disagree)
}

/// Where a run keeps the builder's buffers (dword offsets).
struct Layout {
    out: usize,
    counters: usize,
    parents: usize,
    children: usize,
    node_counter: usize,
    bounds: usize,
    index: usize,
}

/// The buffers for tree `t` of n leaves: leaf j's box spans x 1000 + j to
/// 1001 + j and y and z 0 to 1; internal bounds are the unions.
fn image(n: usize, t: &Tree) -> (Image, Layout, Vec<[f32; 6]>) {
    let nodes = 2 * n - 1;
    let mut im = Image {
        memory: Vec::new(),
        pages: 0,
    };
    // The output's header, then its parent info table at +0x40, then the
    // leaf and BVH4 nodes from header dword 13 on.
    let data_base = (0x40 + 4 * nodes).next_multiple_of(0x1000);
    let l = Layout {
        out: im.alloc((data_base + 256 * n) / 4),
        counters: im.alloc(n),
        parents: im.alloc(nodes),
        children: im.alloc(2 * nodes),
        node_counter: im.alloc(64),
        bounds: im.alloc(6 * nodes),
        index: im.alloc(n),
    };
    let m = &mut im.memory;
    m[l.out + 3] = n as u32;
    m[l.out + 13] = data_base as u32;
    m[l.parents..l.parents + nodes].copy_from_slice(&t.parent);
    for (k, c) in t.children.iter().enumerate() {
        m[l.children + 2 * k..l.children + 2 * k + 2].copy_from_slice(c);
    }
    let mut boxes = vec![[0f32; 6]; nodes];
    for j in 0..n {
        boxes[n - 1 + j] = [1000.0 + j as f32, 0.0, 0.0, 1001.0 + j as f32, 1.0, 1.0];
        m[l.index + j] = j as u32;
    }
    for x in (0..n - 1).rev() {
        let [a, b] = t.children[x];
        boxes[x] = union(boxes[a as usize], boxes[b as usize]);
    }
    for (x, b) in boxes.iter().enumerate() {
        for d in 0..6 {
            m[l.bounds + 6 * x + d] = b[d].to_bits();
        }
    }
    (im, l, boxes)
}

/// ConstructBVH4 over the tree in `im`: its internal bounds wrong and still
/// unset (counting the root), the root's, and its counters' largest value.
fn construct(
    n: usize,
    im: &mut Image,
    l: &Layout,
    expect: &[[f32; 6]],
) -> Option<(usize, usize, [f32; 6], u32)> {
    let m = &mut im.memory;
    m[l.counters..l.counters + n].fill(0);
    for x in 0..n - 1 {
        for d in 0..6 {
            m[l.bounds + 6 * x + d] = UNSET[d].to_bits();
        }
    }
    let slots = [
        address(l.out),
        address(l.counters),
        address(l.parents),
        address(l.children),
        address(l.out) + 0x40,
        address(l.node_counter),
        address(l.bounds),
        address(l.index),
    ];
    let r = dispatch(
        "VICTORY_REFIT",
        &im.memory,
        &slots,
        &[0, 0, 0],
        n.div_ceil(32) as u32,
    )?;
    let got =
        |x: usize| -> [f32; 6] { std::array::from_fn(|d| f32::from_bits(r[l.bounds + 6 * x + d])) };
    let unset = (0..n - 1).filter(|&x| got(x)[0] == -SENTINEL).count();
    let wrong = (0..n - 1).filter(|&x| got(x) != expect[x]).count() - unset;
    let most = *r[l.counters..l.counters + n - 1].iter().max().unwrap();
    Some((wrong, unset, got(0), most))
}

#[test]
fn construct_bvh4_bounds_at_scale() {
    if !may_run() {
        return;
    }
    for (n, seed) in [(4044, 1), (4044, 2), (16384, 3)] {
        let t = tree(n, seed);
        let (mut im, l, expect) = image(n, &t);
        let Some((wrong, unset, root, most)) = construct(n, &mut im, &l, &expect) else {
            eprintln!("VICTORY_REFIT unset or no GPU: skipped");
            return;
        };
        eprintln!(
            "{n} leaves: internal bounds wrong {wrong}, unset {unset} of {}; root {root:?}; counters at most {most}",
            n - 1
        );
        if super::coherent_memory() {
            assert_eq!((wrong, unset, root), (0, 0, expect[0]));
        }
    }
}

#[test]
fn find_treelets_then_construct_bvh4() {
    if !may_run() {
        return;
    }
    // (leaves, seed, minLeavesPerTreelet, walkMaxLevels)
    for (n, seed, min, walk) in [(4044, 1, 7, 32), (4044, 2, 16, 64), (16384, 3, 7, 32)] {
        let t = tree(n, seed);
        let (mut im, l, mut boxes) = image(n, &t);
        let nodes = 2 * n - 1;
        let slots = [
            address(l.parents),
            address(l.children),
            address(l.counters),
            address(l.bounds),
            0,
            0,
        ];
        let header = address(l.out);
        // partitionId -1: the leaf count is header dword 3.
        let consts = [header as u32, (header >> 32) as u32, min, walk, u32::MAX];
        let Some(r) = dispatch(
            "VICTORY_TREELETS",
            &im.memory,
            &slots,
            &consts,
            n.div_ceil(32) as u32,
        ) else {
            eprintln!("VICTORY_TREELETS unset or no GPU: skipped");
            return;
        };
        im.memory = r;
        let m = &im.memory;
        let children = &m[l.children..l.children + 2 * nodes];
        let (twice, never, disagree) = topology(n, &m[l.parents..l.parents + nodes], children);
        let moved = (0..n - 1)
            .filter(|&x| children[2 * x..2 * x + 2] != t.children[x])
            .count();
        eprintln!(
            "{n} leaves, treelets of {min}+ leaves, {walk} levels: FindTreelets rewired {moved} nodes; reached twice {twice}, never {never}, parents disagreeing {disagree}"
        );
        let sound = (twice, never, disagree) == (0, 0, 0);
        if sound {
            let mut order = Vec::new();
            let mut stack = vec![0usize];
            while let Some(x) = stack.pop() {
                if x < n - 1 {
                    order.push(x);
                    stack.extend(children[2 * x..2 * x + 2].iter().map(|&c| c as usize));
                }
            }
            for &x in order.iter().rev() {
                boxes[x] = union(
                    boxes[children[2 * x] as usize],
                    boxes[children[2 * x + 1] as usize],
                );
            }
        }
        let Some((wrong, unset, root, most)) = construct(n, &mut im, &l, &boxes) else {
            return;
        };
        eprintln!(
            "  then ConstructBVH4: internal bounds wrong {wrong}, unset {unset} of {}; root {root:?}; counters at most {most}",
            n - 1
        );
        if super::coherent_memory() {
            assert!(sound && moved > 0);
            assert_eq!((wrong, unset, root, most), (0, 0, boxes[0], 4));
        }
    }
}

#[test]
fn prefix_sum64_scans_each_group_s_64_values() {
    // PrefixSum64 (cut from XboxBvhBuilder.dll at 0xa1080, named by
    // `VICTORY_PREFIX_SUM`): each group of 32 lanes takes 64 values, two a
    // lane, and leaves their exclusive scan in place and their sum at the
    // group's dword of the second buffer (root constant bit 0).
    if !super::mask_numbers() {
        eprintln!("without VICTORY_MASK_NUMBERS PrefixSum64 is refused: skipped");
        return;
    }
    let groups = 64;
    let mut im = Image {
        memory: Vec::new(),
        pages: 0,
    };
    let values = im.alloc(64 * groups);
    let sums = im.alloc(groups);
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let input: Vec<u32> = (0..64 * groups)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1000) as u32
        })
        .collect();
    im.memory[values..values + input.len()].copy_from_slice(&input);
    let slots = [address(values), address(sums)];
    let Some(r) = dispatch("VICTORY_PREFIX_SUM", &im.memory, &slots, &[1], groups as u32) else {
        eprintln!("VICTORY_PREFIX_SUM unset or no GPU: skipped");
        return;
    };
    for g in 0..groups {
        let mut total = 0;
        for k in 0..64 {
            assert_eq!(r[values + 64 * g + k], total, "group {g}, value {k}");
            total += input[64 * g + k];
        }
        assert_eq!(r[sums + g], total, "group {g}'s sum");
    }
}

#[test]
fn copy_for_bvh_update_copies_a_tree_once_and_ends() {
    // CopyForBVHUpdate (cut from XboxBvhBuilder.dll at 0xb4fe0, named by
    // `VICTORY_BVH_COPY`): each lane copies 16 bytes, 16 * (root constant 0)
    // bytes apart, until 64 * (the source's dword 1) + 64 bytes; with root
    // constant 1 set, lane 0 then resets the bounds at bytes 16 to 40. Run
    // 158 timed out in its dispatches of 128 groups (4096 threads) whose
    // source and destination were one address.
    let n = 5000;
    let dwords = 16 * (n + 1);
    let mut im = Image {
        memory: Vec::new(),
        pages: 0,
    };
    let src = im.alloc(dwords);
    let dst = im.alloc(dwords + DW);
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    for k in 0..dwords {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        im.memory[src + k] = s as u32;
    }
    im.memory[src + 1] = n as u32;
    let empty = [
        0x7f7f_ffffu32,
        0x7f7f_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0xff7f_ffff,
        0xff7f_ffff,
    ];
    for (consts, to) in [([4096, 0], dst), ([4096, 1], dst), ([4096, 0], src)] {
        let slots = [address(src), address(to)];
        let start = std::time::Instant::now();
        let Some(r) = dispatch("VICTORY_BVH_COPY", &im.memory, &slots, &consts, 128) else {
            eprintln!("VICTORY_BVH_COPY unset or no GPU: skipped");
            return;
        };
        eprintln!(
            "constants {consts:?}, in place {}: {:?}",
            to == src,
            start.elapsed()
        );
        for k in 0..dwords {
            let want = match k {
                4..10 if consts[1] != 0 => empty[k - 4],
                _ => im.memory[src + k],
            };
            assert_eq!(r[to + k], want, "dword {k}");
        }
        if to == dst {
            assert!(r[dst + dwords..].iter().all(|&d| d == 0), "past the tree");
        }
    }
}
