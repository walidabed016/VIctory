//! 64-lane console waves on 32-lane host waves (added 2026-11-24).
//!
//! A wave64 compute program whose lanes interact declares `WaveSize(64)`
//! so that host lane n is console lane n; NVIDIA and Intel GPUs run only
//! 32-lane waves and refuse such a pipeline. Where the translation can show
//! that the result is the console's, it declares `WaveSize(32)` instead and
//! each console wave runs as two host waves, its halves: lanes 0-31 and
//! 32-63.
//!
//! * **Lanes.** Host waves of a 2-D group need not hold consecutive thread
//!   ids (the RTX 4090 packs 2x2 quads: a wave of a 32x2 group holds
//!   threads 0-15 and 32-47). So each host wave takes the next 32 console
//!   threads from a group-shared counter and its lanes play them in order:
//!   console thread = block + host lane, console lane = thread % 64, half =
//!   bit 5, console wave = thread / 64. Only groups of whole host waves (a
//!   multiple of 32 threads) can.
//! * **Within a half.** `ds_swizzle`, DPP and the lane-index arithmetic of
//!   `v_mbcnt` stay inside 32 lanes on the console too, so a host wave
//!   reads its own lanes (`WaveReadLaneAt(x, l % 32)`).
//! * **Across halves.** `v_readlane`, `v_readfirstlane`, lane masks read
//!   as data (ballots), and whole-wave reductions and prefixes need the
//!   other half: each half writes its word to group-shared memory (a place
//!   per instruction), a group barrier, and both read both words; another
//!   barrier after the reads where a loop comes back to the instruction. A
//!   barrier is only sound where every thread of the group gets as often,
//!   so these exchanges sit only in [`meeting_blocks`]; anywhere else the
//!   program keeps the console's 64-lane waves, with the reason.
//! * **One lane.** Where one lane of each console wave at most gets
//!   ([`alone_blocks`]: `if (index == 0)`), `v_readfirstlane` reads that
//!   lane's own value, with no exchange.
//! * **Lockstep.** A group of one console wave that the halves cannot run
//!   lane by lane follows that wave's own control flow instead: both halves
//!   take each branch the wave takes (on values every thread has alike),
//!   EXEC and the lane masks are values (the wave's ballots), and each
//!   vector instruction runs under the thread's own EXEC bit, so every
//!   block meets. A whole-wave section's scalar instructions run there too,
//!   so the masks it makes from the saved EXEC are values like any other.
//! * **In turn.** A group of several console waves that never meet (no
//!   LDS, no barrier, nothing in a loop through which one could wait for
//!   another) runs in lockstep too: the host group is one console wave,
//!   whose halves play the group's waves one after the other (console
//!   thread = turn * 64 + its own); the console's `s_endpgm` ends a turn.
//!   Waves that never wait for each other may run in any order on the
//!   console, this one included.
//! * **LDS of a one-wave group.** The console compiler leaves out barriers
//!   when the group is one wave, which its two halves no longer are: every
//!   LDS access gets a barrier before it, so they happen in program order
//!   across the group as in one wave; again only in meeting blocks.
//!
//! The halves keep every scalar value equal (the exchanges make what lane
//! data reaches a scalar the same in both), so uniform branches go the same
//! way in both, as in the console's one wave.

use std::collections::HashSet;

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, Bools, LaneExit, Structured};

/// Per block: whether every thread of the group gets there equally often
/// (in the same order as to the other such blocks), so a barrier there is
/// reached by all of them. Those are the blocks no lane-dependent branch
/// can take some threads around: none lies on a way from such a branch to
/// its nearest post-dominator. Lane-dependent are the per-lane branches
/// and the branches on scalars that differ between threads: those made
/// from `lane_values` (scalars that hold each lane's own value on the
/// host), and unless the group is `one_wave`, those made from other lanes
/// (equal within a console wave only). A loop's iteration limit leaves
/// from its header in the threads that reach it, together where the header
/// meets, else on their own; the end of the program and an indirect jump
/// are exits too. A block holding the console's own `s_barrier` meets as
/// well: its translation already takes every thread of the group to get
/// there equally often and in step with the other barriers (a thread the
/// iteration limit stops leaves that barrier short too).
pub(crate) fn meeting_blocks(
    p: &Program,
    st: &Structured,
    one_wave: bool,
    lane_values: &[Value],
) -> Vec<bool> {
    let n = p.blocks.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        rpo[b] = i;
    }
    let mut varying = spread(p, lane_values.iter().copied().collect(), |_| false);
    if !one_wave {
        let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
        varying = spread(p, varying, |i| {
            matches!(i.name(), "v_readlane_b32" | "v_readfirstlane_b32")
                || matches!(i.fields, Fields::Sop { .. })
                    && i.reads.iter().any(|&(r, v)| {
                        is_mask(v) || matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1)
                    })
        });
    }
    let lane_branch = |b: usize| match st.exits[b] {
        LaneExit::Branch { when, .. } => st.bools.support(when).iter().any(|a| match *a {
            Atom::Uniform(v) => varying.contains(&v),
            _ => true,
        }),
        _ => false,
    };
    // Successors; `n` is the end of the program.
    let mut succ: Vec<Vec<usize>> = (0..n)
        .map(|b| match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => vec![n],
        })
        .collect();
    succ.push(Vec::new());
    let mut splits: Vec<usize> = (0..n)
        .filter(|&b| rpo[b] != usize::MAX && lane_branch(b))
        .collect();
    let Some(first) = diverging(&succ, &order, &splits) else {
        return vec![false; n];
    };
    let rpo = &rpo;
    let headers: HashSet<usize> = order
        .iter()
        .flat_map(|&b| {
            succ[b]
                .iter()
                .copied()
                .filter(move |&t| t < n && rpo[t] <= rpo[b])
        })
        .collect();
    for &h in &headers {
        succ[h].push(n);
        if first[h] {
            splits.push(h);
        }
    }
    let Some(apart) = diverging(&succ, &order, &splits) else {
        return vec![false; n];
    };
    let barrier = |b: usize| p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier");
    (0..n)
        .map(|b| rpo[b] != usize::MAX && (!apart[b] || barrier(b)))
        .collect()
}

/// Per block: whether a loop can come back to it.
pub(crate) fn looping_blocks(p: &Program, st: &Structured) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    (0..n)
        .map(|b| {
            let mut seen = vec![false; n];
            let mut stack = succ(b);
            while let Some(x) = stack.pop() {
                if x == b {
                    return true;
                }
                if !std::mem::replace(&mut seen[x], true) {
                    stack.extend(succ(x));
                }
            }
            false
        })
        .collect()
}

/// Per block: at most one lane of each console wave gets there. Every lane
/// that does passed a compare holding in one lane of a wave at most: the
/// equality of a value the wave shares (a constant, or an SGPR not made
/// from `lane_values`) with one each lane has its own: `distinct` (the
/// thread's x index where a wave's lanes have different ones), plus, minus
/// or xor a shared value, written wherever it is reached. The compare and
/// what makes the lane's value lie outside loops, so a wave's lanes have
/// them from the same pass. A `v_readfirstlane` there in every lane that
/// gets there reads the lane's own value.
pub(crate) fn alone_blocks(
    p: &Program,
    st: &Structured,
    distinct: Option<Value>,
    lane_values: &[Value],
) -> Vec<bool> {
    let n = p.blocks.len();
    let Some(seed) = distinct else {
        return vec![false; n];
    };
    let varying = spread(p, lane_values.iter().copied().collect(), |_| false);
    let shared = |i: &Inst, op: &Operand| match *op {
        Operand::Const(_) | Operand::Literal(_) => true,
        Operand::Scalar { code, .. } if code != NULL => {
            i.read(Reg::S(code)).is_some_and(|v| !varying.contains(&v))
        }
        _ => false,
    };
    let vgpr = |i: &Inst, op: &Operand| match *op {
        Operand::Vector { reg, .. } => i.read(Reg::V(reg)),
        _ => None,
    };
    let looped = looping_blocks(p, st);
    let mut own = HashSet::from([seed]);
    let mut singles = Vec::new();
    loop {
        let before = own.len();
        singles.clear();
        for (b, block) in p.blocks.iter().enumerate().filter(|b| !looped[b.0]) {
            for (k, i) in block.insts.iter().enumerate() {
                let Fields::Valu(v) = &i.fields else {
                    continue;
                };
                let is_own = |op: &Operand| vgpr(i, op).is_some_and(|x| own.contains(&x));
                let plain = v.dpp.is_none()
                    && v.sdwa.is_none()
                    && !v.clamp
                    && v.omod == 0
                    && v.opsel == 0
                    && v.abs == 0
                    && v.neg == 0;
                // One source the lane's own, the others shared.
                let one = plain
                    && v.src.iter().filter(|op| is_own(op)).count() == 1
                    && v.src.iter().all(|op| is_own(op) || shared(i, op));
                if !one {
                    continue;
                }
                let name = i.name();
                if let Some(rest) = name
                    .strip_prefix("v_cmp_eq_")
                    .or(name.strip_prefix("v_cmpx_eq_"))
                {
                    let code = match v.dst {
                        _ if name.starts_with("v_cmpx") => Some(EXEC),
                        Some(Operand::Scalar { code, .. }) => Some(code),
                        _ => None,
                    };
                    if matches!(rest, "u32" | "i32")
                        && let Some(lo) = code.and_then(|c| i.write(Reg::S(c)))
                    {
                        singles.push(Atom::Lane(lo));
                    }
                    continue;
                }
                let fits = match name {
                    "v_mov_b32" | "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32"
                    | "v_xor_b32" | "v_add3_u32" => true,
                    // (s0 << s1) + s2.
                    "v_lshl_add_u32" => v.src.get(2).is_some_and(is_own),
                    _ => false,
                };
                if fits
                    && st.guards[b][k].is_none()
                    && let Some(Operand::Vector { reg, .. }) = v.dst
                    && let Some(w) = i.write(Reg::V(reg))
                {
                    own.insert(w);
                }
            }
        }
        if own.len() == before {
            break;
        }
    }
    (0..n)
        .map(|b| singles.iter().any(|&a| implies(&st.bools, st.reach[b], a)))
        .collect()
}

/// Whether `b` holds only where `atom` does: no way through it to true
/// with `atom` false.
fn implies(bools: &Bools, b: Bool, atom: Atom) -> bool {
    let mut stack = vec![b];
    let mut seen = HashSet::new();
    while let Some(x) = stack.pop() {
        if x == Bool::TRUE {
            return false;
        }
        if x == Bool::FALSE || !seen.insert(x) {
            continue;
        }
        let (a, lo, hi) = bools.node(x).expect("not a constant");
        stack.push(lo);
        if a != atom {
            stack.push(hi);
        }
    }
    true
}

/// `out` and the values computed from it, or written by an instruction
/// `from` picks.
fn spread(p: &Program, mut out: HashSet<Value>, from: impl Fn(&Inst) -> bool) -> HashSet<Value> {
    loop {
        let before = out.len();
        for block in &p.blocks {
            for phi in &block.phis {
                if phi.args.iter().any(|a| out.contains(a)) {
                    out.insert(phi.value);
                }
            }
            for i in &block.insts {
                if from(i) || i.reads.iter().any(|r| out.contains(&r.1)) {
                    out.extend(i.writes.iter().map(|w| w.1));
                }
            }
        }
        if out.len() == before {
            return out;
        }
    }
}

/// Per block of `succ` (the last entry the end): whether it lies on a way
/// from one of `splits` to that block's nearest post-dominator; none when
/// some reachable block cannot get to the end.
fn diverging(succ: &[Vec<usize>], order: &[usize], splits: &[usize]) -> Option<Vec<bool>> {
    let end = succ.len() - 1;
    let words = succ.len().div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = vec![full.clone(); succ.len()];
    pdom[end] = vec![0; words];
    pdom[end][end / 64] |= 1 << (end % 64);
    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            let mut set = full.clone();
            for &s in &succ[b] {
                for (w, x) in set.iter_mut().zip(&pdom[s]) {
                    *w &= x;
                }
            }
            set[b / 64] |= 1 << (b % 64);
            if set != pdom[b] {
                pdom[b] = set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let has = |s: &[u64], x: usize| s[x / 64] >> (x % 64) & 1 != 0;
    let count = |s: &[u64]| s.iter().map(|w| w.count_ones()).sum::<u32>();
    let mut ends = vec![false; succ.len()];
    ends[end] = true;
    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            if !ends[b] && succ[b].iter().any(|&s| ends[s]) {
                ends[b] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if order.iter().any(|&b| !ends[b]) {
        return None;
    }
    let mut apart = vec![false; succ.len()];
    for &d in splits {
        let join = (0..succ.len())
            .filter(|&x| x != d && has(&pdom[d], x))
            .max_by_key(|&x| count(&pdom[x]))
            .expect("the end post-dominates every block");
        let mut stack: Vec<usize> = succ[d].clone();
        let mut seen = vec![false; succ.len()];
        while let Some(x) = stack.pop() {
            if x == join || std::mem::replace(&mut seen[x], true) {
                continue;
            }
            apart[x] = true;
            stack.extend(&succ[x]);
        }
    }
    apart.truncate(end);
    Some(apart)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each block's offset and whether it meets.
    fn meeting(words: &[u32]) -> Vec<(usize, bool)> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 16).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let m = meeting_blocks(&p, &st, true, &[]);
        p.blocks.iter().map(|b| b.offset).zip(m).collect()
    }

    /// The same for a group of several console waves.
    fn meeting_in_waves(words: &[u32]) -> Vec<(usize, bool)> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 16).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let m = meeting_blocks(&p, &st, false, &[]);
        p.blocks.iter().map(|b| b.offset).zip(m).collect()
    }

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// `v_cmpx_eq_u32 exec, 0, v0`.
    const V_CMPX_EQ_U32_0_V0: u32 = 0x7DA4_0080;
    /// `s_cbranch_execz` over one dword.
    const S_CBRANCH_EXECZ_1: u32 = 0xBF88_0001;
    /// `s_cbranch_execnz` back three dwords.
    const S_CBRANCH_EXECNZ_BACK_3: u32 = 0xBF89_FFFD;
    /// `v_mov_b32 v1, 0`.
    const V_MOV_V1_0: u32 = 0x7E02_0280;
    /// `s_mov_b64 s[0:1], exec` and back.
    const S_SAVE: u32 = 0xBE80_047E;
    const S_RESTORE: u32 = 0xBEFE_0400;

    /// `s_mov_b32 s0, 4`.
    const S_MOV_S0_4: u32 = 0xBE80_0384;
    /// `v_readfirstlane_b32 s0, v0`.
    const V_READFIRSTLANE_S0_V0: u32 = 0x7E00_0500;
    /// `s_sub_u32 s0, s0, 1`.
    const S_SUB_S0_1: u32 = 0x8080_8100;
    /// `s_cmp_lg_u32 s0, 0`.
    const S_CMP_LG_S0_0: u32 = 0xBF07_8000;
    /// `s_cbranch_scc1` back four dwords.
    const S_CBRANCH_SCC1_BACK_4: u32 = 0xBF85_FFFC;

    /// Each block's offset and whether one lane of a console wave at most
    /// gets there, v0 holding each lane's own x index.
    fn alone(words: &[u32]) -> Vec<(usize, bool)> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 16).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let x = p.inputs.iter().find(|i| i.0 == Reg::V(0)).map(|i| i.1);
        let a = alone_blocks(&p, &st, x, &[]);
        p.blocks.iter().map(|b| b.offset).zip(a).collect()
    }

    /// `v_lshl_add_u32 v2, s1, 8, v0`, `v_and_b32 v2, 31, v0`.
    const V_LSHL_ADD_V2: [u32; 2] = [0xD746_0002, 0x0401_1001];
    const V_AND_V2_31_V0: u32 = 0x3604_009F;
    /// `v_cmpx_eq_u32 exec, 0, v2`.
    const V_CMPX_EQ_U32_0_V2: u32 = 0x7DA4_0480;

    #[test]
    fn one_lane_of_a_wave_gets_past_an_equality_with_its_own_index() {
        // if (s1 * 256 + x == 0) v1 = 0.
        let a = alone(&[
            V_LSHL_ADD_V2[0],
            V_LSHL_ADD_V2[1],
            V_CMPX_EQ_U32_0_V2,
            S_CBRANCH_EXECZ_1,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert_eq!(a, [(0, false), (16, true), (20, false)]);
    }

    #[test]
    fn lanes_that_share_a_masked_index_are_not_alone() {
        // if ((x & 31) == 0) v1 = 0: lanes 0 and 32 of each wave.
        let a = alone(&[
            V_AND_V2_31_V0,
            V_CMPX_EQ_U32_0_V2,
            S_CBRANCH_EXECZ_1,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert_eq!(a, [(0, false), (12, false), (16, false)]);
    }

    #[test]
    fn an_index_made_in_a_loop_is_not_the_lanes_own() {
        // for (s0 = 4; s0 != 0; s0--) v2 = s0 + x; if (v2 == 0) v1 = 0:
        // lanes leaving in different passes could share v2.
        let a = alone(&[
            S_MOV_S0_4,
            V_ADD_V2_S0_V0,
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            S_CBRANCH_SCC1_BACK_4,
            V_CMPX_EQ_U32_0_V2,
            S_CBRANCH_EXECZ_1,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert!(a.iter().all(|b| !b.1), "{a:?}");
    }

    /// `v_add_nc_u32 v2, s0, v0`.
    const V_ADD_V2_S0_V0: u32 = 0x4A04_0000;
    /// `s_cbranch_execnz` back three dwords; `s_barrier`.
    const S_BARRIER: u32 = 0xBF8A_0000;

    #[test]
    fn a_block_with_the_programs_barrier_meets_after_a_lane_loop() {
        // do v1 = 0 while (x == 0), lane by lane; then v1 = 0 after the
        // program's barrier or without one.
        let lane_loop = [
            S_SAVE,
            V_MOV_V1_0,
            V_CMPX_EQ_U32_0_V0,
            S_CBRANCH_EXECNZ_BACK_3,
            S_RESTORE,
        ];
        let with = [&lane_loop[..], &[S_BARRIER, V_MOV_V1_0, S_ENDPGM]].concat();
        let without = [&lane_loop[..], &[V_MOV_V1_0, S_ENDPGM]].concat();
        assert_eq!(meeting_in_waves(&with), [(0, true), (4, false), (16, true)]);
        assert_eq!(
            meeting_in_waves(&without),
            [(0, true), (4, false), (16, false)]
        );
    }

    #[test]
    fn a_uniform_loop_and_what_follows_it_meet() {
        // for (s0 = 4; s0 != 0; s0--) v1 = 0; v1 = 0.
        let m = meeting(&[
            S_MOV_S0_4,
            V_MOV_V1_0,
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            S_CBRANCH_SCC1_BACK_4,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert_eq!(m, [(0, true), (4, true), (20, true)]);
    }

    #[test]
    fn a_loop_counted_by_one_lane_meets_only_within_one_wave() {
        // for (s0 = first lane's v0; s0 != 0; s0--) v1 = 0; v1 = 0.
        let words = [
            V_READFIRSTLANE_S0_V0,
            V_MOV_V1_0,
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            S_CBRANCH_SCC1_BACK_4,
            V_MOV_V1_0,
            S_ENDPGM,
        ];
        assert_eq!(meeting(&words), [(0, true), (4, true), (20, true)]);
        assert_eq!(
            meeting_in_waves(&words),
            [(0, true), (4, false), (20, false)]
        );
    }

    #[test]
    fn a_skipped_region_does_not_meet_but_its_join_does() {
        // if (v0 == 0) { v1 = 0 }; v1 = 0.
        let m = meeting(&[
            S_SAVE,
            V_CMPX_EQ_U32_0_V0,
            S_CBRANCH_EXECZ_1,
            V_MOV_V1_0,
            S_RESTORE,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert_eq!(m, [(0, true), (12, false), (16, true)]);
    }

    #[test]
    fn lanes_that_leave_early_keep_later_blocks_from_meeting() {
        // if (v0 == 0) end; v1 = 0.
        let m = meeting(&[
            V_CMPX_EQ_U32_0_V0,
            S_CBRANCH_EXECZ_1,
            S_ENDPGM,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert_eq!(m, [(0, true), (8, false), (12, false)]);
    }

    #[test]
    fn loops_and_what_follows_them_do_not_meet() {
        // do { v1 = 0 } while (v0 == 0 in some lane); v1 = 0.
        let m = meeting(&[
            S_SAVE,
            V_MOV_V1_0,
            V_CMPX_EQ_U32_0_V0,
            S_CBRANCH_EXECNZ_BACK_3,
            S_RESTORE,
            V_MOV_V1_0,
            S_ENDPGM,
        ]);
        assert_eq!(m, [(0, true), (4, false), (16, false)]);
    }
}
