//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be: the reduction or
//! prefix of one operation over the lanes' (masked) inputs, which DXIL's
//! wave operations compute over the host's active lanes. No instruction
//! sequence is matched; any step the simulation does not model leaves the
//! values it writes unknown. An f16 sum rounds at every step, so the order
//! of its steps is its result: it is followed only where each step adds two
//! aligned blocks of lanes that make one, as the console's tree does, and
//! the host adds them in that order.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            // A 64-bit sum's low dword is the sum of the low dwords; its
            // carry, and so the high dword, stays unknown.
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f32" => Op::AddF,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF => k == 0 || k == 0x8000_0000,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(self, Op::AddU | Op::AddF | Op::Xor)
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// The lane's own value of a VGPR value from before the section.
    Own(Value),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
    },
    /// In the low half, the f16 sum of the low halves of `lanes`'
    /// `mask ? src : ident`, each aligned block of 2n lanes the sum of its
    /// two blocks of n (`v_add_f16`); in the high half, lane `high`'s.
    F16Sum {
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
        high: u32,
    },
    /// The low (`!high`) or high (`high`) dword of the double sum over
    /// `lanes` of `mask ? f64(lo, hi) : +0.0` (`v_add_f64`).
    Red64F {
        lo: Value,
        hi: Value,
        mask: Value,
        lanes: u64,
        high: bool,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
    /// [`Sym::F16Sum`] over every lane: `src`, `mask`, `ident`, `high`.
    F16Sum(Value, Value, u32, u32),
    /// [`Sym::Red64F`] over every lane: `lo`, `hi`, `mask`, `high`.
    ActiveF64(Value, Value, Value, bool),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
    /// A section the block leaves for its one successor, which restores
    /// EXEC: that block and the restore's index there (`end` is then the
    /// block's own end, or its `s_branch`); the shapes cover what that
    /// block writes before the restore, coming from this one.
    pub tail: Option<(usize, usize)>,
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut i = 0;
    while i < insts.len() {
        let inst = &insts[i];
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        let all = match inst.name() {
            "s_or_saveexec_b64" | "s_or_saveexec_b32" => Some(Operand::Const(193)),
            "s_nand_saveexec_b64" | "s_nand_saveexec_b32" => Some(Operand::Const(128)),
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => inst.write(Reg::S(*code)),
            _ => None,
        };
        let Some(saved) = saved else {
            i += 1;
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            insts[j].write(Reg::S(EXEC)).is_some()
                && insts[j].read(Reg::S(saved_code(inst))) == Some(saved)
        });
        let Some(end) = end else {
            // The console compiler shares a reduction's last steps and the
            // restore between the sections of a uniform branch's two ways:
            // the section goes on into the block they join in, as each way
            // brings it.
            if super::shared_restores()
                && let Exit::Jump(d) = p.blocks[block].exit
                && let Some(n) = p.blocks[d].preds.iter().position(|&q| q == block)
                && let Some(phi) = p.blocks[d]
                    .phis
                    .iter()
                    .find(|f| f.reg == Reg::S(saved_code(inst)) && f.args[n] == saved)
                && let Some(restore) = p.blocks[d].insts.iter().position(|j| {
                    j.write(Reg::S(EXEC)).is_some()
                        && j.read(Reg::S(saved_code(inst))) == Some(phi.value)
                })
            {
                let end = match insts.last() {
                    Some(l) if l.name() == "s_branch" => insts.len() - 1,
                    _ => insts.len(),
                };
                let joins: Vec<(Value, Value)> = p.blocks[d]
                    .phis
                    .iter()
                    .map(|f| (f.value, f.args[n]))
                    .collect();
                let rest = (&p.blocks[d].insts[..restore], &joins[..]);
                let shapes = simulate(&insts[(i + 1).min(end)..end], Some(rest), wave, saved);
                out.insert(
                    i,
                    Section {
                        end,
                        shapes,
                        tail: Some((d, restore)),
                    },
                );
            }
            break;
        };
        let shapes = simulate(&insts[i + 1..end], None, wave, saved);
        out.insert(
            i,
            Section {
                end,
                shapes,
                tail: None,
            },
        );
        i = end + 1;
    }
    out
}

fn saved_code(inst: &Inst) -> u8 {
    match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } => *code,
        _ => NULL,
    }
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
            }
        }
        _ => Sym::Unknown,
    }
}

/// `v_add_f16` of `a` and `b` into `dst`: the sums of two aligned blocks
/// of n lanes that make one of 2n, else unknown; the high half stays
/// `dst`'s.
fn f16_sum(a: Sym, b: Sym, dst: Sym) -> Sym {
    let part = |s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } => Some(((src, mask, ident), lanes, lanes.trailing_zeros())),
        Sym::F16Sum {
            src,
            mask,
            ident,
            lanes,
            high,
        } => Some(((src, mask, ident), lanes, high)),
        _ => None,
    };
    // An aligned block of n lanes, n a power of two: its first lane and n.
    let block = |lanes: u64| {
        let (at, n) = (lanes.trailing_zeros(), lanes.count_ones());
        (n.is_power_of_two() && at.is_multiple_of(n) && lanes == u64::MAX >> (64 - n) << at)
            .then_some((at, n))
    };
    let (Some(x), Some(y), Some(d)) = (part(a), part(b), part(dst)) else {
        return Sym::Unknown;
    };
    match (block(x.1), block(y.1)) {
        (Some((at, n)), Some((bt, m))) if n == m && at ^ bt == n && x.0 == y.0 && x.0 == d.0 => {
            let (src, mask, ident) = x.0;
            Sym::F16Sum {
                src,
                mask,
                ident,
                lanes: x.1 | y.1,
                high: d.2,
            }
        }
        _ => Sym::Unknown,
    }
}

/// The shapes of what `insts` write, then (`tail`) what the instructions of
/// the block the section goes on into write, its joins taking what this way
/// brings them.
fn simulate(
    insts: &[Inst],
    tail: Option<(&[Inst], &[(Value, Value)])>,
    wave: u32,
    saved: Value,
) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    // Masks the section derives from the saved EXEC (`s_and` with it).
    let mut masked = vec![saved];
    let (rest, joins) = tail.unwrap_or_default();
    for (k, inst) in insts.iter().chain(rest).enumerate() {
        if k == insts.len() {
            for &(phi, arg) in joins {
                let l = vec.get(&arg).cloned();
                vec.insert(phi, l.unwrap_or_else(|| vec![Sym::Own(arg); wave as usize]));
                uni.insert(phi, uni.get(&arg).copied().unwrap_or(Sym::Uni(arg)));
                if masked.contains(&arg) {
                    masked.push(phi);
                }
            }
        }
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let lo = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten();
            let Some(k) = lo else {
                return shapes(vec, uni, wave);
            };
            exec = if name == "s_mov_b64" {
                k
            } else {
                exec & !0xFFFF_FFFF | k & 0xFFFF_FFFF
            };
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten()
            else {
                return shapes(vec, uni, wave);
            };
            exec = exec & 0xFFFF_FFFF | k << 32;
            continue;
        }
        if let Fields::Sop { src, .. } = &inst.fields {
            if name.starts_with("s_and_b")
                && src.iter().any(|o| matches!(*o, Operand::Scalar { code, .. } if inst.read(Reg::S(code)).is_some_and(|v| masked.contains(&v))))
                && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
            {
                masked.push(w);
            }
            // A scalar copy (of a sum read from a lane, say).
            if name == "s_mov_b32"
                && super::f64_sums()
                && let Some(&Operand::Scalar { code, .. }) = src.first()
                && code != NULL
                && let (Some(x), Some(&(_, w))) = (inst.read(Reg::S(code)), inst.writes.first())
            {
                uni.insert(w, uni.get(&x).copied().unwrap_or(Sym::Uni(x)));
            }
            for &(_, w) in &inst.writes {
                uni.entry(w).or_insert(Sym::Unknown);
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|_| Sym::Own(x)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                let s = l
                    .and_then(|l| x.get(l as usize).copied())
                    .unwrap_or(Sym::Unknown);
                uni.insert(w, s);
            }
            continue;
        }
        if name == "v_add_f64"
            && super::f64_sums()
            && let Some(Operand::Vector { reg: dreg, .. }) = v.dst
        {
            // Each operand pair's dwords, per lane.
            let pair = |op: Option<&Operand>| -> Option<[Vec<Sym>; 2]> {
                let (lo, hi) = match *op? {
                    Operand::Vector { reg, .. } => (
                        Operand::Vector { reg, dwords: 1 },
                        Operand::Vector {
                            reg: reg + 1,
                            dwords: 1,
                        },
                    ),
                    Operand::Scalar { code, .. } if code != NULL => (
                        Operand::Scalar { code, dwords: 1 },
                        Operand::Scalar {
                            code: code + 1,
                            dwords: 1,
                        },
                    ),
                    _ => return None,
                };
                Some([lanes(&lo, &vec, &uni), lanes(&hi, &vec, &uni)])
            };
            // A lane's term: one lane's masked double (+0.0 outside the
            // mask), or a sum over some lanes.
            let term = |lo: Sym, hi: Sym| match (lo, hi) {
                (
                    Sym::Red {
                        op: None,
                        src: a,
                        mask,
                        ident: 0,
                        lanes,
                    },
                    Sym::Red {
                        op: None,
                        src: b,
                        mask: m,
                        ident: 0,
                        lanes: l,
                    },
                ) if (mask, lanes) == (m, l) => Some((a, b, mask, lanes)),
                (
                    Sym::Red64F {
                        lo: a,
                        hi: b,
                        mask,
                        lanes,
                        high: false,
                    },
                    Sym::Red64F {
                        lo,
                        hi,
                        mask: m,
                        lanes: l,
                        high: true,
                    },
                ) if (a, b, mask, lanes) == (lo, hi, m, l) => Some((a, b, mask, lanes)),
                _ => None,
            };
            let old = |r: u8| lanes(&Operand::Vector { reg: r, dwords: 1 }, &vec, &uni);
            let old = [old(dreg), old(dreg + 1)];
            let mut out = [
                vec![Sym::Unknown; wave as usize],
                vec![Sym::Unknown; wave as usize],
            ];
            if !modifiers
                && v.dpp.is_none()
                && let (Some([alo, ahi]), Some([blo, bhi])) =
                    (pair(v.src.first()), pair(v.src.get(1)))
            {
                for l in 0..wave as usize {
                    let sum = match (term(alo[l], ahi[l]), term(blo[l], bhi[l])) {
                        (Some((lo, hi, mask, a)), Some((lo2, hi2, mask2, b)))
                            if (lo, hi, mask) == (lo2, hi2, mask2) && a & b == 0 =>
                        {
                            Some((lo, hi, mask, a | b))
                        }
                        _ => None,
                    };
                    for (k, high) in [(0, false), (1, true)] {
                        out[k][l] = if exec >> l & 1 == 0 {
                            old[k][l]
                        } else {
                            sum.map_or(Sym::Unknown, |(lo, hi, mask, lanes)| Sym::Red64F {
                                lo,
                                hi,
                                mask,
                                lanes,
                                high,
                            })
                        };
                    }
                }
            }
            for (r, o) in [dreg, dreg + 1].into_iter().zip(out) {
                if let Some(w) = inst.write(Reg::V(r)) {
                    vec.insert(w, o);
                }
            }
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => vec![Sym::Unknown; wave as usize],
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied(), mask) {
                        (Sym::K(ident), Some(Sym::Own(x)), Some(m)) if masked.contains(&m) => {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m,
                                ident,
                                lanes: 1 << l,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read the other row of their half; lane
                // i's pick is nibble i of {src2, src1}.
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    _ => None,
                };
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        (0..wave)
                            .map(|l| {
                                let other = (l & !31) | ((l & 16) ^ 16);
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                a[(other | pick) as usize]
                            })
                            .collect()
                    }
                    _ => vec![Sym::Unknown; wave as usize],
                }
            }
            ("v_sub_nc_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (
                            Sym::Red {
                                op: None | Some(Op::AddU),
                                src,
                                mask,
                                ident: 0,
                                lanes,
                            },
                            Some(Sym::Red {
                                op: None | Some(Op::AddU),
                                src: s2,
                                mask: m2,
                                ident: 0,
                                lanes: l2,
                            }),
                        ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                            let rest = lanes & !l2;
                            if rest == 0 {
                                Sym::K(0)
                            } else {
                                Sym::Red {
                                    op: Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes: rest,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_add_f16", Some(a)) if v.opsel == 0 => {
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| b.get(l).map_or(Sym::Unknown, |&b| f16_sum(a[l], b, old[l])))
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l]))
                    .collect(),
                _ => vec![Sym::Unknown; wave as usize],
            },
            _ => vec![Sym::Unknown; wave as usize],
        };
        let merged = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        vec.insert(dst, merged);
    }
    shapes(vec, uni, wave)
}

fn constant(inst: &Inst) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    match *src.first()? {
        Operand::Const(c) => {
            let x = crate::resource::inline(c)?;
            Some(if (x as i32) < 0 {
                u64::MAX
            } else {
                u64::from(x)
            })
        }
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value, where every lane agrees on one.
fn shapes(
    vec: HashMap<Value, Vec<Sym>>,
    uni: HashMap<Value, Sym>,
    wave: u32,
) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let single = |s: Sym| match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Own(x) | Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        Sym::F16Sum {
            src,
            mask,
            ident,
            lanes,
            high,
        } if lanes == all => Some(Shape::F16Sum(src, mask, ident, high)),
        Sym::Red64F {
            lo,
            hi,
            mask,
            lanes,
            high,
        } if lanes == all => Some(Shape::ActiveF64(lo, hi, mask, high)),
        _ => None,
    };
    for (v, s) in uni {
        if let Some(sh) = single(s) {
            out.insert(v, sh);
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
        }
    }
    out
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return match first {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) | Sym::Own(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            Sym::F16Sum {
                src,
                mask,
                ident,
                lanes,
                high,
            } if lanes == all => Some(Shape::F16Sum(src, mask, ident, high)),
            Sym::Red64F {
                lo,
                hi,
                mask,
                lanes,
                high,
            } if lanes == all => Some(Shape::ActiveF64(lo, hi, mask, high)),
            _ => None,
        };
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut found: Option<(Op, Value, Value, u32, bool)> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | 1 << l;
        if !inclusive && set != below {
            return None;
        }
        let op = match (op, found) {
            (Some(o), _) => o,
            (None, Some(f)) => f.0,
            (None, None) => continue,
        };
        let me = (op, src, mask, ident, inclusive);
        match found {
            None => found = Some(me),
            Some(f) if f == me => {}
            _ => return None,
        }
    }
    let (op, src, mask, ident, inclusive) = found?;
    if let Sym::K(k) = lanes[0]
        && (inclusive || k != ident)
    {
        return None;
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    #[test]
    fn f16_sums_follow_only_the_consoles_tree() {
        let (x, m) = (Value(1), Value(2));
        let one = |l: u32| Sym::Red {
            op: None,
            src: x,
            mask: m,
            ident: 0,
            lanes: 1 << l,
        };
        let sum = |lanes: u64, high: u32| Sym::F16Sum {
            src: x,
            mask: m,
            ident: 0,
            lanes,
            high,
        };
        assert_eq!(f16_sum(one(1), one(0), one(0)), sum(0b11, 0));
        assert_eq!(
            f16_sum(sum(0b1100, 3), sum(0b11, 0), one(1)),
            sum(0b1111, 1)
        );
        // Lanes 0 and 2, a pair and a lane, pairs 2-3 and 4-5 and a pair
        // twice are no step of the tree.
        assert_eq!(f16_sum(one(0), one(2), one(0)), Sym::Unknown);
        assert_eq!(f16_sum(sum(0b11, 0), one(2), one(0)), Sym::Unknown);
        assert_eq!(
            f16_sum(sum(0b1100, 2), sum(0b11_0000, 4), one(2)),
            Sym::Unknown
        );
        assert_eq!(f16_sum(sum(0b11, 1), sum(0b11, 0), one(0)), Sym::Unknown);
        // The high half kept is the destination's.
        let all = f16_sum(sum(u64::MAX << 32, 32), sum(u64::MAX >> 32, 0), one(5));
        assert_eq!(all, sum(u64::MAX, 5));
        assert_eq!(
            per_lane(&[all; 64], u64::MAX),
            Some(Shape::F16Sum(x, m, 0, 5))
        );
    }
}
