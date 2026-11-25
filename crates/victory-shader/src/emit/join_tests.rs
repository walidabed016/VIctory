//! End-to-end tests of joins of values that have no DXIL form: synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present, and
//! run on the host GPU and WARP. A lane mask or a descriptor dword joined with
//! data stands only where no lane that brings it reads it; addresses into the
//! program's constant table join as offsets each predecessor computes.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::ShaderError;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;
const LITERAL: u8 = 0xFF;
/// Inline constants: 0, 1, 2 and -1.
const ZERO: u8 = 128;
const ONE: u8 = 129;
const TWO: u8 = 130;
const ONES: u8 = 193;
/// Console threads, one wave64.
const THREADS: u32 = 64;

const fn v(n: u8) -> u16 {
    256 + n as u16
}

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// A branch from dword `at` to dword `to`.
const fn sopp(op: u8, at: usize, to: usize) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | (to as i16 - at as i16 - 1) as u16 as u32
}

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// A loop of two turns, as the ped pre-skinning mesh programs run it. Each
/// turn writes the lane mask `v0 < 16` to s[2:3] and counts it in v5; the
/// first turn then overwrites s3 with 0x100 and sets v1 to -1, the second
/// skips that with v1 at `skipped`, and both join. A lane reads s3 there
/// (adding it to v5) only when v1 is -1. With `joins` 1, s3 first passes
/// through a second join (the first turn writes 0x100 to it again, the
/// second skips that), as the ped pixel programs' unrolled decal loop does,
/// and the read tests the first join's compare. With `joins` 2, the first
/// turn's path to that join has a join of its own where s3 also arrives
/// unwritten, so s3 reaches the second join along two paths. With
/// `compare`, the first join also compares s3 with 0, as the ped pixel
/// programs do, and ORs the mask that answer selects with `v1 != -1`
/// before branching on it. Each thread stores v5.
fn two_turns(skipped: u8, joins: u8, compare: bool) -> Vec<u32> {
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0; v4 = v0 * 4; s4 = 0; v5 = 0.
    let mut w = vec![sop1(0x03, 1, ZERO), 0xF408_0200, 0xFA00_0000];
    w.extend([vop2(0x1A, 4, u16::from(TWO), 0), sop1(0x03, 4, ZERO)]);
    w.push(vop1(0x01, 5, u16::from(ZERO)));
    let turn = w.len();
    // v_cmp_gt_u32 s[2:3], 16, v0; v_cndmask_b32 v6, 0, 1, s[2:3]; v5 += v6.
    w.extend(vop3(0xC4, 2, 128 + 16, v(0), 0));
    w.extend(vop3(0x101, 6, u16::from(ZERO), u16::from(ONE), 2));
    w.push(vop2(0x25, 5, v(5), 6));
    // v1 = skipped; s_cmp_eq_u32 s4, 1; s_cbranch_scc1 join.
    w.extend([vop1(0x01, 1, u16::from(skipped)), sopc(0x06, 4, ONE)]);
    let skip = w.len();
    w.push(0);
    // s3 = 0x100; v1 = -1.
    w.extend([
        sop1(0x03, 3, LITERAL),
        0x100,
        vop1(0x01, 1, u16::from(ONES)),
    ]);
    let join = w.len();
    w[skip] = sopp(0x05, skip, join);
    // v_cmp_eq_u32 vcc, -1, v1; s_andn2_b64 vcc, exec, vcc; s_cbranch_vccnz next.
    w.push(vopc(0xC2, u16::from(ONES), 1));
    if compare {
        // s_cmp_eq_u32 s3, 0; s_cselect_b64 s[12:13], exec, 0;
        // s_xor_b64 s[14:15], vcc, -1; s_or_b64 s[12:13], s[12:13], s[14:15];
        // s_and_b64 vcc, exec, s[12:13].
        w.extend([
            sopc(0x06, 3, ZERO),
            sop2(0x0B, 12, EXEC_LO, ZERO),
            sop2(0x13, 14, VCC_LO, ONES),
            sop2(0x11, 12, 12, 14),
            sop2(0x0F, VCC_LO, EXEC_LO, 12),
        ]);
    } else {
        w.push(sop2(0x15, VCC_LO, EXEC_LO, VCC_LO));
    }
    if joins > 0 {
        // s[6:7] = vcc; s_cmp_eq_u32 s4, 0; s_cbranch_scc0 second; s3 = 0x100;
        // second: vcc = s[6:7].
        w.extend([sop1(0x04, 6, VCC_LO), sopc(0x06, 4, ZERO)]);
        let skip = w.len();
        w.push(0);
        if joins > 1 {
            // s_cmp_eq_u32 s4, 1; s_cbranch_scc1 first; s3 = 0x100; first: s_nop.
            w.push(sopc(0x06, 4, ONE));
            let keep = w.len();
            w.extend([0, sop1(0x03, 3, LITERAL), 0x100]);
            w[keep] = sopp(0x05, keep, w.len());
            w.push(0xBF80_0000);
        } else {
            w.extend([sop1(0x03, 3, LITERAL), 0x100]);
        }
        w[skip] = sopp(0x04, skip, w.len());
        w.push(sop1(0x04, VCC_LO, 6));
    }
    let unread = w.len();
    w.push(0);
    // s5 = s3 & 0x100; v5 += s5.
    w.extend([sop2(0x0E, 5, 3, LITERAL), 0x100, vop2(0x25, 5, 5, 5)]);
    let next = w.len();
    w[unread] = sopp(0x07, unread, next);
    // s4 += 1; s_cmp_lt_u32 s4, 2; s_cbranch_scc1 turn.
    w.extend([sop2(0x00, 4, 4, ONE), sopc(0x0A, 4, TWO)]);
    w.push(sopp(0x05, w.len(), turn));
    // buffer_store_dword v5, v4, s[8:11], 0 offen.
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4,
    ]);
    w.push(S_ENDPGM);
    w
}

/// Lane `l` loads a dword of a table after the code, read through
/// `s_getpc_b64` at a join, and stores it: dword `l & 7` on the path
/// `s_cmp_<cmp>_u32 s0, 0` (s0 is 0) falls through to the load on, dword
/// `8 + (l & 7)` on the path that branches back to it (as the hair strand
/// programs do). The table's dwords are 1000 and on.
fn table_at_a_join(cmp: u8) -> Vec<u32> {
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0; v4 = v0 * 4; v1 = v0 & 7.
    let mut w = vec![sop1(0x03, 1, ZERO), 0xF408_0200, 0xFA00_0000];
    w.extend([vop2(0x1A, 4, u16::from(TWO), 0), vop2(0x1B, 1, 128 + 7, 0)]);
    // s_getpc_b64 s[16:17] (the next instruction's byte); s16 += the table's
    // distance from it; s_addc_u32 s17, s17, 0.
    let pc = 4 * (w.len() + 1);
    w.extend([sop1(0x1F, 16, 0), sop2(0x00, 16, 16, LITERAL), 0]);
    let distance = w.len() - 1;
    w.push(sop2(0x04, 17, 17, ZERO));
    // s_cmp_<cmp>_u32 s0, 0; s_cbranch_scc0 back.
    w.push(sopc(cmp, 0, ZERO));
    let skip = w.len();
    w.push(0);
    // v_add_co_u32 v5, vcc, v2, s16; v_add_co_ci_u32 v6, vcc, 0, s17, vcc.
    let address = |w: &mut Vec<u32>| {
        for [w0, w1] in [
            vop3(0x30F, 5, v(2), 16, 0),
            vop3(0x128, 6, u16::from(ZERO), 17, u16::from(VCC_LO)),
        ] {
            w.extend([w0 | u32::from(VCC_LO) << 8, w1]);
        }
    };
    // v2 = v1 * 4.
    w.push(vop2(0x1A, 2, u16::from(TWO), 1));
    address(&mut w);
    // global_load_dword v7, v[5:6], off; s_waitcnt vmcnt(0);
    // buffer_store_dword v7, v4, s[8:11], 0 offen.
    let load = w.len();
    w.extend([0xDC30_8000, 7 << 24 | 0x7D << 16 | 5, 0xBF8C_3F70]);
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 7 << 8 | 4,
    ]);
    w.push(S_ENDPGM);
    // v2 = v1 * 4 + 32; s_branch load.
    let back = w.len();
    w[skip] = sopp(0x04, skip, back);
    w.extend([vop2(0x1A, 2, u16::from(TWO), 1), vop2(0x25, 2, 128 + 32, 2)]);
    address(&mut w);
    w.push(sopp(0x02, w.len(), load));
    w[distance] = (4 * w.len() - pc) as u32;
    w.extend(1000..1016);
    w
}

/// Every lane stores a dword of a table after the code, read with
/// `s_load_dword` through `s_getpc_b64` at a join: dword 1, its address
/// added in VGPRs and read back with `v_readfirstlane_b32`, on the path
/// `s_cmp_<cmp>_u32 s0, 0` (s0 is 0) falls through to the load on; dword 9,
/// added in SGPRs, on the path that branches back to it (as the
/// reconstruct-raymarch scattering programs do). The table's dwords are
/// 1000 and on.
fn scalar_table_at_a_join(cmp: u8) -> Vec<u32> {
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0; v4 = v0 * 4.
    let mut w = vec![sop1(0x03, 1, ZERO), 0xF408_0200, 0xFA00_0000];
    w.push(vop2(0x1A, 4, u16::from(TWO), 0));
    // s_getpc_b64 s[16:17] (the next instruction's byte); s16 += the table's
    // distance from it; s_addc_u32 s17, s17, 0.
    let pc = 4 * (w.len() + 1);
    w.extend([sop1(0x1F, 16, 0), sop2(0x00, 16, 16, LITERAL), 0]);
    let distance = w.len() - 1;
    w.push(sop2(0x04, 17, 17, ZERO));
    // s_cmp_<cmp>_u32 s0, 0; s_cbranch_scc0 back.
    w.push(sopc(cmp, 0, ZERO));
    let skip = w.len();
    w.push(0);
    // v2 = 4; v_add_co_u32 v5, vcc, v2, s16; v_add_co_ci_u32 v6, vcc, 0,
    // s17, vcc; v_readfirstlane_b32 s2, v5; v_readfirstlane_b32 s3, v6.
    w.push(vop1(0x01, 2, 128 + 4));
    for [w0, w1] in [
        vop3(0x30F, 5, v(2), 16, 0),
        vop3(0x128, 6, u16::from(ZERO), 17, u16::from(VCC_LO)),
    ] {
        w.extend([w0 | u32::from(VCC_LO) << 8, w1]);
    }
    w.extend([vop1(0x02, 2, v(5)), vop1(0x02, 3, v(6))]);
    // s_load_dword s4, s[2:3], null; s_waitcnt lgkmcnt(0); v7 = s4;
    // buffer_store_dword v7, v4, s[8:11], 0 offen.
    let load = w.len();
    w.extend([0xF400_0101, 0xFA00_0000, 0xBF8C_C07F, vop1(0x01, 7, 4)]);
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 7 << 8 | 4,
    ]);
    w.push(S_ENDPGM);
    // s_add_u32 s2, s16, 36; s_addc_u32 s3, s17, 0; s_branch load.
    let back = w.len();
    w[skip] = sopp(0x04, skip, back);
    w.extend([sop2(0x00, 2, 16, 128 + 36), sop2(0x04, 3, 17, ZERO)]);
    w.push(sopp(0x02, w.len(), load));
    w[distance] = (4 * w.len() - pc) as u32;
    w.extend(1000..1016);
    w
}

/// Each thread stores `v0 + 1000` through the UAV descriptor the scalar
/// code loads on one of three paths, chosen by `s4 = path`: paths 0 and 1
/// join first (each adds its own number to the value, then the join adds
/// 16), and that join meets path 2 (as the screen-probe ray programs merge
/// code tails, each path loading the same table entry).
fn loaded_on_three_paths(path: u8) -> Vec<u32> {
    // v4 = v0 * 4; v5 = v0 + 1000; s1 = 0; s4 = path.
    let mut w = vec![
        vop2(0x1A, 4, u16::from(TWO), 0),
        vop2(0x25, 5, 0xFF, 0),
        1000,
    ];
    w.extend([sop1(0x03, 1, ZERO), sop1(0x03, 4, 128 + path)]);
    // s_cmp_eq_u32 s4, 0; s_cbranch_scc1 first; s_cmp_eq_u32 s4, 1;
    // s_cbranch_scc1 second.
    w.push(sopc(0x06, 4, ZERO));
    let to_first = w.len();
    w.push(0);
    w.push(sopc(0x06, 4, ONE));
    let to_second = w.len();
    w.push(0);
    // s_load_dwordx4 s[8:11], s[0:1], 0; v5 += 2; s_branch last.
    let load = [0xF408_0200, 0xFA00_0000];
    w.extend(load);
    w.push(vop2(0x25, 5, u16::from(TWO), 5));
    let to_last = w.len();
    w.push(0);
    // first: the load; s_branch inner.
    w[to_first] = sopp(0x05, to_first, w.len());
    w.extend(load);
    let to_inner = w.len();
    w.push(0);
    // second: the load; v5 += 1.
    w[to_second] = sopp(0x05, to_second, w.len());
    w.extend(load);
    w.push(vop2(0x25, 5, u16::from(ONE), 5));
    // inner: v5 += 16.
    w[to_inner] = sopp(0x02, to_inner, w.len());
    w.push(vop2(0x25, 5, 128 + 16, 5));
    // last: s_waitcnt lgkmcnt(0); buffer_store_dword v5, v4, s[8:11], 0 offen.
    w[to_last] = sopp(0x02, to_last, w.len());
    w.push(0xBF8C_C07F);
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4,
    ]);
    w.push(S_ENDPGM);
    w
}

/// Lanes 0-15 (`v_cmp_gt_u32`, `s_and_saveexec_b64`) run an arm that sets
/// s1 to 0 and loads the UAV descriptor to s[8:11]; after it each stores
/// `v0 + 1000` through that descriptor and `v0 + 2000` at dword 16 on,
/// through the one it then loads from s[0:1]. The arm's mask is EXEC and
/// the mask on entry, as the light-probe schedule program keeps its lanes
/// apart from the quads it samples in, so every lane that stores ran the
/// arm. With `inner`, the mask is EXEC and `v0 < inner` instead, and
/// lanes that skipped the arm store too.
fn loaded_in_an_arm(inner: Option<u8>) -> Vec<u32> {
    // v4 = v0 * 4; v6 = v4 + 64; v5 = v0 + 1000; v7 = v0 + 2000;
    // s[12:13] = exec.
    let mut w = vec![vop2(0x1A, 4, u16::from(TWO), 0), vop2(0x25, 6, 128 + 64, 4)];
    w.extend([vop2(0x25, 5, 0xFF, 0), 1000, vop2(0x25, 7, 0xFF, 0), 2000]);
    w.push(sop1(0x04, 12, EXEC_LO));
    // v_cmp_gt_u32 vcc, 16, v0; s_and_saveexec_b64 s[6:7], vcc;
    // s_cbranch_execz end.
    w.extend([vopc(0xC4, 128 + 16, 0), sop1(0x24, 6, VCC_LO)]);
    let to_end = w.len();
    w.push(0);
    // s_and_saveexec_b64 s[14:15], s[12:13] (or vcc = v0 < inner);
    // s_cbranch_execz join.
    match inner {
        None => w.push(sop1(0x24, 14, 12)),
        Some(n) => w.extend([vopc(0xC4, 128 + u16::from(n), 0), sop1(0x24, 14, VCC_LO)]),
    }
    let to_join = w.len();
    w.push(0);
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0.
    w.extend([sop1(0x03, 1, ZERO), 0xF408_0200, 0xFA00_0000]);
    // join: exec = s[14:15]; s_load_dwordx4 s[16:19], s[0:1], 0;
    // s_waitcnt lgkmcnt(0); buffer_store_dword v5, v4, s[8:11], 0 offen;
    // buffer_store_dword v7, v6, s[16:19], 0 offen.
    w[to_join] = sopp(0x08, to_join, w.len());
    w.extend([
        sop1(0x04, EXEC_LO, 14),
        0xF408_0400,
        0xFA00_0000,
        0xBF8C_C07F,
    ]);
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4,
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 4 << 16 | 7 << 8 | 6,
    ]);
    // end.
    w[to_end] = sopp(0x08, to_end, w.len());
    w.push(S_ENDPGM);
    w
}

/// Lanes 0-15 run an arm under EXEC and the mask on entry (or `v0 < inner`),
/// as in [`loaded_in_an_arm`], whose scalar code points s[2:3] into a table
/// after the code, read through `s_getpc_b64`: at dword 1 on the path
/// `s_cmp_<cmp>_u32 s0, 0` (s0 is 0) falls through on, at dword 9 on the
/// one it branches on. After the arm each lane loads the dword at s[2:3]
/// with `s_load_dword` and stores it at dword `v0`; on the edge that skips
/// the arm s[2:3] hold what they held on entry. The table's dwords are 1000
/// and on.
fn table_pointed_at_in_an_arm(cmp: u8, inner: Option<u8>) -> Vec<u32> {
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0; v4 = v0 * 4; s[12:13] = exec.
    let mut w = vec![sop1(0x03, 1, ZERO), 0xF408_0200, 0xFA00_0000];
    w.extend([vop2(0x1A, 4, u16::from(TWO), 0), sop1(0x04, 12, EXEC_LO)]);
    // s_getpc_b64 s[16:17] (the next instruction's byte); s16 += the table's
    // distance from it; s_addc_u32 s17, s17, 0.
    let pc = 4 * (w.len() + 1);
    w.extend([sop1(0x1F, 16, 0), sop2(0x00, 16, 16, LITERAL), 0]);
    let distance = w.len() - 1;
    w.push(sop2(0x04, 17, 17, ZERO));
    // v_cmp_gt_u32 vcc, 16, v0; s_and_saveexec_b64 s[6:7], vcc;
    // s_cbranch_execz end.
    w.extend([vopc(0xC4, 128 + 16, 0), sop1(0x24, 6, VCC_LO)]);
    let to_end = w.len();
    w.push(0);
    // s_and_saveexec_b64 s[14:15], s[12:13] (or vcc = v0 < inner);
    // s_cbranch_execz join.
    match inner {
        None => w.push(sop1(0x24, 14, 12)),
        Some(n) => w.extend([vopc(0xC4, 128 + u16::from(n), 0), sop1(0x24, 14, VCC_LO)]),
    }
    let to_join = w.len();
    w.push(0);
    // s_cmp_<cmp>_u32 s0, 0; s_cbranch_scc0 far; s_add_u32 s2, s16, 4;
    // s_addc_u32 s3, s17, 0; s_branch join; far: s_add_u32 s2, s16, 36;
    // s_addc_u32 s3, s17, 0.
    w.push(sopc(cmp, 0, ZERO));
    let to_far = w.len();
    w.push(0);
    w.extend([sop2(0x00, 2, 16, 128 + 4), sop2(0x04, 3, 17, ZERO)]);
    let near = w.len();
    w.push(0);
    w[to_far] = sopp(0x04, to_far, w.len());
    w.extend([sop2(0x00, 2, 16, 128 + 36), sop2(0x04, 3, 17, ZERO)]);
    // join: exec = s[14:15]; s_load_dword s4, s[2:3], null;
    // s_waitcnt lgkmcnt(0); v7 = s4; buffer_store_dword v7, v4, s[8:11], 0
    // offen.
    w[to_join] = sopp(0x08, to_join, w.len());
    w[near] = sopp(0x02, near, w.len());
    w.extend([sop1(0x04, EXEC_LO, 14), 0xF400_0101, 0xFA00_0000]);
    w.extend([0xBF8C_C07F, vop1(0x01, 7, 4)]);
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 7 << 8 | 4,
    ]);
    // end.
    w[to_end] = sopp(0x08, to_end, w.len());
    w.push(S_ENDPGM);
    w[distance] = (4 * w.len() - pc) as u32;
    w.extend(1000..1016);
    w
}

/// Each lane counts an index up from `min(v0 & 7, 3)` until it passes
/// `min(v0 >> 3, 3)` (or, without `clamp`, `v0 >> 3`), at least once,
/// adding `index + 1` to element `index` of an array in v20..v23 through
/// `v_movrels_b32` and `v_movreld_b32` in waterfall loops over the index
/// (M0 from `v_readfirstlane_b32`), and stores the array's sum. As in the
/// light-loop prescalarisation, the array's registers run on into two that
/// hold descriptor dwords (v24, v25) and into the index and its end (v28,
/// v29), to the program's last VGPR: each move names them all.
fn index_loop(clamp: bool) -> Vec<u32> {
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0; v4 = v0 * 4;
    // s[16:17] = exec.
    let mut w = vec![sop1(0x03, 1, ZERO), 0xF408_0200, 0xFA00_0000];
    w.extend([vop2(0x1A, 4, u16::from(TWO), 0), sop1(0x04, 16, EXEC_LO)]);
    // v7 = v0 & 7; v8 = v0 >> 3; v27 = med3_u32(v7, 0, 3);
    // v29 = med3_u32(v8, 0, 3) (or v8).
    w.extend([vop2(0x1B, 7, 128 + 7, 0), vop2(0x16, 8, 128 + 3, 0)]);
    w.extend(vop3(0x159, 27, v(7), u16::from(ZERO), 128 + 3));
    if clamp {
        w.extend(vop3(0x159, 29, v(8), u16::from(ZERO), 128 + 3));
    } else {
        w.push(vop1(0x01, 29, v(8)));
    }
    // v20..v23, v26, v30, v31 = 0; s_waitcnt lgkmcnt(0); v24 = s8; v25 = s9.
    for r in [20, 21, 22, 23, 26, 30, 31] {
        w.push(vop1(0x01, r, u16::from(ZERO)));
    }
    w.extend([0xBF8C_C07F, vop1(0x01, 24, 8), vop1(0x01, 25, 9)]);
    // s[12:13] = exec (the lanes still counting); v28 = v27; s_branch body.
    w.extend([sop1(0x04, 12, EXEC_LO), vop1(0x01, 28, v(27))]);
    let to_body = w.len();
    w.push(0);
    // head: exec = s[14:15] & s[12:13].
    let head = w.len();
    w.push(sop2(0x0F, EXEC_LO, 14, 12));
    w[to_body] = sopp(0x02, to_body, w.len());
    // A move in a waterfall over v28: s[6:7] = exec; again: s2 = the first
    // lane's v28; vcc = v28 == s2; s_and_saveexec_b64 s[4:5], vcc; m0 = s2;
    // the move; exec ^= s[4:5]; s_cbranch_execnz again; exec = s[6:7].
    let waterfall = |w: &mut Vec<u32>, mov: u32| {
        w.push(sop1(0x04, 6, EXEC_LO));
        let again = w.len();
        w.extend([
            vop1(0x02, 2, v(28)),
            vopc(0xC2, 2, 28),
            sop1(0x24, 4, VCC_LO),
        ]);
        w.extend([sop1(0x03, 124, 2), mov, sop2(0x13, EXEC_LO, EXEC_LO, 4)]);
        w.push(sopp(0x09, w.len(), again));
        w.push(sop1(0x04, EXEC_LO, 6));
    };
    // body: v_movrels_b32 v5, v20; v5 += v28 + 1; v_movreld_b32 v20, v5.
    waterfall(&mut w, vop1(0x43, 5, v(20)));
    w.extend([vop2(0x25, 5, v(28), 5), vop2(0x25, 5, u16::from(ONE), 5)]);
    waterfall(&mut w, vop1(0x42, 20, v(5)));
    // v28 += 1; s[14:15] = exec; v_cmpx_gt_u32 exec, v28, v29 (the lanes
    // done); s[12:13] &= ~exec; s_cbranch_scc1 head.
    w.extend([vop2(0x25, 28, u16::from(ONE), 28), sop1(0x04, 14, EXEC_LO)]);
    w.extend([vopc(0xD4, v(28), 29), sop2(0x15, 12, 12, EXEC_LO)]);
    w.push(sopp(0x05, w.len(), head));
    // exec = s[16:17]; v6 = v20 + v21 + v22 + v23;
    // buffer_store_dword v6, v4, s[8:11], 0 offen.
    w.push(sop1(0x04, EXEC_LO, 16));
    w.extend([
        vop2(0x25, 6, v(20), 21),
        vop2(0x25, 6, v(22), 6),
        vop2(0x25, 6, v(23), 6),
    ]);
    w.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 6 << 8 | 4,
    ]);
    w.push(S_ENDPGM);
    w
}

/// Lowers a wave64 compute program of one wave whose UAV table is at s0.
fn lower_cs(words: &[u32]) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            // One user SGPR, thread ids x.
            rsrc2: 1 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        view_strides: lowered.view_strides,
        heap_indices: Vec::new(),
        heap_unknown: 0,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
    })
}

/// Runs `shader` (one group) on the hardware GPU, which must take it, and
/// on WARP if it takes it; returns, per one that ran, the UAV's dwords.
fn run(shader: &HostShader) -> Vec<(&'static str, Vec<u32>)> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) if warp => {
                eprintln!("WARP refused the pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(THREADS)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    assert!(!out.is_empty(), "no SM 6.6 device");
    out
}

#[test]
fn a_mask_joined_with_data_runs_where_only_the_data_is_read() {
    let shader = lower_cs(&two_turns(ZERO, 0, false)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|x| 0x100 + if x < 16 { 2 } else { 0 })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_mask_a_second_join_passes_on_runs_where_only_the_data_is_read() {
    let shader = lower_cs(&two_turns(ZERO, 1, false)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|x| 0x100 + if x < 16 { 2 } else { 0 })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_mask_reaching_a_join_along_two_paths_runs_where_only_the_data_is_read() {
    let shader = lower_cs(&two_turns(ZERO, 2, false)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|x| 0x100 + if x < 16 { 2 } else { 0 })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_mask_a_compare_reads_where_an_or_overrides_its_answer_runs() {
    let shader = lower_cs(&two_turns(ZERO, 0, true)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|x| 0x100 + if x < 16 { 2 } else { 0 })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_table_address_joined_from_two_paths_reads_each_paths_dword() {
    for (cmp, first) in [(0x06, 1000), (0x07, 1008)] {
        let shader = lower_cs(&table_at_a_join(cmp)).expect("lower");
        let want: Vec<u32> = (0..THREADS).map(|l| first + (l & 7)).collect();
        for (label, got) in run(&shader) {
            assert_eq!(got, want, "{label}, s_cmp {cmp:#x}");
        }
    }
}

#[test]
fn a_scalar_table_address_joined_from_two_paths_reads_each_paths_dword() {
    for (cmp, dword) in [(0x06, 1001), (0x07, 1009)] {
        let shader = lower_cs(&scalar_table_at_a_join(cmp)).expect("lower");
        let want = vec![dword; THREADS as usize];
        for (label, got) in run(&shader) {
            assert_eq!(got, want, "{label}, s_cmp {cmp:#x}");
        }
    }
}

#[test]
fn a_descriptor_joined_twice_is_the_one_each_path_loads() {
    for (path, add) in [(0, 16), (1, 17), (2, 2)] {
        let shader = lower_cs(&loaded_on_three_paths(path)).expect("lower");
        let want: Vec<u32> = (0..THREADS).map(|l| 1000 + l + add).collect();
        for (label, got) in run(&shader) {
            assert_eq!(got, want, "{label}, path {path}");
        }
    }
}

#[test]
fn scalars_set_in_an_arm_every_storing_lane_runs_are_the_arms() {
    let shader = lower_cs(&loaded_in_an_arm(None)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|d| match d {
            0..16 => 1000 + d,
            16..32 => 2000 + d - 16,
            _ => 0,
        })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn scalars_set_in_an_arm_are_refused_where_a_lane_that_skipped_it_reads_them() {
    match lower_cs(&loaded_in_an_arm(Some(8))) {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "an address into a descriptor heap with an untracked high half");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

#[test]
fn a_table_address_an_arm_sets_is_read_on_each_path_its_lanes_take() {
    for (cmp, dword) in [(0x06, 1001), (0x07, 1009)] {
        let shader = lower_cs(&table_pointed_at_in_an_arm(cmp, None)).expect("lower");
        let want: Vec<u32> = (0..THREADS)
            .map(|l| if l < 16 { dword } else { 0 })
            .collect();
        for (label, got) in run(&shader) {
            assert_eq!(got, want, "{label}, s_cmp {cmp:#x}");
        }
    }
}

#[test]
fn a_table_address_an_arm_sets_is_refused_where_a_lane_that_skipped_it_reads_it() {
    for cmp in [0x06, 0x07] {
        match lower_cs(&table_pointed_at_in_an_arm(cmp, Some(8))) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(
                    what, "an address in an SGPR that is no root parameter",
                    "s_cmp {cmp:#x}"
                );
            }
            other => panic!("s_cmp {cmp:#x}: {:?}", other.map(|_| ())),
        }
    }
}

#[test]
fn an_m0_index_a_loop_keeps_in_range_reads_no_register_past_it() {
    let shader = lower_cs(&index_loop(true)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|l| {
            let (first, last) = ((l & 7).min(3), (l >> 3).min(3));
            (first..=last.max(first)).map(|k| k + 1).sum()
        })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn an_m0_index_no_bound_shows_for_reads_every_register_it_names() {
    match lower_cs(&index_loop(false)) {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a descriptor's address or format read as data");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

#[test]
fn a_mask_joined_with_data_is_refused_where_a_lane_bringing_it_reads_it() {
    for (joins, compare) in [(0, false), (1, false), (2, false), (0, true)] {
        match lower_cs(&two_turns(ONES, joins, compare)) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(what, "a lane mask read as data", "joins {joins} {compare}");
            }
            other => panic!("joins {joins} {compare}: {:?}", other.map(|_| ())),
        }
    }
}
