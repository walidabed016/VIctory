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
/// unwritten, so s3 reaches the second join along two paths. Each thread
/// stores v5.
fn two_turns(skipped: u8, joins: u8) -> Vec<u32> {
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
    w.extend([
        vopc(0xC2, u16::from(ONES), 1),
        sop2(0x15, VCC_LO, EXEC_LO, VCC_LO),
    ]);
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
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
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
    let shader = lower_cs(&two_turns(ZERO, 0)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|x| 0x100 + if x < 16 { 2 } else { 0 })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_mask_a_second_join_passes_on_runs_where_only_the_data_is_read() {
    let shader = lower_cs(&two_turns(ZERO, 1)).expect("lower");
    let want: Vec<u32> = (0..THREADS)
        .map(|x| 0x100 + if x < 16 { 2 } else { 0 })
        .collect();
    for (label, got) in run(&shader) {
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_mask_reaching_a_join_along_two_paths_runs_where_only_the_data_is_read() {
    let shader = lower_cs(&two_turns(ZERO, 2)).expect("lower");
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
fn a_mask_joined_with_data_is_refused_where_a_lane_bringing_it_reads_it() {
    for joins in 0..3 {
        match lower_cs(&two_turns(ONES, joins)) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(what, "a lane mask read as data", "joins {joins}");
            }
            other => panic!("joins {joins}: {:?}", other.map(|_| ())),
        }
    }
}
