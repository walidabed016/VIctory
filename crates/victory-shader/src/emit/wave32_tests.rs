//! End-to-end tests of cross-lane code in wave32 compute programs
//! (`ds_bpermute_b32`, whole-wave sections): synthetic GFX10 programs
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU (and WARP when it takes `WaveSize(32)`). Each thread stores
//! what the console's wave gives it.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;
/// Inline constant 0.
const ZERO: u16 = 128;
/// Threads: two console waves of 32 lanes.
const THREADS: u32 = 64;
/// Dwords each thread stores.
const SLOTS: u32 = 4;
/// What a lane outside EXEC keeps.
const DEAD: u32 = 0xDEAD;

const fn inline(n: u16) -> u16 {
    128 + n
}

const fn v(n: u8) -> u16 {
    256 + n as u16
}

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

const fn sopp(op: u8, simm: i16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | simm as u16 as u32
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

/// `v_mov_b32 vdst, k` (a literal).
const fn mov_k(vdst: u8, k: u32) -> [u32; 2] {
    [vop1(0x001, vdst, 255), k]
}

/// `ds_bpermute_b32 vdst, vaddr, vdata offset:offset`.
const fn bpermute(vdst: u8, addr: u8, data0: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | 0xB3 << 18 | offset as u32,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

/// `buffer_store_dword vdata, vaddr, s[8:11], 0 offen offset:byte`.
const fn store(vdata: u8, vaddr: u8, byte: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte as u32 & 0xFFF),
        0x80 << 24 | 2 << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// `s_mov_b32 s12, exec_lo` and back; `v_cmpx_ne_u32 exec_lo, 0, v`.
const SAVE_EXEC: u32 = sop1(0x03, 12, EXEC_LO);
const RESTORE_EXEC: u32 = sop1(0x03, EXEC_LO, 12);
const fn cmpx_ne_0(v: u8) -> u32 {
    vopc(0xD5, ZERO, v)
}

/// Lowers a wave32 compute program of [`THREADS`] threads whose UAV table
/// is at s0.
fn lower32(words: &[u32]) -> Result<lower::Lowered, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
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
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
}

/// [`lower32`], which must take the program, validated.
fn build(words: &[u32]) -> HostShader {
    let lowered = lower32(words).expect("lower");
    let shader = HostShader {
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
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` (one group) on the hardware GPU, which must take it, and
/// on WARP if it takes it; returns, per one that ran, the UAV's dwords.
fn run(shader: &HostShader) -> Vec<(&'static str, Vec<u32>)> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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
                    views: &[View::RawUav(THREADS * SLOTS)],
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
fn backward_permutes_read_lanes_of_the_wave_and_0_outside_exec() {
    // Thread x holds 1000 + x and reads lane (20x + 12) / 4 mod 32 of its
    // wave: with every lane, with the lanes whose x & 5 is not 0
    // (`v_cmpx` alone) and with those whose x & 6 is not 0 (`v_cmpx`,
    // `s_cbranch_execz`). A source lane outside EXEC gives 0; a lane
    // outside EXEC keeps its register.
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    w.push(vop2(0x1A, 4, inline((SLOTS * 4).trailing_zeros() as u16), 0));
    w.extend([vop2(0x25, 1, 255, 0), 1000]);
    w.push(vop2(0x0B, 2, inline(20), 0));
    w.extend(bpermute(5, 2, 1, 12));
    w.extend(mov_k(6, DEAD));
    w.push(vop2(0x1B, 7, inline(5), 0));
    w.extend([SAVE_EXEC, cmpx_ne_0(7)]);
    w.extend(bpermute(6, 2, 1, 12));
    w.push(RESTORE_EXEC);
    w.extend(mov_k(8, DEAD));
    w.push(vop2(0x1B, 9, inline(6), 0));
    w.extend([SAVE_EXEC, cmpx_ne_0(9), sopp(0x08, 2)]);
    w.extend(bpermute(8, 2, 1, 12));
    w.push(RESTORE_EXEC);
    for (k, r) in [5u8, 6, 8].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    let shader = build(&w);
    let read = |x: u32, on: &dyn Fn(u32) -> bool| {
        let src = x & !31 | (5 * x + 3) & 31;
        match (on(x), on(src)) {
            (false, _) => DEAD,
            (true, false) => 0,
            (true, true) => 1000 + src,
        }
    };
    for (label, got) in run(&shader) {
        for x in 0..THREADS {
            let want = [
                read(x, &|_| true),
                read(x, &|f| f & 5 != 0),
                read(x, &|f| f & 6 != 0),
            ];
            let at = (x * SLOTS) as usize;
            assert_eq!(got[at..at + 3], want, "{label}: thread {x}");
        }
    }
}

/// A VOP3 instruction.
const fn vop3(op: u16, vdst: u8, src: [u16; 3]) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src[0] as u32 & 0x1FF) | (src[1] as u32 & 0x1FF) << 9 | (src[2] as u32 & 0x1FF) << 18,
    ]
}

/// `v_add_co_u32 vdst, vcc_lo, src0, src1` (VOP3b).
const fn add_co(vdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    [
        0x35 << 26 | 0x30F << 16 | (VCC_LO as u32) << 8 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9,
    ]
}

/// `v_mov_b32 vdst, vsrc` with DPP8 lane selects `sel`, or (`sel` none)
/// `row_mirror bound_ctrl:1`.
const fn mov_dpp(vdst: u8, vsrc: u8, sel: Option<[u32; 8]>) -> [u32; 2] {
    let Some(sel) = sel else {
        return [
            vop1(0x001, vdst, 0xFA),
            vsrc as u32 | 0x140 << 8 | 1 << 19 | 0xF << 24 | 0xF << 28,
        ];
    };
    let mut s = 0;
    let mut i = 0;
    while i < 8 {
        s |= sel[i] << (3 * i);
        i += 1;
    }
    [vop1(0x001, vdst, 0xE9), vsrc as u32 | s << 8]
}

/// A whole-wave 64-bit sum as the console compiler expands it in a wave32
/// program: of v[6:7] in the lanes of the saved EXEC (s13), with v[2:5]
/// and VCC; the low dword to s2, the high one to s3.
fn wave_sum64() -> Vec<u32> {
    let mut w = vec![
        sop1(0x3D, 13, 193),
        vop1(0x001, 2, ZERO),
        vop1(0x001, 3, ZERO),
    ];
    w.extend(vop3(0x101, 2, [v(2), v(6), 13]));
    w.extend(vop3(0x101, 3, [v(3), v(7), 13]));
    let add = |w: &mut Vec<u32>| {
        w.extend(add_co(2, v(4), v(2)));
        w.push(vop2(0x28, 3, v(5), 3));
    };
    for sel in [
        Some([1, 0, 3, 2, 5, 4, 7, 6]),
        Some([2, 2, 0, 0, 6, 6, 4, 4]),
        Some([4, 4, 4, 4, 0, 0, 0, 0]),
        None,
    ] {
        w.extend(mov_dpp(4, 2, sel));
        w.extend(mov_dpp(5, 3, sel));
        add(&mut w);
    }
    w.extend(vop3(0x378, 4, [v(2), ZERO, ZERO]));
    w.extend(vop3(0x378, 5, [v(3), ZERO, ZERO]));
    add(&mut w);
    w.extend(vop3(0x360, 2, [v(2), ZERO, 0]));
    w.extend(vop3(0x360, 3, [v(3), ZERO, 0]));
    w.push(sop1(0x03, EXEC_LO, 13));
    w
}

#[test]
fn a_whole_wave_64_bit_sum_gives_the_sum_of_the_low_dwords() {
    // In each wave the lanes whose x & 3 is not 0 (`s_and_saveexec`,
    // `s_cbranch_execz`) sum (0x9000_0000 + x, x) over the wave, and each
    // stores the low dword (s2), or the high one (s3), which needs the
    // carries and is refused.
    let program = |high: bool| {
        let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
        let shift = inline((SLOTS * 4).trailing_zeros() as u16);
        w.push(vop2(0x1A, 9, shift, 0));
        w.extend([vop2(0x25, 6, 255, 0), 0x9000_0000]);
        w.push(vop1(0x001, 7, v(0)));
        w.extend(mov_k(8, DEAD));
        w.push(vop2(0x1B, 12, inline(3), 0));
        w.extend([vopc(0xC5, ZERO, 12), sop1(0x3C, 12, VCC_LO)]);
        let mut region = wave_sum64();
        region.push(vop1(0x001, 8, if high { 3 } else { 2 }));
        w.push(sopp(0x08, region.len() as i16));
        w.extend(region);
        w.push(sop1(0x03, EXEC_LO, 12));
        w.extend(store(8, 9, 0));
        w.push(S_ENDPGM);
        w
    };
    let refused = lower32(&program(true)).err();
    assert!(
        matches!(
            refused,
            Some(crate::ShaderError::Unsupported {
                what: "a whole-wave section step the simulation does not follow",
                ..
            })
        ),
        "{refused:?}"
    );
    let shader = build(&program(false));
    let on = |x: u32| x & 3 != 0;
    for (label, got) in run(&shader) {
        for x in 0..THREADS {
            let wave = x & !31..(x & !31) + 32;
            let want = if on(x) {
                wave.filter(|&g| on(g))
                    .fold(0u32, |s, g| s.wrapping_add(0x9000_0000 + g))
            } else {
                DEAD
            };
            assert_eq!(got[(x * SLOTS) as usize], want, "{label}: thread {x}");
        }
    }
}

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

#[test]
fn masks_of_lanes_outside_exec_read_as_data_are_the_scalar_units() {
    // In the lanes whose x & 5 is not 0 (`s_and_saveexec`), each stores the
    // other lanes of its wave three ways: the saved EXEC without these
    // (`s_xor_b32 s12, exec_lo, s12`, as an if's else mask), the compare's
    // complement (`s_not_b32 s13, vcc_lo`) and a copy of the first
    // (`s_mov_b32 s14, s12`). No lane they hold runs where they are
    // written, so no ballot there sees them; the scalar unit's operations
    // on the masks it reads give them.
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    let shift = inline((SLOTS * 4).trailing_zeros() as u16);
    w.push(vop2(0x1A, 4, shift, 0));
    for r in [5, 6, 8] {
        w.extend(mov_k(r, DEAD));
    }
    w.push(vop2(0x1B, 7, inline(5), 0));
    w.extend([vopc(0xC5, ZERO, 7), sop1(0x3C, 12, VCC_LO)]);
    w.push(sop2(0x12, 12, EXEC_LO, 12));
    w.push(sop1(0x07, 13, VCC_LO));
    w.push(sop1(0x03, 14, 12));
    w.extend([vop1(0x001, 5, 12), vop1(0x001, 6, 13), vop1(0x001, 8, 14)]);
    w.push(sop2(0x10, EXEC_LO, EXEC_LO, 12));
    for (k, r) in [5u8, 6, 8].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    let shader = build(&w);
    let on = |x: u32| x & 5 != 0;
    let others = (0..32).filter(|&l| !on(l)).fold(0u32, |m, l| m | 1 << l);
    for (label, got) in run(&shader) {
        for x in 0..THREADS {
            let want = if on(x) { others } else { DEAD };
            let at = (x * SLOTS) as usize;
            assert_eq!(got[at..at + 3], [want; 3], "{label}: thread {x}");
        }
    }
}

#[test]
fn a_lane_mask_a_loop_carries_reads_as_the_lanes_it_still_holds() {
    // A waterfall loop over the lanes whose x & 5 is not 0, kept in s2
    // (not EXEC): each turn takes the first lane left (`s_ff1`), which
    // stores how many were left (`s_bcnt1`), and drops it from s2. The
    // lane of rank k stores n - k; the others keep their register.
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    w.push(vop2(0x1A, 9, inline((SLOTS * 4).trailing_zeros() as u16), 0));
    w.push(vop2(0x1B, 1, inline(31), 0));
    w.extend(mov_k(8, DEAD));
    w.push(vop2(0x1B, 12, inline(5), 0));
    w.push(vopc(0xC5, ZERO, 12));
    // s_and_b32 s2, vcc_lo, exec_lo
    w.push(0x8000_0000 | 14 << 23 | 2 << 16 | (EXEC_LO as u32) << 8 | VCC_LO as u32);
    let head = w.len();
    w.push(sop1(0x13, 4, 2));
    w.push(sop1(0x0F, 3, 2));
    w.push(vopc(0xC2, 4, 1));
    w.push(sop1(0x3C, 6, VCC_LO));
    w.push(vop1(0x001, 8, 3));
    w.push(sop1(0x03, EXEC_LO, 6));
    // s_lshl_b32 s5, 1, s4; s_andn2_b32 s2, s2, s5; s_cmp_lg_u32 s2, 0
    w.push(0x8000_0000 | 30 << 23 | 5 << 16 | 4 << 8 | inline(1) as u32);
    w.push(0x8000_0000 | 20 << 23 | 2 << 16 | 5 << 8 | 2);
    w.push(0xBF07_8002);
    w.push(sopp(0x05, head as i16 - w.len() as i16 - 1));
    w.extend(store(8, 9, 0));
    w.push(S_ENDPGM);
    let shader = build(&w);
    let on = |x: u32| x & 5 != 0;
    let n = (0..32).filter(|&l| on(l)).count() as u32;
    for (label, got) in run(&shader) {
        for x in 0..THREADS {
            let rank = (x & !31..x).filter(|&g| on(g)).count() as u32;
            let want = if on(x) { n - rank } else { DEAD };
            assert_eq!(got[(x * SLOTS) as usize], want, "{label}: thread {x}");
        }
    }
}

/// A loop over the lanes whose x & 5 is not 0, kept in s2: each turn the
/// lanes left (`s_and_saveexec_b32` of s2) count themselves in v8 and the
/// first one left (`s_ff1`) drops out. The lane of rank k counts k + 1;
/// the others 0. `carry` gives the words that set s2 to s7, the lanes left
/// for the next turn.
fn lanes_left_loop(carry: &[u32]) -> Vec<u32> {
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    w.push(vop2(0x1A, 9, inline((SLOTS * 4).trailing_zeros() as u16), 0));
    w.push(vop1(0x001, 8, ZERO));
    w.push(vop2(0x1B, 12, inline(5), 0));
    w.push(vopc(0xC5, ZERO, 12));
    // s_and_b32 s2, vcc_lo, exec_lo
    w.push(0x8000_0000 | 14 << 23 | 2 << 16 | (EXEC_LO as u32) << 8 | VCC_LO as u32);
    let head = w.len();
    w.push(sop1(0x13, 4, 2));
    w.push(sop1(0x3C, 6, 2));
    w.push(vop2(0x25, 8, inline(1), 8));
    w.push(sop1(0x03, EXEC_LO, 6));
    // s_lshl_b32 s5, 1, s4; s_andn2_b32 s7, s2, s5
    w.push(0x8000_0000 | 30 << 23 | 5 << 16 | 4 << 8 | inline(1) as u32);
    w.push(0x8000_0000 | 20 << 23 | 7 << 16 | 5 << 8 | 2);
    w.extend(carry);
    // s_cmp_lg_u32 s2, 0
    w.push(0xBF07_8002);
    w.push(sopp(0x05, head as i16 - w.len() as i16 - 1));
    w.extend(store(8, 9, 0));
    w.push(S_ENDPGM);
    w
}

fn check_lanes_left(shader: &HostShader) {
    let on = |x: u32| x & 5 != 0;
    for (label, got) in run(shader) {
        for x in 0..THREADS {
            let rank = (x & !31..x).filter(|&g| on(g)).count() as u32;
            let want = if on(x) { rank + 1 } else { 0 };
            assert_eq!(got[(x * SLOTS) as usize], want, "{label}: thread {x}");
        }
    }
}

#[test]
fn a_lane_mask_a_loop_carries_through_a_vgpr_keeps_its_lanes() {
    // The lanes left go round as the console compiler may carry them: into
    // v10 by `v_mov_b32` on two uniform paths (s7 itself, or a copy in s13)
    // and back by `v_readfirstlane_b32`.
    let carry = [
        // s_bitcmp0_b32 s4, 0; s_cbranch_scc1 +3
        0xBF0C_8004,
        sopp(0x05, 3),
        sop1(0x03, 13, 7),
        vop1(0x001, 10, 13),
        sopp(0x02, 1),
        vop1(0x001, 10, 7),
        vop1(0x002, 2, v(10)),
    ];
    check_lanes_left(&build(&lanes_left_loop(&carry)));
}

#[test]
fn a_lane_mask_a_loop_carries_as_a_number_keeps_its_lanes() {
    // The lanes left go round through a VGPR lane picked at run time
    // (`v_writelane_b32 v13, s7, s4`, `v_readlane_b32 s2, v13, s4`), as a
    // traversal stack holds them: a number, whose bits are console lanes.
    let carry = [0xD761_000D, 7 | 4 << 9, 0xD760_0002, v(13) as u32 | 4 << 9];
    check_lanes_left(&build(&lanes_left_loop(&carry)));
}

/// `v_cndmask_b32 vdst, src0, src1, sgpr` (VOP3).
const fn cndmask(vdst: u8, src0: u16, src1: u16, mask: u8) -> [u32; 2] {
    [
        0xD501_0000 | vdst as u32,
        src0 as u32 | (src1 as u32) << 9 | (mask as u32) << 18,
    ]
}

/// `buffer_store_short vdata, vaddr, s[8:11], 0 offen offset:byte`.
const fn store_short(vdata: u8, vaddr: u8, byte: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1A << 18 | 1 << 12 | (byte as u32 & 0xFFF),
        0x80 << 24 | 2 << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

#[test]
fn a_select_on_a_mask_held_as_a_number_tests_the_lane_s_own_bit() {
    // PrefixSum64's scan: s7 = 0xaaaaaaaa (the odd lanes) by `s_mov_b32`,
    // `v_cndmask_b32 v5, 0, v1, s7`; a backward permute makes the program
    // one of console lanes.
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    w.push(vop2(0x1A, 4, inline((SLOTS * 4).trailing_zeros() as u16), 0));
    w.extend([vop2(0x25, 1, 255, 0), 1000]);
    w.extend([sop1(0x03, 7, 255), 0xAAAA_AAAA]);
    w.extend(cndmask(5, ZERO, v(1), 7));
    w.push(vop2(0x1A, 2, inline(2), 0));
    w.extend(bpermute(6, 2, 1, 0));
    w.extend(store(5, 4, 0));
    w.extend(store(6, 4, 4));
    w.push(S_ENDPGM);
    if !crate::emit::mask_numbers() {
        let e = lower32(&w).err().expect("refused without VICTORY_MASK_NUMBERS");
        assert!(e.to_string().contains("a select on an untracked mask"), "{e}");
        return;
    }
    let shader = build(&w);
    for (label, got) in run(&shader) {
        for x in 0..THREADS {
            let odd = 0xAAAA_AAAAu32 >> (x & 31) & 1 != 0;
            let at = (x * SLOTS) as usize;
            assert_eq!(got[at], if odd { 1000 + x } else { 0 }, "{label}: thread {x}");
            assert_eq!(got[at + 1], 1000 + x, "{label}: thread {x}");
        }
    }
}

#[test]
fn short_stores_keep_the_other_half_of_their_dword() {
    // Thread x stores the short 1000 + x at byte 2x (two threads per dword,
    // at once), and at byte 256 + 4x over the dword 0xBEEFFFFF it stored
    // there itself.
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    w.extend([vop2(0x25, 1, 255, 0), 1000]);
    w.push(vop2(0x1A, 4, inline(1), 0));
    w.push(vop2(0x1A, 8, inline(2), 0));
    w.extend(mov_k(7, 0xBEEF_FFFF));
    w.extend(store(7, 8, 256));
    w.extend(store_short(1, 4, 0));
    w.extend(store_short(1, 8, 256));
    w.push(S_ENDPGM);
    if !crate::emit::subdword_stores() {
        let e = lower32(&w).err().expect("refused without VICTORY_SUBDWORD_STORES");
        assert!(e.to_string().contains("buffer_store_short"), "{e}");
        return;
    }
    let shader = build(&w);
    for (label, got) in run(&shader) {
        for k in 0..THREADS / 2 {
            let want = (1000 + 2 * k) | (1001 + 2 * k) << 16;
            assert_eq!(got[k as usize], want, "{label}: dword {k}");
        }
        for x in 0..THREADS {
            assert_eq!(got[64 + x as usize], 0xBEEF_0000 | (1000 + x), "{label}: thread {x}");
        }
    }
}

#[test]
fn nested_loops_share_the_loop_budget() {
    // 64 passes of a loop round 4096 passes of one that can wait on memory
    // (`s_dcache_inv`): each loop within its own limit, 262144 in all.
    // Each outer pass stores the count so far.
    let w = [
        sop1(0x03, 1, 0x80),
        0xF408_0200,
        0xFA00_0000,
        vop2(0x1A, 4, inline((SLOTS * 4).trailing_zeros() as u16), 0),
        sop1(0x03, 3, 0x80),
        sop1(0x03, 4, 0x80),
        sop1(0x03, 2, 0x80),
        0xF480_0000,
        0x0000_0000,
        0x8102_8102, // s_add_i32 s2, s2, 1
        0x8103_8103, // s_add_i32 s3, s3, 1
        0xBF0A_FF02, // s_cmp_lt_u32 s2, 4096
        4096,
        sopp(0x05, -7),
        vop1(0x001, 5, 3),
        store(5, 4, 0)[0],
        store(5, 4, 0)[1],
        0x8104_8104, // s_add_i32 s4, s4, 1
        0xBF0A_C004, // s_cmp_lt_u32 s4, 64
        sopp(0x05, -14),
        S_ENDPGM,
    ];
    let shader = build(&w);
    let budget = crate::emit::loop_budget_passes().min(crate::emit::loop_limit());
    for (label, got) in run(&shader) {
        for x in 0..THREADS {
            let n = got[(x * SLOTS) as usize];
            if crate::emit::loop_budget() {
                assert!(n <= budget && n + 2 * 4096 > budget, "{label}: thread {x} counted {n}");
            } else {
                assert_eq!(n, 64 * 4096, "{label}: thread {x}");
            }
        }
    }
}
