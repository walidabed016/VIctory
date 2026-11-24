//! End-to-end tests of wave64 compute programs on 32-lane halves
//! ([`super::halves`]): synthetic GFX10 wave64 programs lowered to DXIL,
//! validated by `dxil.dll` when present, and run on the host GPU (and WARP
//! when it takes `WaveSize(32)`). Each console thread stores what the
//! console's 64-lane waves give it, checked against a model of them.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, Waves, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const VCC_HI: u8 = 107;
const EXEC_LO: u8 = 126;
/// Inline constants: 0 and -1.
const ZERO: u16 = 128;
const ONES: u16 = 193;

const fn inline(n: u16) -> u16 {
    128 + n
}

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

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9,
    ]
}

/// `v_mov_b32_dpp vdst, vsrc` with DPP control `ctrl`, all rows and banks.
const fn mov_dpp(vdst: u8, vsrc: u8, ctrl: u16) -> [u32; 2] {
    [
        vop1(0x001, vdst, 0xFA),
        vsrc as u32 | (ctrl as u32) << 8 | 0xF << 24 | 0xF << 28,
    ]
}

const fn ds(op: u8, vdst: u8, addr: u8, data0: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18 | offset as u32,
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

/// Dwords each console thread stores.
const SLOTS: u32 = 16;

/// The start every program shares: the UAV descriptor in s[8:11], the
/// thread's flat index in v3 and its slots' byte address in v4.
fn prologue(threads: [u32; 3]) -> Vec<u32> {
    let [x, y, _] = threads;
    assert!(x.is_power_of_two() && y.is_power_of_two());
    // s1 = 0; s_load_dwordx4 s[8:11], s[0:1], 0.
    let mut w = vec![sop1(0x03, 1, 0x80), 0xF408_0200, 0xFA00_0000];
    w.extend([
        vop2(0x1A, 3, inline(x.trailing_zeros() as u16), 1),
        vop2(0x25, 3, v(3), 0),
        vop2(0x1A, 4, inline((x * y).trailing_zeros() as u16), 2),
        vop2(0x25, 3, v(3), 4),
        vop2(0x1A, 4, inline((SLOTS * 4).trailing_zeros() as u16), 3),
    ]);
    w
}

/// Lowers a wave64 compute program of `threads` (`lds` bytes of LDS)
/// whose UAV table is at s0; returns the shader, how its waves run and the
/// group barriers the halves add.
fn build(words: &[u32], threads: [u32; 3], lds: u32) -> (HostShader, Waves, u32) {
    let lowered = lower64(words, threads, lds);
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
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    (shader, lowered.waves, lowered.halves_barriers)
}

/// [`build`]'s lowering.
fn lower64(words: &[u32], threads: [u32; 3], lds: u32) -> lower::Lowered {
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
            // One user SGPR, thread ids x, y and z.
            rsrc2: 1 << 1 | 2 << 11 | lds.div_ceil(512) << 15,
            threads,
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
    .expect("lower")
}

/// Runs `shader` (one group) on the hardware GPU, which must take it, and
/// on WARP if it takes it; returns, per one that ran, the UAV's dwords.
fn run(shader: &HostShader, dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
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
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(dwords)],
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

/// Lane reads, ballots and lane counts of 64-lane waves: each console
/// thread stores
/// 0. its lane (`v_mbcnt` of all lanes),
/// 1. its index + 1,
/// 2. the count of its wave's lanes whose index has bit 0 or 3 set (`s_bcnt1`),
/// 3. how many of those lie below it (`v_mbcnt` of that mask),
/// 4. the first of them (`s_ff1`),
/// 5. lane 40's index (`v_readlane` of a constant lane),
/// 6. that first lane's index (`v_readlane` of a computed lane),
/// 7. the index of lane ^ 1 (`ds_swizzle`),
/// 8. the index of lane ^ 15 (DPP `row_mirror`),
/// 9. the index of the first lane from 37 on (`v_readfirstlane` under
///    EXEC).
fn lanes_program(threads: [u32; 3]) -> Vec<u32> {
    let mut w = prologue(threads);
    w.extend(vop3(0x365, 5, ONES, ZERO));
    w.extend(vop3(0x366, 5, ONES, v(5)));
    w.extend([
        vop2(0x25, 6, inline(1), 3),
        vop2(0x1B, 7, inline(9), 3),
        vopc(0xC5, ZERO, 7),
        sop1(0x10, 12, VCC_LO),
        vop1(0x001, 7, 12),
    ]);
    w.extend(vop3(0x365, 8, VCC_LO as u16, ZERO));
    w.extend(vop3(0x366, 8, VCC_HI as u16, v(8)));
    w.extend([sop1(0x14, 13, VCC_LO), vop1(0x001, 9, 13)]);
    w.extend(vop3(0x360, 14, v(3), inline(40)));
    w.extend(vop3(0x360, 15, v(3), 13));
    w.extend([vop1(0x001, 10, 14), vop1(0x001, 11, 15)]);
    w.extend(ds(0x35, 12, 3, 0, 0x041F));
    w.extend(mov_dpp(13, 3, 0x140));
    w.extend([
        vopc(0xC3, inline(37), 5),
        sop1(0x24, 16, VCC_LO),
        vop1(0x002, 18, v(3)),
        sop1(0x04, EXEC_LO, 16),
        vop1(0x001, 14, 18),
    ]);
    for (k, r) in [5u8, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        .into_iter()
        .enumerate()
    {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    w
}

/// What `lanes_program` stores on the console, per thread (`None`: not
/// defined there).
fn lanes_model(total: u32) -> Vec<[Option<u32>; 10]> {
    let set = |f: u32| f & 9 != 0;
    (0..total)
        .map(|f| {
            let (wave, lane) = (f / 64 * 64, f % 64);
            let lanes: Vec<u32> = (wave..total.min(wave + 64)).collect();
            let has = |l: u32| wave + l < total;
            let first = lanes.iter().position(|&g| set(g)).unwrap() as u32;
            [
                Some(lane),
                Some(f + 1),
                Some(lanes.iter().filter(|&&g| set(g)).count() as u32),
                Some(lanes.iter().filter(|&&g| g < f && set(g)).count() as u32),
                Some(first),
                has(40).then_some(wave + 40),
                Some(wave + first),
                Some(f ^ 1),
                Some(f ^ 15),
                has(37).then_some(wave + 37),
            ]
        })
        .collect()
}

fn check_lanes(threads: [u32; 3]) {
    let (shader, wave64, _) = build(&lanes_program(threads), threads, 0);
    assert_eq!(wave64, Waves::Halves, "{threads:?}");
    let total: u32 = threads.iter().product();
    let model = lanes_model(total);
    for (label, got) in run(&shader, total * SLOTS) {
        for (f, want) in model.iter().enumerate() {
            let at = f * SLOTS as usize;
            for (k, w) in want.iter().enumerate() {
                if let Some(w) = *w {
                    assert_eq!(got[at + k], w, "{label} {threads:?}: thread {f}, slot {k}");
                }
            }
        }
    }
}

#[test]
fn lanes_of_one_wave_in_a_2d_group_match_64_lane_waves() {
    // The RTX 4090 packs 2x2 quads of a 32x2 group into its waves.
    check_lanes([32, 2, 1]);
}

#[test]
fn lanes_of_several_waves_match_64_lane_waves() {
    check_lanes([8, 8, 4]);
    check_lanes([256, 1, 1]);
}

#[test]
fn a_last_wave_of_32_lanes_has_no_upper_half() {
    check_lanes([32, 1, 3]);
    check_lanes([32, 1, 1]);
}

/// `s_mov_b64 s[16:17], exec` and back; `v_cmpx_eq_u32 exec, 0, v0`.
const SAVE_EXEC: u32 = 0xBE90_047E;
const RESTORE_EXEC: u32 = 0xBEFE_0410;
const V_CMPX_EQ_U32_0_V0: u32 = 0x7DA4_0080;

#[test]
fn cross_lane_reads_after_lanes_split_and_in_a_uniform_loop_match_64_lane_waves() {
    // if (x == 0) v1 = 0; then four times: v5 += v3 of lane 40, v3 += 1000.
    let threads = [64, 2, 1];
    let mut w = prologue(threads);
    w.extend([
        SAVE_EXEC,
        V_CMPX_EQ_U32_0_V0,
        sopp(0x08, 1),
        vop1(0x001, 1, ZERO),
    ]);
    w.push(RESTORE_EXEC);
    w.extend([vop1(0x001, 5, ZERO), sop1(0x03, 20, inline(4) as u8)]);
    w.extend(vop3(0x360, 14, v(3), inline(40)));
    w.push(vop2(0x25, 5, 14, 5));
    w.extend([vop2(0x25, 3, 255, 3), 1000]);
    w.extend([
        sop2(0x01, 20, 20, inline(1) as u8),
        sopc(0x07, 20, 0x80),
        sopp(0x05, -8),
    ]);
    w.extend(store(5, 4, 0));
    w.push(S_ENDPGM);
    let (shader, waves, barriers) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Halves);
    // The counter's, and two for the exchange in the loop.
    assert_eq!(barriers, 3);
    for (label, got) in run(&shader, 128 * SLOTS) {
        for f in 0..128u32 {
            let want = 4 * (f / 64 * 64 + 40) + 6000;
            assert_eq!(got[(f * SLOTS) as usize], want, "{label}: thread {f}");
        }
    }
}

#[test]
fn a_cross_lane_read_some_lanes_skip_is_the_console_waves_in_turn() {
    // Two console waves, the second skipping the read: they never meet, so
    // they run in lockstep one after the other.
    let threads = [128, 1, 1];
    let mut w = prologue(threads);
    w.extend([SAVE_EXEC, V_CMPX_EQ_U32_0_V0, sopp(0x08, 3)]);
    w.extend(vop3(0x360, 14, v(3), inline(40)));
    w.extend([vop1(0x001, 5, 14), RESTORE_EXEC]);
    w.extend(store(5, 4, 0));
    w.push(S_ENDPGM);
    let (shader, waves, _) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Lockstep);
    for (label, got) in run(&shader, 128 * SLOTS) {
        assert_eq!(got[0], 40, "{label}: thread 0");
    }
}

#[test]
fn a_group_of_no_whole_host_waves_keeps_64_lane_waves() {
    let threads = [16, 3, 1];
    let mut w = prologue([16, 4, 1]);
    w.extend(vop3(0x360, 14, v(3), inline(40)));
    w.extend([vop1(0x001, 5, 14)]);
    w.extend(store(5, 4, 0));
    w.push(S_ENDPGM);
    let (_, wave64, _) = build(&w, threads, 0);
    assert_eq!(
        wave64,
        Waves::Wave64("a group of threads that is no multiple of 32 on 32-lane halves")
    );
}

#[test]
fn one_wave_lds_on_halves_is_in_program_order() {
    // Rounds of LDS[i] = i + 64r, then a read of LDS[i ^ 33], in the other
    // half, each read summed: the console compiler puts no barrier between
    // them in a one-wave group.
    const ROUNDS: u32 = 32;
    let threads = [32, 2, 1];
    let mut w = prologue(threads);
    w.push(vop2(0x1A, 5, inline(2), 3));
    w.push(vop2(0x1D, 6, inline(33), 3));
    w.push(vop2(0x1A, 6, inline(2), 6));
    w.push(vop1(0x001, 10, ZERO));
    for r in 0..ROUNDS {
        w.extend([vop2(0x25, 8, 255, 3), 64 * r]);
        w.extend(ds(0x0D, 0, 5, 8, 0));
        w.extend(ds(0x36, 9, 6, 0, 0));
        w.push(vop2(0x25, 10, v(9), 10));
    }
    w.extend(store(10, 4, 0));
    w.push(S_ENDPGM);
    let (shader, wave64, barriers) = build(&w, threads, 256);
    assert_eq!(wave64, Waves::Halves);
    // One before each LDS access (the host GPU need not show their lack).
    assert_eq!(barriers, 2 * ROUNDS);
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            let want = (0..ROUNDS).map(|r| (f ^ 33) + 64 * r).sum::<u32>();
            assert_eq!(got[(f * SLOTS) as usize], want, "{label}: thread {f}");
        }
    }
}

/// `v_lshl_add_u32 v7, s1, 8, v0` (s1 is 0).
const V_LSHL_ADD_V7: [u32; 2] = [0xD746_0007, 0x0401_1001];

/// Thread 0 alone runs a loop summing LDS[1..4] into a running prefix that
/// it moves through an SGPR (`v_readfirstlane`), as the console compiler
/// writes `if (index == 0) { for ... }`; then every thread stores
/// LDS[index % 4], and its `v_mbcnt` (which makes it a program of
/// console lanes). `compare` picks the threads that run the loop: `v7`
/// holds s1 * 256 + x.
fn serial_prefix(threads: [u32; 3], compare: &[u32]) -> Vec<u32> {
    let mut w = prologue(threads);
    w.extend(vop3(0x365, 13, ONES, ZERO));
    w.extend(V_LSHL_ADD_V7);
    w.extend([vop2(0x1A, 5, inline(2), 3), vop2(0x0B, 6, inline(3), 3)]);
    w.extend(ds(0x0D, 0, 5, 6, 0));
    w.push(sopp(0x0A, 0));
    w.extend(compare);
    w.push(sop1(0x24, 2, VCC_LO));
    let mut body = vec![
        sop2(0x1E, 6, 7, inline(2) as u8),
        sop2(0x02, 7, 7, inline(1) as u8),
        vop1(0x001, 10, 6),
        sopc(0x08, 7, inline(3) as u8),
    ];
    body.extend(ds(0x36, 9, 10, 0, 0));
    body.push(vop2(0x25, 9, 16, 9));
    body.extend(ds(0x0D, 0, 10, 9, 0));
    let mut region = vec![
        vop1(0x001, 9, ZERO),
        sop1(0x03, 16, 0x80),
        sop1(0x03, 7, inline(1) as u8),
        sopp(0x02, 1),
        vop1(0x002, 16, v(9)),
    ];
    let back = -(body.len() as i16 + 2);
    region.extend(body);
    region.push(sopp(0x04, back));
    w.push(sopp(0x08, region.len() as i16));
    w.extend(region);
    w.extend([sop1(0x04, EXEC_LO, 2), sopp(0x0A, 0)]);
    w.extend([vop2(0x1B, 11, inline(3), 3), vop2(0x1A, 11, inline(2), 11)]);
    w.extend(ds(0x36, 12, 11, 0, 0));
    w.extend(store(12, 4, 0));
    w.extend(store(13, 4, 4));
    w.push(S_ENDPGM);
    w
}

#[test]
fn one_lane_of_each_wave_reads_its_own_first_lane_on_halves() {
    // Two console waves; the first has the one lane where x == 0.
    let threads = [128, 1, 1];
    let (shader, waves, _) = build(
        &serial_prefix(threads, &[vopc(0xC2, ZERO, 7)]),
        threads,
        512,
    );
    assert_eq!(waves, Waves::Halves);
    for (label, got) in run(&shader, 128 * SLOTS) {
        for f in 0..128u32 {
            let at = (f * SLOTS) as usize;
            let want = [0, 3, 9, 18][(f % 4) as usize];
            assert_eq!(got[at], want, "{label}: thread {f}");
            assert_eq!(got[at + 1], (f % 64).min(32), "{label}: thread {f}");
        }
    }
}

#[test]
fn an_x_index_repeated_in_a_wave_keeps_64_lane_waves() {
    // x == 0 holds in lanes 0 and 32 of a 32x4 group's first wave.
    let threads = [32, 4, 1];
    let (_, waves, _) = build(
        &serial_prefix(threads, &[vopc(0xC2, ZERO, 7)]),
        threads,
        512,
    );
    assert_eq!(
        waves,
        Waves::Wave64("a cross-lane value where the two 32-lane halves do not meet")
    );
}

#[test]
fn lanes_that_share_a_first_lane_keep_64_lane_waves() {
    // x % 32 == 0 holds in two lanes of each console wave.
    let threads = [128, 1, 1];
    let compare = [vop2(0x1B, 8, inline(31), 7), vopc(0xC2, ZERO, 8)];
    let (_, waves, _) = build(&serial_prefix(threads, &compare), threads, 512);
    assert_eq!(
        waves,
        Waves::Wave64("a cross-lane value where the two 32-lane halves do not meet")
    );
}

/// Each thread adds 2 to v5 x % 4 times in a loop of its own (as Sort's
/// per-thread item loops), then, after the program's `s_barrier` or with
/// none, adds lane 31's flat index (`v_readlane`, the lower half's).
fn after_a_lane_loop(threads: [u32; 3], barrier: bool) -> Vec<u32> {
    let mut w = prologue(threads);
    w.extend([
        vop1(0x001, 5, ZERO),
        vop2(0x1B, 6, inline(3), 3),
        SAVE_EXEC,
        vopc(0xD5, ZERO, 6),
        sopp(0x08, 4),
        vop2(0x25, 5, inline(2), 5),
        vop2(0x27, 6, inline(1), 6),
        vopc(0xD5, ZERO, 6),
        sopp(0x09, -4),
        RESTORE_EXEC,
    ]);
    if barrier {
        w.push(sopp(0x0A, 0));
    }
    w.extend(vop3(0x360, 2, v(3), inline(31)));
    w.push(vop2(0x25, 7, 2, 5));
    w.extend(store(7, 4, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn a_cross_lane_read_by_the_programs_barrier_after_a_lane_loop_runs_on_halves() {
    let threads = [128, 1, 1];
    let (shader, waves, barriers) = build(&after_a_lane_loop(threads, true), threads, 0);
    assert_eq!(waves, Waves::Halves);
    // The counter's and the exchange's.
    assert_eq!(barriers, 2);
    for (label, got) in run(&shader, 128 * SLOTS) {
        for f in 0..128u32 {
            let want = f / 64 * 64 + 31 + 2 * (f % 4);
            assert_eq!(got[(f * SLOTS) as usize], want, "{label}: thread {f}");
        }
    }
}

#[test]
fn a_cross_lane_read_after_a_lane_loop_with_no_barrier_is_the_console_waves_in_turn() {
    // The halves do not meet after the loop, but the two console waves
    // never meet either: they run in lockstep one after the other.
    let threads = [128, 1, 1];
    let (shader, waves, _) = build(&after_a_lane_loop(threads, false), threads, 0);
    assert_eq!(waves, Waves::Lockstep);
    for (label, got) in run(&shader, 128 * SLOTS) {
        for f in 0..128u32 {
            let want = f / 64 * 64 + 31 + 2 * (f % 4);
            assert_eq!(got[(f * SLOTS) as usize], want, "{label}: thread {f}");
        }
    }
}

/// Each thread writes index + 1000 to LDS[index] and index + 2000 to a
/// dword past the program's 128, LDS[index + 128] as CS_SortRayTiles does
/// from its third group on (it indexes LDS by the thread's id in the
/// dispatch), or with `near` LDS[index % 4 + 128]; after a barrier it
/// stores LDS[index], that dword and what `ds_add_rtn` returns there. The
/// console drops writes past the allocation and reads 0 there. `lanes`
/// adds a `v_mbcnt` (a program of console lanes, on halves).
fn past_the_lds(threads: [u32; 3], lanes: bool, near: bool) -> Vec<u32> {
    let mut w = prologue(threads);
    if lanes {
        w.extend(vop3(0x365, 13, ONES, ZERO));
    }
    w.push(vop2(0x1A, 5, inline(2), 3));
    if near {
        w.push(vop2(0x1B, 6, inline(3), 3));
        w.push(vop2(0x1A, 6, inline(2), 6));
        w.extend([vop2(0x25, 6, 255, 6), 512]);
    } else {
        w.extend([vop2(0x25, 6, 255, 5), 512]);
    }
    w.extend([vop2(0x25, 7, 255, 3), 1000]);
    w.extend([vop2(0x25, 12, 255, 3), 2000]);
    w.extend(ds(0x0D, 0, 5, 7, 0));
    w.extend(ds(0x0D, 0, 6, 12, 0));
    w.push(sopp(0x0A, 0));
    w.extend(ds(0x36, 8, 5, 0, 0));
    w.extend(ds(0x36, 9, 6, 0, 0));
    w.push(vop1(0x001, 10, inline(1)));
    w.extend(ds(0x20, 11, 6, 10, 0));
    for (k, r) in [8u8, 9, 11].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    w
}

fn check_past_the_lds(lanes: bool, near: bool, want: Waves) {
    let threads = [128, 1, 1];
    let (shader, waves, _) = build(&past_the_lds(threads, lanes, near), threads, 512);
    assert_eq!(waves, want);
    for (label, got) in run(&shader, 128 * SLOTS) {
        for f in 0..128u32 {
            let at = (f * SLOTS) as usize;
            assert_eq!(got[at..at + 3], [f + 1000, 0, 0], "{label}: thread {f}");
        }
    }
}

#[test]
fn lds_past_the_allocation_reads_0_and_takes_no_writes() {
    check_past_the_lds(false, false, Waves::Any);
}

#[test]
fn lds_past_the_allocation_reads_0_and_takes_no_writes_on_halves() {
    check_past_the_lds(true, false, Waves::Halves);
}

#[test]
fn lds_just_past_the_allocation_reads_0_and_takes_no_writes_on_halves() {
    // Within the host's group-shared memory (the halves' own words, or its
    // rounding): unchecked, this reads what was written instead of faulting.
    check_past_the_lds(true, true, Waves::Halves);
}

/// `v_mov_b32 vdst, k` (a literal).
const fn mov_k(vdst: u8, k: u32) -> [u32; 2] {
    [vop1(0x001, vdst, 255), k]
}

/// A branch of SOPP opcode `op` past `region`, then the region.
fn skip(op: u8, region: Vec<u32>) -> Vec<u32> {
    let mut w = vec![sopp(op, region.len() as i16)];
    w.extend(region);
    w
}

/// SOPP opcodes: `s_cbranch_scc0`, `s_cbranch_execz`, `s_cbranch_execnz`.
const SCC0: u8 = 0x04;
const EXECZ: u8 = 0x08;
const EXECNZ: u8 = 0x09;
/// What a lane outside EXEC keeps.
const DEAD: u32 = 0xDEAD;

/// Each thread's first `n` stored dwords.
fn slots(got: &[u32], f: u32, n: usize) -> &[u32] {
    let at = (f * SLOTS) as usize;
    &got[at..at + n]
}

#[test]
fn masks_written_where_lanes_split_are_the_console_waves_in_lockstep() {
    // Lanes with bit 0 or 3 of their index go in (`s_and_saveexec`, SCC);
    // each there stores how many of them have bit 1, how many of those lie
    // below it, and the first (`s_bcnt1`, `v_mbcnt`, `s_ff1` of a mask
    // written there; bit 1 is taken before, in every lane).
    let threads = [32, 2, 1];
    let mut w = prologue(threads);
    for r in [5, 6, 7] {
        w.extend(mov_k(r, DEAD));
    }
    w.extend([
        vop2(0x1B, 8, inline(9), 3),
        vop2(0x1B, 9, inline(2), 3),
        vopc(0xC5, ZERO, 8),
        sop1(0x24, 16, VCC_LO),
    ]);
    let mut region = vec![
        vopc(0xC5, ZERO, 9),
        sop1(0x10, 12, VCC_LO),
        vop1(0x001, 5, 12),
    ];
    region.extend(vop3(0x365, 6, VCC_LO as u16, ZERO));
    region.extend(vop3(0x366, 6, VCC_HI as u16, v(6)));
    region.extend([sop1(0x14, 13, VCC_LO), vop1(0x001, 7, 13)]);
    w.extend(skip(SCC0, region));
    w.push(RESTORE_EXEC);
    for (k, r) in [5u8, 6, 7].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    let (shader, waves, _) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Lockstep);
    let inside = |f: u32| f & 9 != 0;
    let marked: Vec<u32> = (0..64).filter(|&f| inside(f) && f & 2 != 0).collect();
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            let want = if inside(f) {
                let below = marked.iter().filter(|&&g| g < f).count() as u32;
                [marked.len() as u32, below, marked[0]]
            } else {
                [DEAD; 3]
            };
            assert_eq!(slots(&got, f, 3), want, "{label}: thread {f}");
        }
    }
}

#[test]
fn reads_of_other_lanes_where_only_the_upper_half_runs_are_the_console_waves() {
    // Lanes 40-63 go in (`v_cmpx`, `s_cbranch_execz`), the lower half none
    // of them; there each stores the first lane's v11, lane 5's (outside
    // EXEC), lane ^ 8's through `ds_swizzle` and lane - 1's through DPP
    // `row_shr:1`, where a lane outside EXEC gives 0 and no write.
    let threads = [64, 1, 1];
    let mut w = prologue(threads);
    for r in [5, 6, 7, 10] {
        w.extend(mov_k(r, DEAD));
    }
    w.extend([vop2(0x25, 11, 255, 3), 256]);
    w.extend([SAVE_EXEC, vopc(0xD3, inline(40), 3)]);
    let mut region = vec![vop1(0x002, 14, v(11))];
    region.extend(vop3(0x360, 15, v(11), inline(5)));
    region.extend([vop1(0x001, 5, 14), vop1(0x001, 6, 15)]);
    region.extend(ds(0x35, 7, 11, 0, 0x201F));
    region.extend(mov_dpp(10, 11, 0x111));
    w.extend(skip(EXECZ, region));
    w.push(RESTORE_EXEC);
    for (k, r) in [5u8, 6, 7, 10].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    let (shader, waves, _) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Lockstep);
    let on = |f: u32| f >= 40;
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            let want = if on(f) {
                let swizzled = if on(f ^ 8) { 256 + (f ^ 8) } else { 0 };
                let shifted = if f % 16 >= 1 && on(f - 1) {
                    256 + f - 1
                } else {
                    DEAD
                };
                [256 + 40, 256 + 5, swizzled, shifted]
            } else {
                [DEAD; 4]
            };
            assert_eq!(slots(&got, f, 4), want, "{label}: thread {f}");
        }
    }
}

#[test]
fn a_loop_taking_lanes_by_key_in_turn_is_the_console_waves() {
    // Each turn takes the lanes whose key is the first remaining lane's
    // (`v_readfirstlane`, `v_cmp`, `s_and_saveexec`, `s_xor` EXEC,
    // `s_cbranch_execnz`); each stores its turn, the key taken and how
    // many lanes took it (`s_bcnt1`).
    let threads = [64, 1, 1];
    let key = |f: u32| (f * 13) & 7;
    let mut w = prologue(threads);
    w.push(vop2(0x0B, 11, inline(13), 3));
    w.push(vop2(0x1B, 11, inline(7), 11));
    w.extend(mov_k(12, DEAD));
    w.extend([sop1(0x03, 20, 0x80), SAVE_EXEC]);
    let body = vec![
        vop1(0x002, 14, v(11)),
        vopc(0xC2, 14, 11),
        sop1(0x24, 18, VCC_LO),
        vop1(0x001, 5, 20),
        vop1(0x001, 12, 14),
        sop1(0x10, 21, VCC_LO),
        vop1(0x001, 13, 21),
        sop2(0x13, EXEC_LO, EXEC_LO, 18),
        sop2(0x00, 20, 20, inline(1) as u8),
    ];
    let back = -(body.len() as i16 + 1);
    w.extend(body);
    w.extend([sopp(EXECNZ, back), RESTORE_EXEC]);
    for (k, r) in [5u8, 12, 13].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    let (shader, waves, _) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Lockstep);
    let mut want = [[0; 3]; 64];
    let mut left: Vec<u32> = (0..64).collect();
    let mut turn = 0;
    while let Some(&first) = left.first() {
        let k = key(first);
        let taken: Vec<u32> = left.iter().copied().filter(|&f| key(f) == k).collect();
        for &f in &taken {
            want[f as usize] = [turn, k, taken.len() as u32];
        }
        left.retain(|&f| key(f) != k);
        turn += 1;
    }
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            assert_eq!(slots(&got, f, 3), want[f as usize], "{label}: thread {f}");
        }
    }
}

#[test]
fn one_wave_lds_written_where_lanes_split_is_in_program_order() {
    // LDS[i] = 0; lanes 40-63 then write LDS[i] = i + 1000 (`v_cmpx`,
    // `s_cbranch_execz`); each lane reads LDS[i ^ 33], mostly the other
    // half's.
    let threads = [64, 1, 1];
    let mut w = prologue(threads);
    w.push(vop2(0x1A, 5, inline(2), 3));
    w.push(vop2(0x1D, 6, inline(33), 3));
    w.push(vop2(0x1A, 6, inline(2), 6));
    w.extend([vop2(0x25, 8, 255, 3), 1000]);
    w.push(vop1(0x001, 9, ZERO));
    w.extend(ds(0x0D, 0, 5, 9, 0));
    w.extend([SAVE_EXEC, vopc(0xD3, inline(40), 3)]);
    w.extend(skip(EXECZ, ds(0x0D, 0, 5, 8, 0).to_vec()));
    w.push(RESTORE_EXEC);
    w.extend(ds(0x36, 9, 6, 0, 0));
    w.extend(store(9, 4, 0));
    w.push(S_ENDPGM);
    let (shader, waves, barriers) = build(&w, threads, 256);
    assert_eq!(waves, Waves::Lockstep);
    // The counter's, the `v_cmpx` ballot's and one before each LDS access
    // (the host GPU need not show their lack).
    assert_eq!(barriers, 5);
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            let g = f ^ 33;
            let want = if g >= 40 { g + 1000 } else { 0 };
            assert_eq!(got[(f * SLOTS) as usize], want, "{label}: thread {f}");
        }
    }
}

#[test]
fn the_first_lane_of_no_lanes_is_lane_0() {
    // `v_readfirstlane` with EXEC zero reads lane 0; each thread stores
    // that and its lane (`v_mbcnt`, which needs the console's lanes).
    let threads = [64, 1, 1];
    let mut w = prologue(threads);
    w.extend([vop2(0x25, 11, 255, 3), 256]);
    w.extend([
        SAVE_EXEC,
        vopc(0xD4, ZERO, 3),
        vop1(0x002, 14, v(11)),
        RESTORE_EXEC,
        vop1(0x001, 5, 14),
    ]);
    w.extend(vop3(0x365, 6, ONES, ZERO));
    w.extend(vop3(0x366, 6, ONES, v(6)));
    w.extend(store(5, 4, 0));
    w.extend(store(6, 4, 4));
    w.push(S_ENDPGM);
    let (shader, waves, _) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Halves);
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            assert_eq!(slots(&got, f, 2), [256, f], "{label}: thread {f}");
        }
    }
}

#[test]
fn lanes_outside_exec_take_slot_0_in_a_waterfall_in_lockstep() {
    // Lanes 40-63 run a console waterfall: v7 = (lane & 1) * 32, the first
    // lane's table offset (`v_readfirstlane`), the lanes with the same one
    // (`v_cmpx_eq`), its entry loaded (`s_load` at that offset) and read
    // through (`buffer_load_dword`). A cross-lane read of lane 5 there keeps
    // the wave in lockstep, where every thread runs the loop's accesses:
    // lanes 0-39, whose v7 still holds 0x7FFF_FFE0, must take slot 0, not
    // slot 0x3FF_FFFF (decal classification, runs 54 and 56).
    let threads = [64, 1, 1];
    let mut w = prologue(threads);
    w.extend(mov_k(7, 0x7FFF_FFE0));
    w.extend(mov_k(9, DEAD));
    w.extend([SAVE_EXEC, vopc(0xD3, inline(40), 3)]);
    let mut region = vec![vop2(0x1B, 7, inline(1), 3), vop2(0x1A, 7, inline(5), 7)];
    region.extend(vop3(0x360, 15, v(3), inline(5)));
    region.extend([
        vop1(0x001, 6, 15),
        0xBE9A_047E, // s_mov_b64 s[26:27], exec
        0xBE96_047E, // s_mov_b64 s[22:23], exec
    ]);
    let looped = [
        vop1(0x002, 20, v(7)),
        vopc(0xD2, 20, 7), // v_cmpx_eq_u32 exec, s20, v7
        0xF408_0300,
        0x2800_0000, // s_load_dwordx4 s[12:15], s[0:1], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_0000,
        0x8003_0900, // buffer_load_dword v9, off, s[12:15], 0
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE96_3A16, // s_andn2_wrexec_b64 s[22:23], s[22:23]
    ];
    region.extend(looped);
    region.extend([
        sopp(EXECNZ, -(looped.len() as i16 + 1)),
        0xBEFE_041A, // s_mov_b64 exec, s[26:27]
    ]);
    w.extend(skip(EXECZ, region));
    w.push(RESTORE_EXEC);
    w.extend(store(6, 4, 0));
    w.extend(store(9, 4, 4));
    w.push(S_ENDPGM);
    let lowered = lower64(&w, threads, 0);
    assert_eq!(lowered.waves, Waves::Lockstep);
    let outside = lowered.nonuniform_outside_exec;
    assert!(!outside.is_empty(), "no per-lane heap index");
    assert!(
        outside.iter().all(|&i| i == 0),
        "slots outside EXEC: {outside:x?}"
    );
}

/// Lanes whose index in their console wave is at least 24 times the
/// wave's index go in (`v_cmpx`, `s_cbranch_execz`): all of wave 0, lanes
/// 24-63 of wave 1, the upper half alone of wave 2 and none of wave 3,
/// which skips the region. Each there stores how many of them have bit 1
/// (`s_bcnt1` of a mask written there), how many of those lie below it
/// (`v_mbcnt`), the first (`s_ff1`), the first lane's v11
/// (`v_readfirstlane`) and lane 50's (`v_readlane`). With `barrier`, an
/// `s_barrier` first.
fn waves_apart(threads: [u32; 3], barrier: bool) -> Vec<u32> {
    let mut w = prologue(threads);
    if barrier {
        w.push(sopp(0x0A, 0));
    }
    for r in [5, 6, 7, 10, 12] {
        w.extend(mov_k(r, DEAD));
    }
    w.extend([vop2(0x25, 11, 255, 3), 256]);
    w.extend([
        vop2(0x16, 8, inline(6), 3),
        vop2(0x0B, 8, inline(24), 8),
        vop2(0x1B, 9, inline(63), 3),
        vop2(0x1B, 13, inline(2), 3),
        SAVE_EXEC,
        vopc(0xD3, v(8), 9),
    ]);
    let mut region = vec![
        vopc(0xC5, ZERO, 13),
        sop1(0x10, 12, VCC_LO),
        vop1(0x001, 5, 12),
    ];
    region.extend(vop3(0x365, 6, VCC_LO as u16, ZERO));
    region.extend(vop3(0x366, 6, VCC_HI as u16, v(6)));
    region.extend([sop1(0x14, 13, VCC_LO), vop1(0x001, 7, 13)]);
    region.extend([vop1(0x002, 14, v(11)), vop1(0x001, 10, 14)]);
    region.extend(vop3(0x360, 15, v(11), inline(50)));
    region.push(vop1(0x001, 12, 15));
    w.extend(skip(EXECZ, region));
    w.push(RESTORE_EXEC);
    for (k, r) in [5u8, 6, 7, 10, 12].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    w
}

fn check_waves_apart(threads: [u32; 3]) {
    let (shader, waves, _) = build(&waves_apart(threads, false), threads, 0);
    assert_eq!(waves, Waves::Lockstep, "{threads:?}");
    let total: u32 = threads.iter().product();
    let inside = |f: u32| f % 64 >= 24 * (f / 64);
    for (label, got) in run(&shader, total * SLOTS) {
        for f in 0..total {
            let wave = f / 64 * 64;
            let want = if inside(f) {
                let lanes = || (wave..wave + 64).filter(|&g| inside(g));
                let marked: Vec<u32> = lanes().filter(|&g| g & 2 != 0).collect();
                let below = marked.iter().filter(|&&g| g < f).count() as u32;
                let first = lanes().next().unwrap();
                [
                    marked.len() as u32,
                    below,
                    marked[0] % 64,
                    256 + first,
                    256 + wave + 50,
                ]
            } else {
                [DEAD; 5]
            };
            assert_eq!(slots(&got, f, 5), want, "{label} {threads:?}: thread {f}");
        }
    }
}

#[test]
fn waves_of_a_group_that_branch_apart_are_the_console_waves_in_turn() {
    check_waves_apart([16, 16, 1]);
    check_waves_apart([8, 16, 1]);
}

#[test]
fn waves_of_a_group_with_a_barrier_keep_64_lane_waves() {
    let threads = [16, 16, 1];
    let (_, waves, _) = build(&waves_apart(threads, true), threads, 0);
    assert_eq!(
        waves,
        Waves::Wave64("a lane mask read as data where the two 32-lane halves do not meet")
    );
}

/// `op_dpp vdst, vsrc0, vsrc1` (VOP2) with DPP8 lane selects `sel`.
const fn dpp8(op: u8, vdst: u8, vsrc0: u8, vsrc1: u8, sel: [u32; 8]) -> [u32; 2] {
    let mut s = 0;
    let mut i = 0;
    while i < 8 {
        s |= sel[i] << (3 * i);
        i += 1;
    }
    [vop2(op, vdst, 0xE9, vsrc1), vsrc0 as u32 | s << 8]
}

/// `op_dpp vdst, vsrc0, vsrc1` (VOP2) with DPP control `ctrl`,
/// `bound_ctrl:1`, all rows and banks.
const fn dpp16(op: u8, vdst: u8, vsrc0: u8, vsrc1: u8, ctrl: u16) -> [u32; 2] {
    [
        vop2(op, vdst, 0xFA, vsrc1),
        vsrc0 as u32 | (ctrl as u32) << 8 | 1 << 19 | 0xF << 24 | 0xF << 28,
    ]
}

/// A VOP3 instruction of three sources.
const fn vop3_3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

const EXEC_HI: u8 = 127;

/// A whole-wave reduction as the console compiler expands it (EXEC every
/// lane, saved in s[20:21]): VOP2 `op` over v`src` in the lanes of s[24:25]
/// and the saved EXEC, from `ident` (in v2 and v3), to s`dst`.
fn wave_reduction(op: u8, ident: u32, src: u8, dst: u8) -> Vec<u32> {
    let mut w = vec![sop1(0x25, 20, ONES as u8)];
    w.extend(mov_k(2, ident));
    w.push(sop2(0x0F, 22, 24, 20));
    w.extend(vop3_3(0x101, 2, v(2), v(src), 22));
    for sel in [
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 2, 0, 0, 6, 6, 4, 4],
        [4, 4, 4, 4, 0, 0, 0, 0],
    ] {
        w.extend(dpp8(op, 2, 2, 2, sel));
    }
    // row_mirror, then the other row of the half, then the other half.
    w.extend(dpp16(op, 2, 2, 2, 0x140));
    w.extend(vop3_3(0x378, 3, v(2), ZERO, ZERO));
    w.push(vop2(op, 2, v(3), 2));
    w.extend(vop3(0x360, 15, v(2), inline(32)));
    w.push(vop2(op, 2, 15, 2));
    w.extend(vop3(0x360, dst, v(2), ZERO));
    w.push(sop1(0x04, EXEC_LO, 20));
    w
}

/// An exclusive whole-wave prefix sum as the console compiler expands it,
/// of v`src` in the lanes of s[24:25] and the saved EXEC, to v2 (with v3
/// and v7, and s14): `row_shr` within rows, the row below through
/// `v_permlanex16` under a constant EXEC, the lower half through
/// `v_readlane`, less the lane's own term.
fn wave_prefix_sum(src: u8) -> Vec<u32> {
    let mut w = vec![
        sop1(0x25, 20, ONES as u8),
        vop1(0x001, 2, ZERO),
        sop2(0x0F, 22, 24, 20),
    ];
    w.extend(vop3_3(0x101, 7, v(2), v(src), 22));
    w.push(vop1(0x001, 2, v(7)));
    for n in [1, 2, 4, 8] {
        w.extend(dpp16(0x25, 2, 2, 2, 0x110 + n));
    }
    w.extend(vop3_3(0x378, 3, v(2), ONES, ONES));
    w.extend([sop1(0x03, EXEC_LO, 255), 0xFFFF_0000]);
    w.extend([sop1(0x03, EXEC_HI, 255), 0xFFFF_0000]);
    w.push(vop2(0x25, 2, v(3), 2));
    w.extend(vop3(0x360, 14, v(2), inline(31)));
    w.extend([
        sop1(0x03, EXEC_LO, ZERO as u8),
        sop1(0x03, EXEC_HI, ONES as u8),
    ]);
    w.push(vop2(0x25, 2, 14, 2));
    w.push(sop1(0x04, EXEC_LO, ONES as u8));
    w.push(vop2(0x26, 2, v(2), 7));
    w.push(sop1(0x04, EXEC_LO, 20));
    w
}

#[test]
fn whole_wave_reductions_and_prefixes_where_lanes_split_are_the_console_waves() {
    // Lanes whose index has bit 0 or 3 set go in (`v_cmpx`,
    // `s_cbranch_execz`); there whole-wave sections over the lanes of a mask
    // (index bit 1 clear) and the saved EXEC give each the sum of 3f + 1,
    // its exclusive prefix sum and the float minimum of f ^ 42.
    let threads = [64, 1, 1];
    let mut w = prologue(threads);
    w.extend([vop2(0x0B, 11, inline(3), 3), vop2(0x25, 11, inline(1), 11)]);
    w.extend([vop2(0x1D, 14, inline(42), 3), vop1(0x006, 14, v(14))]);
    w.extend([
        vop2(0x1B, 12, inline(2), 3),
        vopc(0xC2, ZERO, 12),
        sop1(0x04, 24, VCC_LO),
    ]);
    for r in [5, 6, 8] {
        w.extend(mov_k(r, DEAD));
    }
    w.push(vop2(0x1B, 13, inline(9), 3));
    w.extend([SAVE_EXEC, vopc(0xD5, ZERO, 13)]);
    let mut region = wave_reduction(0x25, 0, 11, 26);
    region.push(vop1(0x001, 5, 26));
    region.extend(wave_prefix_sum(11));
    region.push(vop1(0x001, 6, v(2)));
    region.extend(wave_reduction(0x0F, 0x7F80_0000, 14, 27));
    region.push(vop1(0x001, 8, 27));
    w.extend(skip(EXECZ, region));
    w.push(RESTORE_EXEC);
    for (k, r) in [5u8, 6, 8].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    let (shader, waves, _) = build(&w, threads, 0);
    assert_eq!(waves, Waves::Lockstep);
    let inside = |f: u32| f & 9 != 0;
    let counted: Vec<u32> = (0..64).filter(|&f| inside(f) && f & 2 == 0).collect();
    let sum = |below: u32| -> u32 {
        counted
            .iter()
            .filter(|&&g| g < below)
            .map(|&g| 3 * g + 1)
            .sum()
    };
    let min = counted
        .iter()
        .map(|&g| (g ^ 42) as f32)
        .fold(f32::INFINITY, f32::min)
        .to_bits();
    for (label, got) in run(&shader, 64 * SLOTS) {
        for f in 0..64u32 {
            let want = if inside(f) {
                [sum(64), sum(f), min]
            } else {
                [DEAD; 3]
            };
            assert_eq!(slots(&got, f, 3), want, "{label}: thread {f}");
        }
    }
}
