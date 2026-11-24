//! End-to-end tests of `ds_bpermute_b32` in wave32 compute programs:
//! synthetic GFX10 programs lowered to DXIL, validated by `dxil.dll` when
//! present, and run on the host GPU (and WARP when it takes
//! `WaveSize(32)`). Each thread stores what the console's wave gives it.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
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
/// is at s0, and validates it.
fn build(words: &[u32]) -> HostShader {
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
    })
    .expect("lower");
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
