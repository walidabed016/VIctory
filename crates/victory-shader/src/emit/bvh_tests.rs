//! End-to-end tests of ray intersections (`image_bvh*_intersect_ray`) and
//! the acceleration structure memory around them: synthetic GFX10 programs
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP. The host builds no acceleration structure, so each is
//! taken to be empty: its memory reads 0 and every intersection misses its
//! node, so a traversal ends on its first step.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;
const LITERAL: u8 = 0xFF;
/// Inline constants: 0 and -1.
const ZERO: u16 = 128;
const ONES: u16 = 193;
/// Console threads, one wave32.
const THREADS: u32 = 32;
/// Dwords each thread stores.
const SLOTS: u32 = 8;

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

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// `s_load_dword* sdata, s[sbase:sbase+1], byte`, `op` 0 for one dword, 2
/// for four.
const fn smem(op: u8, sdata: u8, sbase: u8, byte: u32) -> [u32; 2] {
    [
        0xF400_0000 | (op as u32) << 18 | (sdata as u32) << 6 | (sbase as u32) >> 1,
        0xFA00_0000 | byte,
    ]
}

/// `buffer_store_dword vdata, vaddr, s[8:11], 0 offen offset:byte`.
const fn store(vdata: u8, vaddr: u8, byte: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte as u32 & 0xFFF),
        0x80 << 24 | 2 << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// `image_bvh_intersect_ray v[vdata:vdata+3], v[vaddr:vaddr+10], s[srsrc:srsrc+3]
/// dmask:0xf unorm r128`.
const fn intersect(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x3C << 26 | 0x66 << 18 | 1 << 15 | 1 << 12 | 0xF << 8 | 1,
        vaddr as u32 | (vdata as u32) << 8 | (srsrc as u32 >> 2) << 16,
    ]
}

/// A traversal as the shadow programs run one, each step taking the node's
/// closest child (a box node) or testing the triangle's distance as the
/// console compiler does (`t = num * rcp(den)`, a miss where `0 > t` or
/// `t >= tmax`), until no node is left. The acceleration structure's
/// descriptor is table entry 1; its root node is the header's dword at byte
/// 48 plus 5 (a box node) in even lanes, plus 0 (a triangle) in odd ones.
/// The ray's end is +inf. Each thread stores its steps, the last
/// intersection's four dwords, whether a triangle was hit and the header's
/// dword.
fn traversal() -> Vec<u32> {
    // s1 = 0; s[8:11] = the UAV; s[12:15] = the acceleration structure.
    let mut w = vec![sop1(0x03, 1, 0x80)];
    w.extend(smem(2, 8, 0, 0));
    w.extend(smem(2, 12, 0, 0x20));
    // v4 = v0 * SLOTS * 4. The header: s[16:17] = s[12:13] >> 8 (the base
    // in 256 bytes); s[18:19] = s[16:17] << 8 | 48; s20 = its dword.
    w.extend([
        vop2(0x1A, 4, inline((SLOTS * 4).trailing_zeros() as u16), 0),
        sop2(0x21, 16, 12, 128 + 8),
        sop2(0x1F, 18, 16, 128 + 8),
        sop2(0x10, 18, 18, 128 + 48),
    ]);
    w.extend(smem(0, 20, 18, 0));
    // s[24:27] = the intersection's descriptor: the base, then the console
    // compiler's fixed dwords.
    w.extend([
        sop1(0x03, 24, 16),
        sop1(0x03, 25, LITERAL),
        0x8000_0000,
        sop1(0x03, 26, ONES as u8),
        sop1(0x03, 27, LITERAL),
        0x8900_03FF,
    ]);
    // v1 = ((v0 & 1) ^ 1) * 5 + s20; v9 = +inf; v[10:18] = 0; v2 = v5 = 0.
    w.extend([
        vop2(0x1B, 1, inline(1), 0),
        vop2(0x1D, 1, inline(1), 1),
        vop2(0x0B, 1, inline(5), 1),
        vop2(0x25, 1, 20, 1),
        vop1(0x001, 9, u16::from(LITERAL)),
        0x7F80_0000,
    ]);
    w.extend((10..=18).map(|r| vop1(0x001, r, ZERO)));
    w.extend([vop1(0x001, 2, ZERO), vop1(0x001, 5, ZERO)]);
    // s28 = EXEC.
    w.push(sop1(0x03, 28, EXEC_LO));
    let step = w.len();
    // v2 += 1; v8 = v1; v[20:23] = the intersection.
    w.extend([vop2(0x25, 2, inline(1), 2), vop1(0x001, 8, v(1))]);
    w.extend(intersect(20, 8, 24));
    // v19 = v20 * rcp(v21); s30 = 0 > v19 | v19 >= v9 (a miss).
    w.extend([vop1(0x02A, 19, v(21)), vop2(0x07, 19, v(20), 19)]);
    w.extend(vop3(0x004, 30, ZERO, v(19), 0));
    w.extend([vopc(0x06, v(19), 9), sop2(0x10, 30, 30, VCC_LO)]);
    // vcc = a triangle ((v8 & 7) < 4); s31 = a hit there; v5 |= it; v1 =
    // -1 there, else the closest child.
    w.extend([
        vop2(0x1B, 24, inline(7), 8),
        vopc(0xC4, inline(4), 24),
        sop2(0x14, 31, VCC_LO, 30),
    ]);
    w.extend(vop3(0x101, 5, v(5), inline(1), 31));
    w.extend(vop3(0x101, 1, v(20), ONES, u16::from(VCC_LO)));
    // v_cmpx_ne_u32 -1, v1; s_cbranch_execnz step; EXEC = s28.
    w.push(vopc(0xD5, ONES, 1));
    w.push(sopp(0x09, step as i16 - w.len() as i16 - 1));
    w.push(sop1(0x03, EXEC_LO, 28));
    w.push(vop1(0x001, 6, 20));
    for (k, r) in [2u8, 20, 21, 22, 23, 5, 6].into_iter().enumerate() {
        w.extend(store(r, 4, 4 * k as u16));
    }
    w.push(S_ENDPGM);
    w
}

/// Lowers a wave32 compute program of one wave whose UAV table is at s0.
fn lower_cs(words: &[u32]) -> HostShader {
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
        psv: None,
    })
    .expect("lower");
    HostShader {
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
    }
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
fn a_traversal_of_an_empty_acceleration_structure_ends_on_its_first_step() {
    let shader = lower_cs(&traversal());
    for (label, got) in run(&shader) {
        for l in 0..THREADS {
            let at = (l * SLOTS) as usize;
            // Steps, the four dwords, a triangle hit, the header.
            let want = if l % 2 == 0 {
                [1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0]
            } else {
                [1, 0x7F80_0000, 0x3F80_0000, 0, 0, 0, 0]
            };
            assert_eq!(got[at..at + 7], want, "{label}: lane {l}");
        }
    }
}
