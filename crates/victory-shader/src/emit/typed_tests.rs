//! End-to-end tests of typed buffer stores and loads (`buffer_store_format_*`,
//! `buffer_load_format_*`) through typed views in 32-bit formats.
//! Synthetic GFX10 programs lowered to DXIL, validated by `dxil.dll` when
//! present, and run on the host GPU and WARP.

use windows::Win32::Graphics::Dxgi::Common::*;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const LANES: u32 = 32;

/// Lowers a wave32 program of 32 lanes with the resource heap at s0 and a
/// table at s1 whose entry 0 is a UAV.
fn build(words: &[u32]) -> HostShader {
    let param = |root, user_data: u8| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![param(RootType::ResourceHeap, 0), param(RootType::Table, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 1,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [LANES, 1, 1],
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

/// Runs `shader` with `views` at heap indices 0.. on the hardware GPU and on
/// WARP (each skipped without a device): per one that ran, each UAV's
/// dwords in heap order.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // The table starts at heap index 0.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, uavs));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

/// CS_ClearShadowMask's pattern on the table's typed UAV: zeros element 4,
/// copies element 1 to element 2 (`buffer_load_format_x`, then the loaded
/// dword in all four channels of `buffer_store_format_xyzw`) and stores the
/// bits of 0.5 at element 3.
const CLEAR_PATTERN: &[u32] = &[
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0x7E02_0280, // v_mov_b32 v1, 0
    0x7E04_0280, // v_mov_b32 v2, 0
    0x7E06_0280, // v_mov_b32 v3, 0
    0x7E08_0280, // v_mov_b32 v4, 0
    0x7E0A_0284, // v_mov_b32 v5, 4
    0x7E0C_0281, // v_mov_b32 v6, 1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE01C_2000,
    0x8003_0105, // buffer_store_format_xyzw v[1:4], v5, s[12:15], 0 idxen
    0xE000_2000,
    0x8003_0706, // buffer_load_format_x v7, v6, s[12:15], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E10_0307, // v_mov_b32 v8, v7
    0x7E12_0307, // v_mov_b32 v9, v7
    0x7E14_0307, // v_mov_b32 v10, v7
    0x7E16_0307, // v_mov_b32 v11, v7
    0x7E18_0282, // v_mov_b32 v12, 2
    0xE01C_2000,
    0x8003_080C, // buffer_store_format_xyzw v[8:11], v12, s[12:15], 0 idxen
    0x7E1A_02FF,
    0x3F00_0000, // v_mov_b32 v13, 0.5
    0x7E1C_030D, // v_mov_b32 v14, v13
    0x7E1E_030D, // v_mov_b32 v15, v13
    0x7E20_030D, // v_mov_b32 v16, v13
    0x7E22_0283, // v_mov_b32 v17, 3
    0xE01C_2000,
    0x8003_0D11, // buffer_store_format_xyzw v[13:16], v17, s[12:15], 0 idxen
    0xBF81_0000, // s_endpgm
];

/// The emitter declares typed UAVs float4 whatever the view's format; with
/// 32-bit formats (the title's counters are R32_UINT) stores and loads keep
/// every bit, NaNs and denormals included.
#[test]
fn typed_stores_write_32_bit_formats() {
    let shader = build(CLEAR_PATTERN);
    let mut failed = Vec::new();
    for (name, format) in [
        ("R32_UINT", DXGI_FORMAT_R32_UINT),
        ("R32_FLOAT", DXGI_FORMAT_R32_FLOAT),
        ("R32_SINT", DXGI_FORMAT_R32_SINT),
    ] {
        for one in [5, 0x7F80_0001, 0x4049_0FDB, 0xFFFF_FFFF] {
            let data = [11, one, 22, 33, 44, 55];
            let want = vec![vec![11, one, one, 0x3F00_0000, 0, 55]];
            for (label, uavs) in run(&shader, &[View::TypedUavOf(format, &data)]) {
                if uavs != want {
                    failed.push(format!("{label} {name}: got {uavs:x?}, want {want:x?}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
