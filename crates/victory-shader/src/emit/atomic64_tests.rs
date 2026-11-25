//! End-to-end tests of 64-bit image atomics (two data dwords per value):
//! synthetic GFX10 programs lowered to DXIL, validated by `dxil.dll` when
//! present, and run on R32G32_UINT texture UAVs on the host GPU and WARP
//! (each skipped without 64-bit atomics on typed resources in heaps).

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const LITERAL: u8 = 0xFF;
/// Inline constants: 0, 1 and -1.
const ZERO: u8 = 128;
const ONE: u8 = 129;
const ONES: u8 = 193;
/// Console threads, one wave32.
const LANES: u32 = 32;
/// Dwords each lane stores.
const STORED: u32 = 8;

const fn v(n: u8) -> u16 {
    256 + n as u16
}

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// A 2D image operation (unorm addresses) of `dmask` on the descriptor at
/// s[`srsrc`..], addressed by v[`vaddr`:`vaddr` + 1].
const fn mimg(op: u32, dmask: u32, glc: bool, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | (glc as u32) << 13 | 1 << 12 | dmask << 8 | 1 << 3,
        (srsrc as u32 / 4) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// `buffer_store_dwordx2 v[vdata:vdata + 1], v[vaddr], s[srsrc..], 0 offen
/// offset:offset`.
const fn store_x2(vdata: u8, vaddr: u8, srsrc: u8, offset: u32) -> [u32; 2] {
    [
        0xE074_1000 | offset,
        0x80 << 24 | (srsrc as u32 / 4) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// The table at s0 holds a raw UAV, then textures A (8 by 1) and B (32 by
/// 1), 64-bit texels as (low, high) dwords.
///
/// Lane `l` takes A's texel `l & 7` to the 64-bit maximum of it and
/// `(!l, l >> 3)`. On B's texel `l` it loads the texel, adds
/// `(0xFFFF_FFFF, l)`, then `(l + 1, 0)` returning the old value, swaps in
/// `(l ^ 0x1234_5678, 0x9ABC_DEF0)` where the texel is `(l, l + 1)`, and
/// tries again with the same comparison and `(0, 0)`. It stores what each
/// returned and what it loaded at dwords `8 * l..`.
fn program() -> Vec<u32> {
    let mut w = vec![
        sop1(0x03, 1, ZERO),
        // s_load_dwordx4 s[24:27], s[0:1], 0 (the raw UAV)
        0xF408_0600,
        0xFA00_0000,
        // s_load_dwordx8 s[8:15], s[0:1], 0x20 (A)
        0xF40C_0200,
        0xFA00_0020,
        // s_load_dwordx8 s[16:23], s[0:1], 0x40 (B)
        0xF40C_0400,
        0xFA00_0040,
        // s_waitcnt lgkmcnt(0)
        0xBF8C_C07F,
        // v3 = l & 7; v4 = 0; v1 = !l; v2 = l >> 3.
        vop2(0x1B, 3, 128 + 7, 0),
        vop1(0x01, 4, u16::from(ZERO)),
        vop1(0x37, 1, v(0)),
        vop2(0x16, 2, 128 + 3, 0),
    ];
    // image_atomic_umax v[1:2], v[3:4], s[8:15] dmask:0x3
    w.extend(mimg(0x17, 3, false, 1, 3, 8));
    // v5 = l; v6 = 0; image_load v[20:21], v[5:6], s[16:23] dmask:0x3
    w.extend([vop1(0x01, 5, v(0)), vop1(0x01, 6, u16::from(ZERO))]);
    w.extend(mimg(0x00, 3, false, 20, 5, 16));
    // v7 = -1; v8 = l; image_atomic_add v[7:8], v[5:6], s[16:23] dmask:0x3
    w.extend([vop1(0x01, 7, u16::from(ONES)), vop1(0x01, 8, v(0))]);
    w.extend(mimg(0x11, 3, false, 7, 5, 16));
    // v9 = l + 1; v10 = 0; image_atomic_add v[9:10], ... glc
    w.extend([
        vop2(0x25, 9, u16::from(ONE), 0),
        vop1(0x01, 10, u16::from(ZERO)),
    ]);
    w.extend(mimg(0x11, 3, true, 9, 5, 16));
    // v11 = l ^ 0x12345678; v12 = 0x9ABCDEF0; v13 = l; v14 = l + 1;
    // image_atomic_cmpswap v[11:14], v[5:6], s[16:23] dmask:0xf glc
    w.extend([vop2(0x1D, 11, u16::from(LITERAL), 0), 0x1234_5678]);
    w.extend([vop1(0x01, 12, u16::from(LITERAL)), 0x9ABC_DEF0]);
    w.extend([vop1(0x01, 13, v(0)), vop2(0x25, 14, u16::from(ONE), 0)]);
    w.extend(mimg(0x10, 0xF, true, 11, 5, 16));
    // v15 = 0; v16 = 0; v17 = l; v18 = l + 1; the same, failing.
    w.extend([
        vop1(0x01, 15, u16::from(ZERO)),
        vop1(0x01, 16, u16::from(ZERO)),
        vop1(0x01, 17, v(0)),
        vop2(0x25, 18, u16::from(ONE), 0),
    ]);
    w.extend(mimg(0x10, 0xF, true, 15, 5, 16));
    // s_waitcnt vmcnt(0); v19 = l << 5; the stores.
    w.extend([0xBF8C_3F70, vop2(0x1A, 19, 128 + 5, 0)]);
    for (k, reg) in [9u8, 11, 15, 20].into_iter().enumerate() {
        w.extend(store_x2(reg, 19, 24, 8 * k as u32));
    }
    w.push(S_ENDPGM);
    w
}

/// Lowers a wave32 compute program of one wave whose table (at s0) holds
/// a raw UAV and two texture UAVs.
fn lower_cs(words: &[u32]) -> HostShader {
    let uav = |table_offset_dwords, dwords| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
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
        used: vec![uav(0, 4), uav(8, 8), uav(16, 8)],
        io: StageIo::Cs(ComputeIo {
            // One user SGPR, thread ids x.
            rsrc2: 1 << 1,
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

/// Whether `host` has 64-bit atomics on typed resources, in descriptor
/// heaps too.
fn atomic64_textures(host: &Host) -> bool {
    let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
    let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
    // SAFETY: each query writes its own structure of the given size.
    let ok = unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS9,
                (&raw mut typed).cast(),
                size_of_val(&typed) as u32,
            )
            .is_ok()
            && host
                .device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS11,
                    (&raw mut heap).cast(),
                    size_of_val(&heap) as u32,
                )
                .is_ok()
    };
    ok && typed.AtomicInt64OnTypedResourceSupported.as_bool()
        && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool()
}

/// Runs `shader` (one group) on the hardware GPU and WARP, each that has
/// 64-bit atomics on heap textures; returns, per one that ran, the raw
/// UAV's and the textures' dwords.
fn run(shader: &HostShader) -> Vec<(&'static str, Vec<Vec<u32>>)> {
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
        if !atomic64_textures(&host) {
            eprintln!("no 64-bit atomics on heap textures on {label}; skipped");
            continue;
        }
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(STORED * LANES),
                        View::Uint64TextureUav(8, 1),
                        View::Uint64TextureUav(LANES, 1),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        eprintln!("ran on {label}");
        out.push((label, uavs));
    }
    assert!(
        !out.is_empty(),
        "no SM 6.6 device with 64-bit atomics on heap textures"
    );
    out
}

#[test]
fn image_atomics_of_two_dwords_are_64_bit_atomics() {
    let shader = lower_cs(&program());
    // The maximum of each texel is lane `x + 24`'s: its high dword is the
    // greatest, its low one the least.
    let a: Vec<u32> = (0..8).flat_map(|x| [!(x + 24), 3]).collect();
    let b: Vec<u32> = (0..LANES)
        .flat_map(|l| [l ^ 0x1234_5678, 0x9ABC_DEF0])
        .collect();
    // The add's old value; the swap's, (l, l + 1) as the add carried; the
    // failed swap's, what the first swapped in; the load's.
    let returned: Vec<u32> = (0..LANES)
        .flat_map(|l| [!0, l, l, l + 1, l ^ 0x1234_5678, 0x9ABC_DEF0, 0, 0])
        .collect();
    for (label, uavs) in run(&shader) {
        assert_eq!(uavs[0], returned, "{label}: returned");
        assert_eq!(uavs[1], a, "{label}: A");
        assert_eq!(uavs[2], b, "{label}: B");
    }
}
