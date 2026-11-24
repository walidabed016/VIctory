//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a waterfall of [`lower_cs`]: one access with each lane's own
/// heap index, no wave operation.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let (shader, nonuniform_handles, wave_ops) = lower_cs(words, buffers);
    assert_eq!(nonuniform_handles, 1);
    assert!(!wave_ops);
    shader
}

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1; with its
/// non-uniform handles and whether it uses wave operations.
fn lower_cs(words: &[u32], buffers: bool) -> (HostShader, u32, bool) {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
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
    let (nonuniform_handles, wave_ops) = (lowered.nonuniform_handles, lowered.wave_ops);
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
    (shader, nonuniform_handles, wave_ops)
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
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
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

/// A gather at an explicit level (`image_gather4_l`, emulated) returns the
/// texels the hardware's `image_gather4_lz` returns, in gather order and
/// clamped at the edges, and a level past the last is the last.
#[test]
fn a_gather_at_a_level_matches_the_gather_at_level_zero() {
    // Lane l gathers red at u = (l & 7) / 8 + 1/16, v = (l >> 3) / 4 + 0.1
    // from a 4x4 texture whose red is x + 4y, and stores the four texels as
    // a + 16b + 256c + 4096d.
    let program = |gather: [u32; 2], lod: u32| {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xF408_0203,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x3602_0087, // v_and_b32 v1, 7, v0
            0x2C04_0083, // v_lshrrev_b32 v2, 3, v0
            0x7E02_0D01, // v_cvt_f32_u32 v1, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x1002_02FF,
            0x3E00_0000, // v_mul_f32 v1, 0.125, v1
            0x0602_02FF,
            0x3D80_0000, // v_add_f32 v1, 0.0625, v1
            0x1004_04FF,
            0x3E80_0000, // v_mul_f32 v2, 0.25, v2
            0x0604_04FF,
            0x3DCC_CCCD,       // v_add_f32 v2, 0.1, v2
            0x7E06_0200 | lod, // v_mov_b32 v3, lod
            0x3420_0082,       // v_lshlrev_b32 v16, 2, v0
            0xBF8C_C07F,       // s_waitcnt lgkmcnt(0)
        ];
        words.extend(gather);
        words.extend([
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x5608_0AFF,
            0x4180_0000, // v_fmac_f32 v4, 16.0, v5
            0x5608_0CFF,
            0x4380_0000, // v_fmac_f32 v4, 256.0, v6
            0x5608_0EFF,
            0x4580_0000, // v_fmac_f32 v4, 4096.0, v7
            0xE070_1000,
            0x800A_0410, // buffer_store_dword v4, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ]);
        lower_cs(&words, false).0
    };
    // image_gather4_l / _lz v[4:7], v[1:3], s[12:19], s[8:11] dmask:0x1 dim:2D
    let at_level = [0xF110_0108, 0x0043_0401];
    let level_zero = [0xF11C_0108, 0x0043_0401];
    let texels: Vec<[f32; 4]> = (0..16).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
    let views = [
        View::Texture(4, 4, &texels),
        View::Texture(1, 1, &texels[..1]),
        View::RawUav(32),
    ];
    let want: Vec<f32> = (0..32)
        .map(|l: i32| {
            let x = ((l & 7) as f32 / 8.0 + 0.0625) * 4.0 - 0.5;
            let y = ((l >> 3) as f32 / 4.0 + 0.1) * 4.0 - 0.5;
            let (x0, y0) = (x.floor() as i32, y.floor() as i32);
            let red = |x: i32, y: i32| (x.clamp(0, 3) + 4 * y.clamp(0, 3)) as f32;
            [(x0, y0 + 1), (x0 + 1, y0 + 1), (x0 + 1, y0), (x0, y0)]
                .iter()
                .zip([1.0, 16.0, 256.0, 4096.0])
                .map(|(&(x, y), k)| red(x, y) * k)
                .sum()
        })
        .collect();
    // v3 = 0.0 (128) and 1.0 (242).
    for (name, gather, lod) in [
        ("image_gather4_lz", level_zero, 128),
        ("image_gather4_l level 0", at_level, 128),
        ("image_gather4_l level 1", at_level, 242),
    ] {
        for (label, words) in run(&program(gather, lod), &views) {
            let got: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
            assert_eq!(got, want, "{label}: {name}");
        }
    }
}

/// A sample with gradients and a least level of detail
/// (`image_sample_d_cl`) samples no level below it: with zero gradients,
/// even lanes (least level 0) read level 0 of a two-level texture (red 1.0),
/// odd lanes (least level 1) level 1 (red 2.0).
#[test]
fn a_sample_with_gradients_keeps_to_its_least_level() {
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E02_0280, // v_mov_b32 v1, 0 (du/dx)
        0x7E04_0280, // v_mov_b32 v2, 0 (dv/dx)
        0x7E06_0280, // v_mov_b32 v3, 0 (du/dy)
        0x7E08_0280, // v_mov_b32 v4, 0 (dv/dy)
        0x7E0A_02F0, // v_mov_b32 v5, 0.5 (u)
        0x7E0C_02F0, // v_mov_b32 v6, 0.5 (v)
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x7E0E_0D07, // v_cvt_f32_u32 v7, v7 (the least level)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF08C_0108,
        0x0043_0801, // image_sample_d_cl v8, v[1:7], s[12:19], s[8:11] dmask:0x1 dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0810, // buffer_store_dword v8, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let shader = lower_cs(&words, false).0;
    let mut texels = vec![[1.0f32, 0.0, 0.0, 1.0]; 4];
    texels.push([2.0, 0.0, 0.0, 1.0]);
    let views = [
        View::Mips(2, 2, 2, &texels),
        View::Texture(1, 1, &texels[..1]),
        View::RawUav(32),
    ];
    let want: Vec<f32> = (0..32)
        .map(|l| if l & 1 == 0 { 1.0 } else { 2.0 })
        .collect();
    let ran = run(&shader, &views);
    assert!(!ran.is_empty(), "no SM 6.6 device");
    for (label, words) in ran {
        let got: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
        assert_eq!(got, want, "{label}");
    }
}

/// A cube sample with gradients (`image_sample_d`, or with `cl` its `_cl`
/// form with least level 2): lane l samples face min(l / 4, 5) of a cube of
/// 64-texel faces with a gradient of 2^(k + 1) texels in the k-th of ds/dx,
/// dt/dx, ds/dy and dt/dy, k = l & 3.
fn cube_gradients(cl: bool) -> Vec<u32> {
    let sample = if cl {
        // image_sample_d_cl v12, v[1:8], s[12:19], s[8:11] dmask:0x1 dim:CUBE
        0xF08C_0118
    } else {
        // image_sample_d v12, v[1:7], s[12:19], s[8:11] dmask:0x1 dim:CUBE
        0xF088_0118
    };
    vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3612_0083, // v_and_b32 v9, 3, v0 (k)
        0x7E14_0282, // v_mov_b32 v10, 2
        0x3414_1509, // v_lshlrev_b32 v10, v9, v10
        0x7E14_0D0A, // v_cvt_f32_u32 v10, v10
        0x1014_14FF,
        0x3C80_0000, // v_mul_f32 v10, 1/64, v10 (the gradient)
        0x7D84_1280, // v_cmp_eq_u32 vcc_lo, 0, v9
        0x0202_1480, // v_cndmask_b32 v1, 0, v10, vcc_lo (ds/dx)
        0x7D84_1281, // v_cmp_eq_u32 vcc_lo, 1, v9
        0x0204_1480, // v_cndmask_b32 v2, 0, v10, vcc_lo (dt/dx)
        0x7D84_1282, // v_cmp_eq_u32 vcc_lo, 2, v9
        0x0206_1480, // v_cndmask_b32 v3, 0, v10, vcc_lo (ds/dy)
        0x7D84_1283, // v_cmp_eq_u32 vcc_lo, 3, v9
        0x0208_1480, // v_cndmask_b32 v4, 0, v10, vcc_lo (dt/dy)
        0x7E0A_02FF,
        0x3FA0_0000, // v_mov_b32 v5, 1.25 (s)
        0x7E0C_02FF,
        0x3FE0_0000, // v_mov_b32 v6, 1.75 (t)
        0x2C16_0082, // v_lshrrev_b32 v11, 2, v0
        0x2616_1685, // v_min_u32 v11, 5, v11
        0x7E0E_0D0B, // v_cvt_f32_u32 v7, v11 (the face)
        0x7E10_02F4, // v_mov_b32 v8, 2.0 (the least level)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        sample,
        0x0043_0C01,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]
}

#[test]
fn a_cube_sample_with_gradients_takes_the_level_its_face_gradients_give() {
    // Faces of 64 texels, 7 levels; level l of face f reads 10 f + l. (Past a
    // face's width per pixel, the 4090 picks a level below WARP's off the
    // face's centre.)
    let texels: Vec<[f32; 4]> = (0..6u32)
        .flat_map(|f| {
            (0..7u32).flat_map(move |l| {
                let n = (64usize >> l).pow(2);
                std::iter::repeat_n([(10 * f + l) as f32, 0.0, 0.0, 1.0], n)
            })
        })
        .collect();
    let views = [View::Cube(64, 7, &texels), View::Null, View::RawUav(32)];
    for cl in [false, true] {
        let shader = lower_cs(&cube_gradients(cl), false).0;
        let want: Vec<f32> = (0..32u32)
            .map(|l| {
                let level = (l & 3) + 1;
                let level = if cl { level.max(2) } else { level };
                (10 * (l >> 2).min(5) + level) as f32
            })
            .collect();
        let ran = run(&shader, &views);
        assert!(!ran.is_empty(), "no SM 6.6 device");
        for (label, words) in ran {
            let got: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
            assert_eq!(got, want, "{label}, cl {cl}");
        }
    }
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}
