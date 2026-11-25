//! End-to-end tests of indexed buffer accesses whose record stride is only
//! in the descriptor (a heap descriptor no used-resource entry names): the
//! stride, and the bytes the host's raw view starts before the first
//! record, come from the view heap's records at run time; so does that
//! lead for scalar loads of a structured view; a constant buffer's
//! descriptor spilled to VGPR lanes loads once restored. Synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present, and
//! run on the host GPU and WARP against views of each kind.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator, view_kind};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const LANES: u32 = 32;

/// Lowers a wave32 program of 32 lanes with the resource heap at s0 and a
/// table at s1 whose entry 0 is a raw UAV.
fn build(words: &[u32]) -> HostShader {
    let shader = build_with(words, &[]);
    assert!(shader.view_strides, "the stride is read at run time");
    shader
}

/// [`build`] with `more` used-resource entries in the table.
fn build_with(words: &[u32], more: &[UsedResource]) -> HostShader {
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
        used: [UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 1,
            slot: 1,
        }]
        .into_iter()
        .chain(more.iter().copied())
        .collect(),
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

/// Runs `shader` with `views` at heap indices 0.. and their `records`
/// (stride, lead and kind, one per view; bound only when the shader reads
/// them) on the hardware GPU, which must take it, and on WARP if it takes
/// it (each skipped without a device): per one that ran, each UAV's dwords
/// in heap order.
fn run(
    shader: &HostShader,
    views: &[View],
    records: &[[u32; 3]],
) -> Vec<(&'static str, Vec<Vec<u32>>)> {
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
        // The table starts at heap index 0; the view records follow the
        // user data (no root views).
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let mut roots = vec![(0, Root::Constants(ud))];
        if shader.view_strides {
            roots.push((
                1,
                Root::View {
                    srv: true,
                    data: records.iter().flat_map(|&[s, l, k]| [s, l, k, 0]).collect(),
                },
            ));
        }
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots,
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

/// Each lane loads dwords 1 and 2 of record `lane` of the buffer at heap
/// index 1 (`buffer_load_dwordx2 idxen offset:4`, a descriptor read from the
/// heap) and stores them at dword 2 * lane of the UAV.
const LOAD_RECORDS: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (heap index 1)
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE034_2004,
    0x8002_0100, // buffer_load_dwordx2 v[1:2], v0, s[8:11], 0 idxen offset:4
    0x3406_0083, // v_lshlrev_b32 v3, 3, v0
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE074_1000,
    0x8003_0103, // buffer_store_dwordx2 v[1:2], v3, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

/// What [`LOAD_RECORDS`] stores from `data` read in records of `stride`
/// bytes.
fn loaded(data: &[u32], stride: u32) -> Vec<u32> {
    (0..LANES)
        .flat_map(|l| {
            let first = (l * stride / 4 + 1) as usize;
            [data[first], data[first + 1]]
        })
        .collect()
}

// The host makes raw views of untyped buffer views (structured ones
// included), and keeps each view's element size (4 for raw views, the
// structure's stride for structured ones) and lead: the bytes its raw view
// starts before the first record, to start on 16 bytes.
const RECORDS: [[u32; 2]; 5] = [[4, 0], [12, 0], [64, 0], [12, 8], [64, 12]];

#[test]
fn indexed_loads_read_records_of_the_views_own_stride() {
    let shader = build(LOAD_RECORDS);
    let mut failed = Vec::new();
    for [stride, lead] in RECORDS {
        // A record more for the last lane's second dword of 4-byte records.
        let data: Vec<u32> = (0..(lead + LANES * stride) / 4 + 4)
            .map(|i| 1000 + i)
            .collect();
        let views = [
            View::RawUav(2 * LANES),
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
        ];
        let want = loaded(&data[lead as usize / 4..], stride);
        let records = [
            [4, 0, view_kind::RAW_UAV],
            [stride, lead, view_kind::RAW_SRV],
        ];
        for (label, uavs) in run(&shader, &views, &records) {
            if uavs[0] != want {
                failed.push(format!(
                    "{label} {stride}+{lead}: got {:?}, want {want:?}",
                    uavs[0]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane stores its index to dword 1 of record `lane`, and adds 1 to
/// dword 2 of record `lane & 3`, of the UAV at heap index 1 (indexed
/// `buffer_store_dword` and `buffer_atomic_add` through a descriptor read
/// from the heap).
const STORE_RECORDS: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (heap index 1)
    0x3606_0083, // v_and_b32 v3, 3, v0
    0x7E04_0281, // v_mov_b32 v2, 1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_2004,
    0x8002_0000, // buffer_store_dword v0, v0, s[8:11], 0 idxen offset:4
    0xE0C8_2008,
    0x8002_0203, // buffer_atomic_add v2, v3, s[8:11], 0 idxen offset:8
    0xBF81_0000, // s_endpgm
];

#[test]
fn indexed_stores_and_atomics_write_records_of_the_views_own_stride() {
    let shader = build(STORE_RECORDS);
    let mut failed = Vec::new();
    for [stride, lead] in [[12, 0], [64, 0], [12, 4]] {
        let dwords = (lead + LANES * stride) / 4;
        let views = [View::RawUav(1), View::RawUav(dwords)];
        let mut want = vec![0u32; dwords as usize];
        let at = |record: u32, dword: u32| ((lead + record * stride) / 4 + dword) as usize;
        for lane in 0..LANES {
            want[at(lane, 1)] = lane;
        }
        for record in 0..4 {
            want[at(record, 2)] = LANES / 4;
        }
        let records = [
            [4, 0, view_kind::RAW_UAV],
            [stride, lead, view_kind::RAW_UAV],
        ];
        for (label, uavs) in run(&shader, &views, &records) {
            if uavs[1] != want {
                failed.push(format!(
                    "{label} {stride}+{lead}: got {:?}, want {want:?}",
                    uavs[1]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A structured shader resource view of `stride` bytes at entry 1 of the
/// table (heap index 1).
fn structured(stride: u32) -> UsedResource {
    UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 8,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 1,
        slot: 1,
    }
}

/// Each lane stores dword 1 of record 0 of the structured buffer at table
/// entry 1, read with `s_buffer_load_dword`, at dword `lane` of the UAV.
const SCALAR_LOAD: &[u32] = &[
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0xF408_0203,
    0xFA00_0020, // s_load_dwordx4 s[8:11], s[6:7], 0x20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0404,
    0xFA00_0004, // s_buffer_load_dword s16, s[8:11], 0x4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0210, // v_mov_b32 v1, s16
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xE070_1000,
    0x8003_0103, // buffer_store_dword v1, v3, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn scalar_loads_of_structured_views_start_at_the_first_record() {
    let mut failed = Vec::new();
    for [stride, lead] in [[12, 0], [12, 8], [64, 0], [64, 12]] {
        let shader = build_with(SCALAR_LOAD, &[structured(stride)]);
        let data: Vec<u32> = (0..(lead + 4 * stride) / 4).map(|i| 1000 + i).collect();
        let views = [
            View::RawUav(LANES),
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
        ];
        let want = vec![data[(lead / 4 + 1) as usize]; LANES as usize];
        let records = [
            [4, 0, view_kind::RAW_UAV],
            [stride, lead, view_kind::RAW_SRV],
        ];
        for (label, uavs) in run(&shader, &views, &records) {
            if uavs[0] != want {
                failed.push(format!(
                    "{label} {stride}+{lead}: got {:?}, want {want:?}",
                    uavs[0]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane stores dword 0 of the last record of the structured buffer of
/// `stride` bytes at table entry 1, read with `s_buffer_load_dword` at
/// (records - 1) * `stride`: the record count is dword 2 of the buffer's
/// descriptor, read as a number.
fn last_record(stride: u32) -> Vec<u32> {
    vec![
        0xBE86_0301, // s_mov_b32 s6, s1 (the table)
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0303,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
        0xF408_0203,
        0xFA00_0020, // s_load_dwordx4 s[8:11], s[6:7], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x8110_C10A, // s_add_i32 s16, s10, -1
        0x9310_FF10,
        stride, // s_mul_i32 s16, s16, stride
        0xF420_0444,
        0x2000_0000, // s_buffer_load_dword s17, s[8:11], s16
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0211, // v_mov_b32 v1, s17
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8003_0103, // buffer_store_dword v1, v3, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]
}

#[test]
fn a_structured_views_record_count_read_from_its_descriptor_counts_its_records() {
    const RECORDS: u32 = 5;
    let mut failed = Vec::new();
    for [stride, lead] in [[12, 0], [12, 8], [64, 0], [64, 12]] {
        let shader = build_with(&last_record(stride), &[structured(stride)]);
        let data: Vec<u32> = (0..(lead + RECORDS * stride) / 4)
            .map(|i| 1000 + i)
            .collect();
        let views = [
            View::RawUav(LANES),
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
        ];
        let want = vec![data[((lead + (RECORDS - 1) * stride) / 4) as usize]; LANES as usize];
        let records = [
            [4, 0, view_kind::RAW_UAV],
            [stride, lead, view_kind::RAW_SRV],
        ];
        for (label, uavs) in run(&shader, &views, &records) {
            if uavs[0] != want {
                failed.push(format!(
                    "{label} {stride}+{lead}: got {:?}, want {want:?}",
                    uavs[0]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane stores dword 1 of the constant buffer at table entry 1, and 7,
/// at dwords 2 * lane and 2 * lane + 1 of the UAV. The buffer's descriptor
/// and the 7 are spilled to lanes 0-4 of v5 (`v_writelane_b32`) and
/// restored (`v_readlane_b32`) before the load, as the console compiler
/// does under SGPR pressure: the descriptor's dwords have no host value.
const SPILLED_DESCRIPTOR: &[u32] = &[
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0xF408_0203,
    0xFA00_0020, // s_load_dwordx4 s[8:11], s[6:7], 0x20
    0xBE84_0387, // s_mov_b32 s4, 7
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD761_0005,
    0x0001_0008, // v_writelane_b32 v5, s8, 0
    0xD761_0005,
    0x0001_0209, // v_writelane_b32 v5, s9, 1
    0xD761_0005,
    0x0001_040A, // v_writelane_b32 v5, s10, 2
    0xD761_0005,
    0x0001_060B, // v_writelane_b32 v5, s11, 3
    0xD761_0005,
    0x0001_0804, // v_writelane_b32 v5, s4, 4
    0xD760_0010,
    0x0001_0105, // v_readlane_b32 s16, v5, 0
    0xD760_0011,
    0x0001_0305, // v_readlane_b32 s17, v5, 1
    0xD760_0012,
    0x0001_0505, // v_readlane_b32 s18, v5, 2
    0xD760_0013,
    0x0001_0705, // v_readlane_b32 s19, v5, 3
    0xD760_0014,
    0x0001_0905, // v_readlane_b32 s20, v5, 4
    0xF420_0548,
    0xFA00_0004, // s_buffer_load_dword s21, s[16:19], 0x4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0215, // v_mov_b32 v1, s21
    0x7E04_0214, // v_mov_b32 v2, s20
    0x3406_0083, // v_lshlrev_b32 v3, 3, v0
    0xE074_1000,
    0x8003_0103, // buffer_store_dwordx2 v[1:2], v3, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn a_constant_buffers_descriptor_spilled_to_vgpr_lanes_loads_once_restored() {
    let cbv = UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::Table,
        table_offset_dwords: 8,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 1,
        slot: 1,
    };
    let shader = build_with(SPILLED_DESCRIPTOR, &[cbv]);
    let cb: Vec<u32> = (0..64).map(|i| 1000 + i).collect();
    let views = [View::RawUav(2 * LANES), View::Cbv(&cb)];
    let records = [[4, 0, view_kind::RAW_UAV], [0, 0, view_kind::CBV]];
    let want: Vec<u32> = (0..LANES).flat_map(|_| [cb[1], 7]).collect();
    for (label, uavs) in run(&shader, &views, &records) {
        assert_eq!(uavs[0], want, "{label}");
    }
}

/// Each lane stores 2, 3 and 4 dwords (`buffer_store_dwordx4/x3/x2`) into
/// its 36-byte record of the raw UAV at table entry 0: values lane * 16 +
/// 1..=4, 2..=4 and 3..=4 at dwords 0, 4 and 7 of the record.
const WIDE_STORES: &[u32] = &[
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0x340C_0084, // v_lshlrev_b32 v6, 4, v0
    0x4A02_0C81, // v_add_nc_u32 v1, 1, v6
    0x4A04_0C82, // v_add_nc_u32 v2, 2, v6
    0x4A06_0C83, // v_add_nc_u32 v3, 3, v6
    0x4A08_0C84, // v_add_nc_u32 v4, 4, v6
    0x160A_00A4, // v_mul_u32_u24 v5, 36, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE078_1000,
    0x8003_0105, // buffer_store_dwordx4 v[1:4], v5, s[12:15], 0 offen
    0xE07C_1010,
    0x8003_0205, // buffer_store_dwordx3 v[2:4], v5, s[12:15], 0 offen offset:16
    0xE074_101C,
    0x8003_0305, // buffer_store_dwordx2 v[3:4], v5, s[12:15], 0 offen offset:28
    0xBF81_0000, // s_endpgm
];

#[test]
fn raw_stores_of_several_dwords_write_each_dword_in_place() {
    let shader = build_with(WIDE_STORES, &[]);
    let want: Vec<u32> = (0..LANES)
        .flat_map(|l| [1, 2, 3, 4, 2, 3, 4, 3, 4].map(|k| l * 16 + k))
        .collect();
    let views = [View::RawUav(9 * LANES)];
    let mut failed = Vec::new();
    for (label, uavs) in run(&shader, &views, &[[4, 0, view_kind::RAW_UAV]]) {
        if uavs[0] != want {
            failed.push(format!("{label}: got {:?}, want {want:?}", uavs[0]));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
