//! End-to-end tests of buffer accesses through view-heap slots whose kind
//! no used-resource entry gives (a descriptor index in a root constant, or
//! one each lane chooses): each goes by the kind the view heap's record of
//! the slot names, so a constant buffer is read as one, a raw view raw, and
//! a slot holding a texture or nothing reads zeros and drops writes, where
//! a handle of the wrong kind is undefined (a GPU fault on NVIDIA). Synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present, and
//! run on the host GPU and WARP.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HeapIndices, HostShader, heap_indices, lower, validator, view_kind};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const LANES: u32 = 32;

/// Lowers a wave32 program of 32 lanes with the resource heap at s0, a
/// table at s1 whose entry 0 is a raw UAV, and a root constant at s2; with
/// the heap slots it reaches through its user data.
fn build(words: &[u32]) -> (HostShader, HeapIndices) {
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
        layout: vec![
            param(RootType::ResourceHeap, 0),
            param(RootType::Table, 1),
            param(RootType::Constants, 2),
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
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
        psv: None,
    })
    .expect("lower");
    assert!(lowered.view_strides, "the slot's kind is read at run time");
    let heaps = heap_indices(&p, &native, &res);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        view_strides: lowered.view_strides,
        heap_indices: heaps.known.clone(),
        heap_unknown: heaps.unknown.len() as u32,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    (shader, heaps)
}

/// The stage's user data with `constant` in the root constant (the table
/// starts at heap index 0).
fn user_data(shader: &HostShader, constant: u32) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if u.root == RootType::Constants {
            ud[u.index as usize] = constant;
        }
    }
    ud
}

/// Runs `shader` with `constant` in its root constant and `views` at heap
/// indices 0.. with their records (stride, lead and kind) on the hardware
/// GPU and on WARP (each skipped without a device): per one that ran, each
/// UAV's dwords in heap order.
fn run(
    shader: &HostShader,
    constant: u32,
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
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(user_data(shader, constant))),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: records.iter().flat_map(|&[s, l, k]| [s, l, k, 0]).collect(),
                            },
                        ),
                    ],
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

/// The dwords of every buffer in [`heap`].
fn data() -> Vec<u32> {
    (0..2 * LANES).map(|i| 1000 + i).collect()
}

/// A dword more for [`heap`]'s last slot.
fn other() -> Vec<u32> {
    (0..2 * LANES).map(|i| 5000 + i).collect()
}

const TEXEL: [[f32; 4]; 1] = [[1.0; 4]];

/// The view heap of the tests, a view of each kind the host makes, and its
/// records: 0 the table's UAV (zeroed), 1 a constant buffer, 2 a raw SRV,
/// 3 a raw SRV whose records start 8 bytes in (a structured view the host
/// made raw), 4 a raw UAV, 5 a texture, 6 nothing, 7 a raw SRV of [`other`].
fn heap<'a>(data: &'a [u32], other: &'a [u32]) -> ([View<'a>; 8], [[u32; 3]; 8]) {
    let raw = |data| View::Buffer {
        data,
        stride: Stride::Raw,
    };
    (
        [
            View::RawUav(2 * LANES),
            View::Cbv(data),
            raw(data),
            raw(data),
            View::RawUavOf(data),
            View::Texture(1, 1, &TEXEL),
            View::Null,
            raw(other),
        ],
        [
            [4, 0, view_kind::RAW_UAV],
            [0, 0, view_kind::CBV],
            [4, 0, view_kind::RAW_SRV],
            [4, 8, view_kind::RAW_SRV],
            [4, 0, view_kind::RAW_UAV],
            [0, 0, view_kind::TEXTURE],
            [0, 0, view_kind::NONE],
            [4, 0, view_kind::RAW_SRV],
        ],
    )
}

/// What reading byte `byte` of heap slot `slot` gives.
fn read(slot: u32, byte: u32) -> u32 {
    let at = |lead: u32| ((lead + byte) / 4) as usize;
    match slot {
        1 | 2 | 4 => data()[at(0)],
        3 => data()[at(8)],
        7 => other()[at(0)],
        _ => 0,
    }
}

/// Each lane loads dword `lane` of the buffer at the heap slot the root
/// constant names (`buffer_load_dword`, the descriptor read from the heap
/// at constant * 32) and stores it at dword `lane` of the UAV.
const LOAD_BY_CONSTANT: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0x8F03_8502, // s_lshl_b32 s3, s2, 5 (the slot in the root constant)
    0xF408_0202,
    0x0600_0000, // s_load_dwordx4 s[8:11], s[4:5], s3
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_1000,
    0x8002_0103, // buffer_load_dword v1, v3, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8003_0103, // buffer_store_dword v1, v3, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

/// Each lane loads the short at byte 2 * lane of the buffer at the heap slot
/// the root constant names (`buffer_load_ushort`) and stores it at dword
/// `lane` of the UAV.
const LOAD_SHORT_BY_CONSTANT: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0x8F03_8502, // s_lshl_b32 s3, s2, 5 (the slot in the root constant)
    0xF408_0202,
    0x0600_0000, // s_load_dwordx4 s[8:11], s[4:5], s3
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0x3406_0081, // v_lshlrev_b32 v3, 1, v0
    0x3408_0082, // v_lshlrev_b32 v4, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE028_1000,
    0x8002_0103, // buffer_load_ushort v1, v3, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8003_0104, // buffer_store_dword v1, v4, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

/// A short load reads its half of the dword holding it, zero-extended, from
/// raw views; constant buffer slots give nothing.
#[test]
fn short_loads_read_their_half_of_a_dword() {
    let (shader, heaps) = build(LOAD_SHORT_BY_CONSTANT);
    assert!(heaps.unknown.is_empty(), "{:?}", heaps.unknown);
    let (data, other) = (data(), other());
    let (views, records) = heap(&data, &other);
    let mut failed = Vec::new();
    for slot in 1..8 {
        let mut want: Vec<u32> = (0..LANES)
            .map(|l| match slot {
                1 => 0,
                _ => read(slot, 2 * l) >> (16 * (l % 2)) & 0xffff,
            })
            .collect();
        want.resize(2 * LANES as usize, 0);
        for (label, uavs) in run(&shader, slot, &views, &records) {
            if uavs[0] != want {
                failed.push(format!(
                    "{label} slot {slot}: got {:?}, want {want:?}",
                    uavs[0]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn vector_loads_read_a_slot_by_the_kind_of_its_view() {
    let (shader, heaps) = build(LOAD_BY_CONSTANT);
    assert!(heaps.unknown.is_empty(), "{:?}", heaps.unknown);
    let (data, other) = (data(), other());
    let (views, records) = heap(&data, &other);
    let mut failed = Vec::new();
    for slot in 1..8 {
        let ud = user_data(&shader, slot);
        let slots: Vec<u32> = heaps.known.iter().map(|h| h.slot(&ud)).collect();
        assert_eq!(slots, [slot], "the host makes the slot the shader reads");
        let mut want: Vec<u32> = (0..LANES).map(|l| read(slot, 4 * l)).collect();
        want.resize(2 * LANES as usize, 0);
        for (label, uavs) in run(&shader, slot, &views, &records) {
            if uavs[0] != want {
                failed.push(format!(
                    "{label} slot {slot}: got {:?}, want {want:?}",
                    uavs[0]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane loads dwords 2 and 3 of the buffer at the heap slot the root
/// constant's low 10 bits name (`s_buffer_load_dwordx2`, a constant
/// buffer read on the console) and stores them at dword 2 * lane of the
/// UAV.
const SCALAR_LOAD_BY_CONSTANT: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0x8703_FF02,
    0x0000_03FF, // s_and_b32 s3, s2, 0x3ff
    0x8F03_8503, // s_lshl_b32 s3, s3, 5
    0xF408_0202,
    0x0600_0000, // s_load_dwordx4 s[8:11], s[4:5], s3
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF424_0404,
    0xFA00_0008, // s_buffer_load_dwordx2 s[16:17], s[8:11], 0x8
    0x3406_0083, // v_lshlrev_b32 v3, 3, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0210, // v_mov_b32 v1, s16
    0x7E04_0211, // v_mov_b32 v2, s17
    0xE074_1000,
    0x8003_0103, // buffer_store_dwordx2 v[1:2], v3, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn scalar_loads_read_a_slot_by_the_kind_of_its_view() {
    let (shader, heaps) = build(SCALAR_LOAD_BY_CONSTANT);
    assert!(heaps.unknown.is_empty(), "{:?}", heaps.unknown);
    let (data, other) = (data(), other());
    let (views, records) = heap(&data, &other);
    let mut failed = Vec::new();
    for slot in 1..8 {
        // Bits above the 10 the shader keeps are not the slot's.
        let constant = 0x2A5 << 10 | slot;
        let ud = user_data(&shader, constant);
        let slots: Vec<u32> = heaps.known.iter().map(|h| h.slot(&ud)).collect();
        assert_eq!(slots, [slot], "the host makes the slot the shader reads");
        let want: Vec<u32> = (0..LANES)
            .flat_map(|_| [read(slot, 8), read(slot, 12)])
            .collect();
        for (label, uavs) in run(&shader, constant, &views, &records) {
            if uavs[0] != want {
                failed.push(format!(
                    "{label} slot {slot}: got {:?}, want {want:?}",
                    uavs[0]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane stores its index at dword `lane` of the buffer at the heap
/// slot the root constant names.
const STORE_BY_CONSTANT: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0x8F03_8502, // s_lshl_b32 s3, s2, 5 (the slot in the root constant)
    0xF408_0202,
    0x0600_0000, // s_load_dwordx4 s[8:11], s[4:5], s3
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x8002_0003, // buffer_store_dword v0, v3, s[8:11], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn vector_stores_write_only_slots_holding_raw_unordered_views() {
    let (shader, heaps) = build(STORE_BY_CONSTANT);
    assert!(heaps.unknown.is_empty(), "{:?}", heaps.unknown);
    assert!(heaps.known.iter().all(|h| h.write), "{:?}", heaps.known);
    let (data, other) = (data(), other());
    let (views, records) = heap(&data, &other);
    let mut failed = Vec::new();
    for slot in 1..8 {
        // The UAVs: the table's (untouched) and slot 4's.
        let mut written = data.clone();
        if slot == 4 {
            written[..LANES as usize].copy_from_slice(&(0..LANES).collect::<Vec<_>>());
        }
        let want = vec![vec![0; 2 * LANES as usize], written];
        for (label, uavs) in run(&shader, slot, &views, &records) {
            if uavs != want {
                failed.push(format!("{label} slot {slot}: got {uavs:?}, want {want:?}"));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane loads dword `lane` of the buffer at heap slot `lane & 7`
/// (its own: the console's loop over the first active lane's slot,
/// narrowed with `v_cmpx_eq_u32`) and stores it at dword `lane` of the
/// UAV.
const LOAD_PER_LANE: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the resource heap)
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1 (the table)
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], null
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0x360E_0087, // v_and_b32 v7, 7, v0
    0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBE97_037E, // s_mov_b32 s23, exec_lo
    0xBE95_037E, // s_mov_b32 s21, exec_lo
    0x7E28_0507, // 0x30: v_readfirstlane_b32 s20, v7
    0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
    0xF408_0202,
    0x2800_0000, // s_load_dwordx4 s[8:11], s[4:5], s20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_1000,
    0x8002_0110, // buffer_load_dword v1, v16, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
    0xBF89_FFF6, // s_cbranch_execnz -> 0x30
    0xBEFE_0317, // s_mov_b32 exec_lo, s23
    0xE070_1000,
    0x8003_0110, // buffer_store_dword v1, v16, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn lanes_read_their_own_slot_by_the_kind_of_its_view() {
    let (shader, heaps) = build(LOAD_PER_LANE);
    assert!(heaps.known.is_empty(), "{:?}", heaps.known);
    assert!(!heaps.unknown.is_empty(), "the slots are the lanes' own");
    let (data, other) = (data(), other());
    let (views, records) = heap(&data, &other);
    // Slot 0 is the UAV itself, zeroed when the loads read it.
    let mut want: Vec<u32> = (0..LANES).map(|l| read(l & 7, 4 * l)).collect();
    want.resize(2 * LANES as usize, 0);
    for (label, uavs) in run(&shader, 0, &views, &records) {
        assert_eq!(uavs[0], want, "{label}");
    }
}
