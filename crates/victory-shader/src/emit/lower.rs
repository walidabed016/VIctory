//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, VIEW_RECORD_BYTES, flags, view_kind};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, VCC, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, BVH, CODE, Moved, Resources, Role, Site, Source, restored};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    /// Whether it reads the view heap's records.
    pub view_strides: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// How the waves of a wave64 compute program run.
    pub waves: super::Waves,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Group barriers the 32-lane halves add.
    #[cfg(test)]
    pub halves_barriers: u32,
    /// The index of each non-uniform heap handle in a lane outside every
    /// EXEC the program set ([`super::bitcode::Module::second_arms`]).
    #[cfg(test)]
    pub nonuniform_outside_exec: Vec<u32>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    if !wave64_lanes(parts) {
        return lower_as(parts, Split::No);
    }
    // On 32-lane halves where the translation shows it exact: lane by lane,
    // else, for a group of whole console waves, in each wave's own control
    // flow; else on the console's 64-lane waves with the reason.
    let why = |e| match e {
        ShaderError::Unsupported { what, .. } => what,
        _ => "an error on 32-lane halves",
    };
    let mut reason = match lower_as(parts, Split::Halves) {
        Ok(l) => return Ok(l),
        Err(e) => why(e),
    };
    let StageIo::Cs(cs) = parts.native.io else {
        unreachable!("wave64 lanes are compute")
    };
    let threads = cs.threads.iter().product::<u32>();
    if threads <= 64 || threads.is_multiple_of(64) {
        match lower_as(parts, Split::Lockstep) {
            Ok(l) => return Ok(l),
            Err(e) => reason = why(e),
        }
    }
    let mut l = lower_as(parts, Split::No)?;
    l.waves = super::Waves::Wave64(reason);
    Ok(l)
}

/// How the console's waves of a wave64 compute program that needs its lanes
/// run on the host ([`super::halves`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    /// Each as one host wave of its size.
    No,
    /// As two 32-lane halves, each lane on its own way through the program.
    Halves,
    /// As two 32-lane halves following the console wave's own branches,
    /// EXEC a value; groups of one console wave, or of several: played in
    /// turn where they never meet ([`Serial`]), else each wave's halves
    /// waiting for each other alone ([`Halves::pairs`]).
    Lockstep,
}

/// [`lower`], with the console's waves run as `split` says.
fn lower_as(parts: &Parts, split: Split) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    let mut lanes = false;
    loop {
        match Lower::new(parts, &bad, split, lanes).run() {
            Ok(l) => return Ok(l),
            // Again on the console's lanes, where what had no DXIL form off
            // them may have one.
            Err(Stop::Fail(ShaderError::Unsupported {
                what: HOST_LANES, ..
            })) if !lanes => {
                lanes = true;
                bad.clear();
            }
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// Lockstep: this thread's bit of a lane mask, by DXIL block and the
    /// mask's dwords.
    pair_bits: HashMap<(usize, Value, Value), V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, u32, u32), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes (a loop's last pass included):
    /// thread ids follow the host lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values, by row: where each
    /// half comes from when two per dword ([`packed_halves`]).
    half_outputs: HashMap<u8, Option<HalfSources>>,
    /// Pixel inputs holding 16-bit values two per dword, by row: where each
    /// half comes from ([`packed_halves`]).
    half_inputs: HashMap<u8, HalfSources>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// Wave64 on 32-lane halves ([`super::halves`]).
    halves: Option<Halves>,
    /// Halves: why a mask read as data has no ballot where it is written.
    ballot_fail: HashMap<Value, Why>,
    /// Loops on one block whose scalars code after them reads as the last
    /// pass's (see [`Lower::last_pass_after`]).
    last_pass_loops: HashSet<usize>,
    /// Those scalars as that code reads them, with their loop's block.
    last_pass: HashMap<Value, (usize, Val)>,
    /// Values made from image descriptor dwords read as data (see [`Dim`]).
    dims: HashMap<Value, Dim>,
    /// The site a join stands for that takes dwords of the descriptors
    /// loaded at these sites, by block and incoming edge (see
    /// [`Lower::phi_dim`]).
    phi_sites: HashMap<(usize, Vec<Site>), Site>,
    #[cfg(test)]
    halves_barriers: u32,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";
/// What a lane mask with lanes outside EXEC where it is written stops with.
const ELSEWHERE: Why = "a lane mask holding lanes elsewhere read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Compute code that depends on the console's lane layout: it reads other
/// lanes or lane masks as numbers, or tests a mask held as a number or a
/// constant lane set (the lane tests its own bit by its console lane).
fn reads_console_lanes(x: &Parts) -> bool {
    let insts = || x.p.blocks.iter().flat_map(|b| &b.insts);
    let crossing = insts().any(|i| {
        let n = i.name();
        n.starts_with("ds_swizzle")
            || n.contains("permute")
            || n.starts_with("v_mbcnt")
            || matches!(n, "v_readlane_b32" | "v_writelane_b32")
            || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
            || reads_mask_as_data(x.st, i)
    });
    if crossing {
        return true;
    }
    let whens = x.st.exits.iter().filter_map(|e| match *e {
        LaneExit::Branch { when, .. } => Some(when),
        _ => None,
    });
    let mut all: Vec<Bool> =
        x.st.guards
            .iter()
            .flatten()
            .flatten()
            .copied()
            .chain(whens)
            .collect();
    all.sort_unstable();
    all.dedup();
    all.into_iter().any(|g| {
        x.st.bools
            .support(g)
            .iter()
            .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
    })
}

/// A wave64 compute program that needs the console's lane layout or one
/// wave per group: it declares a wave size, 32 on halves or else 64.
fn wave64_lanes(x: &Parts) -> bool {
    let StageIo::Cs(cs) = x.native.io else {
        return false;
    };
    let one_wave = cs.threads.iter().product::<u32>() <= 64;
    x.native.wave64 && (reads_console_lanes(x) || one_wave && cs.lds_bytes() > 0)
}

/// Whether the console waves of a wave64 compute group never meet, so that
/// one host group can play them one after the other: whole console waves,
/// no LDS or GDS, no `s_barrier`, and in loops nothing through which a
/// wave could wait for another (an atomic, a coherent `glc` or `dlc`
/// access, a cache invalidation).
fn apart_waves(x: &Parts) -> bool {
    let StageIo::Cs(cs) = x.native.io else {
        return false;
    };
    let threads: u32 = cs.threads.iter().product();
    let looped = super::halves::looping_blocks(x.p, x.st);
    let shares = |i: &Inst| match &i.fields {
        Fields::Ds { .. } => !matches!(
            i.name(),
            "ds_swizzle_b32" | "ds_permute_b32" | "ds_bpermute_b32"
        ),
        Fields::Buffer { lds, .. } | Fields::Flat { lds, .. } => *lds,
        _ => i.name() == "s_barrier",
    };
    let waits = |i: &Inst| {
        let coherent = match &i.fields {
            Fields::Smem { cache, .. }
            | Fields::Buffer { cache, .. }
            | Fields::Image { cache, .. }
            | Fields::Flat { cache, .. } => cache.glc || cache.dlc,
            _ => false,
        };
        coherent || i.name().contains("atomic") || i.name().ends_with("_inv")
    };
    threads.is_multiple_of(64)
        && cs.lds_bytes() == 0
        && x.p.blocks.iter().enumerate().all(|(b, block)| {
            block
                .insts
                .iter()
                .all(|i| !shares(i) && !(looped[b] && waits(i)))
        })
}

/// The low dwords of the lane masks some instruction reads as data.
fn data_masks(p: &Program, st: &Structured) -> HashSet<Value> {
    let mut out = HashSet::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        if !reads_mask_as_data(st, i) {
            continue;
        }
        for &(r, v) in &i.reads {
            if !matches!(r, Reg::S(_)) {
                continue;
            }
            if st.masks.contains_key(&v) {
                out.insert(v);
            } else if let Some(&lo) = st.highs.get(&v) {
                out.insert(lo);
            }
        }
    }
    out
}

/// A wave64 compute program on 32-lane halves ([`super::halves`]).
struct Halves {
    /// Per IR block: every thread of the group gets there equally often, so
    /// the halves can meet there.
    meet: Vec<bool>,
    /// Per IR block: a loop can come back to it.
    looped: Vec<bool>,
    /// Per IR block: one lane of each console wave gets there at most
    /// ([`super::halves::alone_blocks`]): a `v_readfirstlane` there under
    /// no guard reads the lane's own value.
    alone: Vec<bool>,
    /// Masks read as data (low dwords): balloted where they are written.
    data_masks: HashSet<Value>,
    /// One console wave per group with LDS: a barrier before each access.
    lds_barriers: bool,
    /// The console thread each host thread plays (programs reading other
    /// lanes; made by the prologue).
    ids: Option<HalfIds>,
    /// Group-shared dwords the exchanges take.
    dwords: u32,
    /// One console wave per group, which both halves run in lockstep: the
    /// console wave's branches (on values every thread has alike), EXEC
    /// and every lane mask a value, each vector instruction under the
    /// thread's own bit of EXEC; so every block meets.
    lockstep: bool,
    /// Lockstep in a group of several console waves that never meet.
    serial: Option<Serial>,
    /// Lockstep in a group of several console waves that may meet: the two
    /// halves of a console wave wait for each other through a pair barrier
    /// of their own ([`Lower::pair_barrier`]) where a one-wave group's
    /// halves take a group barrier; the console's `s_barrier` stays a
    /// group barrier.
    pairs: bool,
}

/// Lockstep for a group of several console waves that never meet
/// ([`apart_waves`]): the host group is one console wave, whose halves
/// play the group's waves one after the other, around a loop over them.
#[derive(Debug, Clone, Copy)]
struct Serial {
    /// The console's group.
    threads: [u32; 3],
    /// Its console waves.
    waves: u32,
    /// The DXIL block where a wave's turn ends (made by [`Lower::turns`]).
    next: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct HalfIds {
    /// The console thread's index in the group.
    thread: V,
    /// Its console lane (0-63), half (0 or 1) and wave in the host group.
    lane: V,
    half: V,
    wave: V,
    /// Whether its console wave has lanes 32-63; none when all do.
    upper: Option<V>,
    /// Pairs: the group-shared counter of its console wave's pair barrier.
    pair: Option<V>,
}

/// Why a value from the other half has no form: the halves do not meet.
const APART: Why = "a cross-lane value where the two 32-lane halves do not meet";
const APART_MASK: Why = "a lane mask read as data where the two 32-lane halves do not meet";
const APART_LDS: Why = "one-wave LDS where the two 32-lane halves do not meet";
const NO_ROOM: Why = "LDS with no room for the 32-lane halves' exchanges";

/// Where each half of a parameter holding 16-bit values two per dword
/// comes from: for half h (dword h / 2, the high half when odd), a row and
/// channel of the signature, none for padding.
type HalfSources = [Option<(u8, u8)>; 8];

/// Rows of 16-bit values two per dword (`packed`), each with the row
/// packed after it: the next 16-bit row of the signature (`elements`) that
/// no stage data entry (`named`) names, taken in row order.
fn packed_rows(elements: &[SigElement], named: &[u8], packed: &[u8]) -> Vec<(u8, Option<u8>)> {
    let mut free: Vec<u8> = elements
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|r| !named.contains(r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut packed = packed.to_vec();
    packed.sort_unstable();
    packed
        .into_iter()
        .map(|a| (a, free.iter().position(|&r| r > a).map(|i| free.remove(i))))
        .collect()
}

/// The halves of the parameters packing `rows` (see [`packed_rows`]), by
/// first row. Channel c of the first row is half c and the second row
/// starts at the dword after the first's last channel, but an element
/// starting in a high half takes the next whole dword instead; one that
/// then passes its row's two dwords moves to the first later parameter
/// with whole dwords to spare, after its rows (as the ped programs' 1-channel
/// TEXCOORD6 after the 3-channel TEXCOORD4 does).
fn packed_halves(elements: &[SigElement], rows: &[(u8, Option<u8>)]) -> HashMap<u8, HalfSources> {
    let mut moved: Vec<(u8, &SigElement)> = Vec::new();
    let mut out = HashMap::new();
    for &(a, b) in rows {
        let mut halves = [None; 8];
        let mut used = 0;
        let mut out_of_row = Vec::new();
        for r in [a].into_iter().chain(b) {
            let mut row: Vec<_> = elements
                .iter()
                .filter(|e| e.row == Some(u32::from(r)))
                .collect();
            row.sort_by_key(|e| e.col);
            let mut end = 0;
            for e in row {
                let at = e.col.next_multiple_of(2);
                if at + e.cols > 4 {
                    out_of_row.push((r, e));
                    continue;
                }
                for k in 0..e.cols {
                    halves[usize::from(used + at + k)] = Some((r, e.col + k));
                }
                end = at + e.cols;
            }
            used += end.next_multiple_of(2);
        }
        moved.retain(|&(r, e)| {
            if used + e.cols > 8 {
                return true;
            }
            for k in 0..e.cols {
                halves[usize::from(used + k)] = Some((r, e.col + k));
            }
            used += e.cols.next_multiple_of(2);
            false
        });
        moved.extend(out_of_row);
        out.insert(a, halves);
    }
    out
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]).
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, Option<HalfSources>> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex: Vec<_> = entries.iter().filter(|o| o.param & 0x80 == 0).collect();
    let named: Vec<u8> = per_vertex.iter().map(|o| o.row).collect();
    let packed: Vec<u8> = per_vertex
        .iter()
        .filter(|o| o.fp16 && o.pack2)
        .map(|o| o.row)
        .collect();
    let halves = packed_halves(outputs, &packed_rows(outputs, &named, &packed));
    per_vertex
        .iter()
        .filter(|o| o.fp16)
        .map(|o| (o.row, o.pack2.then(|| halves[&o.row])))
        .collect()
}

/// Pixel inputs of 16-bit values two per dword (see
/// [`Lower::half_inputs`]): none where the stage data names a second row
/// other than the one packed after.
fn half_inputs(io: &StageIo, elements: &[SigElement]) -> HashMap<u8, HalfSources> {
    let StageIo::Ps { inputs, .. } = io else {
        return HashMap::new();
    };
    let named: Vec<u8> = inputs.iter().map(|i| i.row).collect();
    let packed: Vec<u8> = inputs
        .iter()
        .filter(|i| i.fp16 && i.pack2)
        .map(|i| i.row)
        .collect();
    let rows = packed_rows(elements, &named, &packed);
    let agree = inputs
        .iter()
        .filter(|i| i.fp16 && i.pack2 && i.row2 != i.row)
        .all(|i| rows.contains(&(i.row, Some(i.row2))));
    if !agree {
        return HashMap::new();
    }
    packed_halves(elements, &rows)
}

/// A scalar a waterfall loop reads from a descriptor and code after the loop
/// reads, where it cannot be the loop's last pass's ([`Lower::last_pass_exact`]).
const LAST_PASS: Why = "a scalar a waterfall loop reads from a descriptor, used after the loop";
/// [`LAST_PASS`] where only the host's lanes stand in the way: [`lower_as`]
/// lowers the program again on the console's.
const HOST_LANES: Why = "a scalar a waterfall loop leaves, off the console's lanes";

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// One search of [`Lower::bound`].
struct Bounding {
    /// M0-relative moves are taken to change no register past this index.
    past: u32,
    /// The moves the search took so (block and instruction).
    used: Vec<Site>,
    /// The values being bounded.
    path: Vec<Value>,
    budget: usize,
}

/// What a value made from the dwords of an image descriptor read as data
/// holds. The host's dwords describe the view as a texture of its own: its
/// first level's size, BASE_LEVEL 0 (see `descriptor_data`); the console's,
/// the size of the texture's first level and the view's first level. The
/// console compiler takes `GetDimensions` as (WIDTH + 1) >> BASE_LEVEL, and
/// the same of HEIGHT, unclamped: both give the size of the view's first
/// level that way, except where the console's comes out 0 (a view whose
/// first level is past that side's last, 1 on the host). That size and bits
/// of it are numbers like any other; any other read of WIDTH, HEIGHT or
/// BASE_LEVEL is refused ([`FIELDS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dim {
    Bits(Bits),
    /// WIDTH or HEIGHT of the descriptor loaded at a site, plus 1.
    Size(Site),
    /// That descriptor's BASE_LEVEL as an `s_bfe_u32` control's offset,
    /// with a constant width.
    Control(Site),
    /// Spilled or joined with other values.
    Joined,
}

/// Bits `start..start + len` of the image descriptor loaded at `site`, at
/// bit `at` of a dword, the others 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Bits {
    site: Site,
    start: u32,
    len: u32,
    at: u32,
}

impl Bits {
    fn shr(self, n: u32) -> Bits {
        match n.checked_sub(self.at) {
            None => Bits {
                at: self.at - n,
                ..self
            },
            Some(k) => Bits {
                start: self.start + k,
                len: self.len.saturating_sub(k),
                at: 0,
                ..self
            },
        }
    }

    fn shl(self, n: u32) -> Bits {
        let at = (self.at + n).min(32);
        Bits {
            len: self.len.min(32 - at),
            at,
            ..self
        }
    }

    /// Bits `at..at + len` of these, in place.
    fn field(self, at: u32, len: u32) -> Bits {
        let low = self.shr(at);
        Bits {
            len: low.len.min(len.saturating_sub(low.at)),
            ..low
        }
        .shl(at)
    }

    /// Two runs of bits that adjoin, in the descriptor and in the dword.
    fn or(self, b: Bits) -> Option<Bits> {
        let (lo, hi) = if self.at <= b.at {
            (self, b)
        } else {
            (b, self)
        };
        match (lo.len, hi.len) {
            (0, _) => Some(hi),
            (_, 0) => Some(lo),
            _ => (lo.site == hi.site && lo.at + lo.len == hi.at && lo.start + lo.len == hi.start)
                .then_some(Bits {
                    len: lo.len + hi.len,
                    ..lo
                }),
        }
    }

    /// Exactly the field `(start, len)` of the descriptor.
    fn is(self, (start, len): (u32, u32)) -> bool {
        (self.start, self.len, self.at) == (start, len, 0)
    }

    /// Whether any of these is in the field `(start, len)`.
    fn touches(self, (start, len): (u32, u32)) -> bool {
        self.len > 0 && self.start < start + len && start < self.start + self.len
    }
}

/// GFX10 image descriptor fields by first bit and width, as the console
/// compiler extracts them.
const WIDTH: (u32, u32) = (62, 16);
const HEIGHT: (u32, u32) = (78, 16);
const BASE_LEVEL: (u32, u32) = (108, 4);

/// See [`Dim`].
const FIELDS: Why =
    "an image's size or first level read from its descriptor other than as its first level's size";

/// The values an instruction writes, each with its [`Dim`] (none: a number).
type Dims = Vec<(Value, Option<Dim>)>;

impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>, split: Split, lanes: bool) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let mut console_lanes = false;
        let mut halves = None;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so does LDS shared within one wave, where the
            // console compiler leaves out barriers. In lockstep, EXEC itself
            // is console lanes.
            // `lanes`: a loop's scalars read after it are its last pass's
            // (`HOST_LANES`).
            let lockstep = split == Split::Lockstep;
            console_lanes = reads_console_lanes(x) || lockstep || lanes;
            let wave = if x.native.wave64 { 64 } else { 32 };
            let threads: u32 = cs.threads.iter().product();
            let one_wave = threads <= wave;
            if console_lanes || one_wave && s.lds_dwords > 0 {
                s.wave_size = Some(wave);
                if split != Split::No && wave == 64 {
                    s.wave_size = Some(32);
                    // Several console waves in lockstep: the host group is
                    // one, which plays them in turn.
                    let serial = (lockstep && !one_wave && apart_waves(x)).then(|| {
                        s.threads = Some([64, 1, 1]);
                        Serial {
                            threads: cs.threads,
                            waves: threads / 64,
                            next: None,
                        }
                    });
                    let pairs = lockstep && !one_wave && serial.is_none();
                    // A wave's lanes have their own x index. (In lockstep
                    // every lane goes where its wave does: none is alone.)
                    let distinct = (!lockstep
                        && (cs.threads[0] % 64 == 0 || cs.threads[1] * cs.threads[2] == 1))
                        .then(|| {
                            x.io.inputs
                                .iter()
                                .find(|i| i.1 == Input::ThreadIdInGroup(0))
                                .map(|i| i.0)
                        })
                        .flatten();
                    let alone =
                        super::halves::alone_blocks(p, x.st, distinct, &x.res.waterfall.reads);
                    // A first lane read as the lane's own differs between
                    // threads (see `Halves::alone`).
                    let mut lane_values = x.res.waterfall.reads.clone();
                    for (b, block) in p.blocks.iter().enumerate().filter(|b| alone[b.0]) {
                        for (i, inst) in block.insts.iter().enumerate() {
                            if inst.name() == "v_readfirstlane_b32" && x.st.guards[b][i].is_none() {
                                lane_values.extend(inst.writes.iter().map(|w| w.1));
                            }
                        }
                    }
                    halves = Some(Halves {
                        meet: if lockstep {
                            vec![true; p.blocks.len()]
                        } else {
                            super::halves::meeting_blocks(p, x.st, one_wave, &lane_values)
                        },
                        alone,
                        looped: super::halves::looping_blocks(p, x.st),
                        data_masks: if lockstep {
                            HashSet::new()
                        } else {
                            data_masks(p, x.st)
                        },
                        lds_barriers: s.lds_dwords > 0 && (one_wave && threads > 32 || pairs),
                        ids: None,
                        dwords: 0,
                        lockstep,
                        serial,
                        pairs,
                    });
                }
            }
        }
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            pair_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            src0: None,
            console_lanes,
            flat: None,
            no_barrier,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            half_inputs: half_inputs(&x.native.io, &x.inputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            halves,
            ballot_fail: HashMap::new(),
            last_pass_loops: HashSet::new(),
            last_pass: HashMap::new(),
            dims: HashMap::new(),
            phi_sites: HashMap::new(),
            #[cfg(test)]
            halves_barriers: 0,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// The bytes past `least` of the address a phi of `block` joins (see
    /// [`Resources::joined`]): each predecessor computes its address's at
    /// its end, and the join picks one. An edge no lane reading it takes
    /// gives 0.
    fn joined_offset(
        &mut self,
        block: usize,
        addresses: &[Option<Address>],
        least: u32,
    ) -> Result<V, Why> {
        let mut incoming = Vec::new();
        for (&q, a) in self.x.p.blocks[block].preds.iter().zip(addresses) {
            if self.block[q].is_none() {
                continue;
            }
            let Some(tail) = self.tail[q] else {
                return Err("an address joined by a loop");
            };
            let at = self.edges.get(&(q, block)).copied().unwrap_or(tail);
            let Some(a) = a else {
                incoming.push((self.s.m.i32(0), at));
                continue;
            };
            let here = std::mem::replace(&mut self.cur, at);
            let mut off = self.s.m.i32(a.offset.wrapping_sub(least));
            let d = a.dynamic.map(|d| self.i32_of(d)).transpose();
            if let Ok(Some(d)) = d {
                off = self.s.m.bin(at, Bin::Add, off, d);
            }
            self.cur = here;
            d?;
            incoming.push((off, at));
        }
        let join = self.block[block].ok_or("an address joined in a block not lowered")?;
        let phi = self.s.m.phi(join, self.s.t.i32);
        self.s.m.set_phi(phi, incoming);
        Ok(phi)
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        if a.param == BVH {
            let zero = self.s.m.i32(0);
            for k in 0..n {
                self.write(inst, Reg::V(reg + k as u8), zero, guard);
            }
            return Ok(());
        }
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        let x = self.x;
        if let Some((_, block, addresses)) = x.res.joined.iter().find(|j| j.0 == (b, i)) {
            let d = self.joined_offset(*block, addresses, a.offset)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        } else if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be. In
    /// lockstep, where lane masks are values, scalar instructions (those
    /// that make the section's masks from the saved EXEC) run as anywhere.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) -> Result<(), Stop> {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        if self.lockstep() && matches!(inst.fields, Fields::Sop { .. }) {
            return self.inst(b, i);
        }
        if inst.reads.iter().any(|r| self.dims.contains_key(&r.1)) {
            return Err(unsupported(inst.machine.offset, FIELDS));
        }
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
        Ok(())
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let on = self.mask_bit(mask, "a wave reduction's untracked mask")?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
            Shape::F16Sum(src, mask, ident, high) => return self.f16_sum(src, mask, ident, high),
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let on = self.mask_bit(mask, "a wave reduction's untracked mask")?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (_, None) => self.wave_active(op, arg),
        };
        let r = if float { self.as_int(r) } else { r };
        let Some(ids) = self.halves.as_ref().map(|h| h.ids) else {
            return Ok(r);
        };
        let ids = ids.ok_or(APART)?;
        // Halves: a reduction combines both halves' results; a prefix adds
        // the lower half's total in the upper half.
        if prefix.is_none() {
            let [(low, high)] = self.exchange([r], APART)?;
            let high = match ids.upper {
                Some(u) => {
                    let id = self.s.m.i32(ident);
                    self.s.m.select(self.cur, u, high, id)
                }
                None => high,
            };
            return Ok(self.combine(op, low, high));
        }
        let total = self.wave_active(op, arg);
        let total = if float { self.as_int(total) } else { total };
        let [(low, _)] = self.exchange([total], APART)?;
        // -0.0 leaves every float sum as it is.
        let none = self.s.m.i32(if float { 0x8000_0000 } else { 0 });
        let z = self.s.m.i32(0);
        let c = self.cur;
        let upper = self.s.m.cmp(c, pred::NE, ids.half, z);
        let below = self.s.m.select(c, upper, low, none);
        Ok(self.combine(op, r, below))
    }

    /// `op` over the host wave's active lanes (`arg` a float for float
    /// operations).
    fn wave_active(&mut self, op: super::wave::Op, arg: V) -> V {
        use super::wave::Op;
        let c = self.cur;
        let i32t = self.s.t.i32;
        if let Op::Or | Op::And | Op::Xor = op {
            let k = self.s.m.i8(match op {
                Op::And => 0,
                Op::Or => 1,
                _ => 2,
            });
            return self
                .s
                .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k]);
        }
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (ty, sfx) = if float {
            (self.s.t.f32, "f32")
        } else {
            (i32t, "i32")
        };
        let (code, unsigned) = match op {
            Op::AddU => (0, 1),
            Op::AddF => (0, 0),
            Op::MinU => (2, 1),
            Op::MaxU => (3, 1),
            Op::MinI | Op::MinF => (2, 0),
            _ => (3, 0),
        };
        let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
        self.s.op(
            c,
            &format!("waveActiveOp.{sfx}"),
            Attr::NoUnwind,
            ty,
            119,
            &[arg, k, u],
        )
    }

    /// `a` `op` `b` for two dwords (float bits for float operations), as
    /// the console's VALU computes it.
    fn combine(&mut self, op: super::wave::Op, a: V, b: V) -> V {
        use super::wave::Op;
        let c = self.cur;
        let pick = |l: &mut Self, p: u8| {
            let lt = l.s.m.cmp(c, p, a, b);
            l.s.m.select(c, lt, a, b)
        };
        match op {
            Op::AddU => self.s.m.bin(c, Bin::Add, a, b),
            Op::Or => self.s.m.bin(c, Bin::Or, a, b),
            Op::And => self.s.m.bin(c, Bin::And, a, b),
            Op::Xor => self.s.m.bin(c, Bin::Xor, a, b),
            Op::MinU => pick(self, pred::IULT),
            Op::MaxU => pick(self, pred::IUGT),
            Op::MinI => pick(self, pred::SLT),
            Op::MaxI => pick(self, pred::SGT),
            Op::AddF | Op::MinF | Op::MaxF => {
                let (fa, fb) = (self.as_float(a), self.as_float(b));
                let r = match op {
                    Op::AddF => self.s.m.bin(c, Bin::Add, fa, fb),
                    _ => self.max_min(op == Op::MaxF, fa, fb),
                };
                self.as_int(r)
            }
        }
    }

    /// [`Shape::F16Sum`](super::wave::Shape::F16Sum) in lockstep, where
    /// every lane of both halves runs: the console's tree in order (in each
    /// half lanes l and l ^ n for n = 1, 2, 4, 8, 16, then the halves),
    /// each sum exact in f32 and rounded to the nearest even f16 as
    /// `v_add_f16` rounds it ([`Self::f16_nearest`]); in the high half,
    /// lane `high`'s.
    fn f16_sum(&mut self, src: Value, mask: Value, ident: u32, high: u32) -> Result<V, Why> {
        if !self.lockstep() {
            return Err("an f16 wave sum outside lockstep");
        }
        let ids = self.halves.as_ref().and_then(|h| h.ids).ok_or(APART)?;
        let c = self.cur;
        let on = self.mask_bit(mask, "a wave reduction's untracked mask")?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let add = |l: &mut Self, a: V, b: V| {
            let (c, f32t) = (l.cur, l.s.t.f32);
            let fa = l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[a]);
            let fb = l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[b]);
            let r = l.s.m.bin(c, Bin::Add, fa, fb);
            l.f16_nearest(r)
        };
        let low = self.s.m.i32(0xFFFF);
        let mut sum = self.s.m.bin(c, Bin::And, x, low);
        for n in [1, 2, 4, 8, 16] {
            let n = self.s.m.i32(n);
            let other = self.s.m.bin(c, Bin::Xor, ids.lane, n);
            let theirs = self.read_lane(sum, other);
            sum = add(self, sum, theirs);
        }
        let lane = self.s.m.i32(high);
        let top = self.read_lane(x, lane);
        let [(sum0, sum1), (top0, top1)] = self.exchange([sum, top], APART)?;
        // A console wave with no upper half adds its 32 lanes of `ident`.
        let (sum1, top1) = match ids.upper {
            None => (sum1, top1),
            Some(u) if ident & 0x7FFF == 0 => {
                let (c, empty) = (self.cur, self.s.m.i32(ident & 0xFFFF));
                (
                    self.s.m.select(c, u, sum1, empty),
                    self.s.m.select(c, u, top1, id),
                )
            }
            Some(_) => return Err("an f16 wave sum of a nonzero identity over a half wave"),
        };
        let total = add(self, sum1, sum0);
        let c = self.cur;
        let keep = self.s.m.i32(0xFFFF_0000);
        let top = if high < 32 { top0 } else { top1 };
        let top = self.s.m.bin(c, Bin::And, top, keep);
        Ok(self.s.m.bin(c, Bin::Or, top, total))
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.cur_ir = order[0];
        self.find_sections();
        if self.halves.is_some() && self.console_lanes {
            self.halves_prologue().map_err(|w| unsupported(0, w))?;
        }
        if self.halves.as_ref().is_some_and(|h| h.serial.is_some()) {
            if self.counters.contains_key(&order[0]) {
                return Err(unsupported(0, "a loop back to the start of waves in turn"));
            }
            self.turns();
            self.block[order[0]] = Some(self.cur);
        }
        self.inputs()?;
        for &(_, v) in &p.inputs {
            self.ballot_where_written(v);
        }
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if (c == EXEC || c == EXEC + 1) && !self.lockstep() => {
                        Val::Missing("a lane mask")
                    }
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
                if let Some(d) = self.phi_dim(b, phi.value, &phi.args) {
                    self.dims.insert(phi.value, d);
                }
            }
            for phi in &p.blocks[b].phis {
                self.ballot_where_written(phi.value);
            }
            self.last_pass_after(b)
                .map_err(|w| unsupported(p.blocks[b].offset, w))?;
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                self.inst(b, i)?;
                for &(_, w) in &p.blocks[b].insts[i].writes {
                    self.ballot_where_written(w);
                }
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i)?;
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let cmp_samplers = self.s.cmp_samplers;
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            view_strides: self.s.view_strides.is_some(),
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            halves_barriers: self.halves_barriers,
            #[cfg(test)]
            nonuniform_outside_exec: self
                .s
                .nonuniform_indices
                .iter()
                .map(|&i| self.s.m.second_arms(i))
                .collect(),
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
            waves: match &self.halves {
                Some(h) if h.lockstep => super::Waves::Lockstep,
                Some(_) => super::Waves::Halves,
                None => super::Waves::Any,
            },
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                Reg::S(n) if (n == EXEC || n == EXEC + 1) && self.lockstep() => {
                    // Every lane of the group's one console wave.
                    let threads: u32 = self.s.threads.expect("compute").iter().product();
                    let lanes = if n == EXEC {
                        threads.min(32)
                    } else {
                        threads.saturating_sub(32).min(32)
                    };
                    Val::I32(self.s.m.i32(((1u64 << lanes) - 1) as u32))
                }
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            // Channels y and z of SV_Barycentrics (Belobug 416ac79). With
            // `VICTORY_NO_BELOBUG=barycentrics` they read 0, as before.
            Input::Barycentric { kind, index } if super::belobug("barycentrics") => {
                return self.barycentric(kind, index);
            }
            Input::Barycentric { .. } => return Val::I32(self.s.m.i32(0)),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        match self.last_pass.get(&v) {
            Some(&(_, Val::Missing(w))) => return Err(w),
            Some(&(_, val)) => return Ok(val),
            None => {}
        }
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = !self.lockstep()
            && (self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v));
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values
                // (Belobug's tip).
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let (x, y) = self.mask_dwords(lo)?;
        Ok(if high { y } else { x })
    }

    /// Both dwords of the lane mask whose low dword is `lo` (see
    /// [`Lower::mask_dword`]).
    fn mask_dwords(&mut self, lo: Value) -> Result<(V, V), Why> {
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if let Some(&xy) = self.ballots.get(&lo) {
            return Ok(xy);
        }
        if !self.x.st.active.contains(&lo) {
            if self.halves.is_some() || self.mask_data != MaskData::Console {
                return Err(ELSEWHERE);
            }
            let xy = self.mask_elsewhere(lo)?;
            self.ballots.insert(lo, xy);
            return Ok(xy);
        }
        if self.halves.is_some() {
            // Halves: only the ballot taken where the mask is written.
            return Err(self.ballot_fail.get(&lo).copied().unwrap_or(APART_MASK));
        }
        let at = match self.def_block.get(&lo) {
            // Program inputs (EXEC on entry).
            None => self.s.entry,
            Some(&b) if b == self.cur_ir => self.cur,
            Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let bit = self.boolean(m, at);
        self.balloting.remove(&lo);
        let bit = bit?;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
        self.ballots.insert(lo, xy);
        Ok(xy)
    }

    /// Compute: a lane mask with lanes outside EXEC where it is written (no
    /// ballot there sees them) as the console's scalar unit makes it from
    /// data: a copy of a mask ([`Lower::copied_mask`]), or a bitwise
    /// operation of masks and other scalars.
    fn mask_elsewhere(&mut self, lo: Value) -> Result<(V, V), Why> {
        let root = self.copied_mask(lo, &mut Vec::new()).unwrap_or(lo);
        if root != lo {
            return self.mask_dwords(root);
        }
        let p = self.x.p;
        let b = *self.def_block.get(&lo).ok_or(ELSEWHERE)?;
        let inst: &'a Inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))
            .ok_or(ELSEWHERE)?;
        let Fields::Sop { src, .. } = &inst.fields else {
            return Err(ELSEWHERE);
        };
        let op = inst
            .name()
            .strip_prefix("s_")
            .and_then(|n| n.strip_suffix(if p.wave64 { "_b64" } else { "_b32" }))
            .filter(|op| {
                matches!(
                    *op,
                    "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not"
                )
            })
            .ok_or(ELSEWHERE)?;
        let at = if b == self.cur_ir {
            self.cur
        } else {
            self.tail[b].ok_or("a lane mask written after its use")?
        };
        let mut words = Vec::new();
        for s in src {
            words.push(match *s {
                Operand::Scalar { code, .. } if code != NULL => {
                    let x = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                    if self.is_mask(Some(x)) {
                        self.mask_dwords(x)?
                    } else if p.wave64 {
                        let y = inst.read(Reg::S(code + 1)).ok_or("an unread operand")?;
                        (self.i32_of(x)?, self.i32_of(y)?)
                    } else {
                        (self.i32_of(x)?, self.s.m.i32(0))
                    }
                }
                Operand::Const(c) => {
                    let x = crate::resource::inline(c).ok_or("a special operand")?;
                    let y = if (x as i32) < 0 { u32::MAX } else { 0 };
                    (self.s.m.i32(x), self.s.m.i32(y))
                }
                Operand::Literal(x) => (self.s.m.i32(x), self.s.m.i32(0)),
                _ => return Err(ELSEWHERE),
            });
        }
        let ones = self.s.m.i32(u32::MAX);
        // s_not is the second operand all ones of an XOR.
        let [(a, c), (b, d)] = match words[..] {
            [x] if op == "not" => [x, (ones, ones)],
            [x, y] if op != "not" => [x, y],
            _ => return Err(ELSEWHERE),
        };
        // The operation, whether it inverts its second operand, its result.
        let (bin, not_b, not_r) = match op {
            "and" => (Bin::And, false, false),
            "or" => (Bin::Or, false, false),
            "andn2" => (Bin::And, true, false),
            "orn2" => (Bin::Or, true, false),
            "nand" => (Bin::And, false, true),
            "nor" => (Bin::Or, false, true),
            "xnor" => (Bin::Xor, false, true),
            _ => (Bin::Xor, false, false),
        };
        let mut dword = |a: V, b: V| {
            let m = &mut self.s.m;
            let b = if not_b {
                m.bin(at, Bin::Xor, b, ones)
            } else {
                b
            };
            let r = m.bin(at, bin, a, b);
            if not_r {
                m.bin(at, Bin::Xor, r, ones)
            } else {
                r
            }
        };
        Ok((dword(a, b), dword(c, d)))
    }

    /// The mask `v` copies: through `s_mov`, `v_readlane_b32` restoring
    /// a spill ([`Lower::restored`]) and joins whose incoming values all
    /// copy one mask (`seen`: joins on the way, where a
    /// loop back to one adds nothing); `v` when it copies none.
    fn copied_mask(&self, v: Value, seen: &mut Vec<Value>) -> Option<Value> {
        let p = self.x.p;
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            if seen.contains(&v) {
                return None;
            }
            seen.push(v);
            let mut root = None;
            for &a in &p.blocks[b].phis[i].args {
                match self.copied_mask(a, seen) {
                    None => {}
                    Some(r) if root.is_none_or(|x| x == r) => root = Some(r),
                    Some(_) => return Some(v),
                }
            }
            return Some(root.unwrap_or(v));
        }
        let Some(&b) = self.def_block.get(&v) else {
            return Some(v);
        };
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        let mov = if p.wave64 { "s_mov_b64" } else { "s_mov_b32" };
        match &inst.fields {
            Fields::Sop { src, .. } if inst.name() == mov => match src.first() {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let from = inst.read(Reg::S(code))?;
                    self.copied_mask(from, seen)
                }
                _ => Some(v),
            },
            Fields::Valu(_) => match self.restored(inst) {
                Some(Moved::Value(from)) => self.copied_mask(from, seen),
                _ => Some(v),
            },
            _ => Some(v),
        }
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: DXIL's
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `f` as an f16 in the low bits, rounded to nearest even (an overflow
    /// to infinity) as the console rounds f16 results (the MODE default):
    /// DXIL's `fptrunc` to half under native 16-bit types. The RTX 4090
    /// rounds it so; WARP rounds it toward zero, as both do
    /// `legacyF32ToF16`, which also turns an overflow into the largest
    /// finite half.
    fn f16_nearest(&mut self, f: V) -> V {
        let c = self.cur;
        self.s.flags |= flags::NATIVE_16;
        let (half, i16t) = (self.s.m.half(), self.s.m.int(16));
        let h = self.s.m.cast(c, Cast::FpTrunc, f, half);
        let h = self.s.m.cast(c, Cast::Bitcast, h, i16t);
        self.s.m.cast(c, Cast::ZExt, h, self.s.t.i32)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        if let Atom::Uniform(v) | Atom::Opaque(v) = atom
            && self.dims.contains_key(&v)
        {
            return Err(FIELDS);
        }
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => self.own_bit(v, at),
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// With console lanes: this lane's bit of the mask whose low dword is
    /// `v`, read from the number it holds.
    fn own_bit(&mut self, v: Value, at: usize) -> Result<V, Why> {
        let hi = match self.pair_high(v) {
            Some(h) => Some(self.get(h)?),
            None => None,
        };
        let lo = self.get(v)?;
        let word = |w: Val| match w {
            Val::I32(x) => Ok(x),
            _ => Err("an opaque lane mask"),
        };
        let lo = word(lo)?;
        let hi = hi.map(word).transpose()?;
        Ok(self.lane_bit(at, lo, hi))
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        let lane = match self.halves.as_ref().and_then(|h| h.ids) {
            Some(ids) => ids.lane,
            None => self.s.op(
                at,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            ),
        };
        self.lane_bit_of(at, lane, lo, hi)
    }

    /// Lane `lane`'s bit of the mask `lo`, `hi` (wave64).
    fn lane_bit_of(&mut self, at: usize, lane: V, lo: V, hi: Option<V>) -> V {
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    /// What `v_readlane_b32` of a constant lane reads where every path
    /// spilled one scalar there with `v_writelane_b32` (see [`restored`]).
    fn restored(&self, inst: &Inst) -> Option<Moved> {
        let p = self.x.p;
        let def = |v: Value| {
            let b = *self.def_block.get(&v)?;
            p.blocks[b]
                .insts
                .iter()
                .find(|i| i.writes.iter().any(|w| w.1 == v))
        };
        let args = |v: Value| {
            self.phi_at
                .get(&v)
                .map(|&(b, i)| &p.blocks[b].phis[i].args[..])
        };
        restored(inst, &def, &args)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        if self.lockstep() {
            let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
            return self.exec_bit(inst);
        }
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    /// The halves run in lockstep ([`Halves::lockstep`]).
    fn lockstep(&self) -> bool {
        self.halves.as_ref().is_some_and(|h| h.lockstep)
    }

    /// Lockstep: this thread's bit of the EXEC `inst` reads; none when that
    /// is the group's every lane, as on entry.
    fn exec_bit(&mut self, inst: &Inst) -> Result<Option<V>, Why> {
        let (Some(lo), Some(hi)) = (inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1))) else {
            return Ok(None);
        };
        let input = |v: Value| self.x.p.inputs.iter().any(|i| i.1 == v);
        if input(lo) && input(hi) {
            return Ok(None);
        }
        self.pair_bit(lo, hi).map(Some)
    }

    /// Lockstep: the dwords of the EXEC `inst` reads; none when that is
    /// the group's every lane.
    fn exec_words(&mut self, inst: &Inst) -> Result<Option<(V, V)>, Why> {
        let (Some(lo), Some(hi)) = (inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1))) else {
            return Ok(None);
        };
        let input = |v: Value| self.x.p.inputs.iter().any(|i| i.1 == v);
        if input(lo) && input(hi) {
            return Ok(None);
        }
        Ok(Some((self.i32_of(lo)?, self.i32_of(hi)?)))
    }

    /// Lockstep: this thread's bit of the lane mask held by `lo` and `hi`.
    fn pair_bit(&mut self, lo: Value, hi: Value) -> Result<V, Why> {
        if let Some(&x) = self.pair_bits.get(&(self.cur, lo, hi)) {
            return Ok(x);
        }
        let (l, h) = (self.i32_of(lo)?, self.i32_of(hi)?);
        let x = self.lane_bit(self.cur, l, Some(h));
        self.pair_bits.insert((self.cur, lo, hi), x);
        Ok(x)
    }

    /// This lane's bit of the lane mask whose low dword is `lo`.
    fn mask_bit(&mut self, lo: Value, untracked: Why) -> Result<V, Why> {
        if self.lockstep() {
            let hi = self.pair_high(lo).ok_or(untracked)?;
            return self.pair_bit(lo, hi);
        }
        let &m = self.x.st.masks.get(&lo).ok_or(untracked)?;
        self.boolean(m, self.cur)
    }

    /// This lane's bit of the lane mask `inst` reads from SGPRs `code` and
    /// `code + 1` (from `code` alone in wave32).
    fn read_mask_bit(&mut self, inst: &Inst, code: u8, untracked: Why) -> Result<V, Why> {
        let lo = inst.read(Reg::S(code)).ok_or(untracked)?;
        if self.lockstep() {
            let hi = inst.read(Reg::S(code + 1)).ok_or(untracked)?;
            return self.pair_bit(lo, hi);
        }
        self.mask_bit(lo, untracked)
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if let Some(ids) = self.halves.as_ref().and_then(|h| h.ids) {
            // Halves: the console thread the prologue gave this one.
            return Val::I32(self.group_coord(self.cur, ids.thread, c));
        }
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.lane_index();
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        Val::I32(self.group_coord(e, f, c))
    }

    /// Component `c` of the thread id of the thread with index `f` in the
    /// group, computed in block `at`.
    fn group_coord(&mut self, at: usize, f: V, c: u8) -> V {
        let [x, y, _] = match self.halves.as_ref().and_then(|h| h.serial) {
            Some(s) => s.threads,
            None => self.s.threads.expect("compute"),
        };
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        match c {
            0 => self.s.m.bin(at, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(at, Bin::UDiv, f, xk);
                self.s.m.bin(at, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(at, Bin::UDiv, f, xy),
        }
    }

    /// The number of console waves in a compute group.
    fn console_waves(&self) -> u32 {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        threads.div_ceil(64)
    }

    /// A group barrier that fences group-shared memory.
    fn group_barrier(&mut self) {
        #[cfg(test)]
        {
            self.halves_barriers += 1;
        }
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Halves: each host wave takes the next 32 console threads of the
    /// group from a group-shared counter, whichever threads the host packed
    /// into it ([`super::halves`]). Only for groups of whole host waves.
    fn halves_prologue(&mut self) -> Result<(), Why> {
        use super::bitcode::Rmw;
        let total: u32 = self.s.threads.expect("compute").iter().product();
        if !total.is_multiple_of(32) {
            return Err("a group of threads that is no multiple of 32 on 32-lane halves");
        }
        if self.s.lds_dwords >= 8192 {
            return Err(NO_ROOM);
        }
        let i32t = self.s.t.i32;
        // Pairs: a counter per console wave after the group's.
        let pairs = self
            .halves
            .as_ref()
            .is_some_and(|h| h.pairs)
            .then_some(total / 64);
        let dwords = 1 + pairs.unwrap_or(0);
        let arr = self.s.m.array(dwords, i32t);
        let area = self.s.m.global("halves", arr, 3);
        if let Some(h) = &mut self.halves {
            h.dwords = dwords;
        }
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let counter = self.s.m.element(c, area, zero);
        let lane = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        // Every thread clears the counter (and lane n of each host wave
        // pair counter n), then each wave's first lane adds 32 (the others
        // 0): it gets where the wave's threads start.
        self.s.m.store(c, counter, zero);
        let one = self.s.m.i32(1);
        if let Some(waves) = pairs {
            let waves = self.s.m.i32(waves);
            let n = self.s.m.bin(c, Bin::URem, lane, waves);
            let n = self.s.m.bin(c, Bin::Add, n, one);
            let p = self.s.m.element(c, area, n);
            self.s.m.store(c, p, zero);
        }
        self.group_barrier();
        let k32 = self.s.m.i32(32);
        let lead = self.s.m.cmp(c, pred::EQ, lane, zero);
        let add = self.s.m.select(c, lead, k32, zero);
        let taken = self.s.m.atomic(c, Rmw::Add, counter, add);
        let start = self.s.op(
            c,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            i32t,
            117,
            &[taken, zero],
        );
        let thread = self.s.m.bin(c, Bin::Add, start, lane);
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (k63, k5, k1, k6) = (k(self, 63), k(self, 5), k(self, 1), k(self, 6));
        let lane = self.s.m.bin(c, Bin::And, thread, k63);
        let half = self.s.m.bin(c, Bin::LShr, lane, k5);
        let half = self.s.m.bin(c, Bin::And, half, k1);
        let wave = self.s.m.bin(c, Bin::LShr, thread, k6);
        // The last console wave of a group of 64n + 32 threads has no upper
        // half.
        let upper = (total % 64 == 32).then(|| {
            let not63 = self.s.m.i32(!63);
            let start = self.s.m.bin(c, Bin::And, thread, not63);
            let up = self.s.m.bin(c, Bin::Add, start, k32);
            let total = self.s.m.i32(total);
            self.s.m.cmp(c, pred::IULT, up, total)
        });
        let pair = pairs.map(|_| {
            let n = self.s.m.bin(c, Bin::Add, wave, one);
            self.s.m.element(c, area, n)
        });
        self.s.flags |= flags::WAVE_OPS;
        if let Some(h) = &mut self.halves {
            h.ids = Some(HalfIds {
                thread,
                lane,
                half,
                wave,
                upper,
                pair,
            });
        }
        Ok(())
    }

    /// Halves: where the two halves of a console wave wait for each other
    /// (around an exchange, before an LDS access): a group barrier, or in
    /// a group of several console waves in lockstep their pair barrier.
    fn halves_barrier(&mut self) {
        match self
            .halves
            .as_ref()
            .and_then(|h| h.ids)
            .and_then(|i| i.pair)
        {
            Some(p) => self.pair_barrier(p),
            None => self.group_barrier(),
        }
    }

    /// Pairs ([`Halves::pairs`]): the two halves of this thread's console
    /// wave wait for each other through its counter `p` in group-shared
    /// memory. Each half's first lane adds 1 where it gets there, so after
    /// both halves' nth barrier the count is 2n; a half goes on once it
    /// reaches the end of the pair its own arrival belongs to. Both halves
    /// follow their wave's branches, so they get to the same barriers in
    /// the same order. Group-shared fences before and after make what one
    /// half wrote before visible to the other after.
    // shortcut: a half stops waiting after `PAIR_SPINS` reads, which only
    // bounds a fault: both host waves of a group are resident and run.
    fn pair_barrier(&mut self, p: V) {
        use super::bitcode::Rmw;
        const PAIR_SPINS: u32 = 1 << 16;
        let void = self.s.t.void;
        let i32t = self.s.t.i32;
        let fence = self.s.m.i32(8);
        let c = self.cur;
        self.s
            .op(c, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        let (zero, one) = (self.s.m.i32(0), self.s.m.i32(1));
        let lane = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let lead = self.s.m.cmp(c, pred::EQ, lane, zero);
        let add = self.s.m.select(c, lead, one, zero);
        let old = self.s.m.atomic(c, Rmw::Add, p, add);
        let old = self.s.op(
            c,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            i32t,
            117,
            &[old, zero],
        );
        let even = self.s.m.i32(!1);
        let start = self.s.m.bin(c, Bin::And, old, even);
        let two = self.s.m.i32(2);
        let end = self.s.m.bin(c, Bin::Add, start, two);
        let (spin, after) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(c, Term::Br(spin));
        let n = self.s.m.phi(spin, i32t);
        let now = self.s.m.atomic(spin, Rmw::Or, p, zero);
        let now = self.s.op(
            spin,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            i32t,
            117,
            &[now, zero],
        );
        let ahead = self.s.m.bin(spin, Bin::Sub, now, end);
        let met = self.s.m.cmp(spin, pred::SGE, ahead, zero);
        let next = self.s.m.bin(spin, Bin::Add, n, one);
        let limit = self.s.m.i32(PAIR_SPINS);
        let tired = self.s.m.cmp(spin, pred::IUGE, next, limit);
        let leave = self.s.m.bin(spin, Bin::Or, met, tired);
        self.s.m.terminate(spin, Term::CondBr(leave, after, spin));
        self.s.m.set_phi(n, vec![(zero, c), (next, spin)]);
        self.cur = after;
        self.s
            .op(after, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        self.s.flags |= flags::WAVE_OPS;
    }

    /// Lockstep in turn ([`Serial`]): a loop over the console group's
    /// waves around the program, made after the halves' prologue; this
    /// thread plays console thread turn * 64 + its own, and where the
    /// console wave ends, its turn does. A group barrier ends each turn, so
    /// every thread has read one wave's exchanges before any writes the
    /// next one's.
    fn turns(&mut self) {
        let Some((serial, ids)) = self.halves.as_ref().and_then(|h| Some((h.serial?, h.ids?)))
        else {
            return;
        };
        let entry = self.cur;
        let (head, next, done) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(entry, Term::Br(head));
        let turn = self.s.m.phi(head, self.s.t.i32);
        self.cur = next;
        self.group_barrier();
        let one = self.s.m.i32(1);
        let after = self.s.m.bin(next, Bin::Add, turn, one);
        let waves = self.s.m.i32(serial.waves);
        let more = self.s.m.cmp(next, pred::IULT, after, waves);
        self.s.m.terminate(next, Term::CondBr(more, head, done));
        self.s.m.terminate(done, Term::Ret);
        let zero = self.s.m.i32(0);
        self.s.m.set_phi(turn, vec![(zero, entry), (after, next)]);
        let k6 = self.s.m.i32(6);
        let first = self.s.m.bin(head, Bin::Shl, turn, k6);
        let thread = self.s.m.bin(head, Bin::Add, first, ids.thread);
        if let Some(h) = &mut self.halves {
            h.ids = Some(HalfIds { thread, ..ids });
            h.serial = Some(Serial {
                next: Some(next),
                ..serial
            });
        }
        self.cur = head;
    }

    /// Lockstep in turn: the block where the console wave's turn ends.
    fn next_turn(&self) -> Option<usize> {
        self.halves.as_ref()?.serial?.next
    }

    /// Halves: each half gives `words` (the same in all its lanes) and each
    /// lane gets both halves' (the lower's, the upper's), through
    /// group-shared memory of this instruction's own and a group barrier
    /// (and another after the reads where a loop comes back); only where
    /// every thread of the group gets equally often (else `why`).
    fn exchange<const N: usize>(&mut self, words: [V; N], why: Why) -> Result<[(V, V); N], Why> {
        let waves = self.console_waves();
        let at = self.cur_ir;
        let single = self.s.threads.expect("compute").iter().product::<u32>() == 32;
        let lds = self.s.lds_dwords;
        let h = self.halves.as_mut().ok_or(why)?;
        let ids = h.ids.ok_or(why)?;
        if single {
            // One host wave, the lower half: the upper one has no lanes.
            let z = self.s.m.i32(0);
            return Ok(words.map(|w| (w, z)));
        }
        if !h.meet[at] {
            return Err(why);
        }
        let size = N as u32 * waves * 2;
        if lds + h.dwords + size > 8192 {
            return Err(NO_ROOM);
        }
        h.dwords += size;
        let looped = h.looped[at];
        let arr = self.s.m.array(size, self.s.t.i32);
        let area = self.s.m.global(&format!("halves.{}", h.dwords), arr, 3);
        // Word k of half h of console wave w at (k * waves + w) * 2 + h.
        let c = self.cur;
        let two = self.s.m.i32(2);
        let pair = self.s.m.bin(c, Bin::Mul, ids.wave, two);
        let slot = |l: &mut Self, k: usize, at: V| {
            let base = l.s.m.i32(k as u32 * waves * 2);
            let i = l.s.m.bin(c, Bin::Add, at, base);
            l.s.m.element(c, area, i)
        };
        let mine = self.s.m.bin(c, Bin::Add, pair, ids.half);
        for (k, &w) in words.iter().enumerate() {
            let p = slot(self, k, mine);
            self.s.m.store(c, p, w);
        }
        self.halves_barrier();
        let c = self.cur;
        let one = self.s.m.i32(1);
        let upper = self.s.m.bin(c, Bin::Add, pair, one);
        let out = std::array::from_fn(|k| {
            let base = self.s.m.i32(k as u32 * waves * 2);
            let (i0, i1) = (
                self.s.m.bin(c, Bin::Add, pair, base),
                self.s.m.bin(c, Bin::Add, upper, base),
            );
            let (p0, p1) = (self.s.m.element(c, area, i0), self.s.m.element(c, area, i1));
            (self.s.m.load(c, p0), self.s.m.load(c, p1))
        });
        if looped {
            // Every thread has read before any writes again.
            self.halves_barrier();
        }
        Ok(out)
    }

    /// Halves: the ballot of a mask some instruction reads as data, taken
    /// where the mask is written so that the halves meet there.
    fn ballot_where_written(&mut self, lo: Value) {
        let wanted = self
            .halves
            .as_ref()
            .is_some_and(|h| h.data_masks.contains(&lo));
        if !wanted || self.ballots.contains_key(&lo) {
            return;
        }
        match self.half_ballot(lo) {
            Ok(xy) => {
                self.ballots.insert(lo, xy);
            }
            Err(why) => {
                self.ballot_fail.insert(lo, why);
            }
        }
    }

    fn half_ballot(&mut self, lo: Value) -> Result<(V, V), Why> {
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) {
            return Err(ELSEWHERE);
        }
        let bit = self.boolean(m, self.cur)?;
        self.ballot_halves(bit, APART_MASK)
    }

    /// Halves: the console wave's ballot of `bit` (its low dword, its high
    /// one), through an exchange (else `why`).
    fn ballot_halves(&mut self, bit: V, why: Why) -> Result<(V, V), Why> {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let mine = self.s.m.extract(c, r, 0);
        let [(low, high)] = self.exchange([mine], why)?;
        let upper = self
            .halves
            .as_ref()
            .and_then(|h| h.ids)
            .and_then(|i| i.upper);
        let high = match upper {
            Some(u) => {
                let z = self.s.m.i32(0);
                self.s.m.select(self.cur, u, high, z)
            }
            None => high,
        };
        Ok((low, high))
    }

    /// Halves: `x` in the console wave's first lane where `active` holds
    /// (none: every lane), the lower half's if it has one; lane 0's where
    /// it holds in none, as `v_readfirstlane` reads with EXEC zero.
    fn first_lane_of_halves(&mut self, x: V, active: Option<V>) -> Result<V, Why> {
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let Some(g) = active else {
            let w = self.read_lane(x, zero);
            let [(low, _)] = self.exchange([w], APART)?;
            return Ok(low);
        };
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[g]);
        let bits = self.s.m.extract(c, r, 0);
        // First bit low: -1 without one, which `read_lane` takes as lane 31.
        let first = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[bits]);
        let w = self.read_lane(x, first);
        // With no lane in EXEC, the console reads lane 0.
        let w0 = self.read_lane(x, zero);
        let [(low_bits, high_bits), (low, high), (lane0, _)] =
            self.exchange([bits, w, w0], APART)?;
        let c = self.cur;
        let any_high = self.s.m.cmp(c, pred::NE, high_bits, zero);
        let r = self.s.m.select(c, any_high, high, lane0);
        let any_low = self.s.m.cmp(c, pred::NE, low_bits, zero);
        Ok(self.s.m.select(c, any_low, low, r))
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if let Some(ids) = self.halves.as_ref().and_then(|h| h.ids) {
            return ids.lane;
        }
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// Whether a read of another lane of this lane's quad must reach helper
    /// lanes: in a pixel shader, always (Belobug 03629f3; off with
    /// `VICTORY_NO_BELOBUG=quad`). At a triangle's edge the quad's other
    /// pixels are helpers (and so are discarded ones), which only quad reads
    /// see: wave reads give 0 from them, so a derivative taken by hand
    /// (`quad_perm`) was the whole value there, the texture read that used
    /// it took its smallest level, and the triangles' edges showed as lines
    /// on car paint, glass and skin.
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel && super::belobug("quad")
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead (Belobug f520772, upstream
    /// 0601955; off with `VICTORY_NO_BELOBUG=readfirstlane`).
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out: once only helper lanes are
    /// left in such a loop, `WaveReadLaneFirst` reads no lane, no lane
    /// matches, and the loop runs to its limit in every such lane, with
    /// whatever descriptor the read gave. With its own value a helper lane
    /// matches in the first pass and leaves; what it computes only feeds
    /// derivatives (those of a texture read's dependent coordinates, at
    /// triangle edges), and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel || !super::belobug("readfirstlane") {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// `x` of lane `lane`; on halves, a console lane of this lane's half.
    fn read_lane(&mut self, x: V, lane: V) -> V {
        let lane = match self.halves.as_ref().and_then(|h| h.ids) {
            Some(_) => {
                let k31 = self.s.m.i32(31);
                self.s.m.bin(self.cur, Bin::And, lane, k31)
            }
            None => lane,
        };
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let shares = !(name == "ds_swizzle_b32" || name.contains("permute"));
        if let Some(h) = &self.halves
            && h.lds_barriers
            && shares
            && self.s.lds_dwords > 0
        {
            // Halves of one console wave: LDS accesses in program order.
            if !h.meet[self.cur_ir] {
                return Err(APART_LDS);
            }
            self.halves_barrier();
        }
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let mut r = if mesh_quad || self.helpers_read(quad) {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            // Lockstep: a source lane outside EXEC reads 0.
            if self.lockstep()
                && let Some((lo, hi)) = self.exec_words(inst)?
            {
                let on = self.lane_bit_of(c, src, lo, Some(hi));
                let z = k(self, 0);
                r = self.s.m.select(c, on, r, z);
            }
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if name == "ds_bpermute_b32" {
            // Each lane reads data0 of lane (address + offset) / 4 mod 32;
            // a source lane outside EXEC gives 0. No LDS is involved.
            if self.x.native.wave64 {
                return Err("ds_bpermute_b32 in a 64-lane wave");
            }
            self.cross_lane(false)?;
            let a = self.vgprs(inst, addr.as_ref().ok_or("a permute address")?, 1)?[0];
            let x = self.vgprs(inst, data.first().ok_or("a permute source")?, 1)?[0];
            let (o, two, k31) = (self.s.m.i32(offset), self.s.m.i32(2), self.s.m.i32(31));
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let src = self.s.m.bin(c, Bin::LShr, at, two);
            let src = self.s.m.bin(c, Bin::And, src, k31);
            let r = self.read_lane(x, src);
            // EXEC: the host lanes here, less those the guard leaves out.
            let active = match guard {
                Some(g) => g,
                None => self.s.m.bool(true),
            };
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let exec = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[active]);
            let exec = self.s.m.extract(c, exec, 0);
            let on = self.lane_bit_of(c, src, exec, None);
            let z = self.s.m.i32(0);
            let r = self.s.m.select(c, on, r, z);
            let reg = dst_reg.ok_or("a permute result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as dword-aligned.
        // A dword past the program's allocation reads 0 and takes no write,
        // as on the console (the host's group-shared memory would fault):
        // its pointer (dword 0's then) and whether it lies inside.
        let dword = |l: &mut Self, bytes: u32| -> (V, V) {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let size = l.s.m.i32(l.s.lds_dwords);
            let inside = l.s.m.cmp(c, pred::IULT, index, size);
            let z = l.s.m.i32(0);
            let index = l.s.m.select(c, inside, index, z);
            let lds = l.s.lds();
            (l.s.m.element(c, lds, index), inside)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                // Bytes and halves: the dword holding them, shifted down to
                // bit 0 and extended; the d16 forms keep the other half.
                let bits = match rest {
                    "_u8" | "_i8" | "_u8_d16" | "_u8_d16_hi" | "_i8_d16" | "_i8_d16_hi" => 8,
                    "_u16" | "_i16" | "_u16_d16" | "_u16_d16_hi" => 16,
                    _ => return Err("an LDS read of bytes or halves"),
                };
                let reg = dst_reg.ok_or("an LDS read result")?;
                let (p, inside) = dword(self, offset);
                let w = self.s.m.load(c, p);
                let z = self.s.m.i32(0);
                let w = self.s.m.select(c, inside, w, z);
                let o = self.s.m.i32(offset);
                let at = self.s.m.bin(c, Bin::Add, a, o);
                let align = self.s.m.i32(if bits == 8 { 3 } else { 2 });
                let at = self.s.m.bin(c, Bin::And, at, align);
                let three = self.s.m.i32(3);
                let sh = self.s.m.bin(c, Bin::Shl, at, three);
                let k32 = self.s.m.i32(32 - bits);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let x = self.s.m.bin(c, Bin::Shl, w, left);
                let signed = if rest.starts_with("_i") {
                    Bin::AShr
                } else {
                    Bin::LShr
                };
                let x = self.s.m.bin(c, signed, x, k32);
                let x = match rest.strip_suffix("_hi") {
                    _ if !rest.contains("_d16") => x,
                    Some(_) => {
                        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                            Some(Ok(o)) => o,
                            _ => self.s.m.i32(0),
                        };
                        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                        let old = self.s.m.bin(c, Bin::And, old, low);
                        let x = self.s.m.bin(c, Bin::Shl, x, k16);
                        self.s.m.bin(c, Bin::Or, old, x)
                    }
                    None => self.low_half(inst, reg, x),
                };
                self.write(inst, Reg::V(reg), x, guard);
                return Ok(());
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let (p, inside) = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    let z = self.s.m.i32(0);
                    let x = self.s.m.select(self.cur, inside, x, z);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                // Bytes and halves: cleared, then set, in the dword holding
                // them, each atomically, so lanes writing other bytes of the
                // same dword keep theirs.
                let bits = match rest {
                    "_b8" | "_b8_d16_hi" => 8,
                    "_b16" | "_b16_d16_hi" => 16,
                    _ => return Err("an LDS write of bytes or halves"),
                };
                let mut x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if rest.ends_with("_d16_hi") {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let o = self.s.m.i32(offset);
                let at = self.s.m.bin(c, Bin::Add, a, o);
                let align = self.s.m.i32(if bits == 8 { 3 } else { 2 });
                let at = self.s.m.bin(c, Bin::And, at, align);
                let three = self.s.m.i32(3);
                let sh = self.s.m.bin(c, Bin::Shl, at, three);
                let field = self.s.m.i32(((1u64 << bits) - 1) as u32);
                let x = self.s.m.bin(c, Bin::And, x, field);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let mask = self.s.m.bin(c, Bin::Shl, field, sh);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, mask, ones);
                let (p, inside) = dword(self, offset);
                use super::bitcode::Rmw;
                let at = self.begin_guarded(guard);
                let only = self.begin_guarded(Some(inside));
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, x);
                self.end_guarded(only, &[]);
                self.end_guarded(at, &[]);
                return Ok(());
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let (p, inside) = dword(self, off);
                let only = self.begin_guarded(Some(inside));
                self.s.m.store(self.cur, p, x);
                self.end_guarded(only, &[]);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        // Past the allocation an atomic changes nothing and returns 0.
        let (p, inside) = dword(self, offset);
        let z = self.s.m.i32(0);
        let (run, outside) = match guard {
            Some(g) => (
                self.s.m.bin(self.cur, Bin::And, g, inside),
                self.s.m.select(self.cur, g, z, outside),
            ),
            None => (inside, z),
        };
        let at = self.begin_guarded(Some(run));
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let dims = self.field_use(inst, self.x.st.guards[b][i].is_some());
        let r = dims
            .as_ref()
            .map_err(|&w| w)
            .and_then(|_| match &inst.fields {
                Fields::Sop { .. } => self.sop(inst),
                Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => {
                    Ok(())
                }
                Fields::Sopp { .. } if name == "s_barrier" => {
                    // Thread-group sync with a shared-memory fence, and a UAV
                    // fence when the program writes UAVs (the waits before an
                    // s_barrier make those writes visible on the console).
                    let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                    let mode = self.s.m.i32(mode);
                    let void = self.s.t.void;
                    self.s
                        .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                    Ok(())
                }
                Fields::Sopp { .. } => Ok(()),
                Fields::Smem { .. } => self.smem(inst, (b, i)),
                Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
                Fields::Valu(_) => self.valu(inst, b, i),
                Fields::Vintrp { .. } => self.interp(inst, (b, i)),
                Fields::Image { .. } => self.image(inst, b, i),
                Fields::Buffer { .. } => self.buffer(inst, b, i),
                Fields::Exp { .. } => self.export(b, i),
                Fields::Ds { .. } => self.ds(inst, b, i),
                Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
                Fields::Flat { .. } => Err("flat memory"),
            });
        if let (Ok(()), Ok(Some(outs))) = (&r, dims) {
            self.field_outputs(inst, &outs);
        }
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if !self.lockstep() && (inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo)) {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                _ if name.contains("_saveexec_") || name.contains("_wrexec_") => {
                    // Lockstep (refused above otherwise): EXEC becomes
                    // src0 op EXEC; the destination takes EXEC from before
                    // (saveexec) or after (wrexec); SCC: EXEC is not zero.
                    let (op, _) = name[2..]
                        .split_once("_saveexec")
                        .or(name[2..].split_once("_wrexec"))
                        .ok_or(name)?;
                    let ones = self.s.m.i32(u32::MAX);
                    let (mut old, mut new) = ([ones; 2], [ones; 2]);
                    for h in 0..2u8 {
                        let e = inst.read(Reg::S(EXEC + h)).ok_or("an unread EXEC")?;
                        let e = self.i32_of(e)?;
                        let a = half(self, 0, h)?;
                        let not = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Xor, x, ones);
                        let (a, e2, op, flip) = match op {
                            "and" => (a, e, Bin::And, false),
                            "or" => (a, e, Bin::Or, false),
                            "xor" => (a, e, Bin::Xor, false),
                            "andn2" => (a, not(self, e), Bin::And, false),
                            "orn2" => (a, not(self, e), Bin::Or, false),
                            "andn1" => (not(self, a), e, Bin::And, false),
                            "orn1" => (not(self, a), e, Bin::Or, false),
                            "nand" => (a, e, Bin::And, true),
                            "nor" => (a, e, Bin::Or, true),
                            "xnor" => (a, e, Bin::Xor, true),
                            _ => return Err(name),
                        };
                        let x = self.s.m.bin(c, op, a, e2);
                        new[usize::from(h)] = if flip { not(self, x) } else { x };
                        old[usize::from(h)] = e;
                    }
                    for h in 0..2u8 {
                        let x = new[usize::from(h)];
                        self.set(inst.write(Reg::S(EXEC + h)), Val::I32(x));
                    }
                    let saved = if name.contains("_saveexec_") {
                        old
                    } else {
                        new
                    };
                    self.set(lo, Val::I32(saved[0]));
                    self.set(hi, Val::I32(saved[1]));
                    let s = nonzero(self, new[0], new[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_wqm_b64" if self.lockstep() => {
                    // Every lane of a quad with a lane set.
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
                        let (one, two, quads, fill) =
                            (k(self, 1), k(self, 2), k(self, 0x1111_1111), k(self, 15));
                        let t = self.s.m.bin(c, Bin::LShr, a, one);
                        let t = self.s.m.bin(c, Bin::Or, a, t);
                        let u = self.s.m.bin(c, Bin::LShr, t, two);
                        let t = self.s.m.bin(c, Bin::Or, t, u);
                        let t = self.s.m.bin(c, Bin::And, t, quads);
                        r.push(self.s.m.bin(c, Bin::Mul, t, fill));
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), None)
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_absdiff_i32" => {
                // |a - b| of the wrapped difference; SCC is set below as for
                // the other results (nonzero).
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let z = self.s.m.i32(0);
                let t = self.s.m.bin(c, Bin::Sub, a, b);
                let neg = self.s.m.cmp(c, pred::SLT, t, z);
                let n = self.s.m.bin(c, Bin::Sub, z, t);
                (self.s.m.select(c, neg, n, t), None)
            }
            "s_sext_i32_i8" | "s_sext_i32_i16" => {
                let a = src32(self, 0)?;
                let k = self.s.m.i32(if name.ends_with("i8") { 24 } else { 16 });
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::AShr, t, k), None)
            }
            "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hh_b32_b16" => {
                // The low half of the result is half of src0 (low for ll and
                // lh, high for hh), the high half the one of src1 (low for ll,
                // high otherwise).
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, low, high) = (
                    self.s.m.i32(16),
                    self.s.m.i32(0xFFFF),
                    self.s.m.i32(0xFFFF_0000),
                );
                let lo_half = if name == "s_pack_hh_b32_b16" {
                    self.s.m.bin(c, Bin::LShr, a, k16)
                } else {
                    self.s.m.bin(c, Bin::And, a, low)
                };
                let hi_half = if name == "s_pack_ll_b32_b16" {
                    self.s.m.bin(c, Bin::Shl, b, k16)
                } else {
                    self.s.m.bin(c, Bin::And, b, high)
                };
                (self.s.m.bin(c, Bin::Or, lo_half, hi_half), None)
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, site, a);
        }
        if self.scalar.get(&site).is_some_and(|a| a.param == BVH) {
            let (Some(Operand::Scalar { code, dwords }), true) =
                (*data, name.starts_with("s_load_"))
            else {
                return Err("a write to an acceleration structure");
            };
            let zero = self.s.m.i32(0);
            for k in 0..dwords {
                self.write(inst, Reg::S(code + k), zero, None);
            }
            return Ok(());
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let xs = if self.by_slot_kind(src) {
            // A raw view's bytes start at its lead (structured views made
            // raw start before their first record).
            let (index, per_lane) = self.heap_slot(src, false)?;
            let (_, lead, kind) = self.slot_record(index);
            let kinds = [view_kind::CBV, view_kind::RAW_SRV, view_kind::RAW_UAV];
            self.by_kind(kind, &kinds, count as usize, |l, k| {
                let props = match k {
                    view_kind::CBV => (13, 0x1_0000),
                    view_kind::RAW_UAV => (11 | 0x1000, 0),
                    _ => (11, 0),
                };
                let h = l.s.heap_handle(l.cur, index, false, per_lane, props);
                if k == view_kind::CBV {
                    return Ok(l.cbuffer_dwords(h, addr, constant, count));
                }
                let at = l.s.m.bin(l.cur, Bin::Add, addr, lead);
                Ok(l.raw_dwords(h, at, count))
            })?
        } else if matches!(kind, Some(u) if u.kind == ResourceType::Srv) {
            let h = self.handle(site, src, false, (11, 0))?;
            // A structured view's raw view starts its lead before the first
            // record (a raw one's has none).
            if kind.is_some_and(|u| u.stride != 0) && !matches!(src, Source::Built { .. }) {
                let (_, lead) = self.view_record(src)?;
                addr = self.s.m.bin(c, Bin::Add, addr, lead);
            }
            self.raw_dwords(h, addr, count)
        } else {
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            self.cbuffer_dwords(h, addr, constant, count)
        };
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// `count` dwords from byte `addr` of the raw buffer `h`.
    fn raw_dwords(&mut self, h: V, addr: V, count: u32) -> Vec<V> {
        let c = self.cur;
        (0..count)
            .map(|k| {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                self.as_int(x)
            })
            .collect()
    }

    /// `count` dwords from byte `addr` of the constant buffer `h`; whole
    /// rows read once where the byte is `constant`.
    fn cbuffer_dwords(&mut self, h: V, addr: V, constant: Option<u32>, count: u32) -> Vec<V> {
        let c = self.cur;
        let mut rows: HashMap<u32, V> = HashMap::new();
        (0..count)
            .map(|k| match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            })
            .collect()
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind: a
        // vector memory or image access best, else a scalar buffer load;
        // without one, a use of the descriptor loaded from the same address
        // elsewhere (a waterfall loop that only reads a per-lane image's
        // size, beside the one that samples it).
        let uses_of = |load_of: &dyn Fn(Site) -> bool| -> Vec<(Site, u8)> {
            self.x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .filter_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if load_of(load) => Some((d.site, first)),
                        _ => None,
                    })
                })
                .collect()
        };
        let mut uses = uses_of(&|load| load == site);
        if uses.is_empty() {
            uses = uses_of(&|load| self.same_address(load, site));
            if !uses.is_empty() && !self.loop_scalars(site.0).is_empty() {
                if !self.last_pass_exact(site.0) {
                    return Err(LAST_PASS);
                }
                if !self.console_lanes {
                    return Err(HOST_LANES);
                }
                self.last_pass_loops.insert(site.0);
            }
        }
        let scalar = |at: Site| {
            matches!(
                self.x.p.blocks[at.0].insts[at.1].fields,
                Fields::Smem { .. }
            )
        };
        let Some(&(at, first)) = uses.iter().find(|u| !scalar(u.0)).or(uses.first()) else {
            const DATA: Why = "a descriptor read only as data";
            let bases = &self.x.res.bvh_bases;
            if !inst.writes.iter().any(|w| bases.contains(&w.1)) {
                return Err(DATA);
            }
            // An acceleration structure's descriptor: its base is 0.
            let zero = self.s.m.i32(0);
            for &(r, v) in &inst.writes {
                if bases.contains(&v) {
                    self.write(inst, r, zero, None);
                } else {
                    self.vals.insert(v, Val::Missing(DATA));
                }
            }
            return Ok(());
        };
        let src = Source::Loaded { load: site, first };
        let used = self.used_kind(&src);
        let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
        let un = ui.name();
        let uav = un.contains("store")
            || un.contains("atomic")
            || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        // Indexed raw accesses count records, of the used-resource entry's
        // stride or the view's own, past the host raw view's lead.
        let mut records = None;
        let (props, image) = match &ui.fields {
            Fields::Image { dim, .. } => {
                let kind = match dim {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    4 => 6,
                    5 => 7,
                    _ => return Err("an image dimension (cube or MSAA)"),
                };
                ((kind | u, 0x409), Some(kind))
            }
            Fields::Buffer { idxen, format, .. } => {
                let stride = used.map_or(0, |u| u.stride);
                let props = if un.starts_with("buffer_load_format")
                    || un.starts_with("buffer_store_format")
                {
                    (10 | u, 0x409)
                } else {
                    if *idxen && format.is_none() {
                        records = Some((stride >= 4 && stride.is_multiple_of(4)).then_some(stride));
                    }
                    (11 | u, 0)
                };
                if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                    return Err("a constant buffer's descriptor read as data");
                }
                (props, None)
            }
            // Scalar buffer loads read the raw view the host makes of a
            // buffer view: its descriptor counts the records of a structured
            // view (the used-resource entry's stride), the bytes of a raw one.
            Fields::Smem { .. } => {
                let stride = match used {
                    Some(u) if u.kind != ResourceType::Cbv => u.stride,
                    Some(_) => return Err("a constant buffer's descriptor read as data"),
                    None => return Err("a scalar memory read as data"),
                };
                if stride != 0 {
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a record count of a stride off whole dwords");
                    }
                    records = Some(Some(stride));
                }
                ((11 | u, 0), None)
            }
            _ => return Err("a scalar memory read as data"),
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => {
                let mut width = field(self, 0);
                if let Some(known) = records {
                    let (s, lead) = self.view_record(&src)?;
                    let s = match known {
                        Some(stride) => self.s.m.i32(stride),
                        None => {
                            // A slot without a buffer view has none (0).
                            let (zero, one) = (self.s.m.i32(0), self.s.m.i32(1));
                            let none = self.s.m.cmp(c, pred::EQ, s, zero);
                            self.s.m.select(c, none, one, s)
                        }
                    };
                    width = self.s.m.bin(c, Bin::Sub, width, lead);
                    width = self.s.m.bin(c, Bin::UDiv, width, s);
                }
                vec![(2, width)]
            }
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                    if let (Some(v), Some(_)) = (inst.write(Reg::S(code + k as u8)), image) {
                        let bits = Bits {
                            site,
                            start: 32 * rel,
                            len: 32,
                            at: 0,
                        };
                        self.dims.insert(v, Dim::Bits(bits));
                    }
                }
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// What `inst` makes of image descriptor dwords it reads as data (see
    /// [`Dim`]): none when it reads none, else each value it writes with its
    /// own [`Dim`] (none: a number like any other); or [`FIELDS`]. Steps let
    /// through: copies and spills, bit extractions at constant positions (a
    /// field but WIDTH, HEIGHT and BASE_LEVEL is a number), WIDTH or HEIGHT
    /// plus 1, and that shifted right by its descriptor's BASE_LEVEL or bits
    /// of it from there (the size of the view's first level).
    fn field_use(&mut self, inst: &Inst, guarded: bool) -> Result<Option<Dims>, Why> {
        if self.dims.is_empty() {
            return Ok(None);
        }
        let regs = |op: &Operand| match *op {
            Operand::Scalar { code, dwords } => (0..dwords).map(|k| Reg::S(code + k)).collect(),
            Operand::Vector { reg, dwords } => (0..dwords).map(|k| Reg::V(reg + k)).collect(),
            _ => Vec::new(),
        };
        let src: Vec<Reg> = match &inst.fields {
            Fields::Sop { src, .. } => src.iter().flat_map(regs).collect(),
            Fields::Valu(v) => v.src.iter().flat_map(regs).collect(),
            _ => Vec::new(),
        };
        // A VGPR write reads the register's old value for the lanes or bits
        // it keeps: a guarded write's, an SDWA field's, and those of writes
        // that merge into the register; no other's.
        let old: Vec<Reg> = match &inst.fields {
            Fields::Valu(v)
                if !guarded
                    && v.dpp.is_none()
                    && Self::sdwa_dst(inst).is_none_or(|(_, unused)| unused != 2)
                    && !["mac", "dot", "lane", "rel"]
                        .iter()
                        .any(|k| inst.name().contains(k)) =>
            {
                (inst.writes.iter())
                    .map(|w| w.0)
                    .filter(|r| matches!(r, Reg::V(_)) && !src.contains(r))
                    .collect()
            }
            _ => Vec::new(),
        };
        // What an access takes as its resource or sampler is no data.
        let held: Vec<Reg> = match &inst.fields {
            Fields::Image { rsrc, sampler, .. } => {
                let mut r = regs(rsrc);
                r.extend(sampler.iter().flat_map(regs));
                r
            }
            Fields::Buffer { rsrc, .. } => regs(rsrc),
            Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_load") => regs(base),
            _ => Vec::new(),
        };
        if !(inst.reads.iter())
            .any(|(r, v)| !held.contains(r) && !old.contains(r) && self.dims.contains_key(v))
        {
            return Ok(None);
        }
        // An operand that has failed fails the instruction for its reason
        // (one a retry may clear).
        let failed =
            (src.iter().filter_map(|&r| inst.read(r))).find_map(|v| match self.vals.get(&v) {
                Some(&Val::Missing(why)) => Some(why),
                _ => None,
            });
        let r = failed.map_or_else(|| self.field_step(inst), Err);
        if let Err(why) = r {
            for &(_, w) in &inst.writes {
                self.lane_fail.insert(w, why);
            }
        }
        r.map(Some)
    }

    /// [`Lower::field_use`] of an instruction that reads such dwords.
    fn field_step(&mut self, inst: &Inst) -> Result<Dims, Why> {
        let name = inst.name();
        let (src, dst, sdwa) = match &inst.fields {
            Fields::Sop { dst, src, .. } => (src, dst, None),
            Fields::Valu(v)
                if v.abs | v.neg | v.opsel == 0 && !v.clamp && v.omod == 0 && v.dpp.is_none() =>
            {
                (&v.src, &v.dst, v.sdwa)
            }
            _ => return Err(FIELDS),
        };
        let reg = |op: &Operand| match *op {
            Operand::Scalar { code, .. } => Some(Reg::S(code)),
            Operand::Vector { reg, .. } => Some(Reg::V(reg)),
            _ => None,
        };
        let out = |r: Option<Reg>| r.and_then(|r| inst.write(r)).ok_or(FIELDS);
        let dim_of = |l: &Self, v: Value| l.dims.get(&v).copied();
        if name == "s_mov_b64" {
            let (Some(Operand::Scalar { code: d, .. }), [Operand::Scalar { code: s, .. }]) =
                (dst, &src[..])
            else {
                return Err(FIELDS);
            };
            return (0..2)
                .map(|k| {
                    let from = inst.read(Reg::S(s + k)).and_then(|v| dim_of(self, v));
                    Ok((out(Some(Reg::S(d + k)))?, from))
                })
                .collect();
        }
        let dst = out(dst.as_ref().and_then(reg))?;
        // Each operand's dim, through its SDWA select.
        let mut dims = Vec::new();
        for (k, op) in src.iter().enumerate() {
            let d = reg(op)
                .and_then(|r| inst.read(r))
                .and_then(|v| dim_of(self, v));
            dims.push(match (sdwa, d) {
                (_, None) | (None, _) => d,
                (Some(s), Some(Dim::Bits(b))) if s.dst_sel == 6 && s.sext >> k & 1 == 0 => {
                    let sel = u32::from(s.src_sel.get(k).copied().unwrap_or(6));
                    Some(Dim::Bits(match sel {
                        0..=3 => b.shr(8 * sel).field(0, 8),
                        4 | 5 => b.shr(16 * (sel - 4)).field(0, 16),
                        _ => b,
                    }))
                }
                _ => return Err(FIELDS),
            });
        }
        let konst = |l: &mut Self, k: usize| {
            let x = l.operand(inst, &src[k]).ok()?;
            if let Some(c) = l.s.m.int_value(x) {
                return Some(c as u32);
            }
            // A join of the same constant on every edge into it (a mask each
            // way into a loop's body sets).
            let v = reg(&src[k]).and_then(|r| inst.read(r))?;
            let &(b, i) = l.phi_at.get(&v)?;
            let p = l.x.p;
            let mut c = None;
            for (n, &a) in p.blocks[b].phis[i].args.iter().enumerate() {
                if a == v || l.block[p.blocks[b].preds[n]].is_none() {
                    continue;
                }
                let Some(&Val::I32(x)) = l.vals.get(&a) else {
                    return None;
                };
                let y = l.s.m.int_value(x)?;
                if c.is_some_and(|c| c != y) {
                    return None;
                }
                c = Some(y);
            }
            c.map(|c| c as u32)
        };
        let same = |l: &Self, a: Site, b: Site| a == b || l.same_address(a, b);
        let next = match (name, &dims[..]) {
            ("s_mov_b32" | "v_mov_b32" | "v_readfirstlane_b32", &[d]) => d,
            // A spill: the lane's scalar comes back where `v_readlane_b32`
            // restores it.
            ("v_writelane_b32", _) => Some(Dim::Joined),
            ("v_readlane_b32", _) => match self.restored(inst) {
                Some(Moved::Value(v)) => dim_of(self, v),
                _ => return Err(FIELDS),
            },
            (
                "s_add_u32" | "s_add_i32" | "v_add_nc_u32" | "v_add_u32" | "v_add_co_u32",
                &[a, b],
            ) => {
                let (f, k) = match (a, b) {
                    (Some(Dim::Bits(f)), None) => (f, 1),
                    (None, Some(Dim::Bits(f))) => (f, 0),
                    _ => return Err(FIELDS),
                };
                if !(f.is(WIDTH) || f.is(HEIGHT)) || konst(self, k) != Some(1) {
                    return Err(FIELDS);
                }
                Some(Dim::Size(f.site))
            }
            ("s_lshr_b32", &[Some(Dim::Size(s)), Some(Dim::Bits(b))])
            | ("v_lshrrev_b32", &[Some(Dim::Bits(b)), Some(Dim::Size(s))])
            | ("v_bfe_u32", &[Some(Dim::Size(s)), Some(Dim::Bits(b)), None])
                if b.is(BASE_LEVEL) && same(self, s, b.site) =>
            {
                None
            }
            ("s_bfe_u32", &[Some(Dim::Size(s)), Some(Dim::Control(c))]) if same(self, s, c) => None,
            ("s_or_b32", &[Some(Dim::Bits(b)), None] | &[None, Some(Dim::Bits(b))])
                if b.is(BASE_LEVEL)
                    && konst(self, usize::from(dims[0].is_some()))
                        .is_some_and(|c| c & 0xFFFF == 0) =>
            {
                Some(Dim::Control(b.site))
            }
            _ => {
                // A bit extraction at constant positions.
                let bits = |k: usize| match dims.get(k) {
                    Some(&Some(Dim::Bits(b))) => Some(b),
                    _ => None,
                };
                let b = match (name, dims.len()) {
                    ("s_lshr_b32", 2) => bits(0).zip(konst(self, 1)).map(|(b, n)| b.shr(n & 31)),
                    ("v_lshrrev_b32", 2) => bits(1).zip(konst(self, 0)).map(|(b, n)| b.shr(n & 31)),
                    ("s_lshl_b32", 2) => bits(0).zip(konst(self, 1)).map(|(b, n)| b.shl(n & 31)),
                    ("v_lshlrev_b32", 2) => bits(1).zip(konst(self, 0)).map(|(b, n)| b.shl(n & 31)),
                    ("s_and_b32" | "v_and_b32" | "s_or_b32" | "v_or_b32", 2) => {
                        match (bits(0), bits(1)) {
                            (Some(a), Some(b)) if name.contains("or") => {
                                let site = if same(self, a.site, b.site) {
                                    a.site
                                } else {
                                    b.site
                                };
                                a.or(Bits { site, ..b })
                            }
                            (Some(b), None) | (None, Some(b)) => {
                                let m =
                                    konst(self, usize::from(bits(0).is_some())).ok_or(FIELDS)?;
                                let at = m.trailing_zeros().min(32);
                                let len = m.checked_shr(at).unwrap_or(0).trailing_ones();
                                match (name.contains("and"), m) {
                                    (false, 0) => Some(b),
                                    (true, _) if m.checked_shr(at + len).unwrap_or(0) == 0 => {
                                        Some(b.field(at, len))
                                    }
                                    _ => None,
                                }
                            }
                            _ => None,
                        }
                    }
                    ("s_bfe_u32", 2) => bits(0)
                        .zip(konst(self, 1))
                        .map(|(b, x)| b.shr(x & 31).field(0, x >> 16 & 0x7F)),
                    ("v_bfe_u32", 3) => match (bits(0), konst(self, 1), konst(self, 2)) {
                        (Some(b), Some(o), Some(w)) => Some(b.shr(o & 31).field(0, w & 31)),
                        _ => None,
                    },
                    ("v_alignbit_b32", 3) => match (bits(0), bits(1), konst(self, 2)) {
                        (Some(_), Some(lo), Some(0)) => Some(lo),
                        (Some(hi), Some(lo), Some(n)) => lo.shr(n & 31).or(hi.shl(32 - (n & 31))),
                        _ => None,
                    },
                    _ => None,
                };
                let b = b.ok_or(FIELDS)?;
                ([WIDTH, HEIGHT, BASE_LEVEL].into_iter())
                    .any(|f| b.touches(f))
                    .then_some(Dim::Bits(b))
            }
        };
        Ok(vec![(dst, next)])
    }

    /// The [`Dim`] of a join whose every other incoming value has one, the
    /// same but for the descriptor: that of each edge's, a site of its own
    /// where they differ. None where an incoming value is not yet lowered (a
    /// loop's), so that a read of the join is refused in `fill_phis`.
    fn phi_dim(&mut self, b: usize, value: Value, args: &[Value]) -> Option<Dim> {
        let preds = &self.x.p.blocks[b].preds;
        let mut kind = None;
        let mut sites = Vec::new();
        for (n, &arg) in args.iter().enumerate() {
            if arg == value || self.block[preds[n]].is_none() {
                continue;
            }
            let (site, k) = match *self.dims.get(&arg)? {
                Dim::Bits(x) => (x.site, Dim::Bits(Bits { site: (0, 0), ..x })),
                Dim::Size(s) => (s, Dim::Size((0, 0))),
                Dim::Control(s) => (s, Dim::Control((0, 0))),
                Dim::Joined => return None,
            };
            if kind.is_some_and(|d| d != k) {
                return None;
            }
            kind = Some(k);
            sites.push(site);
        }
        let kind = kind?;
        let site = if sites.iter().all(|&s| s == sites[0]) {
            sites[0]
        } else {
            let next = (usize::MAX, self.phi_sites.len());
            *self.phi_sites.entry((b, sites)).or_insert(next)
        };
        Some(match kind {
            Dim::Bits(x) => Dim::Bits(Bits { site, ..x }),
            Dim::Size(_) => Dim::Size(site),
            Dim::Control(_) => Dim::Control(site),
            Dim::Joined => Dim::Joined,
        })
    }

    /// After `inst` lowered: the dims [`Lower::field_use`] gave the values
    /// it writes. Any other value it writes (SCC, a carry) is data refused.
    fn field_outputs(&mut self, inst: &Inst, outs: &[(Value, Option<Dim>)]) {
        for &(_, w) in &inst.writes {
            match outs.iter().find(|o| o.0 == w) {
                Some(&(_, Some(d))) => {
                    self.dims.insert(w, d);
                }
                Some(_) => {}
                None => {
                    self.vals.insert(w, Val::Missing(FIELDS));
                    self.lane_bits.remove(&w);
                    self.lane_fail.insert(w, FIELDS);
                }
            }
        }
    }

    /// Whether the scalar loads at `a` and `b` read the same address: the
    /// same root parameter and offset, and the same computed term, or terms
    /// that `v_readfirstlane_b32` reads from the same vector value (the
    /// first lane's entry of a per-lane table index, as each waterfall loop
    /// over the index reads it).
    fn same_address(&self, a: Site, b: Site) -> bool {
        let (Some(x), Some(y)) = (self.scalar.get(&a), self.scalar.get(&b)) else {
            return false;
        };
        let first_lane = |v: Value| {
            let def = self
                .x
                .p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.writes.iter().any(|w| w.1 == v))?;
            let Fields::Valu(f) = &def.fields else {
                return None;
            };
            match f.src.first() {
                Some(&Operand::Vector { reg, .. }) if def.name() == "v_readfirstlane_b32" => {
                    def.read(Reg::V(reg))
                }
                _ => None,
            }
        };
        x.param == y.param
            && x.offset == y.offset
            && match (x.dynamic, y.dynamic) {
                (None, None) => true,
                (Some(d), Some(e)) => {
                    d == e || first_lane(d).is_some_and(|v| first_lane(e) == Some(v))
                }
                _ => false,
            }
    }

    /// The scalars (not lane masks) block `b` computes that code after it
    /// reads, when `b` loops on itself. Lanes run such a loop each until its
    /// own pass (see `structure`), so a scalar is the lane's own pass's; on
    /// the console it is the last pass's, the same in every lane.
    fn loop_scalars(&self, b: usize) -> Vec<Value> {
        let p = self.x.p;
        if !matches!(p.blocks[b].exit, Exit::Branch { taken, not_taken } if taken == b || not_taken == b)
        {
            return Vec::new();
        }
        let st = self.x.st;
        let scalar = |(r, v): &(Reg, Value)| {
            let data = matches!(*r, Reg::S(c) if c != EXEC && c != EXEC + 1) || *r == Reg::Scc;
            data && !st.masks.contains_key(v) && !st.highs.contains_key(v)
        };
        let block = &p.blocks[b];
        let written: HashSet<Value> = block
            .phis
            .iter()
            .map(|ph| (ph.reg, ph.value))
            .chain(block.insts.iter().flat_map(|i| i.writes.iter().copied()))
            .filter(scalar)
            .map(|w| w.1)
            .collect();
        let mut read: Vec<Value> = (p.blocks.iter().enumerate())
            .filter(|&(c, _)| c != b)
            .flat_map(|(_, other)| {
                let insts = other.insts.iter().flat_map(|i| i.reads.iter().map(|r| r.1));
                insts.chain(other.phis.iter().flat_map(|ph| ph.args.iter().copied()))
            })
            .filter(|v| written.contains(v))
            .collect();
        read.sort_unstable();
        read.dedup();
        read
    }

    /// Whether the scalars loop `l` leaves can be its last pass's in each
    /// lane after it (see [`Lower::last_pass_after`]), on the console's
    /// lanes: host waves can be the console's, lane for lane (compute, not
    /// on halves, and a 64-lane program only where it already runs on 64-lane
    /// waves), every lane runs the loop on its own (no pass serves several,
    /// as in a proven waterfall), and the block after it follows it alone.
    fn last_pass_exact(&self, l: usize) -> bool {
        let p = self.x.p;
        let Exit::Branch { taken, not_taken } = p.blocks[l].exit else {
            return false;
        };
        let after = if taken == l { not_taken } else { taken };
        (self.console_lanes || !self.x.native.wave64)
            && self.halves.is_none()
            && self.s.threads.is_some()
            && self.s.mesh.is_none()
            && after != l
            && p.blocks[after].preds == [l]
            && !self.x.res.waterfall.compares.iter().any(|c| c.0 == l)
    }

    /// At the start of block `b`, when it follows alone a loop of
    /// [`Lower::last_pass_loops`]: each scalar the loop computes that code
    /// after it reads becomes its last pass's. The console runs the loop's
    /// scalar code while any lane is left in it, so all its lanes leave it
    /// with the last pass's; lowered per lane, each leaves with its own. The
    /// lanes of the last pass ran the most iterations (the header's count).
    fn last_pass_after(&mut self, b: usize) -> Result<(), Why> {
        use super::wave::Op;
        let preds: Vec<usize> = (self.x.p.blocks[b].preds.iter())
            .copied()
            .filter(|&q| self.block[q].is_some())
            .collect();
        let [l] = preds[..] else {
            return Ok(());
        };
        if l == b || !self.last_pass_loops.contains(&l) {
            return Ok(());
        }
        self.cross_lane(false)?;
        let c = self.cur;
        let (count, _) = self.counters[&l];
        let most = self.wave_active(Op::MaxU, count);
        let last = self.s.m.cmp(c, pred::EQ, count, most);
        let (lane, none) = (self.lane_index(), self.s.m.i32(u32::MAX));
        let lane = self.s.m.select(c, last, lane, none);
        let lane = self.wave_active(Op::MinU, lane);
        for v in self.loop_scalars(l) {
            let val = match self.vals.get(&v).copied() {
                Some(Val::I32(x)) => Val::I32(self.read_lane(x, lane)),
                Some(Val::Bit(x)) => {
                    let x = self.s.m.cast(c, Cast::ZExt, x, self.s.t.i32);
                    let x = self.read_lane(x, lane);
                    let zero = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, x, zero))
                }
                Some(missing @ Val::Missing(_)) => missing,
                None => continue,
            };
            self.floats.remove(&v);
            self.last_pass.insert(v, (l, val));
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, site: Site, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        let x = self.x;
        if let Some((_, block, addresses)) = x.res.joined.iter().find(|j| j.0 == site) {
            let d = self.joined_offset(*block, addresses, a.offset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        } else if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Select { args, .. } | Source::Lanes { args, .. } = src {
            return self.used_kind(args.first()?);
        }
        if let Source::Phi { args, .. } = src {
            return self.used_kind(args.iter().flatten().next()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props.0, props.1);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let (index, per_lane) = self.heap_slot(src, sampler)?;
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if props.0 & 0x8000 != 0 {
                    // shortcut: one sampler used with and without comparison
                    // would need two static samplers; it becomes a comparison one.
                    self.s.cmp_samplers |= 1 << n;
                }
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    _ => Err("a root UAV or typed root view"),
                }
            }
            _ => {
                let _ = site;
                let (index, per_lane) = self.heap_slot(src, sampler)?;
                Ok(self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props))
            }
        }
    }

    /// The largest M0 the M0-relative move `inst` reads, when one shows
    /// (see [`Self::bound`]): guessed with no such move changing a register
    /// past its first, then proved for every move the proof takes to change
    /// none past the guess. By induction over time: the first move to read
    /// a larger M0 would read one bounded by values set before it, while
    /// every move kept to the guess.
    fn m0_bound(&self, inst: &Inst) -> Option<u32> {
        let at = |inst: &Inst, past: u32| {
            let m0 = inst.read(Reg::S(crate::ir::M0))?;
            let mut b = Bounding {
                past,
                used: Vec::new(),
                path: Vec::new(),
                budget: 256,
            };
            Some((self.bound(m0, &mut b)?, b.used))
        };
        let (guess, _) = at(inst, 0)?;
        let (m0, mut todo) = at(inst, guess)?;
        let mut done = Vec::new();
        if m0 > guess {
            return None;
        }
        while let Some(s) = todo.pop() {
            if done.contains(&s) {
                continue;
            }
            done.push(s);
            let (m0, used) = at(&self.x.p.blocks[s.0].insts[s.1], guess)?;
            if m0 > guess {
                return None;
            }
            todo.extend(used);
        }
        Some(guess)
    }

    /// An upper bound on the unsigned value `v` holds in each lane that
    /// passes where it is set, when one shows: a constant; a copy, the first
    /// lane's copy of a vector value, or a register an M0-relative move
    /// keeps (one past `b.past`); a median; a join of values each bounded,
    /// or bounded on its edge by a compare that answers the same in every
    /// lane taking the edge (a loop index each lane leaves the loop past).
    /// No step gives more than the bound of something it reads, so a value
    /// met again around a loop adds nothing.
    fn bound(&self, v: Value, b: &mut Bounding) -> Option<u32> {
        if b.path.contains(&v) {
            return Some(0);
        }
        b.budget = b.budget.checked_sub(1)?;
        b.path.push(v);
        let x = self.bound_of(v, b);
        b.path.pop();
        x
    }

    fn bound_of(&self, v: Value, b: &mut Bounding) -> Option<u32> {
        let (p, st) = (self.x.p, self.x.st);
        if let Some(&(h, i)) = self.phi_at.get(&v) {
            let mut max = 0;
            for (j, &q) in p.blocks[h].preds.iter().enumerate() {
                let mut known = vec![(st.reach[q], true)];
                if let LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } = st.exits[q]
                    && taken != not_taken
                {
                    known.push((when, taken == h));
                }
                if !st.bools.meet(&known) {
                    continue;
                }
                let arg = p.blocks[h].phis[i].args[j];
                let x = match self.compared(arg, &known, b) {
                    Some(x) => x,
                    None => self.bound(arg, b)?,
                };
                max = max.max(x);
            }
            return Some(max);
        }
        let (site, inst) = p
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(k, block)| {
                block
                    .insts
                    .iter()
                    .enumerate()
                    .map(move |(i, x)| ((k, i), x))
            })
            .find(|(_, x)| x.writes.iter().any(|w| w.1 == v))?;
        let reg = inst.writes.iter().find(|w| w.1 == v)?.0;
        let x = match &inst.fields {
            Fields::Sop { src, .. } if inst.name() == "s_mov_b32" => {
                self.operand_bound(inst, src.first()?, b)?
            }
            Fields::Valu(f) if f.sdwa.is_none() && f.dpp.is_none() && f.neg | f.abs == 0 => {
                match (inst.name(), reg, f.dst) {
                    (
                        "v_movreld_b32" | "v_movrelsd_b32",
                        Reg::V(r),
                        Some(Operand::Vector { reg: d, .. }),
                    ) => {
                        if u32::from(r.checked_sub(d)?) <= b.past {
                            return None;
                        }
                        b.used.push(site);
                        return self.bound(inst.read(reg)?, b);
                    }
                    ("v_mov_b32" | "v_readfirstlane_b32", ..) => {
                        self.operand_bound(inst, f.src.first()?, b)?
                    }
                    ("v_med3_u32", ..) => {
                        // At most the larger of any two operands.
                        let mut x: Vec<u32> = f
                            .src
                            .iter()
                            .filter_map(|o| self.operand_bound(inst, o, b))
                            .collect();
                        x.sort_unstable();
                        *x.get(1)?
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        // Lanes outside EXEC keep a VGPR's old value.
        match (reg, st.guards[site.0][site.1]) {
            (Reg::V(_), Some(_)) => Some(x.max(self.bound(inst.read(reg)?, b)?)),
            _ => Some(x),
        }
    }

    fn operand_bound(&self, inst: &Inst, op: &Operand, b: &mut Bounding) -> Option<u32> {
        match *op {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Literal(x) => Some(x),
            Operand::Scalar { code, .. } => self.bound(inst.read(Reg::S(code))?, b),
            Operand::Vector { reg, .. } => self.bound(inst.read(Reg::V(reg))?, b),
            _ => None,
        }
    }

    /// A bound on `a` in the lanes where `known` holds, from an unsigned
    /// compare of `a` with a bounded value that answers the same in each.
    fn compared(&self, a: Value, known: &[(Bool, bool)], b: &mut Bounding) -> Option<u32> {
        let st = self.x.st;
        for inst in self.x.p.blocks.iter().flat_map(|k| &k.insts) {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let name = inst.name();
            let Some(op) = name
                .strip_prefix("v_cmpx_")
                .or(name.strip_prefix("v_cmp_"))
                .and_then(|r| r.strip_suffix("_u32"))
            else {
                continue;
            };
            let (Some(x), Some(y), None, None) = (f.src.first(), f.src.get(1), f.sdwa, f.dpp)
            else {
                continue;
            };
            let mask = match f.dst {
                _ if name.starts_with("v_cmpx_") => inst.write(Reg::S(EXEC)),
                Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)),
                _ => None,
            };
            let Some(lane) = mask.and_then(|m| st.bools.find(Atom::Lane(m))) else {
                continue;
            };
            let reads = |o: &Operand| match *o {
                Operand::Scalar { code, .. } => inst.read(Reg::S(code)) == Some(a),
                Operand::Vector { reg, .. } => inst.read(Reg::V(reg)) == Some(a),
                _ => false,
            };
            // Whether every lane where `known` holds gets `answer`.
            let all = |answer: bool| {
                let mut k = known.to_vec();
                k.push((lane, !answer));
                !st.bools.meet(&k)
            };
            let answer = match (all(true), all(false)) {
                (true, _) => true,
                (_, true) => false,
                _ => continue,
            };
            // `a` at most, or below, the other operand.
            let (other, below) = match (op, reads(x), reads(y), answer) {
                ("gt", true, _, false) | ("le", true, _, true) => (y, false),
                ("ge", true, _, false) | ("lt", true, _, true) => (y, true),
                ("lt", _, true, false) | ("ge", _, true, true) => (x, false),
                ("le", _, true, false) | ("gt", _, true, true) => (x, true),
                _ => continue,
            };
            let Some(m) = self.operand_bound(inst, other, b) else {
                continue;
            };
            return Some(if below { m.saturating_sub(1) } else { m });
        }
        None
    }

    /// The heap index of the descriptor `src` (one read from a table or
    /// heap, or chosen among such), computed in the current block, and
    /// whether it may differ between lanes.
    fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<(V, bool), Why> {
        match src {
            Source::Loaded { load, first } => {
                let index = self.heap_index(*load, *first, sampler)?;
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                Ok((index, self.x.res.waterfall.loads.contains(load)))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { .. } => Err("a descriptor built around a root descriptor"),
            Source::Phi { block, args } => {
                // Each predecessor computes its heap index at its end; the
                // join picks one with a phi. A phi an edge brings is one of
                // an earlier join (code paths that merge, then merge again),
                // which dominates the edge. An edge no lane reading it takes
                // gives index 0.
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let Some(tail) = self.tail[q] else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(Source::Loaded { load, first }) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.heap_index(*load, *first, sampler);
                            self.cur = here;
                            index?
                        }
                        Some(arg @ Source::Phi { .. }) => self.heap_slot(arg, sampler)?.0,
                        Some(_) => return Err("a descriptor chosen by control flow"),
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok((phi, false))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                Ok((self.s.m.select(self.cur, cond, set, clear), false))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                Ok((self.lane_heap_index(src, sampler)?, true))
            }
        }
    }

    /// The record stride of the buffer descriptor `src` (one read from a
    /// table or the heap) and the bytes the host's view of it starts before
    /// the first record, read at run time from the view heap's records at
    /// its heap index (see the module notes of `emit`).
    fn view_record(&mut self, src: &Source) -> Result<(V, V), Why> {
        let (index, _) = self.heap_slot(src, false)?;
        let (stride, lead, _) = self.slot_record(index);
        Ok((stride, lead))
    }

    /// The record of heap slot `index` in the view heap's records: the
    /// record stride, the lead and the kind ([`view_kind`]) of its view.
    fn slot_record(&mut self, index: V) -> (V, V, V) {
        let c = self.cur;
        let h = self.s.view_strides_handle();
        let size = self.s.m.i32(VIEW_RECORD_BYTES);
        let at = self.s.m.bin(c, Bin::Mul, index, size);
        let (u, mask, align) = (
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8(7),
            self.s.m.i32(4),
        );
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, at, u, mask, align],
        );
        let [stride, lead, kind] = [0, 1, 2].map(|k| {
            let x = self.s.m.extract(c, r, k);
            self.as_int(x)
        });
        (stride, lead, kind)
    }

    /// Whether an access through `src` goes by the kind of view its heap
    /// slot holds ([`Lower::by_kind`]): a heap descriptor no used-resource
    /// entry names, whose kind only the slot's record tells (a descriptor
    /// index in a root constant or in memory, say). Reading a slot through
    /// a handle of another kind than its view's is undefined: a constant
    /// buffer read of a buffer view is a GPU fault on NVIDIA.
    fn by_slot_kind(&self, src: &Source) -> bool {
        self.used_kind(src).is_none() && !matches!(src, Source::Built { .. } | Source::Constant(_))
    }

    /// Runs `access` for each kind of view in `kinds`, each in a block only
    /// lanes whose slot has that `kind` (as its record names it) enter, so
    /// no lane reads a slot through a handle of another kind (a branch,
    /// uniform when the slot is). `access` returns `n` dwords, 0 in lanes
    /// whose slot has none of `kinds`; writes there are dropped.
    fn by_kind(
        &mut self,
        kind: V,
        kinds: &[u32],
        n: usize,
        mut access: impl FnMut(&mut Self, u32) -> Result<Vec<V>, Why>,
    ) -> Result<Vec<V>, Why> {
        let zero = self.s.m.i32(0);
        let mut out = vec![zero; n];
        for &k in kinds {
            let want = self.s.m.i32(k);
            let is = self.s.m.cmp(self.cur, pred::EQ, kind, want);
            let at = self.begin_guarded(Some(is));
            let got = access(self, k)?;
            let got = self.end_guarded(at, &got.iter().map(|&x| (x, zero)).collect::<Vec<_>>());
            for (o, g) in out.iter_mut().zip(got) {
                *o = self.s.m.bin(self.cur, Bin::Or, *o, g);
            }
        }
        Ok(out)
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    self.mask_bit(*mask, "a descriptor chosen per lane by an untracked mask")?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            return self.lane(inst, Some(true));
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        // Lockstep: a source lane outside EXEC is invalid unless the
        // instruction fetches inactive lanes; DPP8 then reads 0, DPP16 takes
        // it as outside its row.
        let (inside, zero) = match dpp {
            Dpp::Row { fi: false, .. } | Dpp::Lanes { fi: false, .. } if self.lockstep() => {
                match self.exec_words(inst)? {
                    Some((lo, hi)) => {
                        let on = self.lane_bit_of(c, src, lo, Some(hi));
                        let valid = match inside {
                            Some(i) => self.s.m.bin(c, Bin::And, i, on),
                            None => on,
                        };
                        (Some(valid), zero || matches!(dpp, Dpp::Lanes { .. }))
                    }
                    None => (inside, zero),
                }
            }
            _ => (inside, zero),
        };
        let mut r = if mesh_quad || self.helpers_read(quad) {
            self.quad_pick(x, src)
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes),
                // or to the largest M0 that shows: the registers past it
                // keep their values.
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let last = self.m0_bound(inst).map_or(u8::MAX, |m| m.min(255) as u8);
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg).min(last) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(new) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    if k > last {
                        let kept = match inst.read(Reg::V(dreg + k)) {
                            Some(o) => self.get(o).unwrap_or_else(Val::Missing),
                            None => Val::Missing("an M0-relative move"),
                        };
                        self.vals.insert(new, kept);
                        continue;
                    }
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => {
                        let m = self.legacy_mul(a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                };
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                let h = self.f16_nearest(r);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self.f16_nearest(a);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                const UNTRACKED: Why = "a select on an untracked mask";
                let Some(&Operand::Scalar { code, .. }) = v.src.get(2) else {
                    return Err(UNTRACKED);
                };
                let cond = self.read_mask_bit(inst, code, UNTRACKED)?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    const UNTRACKED: Why = "a carry-in from an untracked mask";
                    let Some(&Operand::Scalar { code, .. }) = v.src.get(2) else {
                        return Err(UNTRACKED);
                    };
                    Some(self.read_mask_bit(inst, code, UNTRACKED)?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry {
                    if self.lockstep() {
                        return self.write_lane_mask(inst, code, carry);
                    }
                    if let Some(w) = inst.write(Reg::S(code)) {
                        self.lane_bits.insert(w, carry);
                    }
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                // A constant lane that `v_writelane` spilled one scalar to on
                // every path (`resource::restored`) reads back that scalar:
                // lane writes and reads ignore EXEC, and each host lane keeps
                // its own wave's scalar. Outside compute the lane is the
                // console's, not the host's; in compute the VGPR need not
                // have a value (a constant buffer's descriptor spilled beside
                // numbers).
                if let Some(m) = self.restored(inst) {
                    let x = match m {
                        Moved::Value(v) => self.i32_of(v)?,
                        Moved::Const(k) => self.s.m.i32(k),
                    };
                    return int(self, x);
                }
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                if self.halves.is_some() {
                    // Halves: lane % 32 of each half, then the half lane bit 5
                    // names.
                    self.cross_lane(false)?;
                    let w = self.read_lane(a, lane);
                    let [(low, high)] = self.exchange([w], APART)?;
                    let c = self.cur;
                    let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
                    let bit = self.s.m.bin(c, Bin::And, lane, k32);
                    let up = self.s.m.cmp(c, pred::NE, bit, z);
                    let r = self.s.m.select(c, up, high, low);
                    return int(self, r);
                }
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                // Lanes outside EXEC take 0, slot 0's: in lockstep they run
                // the loop's accesses too, and their register holds anything.
                let a = s(self, 0)?;
                let a = match guard {
                    Some(g) => {
                        let zero = self.s.m.i32(0);
                        self.s.m.select(c, g, a, zero)
                    }
                    None => a,
                };
                int(self, a)
            }
            "v_readfirstlane_b32"
                if guard.is_none()
                    && self.halves.as_ref().is_some_and(|h| h.alone[self.cur_ir]) =>
            {
                // Halves: the one lane of its console wave that gets here.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" if self.halves.is_some() => {
                let a = s(self, 0)?;
                self.cross_lane(false)?;
                let r = self.first_lane_of_halves(a, guard)?;
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.join64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// The i64 of a low and a high dword.
    fn join64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, i64t);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match self.src0 {
            Some(x) if k == 0 => x,
            _ => self.f16_operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad; a sign extension above the high word adds nothing).
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h = l.f16_nearest(f);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.f16_nearest(f)
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_sub_nc_u32"
            | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                let lane = self.lane_index();
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = self.s.m.bin(c, Bin::And, a, m);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let f16 = ty == "f16";
                let x = if f16 {
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let mut h = self.half_src(inst, 0)?;
                    if v.abs & 1 != 0 {
                        let m = self.s.m.i32(0x7FFF);
                        h = self.s.m.bin(c, Bin::And, h, m);
                    }
                    if v.neg & 1 != 0 {
                        let s = self.s.m.i32(0x8000);
                        h = self.s.m.bin(c, Bin::Xor, h, s);
                    }
                    h
                } else {
                    let x = self.fsrc(inst, 0)?;
                    self.as_int(x)
                };
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
                    (10, 15, 0x1F, 0x03FF, 0x0200)
                } else {
                    (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
                };
                let (k_exp, k_sign, k_emax, mant, quiet) = (
                    k(self, exp_sh),
                    k(self, sign_sh),
                    k(self, exp_mask),
                    k(self, mant_mask),
                    k(self, quiet_bit),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
                let e = self.s.m.bin(c, Bin::And, e, k_emax);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
                let zero = k(self, 0);
                let (e0, emax) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, k_emax),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, emax, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, emax);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                if op == "lg" {
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    let lt = self.s.m.cmp(c, pred::OLT, a, b);
                    let gt = self.s.m.cmp(c, pred::OGT, a, b);
                    self.s.m.bin(c, Bin::Or, lt, gt)
                } else {
                    let p = match op {
                        "f" => pred::FALSE,
                        "lt" => pred::OLT,
                        "eq" => pred::OEQ,
                        "le" => pred::OLE,
                        "gt" => pred::OGT,
                        "ge" => pred::OGE,
                        "o" => pred::ORD,
                        "u" => pred::UNO,
                        "nge" => pred::ULT,
                        "nlg" => pred::UEQ,
                        "ngt" => pred::ULE,
                        "nle" => pred::UGT,
                        "neq" => pred::UNE,
                        "nlt" => pred::UGE,
                        "tru" => pred::TRUE,
                        _ => return Err(inst.name()),
                    };
                    self.s.m.cmp(c, p, a, b)
                }
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => return self.lane(inst, Some(op == "t")),
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r)
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) -> Result<(), Why> {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b)
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Ok(());
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if self.lockstep()
            && let Some(code) = code
        {
            return self.write_lane_mask(inst, code, r);
        }
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
        Ok(())
    }

    /// Lockstep: SGPRs `code` and `code + 1` of `inst` take the lane mask
    /// whose bit is `bit` in the lanes EXEC holds and 0 in the others (a
    /// compare's or a carry's), the console wave's ballot.
    fn write_lane_mask(&mut self, inst: &Inst, code: u8, bit: V) -> Result<(), Why> {
        let bit = match self.exec_bit(inst)? {
            Some(e) => self.s.m.bin(self.cur, Bin::And, bit, e),
            None => bit,
        };
        let (lo, hi) = self.ballot_halves(bit, APART_MASK)?;
        self.set(inst.write(Reg::S(code)), Val::I32(lo));
        self.set(inst.write(Reg::S(code + 1)), Val::I32(hi));
        Ok(())
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                (v.dst.ok_or("an interpolation")?, v.src.get(1).copied())
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        // What the result adds to the evaluated channel (a second half's
        // first half, when the halves evaluate on their own).
        let mut add = None;
        let (attribute, eval) = match *interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => {
                let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                let eval = if self.per_vertex(attribute.row) {
                    // The element is read per vertex elsewhere, so it is
                    // constant: weigh its vertices with SV_Barycentrics.
                    let mut ij = [None; 2];
                    for (index, w) in ij.iter_mut().enumerate() {
                        let x = match self.barycentric(barycentric, index as u8) {
                            Val::I32(x) => x,
                            Val::Missing(w) => return Err(w),
                            Val::Bit(_) => unreachable!(),
                        };
                        *w = Some(self.as_float(x));
                    }
                    Eval::Weights {
                        p0: true,
                        i: ij[0],
                        j: ij[1],
                    }
                } else {
                    Eval::Mode(mode)
                };
                (attribute, eval)
            }
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, Eval::Mode(1)),
            // P10 and P20: what the first and second halves weigh, P1 - P0
            // and P2 - P0, which is what the hardware holds for an
            // attribute the program also interpolates.
            Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                let one = Some(self.s.m.f32(1f32.to_bits()));
                let (i, j) = if vertex == 0 {
                    (one, None)
                } else {
                    (None, one)
                };
                (attribute, Eval::Weights { p0: false, i, j })
            }
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            // A half of an interpolation with computed i or j: P0 + i P10,
            // or the old dst (f16: the third operand) + j P20.
            Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                let i = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: true,
                        i,
                        j: None,
                    },
                )
            }
            Interp::Weighted { attribute } => {
                let old = match &inst.fields {
                    Fields::Valu(v) => v.src.get(2).copied(),
                    _ => Some(dst),
                };
                add = Some(float(self, old)?);
                let j = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: false,
                        i: None,
                        j,
                    },
                )
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| i.pack2),
            _ => None,
        };
        let f16 = name.ends_with("_f16");
        // The evaluated channel plus `add`, as a float.
        let total = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let x = l.eval_channel(row, ch, eval)?;
            let x = l.as_float(x);
            Ok(match add {
                Some(a) => l.s.m.bin(l.cur, Bin::Add, a, x),
                None => x,
            })
        };
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let x = total(self, attribute.row, attribute.channel)?;
                self.as_int(x)
            }
            Some(pack2) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_halves`); the host's are 32-bit floats.
                let packed = if pack2 {
                    let p = self.half_inputs.get(&attribute.row);
                    Some(*p.ok_or("16-bit input rows paired unlike the stage data says")?)
                } else {
                    None
                };
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = match packed {
                        Some(p) => p[usize::from(h)],
                        None => h.is_multiple_of(2).then_some((attribute.row, h / 2)),
                    };
                    let Some((row, ch)) = src else {
                        return Ok(None);
                    };
                    total(l, row, ch).map(Some)
                };
                let to_f16 = |l: &mut Self, f: V| l.f16_nearest(f);
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values.
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    self.as_int(f)
                } else if f16 {
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let lo = to_f16(self, f);
                    self.low_half(inst, reg, lo)
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20): then its
    /// element is declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated { .. } => false,
        })
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { p0, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = p0.then_some(v0);
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name.starts_with("image_bvh") {
            // shortcut: the host builds no acceleration structure, so each
            // is taken to be empty (see `BVH`) and the ray misses the node,
            // as the hardware reports it by the node's type (the pointer's
            // low 3 bits): a triangle's (0-3) distance +inf over 1 and no
            // barycentrics, past any ray's end, which every compare of the
            // distance with the ray's range takes as a miss (a NaN would
            // not be); no child of a box node (0xFFFFFFFF in each slot).
            let (Operand::Vector { reg, .. }, Some(&Operand::Vector { reg: node, .. })) =
                (*data, addr.first())
            else {
                return Err("an intersection");
            };
            let node = inst.read(Reg::V(node)).ok_or("a node pointer")?;
            let node = self.i32_of(node)?;
            let guard = self.guard(b, i)?;
            let c = self.cur;
            let (seven, four, none) = (self.s.m.i32(7), self.s.m.i32(4), self.s.m.i32(u32::MAX));
            let kind = self.s.m.bin(c, Bin::And, node, seven);
            let triangle = self.s.m.cmp(c, pred::IULT, kind, four);
            for (k, miss) in (0..).zip([0x7F80_0000, 0x3F80_0000, 0, 0]) {
                let miss = self.s.m.i32(miss);
                let x = self.s.m.select(c, triangle, miss, none);
                self.write(inst, Reg::V(reg + k), x, guard);
            }
            return Ok(());
        }
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        if *a16 && name.starts_with("image_sample_b") {
            return Err("a biased image sample with 16-bit coordinates");
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // With a16, two 16-bit components per VGPR, low half first.
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs
                .get(if *a16 { k / 2 } else { k })
                .ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            let x = l.i32_of(v)?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (cmpswap: twice the value's), two per
            // value for a 64-bit atomic: SM 6.6's on a uint64 texture UAV.
            // shortcut: D3D12 requires an R32G32_UINT view for those; the
            // title's view is taken to be one.
            let cmpswap = code == u32::MAX;
            let wide = match (*dmask, cmpswap) {
                (1, false) | (3, true) => false,
                (3, false) | (0xF, true) => true,
                _ => return Err("an image atomic of other channels"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let per = if wide { 2 } else { 1 };
            let mut vals = self.vgprs(inst, data, if cmpswap { 2 * per } else { per })?;
            let mut olds = Vec::new();
            for k in 0..per {
                olds.push(match inst.read(Reg::V(reg + k)) {
                    Some(v) if cache.glc => self.i32_of(v)?,
                    _ => self.s.m.i32(0),
                });
            }
            let mut old = olds[0];
            let ty = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC64_TYPED | flags::ATOMIC64_HEAP;
                vals = vals.chunks(2).map(|p| self.join64(p[0], p[1])).collect();
                old = self.join64(olds[0], olds[1]);
                self.s.m.int(64)
            } else {
                self.s.t.i32
            };
            let mut value = vals[0];
            if op == "sub" {
                let z = self.s.m.int_const(ty, 0);
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let comp = if wide { 0x107 } else { 0x105 };
            let h = self.handle(site, src, false, (kind | 0x1000, comp))?;
            let at = self.begin_guarded(guard);
            let bits = if wide { 64 } else { 32 };
            let mut args = vec![h];
            let r = if cmpswap {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                args.extend([vals[1], vals[0]]);
                let name = format!("atomicCompareExchange.i{bits}");
                self.s.op(self.cur, &name, Attr::NoUnwind, ty, 79, &args)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                let name = format!("atomicBinOp.i{bits}");
                self.s.op(self.cur, &name, Attr::NoUnwind, ty, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if cache.glc {
                let r = if wide {
                    let (lo, hi) = self.split64(r);
                    vec![lo, hi]
                } else {
                    vec![r]
                };
                for (k, x) in (0..).zip(r) {
                    self.write(inst, Reg::V(reg + k), x, None);
                }
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format).
            if *dmask != 0xF {
                return Err("an image store of some channels");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for k in 0..4u8 {
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let (op, sample) = match name {
            "image_sample" | "image_sample_o" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" | "image_gather4_l"
            | "image_gather4_l_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            let sh = self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let offset_word = name.ends_with("_o");
            let first =
                usize::from(offset_word) + usize::from(name == "image_sample_b") + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates, a cube's
            // two per direction, along its face's s and t.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            let per = if kind == 5 { 2 } else { spatial };
            let grads = if op == 63 { 2 * per } else { 0 };
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates and LOD are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                let x = addr_val(self, first + grads + k)?;
                *slot = to_float(self, x);
            }
            let face = (kind == 5).then_some(cs[2]);
            if let Some(face) = face {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                let dir = self.cube_direction(cs[0], cs[1], face);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, cs[0], cs[1], cs[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            match op {
                60 => {
                    if offset_word {
                        // DXIL takes immediate offsets in -8..=7; the
                        // console's are 6-bit and may be computed.
                        let w = addr_val(self, 0)?;
                        let w = self
                            .s
                            .m
                            .int_value(w)
                            .ok_or("a sample offset that is not constant")?
                            as u32;
                        for (k, o) in os.iter_mut().enumerate().take(offsets) {
                            let x = (w >> (8 * k) & 0x3F) as i32;
                            let x = if x >= 32 { x - 64 } else { x };
                            if !(-8..=7).contains(&x) {
                                return Err("a sample offset past -8..7");
                            }
                            *o = self.s.m.i32(x as u32);
                        }
                    }
                    args.extend(&os);
                    args.push(fu);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = addr_val(self, 0)?;
                    let bias = self.as_float(bias);
                    args.extend(&os);
                    args.extend([bias, fu]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 => {
                    let mut g = vec![fu; 6];
                    for k in 0..per {
                        let x = addr_val(self, first + k)?;
                        g[k] = to_float(self, x);
                        let y = addr_val(self, first + per + k)?;
                        g[3 + k] = to_float(self, y);
                    }
                    if let Some(face) = face {
                        // A face coordinate is sc / (2 |ma|) + 1.5 and the
                        // direction's |ma| is 1: the direction moves twice the
                        // face coordinates' gradient along the face's axes and
                        // not along its major axis.
                        let (two, zero) = (self.s.m.f32(2.0f32.to_bits()), self.s.m.f32(0));
                        for d in [0, 3] {
                            let u = self.s.m.bin(c, Bin::Mul, g[d], two);
                            let v = self.s.m.bin(c, Bin::Mul, g[d + 1], two);
                            let r = self.on_cube_face(face, [u, v], [zero, zero]);
                            g[d..d + 3].copy_from_slice(&r);
                        }
                    }
                    args.extend(&os);
                    args.extend(g);
                    // The least level of detail (`_cl`) follows the coordinates.
                    let clamp = if name.ends_with("_cl") {
                        let x = addr_val(self, first + grads + coords)?;
                        to_float(self, x)
                    } else {
                        fu
                    };
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_val(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(fu);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if name == "image_sample_l" {
                        let x = addr_val(self, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if offset_word {
                        let w = addr_val(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    if name.starts_with("image_gather4_l") && !name.starts_with("image_gather4_lz")
                    {
                        if !matches!(kind, 2 | 7) || *d16 {
                            return Err("a gather at a level of a cube, 1D, 3D or d16 view");
                        }
                        let lod = addr_val(self, first + coords)?;
                        let lod = to_float(self, lod);
                        let Operand::Vector { reg, .. } = *data else {
                            return Err("an image result");
                        };
                        let texels = self.gather_at_level(
                            (h, sh),
                            &cs,
                            [os[0], os[1]],
                            lod,
                            dmask.trailing_zeros(),
                        );
                        for (k, x) in texels.into_iter().enumerate() {
                            let x = self.as_int(x);
                            self.write(inst, Reg::V(reg + k as u8), x, guard);
                        }
                        return Ok(());
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    self.s.op(
                        c,
                        "textureGather.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        73,
                        &args,
                    )
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                self.f16_nearest(x)
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// A gather at an explicit level (`image_gather4_l`), which DXIL lacks:
    /// the four texels of the 2×2 footprint at the nearest level, each
    /// sampled at its centre through the same sampler (so its address mode
    /// holds), in gather order: (-,+), (+,+), (+,-), (-,-).
    fn gather_at_level(
        &mut self,
        (h, sh): (V, V),
        cs: &[V],
        offsets: [V; 2],
        lod: V,
        channel: u32,
    ) -> [V; 4] {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let z = self.s.m.i32(0);
        let top = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
        let levels = self.s.m.extract(c, top, 3);
        let one = self.s.m.i32(1);
        let last = self.s.m.bin(c, Bin::Sub, levels, one);
        let nearest = self.unary(26, lod);
        let fz = self.s.m.f32(0);
        let nearest = self.binary(35, nearest, fz);
        let level = self.s.m.cast(c, Cast::FpToSi, nearest, i32t);
        let level = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 37, &[level, z]);
        let level = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 38, &[level, last]);
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, level]);
        let (half, three_halves) = (
            self.s.m.f32(0.5f32.to_bits()),
            self.s.m.f32(1.5f32.to_bits()),
        );
        let mut centres = [[fz; 2]; 2];
        for k in 0..2 {
            let n = self.s.m.extract(c, d, k as u32);
            let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
            let x = self.s.m.bin(c, Bin::Mul, cs[k], n);
            let x = self.s.m.bin(c, Bin::Sub, x, half);
            let x = self.unary(27, x);
            let o = self.s.m.cast(c, Cast::SiToFp, offsets[k], f32t);
            let x = self.s.m.bin(c, Bin::Add, x, o);
            for (j, step) in [half, three_halves].into_iter().enumerate() {
                let t = self.s.m.bin(c, Bin::Add, x, step);
                centres[k][j] = self.s.m.bin(c, Bin::SDiv, t, n);
            }
        }
        let level = self.s.m.cast(c, Cast::SiToFp, level, f32t);
        let (fu, iu) = (self.s.m.undef(f32t), self.s.m.undef(i32t));
        [(0, 1), (1, 1), (1, 0), (0, 0)].map(|(i, j)| {
            let args = [
                h,
                sh,
                centres[0][i],
                centres[1][j],
                cs[2],
                fu,
                iu,
                iu,
                iu,
                level,
            ];
            let r = self.s.op(
                c,
                "sampleLevel.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                62,
                &args,
            );
            self.s.m.extract(c, r, channel)
        })
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        self.on_cube_face(face, [u, v], [one, mone])
    }

    /// The vector with `u` and `v` along the s and t axes of cube face
    /// `face` (a float id, as `v_cubeid` gives) and `plus` along the major
    /// axis of a positive face, `minus` along that of a negative one.
    fn on_cube_face(&mut self, face: V, [u, v]: [V; 2], [plus, minus]: [V; 2]) -> [V; 3] {
        let c = self.cur;
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [plus, minus, u, u, u, nu],
            [nv, nv, plus, minus, nv, nv],
            [nu, u, v, nv, plus, minus],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // A heap descriptor no used-resource entry names goes by the kind
        // of view its slot holds: its index, whether lanes differ there, and
        // its record (stride, lead, kind).
        let slot = if self.by_slot_kind(src) {
            let (index, per_lane) = self.heap_slot(src, false)?;
            let (stride, lead, kind) = self.slot_record(index);
            Some((index, per_lane, stride, lead, kind))
        } else {
            None
        };
        // A constant buffer read per lane (a dynamically indexed array).
        // shortcut: the console's CBV descriptor is taken to have a 16-byte
        // stride, as the vector loads of whole rows suggest.
        let cbv_byte = match index {
            Some(i) => {
                let sixteen = self.s.m.i32(16);
                let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                self.s.m.bin(c, Bin::Add, rows, off)
            }
            None => off,
        };
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self
                .cbuffer_dwords(h, cbv_byte, None, count)
                .into_iter()
                .enumerate()
            {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        // Dword accesses go through the raw view the host makes of every
        // untyped buffer view; indexed ones address the record themselves,
        // index * stride + offset, with the used-resource entry's stride or,
        // without one (a descriptor the program reads from the heap itself),
        // the view's own read at run time. A heap view's records start past
        // the bytes its host view starts before them, read likewise; so do
        // the bytes of a slot read by its kind.
        let off = match index {
            Some(i) if !typed && format.is_none() => {
                let stride = used.map_or(0, |u| u.stride);
                let known = (stride >= 4 && stride.is_multiple_of(4)).then(|| self.s.m.i32(stride));
                let (s, lead) = match (src, known, slot) {
                    (Source::Built { .. }, Some(s), _) => (s, None),
                    (Source::Built { .. }, None, _) => {
                        return Err("a structured buffer of unknown stride");
                    }
                    (_, _, Some((_, _, s, lead, _))) => (known.unwrap_or(s), Some(lead)),
                    _ => {
                        let (s, lead) = self.view_record(src)?;
                        (known.unwrap_or(s), Some(lead))
                    }
                };
                let record = self.s.m.bin(c, Bin::Mul, i, s);
                let at = self.s.m.bin(c, Bin::Add, record, off);
                match lead {
                    Some(lead) => self.s.m.bin(c, Bin::Add, at, lead),
                    None => at,
                }
            }
            _ => match slot {
                Some((.., lead, _)) if !typed => self.s.m.bin(c, Bin::Add, off, lead),
                _ => off,
            },
        };
        // A raw (or with `UAV`, typed) slot's handle of the kind `k`.
        let props = |k: u32| {
            let u = if matches!(k, view_kind::RAW_UAV | view_kind::TYPED_UAV) {
                0x1000
            } else {
                0
            };
            if matches!(k, view_kind::TYPED_SRV | view_kind::TYPED_UAV) {
                (10 | u, 0x409)
            } else {
                (11 | u, 0)
            }
        };
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let known = match slot {
                None => Some(self.handle(site, src, false, (11 | u, 0))?),
                Some(_) => None,
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let r = match (known, slot) {
                (Some(h), _) => self.buffer_atomic(h, op, off, &vals),
                (None, Some((index, per_lane, .., kind))) => {
                    let kinds = [view_kind::RAW_UAV];
                    self.by_kind(kind, &kinds, 1, |l, k| {
                        let h = l.s.heap_handle(l.cur, index, false, per_lane, props(k));
                        Ok(vec![l.buffer_atomic(h, op, off, &vals)])
                    })?[0]
                }
                (None, None) => unreachable!(),
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            // shortcut: typed stores are taken to write all four channels
            // of a float format.
            if typed && (count != 4 || *offen || *offset != 0) {
                return Err("a typed buffer store of some channels or at an offset");
            }
            let at = index.unwrap_or(self.s.m.i32(0));
            let known = match slot {
                None if typed => Some(self.handle(site, src, false, (10 | u, 0x409))?),
                None => Some(self.handle(site, src, false, (11 | u, 0))?),
                Some(_) => None,
            };
            let guarded = self.begin_guarded(guard);
            match (known, slot) {
                (Some(h), _) => self.buffer_store(h, typed, at, off, &vals),
                (None, Some((index, per_lane, .., kind))) => {
                    let kinds = [if typed {
                        view_kind::TYPED_UAV
                    } else {
                        view_kind::RAW_UAV
                    }];
                    self.by_kind(kind, &kinds, 0, |l, k| {
                        let h = l.s.heap_handle(l.cur, index, false, per_lane, props(k));
                        l.buffer_store(h, typed, at, off, &vals);
                        Ok(Vec::new())
                    })?;
                }
                (None, None) => unreachable!(),
            }
            self.end_guarded(guarded, &[]);
            return Ok(());
        }
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let xs = match slot {
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.format_load(h, off, f, count)?
                }
                Some((index, per_lane, .., kind)) => {
                    let kinds = [view_kind::RAW_SRV, view_kind::RAW_UAV];
                    self.by_kind(kind, &kinds, count as usize, |l, k| {
                        let h = l.s.heap_handle(l.cur, index, false, per_lane, props(k));
                        l.format_load(h, off, f, count)
                    })?
                }
            };
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let xs = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            let index = index.unwrap_or(self.s.m.i32(0));
            match slot {
                None => {
                    if uav {
                        self.s.flags |= flags::TYPED_UAV_LOADS;
                    }
                    let h = self.handle(site, src, false, (10 | u, 0x409))?;
                    self.typed_load(h, index, count)
                }
                Some((slot, per_lane, .., kind)) => {
                    let kinds = [view_kind::TYPED_SRV, view_kind::TYPED_UAV];
                    self.by_kind(kind, &kinds, count as usize, |l, k| {
                        if k == view_kind::TYPED_UAV {
                            l.s.flags |= flags::TYPED_UAV_LOADS;
                        }
                        let h = l.s.heap_handle(l.cur, slot, false, per_lane, props(k));
                        Ok(l.typed_load(h, index, count))
                    })?
                }
            }
        } else {
            match slot {
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.raw_load(h, off, count)
                }
                Some((index, per_lane, .., kind)) => {
                    let kinds = [view_kind::CBV, view_kind::RAW_SRV, view_kind::RAW_UAV];
                    self.by_kind(kind, &kinds, count as usize, |l, k| {
                        if k == view_kind::CBV {
                            let h =
                                l.s.heap_handle(l.cur, index, false, per_lane, (13, 0x1_0000));
                            return Ok(l.cbuffer_dwords(h, cbv_byte, None, count));
                        }
                        let h = l.s.heap_handle(l.cur, index, false, per_lane, props(k));
                        Ok(l.raw_load(h, off, count))
                    })?
                }
            }
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.as_float(xs[ch as usize]);
                    let h = self.f16_nearest(f);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(self.cur, Bin::Shl, hi, k16);
                            self.s.m.bin(self.cur, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The atomic operation `op` (DXIL's numbering; `u32::MAX` for
    /// compare-exchange, `vals` new and compare) at byte `off` of the raw
    /// UAV `h`: the old value.
    fn buffer_atomic(&mut self, h: V, op: u32, off: V, vals: &[V]) -> V {
        let iu = self.s.m.undef(self.s.t.i32);
        let coords = [off, iu, iu];
        let i32t = self.s.t.i32;
        if op == u32::MAX {
            // cmpswap: data is (new, compare).
            let mut args = vec![h];
            args.extend(coords);
            args.extend([vals[1], vals[0]]);
            return self.s.op(
                self.cur,
                "atomicCompareExchange.i32",
                Attr::NoUnwind,
                i32t,
                79,
                &args,
            );
        }
        let mut args = vec![h, self.s.m.i32(op)];
        args.extend(coords);
        args.push(vals[0]);
        self.s
            .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
    }

    /// Stores `vals` through `h`: a typed UAV's element `index` (four
    /// float channels), or byte `off` of a raw UAV. Raw stores go out one
    /// dword at a time, in order: on the 4090, stores of several dwords to
    /// the system-memory buffers that hold title UAVs run more than a hundred
    /// times slower than single-dword stores.
    fn buffer_store(&mut self, h: V, typed: bool, index: V, off: V, vals: &[V]) {
        let iu = self.s.m.undef(self.s.t.i32);
        let void = self.s.t.void;
        if typed {
            let mut args = vec![h, index, iu];
            for &x in vals {
                args.push(self.as_float(x));
            }
            args.push(self.s.m.i8(0xF));
            self.s
                .op(self.cur, "bufferStore.f32", Attr::NoUnwind, void, 69, &args);
            return;
        }
        let (one, align) = (self.s.m.i8(1), self.s.m.i32(4));
        for (k, &x) in vals.iter().enumerate() {
            let at = match k {
                0 => off,
                _ => {
                    let step = self.s.m.i32(4 * k as u32);
                    self.s.m.bin(self.cur, Bin::Add, off, step)
                }
            };
            self.s.op(
                self.cur,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                void,
                140,
                &[h, at, iu, x, iu, iu, iu, one, align],
            );
        }
    }

    /// `count` channels of element `index` of the typed buffer `h`.
    fn typed_load(&mut self, h: V, index: V, count: u32) -> Vec<V> {
        let iu = self.s.m.undef(self.s.t.i32);
        let r = self.s.op(
            self.cur,
            "bufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            68,
            &[h, index, iu],
        );
        (0..count)
            .map(|k| {
                let x = self.s.m.extract(self.cur, r, k);
                self.as_int(x)
            })
            .collect()
    }

    /// `count` dwords from byte `off` of the raw buffer `h`, in one load.
    fn raw_load(&mut self, h: V, off: V, count: u32) -> Vec<V> {
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8((1u8 << count) - 1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            self.cur,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, off, iu, mask, align],
        );
        (0..count)
            .map(|k| {
                let x = self.s.m.extract(self.cur, r, k);
                self.as_int(x)
            })
            .collect()
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, packed)) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = match packed {
                        Some(p) => p[usize::from(h)],
                        None => (!high).then_some((row, ch)),
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let vertices = self.s.mesh.ok_or("a primitive export outside a mesh shader")?.max_vertices;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let vertices = self.s.m.i32(vertices);
        // An index past the declared vertices is undefined on the host (a
        // driver reset on NVIDIA); such a primitive is culled.
        let mut culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut indices = Vec::with_capacity(3);
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            let outside = self.s.m.cmp(c, pred::IUGE, i, vertices);
            culled = self.s.m.bin(c, Bin::Or, culled, outside);
            indices.push(i);
        }
        let mut args = vec![self.mesh_thread()];
        for i in indices {
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    _ if self.lockstep() => self
                        .wave_branch(b, taken, not_taken)
                        .map_err(|w| unsupported(offset, w))?,
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT);
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE);
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] {
                            self.cur
                        } else {
                            via
                        };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                self.next_turn().map_or(Term::Ret, Term::Br)
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// Lockstep: the console wave's branch ending block `b`, on values every
    /// thread has alike: the condition, the block it goes to when that
    /// holds and the other.
    fn wave_branch(
        &mut self,
        b: usize,
        taken: usize,
        not_taken: usize,
    ) -> Result<(V, usize, usize), Why> {
        let inst: &'a Inst = self.x.p.blocks[b]
            .insts
            .last()
            .ok_or("a branch without its instruction")?;
        let c = self.cur;
        // A register pair (EXEC, VCC) is not zero.
        let nonzero = |l: &mut Self, code: u8| -> Result<V, Why> {
            let read = |h: u8| inst.read(Reg::S(code + h)).ok_or("an unread branch mask");
            let (lo, hi) = (l.i32_of(read(0)?)?, l.i32_of(read(1)?)?);
            let o = l.s.m.bin(c, Bin::Or, lo, hi);
            let z = l.s.m.i32(0);
            Ok(l.s.m.cmp(c, pred::NE, o, z))
        };
        Ok(match inst.name() {
            n @ ("s_cbranch_scc0" | "s_cbranch_scc1") => {
                let scc = inst.read(Reg::Scc).ok_or("an unread SCC")?;
                let set = match self.get(scc)? {
                    Val::Bit(x) => x,
                    Val::I32(x) => {
                        let z = self.s.m.i32(0);
                        self.s.m.cmp(c, pred::NE, x, z)
                    }
                    Val::Missing(w) => return Err(w),
                };
                if n == "s_cbranch_scc1" {
                    (set, taken, not_taken)
                } else {
                    (set, not_taken, taken)
                }
            }
            "s_cbranch_execnz" => (nonzero(self, EXEC)?, taken, not_taken),
            "s_cbranch_execz" => (nonzero(self, EXEC)?, not_taken, taken),
            "s_cbranch_vccnz" => (nonzero(self, VCC)?, taken, not_taken),
            "s_cbranch_vccz" => (nonzero(self, VCC)?, not_taken, taken),
            _ => return Err("a branch the console wave's halves do not follow"),
        })
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = self.s.m.i32(super::loop_limit());
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through (in turn,
    /// its console wave's turn ends there).
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        let end = self.next_turn().map_or(Term::Ret, Term::Br);
        self.s.m.terminate(a, end);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// Whether no lane that enters block `b` from its `n`th predecessor reads
    /// `v`, a value joined there, so that what the edge brings never matters:
    /// compares in `b` that every such lane answers alike (against a
    /// constant the edge brings) rule out each block that reads it. A join it
    /// feeds on a forward edge reads it only where no lane bringing it there
    /// reads that join: `facts` hold for every lane that brings `v` to `b`.
    /// A join can be reached along many paths, so `budget` bounds the joins
    /// followed.
    fn unread_from(
        &self,
        v: Value,
        b: usize,
        n: usize,
        facts: &[(Bool, bool)],
        budget: &mut u32,
    ) -> bool {
        let (p, st) = (self.x.p, self.x.st);
        let along = |w: Value| match self.phi_at.get(&w) {
            Some(&(pb, i)) if pb == b => p.blocks[b].phis[i].args[n],
            _ => w,
        };
        let immediate = |op: &Operand| match *op {
            Operand::Const(code) => crate::resource::inline(code),
            Operand::Literal(k) => Some(k),
            _ => None,
        };
        // A move of a constant every lane reaching it runs.
        let constant = |w: Value| {
            let (db, di) = p.blocks.iter().enumerate().find_map(|(db, block)| {
                let di = block
                    .insts
                    .iter()
                    .position(|d| d.writes.iter().any(|&(_, x)| x == w))?;
                Some((db, di))
            })?;
            let d = &p.blocks[db].insts[di];
            match &d.fields {
                Fields::Sop { src, .. } if d.name() == "s_mov_b32" => immediate(src.first()?),
                Fields::Valu(f)
                    if d.name() == "v_mov_b32"
                        && st.guards[db][di].is_none()
                        && (f.neg, f.abs, f.clamp, f.omod) == (0, 0, false, 0)
                        && f.sdwa.is_none()
                        && f.dpp.is_none() =>
                {
                    immediate(f.src.first()?)
                }
                _ => None,
            }
        };
        let mut known = facts.to_vec();
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let Some((op, ty)) = inst
                .name()
                .strip_prefix("v_cmp_")
                .and_then(|t| t.split_once('_'))
            else {
                continue;
            };
            if st.guards[b][i].is_some() || f.sdwa.is_some() || f.dpp.is_some() || f.src.len() != 2
            {
                continue;
            }
            let Some(&bit) = inst.writes.iter().find_map(|(_, w)| st.masks.get(w)) else {
                continue;
            };
            let operand = |op: &Operand| match *op {
                Operand::Vector { reg, .. } => constant(along(inst.read(Reg::V(reg))?)),
                Operand::Scalar { code, .. } => constant(along(inst.read(Reg::S(code))?)),
                _ => immediate(op),
            };
            let (Some(x), Some(y)) = (operand(&f.src[0]), operand(&f.src[1])) else {
                continue;
            };
            let order = match ty {
                "u32" => x.cmp(&y),
                "i32" => (x as i32).cmp(&(y as i32)),
                _ => continue,
            };
            let holds = match op {
                "eq" => order.is_eq(),
                "ne" => order.is_ne(),
                "lt" => order.is_lt(),
                "le" => order.is_le(),
                "gt" => order.is_gt(),
                "ge" => order.is_ge(),
                _ => continue,
            };
            known.push((bit, holds));
        }
        if known.is_empty() {
            return false;
        }
        let unreached = |r: usize, edge: Option<(Bool, bool)>| {
            let mut all = known.clone();
            all.push((st.reach[r], true));
            all.extend(edge);
            !st.bools.meet(&all)
        };
        let only_user = |x: Value| {
            let mut users = p
                .blocks
                .iter()
                .flat_map(|bl| &bl.insts)
                .filter(move |u| u.reads.iter().any(|&(_, y)| y == x));
            let joined = p
                .blocks
                .iter()
                .any(|bl| bl.phis.iter().any(|phi| phi.args.contains(&x)));
            match (users.next(), users.next()) {
                (Some(u), None) if !joined => Some(u),
                _ => None,
            }
        };
        // A compare of `v` whose answer only selects a mask that an OR with a
        // mask set in every such lane of block `r` overrides.
        let absorbed = |r: usize, i: &Inst| {
            let Some(sel) = i
                .name()
                .starts_with("s_cmp_")
                .then(|| i.write(Reg::Scc))
                .flatten()
                .and_then(only_user)
                .filter(|s| matches!(s.name(), "s_cselect_b32" | "s_cselect_b64"))
            else {
                return false;
            };
            let Some(or) = sel.writes.first().and_then(|&(_, w)| only_user(w)) else {
                return false;
            };
            let set = |&(_, x): &(Reg, Value)| {
                st.masks.get(&x).is_some_and(|&bit| {
                    let mut all = known.clone();
                    all.extend([(st.reach[r], true), (bit, false)]);
                    !st.bools.meet(&all)
                })
            };
            matches!(or.name(), "s_or_b32" | "s_or_b64")
                && sel
                    .writes
                    .iter()
                    .all(|&(_, w)| only_user(w).is_some_and(|u| std::ptr::eq(u, or)))
                && or
                    .reads
                    .iter()
                    .filter(|&&(_, x)| !sel.writes.iter().any(|&(_, w)| w == x))
                    .any(set)
        };
        for (r, block) in p.blocks.iter().enumerate() {
            let reads = block
                .insts
                .iter()
                .any(|i| i.reads.iter().any(|&(_, x)| x == v) && !absorbed(r, i));
            if reads && !unreached(r, None) {
                return false;
            }
            for phi in &block.phis {
                for (m, _) in phi.args.iter().enumerate().filter(|&(_, &a)| a == v) {
                    let s = block.preds[m];
                    let edge = match st.exits[s] {
                        LaneExit::Branch {
                            when,
                            taken,
                            not_taken,
                        } if taken != not_taken => Some((when, taken == r)),
                        _ => None,
                    };
                    if unreached(s, edge) || !self.read_phis.contains(&phi.value) {
                        continue;
                    }
                    if self.rpo[r] <= self.rpo[s] || *budget == 0 {
                        return false;
                    }
                    *budget -= 1;
                    let mut carried = known.clone();
                    carried.push((st.reach[s], true));
                    carried.extend(edge);
                    if !self.unread_from(phi.value, r, m, &carried, budget) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                }
                let read = |l: &Self| {
                    l.read_phis.contains(&phi.value)
                        && !l.unread_from(phi.value, b, n, &[], &mut 256)
                };
                if self.dims.contains_key(&arg) && !self.dims.contains_key(&phi.value) && read(self)
                {
                    return Err(Stop::Retry(phi.value, FIELDS));
                }
                let got = match self.last_pass.get(&arg) {
                    Some(&(l, val)) if l != q => Some(val),
                    _ => self.vals.get(&arg).copied(),
                };
                let x = match (got, bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Bit(_)), false) if read(self) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    (Some(Val::Missing(why)), _) if read(self) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if read(self) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = match self.x.st.masks.get(&phi.args[n]) {
                    Some(&m) => self.boolean(m, qb),
                    // A mask held only as a number (as one popped off a
                    // stack in VGPR lanes): the console's, so the lane tests
                    // its own bit.
                    None if self.console_lanes => self.own_bit(phi.args[n], qb),
                    None => Err("a lane-mask join of an untracked mask"),
                }
                .map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(p.blocks.iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32"));

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        }).unwrap();

        let container =
            crate::Container::parse(&emitted.dxil).unwrap();

        let features =
            container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    /// Each lane's texture (table entry lane & 1) read in a waterfall loop
    /// that only takes its height from the descriptor, as `GetDimensions`
    /// does, then sampled through the same entry in a second one; the height
    /// is copied to v17 `inside` the first loop, else after it from the SGPR
    /// the loop leaves.
    fn height_then_sample(inside: bool) -> Vec<u32> {
        let copy = 0x7E22_0220; // v_mov_b32 v17, s32
        let mut w = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0203,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x360E_0081, // v_and_b32 v7, 1, v0
            0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xBE97_037E, // s_mov_b32 s23, exec_lo
            0xBE96_037E, // s_mov_b32 s22, exec_lo
            0x7E28_0507, // 0x40: v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
            0xF40C_0602,
            0x2800_0000, // s_load_dwordx8 s[24:31], s[4:5], s20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x93A0_FF1A,
            0x0010_000E, // s_bfe_u32 s32, s26, 0x10000e (height - 1)
            0x8020_8120, // s_add_u32 s32, s32, 1
            0x93A1_FF1B,
            0x0004_000C, // s_bfe_u32 s33, s27, 0x4000c (BASE_LEVEL)
            0x9020_2120, // s_lshr_b32 s32, s32, s33
            0x84A0_8120, // s_max_u32 s32, s32, 1
        ];
        if inside {
            w.push(copy);
        }
        w.extend([
            0xBE96_4716,                                    // s_andn2_wrexec_b32 s22, s22
            if inside { 0xBF89_FFF1 } else { 0xBF89_FFF2 }, // s_cbranch_execnz -> 0x40
            0xBEFE_0317,                                    // s_mov_b32 exec_lo, s23
        ]);
        if !inside {
            w.push(copy);
        }
        w.extend([
            0xBE95_037E, // s_mov_b32 s21, exec_lo
            0x7E28_0507, // v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
            0xF40C_0302,
            0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            0xBF89_FFF6, // s_cbranch_execnz -> v_readfirstlane_b32
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xE070_1080,
            0x800A_1110, // buffer_store_dword v17, v16, s[40:43], 0 offen offset:128
            0xBF81_0000, // s_endpgm
        ]);
        w
    }

    #[test]
    fn descriptor_dwords_a_waterfall_loop_leaves_in_scalars_are_the_last_passs() {
        // The descriptor's kind comes from the sampling loop's load of the
        // same entry; a height kept in a VGPR is each lane's own.
        let (layout, used) = texture_and_uav_tables();
        assert!(lower_wave(&height_then_sample(true), layout, used, false).is_ok());
        // On the console the SGPR holds the last pass's height: each lane
        // reads it from a lane of that pass, on the console's lanes.
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(&height_then_sample(false), layout, used, false).unwrap();
        assert!(l.wave_ops);
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }

    #[test]
    fn scalar_pack_extend_and_absdiff_lower() {
        // Each computes into s2 from two constants and branches on it (only
        // code with an effect needs its operands).
        let head = [
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0383, // s_mov_b32 s1, 3
        ];
        let tail = [
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let ops = [
            0x9602_0100, // s_absdiff_i32 s2, s0, s1
            0xBE82_1901, // s_sext_i32_i8 s2, s1
            0xBE82_1A01, // s_sext_i32_i16 s2, s1
            0x9902_0100, // s_pack_ll_b32_b16 s2, s0, s1
            0x9982_0100, // s_pack_lh_b32_b16 s2, s0, s1
            0x9A02_0100, // s_pack_hh_b32_b16 s2, s0, s1
        ];
        for op in ops {
            let words: Vec<u32> = head.iter().chain(&[op]).chain(&tail).copied().collect();
            lowers(&words).unwrap();
        }
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }
}
