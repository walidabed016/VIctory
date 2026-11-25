//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU
//! by a compute shader of ours (HLSL compiled by the system's compiler; a
//! 5.1 shader with its root signature inside), into a buffer of the list's
//! own that the execution then reads; the title's records stay as written.
//!
//! The clamps run in segments: the first indirect execution after a
//! barrier the title records (or after a dispatch, which may write
//! arguments) records one indirect dispatch of the clamp, whose commands
//! (one per execution: records in, records out, kind, count) the segment's
//! executions add on the CPU before the list runs. A title cannot read
//! arguments a pass wrote without a barrier between them, so what a
//! segment's executions read is written before it starts. A clamp per
//! execution drained the GPU two or three times around every indirect draw
//! (~120 ms per world frame).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_SHADER_MACRO, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#ifdef FUSED
#define RS "UAV(u0), UAV(u1), UAV(u2), UAV(u3), SRV(t0)"
// A segment's executions, a group row each, 8 words apiece: kind, stride,
// how many, flags (as Record's), then where its records, its count and
// its clamped records are from args, count and clamped.
ByteAddressBuffer executions : register(t0);
#else
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
// One execution's records: kind (PC argument numbering), stride, how many,
// flags (1: count records with work and empty ones, 2: a count word
// limits them).
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; };
#endif
// The title's records, only read.
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer count : register(u2);
// What the execution reads: the records clamped, then the count.
RWByteAddressBuffer clamped : register(u3);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID) {
#ifdef FUSED
    uint4 record = executions.Load4(group.y * 32);
    uint3 offsets = executions.Load3(group.y * 32 + 16);
    uint kind = record.x;
    uint stride = record.y;
    uint records = record.z;
    uint flags = record.w;
#else
    uint3 offsets = uint3(0, 0, 0);
#endif
    uint n = records;
    if ((flags & 2) != 0) {
        n = min(count.Load(offsets.y), records);
        if (id.x == 0) {
            clamped.Store(offsets.z + records * stride, n);
        }
    }
    if (id.x >= n) return;
    uint from = offsets.x + id.x * stride;
    uint at = offsets.z + id.x * stride;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint vertices = min(args.Load(from), 1u << 24);
        uint instances = min(args.Load(from + 4), (1u << 26) / max(vertices, 1u));
        cost = vertices * instances;
        clamped.Store4(at, uint4(vertices, instances, args.Load2(from + 8)));
        if (kind == 1) {
            clamped.Store(at + 16, args.Load(from + 16));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(from), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        clamped.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(from);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        clamped.Store3(at, g);
    }
    if ((flags & 1) != 0) {
        // Diagnostics: records with work and empty ones, per kind.
        uint base = kind >= 10 ? 1u : (kind == 2 ? 5u : 3u);
        uint seen;
        budget.InterlockedAdd((base + (cost == 0 ? 1u : 0u)) * 4, 1, seen);
    }
    uint before;
    budget.InterlockedAdd(0, cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        clamped.Store(at, 0);
    }
}
"#;

/// A clamp command as the clamp's command signature reads it (root
/// parameters in order): the record constants, the root views of the
/// title's records, of its count and of where the clamped records go, then
/// the dispatch.
#[repr(C)]
#[derive(Clone, Copy)]
struct Command {
    record: [u32; 4],
    args: u64,
    count: u64,
    clamped: u64,
    groups: [u32; 3],
    _pad: u32,
}

impl Command {
    /// The clamp of an execution's records at host address `args` (and
    /// its count at `count`, if the title gives one) into `clamped`.
    fn new(
        (kind, stride, records): (u32, u32, u32),
        args: u64,
        count: Option<u64>,
        clamped: u64,
    ) -> Self {
        let flags = u32::from(cfg!(feature = "diagnostics")) | if count.is_some() { 2 } else { 0 };
        Command {
            record: [kind, stride, records, flags],
            args,
            count: count.unwrap_or(clamped),
            clamped,
            groups: [records.div_ceil(64), 1, 1],
            _pad: 0,
        }
    }
}

/// Clamp commands one segment holds; more start another.
const COMMANDS: u32 = 256;
/// A segment's bytes in its table: the command count, then the commands.
const SEGMENT: u64 = 8 + COMMANDS as u64 * size_of::<Command>() as u64;
/// Bytes of a table (upload memory) and of a buffer of clamped records.
const TABLE: u64 = 64 * SEGMENT;
const RING: u64 = 4 << 20;
/// Where a fused segment's executions start in its part of the table,
/// after the count word and the dispatch ([`fused`]).
const EXECUTIONS: u64 = 32;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    signature: ID3D12CommandSignature,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(false) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

/// The fused clamp, with `VICTORY_FUSED_CLAMP=1`: a segment's clamp is one
/// dispatch whose group rows are its executions, read from the table, at
/// offsets from views the segment sets once (its first execution's records
/// and count, less 1 GiB), instead of an ExecuteIndirect of a command per
/// execution, which costs ~3.4 µs a command on the host GPU against ~4.4
/// µs for a whole dispatch-only one (VICT-9's probe). An execution whose
/// records or count lie outside the views' 2 GiB opens a segment of its
/// own.
fn fused() -> Option<&'static Clamp> {
    static FUSED: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    FUSED
        .get_or_init(|| {
            if std::env::var("VICTORY_FUSED_CLAMP").is_ok_and(|v| v == "1") {
                tracing::info!("indirect clamps fused, one dispatch a segment (VICTORY_FUSED_CLAMP)");
            } else {
                return None;
            }
            make(true)
                .inspect_err(|e| tracing::warn!(error = %e, "no fused argument clamp: a command per execution"))
                .ok()
        })
        .as_ref()
}

fn make(fused: bool) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let defines = [
        D3D_SHADER_MACRO {
            Name: s!("FUSED"),
            Definition: s!("1"),
        },
        D3D_SHADER_MACRO::default(),
    ];
    // SAFETY: the source, defines and names live through the call; out
    // pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            fused.then_some(defines.as_ptr()),
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    let view = |root: u32| D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW,
        Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
            UnorderedAccessView: D3D12_INDIRECT_ARGUMENT_DESC_0_4 {
                RootParameterIndex: root,
            },
        },
    };
    // In `Command`'s order, which is the root parameters' (the runtime
    // wants them increasing): the constants (parameter 0), root views 1
    // (records), 3 (count) and 4 (clamped), the dispatch.
    let arguments = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: 0,
                    DestOffsetIn32BitValues: 0,
                    Num32BitValuesToSet: 4,
                },
            },
        },
        view(1),
        view(3),
        view(4),
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        let mut signature: Option<ID3D12CommandSignature> = None;
        if fused {
            device.CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 12,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &arguments[4],
                    NodeMask: 0,
                },
                None,
                &mut signature,
            )?;
        } else {
            device.CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: size_of::<Command>() as u32,
                    NumArgumentDescs: arguments.len() as u32,
                    pArgumentDescs: arguments.as_ptr(),
                    NodeMask: 0,
                },
                &root,
                &mut signature,
            )?;
        }
        let signature = signature.ok_or_else(windows::core::Error::empty)?;
        Ok(Clamp {
            root,
            pso,
            signature,
        })
    }
}

/// A list's clamps: the buffers clamped records go to (video memory) and
/// whether each is in the indirect-argument state in the list, the one in
/// use and its bytes used; the tables of clamp commands (upload memory,
/// with their mappings), the one in use and its bytes used; the open
/// segment.
#[derive(Default)]
pub(crate) struct Clamps {
    rings: Vec<(ID3D12Resource, u64)>,
    reading: Vec<bool>,
    ring: usize,
    used: u64,
    tables: Vec<(ID3D12Resource, usize)>,
    table: usize,
    table_used: u64,
    open: Option<Segment>,
}

// SAFETY: the table pointers are persistent mappings of upload buffers,
// written by the list's recording thread only.
unsafe impl Send for Clamps {}

/// An open segment: its table, where in it, and the commands so far; when
/// fused ([`fused`]), its views of records and counts and its widest
/// execution's groups.
struct Segment {
    table: usize,
    at: u64,
    commands: u32,
    views: Option<(u64, u64)>,
    groups: u32,
}

impl Clamps {
    /// Whether an execution taking `need` bytes of a ring, its `len`
    /// bytes of records at `args` and its count at `count`, joins the open
    /// segment: one is open, with room for it, and a fused one's views
    /// reach them.
    fn joinable(&self, need: u64, (args, len): (u64, u64), count: Option<u64>) -> bool {
        let reach = |view: u64, at: u64, len: u64| at >= view && at + len - view <= 1 << 31;
        self.open.as_ref().is_some_and(|segment| {
            segment.commands < COMMANDS
                && segment.views.is_none_or(|(records, counts)| {
                    reach(records, args, len) && count.is_none_or(|c| reach(counts, c, 4))
                })
        }) && self.used + need <= self.rings[self.ring].1
    }

    /// After the list is reset: everything reused from the start.
    pub(crate) fn reset(&mut self) {
        (self.ring, self.used, self.table, self.table_used, self.open) = (0, 0, 0, 0, None);
        self.reading.iter_mut().for_each(|r| *r = false);
    }

    /// The ring with `need` bytes free from `used` on, made if none is.
    fn room(&mut self, need: u64) -> Option<usize> {
        while self.ring < self.rings.len() && self.used + need > self.rings[self.ring].1 {
            (self.ring, self.used) = (self.ring + 1, 0);
        }
        if self.ring == self.rings.len() {
            let size = need.max(RING);
            let buffer = default_buffer(size)?;
            crate::host::name(&buffer, || format!("clamped indirect records {size:#x}"));
            self.rings.push((buffer, size));
            self.reading.push(false);
            self.used = 0;
        }
        Some(self.ring)
    }

    /// Room for one more segment in a table, made if none has it: the
    /// table and the segment's offset.
    fn segment(&mut self) -> Option<(usize, u64)> {
        if self.table < self.tables.len() && self.table_used + SEGMENT > TABLE {
            (self.table, self.table_used) = (self.table + 1, 0);
        }
        if self.table == self.tables.len() {
            let (buffer, data) = host()?.upload_buffer(TABLE)?;
            crate::host::name(&buffer, || "indirect clamp commands".into());
            self.tables.push((buffer, data as usize));
            self.table_used = 0;
        }
        let at = self.table_used;
        self.table_used += SEGMENT;
        Some((self.table, at))
    }
}

/// Bytes an execution of `records` records `stride` bytes apart takes in
/// a ring: the records, then the count.
fn region(stride: u32, records: u32) -> u64 {
    (u64::from(stride) * u64::from(records) + 4).next_multiple_of(256)
}

/// A buffer in video memory shaders write.
fn default_buffer(size: u64) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    let made = unsafe {
        host()?.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, size, "no buffer for clamped indirect records");
        return None;
    }
    buffer
}

/// Before an indirect execution of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart, at host address `args` (its
/// count at `count`), is recorded in `list`: opens a segment unless one it
/// joins is open, recording its clamp. Replaces the list's compute root
/// signature and pipeline then (callers bind theirs after). False if the
/// clamp is not available.
pub(crate) fn open(
    list: &HostList,
    (kind, stride, records): (u32, u32, u32),
    args: u64,
    count: Option<u64>,
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !stride.is_multiple_of(4) {
        return false;
    }
    let need = region(stride, records);
    let len = u64::from(stride) * u64::from(records);
    let mut s = list.clamps();
    if s.joinable(need, (args, len), count) {
        return true;
    }
    s.open = None;
    let (Some(ring), Some((table, at))) = (s.room(need), s.segment()) else {
        return false;
    };
    let (buffer, data) = (s.tables[table].0.clone(), s.tables[table].1);
    let rings = s.rings[ring].0.clone();
    // SAFETY: the segment's count word in the table's mapping, which the
    // GPU reads only once the list runs.
    unsafe { ((data + at as usize) as *mut u32).write(0) };
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("clamp kind {kind} records {records}"));
    #[cfg(not(feature = "diagnostics"))]
    let _ = kind;
    let l = &list.list;
    let mut before = vec![uav_barrier()];
    if s.reading[ring] {
        before.push(transition(
            &rings,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
    }
    let view = |at: u64| at.saturating_sub(1 << 30) & !255;
    let fused = fused().map(|f| (f, view(args), view(count.unwrap_or(args))));
    // SAFETY: an open list; our root signature, pipeline, signature and
    // buffers, which the list's clamps keep; the segment's part of the
    // table's mapping, which the GPU reads only once the list runs.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&before);
        match fused {
            Some((f, records, counts)) => {
                ((data + at as usize + 8) as *mut [u32; 3]).write_unaligned([0, 0, 1]);
                l.SetComputeRootSignature(&f.root);
                l.SetPipelineState(&f.pso);
                l.SetComputeRootUnorderedAccessView(0, records);
                l.SetComputeRootUnorderedAccessView(1, budget);
                l.SetComputeRootUnorderedAccessView(2, counts);
                l.SetComputeRootUnorderedAccessView(3, rings.GetGPUVirtualAddress());
                l.SetComputeRootShaderResourceView(
                    4,
                    buffer.GetGPUVirtualAddress() + at + EXECUTIONS,
                );
                l.ExecuteIndirect(&f.signature, 1, &buffer, at + 8, None, 0);
            }
            None => {
                l.SetComputeRootSignature(&c.root);
                l.SetPipelineState(&c.pso);
                l.SetComputeRootUnorderedAccessView(2, budget);
                l.ExecuteIndirect(&c.signature, COMMANDS, &buffer, at + 8, &buffer, at);
            }
        }
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                &rings,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    s.reading[ring] = true;
    s.open = Some(Segment {
        table,
        at,
        commands: 0,
        views: fused.map(|(_, records, counts)| (records, counts)),
        groups: 0,
    });
    true
}

/// Adds an execution to the open segment ([`open`] it first): the clamp
/// of its records at host address `args` (and the count at `count`, if
/// the title gives one). Returns the buffer the execution reads instead,
/// where its records are and where its count is.
pub(crate) fn add(
    list: &HostList,
    (kind, stride, records): (u32, u32, u32),
    args: u64,
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64, Option<u64>)> {
    if !args.is_multiple_of(4) || count.is_some_and(|c| !c.is_multiple_of(4)) {
        return None;
    }
    let mut s = list.clamps();
    let (ring, at) = (s.ring, s.used);
    let Some(segment) = &s.open else {
        return None;
    };
    let (table, offset, n, views) = (segment.table, segment.at, segment.commands, segment.views);
    let groups = segment.groups.max(records.div_ceil(64));
    let buffer = s.rings[ring].0.clone();
    // SAFETY: a live buffer.
    let clamped = unsafe { buffer.GetGPUVirtualAddress() } + at;
    let len = u64::from(stride) * u64::from(records);
    let command = Command::new((kind, stride, records), args, count, clamped);
    let data = s.tables[table].1 + offset as usize;
    // SAFETY: the segment's part of the table's mapping (room for
    // `COMMANDS` commands after the count, or a fused segment's dispatch
    // and executions), which the GPU reads only once the list runs.
    unsafe {
        match views {
            Some((records, counts)) => {
                let [kind, stride, how_many, flags] = command.record;
                let from = |view: u64, at: u64| (at - view) as u32;
                let counted = count.map_or(0, |c| from(counts, c));
                ((data + EXECUTIONS as usize) as *mut [u32; 8])
                    .add(n as usize)
                    .write_unaligned([
                        kind,
                        stride,
                        how_many,
                        flags,
                        from(records, args),
                        counted,
                        at as u32,
                        0,
                    ]);
                ((data + 8) as *mut [u32; 3]).write_unaligned([groups, n + 1, 1]);
            }
            None => {
                ((data + 8) as *mut Command)
                    .add(n as usize)
                    .write_unaligned(command);
                (data as *mut u32).write(n + 1);
            }
        }
    }
    s.used += region(stride, records);
    let segment = s.open.as_mut()?;
    segment.commands += 1;
    segment.groups = groups;
    if segment.commands == COMMANDS {
        s.open = None;
    }
    Some((buffer, at, count.map(|_| at + len)))
}

/// Whether an execution of `records` records `stride` bytes apart at host
/// address `args` (its count at `count`) joins the open segment's clamp
/// ([`open`] would record none).
pub(crate) fn joins(
    list: &HostList,
    (_, stride, records): (u32, u32, u32),
    args: u64,
    count: Option<u64>,
) -> bool {
    let len = u64::from(stride) * u64::from(records);
    list.clamps()
        .joinable(region(stride, records), (args, len), count)
}

/// Clamps an execution alone in its segment with a dispatch of its own,
/// recorded now: the records at host address `args` (and the count at
/// `count`, if the title gives one). For an indirect dispatch when no
/// segment is open ([`joins`]): its groups may write arguments later
/// executions read, so nothing joins its segment, and a segment's
/// ExecuteIndirect between barriers costs ~35 µs on the host GPU against
/// ~5 µs for a dispatch (VICT-9's probe). Replaces the list's compute root
/// signature and pipeline (callers bind theirs after). Returns as [`add`]
/// does.
pub(crate) fn clamp_alone(
    list: &HostList,
    (kind, stride, records): (u32, u32, u32),
    args: u64,
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64, Option<u64>)> {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !stride.is_multiple_of(4)
        || !args.is_multiple_of(4)
        || count.is_some_and(|c| !c.is_multiple_of(4))
    {
        return None;
    }
    let mut s = list.clamps();
    s.open = None;
    let ring = s.room(region(stride, records))?;
    let (buffer, at) = (s.rings[ring].0.clone(), s.used);
    // SAFETY: a live buffer.
    let clamped = unsafe { buffer.GetGPUVirtualAddress() } + at;
    let command = Command::new((kind, stride, records), args, count, clamped);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("clamp kind {kind} records {records}"));
    let l = &list.list;
    let mut before = vec![uav_barrier()];
    if s.reading[ring] {
        before.push(transition(
            &buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
    }
    // SAFETY: an open list; our root signature and pipeline, the list's
    // buffers, which its clamps keep.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&before);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, command.record.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, command.args);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, command.count);
        l.SetComputeRootUnorderedAccessView(4, command.clamped);
        l.Dispatch(command.groups[0], 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                &buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    s.reading[ring] = true;
    s.used += region(stride, records);
    let len = u64::from(stride) * u64::from(records);
    Some((buffer, at, count.map(|_| at + len)))
}

/// Ends the open segment: the title may write arguments from here on
/// (after a barrier it records, or a dispatch).
pub(crate) fn close(list: &HostList) {
    list.clamps().open = None;
}

/// Diagnostics: indirect records the clamp saw run, as (with work, empty)
/// for mesh dispatches, draws and dispatches, added from each list's
/// budget words when it is used again.
#[cfg(feature = "diagnostics")]
pub(crate) static SEEN: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
        if std::env::var("VICTORY_FUSED_CLAMP").is_ok_and(|v| v == "1") {
            assert!(super::fused().is_some(), "the fused clamp builds");
        }
    }

    /// A compute pipeline whose groups count themselves into a root UAV's
    /// first word.
    fn counting() -> (ID3D12RootSignature, ID3D12PipelineState) {
        const COUNT: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer counted : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { uint was; counted.InterlockedAdd(0, 1, was); }
"#;
        let device = &host().unwrap().device;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe {
            D3DCompile(
                COUNT.as_ptr().cast(),
                COUNT.len(),
                s!("count"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// Indirect dispatches read their records clamped, two to a segment
    /// and one in a segment after a barrier, with a count limiting one;
    /// the title's records stay as written, and the debug layer finds
    /// nothing wrong with the states.
    #[test]
    fn indirect_dispatches_run_their_records_clamped_in_segments() {
        let _serial = crate::watch::serial();
        if !crate::shadow::tests::gpu() {
            return;
        }
        let host = host().unwrap();
        let (title, words) = host.shared_buffer(4096).unwrap();
        let (counted, total) = host.shared_buffer(256).unwrap();
        let words = words.cast::<u32>();
        // At 0: 5 groups, then 70000 x 70000 (past the limits: none). At
        // 64: 3 groups then 4, of which the count at 128 runs one. At 192:
        // 2 groups.
        let records: [(usize, [u32; 3]); 5] = [
            (0, [5, 1, 1]),
            (3, [70000, 70000, 1]),
            (16, [3, 1, 1]),
            (19, [4, 1, 1]),
            (48, [2, 1, 1]),
        ];
        // SAFETY: inside the mapped buffers.
        unsafe {
            for (at, groups) in records {
                words.add(at).cast::<[u32; 3]>().write(groups);
            }
            words.add(32).write(1);
            total.cast::<u32>().write(0);
        }
        // SAFETY: live buffers.
        let (base, out) = unsafe { (title.GetGPUVirtualAddress(), counted.GetGPUVirtualAddress()) };
        let (root, pso) = counting();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: one argument, no root arguments.
        unsafe {
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None,
                    &mut signature,
                )
                .unwrap()
        };
        let signature = signature.unwrap();
        let list = host.new_list().unwrap();
        let run = |args: u64, records: u32, count: Option<u64>| {
            assert!(open(&list, (2, 12, records), args, count));
            let (clamped, at, count_at) = add(&list, (2, 12, records), args, count).unwrap();
            // SAFETY: an open list; live objects.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootUnorderedAccessView(0, out);
                list.list.ExecuteIndirect(
                    &signature,
                    records,
                    &clamped,
                    at,
                    count_at.map(|_| &clamped),
                    count_at.unwrap_or(0),
                );
            }
        };
        run(base, 2, None);
        run(base + 64, 2, Some(base + 128));
        let first = list.clamps().open.as_ref().map(|s| s.commands);
        close(&list);
        run(base + 192, 1, None);
        list.close();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        assert_eq!(first, Some(2), "the first two executions share a segment");
        // SAFETY: the mapped buffers, which the finished list wrote.
        let (ran, kept) = unsafe { (total.cast::<u32>().read_volatile(), words.add(3).read()) };
        assert_eq!(ran, 5 + 3 + 2, "groups run");
        assert_eq!(kept, 70000, "the title's records stay as written");
        assert_eq!(host.log_messages(), None, "debug layer errors");
        list.reset();
    }

    /// Indirect dispatches clamped alone, each by a dispatch of its own,
    /// read their records clamped, one with a count limiting it, one after
    /// another from the same ring; the title's records stay as written, and
    /// the debug layer finds nothing wrong with the states.
    #[test]
    fn indirect_dispatches_clamped_alone_run_their_records_clamped() {
        let _serial = crate::watch::serial();
        if !crate::shadow::tests::gpu() {
            return;
        }
        let host = host().unwrap();
        let (title, words) = host.shared_buffer(4096).unwrap();
        let (counted, total) = host.shared_buffer(256).unwrap();
        let words = words.cast::<u32>();
        // At 0: 5 groups, then 70000 x 70000 (past the limits: none). At
        // 64: 3 groups then 4, of which the count at 128 runs one.
        let records: [(usize, [u32; 3]); 4] = [
            (0, [5, 1, 1]),
            (3, [70000, 70000, 1]),
            (16, [3, 1, 1]),
            (19, [4, 1, 1]),
        ];
        // SAFETY: inside the mapped buffers.
        unsafe {
            for (at, groups) in records {
                words.add(at).cast::<[u32; 3]>().write(groups);
            }
            words.add(32).write(1);
            total.cast::<u32>().write(0);
        }
        // SAFETY: live buffers.
        let (base, out) = unsafe { (title.GetGPUVirtualAddress(), counted.GetGPUVirtualAddress()) };
        let (root, pso) = counting();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: one argument, no root arguments.
        unsafe {
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None,
                    &mut signature,
                )
                .unwrap()
        };
        let signature = signature.unwrap();
        let list = host.new_list().unwrap();
        let run = |args: u64, records: u32, count: Option<u64>| {
            assert!(
                !joins(&list, (2, 12, records), args, count),
                "no segment is open"
            );
            let (clamped, at, count_at) =
                clamp_alone(&list, (2, 12, records), args, count).unwrap();
            // SAFETY: an open list; live objects.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootUnorderedAccessView(0, out);
                list.list.ExecuteIndirect(
                    &signature,
                    records,
                    &clamped,
                    at,
                    count_at.map(|_| &clamped),
                    count_at.unwrap_or(0),
                );
            }
        };
        run(base, 2, None);
        run(base + 64, 2, Some(base + 128));
        list.close();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: the mapped buffers, which the finished list wrote.
        let (ran, kept) = unsafe { (total.cast::<u32>().read_volatile(), words.add(3).read()) };
        assert_eq!(ran, 5 + 3, "groups run");
        assert_eq!(kept, 70000, "the title's records stay as written");
        assert_eq!(host.log_messages(), None, "debug layer errors");
        list.reset();
    }
}
