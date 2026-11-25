//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::HostShader;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// The title-memory page table made for this list (`crate::memory`)
    /// and its view heap.
    memory: Option<(usize, crate::memory::Map)>,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            match (heap, crate::memory::heap()) {
                (MEMORY_HEAP, Some(ours)) => ours.release(first, n),
                _ => crate::descriptor::release_dynamic(heap, first, n),
            }
        }
    }
}

/// The key of our own view heap (`crate::memory::heap`) among the title's.
const MEMORY_HEAP: usize = usize::MAX;

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

/// Depth textures, their unordered twins and the planes they hold.
type Twins = std::cell::RefCell<Vec<(This, This, u32)>>;

thread_local! {
    /// The depth textures the draw or dispatch being recorded on this
    /// thread writes through their unordered twins, and the twins
    /// ([`crate::resource::unordered_twin`]): copied back after it.
    static TWINS: Twins = const { std::cell::RefCell::new(Vec::new()) };
    /// Those of them whose twins hold a plane each, to fill after its
    /// tables ([`twins_in`]).
    static FILLS: Twins = const { std::cell::RefCell::new(Vec::new()) };
    /// Whether [`State::bind`] binds again after filling them.
    static REBINDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Fills the plane twins the draw or dispatch being bound writes through
/// from their textures ([`crate::planes`]). Returns whether that recorded
/// a compute pass, replacing the list's compute root signature, pipeline
/// and root arguments.
fn twins_in(list: &HostList) -> bool {
    const UNORDERED_ACCESS: u32 = 0x8;
    let mut passes = false;
    for (depth, twin, plane) in FILLS.take() {
        passes |= crate::planes::exchange(list, depth, twin, plane, false);
        crate::list::ensure_state(list, twin, UNORDERED_ACCESS);
    }
    passes
}

/// Copies what the draw or dispatch just recorded wrote into unordered
/// twins back into their depth textures. Returns whether that recorded a
/// compute pass, replacing the list's compute root signature and pipeline.
fn twins_back(list: &HostList) -> bool {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let mut passes = false;
    for (depth, twin, plane) in TWINS.take() {
        if crate::planes::split(depth) {
            passes |= crate::planes::exchange(list, depth, twin, plane, true);
            continue;
        }
        let (Some(to), Some(from)) = (
            crate::resource::host_resource(depth),
            crate::resource::host_resource(twin),
        ) else {
            continue;
        };
        crate::list::ensure_state(list, twin, COPY_SOURCE);
        crate::list::ensure_state(list, depth, COPY_DEST);
        // SAFETY: an open list; live textures of one description.
        unsafe { list.list.CopyResource(to, from) };
    }
    passes
}

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// Since the last frame line (diagnostics): view-heap slots made through
/// user data for draws and dispatches ([`State::heap_slots`]), stages bound
/// with heap accesses whose slot only the running shader knows (an index
/// loaded from memory or chosen per lane), and those accesses. The host
/// cannot make those slots for the draw: they hold what the title's writes
/// made.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_SLOTS: [AtomicU64; 3] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// How far past its address a root shader-resource view is read. Root
/// views have no size: Grand Theft Auto VI's skinning reads up to ~35.8 MiB
/// past one (an offset of up to 32 MiB in its first word, then up to 65,535
/// records of up to 60 bytes).
const ROOT_SRV_REACH: u64 = 36 << 20;

/// How far a root constant buffer view is read: 4,096 rows of 16 bytes.
const ROOT_CBV_REACH: u64 = 1 << 16;

fn root_reach(srv: bool) -> u64 {
    if srv { ROOT_SRV_REACH } else { ROOT_CBV_REACH }
}

/// The host address of a root view: its host copy
/// ([`crate::shadow::root`]), brought up to date as far as the view may be
/// read, or for a constant buffer not 256-byte aligned there (the console
/// needs 16) a copy in upload memory. What shaders write through a root UAV
/// goes back to title memory after the submission.
fn root_view(list: &HostList, address: u64, srv: bool, uav: bool) -> Option<u64> {
    let reach = root_reach(srv || uav);
    let (shadow, len) = crate::shadow::root(address, reach)?;
    #[cfg(feature = "diagnostics")]
    crate::shadow::note_root(address, reach, &shadow, len);
    let gpu = shadow.gpu(address);
    #[cfg(feature = "diagnostics")]
    shadow.note_reached(address, len, false, !srv && !uav && gpu % 256 != 0);
    if srv || uav || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        if uav {
            shadow.gpu_writes(list, address, len);
        }
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        (self.host_pipeline, self.host_roots, self.host_heaps) = (0, [0; 2], [usize::MAX; 2]);
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        // A target without a host texture keeps its slot as a null view, so
        // the targets after it stay in theirs.
        let null = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT(28),
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                rtvs.push(host.rtv(None, Some(&null)));
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                let desc = (!record.resource.is_null()).then(|| record.desc());
                rtvs.push(host.rtv(None, Some(&desc.flatten().unwrap_or(null))));
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value, a root
    /// view's address (title memory).
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::View(address)), RootType::Descriptor) => (address >> (32 * dword)) as u32,
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The host addresses of the list's title-memory page table and page
    /// bitmap for the bound view heap, made on first use and again when
    /// one of `roots` (root view addresses) is in no copy it names.
    fn memory_map(&mut self, list: &HostList, roots: &[u64]) -> Option<(u64, u64)> {
        let heap = match self.heaps[0] {
            0 => MEMORY_HEAP,
            h => h,
        };
        if let Some((h, map)) = &self.memory
            && *h == heap
            && roots.iter().all(|&a| map.holds(a))
        {
            return Some((map.table, map.dirty));
        }
        let host = if heap == MEMORY_HEAP {
            crate::memory::heap()?
        } else {
            crate::descriptor::heap(heap as This).host()?
        };
        let mut slots = Vec::new();
        let before = self.memory.as_ref().map(|m| &m.1);
        let map = crate::memory::build(list, host, &mut slots, before);
        if let [first, n] = slots[..] {
            self.dynamic.push((heap, first, n));
        }
        let map = map?;
        let out = (map.table, map.dirty);
        self.memory = Some((heap, map));
        Some(out)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Whether the host pipeline draws rectangle lists itself (its geometry
    /// shader adds each rectangle's fourth corner).
    fn draws_rectangles(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangles)
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
            crate::resource::alias::gpu_wrote(list, t as This);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_access(t as This, true);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
            crate::resource::alias::gpu_wrote(list, self.depth as This);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_access(self.depth as This, true);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        crate::descriptor::host::BINDING.set(self.pipeline);
        if !REBINDING.get() {
            TWINS.take();
            FILLS.take();
        }
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them. Title memory's views need a view heap: ours where
        // the title binds none (the BVH builder's dispatches may not).
        let memory_heap = self.heaps[0] == 0 && p.stages.iter().any(|s| s.memory.is_some());
        self.bind_heaps(list, memory_heap);
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, compute);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_why("no dynamic slots for its tables".to_owned());
                return false;
            }
            let mut values = vec![0u32; stage.user_data.map_or(0, |u| u.1) as usize];
            for u in &t.host.user_data {
                let fresh = made
                    .iter()
                    .find(|m| m.0 == u.slot && u.root == RootType::Table);
                values[u.index as usize] = match fresh {
                    Some(&(_, offset)) => offset,
                    None => self.user_value(compute, u.slot, u.root, u.dword),
                };
            }
            // Root views a shader reads as title memory: the copies holding
            // their addresses go into the page table.
            let mut roots = Vec::new();
            for u in &t.host.user_data {
                if u.root == RootType::Descriptor
                    && u.dword == 0
                    && let Some(Root::View(address)) =
                        self.roots_ref(compute).get(usize::from(u.slot))
                    && let Some(shadow) = crate::shadow::find(*address, 1)
                {
                    let uav = t.used.iter().any(|r| {
                        r.slot == u.slot
                            && r.root == RootType::Descriptor
                            && r.kind == ResourceType::Uav
                    });
                    roots.push((*address, uav.then_some(shadow)));
                }
            }
            if let Some((param, dwords)) = stage.user_data {
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            self.heap_slots(list, &values, &t.host);
            if let Some(param) = stage.view_strides {
                // The record strides of the view heap the shader indexes.
                let records = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.records());
                let Some((buffer, view)) = records else {
                    once!("a shader reads view strides without a view heap that has them: dropped");
                    return false;
                };
                list.keep(buffer);
                // SAFETY: an open list; the view is in the heap the list
                // sets, and the buffer lives while the list does.
                unsafe {
                    if compute {
                        l.SetComputeRootDescriptorTable(param, view);
                    } else {
                        l.SetGraphicsRootDescriptorTable(param, view);
                    }
                }
            }
            for (views, kind) in [(&stage.root_cbvs, 0), (&stage.root_srvs, 1), (&stage.root_uavs, 2)] {
                for &(slot, param) in views {
                    let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some(gpu) = root_view(list, *address, kind == 1, kind == 2) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, 0) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, 1) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (false, _) => l.SetGraphicsRootUnorderedAccessView(param, gpu),
                            (true, 0) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, 1) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, _) => l.SetComputeRootUnorderedAccessView(param, gpu),
                        }
                    }
                }
            }
            if let Some(param) = stage.view_addresses {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.view_addresses());
                let Some(gpu) = gpu else {
                    once!("a shader reading a view's address dropped: no view heap");
                    return false;
                };
                // SAFETY: an open list; the heap's buffer lives with the heap.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
            if let Some(param) = stage.memory {
                let addresses: Vec<u64> = roots.iter().map(|r| r.0).collect();
                let Some((table, dirty)) = self.memory_map(list, &addresses) else {
                    once!("a shader reading title memory dropped: no view heap or dynamic slots");
                    return false;
                };
                crate::memory::note_use(&p.name);
                // What shaders write through a root UAV goes back by its
                // bytes, as through root views ([`root_view`]).
                for (address, shadow) in &roots {
                    if let (Some(shadow), Some((_, map))) = (shadow, &self.memory) {
                        let len = shadow.left(*address).min(1 << 20);
                        shadow.refresh(list, *address, len);
                        shadow.gpu_writes(list, *address, len);
                        map.precise(*address, len);
                    }
                }
                // SAFETY: an open list; list memory alive until it is reset.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, table);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, table);
                    }
                    if let Some(param) = stage.memory_dirty {
                        if compute {
                            l.SetComputeRootUnorderedAccessView(param, dirty);
                        } else {
                            l.SetGraphicsRootUnorderedAccessView(param, dirty);
                        }
                    }
                }
            }
        }
        if twins_in(list) {
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            REBINDING.set(true);
            let bound = self.bind(list, p, compute);
            REBINDING.set(false);
            return bound;
        }
        true
    }

    /// [`twins_back`], forgetting the bound compute root signature and
    /// pipeline when it replaced them.
    fn twins_back(&mut self, list: &HostList) {
        if twins_back(list) {
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
        }
    }

    /// Gives the host list the title's heaps if it has others.
    fn bind_heaps(&mut self, list: &HostList, memory_heap: bool) {
        let want = if memory_heap {
            [MEMORY_HEAP, self.heaps[1]]
        } else {
            self.heaps
        };
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if memory_heap && let Some(h) = crate::memory::heap() {
                heaps.insert(0, Some(h.heap.clone()));
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
    }

    /// `ClearUnorderedAccessViewUint` or `...Float` (`float`): the view the
    /// title made at `cpu` (its CPU handle; the GPU one names the same view
    /// in the bound heap) cleared to `values` on the host, through a
    /// dynamic slot of the bound view heap that the list holds until reset.
    pub(crate) fn clear_unordered(
        &mut self,
        list: &HostList,
        cpu: u64,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
    ) {
        let Some(view) = crate::descriptor::view(cpu) else {
            once!("an unordered access clear of a slot without a view");
            return;
        };
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let (Some(heap), Some(host)) = (heap, crate::host::host()) else {
            once!("an unordered access clear without a bound view heap");
            return;
        };
        let Some(visible) = heap.host() else { return };
        let Some(slot) = visible.dynamic(1) else {
            return;
        };
        self.dynamic
            .push((std::ptr::from_ref(heap) as usize, slot, 1));
        visible.make_at(slot, &view);
        let at = host.cpu_view();
        let Some(resource) = crate::descriptor::host::make_view(&view, at) else {
            once!("an unordered access clear of a view the host cannot make");
            return;
        };
        readable(list, ResourceType::Uav, &view);
        self.bind_heaps(list, false);
        list.keep(&resource);
        let gpu = visible.gpu(slot);
        // SAFETY: an open list; both slots view the live resource.
        unsafe {
            if float {
                let values = values.map(f32::from_bits);
                list.list
                    .ClearUnorderedAccessViewFloat(gpu, at, &resource, &values, rects);
            } else {
                list.list
                    .ClearUnorderedAccessViewUint(gpu, at, &resource, &values, rects);
            }
        }
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // Its groups may write arguments later indirect executions read.
        crate::indirect::close(list);
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        self.twins_back(list);
        if self.ambient(list, p) {
            self.host_heaps = [0; 2];
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// After a dispatch of `p`: the least indirect light (and the probes'
    /// light in reflections), if it is the program that resolves that.
    /// Whether that recorded a pass of its own, replacing the list's view
    /// heaps, compute root signature and pipeline.
    fn ambient(&self, list: &HostList, p: &HostPipeline) -> bool {
        let Some(stage) = p.stages.first().filter(|s| s.shader.resolves_light) else {
            return false;
        };
        let view = |kind: ResourceType, (slot, offset): (u8, u32)| {
            let u = stage.translated().used.iter().find(|u| {
                u.kind == kind
                    && u.root == RootType::Table
                    && (u.slot, u.table_offset_dwords) == (slot, offset)
            })?;
            let Some(Root::Table(table)) = self.roots_ref(true).get(usize::from(u.slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4)
        };
        let light = match view(ResourceType::Uav, crate::ambient::LIGHT) {
            Some(View::Unordered(r, ..)) => r,
            _ => 0,
        };
        let mut recorded = if light == 0 {
            once!("no ambient light: other tables");
            false
        } else {
            crate::ambient::raise(list, light as This)
        };
        if crate::ambient::reflected().is_some() {
            for u in &stage.translated().used {
                let at = (u.slot, u.table_offset_dwords);
                if u.kind == ResourceType::Uav
                    && at != crate::ambient::LIGHT
                    && let Some(View::Unordered(r, _, desc)) = view(u.kind, at)
                    && r != 0
                {
                    recorded |= crate::ambient::reflect(list, r as This, desc.map(|d| d[0]));
                }
            }
        }
        recorded
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            let views = stage.root_cbvs.iter().map(|v| (v, false));
            for (&(slot, _), srv) in views.chain(stage.root_srvs.iter().map(|v| (v, true))) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::root(*address, root_reach(srv)).map(|(s, held)| {
                        // Host address, bytes of the view's reach the copy
                        // holds, the copy's title start and its first word
                        // there.
                        (
                            s.gpu(*address),
                            held,
                            *address - s.offset(*address),
                            s.peek(*address),
                        )
                    });
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.twins_back(list);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    /// The view-heap slots a stage reaches through its user data that no
    /// used-resource entry names (a descriptor index in a root constant,
    /// say; see `victory_shader::emit::HeapIndex`), made for the draw with
    /// their data as table entries are. A stage that also loads slot
    /// indices from memory may read any slot: the buffer views the title
    /// wrote since such a stage last ran get their data for it. `values`
    /// is the stage's user data.
    ///
    /// shortcut: those slots' data is copied once, for the first such stage
    /// after the title wrote them (see [`crate::shadow`]).
    fn heap_slots(&self, list: &HostList, values: &[u32], shader: &HostShader) {
        #[cfg(feature = "diagnostics")]
        if shader.heap_unknown > 0 {
            HEAP_SLOTS[1].fetch_add(1, Ordering::Relaxed);
            HEAP_SLOTS[2].fetch_add(u64::from(shader.heap_unknown), Ordering::Relaxed);
        }
        if (shader.heap_indices.is_empty() && shader.heap_unknown == 0) || self.heaps[0] == 0 {
            return;
        }
        let heap = crate::descriptor::heap(self.heaps[0] as This);
        let Some(host) = heap.host() else {
            return;
        };
        if shader.heap_unknown > 0 {
            for slot in host.take_unread() {
                let view = heap.handle(slot).and_then(crate::descriptor::view);
                let Some(view) = view.filter(View::is_buffer) else {
                    continue;
                };
                let kind = match view {
                    View::Unordered(..) | View::Placed(true, ..) => ResourceType::Uav,
                    _ => ResourceType::Srv,
                };
                readable(list, kind, &view);
                #[cfg(feature = "diagnostics")]
                if let Some(View::Shader(r, _) | View::Unordered(r, ..)) =
                    crate::descriptor::host::placed_texture(&view)
                    && let Some(texture) = crate::resource::host_resource(r as This)
                {
                    crate::descriptor::host::note_bindless(texture, false);
                }
            }
        }
        for h in &shader.heap_indices {
            // Dynamic slots (of tables made afresh) are made with the table.
            let slot = h.slot(values);
            let Some(handle) = heap.handle(slot).filter(|_| slot < host.count()) else {
                continue;
            };
            let Some(view) = host.mirror(slot, handle, true) else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            HEAP_SLOTS[0].fetch_add(1, Ordering::Relaxed);
            let kind = if h.write {
                ResourceType::Uav
            } else {
                ResourceType::Srv
            };
            readable(list, kind, &view);
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            #[cfg(feature = "diagnostics")]
            note_data(&view, u);
            readable(list, u.kind, &view);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], compute: bool) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                #[cfg(feature = "diagnostics")]
                for u in &entries {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        note_data(&view, u);
                    }
                }
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    // An empty one is made null below, as other views.
                    Some(View::Constant(address, size)) if size > 0 => {
                        #[cfg(feature = "diagnostics")]
                        if let Some(u) = used {
                            note_data(&View::Constant(address, size), u);
                        }
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        #[cfg(feature = "diagnostics")]
                        crate::shadow::lux_table_constant(
                            list,
                            used.map(|u| u.lower_bound),
                            address,
                            size,
                            to,
                        );
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            #[cfg(feature = "diagnostics")]
                            note_data(&view, u);
                            readable(list, u.kind, &view);
                        }
                    }
                    None => {
                        host.make_null(index);
                        #[cfg(feature = "diagnostics")]
                        if let Some(u) = used {
                            crate::capture::note_slot(None, u);
                        }
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        // Without the pipeline's geometry shader, rectangles are expanded
        // from their vertex inputs on the CPU.
        let rects = self.topology == RECTANGLES && !self.draws_rectangles();
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            self.twins_back(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        let rects = self.topology == RECTANGLES && !self.draws_rectangles();
        if rects {
            once!("indexed rectangles not drawn");
        }
        let ran = !rects && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            self.twins_back(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        #[cfg(feature = "diagnostics")]
        if kind != DISPATCH {
            let mut raw = [0u8; 4];
            let title = crate::host::read_into(arguments as *const u8, &mut raw).then(|| u32::from_le_bytes(raw));
            let host = args.peek(arguments);
            self.trace(&format!("indirect{kind} title {title:?} host {host:?}"), max, stride);
            if kind == MESH {
                let bucket = |groups: Option<u32>| match groups {
                    None => "unread",
                    Some(0) => "0",
                    Some(1..=64) => "1-64",
                    Some(65..=1024) => "65-1024",
                    Some(1025..=65535) => "1025-65535",
                    Some(_) => "over 65535",
                };
                crate::pipeline::note_why(format!(
                    "(not a drop) indirect mesh groups before recording: title {}, host copy {}",
                    bucket(title),
                    bucket(host)
                ));
            }
        }
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            dropped(self.pipeline);
            return false;
        }
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        let records = args.gpu(arguments);
        let count = counter.as_ref().map(|(s, at)| s.gpu(*at));
        // Clamped first: the clamp binds a compute pipeline of its own. An
        // indirect dispatch's segment holds it alone, so it is clamped now
        // unless it joins an open segment.
        let alone = if kind == DISPATCH
            && !crate::indirect::joins(list, (kind, stride, max), records, count)
        {
            let Some(clamped) =
                crate::indirect::clamp_alone(list, (kind, stride, max), records, count)
            else {
                why("not bounded (no clamp, or a stride, records or count not 4-byte aligned)");
                return false;
            };
            Some(clamped)
        } else {
            if !crate::indirect::open(list, (kind, stride, max), records, count) {
                why("not bounded (no clamp, or a stride not a multiple of 4)");
                return false;
            }
            None
        };
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || self.draws_rectangles())
                    && self.prepare(list, kind == INDEXED)
            }
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                Some(p) if self.topology == RECTANGLES && !p.rectangles => {
                    why("rectangles without the rectangle geometry shader")
                }
                Some(p) => why(&format!("prepare failed (compute {}, mesh {})", p.compute, p.mesh)),
            }
            return false;
        }
        let Some((clamped, at, count_at)) =
            alone.or_else(|| crate::indirect::add(list, (kind, stride, max), records, count))
        else {
            why("not bounded (arguments or count not 4-byte aligned)");
            return false;
        };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // SAFETY: an open list with everything set; the clamped records'
        // buffer is the list's own, the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &clamped,
                at,
                count_at.map(|_| &clamped),
                count_at.unwrap_or(0),
            )
        };
        self.twins_back(list);
        if kind == DISPATCH {
            // Its groups may write arguments later executions read.
            crate::indirect::close(list);
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads (`kind` SRV) or writes (UAV) through a view
/// ready: buffer data copied, textures moved to the state they need.
fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer and constant views read title memory: their range is copied
    // at submission; what shaders write through unordered access goes back
    // to it after.
    let range = match *view {
        View::Constant(address, size) => Some((address, u64::from(size))),
        _ => crate::descriptor::host::buffer_range(view),
    };
    if let Some((address, len)) = range
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        // A view the host cannot make on the copy reads its own.
        let own = crate::descriptor::host::own_copy_of(view, &shadow, (address, len));
        #[cfg(feature = "diagnostics")]
        if !matches!(*view, View::Constant(..)) {
            crate::shadow::probe_lux(
                kind == ResourceType::Uav,
                address,
                len,
                &shadow,
                own.is_some(),
            );
        }
        #[cfg(feature = "diagnostics")]
        if kind != ResourceType::Uav {
            shadow.note_reached(address, len, false, own.is_some());
            crate::shadow::lux_read(
                list,
                matches!(*view, View::Constant(..)),
                address,
                len,
                &shadow,
                own.map(|(_, n)| n),
            );
        }
        if let Some((data, len)) = own {
            list.defer_fill(data, address as *const u8, len as usize);
        }
        if kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            crate::resource::alias::before_read(list, *resource as This);
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_access(*resource as This, false);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, words), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_depth_unordered(view);
            let mut resource = *resource as This;
            crate::resource::alias::gpu_wrote(list, resource);
            // A depth texture's twin starts as the texture is.
            let plane = crate::planes::plane(words.as_ref());
            if let Some(twin) = crate::resource::unordered_twin(resource, plane)
                && let (Some(to), Some(from)) = (
                    crate::resource::host_resource(twin),
                    crate::resource::host_resource(resource),
                )
            {
                let twins = (resource, twin, plane);
                if !TWINS.with_borrow(|t| t.contains(&twins)) {
                    if crate::planes::split(resource) {
                        FILLS.with_borrow_mut(|f| f.push(twins));
                    } else {
                        const COPY_DEST: u32 = 0x400;
                        const COPY_SOURCE: u32 = 0x800;
                        crate::list::ensure_state(list, resource, COPY_SOURCE);
                        crate::list::ensure_state(list, twin, COPY_DEST);
                        // SAFETY: an open list; live textures of one description.
                        unsafe { list.list.CopyResource(to, from) };
                    }
                    TWINS.with_borrow_mut(|t| t.push(twins));
                }
                resource = twin;
            }
            #[cfg(feature = "diagnostics")]
            {
                crate::capture::note_target(resource);
                crate::capture::note_access(resource, true);
            }
            crate::list::ensure_state(list, resource, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// Logs the data of a constant or buffer view a table entry `u` holds, for
/// the programs `VICTORY_DUMP_DATA` names (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn note_data(view: &View, u: &UsedResource) {
    crate::capture::note_slot(Some(view), u);
    let range = match *view {
        View::Constant(address, size) => Some((address, u64::from(size))),
        _ => crate::descriptor::host::buffer_range(view),
    };
    if let Some((address, len)) = range {
        crate::capture::note_data(u.slot, u.table_offset_dwords * 4, address, len);
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::tests::{gpu, release, title_memory, word};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    const MIB: usize = 1 << 20;

    /// Reads eight words at offsets past a root shader-resource view.
    const SOURCE: &str = r#"
#define RS "SRV(t0), UAV(u0), RootConstants(num32BitConstants=8, b0)"
cbuffer Offsets : register(b0) { uint4 a; uint4 b; };
ByteAddressBuffer root : register(t0);
RWByteAddressBuffer read : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    read.Store4(0, uint4(root.Load(a.x), root.Load(a.y), root.Load(a.z), root.Load(a.w)));
    read.Store4(16, uint4(root.Load(b.x), root.Load(b.y), root.Load(b.z), root.Load(b.w)));
}
"#;

    /// The root signature and pipeline of a compute shader in HLSL (shader
    /// model 5.1) with its root signature inside.
    fn compute(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let host = crate::host::host().unwrap();
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        let compiled = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            assert!(compiled.is_ok(), "{}", String::from_utf8_lossy(text));
        }
        compiled.unwrap();
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a shader with its root signature inside.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
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

    /// The words at `offsets` past a root view at title `address`, as a
    /// dispatch reads them through the host address [`root_view`] gives.
    /// Fails without dispatching where the root view's host copy ends
    /// within 36 MiB of it: reading past a copy's end faults the hardware
    /// adapter (`VICTORY_TEST_ROOT_PAST_COPY=1` dispatches anyway).
    fn read_root(address: u64, offsets: [u32; 8]) -> [u32; 8] {
        let host = crate::host::host().unwrap();
        let (root, pso) = compute(SOURCE);
        let list = host.new_list().unwrap();
        let at = root_view(&list, address, true, false).unwrap();
        let held = crate::shadow::host_end(at).map_or(0, |end| end - at);
        assert!(
            held >= ROOT_SRV_REACH || std::env::var_os("VICTORY_TEST_ROOT_PAST_COPY").is_some(),
            "the root view's host copy holds {held:#x} bytes past it, short of 36 MiB"
        );
        let (out, data) = host.shared_buffer(256).unwrap();
        // SAFETY: an open list; live buffers, pipeline and root signature.
        unsafe {
            let l = &list.list;
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootShaderResourceView(0, at);
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstants(2, 8, offsets.as_ptr().cast(), 0);
            l.Dispatch(1, 1, 1);
            l.Close().unwrap();
        }
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: the mapped words the dispatch wrote, waited for.
        std::array::from_fn(|i| unsafe { data.cast::<u32>().add(i).read_volatile() })
    }

    /// A root view in loose memory reads the title's bytes 8 to 36 MiB past
    /// it (Grand Theft Auto VI's skinning, run 40's page fault), brought
    /// up to date at each submission.
    #[test]
    fn a_root_view_reads_title_memory_36_mib_past_it() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let p = title_memory(600, 512 * MIB, 48 * MIB, 48 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        // At an odd offset in its 4 MiB window, as run 40's root3.
        let root = p + MIB + 0x1_4e60;
        let offsets = [
            0,
            12,
            96,
            8 << 20,
            (16 << 20) + 4,
            (24 << 20) + 8,
            (32 << 20) + 0x1fc,
            (36 << 20) - 16,
        ];
        let mut want = offsets.map(|o| word(root + o as usize));
        assert_eq!(read_root(root as u64, offsets), want);
        let at = root + (24 << 20) + 8;
        // SAFETY: committed test memory.
        unsafe { (at as *mut u32).write(0x600d) };
        want[5] = 0x600d;
        assert_eq!(read_root(root as u64, offsets), want);
        release(p);
    }

    /// Past the title's committed memory a root view reads zeros.
    #[test]
    fn a_root_view_reads_zeros_past_the_title_memory() {
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let p = title_memory(700, 512 * MIB, 64 * MIB, 16 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        let root = p + 8 * MIB + 0x40;
        let offsets = [
            0,
            4 << 20,
            (8 << 20) - 0x44,
            (8 << 20) - 0x40,
            20 << 20,
            30 << 20,
            35 << 20,
            (36 << 20) - 4,
        ];
        let want = offsets.map(|o| {
            let at = root + o as usize;
            if at < p + 16 * MIB { word(at) } else { 0 }
        });
        assert_eq!(read_root(root as u64, offsets), want);
        release(p);
    }

    /// What Grand Theft Auto VI's meshlet culling (CS_BatchCullMeshlets)
    /// does first, as translated: load a heap slot's index from a batch
    /// record (dword 0, low 24 bits), read the kind of view the host made
    /// there from the view heap's records, and read the batch's words
    /// through a view of that kind (none: zeros). Its instance culling
    /// reads instance records the same way.
    const CULL: &str = r#"
#define RS "DescriptorTable(SRV(t0, space=2)), SRV(t1), UAV(u0), DescriptorTable(SRV(t0, space=1, numDescriptors=64)), DescriptorTable(CBV(b0, space=1, numDescriptors=64))"
struct Words { uint4 w; };
ByteAddressBuffer records : register(t0, space2);
ByteAddressBuffer batches : register(t1);
RWByteAddressBuffer read : register(u0);
ByteAddressBuffer views[64] : register(t0, space1);
ConstantBuffer<Words> constants[64] : register(b0, space1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    for (uint b = 0; b < 2; b++) {
        uint slot = batches.Load(b * 16) & 0xffffff;
        uint3 record = records.Load3(slot * 16);
        uint word = 0;
        if (record.z == 2) {
            word = views[slot].Load(record.y);
        } else if (record.z == 1) {
            word = constants[slot].w.x;
        }
        read.Store2(b * 8, uint2(word, record.z));
    }
}
"#;

    /// A dispatch that loads its heap slots' indices from memory reads the
    /// title's words through the buffer views the title wrote there, as
    /// on the console: no draw names those slots, so the title's writes
    /// make them, and the first stage that may read slots it does not name
    /// brings their data up to date.
    #[test]
    fn a_dispatch_reads_buffer_views_through_slots_it_loads() {
        use crate::descriptor::tests::{release_heap, view_heap};
        use crate::descriptor::{create_cbv, create_placed_srv};
        use victory_shader::emit::view_kind;
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let host = crate::host::host().unwrap();
        // A title buffer under the in-place size: the host copies it.
        let p = title_memory(800, 512 * MIB, 4 * MIB, 4 * MIB);
        assert_ne!(p, 0, "no free address for test memory");
        let (start, len) = (p as u64, 256u64 << 10);
        crate::shadow::register(p, start, len);
        let h = view_heap(64);
        let heap = crate::descriptor::heap(h);
        let host_heap = heap.host().expect("a host view heap");
        let at = |index: u32| heap.handle(index).expect("a slot of the heap");
        // A raw view of the buffer from byte 256 at slot 37, a constant
        // view of 256 bytes at 0x1000 at slot 9.
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, len as u32);
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&[39, 1, 0x1688, 0, 64, 0, 1024, 0, 1, 0]);
        (desc[10], desc[11]) = (start as u32, (start >> 32) as u32);
        let null = std::ptr::null();
        create_placed_srv(null, &resource, &desc, at(37));
        let constant = start + 0x1000;
        create_cbv(
            null,
            &[constant as u32, (constant >> 32) as u32, 256, 0],
            at(9),
        );
        // Batch records naming them in dword 0's low 24 bits.
        let (batches, records) = host.shared_buffer(256).unwrap();
        // SAFETY: the mapped buffer's first dwords.
        unsafe {
            let words = records.cast::<u32>();
            words.write_volatile(0x5a00_0000 | 37);
            words.add(4).write_volatile(0x0300_0000 | 9);
        }
        let shader = HostShader {
            stage: victory_shader::HwStage::Cs,
            dxil: Vec::new(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            view_strides: true,
            heap_indices: Vec::new(),
            heap_unknown: 2,
            root_uavs: Vec::new(),
            memory: false,
            memory_writes: false,
            view_addresses: false,
        };
        let (root, pso) = compute(CULL);
        let (out, data) = host.shared_buffer(256).unwrap();
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[h]);
        state.heap_slots(&list, &[], &shader);
        let (_, records) = host_heap.records().expect("the view heap's records");
        // SAFETY: an open list; live heap, buffers, pipeline and root
        // signature.
        unsafe {
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootDescriptorTable(0, records);
            l.SetComputeRootShaderResourceView(1, batches.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            let slots = host_heap.heap.GetGPUDescriptorHandleForHeapStart();
            l.SetComputeRootDescriptorTable(3, slots);
            l.SetComputeRootDescriptorTable(4, slots);
            l.Dispatch(1, 1, 1);
            l.Close().unwrap();
        }
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: the mapped words the dispatch wrote, waited for.
        let got: [u32; 4] =
            std::array::from_fn(|i| unsafe { data.cast::<u32>().add(i).read_volatile() });
        let want = [
            word(p + 256),
            view_kind::RAW_SRV,
            word(p + 0x1000),
            view_kind::CBV,
        ];
        drop(list);
        release_heap(h);
        crate::shadow::unregister(p, start);
        release(p);
        assert_eq!(got, want, "words and kinds read through slots 37 and 9");
    }

    /// A constant view of title memory off 256 bytes, or a typed buffer
    /// view off an element of the memory's host copy, reads its own copy,
    /// filled from title memory when its list is submitted.
    #[test]
    fn views_off_their_host_copy_read_their_own() {
        let _serial = crate::watch::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let memory = Box::leak(vec![0u8; 0x2_0000].into_boxed_slice());
        let base = (memory.as_ptr() as u64).next_multiple_of(0x1_0000);
        let title = crate::resource::buffer_at(base, 0x1000);
        let address = base + 0x80;
        let shadow = crate::shadow::find(address, 0x80).expect("a host copy");
        assert_ne!(shadow.gpu(address) % 256, 0, "off 256 bytes on the host");
        for i in 0..0x80 {
            // SAFETY: inside the leaked memory.
            unsafe { ((address + i) as *mut u8).write(i as u8 ^ 0xa5) };
        }
        let list = host.new_list().unwrap();
        readable(&list, ResourceType::Cbv, &View::Constant(address, 0x80));
        // SAFETY: an open list.
        unsafe { list.list.Close() }.unwrap();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        let view = View::Constant(address, 0x80);
        let (data, len) = crate::descriptor::host::own_copy_of(&view, &shadow, (address, 0x80))
            .expect("its own copy");
        assert_eq!(len, 0x80);
        // SAFETY: the copy's mapped bytes, filled at submission.
        let copy = unsafe { std::slice::from_raw_parts(data, 0x80) };
        assert!(copy.iter().enumerate().all(|(i, &b)| b == i as u8 ^ 0xa5));
        // A typed view of four R32G32B32_FLOAT elements 4 bytes into the
        // buffer, off an element of its host copy, reads its own copy too.
        let mut words = [0u32; 16];
        words[..10].copy_from_slice(&[6, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0]);
        (words[10], words[11]) = ((base + 4) as u32, ((base + 4) >> 32) as u32);
        let mut buffer = [0u32; 14];
        (buffer[0], buffer[4]) = (1, 0x100);
        let view = View::Placed(false, buffer, words);
        for i in 0..48 {
            // SAFETY: inside the leaked memory.
            unsafe { ((base + 4 + i) as *mut u8).write(i as u8 ^ 0x3c) };
        }
        let list = host.new_list().unwrap();
        readable(&list, ResourceType::Srv, &view);
        // SAFETY: an open list.
        unsafe { list.list.Close() }.unwrap();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        let range = crate::descriptor::host::buffer_range(&view).expect("its range");
        assert_eq!(range, (base + 4, 48));
        let (data, len) =
            crate::descriptor::host::own_copy_of(&view, &shadow, range).expect("its own copy");
        // SAFETY: the copy's mapped bytes, filled at submission.
        let copy = unsafe { std::slice::from_raw_parts(data, len as usize) };
        assert!(copy.iter().enumerate().all(|(i, &b)| b == i as u8 ^ 0x3c));
        let _ = title;
    }

    /// A depth texture the title also writes as unordered gets those
    /// writes through its twin: the twin starts as the texture is and what
    /// the command writes into it goes back to the texture after.
    #[test]
    fn unordered_writes_to_depth_textures_reach_them() {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let _serial = crate::watch::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let memory = Box::leak(vec![0u8; 0x2_0000].into_boxed_slice());
        let address = (memory.as_ptr() as u64).next_multiple_of(0x1_0000);
        // A 64x64 16-bit depth texture, also unordered.
        let desc = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 53, 1, 0, 0, 0x6, 0];
        let depth = crate::resource::texture_at(address, &desc).expect("a host texture");
        let twin = crate::resource::unordered_twin(depth, 0).expect("a twin");
        let flags = |r: This| {
            let t = crate::resource::host_resource(r).unwrap();
            // SAFETY: a live resource.
            unsafe { t.GetDesc() }.Flags.0 & 0x7
        };
        assert_eq!((flags(depth), flags(twin)), (0x2, 0x4));
        assert_eq!(
            crate::resource::unordered_twin(depth, 0),
            Some(twin),
            "made once"
        );
        // A footprint of the texture's rows, 256 bytes apart.
        let footprint = |buffer: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer; the location never releases it.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT(53),
                        Width: 64,
                        Height: 64,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        let texture = |r: This| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe {
                std::mem::transmute_copy(crate::resource::host_resource(r).unwrap())
            },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let (written, data) = host.shared_buffer(0x4000).unwrap();
        // SAFETY: the buffer's mapped bytes.
        unsafe { data.write_bytes(0x5a, 0x4000) };
        let (read, back) = host.shared_buffer(0x4000).unwrap();
        let list = host.new_list().unwrap();
        let view = View::Unordered(depth as usize, 0, Some([55, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
        readable(&list, ResourceType::Uav, &view);
        // The command's write into the twin.
        crate::list::ensure_state(&list, twin, COPY_DEST);
        // SAFETY: an open list; live resources; whole subresources.
        unsafe {
            list.list
                .CopyTextureRegion(&texture(twin), 0, 0, 0, &footprint(&written), None)
        };
        twins_back(&list);
        crate::list::ensure_state(&list, depth, COPY_SOURCE);
        // SAFETY: as above.
        unsafe {
            list.list
                .CopyTextureRegion(&footprint(&read), 0, 0, 0, &texture(depth), None);
            list.list.Close().unwrap();
        }
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: the mapped bytes the list wrote, waited for.
        let row = unsafe { std::slice::from_raw_parts(back, 0x4000) };
        assert!(
            row.iter().step_by(256).all(|&b| b == 0x5a),
            "the twin's write"
        );
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A compute's writes to a depth-stencil texture's planes through
    /// unordered views reach later depth and stencil tests, and the rest of
    /// the texture stays as it was: the console's D16_UNORM_S8_UINT (D24S8
    /// on the host) written as R16_UNORM and R8_UINT, then tested as Grand
    /// Theft Auto VI's tiled lighting tests it (a material's depth, equal).
    #[test]
    fn unordered_writes_to_depth_stencil_planes_reach_later_tests() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        const DEPTH_WRITE: u32 = 0x10;
        const CS: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<uint> stencil : register(u0);
RWTexture2D<unorm float> depth : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < 32) {
        stencil[id.xy] = 0x25;
        depth[id.xy] = 5.0 / 65535.0;
    }
}
"#;
        const DRAW: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0)"
cbuffer Draw : register(b0) { float z; float color; };
[RootSignature(RS)]
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * float2(2, -2) + float2(-1, 1), z, 1);
}
[RootSignature(RS)]
float4 ps() : SV_Target { return color; }
"#;
        let _serial = crate::watch::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        if !crate::planes::stencil_twins() {
            return; // VICTORY_NO_STENCIL_TWIN=1
        }
        let device = &host.device;
        let memory = Box::leak(vec![0u8; 0x4_0000].into_boxed_slice());
        let address = (memory.as_ptr() as u64).next_multiple_of(0x1_0000);
        // A 64x64 D16_UNORM_S8_UINT texture, also unordered.
        let desc = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 118, 1, 0, 0, 0x6, 0];
        let depth = crate::resource::texture_at(address, &desc).expect("a host texture");
        let texture = crate::resource::host_resource(depth).unwrap();
        let format = |r: This| {
            let t = crate::resource::host_resource(r).unwrap();
            // SAFETY: a live resource.
            unsafe { t.GetDesc() }.Format.0
        };
        let twins =
            [0, 1].map(|plane| crate::resource::unordered_twin(depth, plane).expect("a twin"));
        assert_eq!(twins.map(format), [53, 60]);
        let heap = |kind, flags| -> ID3D12DescriptorHeap {
            // SAFETY: a plain description.
            unsafe {
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: 2,
                    Flags: flags,
                    NodeMask: 0,
                })
            }
            .unwrap()
        };
        let views = heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        );
        let dsvs = heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        );
        let rtvs = heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        );
        // SAFETY: live heaps.
        let (first, step, dsv, rtv) = unsafe {
            (
                views.GetCPUDescriptorHandleForHeapStart(),
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                dsvs.GetCPUDescriptorHandleForHeapStart(),
                rtvs.GetCPUDescriptorHandleForHeapStart(),
            )
        };
        // The views as the title describes them: the stencil plane as
        // R8_UINT, the depth plane as R16_UNORM.
        let unordered = [
            View::Unordered(depth as usize, 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0])),
            View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
        ];
        for (i, view) in unordered.iter().enumerate() {
            let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: first.ptr + i * step as usize,
            };
            assert!(crate::descriptor::host::made(view, at), "view {i}");
        }
        let sample = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };
        // A 64x64 R8_UNORM target for the tests' draws.
        let mut target: Option<ID3D12Resource> = None;
        // SAFETY: plain descriptions; the out pointer ours.
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 64,
                    Height: 64,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT(61),
                    SampleDesc: sample,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
        }
        .unwrap();
        let target = target.unwrap();
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT(45),
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // SAFETY: live resources and heap slots.
        unsafe {
            device.CreateDepthStencilView(texture, Some(std::ptr::from_ref(&dsv_desc)), dsv);
            device.CreateRenderTargetView(&target, None, rtv);
        }
        let (cs_root, cs) = compute(CS);
        let blob = |entry, target| {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call.
            unsafe {
                D3DCompile(
                    DRAW.as_ptr().cast(),
                    DRAW.len(),
                    s!("draw"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    None,
                )
            }
            .unwrap();
            code.unwrap()
        };
        let (vs, ps) = (blob(s!("vs"), s!("vs_5_1")), blob(s!("ps"), s!("ps_5_1")));
        // SAFETY: the blobs' own bytes.
        let bytes = |b: &ID3DBlob| unsafe {
            std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
        };
        let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes(b).as_ptr().cast(),
            BytecodeLength: bytes(b).len(),
        };
        // SAFETY: a shader with its root signature inside.
        let draw_root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, bytes(&vs)) }.unwrap();
        let equal = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_EQUAL,
        };
        let mut pipeline = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer; the description never releases it.
            pRootSignature: unsafe { std::mem::transmute_copy(&draw_root) },
            VS: code(&vs),
            PS: code(&ps),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
                StencilEnable: true.into(),
                StencilReadMask: 0xff,
                StencilWriteMask: 0,
                FrontFace: equal,
                BackFace: equal,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT(45),
            SampleDesc: sample,
            ..Default::default()
        };
        pipeline.RTVFormats[0] = DXGI_FORMAT(61);
        pipeline.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
        // SAFETY: a description of live shaders and root signature.
        let draw: ID3D12PipelineState =
            unsafe { device.CreateGraphicsPipelineState(&pipeline) }.unwrap();
        let (read, back) = host.shared_buffer(64 * 256).unwrap();
        let list = host.new_list().unwrap();
        let l = &list.list;
        crate::list::ensure_state(&list, depth, DEPTH_WRITE);
        let both = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
        // SAFETY: an open list; a live view.
        unsafe { l.ClearDepthStencilView(dsv, both, 1.0, 0x80, None) };
        // The dispatch: its tables, its twins filled, its writes, back.
        for view in &unordered {
            readable(&list, ResourceType::Uav, view);
        }
        assert!(twins_in(&list), "the depth plane's pass");
        // SAFETY: an open list; the heap, root signature and pipeline live.
        unsafe {
            l.SetDescriptorHeaps(&[Some(views.clone())]);
            l.SetComputeRootSignature(&cs_root);
            l.SetPipelineState(&cs);
            l.SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
            l.Dispatch(8, 8, 1);
        }
        assert!(twins_back(&list), "the depth plane's pass");
        // Over the whole target: the material the compute wrote (depth 5 of
        // 65,535, stencil 0x25), then what the clear left (1.0, 0x80).
        crate::list::ensure_state(&list, depth, DEPTH_WRITE);
        let material = 5.0 * f32::from_bits(0x3780_0080);
        let readback = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer; the location never releases it.
            pResource: unsafe { std::mem::transmute_copy(&read) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT(61),
                        Width: 64,
                        Height: 64,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        let drawn = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&target) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: an open list; live views, root signature and pipeline.
        unsafe {
            l.OMSetRenderTargets(
                1,
                Some(std::ptr::from_ref(&rtv)),
                false,
                Some(std::ptr::from_ref(&dsv)),
            );
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 64.0,
                Height: 64.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: 64,
                bottom: 64,
                ..Default::default()
            }]);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            l.SetGraphicsRootSignature(&draw_root);
            l.SetPipelineState(&draw);
            for (z, color, reference) in [(material, 1.0f32, 0x25), (1.0, 0.5, 0x80)] {
                l.SetGraphicsRoot32BitConstants(0, 2, [z, color].as_ptr().cast(), 0);
                l.OMSetStencilRef(reference);
                l.DrawInstanced(3, 1, 0, 0);
            }
            l.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(&readback, 0, 0, 0, &drawn, None);
            l.Close().unwrap();
        }
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: the mapped bytes the list wrote, waited for.
        let rows = unsafe { std::slice::from_raw_parts(back, 64 * 256) };
        for row in rows.chunks(256) {
            assert!(
                row[..32].iter().all(|&b| b == 255),
                "the compute's: {:?}",
                &row[..64]
            );
            assert!(
                row[32..64].iter().all(|&b| b.abs_diff(128) <= 1),
                "the clear's: {:?}",
                &row[..64]
            );
        }
        // SAFETY: a plain device query.
        assert!(unsafe { device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Render targets the host has no texture or no view for keep their
    /// slots as null views, so the targets after them stay in theirs.
    #[test]
    fn missing_render_targets_keep_their_slots() {
        let _serial = crate::watch::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let memory = Box::leak(vec![0u8; 0x2_0000].into_boxed_slice());
        let address = (memory.as_ptr() as u64).next_multiple_of(0x1_0000);
        // A 64x64 R8G8B8A8_UNORM render target the title placed.
        let desc = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 0x1, 0];
        let target = crate::resource::texture_at(address, &desc).expect("a host texture");
        // Render target view records: the tag, whether described, the
        // resource, the description.
        let tag = u64::from(u32::from_le_bytes(*b"vRTV"));
        let records = [[tag, 0, 0, 0, 0], [tag, target as u64, 0, 0, 0]];
        let handles = [0, records[0].as_ptr() as u64, records[1].as_ptr() as u64];
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_targets(&list, &handles, 0);
        assert_eq!(state.targets, [target as usize]);
        // SAFETY: an open list.
        unsafe { list.list.Close() }.unwrap();
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Reads the record of the slot in the root constant, the records'
    /// size, and the records at their end and 1 GiB past it.
    const RECORDS: &str = r#"
#define RS "DescriptorTable(SRV(t0, space=2)), UAV(u0), RootConstants(num32BitConstants=1, b0)"
cbuffer Slot : register(b0) { uint slot; };
ByteAddressBuffer records : register(t0, space2);
RWByteAddressBuffer read : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint bytes;
    records.GetDimensions(bytes);
    read.Store4(0, uint4(records.Load3(slot * 16), bytes));
    read.Store3(16, records.Load3(bytes));
    read.Store3(28, records.Load3(bytes + (1u << 30)));
}
"#;

    /// A shader that reads the view heap's slot records at a heap index
    /// past them (a lane's garbage index, or a title slot past the capped
    /// heap) reads zeros, a slot with no view: through the records' view in
    /// the heap, where a root view read whatever lay past the buffer (GPU
    /// page faults in runs 54, 56 and 58).
    #[test]
    fn slot_records_read_past_their_end_are_zeros() {
        use victory_shader::emit::view_kind;
        let _serial = crate::watch::serial();
        if !gpu() {
            return;
        }
        let host = crate::host::host().unwrap();
        let h = crate::descriptor::tests::view_heap(64);
        let host_heap = crate::descriptor::heap(h).host().expect("a host view heap");
        let (out, data) = host.shared_buffer(256).unwrap();
        let slot = host_heap.dynamic(1).expect("a dynamic slot");
        // SAFETY: a live buffer.
        host_heap.constant_at(slot, unsafe { out.GetGPUVirtualAddress() }, 256);
        let (buffer, records) = host_heap.records().expect("the view heap's records");
        // SAFETY: a live buffer.
        let bytes = unsafe { buffer.GetDesc() }.Width as u32;
        let (root, pso) = compute(RECORDS);
        let list = host.new_list().unwrap();
        // SAFETY: an open list; live heap, buffers, pipeline and root
        // signature.
        unsafe {
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootDescriptorTable(0, records);
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstant(2, slot, 0);
            l.Dispatch(1, 1, 1);
            l.Close().unwrap();
        }
        host.execute(&[&list]);
        crate::pipelined::wait_all();
        // SAFETY: the mapped words the dispatch wrote, waited for.
        let got: [u32; 10] =
            std::array::from_fn(|i| unsafe { data.cast::<u32>().add(i).read_volatile() });
        drop(list);
        host_heap.release(slot, 1);
        crate::descriptor::tests::release_heap(h);
        assert_eq!(
            got,
            [0, 0, view_kind::CBV, bytes, 0, 0, 0, 0, 0, 0],
            "the slot's record, the records' bytes, and zeros past them"
        );
    }
}
