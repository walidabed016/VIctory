//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use victory_shader::emit::{VIEW_RECORD_BYTES, view_kind};

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// Held a little longer once the slot is rewritten (see [`HostHeap::keep`]).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// A record per slot (`VIEW_RECORD_BYTES`) for translated shaders'
    /// accesses through heap descriptors
    /// (`victory_shader::emit::VIEW_STRIDE_SPACE`): the record stride of
    /// the slot's view ([`record_stride`]), the bytes its host view starts
    /// before the first record ([`raw_elements`]) and the kind of view the
    /// host made there (`view_kind`), which accesses no used-resource entry
    /// names go by. An upload buffer and its mapping; none in sampler
    /// heaps, or if the host would not make it. Shaders read it through a
    /// raw view at the heap's last slot ([`HostHeap::records`]).
    strides: Option<(ID3D12Resource, usize)>,
    /// Slots of buffer views made from the title's writes whose data no
    /// stage has asked for since ([`HostHeap::take_unread`]).
    unread: Mutex<std::collections::BTreeSet<u32>>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - DYNAMIC, DYNAMIC)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // Zeroed: no slot holds a view yet (`view_kind::NONE`).
        let strides = if sampler {
            None
        } else {
            host.upload_buffer(u64::from(count + extra) * u64::from(VIEW_RECORD_BYTES))
                .map(|(buffer, data)| (buffer, data as usize))
        };
        // The records' view takes the last dynamic slot.
        let dynamic = extra - u32::from(strides.is_some());
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; dynamic as usize])),
            strides,
            unread: Mutex::default(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        if let Some((buffer, _)) = &h.strides {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (count + extra) * (VIEW_RECORD_BYTES / 4),
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            // SAFETY: the last slot of our heap; the buffer holds a record
            // per slot.
            unsafe {
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), h.cpu(count + dynamic))
            };
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..self.dynamic as usize).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// The slots of buffer views made from the title's writes since the
    /// last call: shaders that load slot indices from memory may read any
    /// of them, and their data has not been copied for a draw.
    pub(crate) fn take_unread(&self) -> Vec<u32> {
        let mut unread = self.unread.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *unread).into_iter().collect()
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Diagnostics: the record (view index and generation) each title slot
    /// was last made from, 0 where none is (see [`HostHeap::mirror`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn made_keys(&self) -> Vec<u64> {
        self.made.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        let record = kept
            .as_ref()
            .map_or([0; 3], |k| [record_stride(view), k.lead, k.kind]);
        self.keep(index, kept.and_then(|k| k.resource), record);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None, [0; 3]);
    }

    /// Slot `index` now views `resource` (or nothing of the host's), in
    /// records of `record[0]` bytes from byte `record[1]` of its host view,
    /// a view of kind `record[2]` (see [`HostHeap::strides`]). The
    /// resource it viewed before is released once the submission running
    /// now has finished ([`crate::host::Host::retire`]).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>, record: [u32; 3]) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            if let Some(old) = std::mem::replace(slot, resource)
                && slot.as_ref() != Some(&old)
                && let Some(host) = host()
            {
                host.retire(old);
            }
            if let Some((_, data)) = self.strides {
                // SAFETY: the slot's record in the mapped buffer; shaders
                // read it only.
                let at = unsafe {
                    std::slice::from_raw_parts(
                        (data as *const AtomicU32).add((VIEW_RECORD_BYTES / 4 * index) as usize),
                        3,
                    )
                };
                for (at, word) in at.iter().zip(record) {
                    at.store(word, Ordering::Relaxed);
                }
            }
        }
    }

    /// The buffer of the records of this heap's slots and the view shaders
    /// read them through, at the heap's last slot: a heap index past the
    /// records reads zeros there, a slot with no view. None in sampler heaps.
    pub(crate) fn records(&self) -> Option<(&ID3D12Resource, D3D12_GPU_DESCRIPTOR_HANDLE)> {
        let (buffer, _) = self.strides.as_ref()?;
        // SAFETY: a live heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let at = u64::from(self.count + self.dynamic) * u64::from(self.step);
        Some((
            buffer,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + at,
            },
        ))
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds; `size` bytes, not 0: an
    /// empty view is made by [`HostHeap::make_at`]).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None, [0, 0, view_kind::CBV]);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_size(size, || format!("constant view of list data {size:#x}")),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made when the title writes them, as textures are: shaders
    /// reach slots through indices they load from memory, which no draw
    /// names (Grand Theft Auto VI's instance and meshlet culling). Views of
    /// memory the title placed a texture in are made only for draws
    /// (`for_draw`), which bring the texture up to date.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.placed_texture();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            if kept.is_some() && !for_draw && view.is_buffer() {
                let mut unread = self.unread.lock().unwrap_or_else(|e| e.into_inner());
                unread.insert(index);
            }
            let record = kept
                .as_ref()
                .map_or([0; 3], |k| [record_stride(&view), k.lead, k.kind]);
            self.keep(index, kept.and_then(|k| k.resource), record);
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length, up to the end of its buffer (the host clamps a view
/// running past it to the buffer's copy, see [`on_copy`]).
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let placed = |resource: &[u32; 14], desc: &[u32; 16]| {
        (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )
    };
    let ((base, size), words): ((u64, u64), &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            ((data as u64, len), desc)
        }
        View::Placed(false, resource, desc) if resource[0] == 1 => {
            (placed(resource, desc), desc[..10].try_into().ok()?)
        }
        View::Unordered(r, _, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            return unordered_range(data as u64, len, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let (base, size) = placed(resource, desc);
            return unordered_range(base, size, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some(in_buffer(
        base,
        size,
        buffer.FirstElement,
        buffer.NumElements,
        element,
    ))
}

/// [`buffer_range`] of an unordered access view of the buffer at `base`
/// (`size` bytes).
fn unordered_range(base: u64, size: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = Some(unordered_element(&desc)).filter(|&e| e > 0)?;
    Some(in_buffer(
        base,
        size,
        buffer.FirstElement,
        buffer.NumElements,
        element,
    ))
}

/// Where `count` elements of `element` bytes from element `first` of the
/// buffer at `base` (`size` bytes) lie in it: their address and length.
fn in_buffer(base: u64, size: u64, first: u64, count: u32, element: u64) -> (u64, u64) {
    let start = base.saturating_add(first.saturating_mul(element));
    let end = start
        .saturating_add(u64::from(count) * element)
        .min(base.saturating_add(size));
    (start, end.saturating_sub(start))
}

/// Bytes per element of a buffer unordered access view.
fn unordered_element(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The record stride of `view` as the console's descriptor of it holds it,
/// which its indexed accesses step by: the element size of a buffer view
/// (a structured view's stride, a typed view's format size, 4 for a raw
/// view); 0 for other views.
pub(crate) fn record_stride(view: &View) -> u32 {
    const BUFFER: u32 = 1;
    let element = match view {
        View::Shader(_, Some(words)) if words[1] == BUFFER => element_size(&srv_desc(words)),
        View::Unordered(_, _, Some(words)) if words[1] == BUFFER => {
            unordered_element(&uav_desc(words))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER => {
            let words = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                unordered_element(&uav_desc(words))
            } else {
                element_size(&srv_desc(words))
            }
        }
        _ => 0,
    };
    u32::try_from(element).unwrap_or(0)
}

/// A structured buffer view's records (`count` of `stride` bytes from
/// record `first`, on a host copy `offset` bytes in) as a raw view's
/// dwords: the first, the count, and the bytes before the first record (the
/// lead). Raw views start on 16 bytes (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`)
/// and records on any dword, so the view starts at the 16 bytes holding the
/// first record and shaders add the lead. None if records do not start on
/// a dword; a partial last dword is left out.
/// Translated shaders address structured buffers' records themselves and
/// read them through raw views (`victory_shader::emit`): a host view of
/// another kind than the shader's declares is undefined, and on NVIDIA a
/// raw access through a structured view reads the wrong bytes.
fn raw_elements(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32, u32)> {
    let start = offset + first * u64::from(stride);
    if !start.is_multiple_of(4) {
        return None;
    }
    let lead = start % u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT);
    let bytes = lead + u64::from(count) * u64::from(stride);
    Some((
        (start - lead) / 4,
        u32::try_from(bytes / 4).ok()?,
        lead as u32,
    ))
}

/// How many of a buffer view's `count` elements of `element` bytes from
/// byte `start` of its host copy lie on the copy: all of them, or the ones
/// before its end (the view is clamped to it); `None` where none do, for a
/// null view of its kind (the title's view is `view`). The host removes
/// the device for a view off its resource (an empty raw view of the world's
/// buffer, 2026-11-24, run 48).
fn on_copy(
    shadow: &crate::shadow::Shadow,
    start: u64,
    count: u32,
    element: u64,
    view: impl FnOnce() -> String,
) -> Option<u32> {
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let left = u32::try_from(width.saturating_sub(start) / element).unwrap_or(u32::MAX);
    let made = count.min(left);
    #[cfg(feature = "diagnostics")]
    if made < count {
        let why = if made == 0 {
            "has no element on its host copy: made null"
        } else {
            "runs past the end of its host copy: clamped to it"
        };
        note_view(why, || {
            format!(
                "{}: {made} of {count} elements of {element} bytes from byte {start:#x} of a {width:#x}-byte copy",
                view()
            )
        });
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = view;
    (made > 0).then_some(made)
}

/// Bytes of the host's largest constant buffer view.
const LARGEST_CONSTANT: u32 = D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16;

/// The host size of a constant view of `size` bytes (the title's is
/// `view`): 256-byte whole, at most [`LARGEST_CONSTANT`], which is all
/// shaders address; the host removes the device for a larger view.
fn constant_size(size: u32, view: impl FnOnce() -> String) -> u32 {
    #[cfg(feature = "diagnostics")]
    if size > LARGEST_CONSTANT {
        note_view(
            "is larger than a host constant view: clamped to 64 KiB",
            view,
        );
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = view;
    size.min(LARGEST_CONSTANT).next_multiple_of(256)
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// The title pipeline whose draw or dispatch makes host slots on this
    /// thread, for [`note_view`] (diagnostics).
    pub(crate) static BINDING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Logs, once per cause, a view of title memory the host does not make as
/// the title describes it: why (`why`), the title's view and the program
/// whose draw or dispatch made it (diagnostics).
#[cfg(feature = "diagnostics")]
fn note_view(why: &'static str, view: impl FnOnce() -> String) {
    static SEEN: Mutex<Vec<&str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&why) {
        return;
    }
    seen.push(why);
    let pipeline = BINDING.get() as crate::object::This;
    let program = crate::pipeline::host_pipeline(pipeline).map_or("none", |p| p.name.as_str());
    tracing::warn!(program, view = %view(), "a view of title memory {why} (logged once)");
}

/// What a slot made by [`make`] holds.
struct Kept {
    /// The host resource it views, if any.
    resource: Option<ID3D12Resource>,
    /// The lead of a raw view of a structured one ([`raw_elements`]).
    lead: u32,
    /// The kind of view (`view_kind`).
    kind: u32,
}

impl Kept {
    fn of(resource: &ID3D12Resource, kind: u32) -> Option<Kept> {
        Some(Kept {
            resource: Some(resource.clone()),
            lead: 0,
            kind,
        })
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let (format, stride, bits) = (desc.Format.0, buffer.StructureByteStride, buffer.Flags.0);
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    let view = || {
        let (at, len) = in_buffer(address, u64::MAX, first, count, element);
        format!(
            "shader resource view (format {format}, stride {stride}, flags {bits:#x}) of title {at:#x}+{len:#x}"
        )
    };
    if count == 0 {
        #[cfg(feature = "diagnostics")]
        note_view("is empty: made null", view);
        return None;
    }
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    if element == 0 || stride == 0 && offset % element != 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    let start = offset.saturating_add(first.saturating_mul(element));
    buffer.NumElements = on_copy(&shadow, start, count, element, view)?;
    let mut lead = 0;
    if stride > 0 {
        let raw = raw_elements(
            offset,
            buffer.FirstElement,
            buffer.NumElements,
            buffer.StructureByteStride,
        );
        let Some(raw) = raw else {
            once!(
                offset,
                element,
                "structured buffer view not on a dword of its host copy"
            );
            return None;
        };
        (buffer.FirstElement, buffer.NumElements, lead) = raw;
        buffer.StructureByteStride = 0;
        buffer.Flags |= D3D12_BUFFER_SRV_FLAG_RAW;
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else {
        buffer.FirstElement += offset / element;
    }
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Kept {
        resource: Some(shadow.buffer.clone()),
        lead,
        kind: if raw {
            view_kind::RAW_SRV
        } else {
            view_kind::TYPED_SRV
        },
    })
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = unordered_element(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let (format, stride, bits) = (desc.Format.0, buffer.StructureByteStride, buffer.Flags.0);
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    let view = || {
        let (at, len) = in_buffer(address, u64::MAX, first, count, element);
        format!(
            "unordered access view (format {format}, stride {stride}, flags {bits:#x}) of title {at:#x}+{len:#x}"
        )
    };
    if count == 0 {
        #[cfg(feature = "diagnostics")]
        note_view("is empty: made null", view);
        return None;
    }
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    let offset = shadow.offset(address);
    if element == 0 || stride == 0 && offset % element != 0 {
        once!(
            offset,
            element,
            "unordered buffer view not on an element of its host copy"
        );
        return None;
    }
    let start = offset.saturating_add(first.saturating_mul(element));
    buffer.NumElements = on_copy(&shadow, start, count, element, view)?;
    buffer.CounterOffsetInBytes = 0;
    let mut lead = 0;
    if stride > 0 {
        let raw = raw_elements(
            offset,
            buffer.FirstElement,
            buffer.NumElements,
            buffer.StructureByteStride,
        );
        let Some(raw) = raw else {
            once!(
                offset,
                element,
                "structured unordered view not on a dword of its host copy"
            );
            return None;
        };
        (buffer.FirstElement, buffer.NumElements, lead) = raw;
        buffer.StructureByteStride = 0;
        buffer.Flags |= D3D12_BUFFER_UAV_FLAG_RAW;
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else {
        buffer.FirstElement += offset / element;
    }
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Kept {
        resource: Some(shadow.buffer.clone()),
        lead,
        kind: if raw {
            view_kind::RAW_UAV
        } else {
            view_kind::TYPED_UAV
        },
    })
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let view = || format!("constant view of title {address:#x}+{size:#x}");
            if size == 0 {
                #[cfg(feature = "diagnostics")]
                note_view("is empty: made null", view);
                return None;
            }
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let size = constant_size(size, view);
            let blocks = on_copy(&shadow, shadow.offset(address), size / 256, 256, view)?;
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: blocks * 256,
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Kept::of(&shadow.buffer, view_kind::CBV)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Kept::of(texture, view_kind::TEXTURE)
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Kept::of(texture, view_kind::TEXTURE_UAV)
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept {
                resource: None,
                lead: 0,
                kind: view_kind::NONE,
            })
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept {
                resource: None,
                lead: 0,
                kind: view_kind::NONE,
            })
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                once!(
                    dimension = resource[0],
                    unordered,
                    "placed texture view without a host texture"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader or unordered access view of a texture the title placed in
/// memory itself, as a view of the resource made for that memory (on first
/// use): what a dispatch writes through one, later draws read through the
/// other.
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    const UNORDERED: u32 = 0x4;
    // The description's flags, after its alignment, size and format words.
    let writable = resource[12] & UNORDERED != 0;
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource);
    #[cfg(feature = "diagnostics")]
    if *unordered || texture.is_none() || resource[12] & 0x7 != 0 {
        let program = crate::pipeline::host_pipeline(BINDING.get() as crate::object::This)
            .map_or("none", |p| p.name.as_str());
        let what = if *unordered {
            "placed unordered view"
        } else {
            "placed view"
        };
        crate::resource::census(what, address, resource, texture.is_some(), program);
    }
    let words = desc[..10].try_into().ok();
    match (*unordered, texture) {
        (false, Some(texture)) => Some(View::Shader(texture as usize, words)),
        (true, Some(texture)) if writable => Some(View::Unordered(texture as usize, 0, words)),
        _ => None,
    }
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view; of a buffer, raw or typed as the host makes the title's view
/// (raw and structured ones raw). Writes to it are dropped, reads give
/// zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Whether the title's buffer view words (its stride at `stride`, its
    // flags at 8) make a raw view: structured or raw.
    let raw = |words: &[u32], stride: usize| words[stride] > 0 || words[8] & 1 != 0;
    // Unordered views: whether a buffer (and then whether raw), and the
    // title's view dimension. Without a description, typed as before.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            (is_buffer(r) || desc.is_some_and(|d| d[1] == 1))
                .then(|| desc.is_some_and(|d| raw(&d, 5))),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => {
            ((resource[0] == 1).then(|| raw(desc, 5)), Some(desc[1]))
        }
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it). Without a description, raw as before.
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(desc.is_none_or(|d| raw(&d, 7)), at);
        }
        View::Placed(false, resource, desc) if resource[0] == 1 => {
            return null_buffer_view(raw(desc, 7), at);
        }
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = match buffer {
        Some(raw) => null_buffer_uav_desc(raw),
        None => null_uav_desc(dimension),
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A null buffer unordered access view of one element (none is empty):
/// raw, or typed (R32_UINT).
fn null_buffer_uav_desc(raw: bool) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: if raw {
            DXGI_FORMAT_R32_TYPELESS
        } else {
            DXGI_FORMAT_R32_UINT
        },
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: if raw {
                    D3D12_BUFFER_UAV_FLAG_RAW
                } else {
                    D3D12_BUFFER_UAV_FLAG_NONE
                },
                ..Default::default()
            },
        },
    }
}

/// A null buffer shader resource view of one element, as
/// [`null_buffer_uav_desc`].
fn null_buffer_srv_desc(raw: bool) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: if raw {
            DXGI_FORMAT_R32_TYPELESS
        } else {
            DXGI_FORMAT_R32_UINT
        },
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: if raw {
                    D3D12_BUFFER_SRV_FLAG_RAW
                } else {
                    D3D12_BUFFER_SRV_FLAG_NONE
                },
                ..Default::default()
            },
        },
    }
}

/// A buffer view that reads zeros, raw or typed ([`null_buffer_srv_desc`]).
fn null_buffer_view(raw: bool, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateShaderResourceView(
            None::<&ID3D12Resource>,
            Some(&null_buffer_srv_desc(raw)),
            at,
        )
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// An unordered access view of a texture the title placed in its memory
    /// is a view of the host texture made for that memory, as its shader
    /// views are, when the texture's description allows unordered access.
    #[test]
    fn placed_texture_uavs_view_the_host_texture() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let memory = Box::leak(vec![0u8; 0x3_0000].into_boxed_slice());
        let address = (memory.as_ptr() as u64).next_multiple_of(0x1_0000);
        // A 64x64 R8G8B8A8_UNORM 2D texture, one mip, flags at word 12.
        let mut texture = [0u32; 14];
        texture[..13].copy_from_slice(&[3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0x4]);
        // A 2D view of it at `address`.
        let mut words = [0u32; 16];
        words[..2].copy_from_slice(&[28, 4]);
        words[10] = address as u32;
        words[11] = (address >> 32) as u32;
        let uav = View::Placed(true, texture, words);
        let Some(View::Unordered(made, 0, Some(desc))) = placed_texture(&uav) else {
            panic!("no unordered view of the placed texture");
        };
        assert_eq!(desc[..2], [28, 4]);
        let srv = placed_texture(&View::Placed(false, texture, words));
        assert!(
            matches!(srv, Some(View::Shader(r, _)) if r == made),
            "one host texture"
        );
        assert!(make(&uav, heap.cpu(0)).is_some());
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        texture[12] = 0;
        assert!(placed_texture(&View::Placed(true, texture, words)).is_none());
        assert!(placed_texture(&View::Placed(false, texture, words)).is_some());
    }

    /// The record stride of a buffer view is its element size as the
    /// title describes it, whatever view the host makes of it; other views
    /// have none.
    #[test]
    fn buffer_views_step_by_their_element_size() {
        // Shader resource views: format, dimension (1 buffer), mapping, then
        // first element, count, structure stride and flags from word 4.
        let srv = |format: u32, dimension: u32, stride: u32, flags: u32| {
            View::Shader(
                0,
                Some([format, dimension, 0x1688, 0, 0, 0, 10, stride, flags, 0]),
            )
        };
        assert_eq!(record_stride(&srv(0, 1, 64, 0)), 64);
        assert_eq!(record_stride(&srv(39, 1, 0, 1)), 4, "raw");
        assert_eq!(record_stride(&srv(2, 1, 0, 0)), 16, "R32G32B32A32_FLOAT");
        assert_eq!(record_stride(&srv(34, 1, 0, 0)), 4, "R16G16_FLOAT");
        assert_eq!(record_stride(&srv(2, 4, 0, 0)), 0, "a 2D texture");
        assert_eq!(record_stride(&View::Shader(0, None)), 0);
        // Unordered access views: the buffer member from word 2, flags at 8.
        let uav = |format: u32, stride: u32, flags: u32| {
            View::Unordered(0, 0, Some([format, 1, 0, 0, 10, stride, 0, 0, flags, 0]))
        };
        assert_eq!(record_stride(&uav(0, 16, 0)), 16);
        assert_eq!(record_stride(&uav(39, 0, 1)), 4, "raw");
        assert_eq!(record_stride(&uav(42, 0, 0)), 4, "R32_UINT");
        // Views of memory: the resource description's dimension (1 buffer).
        let mut words = [0u32; 16];
        words[..10].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 10, 20, 0, 0]);
        let mut buffer = [0u32; 14];
        buffer[0] = 1;
        assert_eq!(record_stride(&View::Placed(false, buffer, words)), 20);
        words[..10].copy_from_slice(&[0, 1, 0, 0, 10, 48, 0, 0, 0, 0]);
        assert_eq!(record_stride(&View::Placed(true, buffer, words)), 48);
        buffer[0] = 3;
        assert_eq!(
            record_stride(&View::Placed(true, buffer, words)),
            0,
            "a texture"
        );
        assert_eq!(record_stride(&View::Constant(0x1000, 256)), 0);
    }

    /// A structured view's records as a raw view's dwords: the view starts
    /// on 16 bytes, the records on a dword after it (the lead); a partial
    /// last dword is left out.
    #[test]
    fn structured_elements_as_raw_dwords() {
        assert_eq!(raw_elements(0, 0, 5, 64), Some((0, 80, 0)));
        assert_eq!(raw_elements(0, 4, 10, 12), Some((12, 30, 0)));
        assert_eq!(raw_elements(0, 2, 10, 12), Some((4, 32, 8)));
        assert_eq!(raw_elements(0, 1, 2, 4), Some((0, 3, 4)));
        // On a host copy partway into its buffer, at a whole record or not.
        assert_eq!(raw_elements(8192, 0, 10, 164), Some((2048, 410, 0)));
        assert_eq!(raw_elements(17_691_280, 3, 2, 32), Some((4_422_844, 16, 0)));
        assert_eq!(raw_elements(4, 1, 1, 8), Some((0, 5, 12)));
        assert_eq!(raw_elements(0, 1, 4, 6), None);
        assert_eq!(raw_elements(0, 2, 3, 6), Some((0, 7, 12)));
    }

    /// Each slot's record stride, lead and kind follow what is made there;
    /// samplers have none.
    #[test]
    fn slots_keep_the_record_stride_of_their_view() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let (_, view) = heap.records().expect("a view heap has record strides");
        assert_ne!(view.ptr, 0);
        let data = heap.strides.as_ref().map_or(0, |s| s.1);
        // SAFETY: dwords of the mapped buffer, a record per slot.
        let read = |i: u32| unsafe { (data as *const [u32; 4]).add(i as usize).read_volatile() };
        let last = heap.count + heap.dynamic - 1;
        heap.keep(2, None, [12, 8, view_kind::RAW_SRV]);
        heap.keep(last, None, [64, 0, view_kind::RAW_UAV]);
        assert_eq!(
            (read(0), read(2), read(last)),
            (
                [0; 4],
                [12, 8, view_kind::RAW_SRV, 0],
                [64, 0, view_kind::RAW_UAV, 0]
            )
        );
        heap.make_null(2);
        assert_eq!(read(2), [0; 4]);
        heap.constant_at(last, 0, 256);
        assert_eq!(read(last), [0, 0, view_kind::CBV, 0]);
        let samplers = HostHeap::new(true, 4).expect("a sampler heap");
        assert!(samplers.records().is_none());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// Bytes of [`Title`] and of its host copy.
    const TITLE_LEN: u64 = 1 << 16;

    /// A title buffer the host copies, at an address its copy's length
    /// divides (so a constant view at its start is 256-byte aligned).
    struct Title {
        _memory: Vec<u8>,
        start: u64,
    }

    impl Title {
        fn new() -> Title {
            let memory = vec![0u8; 2 * TITLE_LEN as usize];
            let start = (memory.as_ptr() as u64).next_multiple_of(TITLE_LEN);
            crate::shadow::register(start as usize, start, TITLE_LEN);
            Title {
                _memory: memory,
                start,
            }
        }

        /// A buffer view of it as the title places one (`words`: the view
        /// description's).
        fn view(&self, unordered: bool, words: [u32; 10]) -> View {
            let mut resource = [0u32; 14];
            (resource[0], resource[4]) = (1, TITLE_LEN as u32);
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&words);
            (desc[10], desc[11]) = (self.start as u32, (self.start >> 32) as u32);
            View::Placed(unordered, resource, desc)
        }

        /// A shader resource view of `count` elements from `first`.
        fn srv(&self, format: u32, stride: u32, flags: u32, first: u32, count: u32) -> View {
            self.view(
                false,
                [format, 1, 0x1688, 0, first, 0, count, stride, flags, 0],
            )
        }

        /// An unordered access view of `count` elements from `first`.
        fn uav(&self, format: u32, stride: u32, flags: u32, first: u32, count: u32) -> View {
            self.view(true, [format, 1, first, 0, count, stride, 0, 0, flags, 0])
        }
    }

    impl Drop for Title {
        fn drop(&mut self) {
            crate::shadow::unregister(self.start as usize, self.start);
        }
    }

    /// Runs `make`, which makes slot `index` of `heap`: the debug layer
    /// reports no error, the device stays alive and the slot's record is
    /// `record` (stride, lead and kind; `[0; 3]` for a null view).
    fn check(heap: &HostHeap, index: u32, what: &str, make: impl FnOnce(), record: [u32; 3]) {
        let host = host().expect("a heap's device");
        let errors = crate::host::errors_of(make).unwrap_or_default();
        assert!(errors.is_empty(), "{what}: {errors:?}");
        // SAFETY: a plain device query.
        let alive = unsafe { host.device.GetDeviceRemovedReason() };
        assert!(alive.is_ok(), "{what}: {alive:?}");
        let data = heap.strides.as_ref().map_or(0, |s| s.1);
        // SAFETY: the slot's record in the mapped buffer.
        let got = unsafe {
            (data as *const [u32; 4])
                .add(index as usize)
                .read_volatile()
        };
        assert_eq!(got[..3], record, "{what}");
    }

    /// Makes each view in a slot of its own, as [`check`].
    fn check_views(cases: &[(&str, View, [u32; 3])]) {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        for (index, (what, view, record)) in (0..).zip(cases) {
            check(&heap, index, what, || heap.make_at(index, view), *record);
        }
    }

    /// Raw views (R32_TYPELESS, flag 1) of the 16384 dwords of a copy: an
    /// empty one and one starting at the copy's end are null and recorded
    /// as nothing; one running past the end is clamped to it.
    #[test]
    fn raw_srvs_stay_on_their_copy() {
        let title = Title::new();
        check_views(&[
            ("empty", title.srv(39, 0, 1, 0, 0), [0; 3]),
            (
                "past the end",
                title.srv(39, 0, 1, 16_000, 1000),
                [4, 0, view_kind::RAW_SRV],
            ),
            ("at the end", title.srv(39, 0, 1, 16_384, 1), [0; 3]),
        ]);
    }

    /// As [`raw_srvs_stay_on_their_copy`], for 8-byte records (made raw).
    #[test]
    fn structured_srvs_stay_on_their_copy() {
        let title = Title::new();
        check_views(&[
            ("empty", title.srv(0, 8, 0, 0, 0), [0; 3]),
            (
                "past the end",
                title.srv(0, 8, 0, 8000, 500),
                [8, 0, view_kind::RAW_SRV],
            ),
            ("at the end", title.srv(0, 8, 0, 8192, 1), [0; 3]),
        ]);
    }

    /// As [`raw_srvs_stay_on_their_copy`], for R32G32B32A32_FLOAT elements.
    #[test]
    fn typed_srvs_stay_on_their_copy() {
        let title = Title::new();
        check_views(&[
            ("empty", title.srv(2, 0, 0, 0, 0), [0; 3]),
            (
                "past the end",
                title.srv(2, 0, 0, 4000, 200),
                [16, 0, view_kind::TYPED_SRV],
            ),
            ("at the end", title.srv(2, 0, 0, 4096, 1), [0; 3]),
        ]);
    }

    /// As [`raw_srvs_stay_on_their_copy`], for unordered access: the empty
    /// raw view is the one that removed the device in the world's dispatch
    /// (2026-11-24, run 48).
    #[test]
    fn raw_uavs_stay_on_their_copy() {
        let title = Title::new();
        check_views(&[
            ("empty", title.uav(39, 0, 1, 0, 0), [0; 3]),
            (
                "past the end",
                title.uav(39, 0, 1, 16_000, 1000),
                [4, 0, view_kind::RAW_UAV],
            ),
            ("at the end", title.uav(39, 0, 1, 16_384, 1), [0; 3]),
        ]);
    }

    /// As [`structured_srvs_stay_on_their_copy`], for unordered access.
    #[test]
    fn structured_uavs_stay_on_their_copy() {
        let title = Title::new();
        check_views(&[
            ("empty", title.uav(0, 8, 0, 0, 0), [0; 3]),
            (
                "past the end",
                title.uav(0, 8, 0, 8000, 500),
                [8, 0, view_kind::RAW_UAV],
            ),
            ("at the end", title.uav(0, 8, 0, 8192, 1), [0; 3]),
        ]);
    }

    /// As [`raw_srvs_stay_on_their_copy`], for R32_UINT unordered access.
    #[test]
    fn typed_uavs_stay_on_their_copy() {
        let title = Title::new();
        check_views(&[
            ("empty", title.uav(42, 0, 0, 0, 0), [0; 3]),
            (
                "past the end",
                title.uav(42, 0, 0, 16_000, 1000),
                [4, 0, view_kind::TYPED_UAV],
            ),
            ("at the end", title.uav(42, 0, 0, 16_384, 1), [0; 3]),
        ]);
    }

    /// Constant views: an empty one is null and recorded as nothing,
    /// whether of title memory or of the list's; ones past or at the end of
    /// a title buffer view the memory after it (a copy of their own).
    #[test]
    fn constant_views_stay_on_their_copy() {
        let title = Title::new();
        let end = title.start + TITLE_LEN;
        check_views(&[
            ("empty", View::Constant(title.start, 0), [0; 3]),
            (
                "past the end",
                View::Constant(end - 256, 512),
                [0, 0, view_kind::CBV],
            ),
            (
                "at the end",
                View::Constant(end, 256),
                [0, 0, view_kind::CBV],
            ),
            (
                "128 KiB",
                View::Constant(title.start, 2 * TITLE_LEN as u32),
                [0, 0, view_kind::CBV],
            ),
        ]);
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let copy = crate::shadow::find(title.start, TITLE_LEN).expect("a copy");
        let gpu = copy.gpu(title.start);
        let list_data = || heap.constant_at(0, gpu, 2 * TITLE_LEN as u32);
        check(
            &heap,
            0,
            "128 KiB of list data",
            list_data,
            [0, 0, view_kind::CBV],
        );
    }

    /// The title memory a buffer view reads ends with its buffer (what the
    /// host refreshes and copies back for it, from its buffer's copy).
    #[test]
    fn buffer_ranges_end_with_their_buffer() {
        let title = Title::new();
        let s = title.start;
        let range = |view: View| buffer_range(&view);
        assert_eq!(
            range(title.srv(39, 0, 1, 16_000, 100)),
            Some((s + 64_000, 400))
        );
        assert_eq!(
            range(title.srv(39, 0, 1, 16_000, 1000)),
            Some((s + 64_000, 1536))
        );
        assert_eq!(range(title.srv(2, 0, 0, 4096, 1)), Some((s + TITLE_LEN, 0)));
        assert_eq!(
            range(title.uav(0, 8, 0, 8000, 500)),
            Some((s + 64_000, 1536))
        );
        assert_eq!(range(title.uav(42, 0, 0, 0, 0)), Some((s, 0)));
    }

    /// Null buffer views are raw or typed, as the views they stand for,
    /// and never empty.
    #[test]
    fn null_buffer_views_are_raw_or_typed() {
        for raw in [true, false] {
            let (srv, uav) = (null_buffer_srv_desc(raw), null_buffer_uav_desc(raw));
            // SAFETY: the buffer members the dimensions name.
            let (s, u) = unsafe { (srv.Anonymous.Buffer, uav.Anonymous.Buffer) };
            assert_eq!((s.NumElements, u.NumElements), (1, 1));
            assert_eq!(s.Flags == D3D12_BUFFER_SRV_FLAG_RAW, raw);
            assert_eq!(u.Flags == D3D12_BUFFER_UAV_FLAG_RAW, raw);
            let format = if raw {
                DXGI_FORMAT_R32_TYPELESS
            } else {
                DXGI_FORMAT_R32_UINT
            };
            assert_eq!((srv.Format, uav.Format), (format, format));
        }
    }

    /// A host copy a slot stops viewing outlives the submission running when
    /// the slot is rewritten, which may have read it: shaders that load slot
    /// indices from memory read slots no list names, so no list holds the
    /// copy (run 54's page fault in the decal classification).
    #[test]
    fn copies_a_slot_stops_viewing_outlive_the_running_submission() {
        use windows::core::Interface;
        let _serial = crate::watch::serial();
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let title = Title::new();
        heap.make_at(0, &title.srv(39, 0, 1, 0, 1024));
        let copy = heap.kept.lock().unwrap()[0]
            .clone()
            .expect("the slot's copy");
        // The title releases the buffer: only the slot holds its copy.
        drop(title);
        let refs = |r: &ID3D12Resource| {
            // SAFETY: a live object; the reference taken is given back.
            unsafe {
                let raw = r.as_raw();
                let vtable = *raw.cast::<*const windows::core::IUnknown_Vtbl>();
                ((*vtable).AddRef)(raw);
                ((*vtable).Release)(raw)
            }
        };
        let held = refs(&copy);
        heap.make_null(0);
        assert_eq!(refs(&copy), held, "released when the slot was rewritten");
        let list = host.new_list().unwrap();
        // SAFETY: an open list.
        unsafe { list.list.Close() }.unwrap();
        host.execute(&[&list]);
        assert_eq!(refs(&copy), held - 1, "held after a submission ran");
    }

    /// Buffer views the title writes into a shader-visible heap get host
    /// slots at once: Grand Theft Auto VI's instance and meshlet culling
    /// reach their slots through indices they load from memory, which no
    /// draw names.
    #[test]
    fn buffer_views_the_title_writes_are_made_at_once() {
        use crate::descriptor::tests::{release_heap, view_heap};
        use crate::descriptor::{create_cbv, create_placed_srv, create_placed_uav};
        type Create =
            extern "system" fn(crate::object::This, *const [u32; 14], *const [u32; 16], u64);
        if host().is_none() {
            return; // no host GPU
        }
        let title = Title::new();
        let h = view_heap(64);
        let heap = crate::descriptor::heap(h);
        let host = heap.host().expect("a host view heap");
        let at = |index: u32| heap.handle(index).expect("a slot of the heap");
        let null = std::ptr::null();
        for (index, view) in [
            (3, title.srv(39, 0, 1, 0, 1024)),
            (4, title.uav(39, 0, 1, 0, 1024)),
        ] {
            let View::Placed(unordered, resource, desc) = view else {
                unreachable!("a placed view");
            };
            let create: Create = if unordered {
                create_placed_uav
            } else {
                create_placed_srv
            };
            create(null, &resource, &desc, at(index));
        }
        let (lo, hi) = (title.start as u32, (title.start >> 32) as u32);
        create_cbv(null, &[lo, hi, 256, 0], at(5));
        let data = host.strides.as_ref().map_or(0, |s| s.1);
        // SAFETY: the slots' records in the mapped buffer.
        let kind = |i: u32| unsafe { (data as *const [u32; 4]).add(i as usize).read_volatile()[2] };
        assert_eq!(
            [kind(3), kind(4), kind(5)],
            [view_kind::RAW_SRV, view_kind::RAW_UAV, view_kind::CBV]
        );
        release_heap(h);
    }
}
