//! The view-heap slots a translated shader reaches through its user data
//! without a used-resource entry naming them (see the module notes of
//! [`super`]): heap indices computed from root constants, table entries
//! past the listed ones. The host makes those slots, with their data, for
//! each draw or dispatch; descriptors each lane loads from memory are
//! beyond it and only counted.

use super::{HeapIndex, HeapTerm};
use crate::ir::Program;
use crate::native::{NativeProgram, RootType};
use crate::resource::{Resources, Role, Site, Source};

/// The slots of [`heap_indices`], and why the slots of the other heap
/// descriptor accesses cannot be known before the shader runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HeapIndices {
    pub known: Vec<HeapIndex>,
    pub unknown: Vec<&'static str>,
}

/// The view-heap slots the resource descriptors of `p` read, as the user
/// data gives them, deduplicated; slots a used-resource entry names (which
/// the host makes through their tables) are left out.
pub fn heap_indices(p: &Program, native: &NativeProgram, res: &Resources) -> HeapIndices {
    let mut out = HeapIndices::default();
    for d in res.descriptors.iter().filter(|d| d.role == Role::Resource) {
        let name = p.blocks[d.site.0].insts[d.site.1].name();
        let write = name.contains("_store") || name.contains("_atomic");
        for leaf in d.source.leaves() {
            let found = match *leaf {
                Source::Loaded { load, first } => slot(native, res, load, first),
                Source::Fetched { .. } => Err("a descriptor each lane loads"),
                _ => continue,
            };
            match found {
                Ok(Some(h)) => match out
                    .known
                    .iter_mut()
                    .find(|k| (&k.terms, k.offset) == (&h.terms, h.offset))
                {
                    Some(k) => k.write |= write,
                    None => out.known.push(HeapIndex { write, ..h }),
                },
                Ok(None) => {}
                Err(why) => out.unknown.push(why),
            }
        }
    }
    out
}

/// The slot of the descriptor at dword `first` of what the scalar load at
/// `load` reads; none outside the view heap or where a used-resource
/// entry names it.
fn slot(
    native: &NativeProgram,
    res: &Resources,
    load: Site,
    first: u8,
) -> Result<Option<HeapIndex>, &'static str> {
    let &(_, a) = res
        .scalar
        .iter()
        .find(|s| s.0 == load)
        .ok_or("an untracked descriptor load")?;
    let Some(rp) = native.layout.get(a.param) else {
        return Ok(None);
    };
    let mut terms = Vec::new();
    match rp.root {
        RootType::Table => {
            let dword = a.offset / 4 + u32::from(first);
            let named = native.used.iter().any(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords == dword
            });
            if named && a.dynamic.is_none() {
                return Ok(None);
            }
            terms.push(HeapTerm {
                user_data: u32::from(rp.user_data),
                mask: u32::MAX,
                scale: 1,
            });
        }
        RootType::ResourceHeap => {}
        _ => return Ok(None),
    }
    let mut offset = a.offset.wrapping_add(4 * u32::from(first));
    if a.dynamic.is_some() {
        let term = res
            .terms
            .iter()
            .find(|t| t.0 == load)
            .ok_or("an untracked descriptor load")?
            .1?;
        if let Some((param, dword)) = term.param {
            terms.push(HeapTerm {
                user_data: u32::from(native.layout[param].user_data) + dword,
                mask: term.mask,
                scale: term.scale,
            });
        }
        offset = offset.wrapping_add(term.offset);
    }
    Ok(Some(HeapIndex {
        terms,
        offset,
        write: false,
    }))
}
