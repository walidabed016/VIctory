//! Rectangle lists on the host. The console draws lists of rectangles
//! (primitive topology 17, pipelines of type 19) from three vertices each
//! and the hardware adds the fourth corner. The host draws them as a
//! triangle list through a geometry shader made from the vertex shader's
//! outputs: each rectangle comes out as a strip of four vertices, the
//! three the vertex shader gave and the corner opposite the one between
//! the other two, with every interpolated value continued linearly to it.
//! Draws whose vertex count the GPU writes (indirect draws) then need
//! nothing from the CPU, and vertex shaders that place their corners by
//! vertex index (Grand Theft Auto VI's tiled lighting) work as well as
//! ones that read vertex buffers.
//!
//! The corner between the other two is the one opposite the longest edge
//! (the diagonal), measured on the positions. Grand Theft Auto VI's
//! full-screen rectangles give top left, bottom left, bottom right (the
//! middle vertex is that corner); Mesa's blits on AMD hardware give the
//! corner first. Either works. Values that are not interpolated (integers)
//! are the first vertex's at all four corners.

use super::bitcode::{Attr, Bin, Term, V, pred};
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The geometry shader that draws the rectangles of a translated vertex
/// shader (its container, as [`super::HostShader::dxil`]): its outputs in
/// and out, four vertices per rectangle.
pub fn rectangles(vs: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(vs)?;
    let part = c
        .part(FourCC::OSG1)
        .ok_or(ShaderError::Malformed("a vertex shader without outputs"))?;
    let mut elements = super::signature(part.data)?;
    let base = crate::dxbc::u32_at(part.data, 4)? as usize;
    for (i, e) in elements.iter_mut().enumerate() {
        // The components written: the row's mask less those it never writes.
        let never = *part
            .data
            .get(base + 32 * i + 25)
            .ok_or(ShaderError::Malformed("truncated"))?;
        let mask = ((1u8 << e.cols) - 1) << e.col;
        e.usage = (mask & !never) >> e.col;
        e.interp = super::default_interp(e);
    }

    let mut s = Shader::new(false);
    s.geometry = Some(4);
    let b = s.entry;
    // Each component written: element, column, its value at the three
    // vertices.
    let mut values: Vec<(u32, u8, [V; 3])> = Vec::new();
    for (id, e) in elements.iter().enumerate() {
        let (ty, suffix) = if e.comp == 9 {
            (s.t.f32, ".f32")
        } else {
            (s.t.i32, ".i32")
        };
        for col in (0..e.cols).filter(|c| e.usage >> c & 1 != 0) {
            let v = [0, 1, 2].map(|vertex| {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(vertex)];
                let name = format!("loadInput{suffix}");
                s.op(b, &name, Attr::ReadNone, ty, 4, &args)
            });
            values.push((id as u32, col, v));
        }
    }

    // Whether the corner is the first vertex, or the second; else the
    // third. Without a position, the second (Grand Theft Auto VI's order).
    let position = |col: u8| {
        values
            .iter()
            .find(|v| elements[v.0 as usize].kind == 3 && v.1 == col)
            .map(|v| v.2)
    };
    let (k0, k1) = match (position(0), position(1)) {
        (Some(x), Some(y)) => {
            let mut edge = |i: usize, j: usize| {
                let dx = s.m.bin(b, Bin::Sub, x[i], x[j]);
                let dy = s.m.bin(b, Bin::Sub, y[i], y[j]);
                let xx = s.m.bin(b, Bin::Mul, dx, dx);
                let yy = s.m.bin(b, Bin::Mul, dy, dy);
                s.m.bin(b, Bin::Add, xx, yy)
            };
            let (d01, d12, d02) = (edge(0, 1), edge(1, 2), edge(0, 2));
            let over01 = s.m.cmp(b, pred::OGE, d12, d01);
            let over02 = s.m.cmp(b, pred::OGE, d12, d02);
            let k0 = s.m.bin(b, Bin::And, over01, over02);
            let t = s.m.bool(true);
            let not_k0 = s.m.bin(b, Bin::Xor, k0, t);
            let longest02 = s.m.cmp(b, pred::OGE, d02, d01);
            let k1 = s.m.bin(b, Bin::And, longest02, not_k0);
            (k0, k1)
        }
        _ => (s.m.bool(false), s.m.bool(true)),
    };

    // The strip: the corner, the two after it in the title's order, then
    // the fourth corner. Both triangles turn as the title's does.
    let void = s.t.void;
    let mut corners: Vec<[V; 4]> = Vec::new();
    for &(id, _, v) in &values {
        let e = &elements[id as usize];
        if e.comp != 9 || e.interp == 1 {
            corners.push([v[0]; 4]);
            continue;
        }
        let mut at = |n: usize| {
            let second = s.m.select(b, k1, v[(1 + n) % 3], v[(2 + n) % 3]);
            s.m.select(b, k0, v[n % 3], second)
        };
        let (corner, next, last) = (at(0), at(1), at(2));
        let sum = s.m.bin(b, Bin::Add, next, last);
        let fourth = s.m.bin(b, Bin::Sub, sum, corner);
        corners.push([corner, next, last, fourth]);
    }
    for slot in 0..4 {
        for (&(id, col, _), c) in values.iter().zip(&corners) {
            let suffix = if elements[id as usize].comp == 9 {
                ".f32"
            } else {
                ".i32"
            };
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), c[slot]];
            let name = format!("storeOutput{suffix}");
            s.op(b, &name, Attr::NoUnwind, void, 5, &args);
        }
        let stream = s.m.i8(0);
        s.op(b, "emitStream", Attr::NoUnwind, void, 97, &[stream]);
    }
    s.m.terminate(b, Term::Ret);

    // Each output scalar depends on its own input scalar, and an
    // interpolated one on the position's x and y too (the corner).
    let scalar = |id: u32, col: u8| {
        let e = &elements[id as usize];
        e.row.unwrap_or(0) * 4 + u32::from(e.col + col)
    };
    let scalars = elements
        .iter()
        .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
        .max()
        .unwrap_or(0);
    let per = scalars.div_ceil(32);
    let mut dependencies = vec![0u32; (scalars * per) as usize];
    let corner_inputs: Vec<u32> = values
        .iter()
        .filter(|v| elements[v.0 as usize].kind == 3 && v.1 < 2)
        .map(|v| scalar(v.0, v.1))
        .collect();
    for (&(id, col, _), c) in values.iter().zip(&corners) {
        let out = scalar(id, col);
        let interpolated = c[0] != c[3];
        let from = std::iter::once(out).chain(
            corner_inputs
                .iter()
                .copied()
                .filter(|_| interpolated && corner_inputs.len() == 2),
        );
        for input in from {
            dependencies[(input * per + out / 32) as usize] |= 1 << (out % 32);
        }
    }
    s.dependencies = Some(dependencies);
    s.inputs = elements.clone();
    s.outputs = elements;
    Ok(s.finish())
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::{Attr, Cast, Term};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::shader::Shader;
    use super::super::validator::Validator;
    use super::super::{HostShader, signature};
    use super::rectangles;
    use crate::dxbc::{Container, FourCC};
    use crate::native::HwStage;

    fn element(name: &str, index: u32, kind: u8, comp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col: 0,
            cols,
            usage: 0,
            half: false,
        }
    }

    fn host_shader(stage: HwStage, dxil: Vec<u8>) -> HostShader {
        HostShader {
            stage,
            dxil,
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            view_strides: false,
            heap_indices: Vec::new(),
            heap_unknown: 0,
            root_uavs: Vec::new(),
            memory: false,
            memory_writes: false,
            view_addresses: false,
        }
    }

    /// A vertex shader passing POSITION on as the position and TEXCOORD0 as
    /// TEXCOORD0 (interpolated), with TEXCOORD0.w as an integer TEXCOORD1
    /// (not interpolated).
    fn vs() -> HostShader {
        let mut s = Shader::new(false);
        let b = s.entry;
        let load = |s: &mut Shader, id: u32, col: u8| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let store = |s: &mut Shader, id: u32, col: u8, v, suffix: &str| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), v];
            let name = format!("storeOutput{suffix}");
            s.op(b, &name, Attr::NoUnwind, s.t.void, 5, &args);
        };
        for id in 0..2 {
            for col in 0..4 {
                let v = load(&mut s, id, col);
                store(&mut s, id, col, v, ".f32");
            }
        }
        let w = load(&mut s, 1, 3);
        let n = s.m.cast(b, Cast::FpToUi, w, s.t.i32);
        store(&mut s, 2, 0, n, ".i32");
        s.m.terminate(b, Term::Ret);
        s.inputs = vec![
            SigElement {
                usage: 15,
                ..element("POSITION", 0, 0, 9, 0, 4)
            },
            SigElement {
                usage: 15,
                ..element("TEXCOORD", 0, 0, 9, 1, 4)
            },
        ];
        s.outputs = vec![
            SigElement {
                usage: 15,
                interp: 4,
                ..element("SV_Position", 0, 3, 9, 0, 4)
            },
            SigElement {
                usage: 15,
                interp: 2,
                ..element("TEXCOORD", 0, 0, 9, 1, 4)
            },
            SigElement {
                usage: 1,
                interp: 1,
                ..element("TEXCOORD", 1, 0, 5, 2, 1)
            },
        ];
        host_shader(HwStage::Vs, s.finish())
    }

    /// A pixel shader writing TEXCOORD0.xy and TEXCOORD1 (as a float).
    fn ps() -> HostShader {
        let mut s = Shader::new(true);
        let b = s.entry;
        let mut out = Vec::new();
        for col in 0..2 {
            let args = [s.m.i32(1), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            out.push(s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args));
        }
        let args = [s.m.i32(2), s.m.i32(0), s.m.i8(0), s.m.undef(s.t.i32)];
        let n = s.op(b, "loadInput.i32", Attr::ReadNone, s.t.i32, 4, &args);
        out.push(s.m.cast(b, Cast::UiToFp, n, s.t.f32));
        out.push(s.m.f32(1f32.to_bits()));
        for (col, v) in out.into_iter().enumerate() {
            let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, Term::Ret);
        s.inputs = vec![
            SigElement {
                interp: 4,
                ..element("SV_Position", 0, 3, 9, 0, 4)
            },
            SigElement {
                usage: 3,
                interp: 2,
                ..element("TEXCOORD", 0, 0, 9, 1, 4)
            },
            SigElement {
                usage: 1,
                interp: 1,
                ..element("TEXCOORD", 1, 0, 5, 2, 1)
            },
        ];
        s.outputs = vec![SigElement {
            usage: 15,
            ..element("SV_Target", 0, 16, 9, 0, 4)
        }];
        host_shader(HwStage::Ps, s.finish())
    }

    #[test]
    fn the_geometry_shader_passes_the_validator_and_keeps_the_signature() {
        let vs = vs();
        let mut gs = rectangles(&vs.dxil).unwrap();
        let path = Validator::default_path();
        match Validator::load(path.as_deref()) {
            Ok(mut v) => v.validate(&mut gs).unwrap(),
            Err(e) => eprintln!("no dxil.dll ({e}); validation skipped"),
        }
        let sig = |dxil: &[u8], fourcc| {
            let c = Container::parse(dxil).unwrap();
            signature(c.part(fourcc).unwrap().data).unwrap()
        };
        let outputs = sig(&vs.dxil, FourCC::OSG1);
        assert_eq!(sig(&gs, FourCC::ISG1), outputs);
        assert_eq!(sig(&gs, FourCC::OSG1), outputs);
    }

    /// One rectangle over an 8x8 target, its three vertices in each order
    /// (the corner between the other two first, second and last): the
    /// whole target is drawn, TEXCOORD0 continues linearly into the added
    /// corner, and TEXCOORD1 is the first vertex's everywhere.
    #[test]
    fn a_rectangle_covers_its_fourth_corner_in_any_order() {
        let Ok(host) = Host::new(std::env::var_os("VICTORY_WARP").is_some()) else {
            eprintln!("no D3D12 device; skipped");
            return;
        };
        let (vs, ps) = (vs(), ps());
        let gs = rectangles(&vs.dxil).unwrap();
        let root = host.root_signature(&vs, &ps).unwrap();
        let pso = host
            .pipeline_with_stages(&root, &vs.dxil, Some(&gs), &ps.dxil)
            .unwrap();
        // Position (x, y), TEXCOORD0 (u, v) = ((x + 1) / 2, (1 - y) / 2),
        // and the vertex's integer in TEXCOORD0.w.
        let corner =
            |x: f32, y: f32, n: f32| [x, y, 0.0, 1.0, (x + 1.0) / 2.0, (1.0 - y) / 2.0, 0.0, n];
        let (tl, bl, br, tr) = ((-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0));
        for order in [[tl, bl, br], [bl, tl, br], [tl, br, bl], [br, tr, tl]] {
            let vertices: Vec<f32> = order
                .iter()
                .zip([3.0, 5.0, 9.0])
                .flat_map(|(&(x, y), n)| corner(x, y, n))
                .collect();
            let draw = Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 32,
                size: (8, 8),
                mesh_groups: 0,
            };
            let texels = host.draw(&root, &pso, &draw).unwrap();
            for (i, t) in texels.iter().enumerate() {
                let (x, y) = ((i % 8) as f32, (i / 8) as f32);
                let want = [(2.0 * x + 1.0) / 16.0, (2.0 * y + 1.0) / 16.0, 3.0, 1.0];
                let close = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                assert!(
                    close,
                    "order {order:?}: texel ({x}, {y}) is {t:?}, not {want:?}"
                );
            }
        }
    }
}
