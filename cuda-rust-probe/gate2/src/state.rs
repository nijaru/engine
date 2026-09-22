//! SIMT state proof: independent allocation per request, persistent across launches.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

mod reference;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel ABI preserves eight independently owned matrices and explicit geometry"
    )]
    pub fn update<'a>(
        mut s0: DisjointSlice<'a, f32>,
        mut s1: DisjointSlice<'a, f32>,
        mut s2: DisjointSlice<'a, f32>,
        mut s3: DisjointSlice<'a, f32>,
        mut s4: DisjointSlice<'a, f32>,
        mut s5: DisjointSlice<'a, f32>,
        mut s6: DisjointSlice<'a, f32>,
        mut s7: DisjointSlice<'a, f32>,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        mut output: DisjointSlice<f32>,
        heads: u32,
        k_heads: u32,
        dim: u32,
        v_offset: u32,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let dim = dim as usize;
        let heads = heads as usize;
        let member = index / (heads * dim);
        let head = index / dim % heads;
        let col = index % dim;
        let state = match member {
            0 => &mut s0,
            1 => &mut s1,
            2 => &mut s2,
            3 => &mut s3,
            4 => &mut s4,
            5 => &mut s5,
            6 => &mut s6,
            _ => &mut s7,
        };
        let qk_base = (member * k_heads as usize + head % k_heads as usize) * dim;
        let base = head * dim * dim;
        let mut sk = 0.0;
        for row in 0..dim {
            // SAFETY: each thread exclusively owns one state column of one
            // head/member; all accesses stay within its validated allocation.
            let element = unsafe { state.get_unchecked_mut(base + row * dim + col) };
            let decayed = *element * decay[member * heads + head];
            *element = decayed;
            sk += decayed * k[qk_base + row];
        }
        let v_index =
            member * (v_offset as usize + heads * dim) + v_offset as usize + head * dim + col;
        let delta = (v[v_index] - sk) * beta[member * heads + head];
        let mut out = 0.0;
        for row in 0..dim {
            // SAFETY: same exclusive column as the first pass; launches use
            // one ordered stream and retain every matrix until completion.
            let element = unsafe { state.get_unchecked_mut(base + row * dim + col) };
            let updated = *element + k[qk_base + row] * delta;
            *element = updated;
            out += updated * q[qk_base + row];
        }
        // SAFETY: the linear thread index uniquely owns this bounded output.
        unsafe {
            *output.get_unchecked_mut(index) =
                out * cuda_device::float::rsqrt_approx_f32(dim as f32);
        }
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: bundled kernels belong to this binary.
    let module = unsafe { kernels::load(&ctx)? };
    let oracle_ctx = cudarc::driver::CudaContext::new(0)?;
    let oracle_stream = oracle_ctx.default_stream();
    let oracle = engine_nvidia::CudaQwen35Ops::from_context(&oracle_ctx, oracle_stream.clone())?;
    for (members, heads, kh, dim) in [
        (1_usize, 2_usize, 1_usize, 16_usize),
        (3, 4, 2, 32),
        (8, 4, 2, 128),
        (3, 48, 16, 128),
        (8, 48, 16, 128),
    ] {
        let state_len = heads * dim * dim;
        let offset = 2 * kh * dim;
        let mut initial = Vec::new();
        let mut matrices = Vec::new();
        let mut baseline = Vec::new();
        for m in 0..8 {
            let values: Vec<f32> = (0..if m < members { state_len } else { 1 })
                .map(|i| {
                    if m < members && m % 3 == 0 {
                        0.0
                    } else {
                        (((i * 7 + m * 11) % 31) as f32 - 15.0) * 0.001
                    }
                })
                .collect();
            matrices.push(DeviceBuffer::from_host(&stream, &values)?);
            baseline.push(oracle_stream.clone_htod(&values)?);
            initial.push(values);
        }
        let mut reference: Vec<Vec<_>> = initial[..members]
            .iter()
            .map(|values| {
                values
                    .iter()
                    .copied()
                    .map(reference::Value::exact)
                    .collect()
            })
            .collect();
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, members * heads * dim)?;
        let mut oracle_out = oracle_stream.alloc_zeros::<f32>(members * heads * dim)?;
        let prepared = module.prepare_update(LaunchConfig1D::new(
            u32::try_from((members * heads * dim).div_ceil(128))?,
            128,
            0,
        ))?;
        let steps = if heads == 4 && members == 3 { 64 } else { 8 };
        for step in 0..steps {
            // Reorder live allocation owners, not their contents; references follow
            // the same histories. Pads remain inactive throughout.
            if step % 2 == 1 {
                matrices.swap(0, members - 1);
                baseline.swap(0, members - 1);
                reference.swap(0, members - 1);
            }
            let mut q: Vec<f32> = (0..members * kh * dim)
                .map(|i| (((i + step * 3) % 17) as f32 - 8.0) * 0.02)
                .collect();
            let mut k: Vec<f32> = (0..q.len())
                .map(|i| (((i + step * 5) % 13) as f32 - 6.0) * 0.015)
                .collect();
            for vector in q.chunks_exact_mut(dim).chain(k.chunks_exact_mut(dim)) {
                let norm = vector
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt() as f32;
                for value in vector {
                    *value /= norm;
                }
            }
            let v: Vec<f32> = (0..members * (offset + heads * dim))
                .map(|i| (((i + step * 7) % 23) as f32 - 11.0) * 0.013)
                .collect();
            let decay: Vec<f32> = (0..members * heads)
                .map(|i| match (i + step) % 11 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => 0.85 + ((i + step) % 7) as f32 * 0.01,
                })
                .collect();
            let beta: Vec<f32> = (0..members * heads)
                .map(|i| match (i + step) % 7 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => 0.1 + ((i + step) % 5) as f32 * 0.03,
                })
                .collect();
            let qd = DeviceBuffer::from_host(&stream, &q)?;
            let kd = DeviceBuffer::from_host(&stream, &k)?;
            let vd = DeviceBuffer::from_host(&stream, &v)?;
            let dd = DeviceBuffer::from_host(&stream, &decay)?;
            let bd = DeviceBuffer::from_host(&stream, &beta)?;
            let oq = oracle_stream.clone_htod(&q)?;
            let ok = oracle_stream.clone_htod(&k)?;
            let ov = oracle_stream.clone_htod(&v)?;
            let od = oracle_stream.clone_htod(&decay)?;
            let ob = oracle_stream.clone_htod(&beta)?;
            let independent = reference::Inputs {
                heads,
                key_heads: kh,
                dim,
                offset,
                q: &q,
                k: &k,
                v: &v,
                decay: &decay,
                beta: &beta,
            }
            .advance(&mut reference);
            let [s0, s1, s2, s3, s4, s5, s6, s7] = matrices.as_mut_slice() else {
                return Err("state slots".into());
            };
            module.update(
                &stream,
                &prepared,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                s7,
                &qd,
                &kd,
                &vd,
                &dd,
                &bd,
                &mut out,
                heads as u32,
                kh as u32,
                dim as u32,
                offset as u32,
            )?;
            let (live, pads) = baseline.split_at_mut(members);
            let mut refs: Vec<_> = live.iter_mut().collect();
            oracle.gdn_state_update_batch(
                &mut refs,
                pads,
                &oq,
                &ok,
                &ov,
                &od,
                &ob,
                &mut oracle_out,
                heads,
                kh,
                dim,
                offset,
            )?;
            let actual = out.to_host_vec(&stream)?;
            let expected = oracle_stream.clone_dtoh(&oracle_out)?;
            compare(&actual, &expected, "output", members, dim, step)?;
            compare_independent(&actual, &independent, "output", step)?;
            for m in 0..8 {
                let actual = matrices[m].to_host_vec(&stream)?;
                let expected = oracle_stream.clone_dtoh(&baseline[m])?;
                compare(&actual, &expected, "state", members, dim, step)?;
                if m < members {
                    compare_independent(&actual, &reference[m], "state", step)?;
                }
                if m >= members && actual != initial[m] {
                    return Err("inactive state was mutated".into());
                }
            }
        }
        println!(
            "GDN m={members} heads={heads}/{kh} dim={dim}: {steps} varied/reordered steps, exact C++ + independent f64 state/output acceptance"
        );
    }
    Ok(())
}

fn compare_independent(
    actual: &[f32],
    expected: &[reference::Value],
    kind: &str,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err("reference length mismatch".into());
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if !expected.accepts(*actual) {
            return Err(format!(
                "GDN independent {kind} step={step} index={index}: {actual} outside {expected:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn compare(
    actual: &[f32],
    expected: &[f32],
    kind: &str,
    members: usize,
    dim: usize,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err("length mismatch".into());
    }
    if let Some(i) = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(format!(
            "GDN {kind} bit mismatch m={members} dim={dim} step={step} index={i}: Rust={} C++={}",
            actual[i], expected[i]
        )
        .into());
    }
    Ok(())
}
