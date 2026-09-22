//! Actual encoded Q4_K x Q8_1 layouts, without production dispatch changes.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use half::f16;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    #[allow(
        clippy::needless_range_loop,
        reason = "fixed index loops are explicitly unrolled to keep member accumulators in registers"
    )]
    pub fn project(
        weights: &[u32],
        input: &[u32],
        mut output: DisjointSlice<f32>,
        k: u32,
        n: u32,
        members: u32,
    ) {
        let index = thread::index_1d().get();
        let row = index / 32;
        let lane = index % 32;
        if row >= n as usize {
            return;
        }
        let chunk = lane & 7;
        let blocks = k as usize / 256;
        let stride = k as usize / 32 * 9;
        let mut acc = [0.0_f32; 8];
        for block_index in 0..blocks {
            let base = (row * blocks + block_index) * 36;
            let header = weights[base];
            let d = cuda_device::convert::cvt_f32_f16x2_lo(header);
            let minimum = cuda_device::convert::cvt_f32_f16x2_hi(header);
            #[unroll]
            for half in 0..2 {
                let group = half * 4 + lane / 8;
                let byte_index = group % 4;
                let first = (weights[base + 1] >> (byte_index * 8)) & 255;
                let second = (weights[base + 2] >> (byte_index * 8)) & 255;
                let third = (weights[base + 3] >> (byte_index * 8)) & 255;
                let scale = if group < 4 {
                    first & 63
                } else {
                    (third & 15) | ((first >> 2) & 48)
                } as f32;
                let minval = if group < 4 {
                    second & 63
                } else {
                    (third >> 4) | ((second >> 2) & 48)
                } as f32;
                let packed = weights[base + 4 + group / 2 * 8 + chunk];
                let quant = (packed >> ((group & 1) * 4)) & 0x0f0f0f0f;
                #[unroll]
                for m in 0..8 {
                    if m < members as usize {
                        let activation = m * stride + (block_index * 8 + group) * 9;
                        let dot =
                            cuda_device::dotprod::dp4a_s32(quant, input[activation + 1 + chunk], 0);
                        let activation_scale =
                            cuda_device::convert::cvt_f32_f16x2_lo(input[activation]);
                        acc[m] += d * scale * activation_scale * dot as f32;
                        if chunk == 0 {
                            let sum = cuda_device::convert::cvt_f32_f16x2_hi(input[activation]);
                            acc[m] -= minimum * minval * sum;
                        }
                    }
                }
            }
        }
        #[unroll]
        for m in 0..8 {
            if m < members as usize {
                let mut total = acc[m];
                #[unroll]
                for offset in [16, 8, 4, 2, 1] {
                    total += warp::shuffle_down_f32(total, offset);
                }
                if lane == 0 {
                    // SAFETY: lane zero owns its row for every active member;
                    // the host allocates exactly members*n output elements.
                    unsafe {
                        *output.get_unchecked_mut(m * n as usize + row) = total;
                    }
                }
            }
        }
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::CudaContext as OracleContext;
    use engine_core::{DataType, WeightTensorSpec};
    use engine_nvidia::{CudaQ4KQ8_1Gemv, CudaWeightStore};
    use std::io::Cursor;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this binary owns the bundle for the module above.
    let module = unsafe { kernels::load(&ctx)? };
    // SAFETY: the parent module's packing bundle belongs to this binary too.
    let packer = unsafe { super::kernels::load(&ctx)? };
    let oracle_ctx = OracleContext::new(0)?;
    let oracle_stream = oracle_ctx.default_stream();
    let oracle = CudaQ4KQ8_1Gemv::new(oracle_stream.clone())?;
    for (k, n) in [(256_usize, 1_usize), (768, 5), (4096, 128), (5120, 5120)] {
        let scales = [1_u8, 17, 33, 63, 32, 47, 62, 48];
        let minima = [63_u8, 32, 17, 1, 47, 48, 33, 62];
        let mut encoded = vec![0_u8; k / 256 * n * 144];
        for (seed, block) in encoded.as_chunks_mut::<144>().0.iter_mut().enumerate() {
            block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x2c00_u16.to_le_bytes());
            for g in 0..4 {
                block[4 + g] = scales[g] | ((scales[g + 4] >> 4) << 6);
                block[8 + g] = minima[g] | ((minima[g + 4] >> 4) << 6);
                block[12 + g] = (scales[g + 4] & 15) | ((minima[g + 4] & 15) << 4);
            }
            for g in 0..8 {
                for i in 0..32 {
                    let q = ((i * 7 + seed * 3 + g) % 16) as u8;
                    block[16 + g / 2 * 32 + i] |= q << ((g % 2) * 4);
                }
            }
        }
        let words: Vec<u32> = encoded
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect();
        let weights = DeviceBuffer::from_host(&stream, &words)?;
        let spec = WeightTensorSpec::new("q4", vec![k as u64, n as u64], DataType::F32)?;
        let mut store = CudaWeightStore::new(oracle_stream.clone());
        store.materialize_quantized(spec, 12, encoded.len() as u64, &mut Cursor::new(&encoded))?;
        let weight = store
            .quantized_tensor("q4")
            .ok_or("missing oracle weight")?;
        for members in [1_usize, 3, 8] {
            let input: Vec<f32> = (0..k * members)
                .map(|i| match i / k % 3 {
                    0 => (((i * 7) % 17) as f32 - 8.0) * 0.125,
                    1 => (((i * 17 + 3) % 101) as f32 - 50.0) * 0.0137,
                    _ => 0.0,
                })
                .collect();
            let packed = super::reference(&input)?;
            let device_float_input = DeviceBuffer::from_host(&stream, &input)?;
            let mut device_input = DeviceBuffer::<u32>::zeroed(&stream, packed.len())?;
            let pack_prepared = packer.prepare_pack_q8(LaunchConfig1D::new(
                u32::try_from((input.len() / 32).div_ceil(4))?,
                128,
                0,
            ))?;
            packer.pack_q8(
                &stream,
                &pack_prepared,
                &device_float_input,
                &mut device_input,
            )?;
            let mut output = DeviceBuffer::<f32>::zeroed(&stream, n * members)?;
            let prepared = module.prepare_project(LaunchConfig1D::new(
                u32::try_from(n.div_ceil(4))?,
                128,
                0,
            ))?;
            module.project(
                &stream,
                &prepared,
                &weights,
                &device_input,
                &mut output,
                k as u32,
                n as u32,
                members as u32,
            )?;
            let actual = output.to_host_vec(&stream)?;
            if device_input.to_host_vec(&stream)? != packed {
                return Err("projection-chain packing differs from the host reference".into());
            }
            let oracle_input = oracle_stream.clone_htod(&packed)?;
            let mut oracle_output = oracle_stream.alloc_zeros::<f32>(n * members)?;
            oracle.execute_batch(weight, &oracle_input, &mut oracle_output, members)?;
            let baseline = oracle_stream.clone_dtoh(&oracle_output)?;
            if actual.len() != n * members || baseline.len() != n * members {
                return Err("projection output length mismatch".into());
            }
            if let Some(i) = actual
                .iter()
                .zip(&baseline)
                .position(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
            {
                return Err(format!(
                    "Q4_K bit mismatch k={k} n={n} m={members} index={i}: Rust={} C++={}",
                    actual[i], baseline[i]
                )
                .into());
            }
            // Independent f64 scalar dot over unpacked nibbles and signed bytes.
            // Uses the existing fixture's magnitude-based f32 accumulation bound.
            for m in 0..members {
                for row in 0..n {
                    let mut reference = 0.0_f64;
                    let mut magnitude = 0.0_f64;
                    let mut float_reference = 0.0_f64;
                    let mut packing_error_bound = 0.0_f64;
                    for b in 0..k / 256 {
                        let block = &encoded[(row * (k / 256) + b) * 144..][..144];
                        for g in 0..8 {
                            let a = 0.125 * f64::from(scales[g]);
                            let minimum = 0.0625 * f64::from(minima[g]);
                            let activation = &packed[m * (k / 32 * 9) + (b * 8 + g) * 9..][..9];
                            let scale = f64::from(f16::from_bits(activation[0] as u16).to_f32());
                            let sum =
                                f64::from(f16::from_bits((activation[0] >> 16) as u16).to_f32());
                            let mut original_sum = 0.0_f64;
                            for i in 0..32 {
                                let q = (block[16 + g / 2 * 32 + i] >> ((g % 2) * 4)) & 15;
                                let x = ((activation[1 + i / 4] >> (8 * (i % 4))) as u8) as i8;
                                let term = a * scale * f64::from(q) * f64::from(x);
                                reference += term;
                                magnitude += term.abs();
                                let original = f64::from(input[m * k + b * 256 + g * 32 + i]);
                                original_sum += original;
                                float_reference += (a * f64::from(q) - minimum) * original;
                                packing_error_bound += a.abs()
                                    * f64::from(q)
                                    * (scale * f64::from(x) - original).abs();
                            }
                            reference -= minimum * sum;
                            magnitude += (minimum * sum).abs();
                            packing_error_bound += minimum.abs() * (sum - original_sum).abs();
                        }
                    }
                    let bound = magnitude * f64::from(f32::EPSILON) * 32.0 + 1e-5;
                    if !reference.is_finite()
                        || !float_reference.is_finite()
                        || !packing_error_bound.is_finite()
                        || (f64::from(actual[m * n + row]) - reference).abs() > bound
                        || (f64::from(actual[m * n + row]) - float_reference).abs()
                            > packing_error_bound + bound
                    {
                        return Err(format!(
                            "Q4_K independent reference failed k={k} n={n} m={members} row={row}"
                        )
                        .into());
                    }
                }
            }
            println!("Q4_K k={k} n={n} m={members}: exact C++ + independent f64 parity");
            if std::env::var_os("BENCH").is_some() && k == 5120 {
                for single in [false, true] {
                    if single && members != 1 {
                        continue;
                    }
                    let variant = if single { "single" } else { "batch" };
                    crate::benchmark::paired(
                        &format!("projection k={k} n={n} m={members} cpp={variant}"),
                        || {
                            crate::benchmark::rust(&stream, || {
                                module.project(
                                    &stream,
                                    &prepared,
                                    &weights,
                                    &device_input,
                                    &mut output,
                                    k as u32,
                                    n as u32,
                                    members as u32,
                                )?;
                                Ok(())
                            })
                        },
                        || {
                            crate::benchmark::cpp(&oracle_stream, || {
                                if single {
                                    oracle.execute(weight, &oracle_input, &mut oracle_output)?;
                                } else {
                                    oracle.execute_batch(
                                        weight,
                                        &oracle_input,
                                        &mut oracle_output,
                                        members,
                                    )?;
                                }
                                Ok(())
                            })
                        },
                    )?;
                    let timed = oracle_stream.clone_dtoh(&oracle_output)?;
                    if actual
                        .iter()
                        .zip(&timed)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Err("timed projection oracle changed output".into());
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn preparation() -> crate::benchmark::Result {
    let ctx = CudaContext::new(0)?;
    let oracle_ctx = cudarc::driver::CudaContext::new(0)?;
    let oracle_stream = oracle_ctx.default_stream();
    crate::benchmark::preparation("rust_projection", || {
        // SAFETY: the binary owns the embedded kernel bundle.
        let module = unsafe { kernels::load(&ctx)? };
        std::hint::black_box(module.prepare_project(LaunchConfig1D::new(1280, 128, 0))?);
        Ok(module)
    })?;
    crate::benchmark::preparation("cpp_projection", || {
        Ok(engine_nvidia::CudaQ4KQ8_1Gemv::new(oracle_stream.clone())?)
    })
}
