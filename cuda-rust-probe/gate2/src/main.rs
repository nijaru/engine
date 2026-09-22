//! Representative CUDA Rust qualification; not a production execution path.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use half::f16;

mod projection;
mod state;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn pack_q8(input: &[f32], mut output: DisjointSlice<u32>) {
        let index = thread::index_1d().get();
        let block = index / 32;
        let lane = index % 32;
        if block >= input.len() / 32 {
            return;
        }
        let x = input[index];
        let mut maximum = x.abs();
        let mut sum = x;
        let mut offset = 16;
        while offset > 0 {
            maximum = maximum.max(warp::shuffle_xor_f32(maximum, offset));
            sum += warp::shuffle_xor_f32(sum, offset);
            offset /= 2;
        }
        let scale = maximum / 127.0;
        let q = if maximum == 0.0 {
            0
        } else {
            (x / scale).round() as i32
        };
        let mut packed = (q & 255) as u32;
        packed |= warp::shuffle_down(packed, 1) << 8;
        packed |= warp::shuffle_down(packed, 2) << 16;
        if lane % 4 == 0 {
            // SAFETY: one lane per four input elements owns this output word;
            // host validation requires exactly nine words per whole input block.
            unsafe {
                *output.get_unchecked_mut(block * 9 + 1 + lane / 4) = packed;
            }
        }
        if lane == 0 {
            let header = cuda_device::convert::cvt_f16x2_f32(scale, sum);
            // SAFETY: lane zero alone writes its block's distinct header.
            unsafe {
                *output.get_unchecked_mut(block * 9) = header;
            }
        }
    }
}

fn packed_words(elements: usize) -> Result<usize, &'static str> {
    if elements == 0 || !elements.is_multiple_of(32) {
        return Err("Q8_1 requires nonempty whole 32-element blocks");
    }
    u32::try_from(elements).map_err(|_| "Q8_1 indexing exceeds u32")?;
    Ok(elements / 32 * 9)
}

// Independent scalar quantization with an explicit butterfly sum: Q8_1's header
// specifies the original input sum, not the sum of quantized integers.
fn reference(input: &[f32]) -> Result<Vec<u32>, &'static str> {
    let mut output = vec![0; packed_words(input.len())?];
    for (values, words) in input
        .as_chunks::<32>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<9>().0.iter_mut())
    {
        let maximum = values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = maximum / 127.0;
        let mut sums = [0.0; 32];
        sums.copy_from_slice(values);
        for offset in [16, 8, 4, 2, 1] {
            let previous = sums;
            for lane in 0..32 {
                sums[lane] = previous[lane] + previous[lane ^ offset];
            }
        }
        words[0] = u32::from(f16::from_f32(scale).to_bits())
            | (u32::from(f16::from_f32(sums[0]).to_bits()) << 16);
        for (index, x) in values.iter().enumerate() {
            let quantized = if maximum == 0.0 {
                0
            } else {
                (x / scale).round() as i32
            };
            words[1 + index / 4] |= ((quantized & 255) as u32) << (8 * (index % 4));
        }
    }
    Ok(output)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this binary owns the bundle compiled from kernels above.
    let module = unsafe { kernels::load(&ctx)? };
    let oracle_context = cudarc::driver::CudaContext::new(0)?;
    let oracle_stream = oracle_context.default_stream();
    let oracle = engine_nvidia::CudaQ8_1Quantizer::new(oracle_stream.clone())?;
    for blocks in [1_usize, 3, 4, 5, 8, 129] {
        let input: Vec<f32> = (0..blocks * 32)
            .map(|i| match i / 32 % 4 {
                0 => 0.0,
                1 => {
                    if i % 32 == 31 {
                        127.0
                    } else {
                        (i % 31) as f32 - 15.5
                    }
                }
                2 => ((i % 29) as f32 - 14.0) / 32.0,
                _ => ((i % 19) as f32 - 9.0) * 0.137,
            })
            .collect();
        let expected = reference(&input)?;
        let x = DeviceBuffer::from_host(&stream, &input)?;
        let mut y = DeviceBuffer::<u32>::zeroed(&stream, expected.len())?;
        let geometry = LaunchConfig1D::new(u32::try_from(blocks.div_ceil(4))?, 128, 0);
        let prepared = module.prepare_pack_q8(geometry)?;
        module.pack_q8(&stream, &prepared, &x, &mut y)?;
        let actual = y.to_host_vec(&stream)?;
        let oracle_x = oracle_stream.clone_htod(&input)?;
        let mut oracle_y = oracle_stream.alloc_zeros::<u32>(expected.len())?;
        oracle.execute(&oracle_x, &mut oracle_y)?;
        let baseline = oracle_stream.clone_dtoh(&oracle_y)?;
        if actual != expected || actual != baseline {
            let mismatch = actual.iter().zip(&expected).position(|(a, b)| a != b);
            return Err(format!(
                "packing mismatch blocks={blocks} host_first={mismatch:?}, cpp_equal={}",
                actual == baseline
            )
            .into());
        }
        println!("Q8_1 blocks={blocks}: exact host + C++ parity");
    }
    projection::run()?;
    state::run()?;
    println!("Gate 2 OPEN: full rejection coverage, preparation and broader timings remain.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_partial_empty_and_overflow() {
        for size in [0, 1, 31, 33, (u32::MAX as usize) + 1] {
            assert!(packed_words(size).is_err());
        }
        assert_eq!(packed_words(32), Ok(9));
        assert_eq!(packed_words(160), Ok(45));
    }
    #[test]
    fn packs_ties_away_from_zero_and_original_sum() {
        let mut values = [0.0; 32];
        values[..5].copy_from_slice(&[127.0, 0.5, -0.5, 1.5, -1.5]);
        let words = reference(&values).unwrap();
        assert_eq!(words[1], 0x02ff017f);
        assert_eq!(words[2] & 255, 254);
        assert_eq!(
            words[0],
            u32::from(f16::ONE.to_bits()) | (u32::from(f16::from_f32(127.0).to_bits()) << 16)
        );
    }
}
