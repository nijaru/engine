use std::sync::LazyLock;

const INPUTS: usize = 768;
const ROWS: usize = 5;

struct Block {
    encoded: [u8; 144],
    quants: [u8; 256],
    scales: [u8; 8],
    minima: [u8; 8],
    d: f64,
    minimum: f64,
}

fn fixture() -> Vec<Block> {
    (0..ROWS * (INPUTS / 256))
        .map(|seed| {
            let scales = [1, 17, 33, 63, 32, 47, 62, 48];
            let minima = [63, 32, 17, 1, 47, 48, 33, 62];
            let d_bits = [0x2c00_u16, 0x3000, 0x3400][seed % 3];
            let minimum_bits = [0x2800_u16, 0x2c00, 0x3000][seed % 3];
            let mut encoded = [0_u8; 144];
            encoded[..2].copy_from_slice(&d_bits.to_le_bytes());
            encoded[2..4].copy_from_slice(&minimum_bits.to_le_bytes());
            for group in 0..4 {
                encoded[4 + group] = scales[group] | ((scales[group + 4] >> 4) << 6);
                encoded[8 + group] = minima[group] | ((minima[group + 4] >> 4) << 6);
                encoded[12 + group] = (scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4);
            }
            let quants = std::array::from_fn(|index| {
                u8::try_from((index * 7 + seed * 3 + index / 32) % 16).unwrap()
            });
            for group in 0..8 {
                for index in 0..32 {
                    encoded[16 + (group / 2) * 32 + index] |=
                        quants[group * 32 + index] << ((group % 2) * 4);
                }
            }
            Block {
                encoded,
                quants,
                scales,
                minima,
                d: decode_half(d_bits),
                minimum: decode_half(minimum_bits),
            }
        })
        .collect()
}

fn decode_half(bits: u16) -> f64 {
    let exponent = i32::from((bits >> 10) & 31);
    let fraction = f64::from(bits & 1023);
    let magnitude = if exponent == 0 {
        fraction * 2.0_f64.powi(-24)
    } else {
        (1.0 + fraction / 1024.0) * 2.0_f64.powi(exponent - 15)
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[allow(
    clippy::float_cmp,
    reason = "round-to-nearest-even requires exact midpoint ties"
)]
fn encode_half(value: f32) -> u16 {
    static POSITIVE_FINITE: LazyLock<Vec<f64>> =
        LazyLock::new(|| (0_u16..=0x7bff).map(decode_half).collect());
    let value_abs = f64::from(value.abs());
    let upper = POSITIVE_FINITE.partition_point(|&candidate| candidate < value_abs);
    assert!(upper < POSITIVE_FINITE.len(), "fixture half overflow");
    let selected = if upper == 0 {
        0
    } else {
        let low_distance = value_abs - POSITIVE_FINITE[upper - 1];
        let high_distance = POSITIVE_FINITE[upper] - value_abs;
        if low_distance < high_distance || (low_distance == high_distance && upper % 2 == 1) {
            upper - 1
        } else {
            upper
        }
    };
    u16::try_from(selected).unwrap() | if value.is_sign_negative() { 0x8000 } else { 0 }
}

fn activations(exact: bool) -> Vec<f32> {
    (0..INPUTS)
        .map(|index| {
            if exact {
                let integer = match index % 32 {
                    0 => 127,
                    1 => -127,
                    _ => i16::try_from((index * 7) % 17).unwrap() - 8,
                };
                f32::from(integer) * [0.25, 0.125, 0.0625][(index / 32) % 3]
            } else {
                f32::from(i16::try_from((index * 17 + 3) % 101).unwrap() - 50) * 0.0137
            }
        })
        .collect()
}

fn pack_host(input: &[f32]) -> Vec<u32> {
    input
        .as_chunks::<32>()
        .0
        .iter()
        .flat_map(|block| {
            let maximum = block
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let scale = maximum / 127.0;
            // Sum using the same pairwise tree specified by the warp packer.
            // Rounding to half is independently selected from representable values.
            let mut sums = *block;
            for offset in [16, 8, 4, 2, 1] {
                let previous = sums;
                for lane in 0..32 {
                    sums[lane] += previous[lane ^ offset];
                }
            }
            let mut words =
                vec![u32::from(encode_half(scale)) | (u32::from(encode_half(sums[0])) << 16)];
            for chunk in block.as_chunks::<4>().0 {
                let bytes: [u8; 4] = std::array::from_fn(|index| {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "fixture clamps finite quantized values to signed byte range"
                    )]
                    let signed = if maximum == 0.0 {
                        0
                    } else {
                        (chunk[index] / scale).round().clamp(-127.0, 127.0) as i8
                    };
                    signed.to_ne_bytes()[0]
                });
                words.push(u32::from_le_bytes(bytes));
            }
            words
        })
        .collect()
}

struct Reference {
    packed_output: f64,
    float_output: f64,
    packing_error_bound: f64,
    arithmetic_magnitude: f64,
}

fn reference(blocks: &[Block], input: &[f32], packed: &[u32]) -> Vec<Reference> {
    blocks
        .as_chunks::<{ INPUTS / 256 }>()
        .0
        .iter()
        .map(|row| {
            let mut result = Reference {
                packed_output: 0.0,
                float_output: 0.0,
                packing_error_bound: 0.0,
                arithmetic_magnitude: 0.0,
            };
            for (block_index, block) in row.iter().enumerate() {
                for group in 0..8 {
                    let input_offset = block_index * 256 + group * 32;
                    let words = &packed[input_offset / 32 * 9..][..9];
                    let header = words[0].to_le_bytes();
                    let scale = decode_half(u16::from_le_bytes([header[0], header[1]]));
                    let sum = decode_half(u16::from_le_bytes([header[2], header[3]]));
                    let q: Vec<i8> = words[1..]
                        .iter()
                        .flat_map(|word| word.to_le_bytes().map(|byte| i8::from_ne_bytes([byte])))
                        .collect();
                    let a = block.d * f64::from(block.scales[group]);
                    let b = block.minimum * f64::from(block.minima[group]);
                    let mut integer_dot = 0_i32;
                    let mut original_sum = 0.0;
                    for index in 0..32 {
                        let weight_q = block.quants[group * 32 + index];
                        let x = f64::from(input[input_offset + index]);
                        integer_dot += i32::from(weight_q) * i32::from(q[index]);
                        original_sum += x;
                        result.float_output += (a * f64::from(weight_q) - b) * x;
                        result.packing_error_bound +=
                            a.abs() * f64::from(weight_q) * (scale * f64::from(q[index]) - x).abs();
                        result.arithmetic_magnitude +=
                            (a * f64::from(weight_q) * scale * f64::from(q[index])).abs();
                    }
                    result.packed_output += a * scale * f64::from(integer_dot) - b * sum;
                    result.arithmetic_magnitude += (b * sum).abs();
                    result.packing_error_bound += b.abs() * (sum - original_sum).abs();
                }
            }
            result
        })
        .collect()
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "dyadic fixtures must be exactly representable"
)]
fn exact_activations_and_fixture_metadata_match_float_arithmetic() {
    let blocks = fixture();
    let input = activations(true);
    let packed = pack_host(&input);
    for block in &blocks {
        let decoded = engine_gguf::dequantize_block(12, &block.encoded).unwrap();
        for (index, value) in decoded.iter().enumerate() {
            assert_eq!(
                f64::from(*value),
                block.d * f64::from(block.scales[index / 32]) * f64::from(block.quants[index])
                    - block.minimum * f64::from(block.minima[index / 32])
            );
        }
    }
    for row in reference(&blocks, &input, &packed) {
        assert_eq!(row.packing_error_bound, 0.0);
        assert_eq!(row.packed_output, row.float_output);
    }
}

#[test]
fn lossy_activations_have_an_explicit_float_error_bound() {
    let blocks = fixture();
    let input = activations(false);
    let packed = pack_host(&input);
    let mut differs_from_float = false;
    for row in reference(&blocks, &input, &packed) {
        let error = (row.packed_output - row.float_output).abs();
        differs_from_float |= error > 1e-4;
        assert!(error <= row.packing_error_bound + 1e-10);
        assert!(row.packing_error_bound > 0.0);
    }
    assert!(differs_from_float, "fixture must exercise lossy arithmetic");
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::CudaContext;
    use engine_core::{DataType, WeightTensorSpec};
    use engine_nvidia::{
        CudaQ4KQ8_1Gemv, CudaQ8_1Quantizer, CudaQuantizedKernelError, CudaWeightStore,
    };
    use std::io::Cursor;

    #[test]
    #[ignore = "requires CUDA GPU and NVRTC"]
    fn q4_k_dp4a_matches_packed_host_arithmetic_and_float_error_bound() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let kernel = CudaQ4KQ8_1Gemv::new(stream.clone()).unwrap();
        let quantizer = CudaQ8_1Quantizer::new(stream.clone()).unwrap();
        let blocks = fixture();
        let encoded: Vec<u8> = blocks.iter().flat_map(|block| block.encoded).collect();
        let spec =
            WeightTensorSpec::new("q4", vec![INPUTS as u64, ROWS as u64], DataType::F32).unwrap();
        let mut store = CudaWeightStore::new(stream.clone());
        store
            .materialize_quantized(spec, 12, encoded.len() as u64, &mut Cursor::new(encoded))
            .unwrap();
        let weight = store.quantized_tensor("q4").unwrap();
        let mut packed_device = stream.alloc_zeros::<u32>(INPUTS / 32 * 9).unwrap();
        let mut output = stream.alloc_zeros::<f32>(ROWS).unwrap();
        for input in [activations(true), activations(false), vec![0.0; INPUTS]] {
            let input_device = stream.clone_htod(&input).unwrap();
            quantizer
                .execute(&input_device, &mut packed_device)
                .unwrap();
            let packed = stream.clone_dtoh(&packed_device).unwrap();
            assert_eq!(packed, pack_host(&input));
            kernel.execute(weight, &packed_device, &mut output).unwrap();
            let actual = stream.clone_dtoh(&output).unwrap();
            for (actual, expected) in actual.iter().zip(reference(&blocks, &input, &packed)) {
                let rounding =
                    expected.arithmetic_magnitude * f64::from(f32::EPSILON) * 32.0 + 1e-5;
                assert!(
                    (f64::from(*actual) - expected.packed_output).abs() <= rounding,
                    "packed output {actual} vs {}",
                    expected.packed_output
                );
                assert!(
                    (f64::from(*actual) - expected.float_output).abs()
                        <= expected.packing_error_bound + rounding
                );
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU and NVRTC"]
    fn q4_k_dp4a_rejects_invalid_storage() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let kernel = CudaQ4KQ8_1Gemv::new(stream.clone()).unwrap();
        let spec = WeightTensorSpec::new("q4", vec![256, 1], DataType::F32).unwrap();
        let mut store = CudaWeightStore::new(stream.clone());
        store
            .materialize_quantized(spec, 12, 144, &mut Cursor::new(fixture()[0].encoded))
            .unwrap();
        let weight = store.quantized_tensor("q4").unwrap();
        let short = stream.alloc_zeros::<u32>(71).unwrap();
        let input = stream.alloc_zeros::<u32>(72).unwrap();
        let mut output = stream.clone_htod(&[123.0_f32]).unwrap();
        assert!(matches!(
            kernel.execute(weight, &short, &mut output),
            Err(CudaQuantizedKernelError::InputLength { .. })
        ));
        let mut long_output = stream.alloc_zeros::<f32>(2).unwrap();
        assert!(matches!(
            kernel.execute(weight, &input, &mut long_output),
            Err(CudaQuantizedKernelError::OutputLength { .. })
        ));
        for (dimensions, value_type, bytes) in [
            (vec![256], 12, 144),
            (vec![257, 1], 12, 144),
            (vec![256, 1], 12, 143),
            (vec![256, 1], 8, 144),
        ] {
            let spec = WeightTensorSpec::new("invalid", dimensions, DataType::F32).unwrap();
            let mut invalid_store = CudaWeightStore::new(stream.clone());
            invalid_store
                .materialize_quantized(
                    spec,
                    value_type,
                    bytes,
                    &mut Cursor::new(vec![0_u8; usize::try_from(bytes).unwrap()]),
                )
                .unwrap();
            assert!(matches!(
                kernel.execute(
                    invalid_store.quantized_tensor("invalid").unwrap(),
                    &input,
                    &mut output
                ),
                Err(CudaQuantizedKernelError::InvalidWeight(_)
                    | CudaQuantizedKernelError::UnsupportedValueType { .. })
            ));
        }
        assert_eq!(stream.clone_dtoh(&output).unwrap(), [123.0]);
    }

    #[test]
    #[ignore = "requires CUDA GPU and NVRTC"]
    fn q4_k_batch_matches_per_member_host_arithmetic() {
        const MEMBERS: usize = 3;
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let kernel = CudaQ4KQ8_1Gemv::new(stream.clone()).unwrap();
        let quantizer = CudaQ8_1Quantizer::new(stream.clone()).unwrap();
        let blocks = fixture();
        let encoded: Vec<u8> = blocks.iter().flat_map(|block| block.encoded).collect();
        let spec =
            WeightTensorSpec::new("q4", vec![INPUTS as u64, ROWS as u64], DataType::F32).unwrap();
        let mut store = CudaWeightStore::new(stream.clone());
        store
            .materialize_quantized(spec, 12, encoded.len() as u64, &mut Cursor::new(encoded))
            .unwrap();
        let weight = store.quantized_tensor("q4").unwrap();
        // Batch-major activations with distinct per-member content: exact,
        // lossy, and zero. The quantizer packs linear block streams, so one
        // call covers the whole batch.
        let member_inputs = [activations(true), activations(false), vec![0.0; INPUTS]];
        let batched: Vec<f32> = member_inputs.concat();
        let batched_device = stream.clone_htod(&batched).unwrap();
        let mut packed_device = stream
            .alloc_zeros::<u32>(MEMBERS * INPUTS / 32 * 9)
            .unwrap();
        quantizer
            .execute(&batched_device, &mut packed_device)
            .unwrap();
        let packed = stream.clone_dtoh(&packed_device).unwrap();
        let mut output = stream.alloc_zeros::<f32>(MEMBERS * ROWS).unwrap();
        kernel
            .execute_batch(weight, &packed_device, &mut output, MEMBERS)
            .unwrap();
        let actual = stream.clone_dtoh(&output).unwrap();
        for (member, input) in member_inputs.iter().enumerate() {
            let member_packed = pack_host(input);
            assert_eq!(
                packed[member * INPUTS / 32 * 9..(member + 1) * INPUTS / 32 * 9],
                member_packed
            );
            let expected = reference(&blocks, input, &member_packed);
            for (row, expected) in expected.iter().enumerate() {
                let value = actual[member * ROWS + row];
                let rounding =
                    expected.arithmetic_magnitude * f64::from(f32::EPSILON) * 32.0 + 1e-5;
                assert!(
                    (f64::from(value) - expected.packed_output).abs() <= rounding,
                    "member {member} row {row}: packed output {value} vs {}",
                    expected.packed_output
                );
                assert!(
                    (f64::from(value) - expected.float_output).abs()
                        <= expected.packing_error_bound + rounding
                );
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU and NVRTC"]
    fn q4_k_batch_rejects_invalid_storage() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let kernel = CudaQ4KQ8_1Gemv::new(stream.clone()).unwrap();
        let spec = WeightTensorSpec::new("q4", vec![256, 1], DataType::F32).unwrap();
        let mut store = CudaWeightStore::new(stream.clone());
        store
            .materialize_quantized(spec, 12, 144, &mut Cursor::new(fixture()[0].encoded))
            .unwrap();
        let weight = store.quantized_tensor("q4").unwrap();
        // One member packs 256 inputs into 72 words with one output row.
        let input = stream.alloc_zeros::<u32>(2 * 72).unwrap();
        let mut output = stream.clone_htod(&[1.0_f32, 2.0]).unwrap();
        assert!(matches!(
            kernel.execute_batch(weight, &input, &mut output, 0),
            Err(CudaQuantizedKernelError::InputLength { .. })
        ));
        assert!(matches!(
            kernel.execute_batch(weight, &input, &mut output, 9),
            Err(CudaQuantizedKernelError::InvalidWeight(_))
        ));
        let short = stream.alloc_zeros::<u32>(2 * 72 - 1).unwrap();
        assert!(matches!(
            kernel.execute_batch(weight, &short, &mut output, 2),
            Err(CudaQuantizedKernelError::InputLength { .. })
        ));
        let mut long_output = stream.alloc_zeros::<f32>(3).unwrap();
        assert!(matches!(
            kernel.execute_batch(weight, &input, &mut long_output, 2),
            Err(CudaQuantizedKernelError::OutputLength { .. })
        ));
        assert_eq!(stream.clone_dtoh(&output).unwrap(), [1.0, 2.0]);
    }
}
