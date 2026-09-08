use std::sync::LazyLock;

const INPUTS: usize = 768;
const ROWS: usize = 5;

struct Block {
    encoded: [u8; 210],
    quants: [i8; 256],
    scales: [i8; 16],
    d: f64,
}

fn fixture() -> Vec<Block> {
    (0..ROWS * (INPUTS / 256))
        .map(|seed| {
            let scales = [
                3, -5, 11, -13, 19, -23, 29, -31, 7, -9, 15, -17, 21, -25, 27, -2,
            ];
            let d_bits = [0x2c00_u16, 0x3000, 0x3400][seed % 3];
            let mut encoded = [0_u8; 210];
            let quants = std::array::from_fn(|index| {
                i8::try_from((index * 7 + seed * 3 + index / 32) % 64).unwrap() - 32
            });
            for group in 0..8 {
                let low_base = (group / 4) * 64 + (group % 2) * 32;
                let low_shift = if group % 4 < 2 { 0 } else { 4 };
                let high_base = 128 + (group / 4) * 32;
                let high_shift = (group % 4) * 2;
                for index in 0..32 {
                    let unsigned = u8::try_from(quants[group * 32 + index] + 32).unwrap();
                    encoded[low_base + index] |= (unsigned & 15) << low_shift;
                    encoded[high_base + index] |= ((unsigned >> 4) & 3) << high_shift;
                }
            }
            for (index, scale) in scales.iter().enumerate() {
                encoded[192 + index] = i8::to_ne_bytes(*scale)[0];
            }
            encoded[208..210].copy_from_slice(&d_bits.to_le_bytes());
            Block {
                encoded,
                quants,
                scales,
                d: decode_half(d_bits),
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
            // The Q6_K kernel ignores the sum word, but packing must still
            // match the device quantizer exactly.
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
                    let q: Vec<i8> = words[1..]
                        .iter()
                        .flat_map(|word| word.to_le_bytes().map(|byte| i8::from_ne_bytes([byte])))
                        .collect();
                    for half in 0..2 {
                        let a = block.d * f64::from(block.scales[group * 2 + half]);
                        let mut integer_dot = 0_i32;
                        let mut activation_sum = 0_i32;
                        for index in 0..16 {
                            let position = half * 16 + index;
                            let weight_q = block.quants[group * 32 + position];
                            let activation_q = q[position];
                            // The kernel dots unsigned (q + 32) bytes and
                            // removes the bias through the packed half sums.
                            integer_dot += i32::from(weight_q + 32) * i32::from(activation_q);
                            activation_sum += i32::from(activation_q);
                            let x = f64::from(input[input_offset + position]);
                            result.float_output += a * f64::from(weight_q) * x;
                            result.packing_error_bound += a.abs()
                                * f64::from(weight_q.abs())
                                * (scale * f64::from(activation_q) - x).abs();
                            result.arithmetic_magnitude +=
                                (a * f64::from(weight_q) * scale * f64::from(activation_q)).abs();
                        }
                        result.packed_output +=
                            a * scale * f64::from(integer_dot - 32 * activation_sum);
                        result.arithmetic_magnitude +=
                            (a * scale * f64::from(integer_dot - 32 * activation_sum)).abs();
                    }
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
        let decoded = engine_gguf::dequantize_block(14, &block.encoded).unwrap();
        for (index, value) in decoded.iter().enumerate() {
            let half = index % 32 / 16;
            assert_eq!(
                f64::from(*value),
                block.d
                    * f64::from(block.scales[(index / 32) * 2 + half])
                    * f64::from(block.quants[index])
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
        CudaQ6KQ8_1Gemv, CudaQ8_1Quantizer, CudaQuantizedKernelError, CudaWeightStore,
    };
    use std::io::Cursor;

    #[test]
    #[ignore = "requires CUDA GPU and NVRTC"]
    fn q6_k_dp4a_matches_packed_host_arithmetic_and_float_error_bound() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let kernel = CudaQ6KQ8_1Gemv::new(stream.clone()).unwrap();
        let quantizer = CudaQ8_1Quantizer::new(stream.clone()).unwrap();
        let blocks = fixture();
        let encoded: Vec<u8> = blocks.iter().flat_map(|block| block.encoded).collect();
        let spec =
            WeightTensorSpec::new("q6", vec![INPUTS as u64, ROWS as u64], DataType::F32).unwrap();
        let mut store = CudaWeightStore::new(stream.clone());
        store
            .materialize_quantized(spec, 14, encoded.len() as u64, &mut Cursor::new(encoded))
            .unwrap();
        let weight = store.quantized_tensor("q6").unwrap();
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
    fn q6_k_dp4a_rejects_invalid_storage() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let kernel = CudaQ6KQ8_1Gemv::new(stream.clone()).unwrap();
        let spec = WeightTensorSpec::new("q6", vec![256, 1], DataType::F32).unwrap();
        let mut store = CudaWeightStore::new(stream.clone());
        store
            .materialize_quantized(spec, 14, 210, &mut Cursor::new(fixture()[0].encoded))
            .unwrap();
        let weight = store.quantized_tensor("q6").unwrap();
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
            (vec![256], 14, 210),
            (vec![257, 1], 14, 210),
            (vec![256, 1], 14, 209),
            (vec![256, 1], 8, 210),
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
}
