//! Kernel-level A/B for K-quant float warp GEMV vs experimental integer-dot.
//!
//! Compares the qualified float warp path against `quantize_q8_1` +
//! integer-dot GEMV on synthetic weights at representative Qwen shapes,
//! for each implemented (`Q4_K`, `Q5_K`) family. This measures kernel time
//! only; it does not establish model parity or serving throughput. Float
//! execution remains the default.
//!
//! ```text
//! cargo run --release -p engine-nvidia --features cuda \
//!   --example q4_kernel_bench -- --iters=100
//! ```

use std::io::Cursor;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaSlice};
use engine_core::{DataType, WeightTensorSpec};
use engine_nvidia::{
    CudaQ4KGemv, CudaQ4KQ8_1Gemv, CudaQ5KGemv, CudaQ5KQ8_1Gemv, CudaQ8_1Quantizer,
    CudaQuantizedKernelError, CudaQuantizedWeight, CudaWeightStore,
};

const SHAPES: [(usize, usize); 3] = [(5120, 5120), (5120, 17_408), (17_408, 5120)];
const WARMUP: usize = 10;

#[derive(Clone, Copy)]
enum Family {
    Q4K,
    Q5K,
}

impl Family {
    const fn tag(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
        }
    }

    const fn tensor_name(self) -> &'static str {
        match self {
            Self::Q4K => "q4",
            Self::Q5K => "q5",
        }
    }

    const fn value_type(self) -> u32 {
        match self {
            Self::Q4K => 12,
            Self::Q5K => 13,
        }
    }

    const fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q5K => 176,
        }
    }

    fn encode(self, inputs: usize, rows: usize) -> Vec<u8> {
        match self {
            Self::Q4K => synthetic_q4_k(inputs, rows),
            Self::Q5K => synthetic_q5_k(inputs, rows),
        }
    }
}

enum FloatKernels {
    Q4K(CudaQ4KGemv),
    Q5K(CudaQ5KGemv),
}

impl FloatKernels {
    fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        match self {
            Self::Q4K(kernel) => kernel.execute_warp(weight, input, output),
            Self::Q5K(kernel) => kernel.execute_warp(weight, input, output),
        }
    }
}

enum IntKernels {
    Q4K(CudaQ4KQ8_1Gemv),
    Q5K(CudaQ5KQ8_1Gemv),
}

impl IntKernels {
    fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        match self {
            Self::Q4K(kernel) => kernel.execute(weight, input, output),
            Self::Q5K(kernel) => kernel.execute(weight, input, output),
        }
    }
}

#[allow(clippy::too_many_lines, reason = "one linear bench script")]
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let iters: usize = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--iters="))
        .map_or(100, |value| {
            value.parse().expect("--iters expects a number")
        });

    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let quantizer = CudaQ8_1Quantizer::new(stream.clone()).expect("Q8_1 quantizer");

    println!(
        "family shape(KxN)  path            median      mean        min      weight-GB/s  max-rel-diff"
    );
    for family in [Family::Q4K, Family::Q5K] {
        let float_gemv = match family {
            Family::Q4K => FloatKernels::Q4K(
                CudaQ4KGemv::from_context(&context, stream.clone()).expect("float kernels"),
            ),
            Family::Q5K => FloatKernels::Q5K(
                CudaQ5KGemv::from_context(&context, stream.clone()).expect("float kernels"),
            ),
        };
        let int_gemv = match family {
            Family::Q4K => {
                IntKernels::Q4K(CudaQ4KQ8_1Gemv::new(stream.clone()).expect("int kernel"))
            }
            Family::Q5K => {
                IntKernels::Q5K(CudaQ5KQ8_1Gemv::new(stream.clone()).expect("int kernel"))
            }
        };
        for (inputs, rows) in SHAPES {
            let encoded = family.encode(inputs, rows);
            let spec = WeightTensorSpec::new(
                family.tensor_name(),
                vec![inputs as u64, rows as u64],
                DataType::F32,
            )
            .expect("weight spec");
            let mut store = CudaWeightStore::new(stream.clone());
            store
                .materialize_quantized(
                    spec,
                    family.value_type(),
                    encoded.len() as u64,
                    &mut Cursor::new(encoded),
                )
                .expect("stage weight");
            let weight = store
                .quantized_tensor(family.tensor_name())
                .expect("staged weight");

            let host_input = synthetic_activations(inputs);
            let input_f32 = stream.clone_htod(&host_input).expect("upload f32");
            let mut packed = stream
                .alloc_zeros::<u32>(inputs / 32 * 9)
                .expect("packed storage");
            let mut out_float = stream.alloc_zeros::<f32>(rows).expect("float output");
            let mut out_int = stream.alloc_zeros::<f32>(rows).expect("int output");

            // Warmup and numeric sanity.
            for _ in 0..WARMUP {
                float_gemv
                    .execute_warp(weight, &input_f32, &mut out_float)
                    .expect("float warmup");
                quantizer
                    .execute(&input_f32, &mut packed)
                    .expect("quant warmup");
                int_gemv
                    .execute(weight, &packed, &mut out_int)
                    .expect("int warmup");
            }
            stream.synchronize().expect("warmup sync");
            let reference = stream.clone_dtoh(&out_float).expect("read float");
            let candidate = stream.clone_dtoh(&out_int).expect("read int");
            let max_abs = reference
                .iter()
                .zip(candidate.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let max_ref = reference
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let max_diff = max_abs / max_ref.max(1.0e-6);

            let float_times = time_iters(iters, || {
                float_gemv
                    .execute_warp(weight, &input_f32, &mut out_float)
                    .expect("float bench");
                stream.synchronize().expect("float sync");
            });
            let int_times = time_iters(iters, || {
                quantizer
                    .execute(&input_f32, &mut packed)
                    .expect("quant bench");
                int_gemv
                    .execute(weight, &packed, &mut out_int)
                    .expect("int bench");
                stream.synchronize().expect("int sync");
            });
            let quant_times = time_iters(iters, || {
                quantizer
                    .execute(&input_f32, &mut packed)
                    .expect("quant-only bench");
                stream.synchronize().expect("quant sync");
            });

            let weight_bytes = rows * inputs / 256 * family.block_bytes();
            let int_label = format!("q8_1+{}-dp4a", family.tag().to_lowercase());
            report(
                family.tag(),
                inputs,
                rows,
                "float-warp",
                &float_times,
                weight_bytes,
                max_diff,
            );
            report(
                family.tag(),
                inputs,
                rows,
                &int_label,
                &int_times,
                weight_bytes,
                max_diff,
            );
            report(
                family.tag(),
                inputs,
                rows,
                "q8_1-only",
                &quant_times,
                weight_bytes,
                max_diff,
            );
        }
    }
}

fn report(
    family: &str,
    inputs: usize,
    rows: usize,
    label: &str,
    times: &[Duration],
    weight_bytes: usize,
    max_diff: f32,
) {
    let mut sorted: Vec<Duration> = times.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let mean = sorted.iter().sum::<Duration>() / u32::try_from(sorted.len()).unwrap_or(1);
    let min = sorted[0];
    #[allow(
        clippy::cast_precision_loss,
        reason = "weight bytes fit exactly in f64 for bench shapes"
    )]
    let gbps = weight_bytes as f64 / median.as_secs_f64() / 1.0e9;
    println!(
        "{family:<6} {inputs:>5}x{rows:<5}  {label:<14} {median:>8.3?} {mean:>8.3?} {min:>8.3?}  {gbps:>10.1}  {max_diff:.4}",
    );
}

fn time_iters(iters: usize, mut step: impl FnMut()) -> Vec<Duration> {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        step();
        times.push(start.elapsed());
    }
    times
}

/// Deterministic synthetic `Q4_K` weights: fixed scales/minima with a
/// repeating 4-bit quant pattern. Timing only; not a parity fixture.
fn synthetic_q4_k(inputs: usize, rows: usize) -> Vec<u8> {
    const D_BITS: u16 = 0x2C00;
    const MIN_BITS: u16 = 0x2800;
    let blocks_per_row = inputs / 256;
    let mut out = Vec::with_capacity(rows * blocks_per_row * 144);
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let seed = row * blocks_per_row + block;
            out.extend_from_slice(&D_BITS.to_le_bytes());
            out.extend_from_slice(&MIN_BITS.to_le_bytes());
            let scales = [32_u8, 40, 24, 48, 16, 56, 8, 60];
            let minima = [4_u8, 12, 20, 28, 36, 44, 52, 60];
            for group in 0..4 {
                out.push(scales[group] | ((scales[group + 4] >> 4) << 6));
            }
            for group in 0..4 {
                out.push(minima[group] | ((minima[group + 4] >> 4) << 6));
            }
            for group in 0..4 {
                out.push((scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4));
            }
            let mut payload = [0_u8; 128];
            for group in 0..8 {
                for index in 0..32 {
                    let q = u8::try_from((index * 7 + seed * 3 + index / 32) % 16).unwrap();
                    payload[(group / 2) * 32 + index] |= q << ((group % 2) * 4);
                }
            }
            out.extend_from_slice(&payload);
        }
    }
    out
}

/// Deterministic synthetic `Q5_K` weights: fixed scales/minima with a
/// repeating 5-bit quant pattern split across the low-nibble payload and
/// the high-bit plane. Timing only; not a parity fixture.
fn synthetic_q5_k(inputs: usize, rows: usize) -> Vec<u8> {
    const D_BITS: u16 = 0x2C00;
    const MIN_BITS: u16 = 0x2800;
    let blocks_per_row = inputs / 256;
    let mut out = Vec::with_capacity(rows * blocks_per_row * 176);
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let seed = row * blocks_per_row + block;
            out.extend_from_slice(&D_BITS.to_le_bytes());
            out.extend_from_slice(&MIN_BITS.to_le_bytes());
            let scales = [32_u8, 40, 24, 48, 16, 56, 8, 60];
            let minima = [4_u8, 12, 20, 28, 36, 44, 52, 60];
            for group in 0..4 {
                out.push(scales[group] | ((scales[group + 4] >> 4) << 6));
            }
            for group in 0..4 {
                out.push(minima[group] | ((minima[group + 4] >> 4) << 6));
            }
            for group in 0..4 {
                out.push((scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4));
            }
            let mut high = [0_u8; 32];
            let mut payload = [0_u8; 128];
            for group in 0..8 {
                for index in 0..32 {
                    let q = u8::try_from((index * 7 + seed * 3 + index / 32) % 32).unwrap();
                    payload[(group / 2) * 32 + index] |= (q & 15) << ((group % 2) * 4);
                    high[index] |= ((q >> 4) & 1) << group;
                }
            }
            out.extend_from_slice(&high);
            out.extend_from_slice(&payload);
        }
    }
    out
}

/// Deterministic pseudo-random activations in `[-1, 1]`.
fn synthetic_activations(len: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = (state >> 11) & 0xFFFF_FFFF;
            #[allow(
                clippy::cast_precision_loss,
                reason = "synthetic bench input; u32 fits in f32 with enough resolution"
            )]
            let unit = bits as f32 / u32::MAX as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}
