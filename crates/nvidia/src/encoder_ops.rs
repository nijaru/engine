//! Transformer-encoder primitives for non-autoregressive model execution.
//!
//! These are the device kernels a bidirectional encoder needs and a decoder path
//! does not: token/position/type embedding summation, `LayerNorm` (mean-centred,
//! unlike the decoder's `RMSNorm`), exact-erf `GELU`, masked scaled-dot-product
//! attention with a stable softmax, row bias, residual add and tanh. They are
//! deliberately separate from the Qwen decoder ops: an encoder and a decoder share
//! arithmetic, not ownership, and a single module keeps the encoder path's
//! numerical behavior independently reviewable.
//!
//! Projections are not here. They are dense GEMMs and use cuBLAS, which keeps this
//! module to elementwise and per-row reductions.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

/// Errors returned by the encoder primitives.
#[derive(Debug)]
pub enum CudaEncoderKernelError {
    Driver(String),
    Nvrtc(String),
    ContextMismatch,
    EmptyInput,
    InputLength { expected: usize, actual: usize },
    WeightLength { expected: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
    InvalidEpsilon,
    InvalidSequence,
    ShapeOverflow,
}

impl fmt::Display for CudaEncoderKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA encoder-kernel driver error: {message}"),
            Self::Nvrtc(message) => write!(f, "CUDA encoder-kernel NVRTC error: {message}"),
            Self::ContextMismatch => {
                f.write_str("encoder-kernel buffers and stream belong to different CUDA contexts")
            }
            Self::EmptyInput => f.write_str("encoder-kernel input must not be empty"),
            Self::InputLength { expected, actual } => write!(
                f,
                "encoder-kernel input has {actual} values, expected {expected}"
            ),
            Self::WeightLength { expected, actual } => write!(
                f,
                "encoder-kernel weight has {actual} values, expected {expected}"
            ),
            Self::OutputLength { expected, actual } => write!(
                f,
                "encoder-kernel output has {actual} values, expected {expected}"
            ),
            Self::InvalidEpsilon => f.write_str("LayerNorm epsilon must be finite and positive"),
            Self::InvalidSequence => {
                f.write_str("encoder-kernel sequence length must be positive and fit one block")
            }
            Self::ShapeOverflow => f.write_str("encoder-kernel shape does not fit CUDA arguments"),
        }
    }
}

impl std::error::Error for CudaEncoderKernelError {}

/// Threads per block for row-wise kernels; also the attention block size when the
/// head width is smaller, since the score loop is distributed over the block.
const ROW_THREADS: u32 = 256;
/// Largest sequence length one attention block can hold in shared memory.
const MAX_ATTENTION_SEQUENCE: usize = 4096;
/// Bytes in one `f32`, for shared-memory sizing without importing a trait.
const F32_BYTES: u32 = 4;

const ENCODER_OPS_SOURCE: &str = r#"
// Block-wide sum reduction over the first `count` scratch slots, leaving the result
// in scratch[0]. Every thread must reach both barriers.
__device__ float block_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

// Block-wide maximum over the same scratch layout.
__device__ float block_maximum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

// Word + position + token-type embedding sum for every position of one sequence.
// `types` may select any row of the token-type table, including the zero row a
// single-segment input uses.
extern "C" __global__ void embedding_sum(
    const unsigned int* __restrict__ tokens,
    const unsigned int* __restrict__ types,
    const float* __restrict__ word,
    const float* __restrict__ position,
    const float* __restrict__ token_type,
    const int sequence,
    const int hidden,
    float* __restrict__ out
) {
    const int row = blockIdx.x;
    if (row >= sequence) {
        return;
    }
    const int word_offset = (int)tokens[row] * hidden;
    const int type_offset = (int)types[row] * hidden;
    const int position_offset = row * hidden;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        out[position_offset + index] =
            word[word_offset + index] + position[position_offset + index] + token_type[type_offset + index];
    }
}

// Mean-centred LayerNorm over one row per block, matching the biased-variance
// definition an encoder checkpoint is trained with. The variance is accumulated
// around the mean rather than from raw moments, which keeps a wide row with a large
// offset from losing its variance to cancellation.
extern "C" __global__ void layer_norm_rows(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const int hidden,
    const float epsilon,
    float* __restrict__ out
) {
    extern __shared__ float scratch[];
    __shared__ float mean_shared;
    __shared__ float scale_shared;

    const int row = blockIdx.x;
    const float* source = input + (long)row * hidden;
    float* destination = out + (long)row * hidden;

    float sum = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        sum += source[index];
    }
    sum = block_sum(scratch, sum);
    if (threadIdx.x == 0) {
        mean_shared = sum / (float)hidden;
    }
    __syncthreads();

    float square_deviation = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        const float deviation = source[index] - mean_shared;
        square_deviation = fmaf(deviation, deviation, square_deviation);
    }
    square_deviation = block_sum(scratch, square_deviation);
    if (threadIdx.x == 0) {
        scale_shared = rsqrtf(square_deviation / (float)hidden + epsilon);
    }
    __syncthreads();

    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        destination[index] =
            (source[index] - mean_shared) * scale_shared * gamma[index] + beta[index];
    }
}

// Exact (erf-based) GELU, the activation an encoder checkpoint trained with
// hidden_act=gelu expects.
extern "C" __global__ void gelu_erf(float* __restrict__ data, const int length) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < length; index += gridDim.x * blockDim.x) {
        const float value = data[index];
        data[index] = 0.5f * value * (1.0f + erff(value * 0.7071067811865476f));
    }
}

// Add one bias vector to every row of a matrix, as a dense projection requires.
extern "C" __global__ void add_row_bias(
    float* __restrict__ data,
    const float* __restrict__ bias,
    const int width
) {
    const int row = blockIdx.x;
    float* target = data + (long)row * width;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        target[index] += bias[index];
    }
}

// Residual add used by both encoder sublayers.
extern "C" __global__ void add_in_place(
    float* __restrict__ destination,
    const float* __restrict__ source,
    const int length
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < length; index += gridDim.x * blockDim.x) {
        destination[index] += source[index];
    }
}

// tanh applied elementwise, for a pooler head.
extern "C" __global__ void tanh_apply(float* __restrict__ data, const int length) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < length; index += gridDim.x * blockDim.x) {
        data[index] = tanhf(data[index]);
    }
}

// Bidirectional scaled dot-product attention for one (query position, head) per
// block. `out` and `q`/`k`/`v` share the [sequence, hidden] layout, so a head is
// the contiguous slice [head * head_width, (head + 1) * head_width) of each row.
//
// Shared memory holds one probability per key position plus the reduction scratch
// `block_sum`/`block_maximum` expect.
//
// Masked keys contribute nothing: their score is -infinity, which becomes an exact
// zero probability after the maximum subtraction. A query whose keys are all masked
// has no normalizer, so it is written as zeros rather than NaN, matching a reference
// that never reads a padding row's output.
extern "C" __global__ void attention_context(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const int* __restrict__ mask,
    const int hidden,
    const int heads,
    const int head_width,
    const float scale,
    float* __restrict__ out
) {
    extern __shared__ float shared[];
    const int sequence = gridDim.x;
    float* probabilities = shared;
    float* scratch = shared + sequence;

    const int query_position = blockIdx.x;
    const int head = blockIdx.y;
    const int offset = head * head_width;
    const float* query_row = query + (long)query_position * hidden + offset;

    float maximum = -INFINITY;
    for (int position = threadIdx.x; position < sequence; position += blockDim.x) {
        const float* key_row = key + (long)position * hidden + offset;
        float score = 0.0f;
        for (int index = 0; index < head_width; ++index) {
            score = fmaf(query_row[index], key_row[index], score);
        }
        score *= scale;
        probabilities[position] = mask[position] != 0 ? score : -INFINITY;
        maximum = fmaxf(maximum, probabilities[position]);
    }
    maximum = block_maximum(scratch, maximum);

    float total = 0.0f;
    for (int position = threadIdx.x; position < sequence; position += blockDim.x) {
        const float score = probabilities[position];
        const float weight = score == -INFINITY ? 0.0f : expf(score - maximum);
        probabilities[position] = weight;
        total += weight;
    }
    total = block_sum(scratch, total);
    const float normalizer = total > 0.0f ? 1.0f / total : 0.0f;

    for (int index = threadIdx.x; index < head_width; index += blockDim.x) {
        float accumulated = 0.0f;
        for (int position = 0; position < sequence; ++position) {
            accumulated = fmaf(probabilities[position], value[(long)position * hidden + offset + index], accumulated);
        }
        out[(long)query_position * hidden + offset + index] = accumulated * normalizer;
    }
}
"#;

/// Compiled encoder primitives bound to one CUDA context and stream.
pub struct CudaEncoderOps {
    stream: Arc<CudaStream>,
    embedding_sum: CudaFunction,
    layer_norm_rows: CudaFunction,
    gelu_erf: CudaFunction,
    add_row_bias: CudaFunction,
    add_in_place: CudaFunction,
    tanh_apply: CudaFunction,
    attention_context: CudaFunction,
}

impl CudaEncoderOps {
    /// Compile and load the encoder primitives on a new context.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] when CUDA, NVRTC or module loading fails.
    pub fn new(device_index: usize) -> Result<Self, CudaEncoderKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the encoder primitives on an existing context/stream pair.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaEncoderKernelError> {
        if context.as_ref() != stream.context().as_ref() {
            return Err(CudaEncoderKernelError::ContextMismatch);
        }
        let ptx = compile_ptx(ENCODER_OPS_SOURCE)
            .map_err(|error| CudaEncoderKernelError::Nvrtc(error.to_string()))?;
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        let load = |name: &str| -> Result<CudaFunction, CudaEncoderKernelError> {
            module
                .load_function(name)
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))
        };
        Ok(Self {
            stream,
            embedding_sum: load("embedding_sum")?,
            layer_norm_rows: load("layer_norm_rows")?,
            gelu_erf: load("gelu_erf")?,
            add_row_bias: load("add_row_bias")?,
            add_in_place: load("add_in_place")?,
            tanh_apply: load("tanh_apply")?,
            attention_context: load("attention_context")?,
        })
    }

    /// The stream these primitives enqueue on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Sum the word, position and token-type embedding rows for one sequence.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for mismatched buffer lengths, an empty
    /// sequence, or a shape that does not fit CUDA launch arguments.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per embedding table keeps the launch explicit"
    )]
    pub fn embedding_sum(
        &self,
        tokens: &CudaSlice<u32>,
        types: &CudaSlice<u32>,
        word: &CudaSlice<f32>,
        position: &CudaSlice<f32>,
        token_type: &CudaSlice<f32>,
        hidden: usize,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaEncoderKernelError> {
        self.check_contexts(tokens, types)?;
        let sequence = tokens.len();
        if sequence == 0 {
            return Err(CudaEncoderKernelError::EmptyInput);
        }
        if types.len() != sequence {
            return Err(CudaEncoderKernelError::InputLength {
                expected: sequence,
                actual: types.len(),
            });
        }
        if hidden == 0
            || !word.len().is_multiple_of(hidden)
            || !position.len().is_multiple_of(hidden)
        {
            return Err(CudaEncoderKernelError::WeightLength {
                expected: hidden,
                actual: word.len().min(position.len()),
            });
        }
        let expected = sequence.saturating_mul(hidden);
        if output.len() != expected {
            return Err(CudaEncoderKernelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        let sequence_u32 =
            u32::try_from(sequence).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        let hidden_i32 =
            i32::try_from(hidden).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        // Safety: cudarc owns every slice, lengths are validated above, and the
        // stream keeps the buffers alive until the kernel completes.
        unsafe {
            self.stream
                .launch_builder(&self.embedding_sum)
                .arg(tokens)
                .arg(types)
                .arg(word)
                .arg(position)
                .arg(token_type)
                .arg(&sequence_u32)
                .arg(&hidden_i32)
                .arg(output)
                .launch(LaunchConfig {
                    grid_dim: (sequence_u32, 1, 1),
                    block_dim: (
                        ROW_THREADS.min(u32::try_from(hidden).unwrap_or(ROW_THREADS)),
                        1,
                        1,
                    ),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply mean-centred `LayerNorm` to every row of `input`.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for mismatched lengths or a non-positive
    /// epsilon.
    pub fn layer_norm_rows(
        &self,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        hidden: usize,
        epsilon: f32,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaEncoderKernelError> {
        self.check_contexts(input, gamma)?;
        if hidden == 0 || !input.len().is_multiple_of(hidden) {
            return Err(CudaEncoderKernelError::WeightLength {
                expected: hidden,
                actual: input.len(),
            });
        }
        if gamma.len() != hidden || beta.len() != hidden {
            return Err(CudaEncoderKernelError::WeightLength {
                expected: hidden,
                actual: gamma.len().min(beta.len()),
            });
        }
        if output.len() != input.len() {
            return Err(CudaEncoderKernelError::OutputLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaEncoderKernelError::InvalidEpsilon);
        }
        let rows = input.len() / hidden;
        let rows_u32 = u32::try_from(rows).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        let hidden_i32 =
            i32::try_from(hidden).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        // Safety: see `embedding_sum`; the scratch size matches the two reductions
        // the kernel performs over `ROW_THREADS` threads.
        unsafe {
            self.stream
                .launch_builder(&self.layer_norm_rows)
                .arg(input)
                .arg(gamma)
                .arg(beta)
                .arg(&hidden_i32)
                .arg(&epsilon)
                .arg(output)
                .launch(LaunchConfig {
                    grid_dim: (rows_u32, 1, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: ROW_THREADS * F32_BYTES,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply exact GELU in place.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for an empty buffer.
    pub fn gelu_erf(&self, data: &mut CudaSlice<f32>) -> Result<(), CudaEncoderKernelError> {
        if data.is_empty() {
            return Err(CudaEncoderKernelError::EmptyInput);
        }
        let (grid, block, length) = elementwise_geometry(data.len())?;
        // Safety: see `embedding_sum`; `length` is the slice's element count.
        unsafe {
            self.stream
                .launch_builder(&self.gelu_erf)
                .arg(data)
                .arg(&length)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply tanh in place.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for an empty buffer.
    pub fn tanh_apply(&self, data: &mut CudaSlice<f32>) -> Result<(), CudaEncoderKernelError> {
        if data.is_empty() {
            return Err(CudaEncoderKernelError::EmptyInput);
        }
        let (grid, block, length) = elementwise_geometry(data.len())?;
        // Safety: see `embedding_sum`; `length` is the slice's element count.
        unsafe {
            self.stream
                .launch_builder(&self.tanh_apply)
                .arg(data)
                .arg(&length)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Add `bias` to every row of `data` in place.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for a zero width or a length that is not
    /// a whole number of rows.
    pub fn add_row_bias(
        &self,
        data: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
    ) -> Result<(), CudaEncoderKernelError> {
        self.check_contexts(data, bias)?;
        let width = bias.len();
        if width == 0 || data.is_empty() || !data.len().is_multiple_of(width) {
            return Err(CudaEncoderKernelError::WeightLength {
                expected: width,
                actual: data.len(),
            });
        }
        let rows =
            u32::try_from(data.len() / width).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        let width_i32 = i32::try_from(width).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        // Safety: see `embedding_sum`; both slices are validated above.
        unsafe {
            self.stream
                .launch_builder(&self.add_row_bias)
                .arg(data)
                .arg(bias)
                .arg(&width_i32)
                .launch(LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (
                        ROW_THREADS.min(u32::try_from(width).unwrap_or(ROW_THREADS)),
                        1,
                        1,
                    ),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Add `source` into `destination` in place.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for mismatched or empty buffers.
    pub fn add_in_place(
        &self,
        destination: &mut CudaSlice<f32>,
        source: &CudaSlice<f32>,
    ) -> Result<(), CudaEncoderKernelError> {
        self.check_contexts(destination, source)?;
        if destination.is_empty() {
            return Err(CudaEncoderKernelError::EmptyInput);
        }
        if destination.len() != source.len() {
            return Err(CudaEncoderKernelError::InputLength {
                expected: destination.len(),
                actual: source.len(),
            });
        }
        let (grid, block, length) = elementwise_geometry(destination.len())?;
        // Safety: see `embedding_sum`; lengths match above.
        unsafe {
            self.stream
                .launch_builder(&self.add_in_place)
                .arg(destination)
                .arg(source)
                .arg(&length)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Compute bidirectional scaled dot-product attention over `[sequence, hidden]`
    /// query, key and value matrices.
    ///
    /// `mask` is one flag per key position; a zero flag removes that key from every
    /// query's attention. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    /// Returns [`CudaEncoderKernelError`] for inconsistent shapes or a sequence
    /// that does not fit one block's shared memory.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the kernel signature so the launch stays auditable"
    )]
    pub fn attention_context(
        &self,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        mask: &CudaSlice<i32>,
        hidden: usize,
        heads: usize,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaEncoderKernelError> {
        self.check_contexts(query, key)?;
        let sequence = mask.len();
        if sequence == 0 || sequence > MAX_ATTENTION_SEQUENCE {
            return Err(CudaEncoderKernelError::InvalidSequence);
        }
        if hidden == 0 || !hidden.is_multiple_of(heads) {
            return Err(CudaEncoderKernelError::WeightLength {
                expected: heads,
                actual: hidden,
            });
        }
        let head_width = hidden / heads;
        let expected = sequence.saturating_mul(hidden);
        for (name, actual) in [
            ("query", query.len()),
            ("key", key.len()),
            ("value", value.len()),
        ] {
            if actual != expected {
                let _ = name;
                return Err(CudaEncoderKernelError::InputLength { expected, actual });
            }
        }
        if output.len() != expected {
            return Err(CudaEncoderKernelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        let hidden_i32 =
            i32::try_from(hidden).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        let heads_u32 = u32::try_from(heads).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        let head_width_i32 =
            i32::try_from(head_width).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        let sequence_u32 =
            u32::try_from(sequence).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
        // Head widths are a validated launch dimension, so the conversion is exact
        // for every width a real encoder uses.
        #[allow(
            clippy::cast_precision_loss,
            reason = "head width is a validated small launch dimension"
        )]
        let scale = (head_width as f32).sqrt().recip();
        let block = ROW_THREADS.min(u32::try_from(head_width.max(heads)).unwrap_or(ROW_THREADS));
        // Safety: see `embedding_sum`; shapes are validated above and the shared
        // array holds exactly one probability per key position.
        unsafe {
            self.stream
                .launch_builder(&self.attention_context)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(mask)
                .arg(&hidden_i32)
                .arg(&heads_u32)
                .arg(&head_width_i32)
                .arg(&scale)
                .arg(output)
                .launch(LaunchConfig {
                    grid_dim: (sequence_u32, heads_u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: (sequence_u32 + block) * F32_BYTES,
                })
                .map_err(|error| CudaEncoderKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    fn check_contexts<A, B>(
        &self,
        left: &CudaSlice<A>,
        right: &CudaSlice<B>,
    ) -> Result<(), CudaEncoderKernelError> {
        if left.context().as_ref() != self.stream.context().as_ref()
            || right.context().as_ref() != self.stream.context().as_ref()
        {
            return Err(CudaEncoderKernelError::ContextMismatch);
        }
        Ok(())
    }
}

/// Grid, block and element count for a flat elementwise launch.
type ElementwiseGeometry = ((u32, u32, u32), (u32, u32, u32), i32);

fn elementwise_geometry(length: usize) -> Result<ElementwiseGeometry, CudaEncoderKernelError> {
    let length_i32 = i32::try_from(length).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
    let blocks = length.div_ceil(ROW_THREADS as usize).max(1);
    let blocks_u32 = u32::try_from(blocks).map_err(|_| CudaEncoderKernelError::ShapeOverflow)?;
    Ok(((blocks_u32, 1, 1), (ROW_THREADS, 1, 1), length_i32))
}
