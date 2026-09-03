use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

/// Errors returned by the reference-oriented Qwen elementwise primitives.
#[derive(Debug)]
pub enum CudaModelKernelError {
    Driver(String),
    Nvrtc(String),
    ContextMismatch,
    EmptyInput,
    InputLength { expected: usize, actual: usize },
    WeightLength { expected: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
    InvalidEpsilon,
    ShapeOverflow,
}

impl fmt::Display for CudaModelKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA model-kernel driver error: {message}"),
            Self::Nvrtc(message) => write!(f, "CUDA model-kernel NVRTC error: {message}"),
            Self::ContextMismatch => {
                f.write_str("model-kernel buffers and stream belong to different CUDA contexts")
            }
            Self::EmptyInput => f.write_str("model-kernel input must not be empty"),
            Self::InputLength { expected, actual } => {
                write!(
                    f,
                    "model-kernel input has {actual} values, expected {expected}"
                )
            }
            Self::WeightLength { expected, actual } => {
                write!(
                    f,
                    "model-kernel weight has {actual} values, expected {expected}"
                )
            }
            Self::OutputLength { expected, actual } => {
                write!(
                    f,
                    "model-kernel output has {actual} values, expected {expected}"
                )
            }
            Self::InvalidEpsilon => f.write_str("RMSNorm epsilon must be finite and positive"),
            Self::ShapeOverflow => f.write_str("model-kernel shape does not fit CUDA arguments"),
        }
    }
}

impl std::error::Error for CudaModelKernelError {}

const MODEL_OPS_SOURCE: &str = r#"
// F16 -> F32 bit conversion without cuda_fp16.h (NVRTC's default include
// path does not ship it): zero-extend to F32, rebias the exponent, and
// denormalize subnormals explicitly.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    const unsigned int sign = (unsigned int)(bits >> 15) & 1u;
    const unsigned int exponent = ((unsigned int)(bits >> 10) & 0x1fu);
    const unsigned int fraction = (unsigned int)(bits & 0x3ffu);
    if (exponent == 0u) {
        // Subnormal or zero: normalize via the smallest normal exponent.
        if (fraction == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        // Find the leading bit position of the fraction.
        int shift = -1;
        unsigned int value = fraction;
        while (value != 0u) {
            value >>= 1u;
            ++shift;
        }
        const int leading = shift; // position of the MSB, 0-based
        const unsigned int normalized = (fraction << (10 - leading)) & 0x3ffu;
        const int new_exponent = -14 - leading + 127;
        const unsigned int result = (sign << 31) | ((unsigned int)new_exponent << 23) | (normalized << 13);
        return __int_as_float(result);
    }
    if (exponent == 31u) {
        // Inf or NaN.
        const unsigned int result =
            (sign << 31) | 0x7f800000u | (fraction ? 0x7fc00000u : 0u);
        return __int_as_float(result);
    }
    const unsigned int result = (sign << 31)
        | ((exponent - 15u + 127u) << 23)
        | (fraction << 13);
    return __int_as_float(result);
}
extern "C" __global__ void rms_norm(
    const float* input,
    const float* weight,
    float* output,
    int length,
    float epsilon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float sum = 0.0f;
    for (int index = 0; index < length; ++index) {
        sum += input[index] * input[index];
    }
    const float inverse_norm = rsqrtf(sum / (float)length + epsilon);
    for (int index = 0; index < length; ++index) {
        output[index] = input[index] * inverse_norm * weight[index];
    }
}

extern "C" __global__ void argmax(
    const float* logits,
    unsigned int* output,
    int length
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float best_value = -3.402823466e+38f;
    unsigned int best_index = 0;
    for (int index = 0; index < length; ++index) {
        if (logits[index] > best_value) {
            best_value = logits[index];
            best_index = (unsigned int)index;
        }
    }
    output[0] = best_index;
}

extern "C" __global__ void silu_mul(
    const float* gate,
    const float* up,
    float* output,
    int length
) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= length) {
        return;
    }
    const float value = gate[index];
    output[index] = (value / (1.0f + expf(-value))) * up[index];
}

extern "C" __global__ void l2_norm(
    const float* input,
    float* output,
    int length,
    float epsilon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float sum = 0.0f;
    for (int index = 0; index < length; ++index) {
        sum += input[index] * input[index];
    }
    const float scale = 1.0f / fmaxf(sqrtf(sum), epsilon);
    for (int index = 0; index < length; ++index) {
        output[index] = input[index] * scale;
    }
}

extern "C" __global__ void gdn_scalar_gate(
    const float* alpha,
    const float* beta_raw,
    const float* dt_bias,
    const float* a,
    float* decay,
    float* beta,
    int heads
) {
    const int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) {
        return;
    }
    // Matches the pinned host reference: decay = exp(a * softplus(alpha +
    // dt_bias)) where a = -exp(A_log), and beta = sigmoid(beta_raw) where
    // beta_raw is the ssm_beta projection (dt_bias does not touch beta).
    const float softplus_arg = alpha[head] + dt_bias[head];
    const float softplus_value = softplus_arg > 20.0f
        ? softplus_arg
        : logf(1.0f + expf(softplus_arg));
    decay[head] = expf(a[head] * softplus_value);
    beta[head] = 1.0f / (1.0f + expf(-beta_raw[head]));
}
extern "C" __global__ void rope_neox(
    float* values,
    long long position,
    int head_count,
    int head_dim,
    int rot_dim_pairs,
    float base
) {
    // One thread per (head, pair). NEOX half-split: pair (p, p + rot/2)
    // over the first rot_dim dims of each head; the remaining dims pass
    // through. All heads share the scalar position.
    const int pair_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_pairs = head_count * rot_dim_pairs;
    if (pair_index >= total_pairs) {
        return;
    }
    const int head = pair_index / rot_dim_pairs;
    const int pair = pair_index % rot_dim_pairs;
    const float pos = (float)position;
    const int half = rot_dim_pairs;
    const float theta =
        pos * powf(base, -(2.0f * (float)pair) / (2.0f * (float)half));
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);
    const int offset = head * head_dim + pair;
    const float x0 = values[offset];
    const float x1 = values[offset + half];
    values[offset] = x0 * cos_theta - x1 * sin_theta;
    values[offset + half] = x0 * sin_theta + x1 * cos_theta;
}

extern "C" __global__ void attn_score_gqa(
    const float* q,
    const unsigned short* keys,
    const unsigned short* values,
    const float* gate_scratch,
    float* scores_scratch,
    float* output,
    int tokens,
    int scores_stride,
    int q_heads,
    int kv_heads,
    int head_dim
) {
    // One thread per q head; serial over cached tokens. Reads the F16 KV
    // cache laid out [token][kv_head][head_dim] and computes the
    // block-mapped GQA decode: scores, softmax, weighted V sum, then the
    // sigmoid gate. scores_scratch is a [q_heads][scores_stride] buffer;
    // only the first `tokens` entries per head are used.
    const int q_head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q_head >= q_heads) {
        return;
    }
    const int q_per_kv = q_heads / kv_heads;
    const int kv_head = q_head / q_per_kv;
    const float scale = rsqrtf((float)head_dim);

    float* scores = scores_scratch + q_head * scores_stride;
    for (int token = 0; token < tokens; ++token) {
        const unsigned short* key =
            keys + ((long long)token * kv_heads + kv_head) * head_dim;
        const float* q_vec = q + q_head * head_dim;
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q_vec[dim] * f16_bits_to_f32(key[dim]);
        }
        scores[token] = dot * scale;
    }
    float max_score = -3.402823466e+38f;
    for (int token = 0; token < tokens; ++token) {
        max_score = fmaxf(max_score, scores[token]);
    }
    float total = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        scores[token] = expf(scores[token] - max_score);
        total += scores[token];
    }
    const float inv_total = 1.0f / total;

    float* out = output + q_head * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
        out[dim] = 0.0f;
    }
    for (int token = 0; token < tokens; ++token) {
        const float weight = scores[token] * inv_total;
        const unsigned short* value =
            values + ((long long)token * kv_heads + kv_head) * head_dim;
        for (int dim = 0; dim < head_dim; ++dim) {
            out[dim] += weight * f16_bits_to_f32(value[dim]);
        }
    }

    // Sigmoid gate from the second half of each 2*head_dim q-head slice.
    const float* gate = gate_scratch + q_head * 2 * head_dim + head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
        const float sigmoid = 1.0f / (1.0f + expf(-gate[dim]));
        out[dim] *= sigmoid;
    }
}

extern "C" __global__ void sigmoid_inplace(
    float* values,
    int length
) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= length) {
        return;
    }
    values[index] = 1.0f / (1.0f + expf(-values[index]));
}

extern "C" __global__ void residual_add(
    float* accumulator,
    const float* increment,
    int length
) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= length) {
        return;
    }
    accumulator[index] += increment[index];
}

extern "C" __global__ void gdn_conv_silu(
    const float* input,
    const float* conv_weight,
    float* history,
    float* output,
    int channels
) {
    // Depthwise causal 4-tap convolution per channel, then in-place history
    // advance (drop oldest, append the new input), then SiLU. The conv output
    // must be computed from the old history before it is overwritten.
    const int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) {
        return;
    }
    float* history_channel = history + channel * 3;
    const float* weight_channel = conv_weight + channel * 4;
    const float acc = history_channel[0] * weight_channel[0]
        + history_channel[1] * weight_channel[1]
        + history_channel[2] * weight_channel[2]
        + input[channel] * weight_channel[3];
    history_channel[0] = history_channel[1];
    history_channel[1] = history_channel[2];
    history_channel[2] = input[channel];
    output[channel] = acc / (1.0f + expf(-acc));
}

extern "C" __global__ void l2_norm_heads(
    float* values,
    int heads,
    int head_dim,
    float epsilon
) {
    // In-place per-head l2 normalization with the ggml eps floor on the norm.
    const int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) {
        return;
    }
    float* head_values = values + (long long)head * head_dim;
    float sum = 0.0f;
    for (int index = 0; index < head_dim; ++index) {
        sum += head_values[index] * head_values[index];
    }
    const float scale = 1.0f / fmaxf(sqrtf(sum), epsilon);
    for (int index = 0; index < head_dim; ++index) {
        head_values[index] *= scale;
    }
}

extern "C" __global__ void strided_rms_norm(
    const float* input,
    const float* weight,
    float* output,
    int heads,
    int head_dim,
    float epsilon
) {
    // Per-head RMSNorm over a [heads][head_dim] buffer with shared raw
    // (already +1'd) weights, one thread per head.
    const int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) {
        return;
    }
    const float* head_input = input + (long long)head * head_dim;
    float* head_output = output + (long long)head * head_dim;
    float sum = 0.0f;
    for (int index = 0; index < head_dim; ++index) {
        sum += head_input[index] * head_input[index];
    }
    const float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (int index = 0; index < head_dim; ++index) {
        head_output[index] = head_input[index] * inverse * weight[index];
    }
}

extern "C" __global__ void gdn_state_update(
    float* matrix,
    const float* q_normed,
    const float* k_normed,
    const float* conv_activated,
    const float* decay,
    const float* beta,
    float* output,
    int v_heads,
    int k_heads,
    int head_dim,
    int v_offset
) {
    // One thread per (v head, column) of the column-oriented recurrent
    // update, matching the pinned equations: decay the state column,
    // sk = S^T k, d = (v - sk) * beta, S += k (outer) d, then
    // o = (S^T q) / sqrt(head_dim). q/k are l2-normalized per K head and
    // tiled across v heads (v head vh reads K head vh % k_heads).
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = v_heads * head_dim;
    if (index >= total) {
        return;
    }
    const int v_head = index / head_dim;
    const int col = index % head_dim;
    const int k_head = v_head % k_heads;
    float* state = matrix + (long long)v_head * head_dim * head_dim;
    const float* q = q_normed + (long long)k_head * head_dim;
    const float* k = k_normed + (long long)k_head * head_dim;
    const float* v = conv_activated + (long long)v_offset + (long long)v_head * head_dim;

    for (int row = 0; row < head_dim; ++row) {
        state[row * head_dim + col] *= decay[v_head];
    }
    float sk = 0.0f;
    for (int row = 0; row < head_dim; ++row) {
        sk += state[row * head_dim + col] * k[row];
    }
    const float d = (v[col] - sk) * beta[v_head];
    for (int row = 0; row < head_dim; ++row) {
        state[row * head_dim + col] += k[row] * d;
    }
    float out = 0.0f;
    for (int row = 0; row < head_dim; ++row) {
        out += state[row * head_dim + col] * q[row];
    }
    output[(long long)v_head * head_dim + col] = out * rsqrtf((float)head_dim);
}

extern "C" __global__ void gdn_gated_norm(
    const float* input,
    const float* z_gate,
    const float* ssm_norm,
    float* output,
    int v_heads,
    int head_dim,
    float epsilon
) {
    // Per-head gated norm: rms(input) * ssm_norm (raw, NOT +1'd) * silu(z).
    const int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= v_heads) {
        return;
    }
    const float* head_input = input + (long long)head * head_dim;
    const float* head_gate = z_gate + (long long)head * head_dim;
    float* head_output = output + (long long)head * head_dim;
    float sum = 0.0f;
    for (int index = 0; index < head_dim; ++index) {
        sum += head_input[index] * head_input[index];
    }
    const float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (int index = 0; index < head_dim; ++index) {
        const float z = head_gate[index];
        const float silu = z / (1.0f + expf(-z));
        head_output[index] = head_input[index] * inverse * ssm_norm[index] * silu;
    }
}
extern "C" __global__ void q_gate_norm(
    const float* q_gate,
    const float* q_norm_weight,
    float* q_normed,
    int q_heads,
    int head_dim,
    float epsilon
) {
    // Per-head RMSNorm over the first half of each [q | gate] slice from the
    // joint attn_q projection: input stride 2*head_dim, packed output.
    const int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= q_heads) {
        return;
    }
    const float* head_input = q_gate + (long long)head * 2 * head_dim;
    float* head_output = q_normed + (long long)head * head_dim;
    float sum = 0.0f;
    for (int index = 0; index < head_dim; ++index) {
        sum += head_input[index] * head_input[index];
    }
    const float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (int index = 0; index < head_dim; ++index) {
        head_output[index] = head_input[index] * inverse * q_norm_weight[index];
    }
}

__device__ __forceinline__ unsigned short f32_to_f16_bits(float value) {
    // Round-to-nearest-even conversion matching __float2half.
    const unsigned int bits = __float_as_int(value);
    const unsigned int sign = (bits >> 16) & 0x8000u;
    const int exponent = (int)((bits >> 23) & 0xffu) - 127 + 15;
    unsigned int mantissa = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        // Inf or NaN.
        return sign | 0x7c00u | (mantissa ? 0x0200u : 0u);
    }
    if (exponent >= 31) {
        // Overflow to infinity.
        return sign | 0x7c00u;
    }
    if (exponent <= 0) {
        // Subnormal or zero.
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        unsigned int result = mantissa >> shift;
        const unsigned int remainder = mantissa & ((1u << shift) - 1u);
        const unsigned int halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (result & 1u) != 0u)) {
            result += 1u;
        }
        return sign | result;
    }
    // Normal: 23-bit mantissa to 10-bit with rounding.
    unsigned int result = mantissa >> 13;
    const unsigned int remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0u)) {
        result += 1u;
        if (result == 0x400u) {
            result = 0;
            if (exponent + 1 >= 31) {
                return sign | 0x7c00u;
            }
            return sign | (((unsigned int)(exponent + 1)) << 10) | result;
        }
    }
    return sign | (((unsigned int)exponent << 10)) | result;
}

extern "C" __global__ void kv_append_f16(
    const float* keys,
    const float* values,
    unsigned short* cache_keys,
    unsigned short* cache_values,
    int token_index,
    int kv_heads,
    int head_dim
) {
    // Convert one token's F32 K/V [kv_heads * head_dim] to F16 and append
    // into the [token][kv_head][dim] cache at token_index.
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = kv_heads * head_dim;
    if (index >= total) {
        return;
    }
    const long long offset = (long long)token_index * total + index;
    cache_keys[offset] = f32_to_f16_bits(keys[index]);
    cache_values[offset] = f32_to_f16_bits(values[index]);
}
"#;

/// Reference-oriented elementwise operations used by Qwen-family model
/// regions. These kernels intentionally prioritize a clear numerical contract
/// over throughput; measured tiled/fused replacements can preserve the API.
pub struct CudaQwen35Ops {
    stream: Arc<CudaStream>,
    argmax: CudaFunction,
    rms_norm: CudaFunction,
    silu_mul: CudaFunction,
    l2_norm: CudaFunction,
    gdn_scalar_gate: CudaFunction,
    rope_neox: CudaFunction,
    sigmoid_inplace: CudaFunction,
    attn_score_gqa: CudaFunction,
    residual_add: CudaFunction,
    gdn_conv_silu: CudaFunction,
    l2_norm_heads: CudaFunction,
    strided_rms_norm: CudaFunction,
    gdn_state_update: CudaFunction,
    gdn_gated_norm: CudaFunction,
    q_gate_norm: CudaFunction,
    kv_append_f16: CudaFunction,
}

impl CudaQwen35Ops {
    /// Compile and load the reference Qwen elementwise kernels on a new CUDA
    /// context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaModelKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the reference Qwen elementwise kernels on an existing
    /// context/stream pair.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaModelKernelError> {
        if context.as_ref() != stream.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        let ptx = compile_ptx(MODEL_OPS_SOURCE)
            .map_err(|error| CudaModelKernelError::Nvrtc(error.to_string()))?;
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let argmax = module
            .load_function("argmax")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let rms_norm = module
            .load_function("rms_norm")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let silu_mul = module
            .load_function("silu_mul")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let l2_norm = module
            .load_function("l2_norm")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let gdn_scalar_gate = module
            .load_function("gdn_scalar_gate")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let rope_neox = module
            .load_function("rope_neox")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let sigmoid_inplace = module
            .load_function("sigmoid_inplace")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let attn_score_gqa = module
            .load_function("attn_score_gqa")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let residual_add = module
            .load_function("residual_add")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let gdn_conv_silu = module
            .load_function("gdn_conv_silu")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let l2_norm_heads = module
            .load_function("l2_norm_heads")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let strided_rms_norm = module
            .load_function("strided_rms_norm")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let gdn_state_update = module
            .load_function("gdn_state_update")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let gdn_gated_norm = module
            .load_function("gdn_gated_norm")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let q_gate_norm = module
            .load_function("q_gate_norm")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let kv_append_f16 = module
            .load_function("kv_append_f16")
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        Ok(Self {
            stream,
            argmax,
            rms_norm,
            silu_mul,
            l2_norm,
            gdn_scalar_gate,
            rope_neox,
            sigmoid_inplace,
            attn_score_gqa,
            residual_add,
            gdn_conv_silu,
            l2_norm_heads,
            strided_rms_norm,
            gdn_state_update,
            gdn_gated_norm,
            q_gate_norm,
            kv_append_f16,
        })
    }

    fn check_contexts(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != input.context().as_ref()
            || context.as_ref() != weight.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        Ok(())
    }

    /// Select the first index with the greatest finite logit value.
    ///
    /// This is a blocking host read because token selection is the boundary
    /// between device logits and the next request token. The launch itself is
    /// submitted asynchronously before the result is copied back.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when the context, input, or launch is
    /// invalid.
    pub fn argmax(&self, logits: &CudaSlice<f32>) -> Result<u32, CudaModelKernelError> {
        if self.stream.context().as_ref() != logits.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if logits.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        let length =
            u32::try_from(logits.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let mut selected = self
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        // Safety: cudarc allocated both slices, the length is checked, and the
        // single-thread launch writes exactly one result element.
        unsafe {
            self.stream
                .launch_builder(&self.argmax)
                .arg(logits)
                .arg(&mut selected)
                .arg(&length)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let selected = self
            .stream
            .clone_dtoh(&selected)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        selected
            .first()
            .copied()
            .ok_or(CudaModelKernelError::EmptyInput)
    }

    /// Apply `RMSNorm` to one F32 vector.
    ///
    /// This operation uses the exact equation `x * rsqrt(mean(x²) + epsilon) *
    /// weight`. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, epsilon, or
    /// launch arguments are invalid.
    pub fn rms_norm(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        epsilon: f32,
    ) -> Result<(), CudaModelKernelError> {
        self.check_contexts(input, weight, output)?;
        if input.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if input.len() != weight.len() {
            return Err(CudaModelKernelError::WeightLength {
                expected: input.len(),
                actual: weight.len(),
            });
        }
        if output.len() != input.len() {
            return Err(CudaModelKernelError::OutputLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let length = u32::try_from(input.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        // Safety: cudarc allocated all slices, lengths are validated, and the
        // one-block launch keeps all pointers alive until the stream observes it.
        unsafe {
            self.stream
                .launch_builder(&self.rms_norm)
                .arg(input)
                .arg(weight)
                .arg(output)
                .arg(&length)
                .arg(&epsilon)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply per-head `l2` normalization with an eps floor on the norm,
    /// matching `ggml_compute_forward_l2_norm_f32`:
    /// `1 / max(sqrt(sum(x²)), eps)`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, epsilon, or
    /// launch arguments are invalid.
    pub fn l2_norm(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        epsilon: f32,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != input.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if input.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if output.len() != input.len() {
            return Err(CudaModelKernelError::OutputLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let length = u32::try_from(input.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        // Safety: cudarc allocated both slices, lengths are validated, and
        // the one-block launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.l2_norm)
                .arg(input)
                .arg(output)
                .arg(&length)
                .arg(&epsilon)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply elementwise `SiLU` to `gate` and multiply it by `up`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, or launch
    /// arguments are invalid.
    pub fn silu_mul(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        self.check_contexts(gate, up, output)?;
        if gate.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if gate.len() != up.len() {
            return Err(CudaModelKernelError::InputLength {
                expected: gate.len(),
                actual: up.len(),
            });
        }
        if output.len() != gate.len() {
            return Err(CudaModelKernelError::OutputLength {
                expected: gate.len(),
                actual: output.len(),
            });
        }
        let length = u32::try_from(gate.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (length.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, lengths are validated, and the
        // launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.silu_mul)
                .arg(gate)
                .arg(up)
                .arg(output)
                .arg(&length)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Compute the per-head GDN scalar gates from projected inputs, matching
    /// the host reference: `decay = exp(a · softplus(alpha + dt_bias))` where
    /// `a = -exp(A_log)`, and `beta = sigmoid(beta_raw)`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, or launch
    /// arguments are invalid.
    pub fn gdn_scalar_gate(
        &self,
        alpha: &CudaSlice<f32>,
        beta_raw: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        decay: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != alpha.context().as_ref()
            || context.as_ref() != beta_raw.context().as_ref()
            || context.as_ref() != dt_bias.context().as_ref()
            || context.as_ref() != a.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        let heads = alpha.len();
        if heads == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        let lengths = [
            beta_raw.len(),
            dt_bias.len(),
            a.len(),
            decay.len(),
            beta.len(),
        ];
        if lengths.iter().any(|&len| len != heads) {
            return Err(CudaModelKernelError::InputLength {
                expected: heads,
                actual: lengths
                    .into_iter()
                    .find(|&len| len != heads)
                    .unwrap_or(heads),
            });
        }
        let heads_u32 = u32::try_from(heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (heads_u32.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, lengths are validated, and the
        // launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.gdn_scalar_gate)
                .arg(alpha)
                .arg(beta_raw)
                .arg(dt_bias)
                .arg(a)
                .arg(&mut *decay)
                .arg(&mut *beta)
                .arg(&heads_u32)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply NEOX half-split rotary position embedding in place over the
    /// first `rot_dims` dims of each head. All heads share the scalar
    /// `position`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    pub fn rope_neox(
        &self,
        values: &mut CudaSlice<f32>,
        position: u64,
        head_count: usize,
        head_dim: usize,
        rot_dims: usize,
        base: f32,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != values.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if head_count == 0 || head_dim == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if values.len() != head_count * head_dim {
            return Err(CudaModelKernelError::InputLength {
                expected: head_count * head_dim,
                actual: values.len(),
            });
        }
        if !rot_dims.is_multiple_of(2) || rot_dims > head_dim {
            return Err(CudaModelKernelError::ShapeOverflow);
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let pairs = u32::try_from(rot_dims / 2).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_count_u32 =
            u32::try_from(head_count).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let total_pairs = head_count
            .checked_mul(rot_dims / 2)
            .ok_or(CudaModelKernelError::ShapeOverflow)?;
        let total_pairs_u32 =
            u32::try_from(total_pairs).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (total_pairs_u32.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated the slice, geometry is validated, and the
        // launch keeps the pointer alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.rope_neox)
                .arg(&mut *values)
                .arg(&position)
                .arg(&head_count_u32)
                .arg(&head_dim_u32)
                .arg(&pairs)
                .arg(&base)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply the elementwise sigmoid in place.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, or launch
    /// arguments are invalid.
    pub fn sigmoid_inplace(&self, values: &mut CudaSlice<f32>) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != values.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if values.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        let length =
            u32::try_from(values.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (length.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated the slice and the validated length bounds
        // every access on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.sigmoid_inplace)
                .arg(&mut *values)
                .arg(&length)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// One full-attention decode step over the F16 KV cache with
    /// block-mapped GQA, softmax, weighted V sum, and sigmoid gate.
    ///
    /// `q` is the rope'd `[q_heads * head_dim]` query vector;
    /// `gate_scratch` holds the raw `[q_heads * 2 * head_dim]` q projection
    /// whose second half per head gates the output; `keys`/`values` are the
    /// `[tokens][kv_heads][head_dim]` F16 cache slices for one layer;
    /// `scores_scratch` is a `[q_heads * scores_stride]` accumulator reset by
    /// the caller; `output` receives `[q_heads * head_dim]`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    #[allow(
        clippy::too_many_arguments,
        reason = "the launch mirrors the kernel's fixed model geometry"
    )]
    pub fn attn_score_gqa(
        &self,
        q: &CudaSlice<f32>,
        keys: &CudaSlice<u16>,
        values: &CudaSlice<u16>,
        gate_scratch: &CudaSlice<f32>,
        scores_scratch: &mut CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        tokens: usize,
        scores_stride: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != q.context().as_ref()
            || context.as_ref() != keys.context().as_ref()
            || context.as_ref() != values.context().as_ref()
            || context.as_ref() != gate_scratch.context().as_ref()
            || context.as_ref() != scores_scratch.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 || tokens == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if !q_heads.is_multiple_of(kv_heads) {
            return Err(CudaModelKernelError::ShapeOverflow);
        }
        if q.len() != q_heads * head_dim
            || gate_scratch.len() != q_heads * 2 * head_dim
            || output.len() != q_heads * head_dim
            || scores_scratch.len() < q_heads * scores_stride
            || scores_stride < tokens
        {
            return Err(CudaModelKernelError::InputLength {
                expected: q_heads * head_dim,
                actual: q.len(),
            });
        }
        if keys.len() < tokens * kv_heads * head_dim || values.len() < tokens * kv_heads * head_dim
        {
            return Err(CudaModelKernelError::InputLength {
                expected: tokens * kv_heads * head_dim,
                actual: keys.len().min(values.len()),
            });
        }
        let tokens_u32 = u32::try_from(tokens).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let stride_u32 =
            u32::try_from(scores_stride).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let q_heads_u32 =
            u32::try_from(q_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let kv_heads_u32 =
            u32::try_from(kv_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (q_heads_u32.div_ceil(32), 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, geometry is validated, and
        // the launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.attn_score_gqa)
                .arg(q)
                .arg(keys)
                .arg(values)
                .arg(gate_scratch)
                .arg(&mut *scores_scratch)
                .arg(&mut *output)
                .arg(&tokens_u32)
                .arg(&stride_u32)
                .arg(&q_heads_u32)
                .arg(&kv_heads_u32)
                .arg(&head_dim_u32)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Add `increment` into `accumulator` elementwise.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, or launch
    /// arguments are invalid.
    pub fn residual_add(
        &self,
        accumulator: &mut CudaSlice<f32>,
        increment: &CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != accumulator.context().as_ref()
            || context.as_ref() != increment.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if accumulator.is_empty() || accumulator.len() != increment.len() {
            return Err(CudaModelKernelError::InputLength {
                expected: accumulator.len(),
                actual: increment.len(),
            });
        }
        let length =
            u32::try_from(accumulator.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (length.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated both slices, lengths are validated, and
        // the launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.residual_add)
                .arg(&mut *accumulator)
                .arg(increment)
                .arg(&length)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Depthwise causal 4-tap convolution over `[history | input]` per
    /// channel, advancing the 3-element history in place, then `SiLU`.
    ///
    /// `conv_weight` holds `(tap, channel)` at `tap + channel * 4`,
    /// matching the GGUF `ssm_conv1d` layout.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, lengths, or launch
    /// arguments are invalid.
    pub fn gdn_conv_silu(
        &self,
        input: &CudaSlice<f32>,
        conv_weight: &CudaSlice<f32>,
        history: &mut CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != input.context().as_ref()
            || context.as_ref() != conv_weight.context().as_ref()
            || context.as_ref() != history.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        let channels = input.len();
        if channels == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if conv_weight.len() != channels * 4
            || history.len() != channels * 3
            || output.len() != channels
        {
            return Err(CudaModelKernelError::InputLength {
                expected: channels * 4,
                actual: conv_weight.len(),
            });
        }
        let channels_u32 =
            u32::try_from(channels).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (channels_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, lengths are validated, and
        // the launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.gdn_conv_silu)
                .arg(input)
                .arg(conv_weight)
                .arg(&mut *history)
                .arg(&mut *output)
                .arg(&channels_u32)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// In-place per-head l2 normalization with the ggml eps floor on the
    /// norm over a `[heads][head_dim]` buffer.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    pub fn l2_norm_heads(
        &self,
        values: &mut CudaSlice<f32>,
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != values.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if heads == 0 || head_dim == 0 || values.len() != heads * head_dim {
            return Err(CudaModelKernelError::InputLength {
                expected: heads * head_dim,
                actual: values.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let heads_u32 = u32::try_from(heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (heads_u32.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated the slice, geometry is validated, and the
        // launch keeps the pointer alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.l2_norm_heads)
                .arg(&mut *values)
                .arg(&heads_u32)
                .arg(&head_dim_u32)
                .arg(&epsilon)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Per-head `RMSNorm` over `[heads][head_dim]` with shared raw weights.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    pub fn strided_rms_norm(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != input.context().as_ref()
            || context.as_ref() != weight.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if heads == 0 || head_dim == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if input.len() != heads * head_dim
            || weight.len() != head_dim
            || output.len() != heads * head_dim
        {
            return Err(CudaModelKernelError::InputLength {
                expected: heads * head_dim,
                actual: input.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let heads_u32 = u32::try_from(heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (heads_u32.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, geometry is validated, and the
        // launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.strided_rms_norm)
                .arg(input)
                .arg(weight)
                .arg(output)
                .arg(&heads_u32)
                .arg(&head_dim_u32)
                .arg(&epsilon)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Column-oriented recurrent state update for all GDN v heads of one
    /// layer, in place on the device state matrix, plus the scaled output.
    ///
    /// `q_normed`/`k_normed` are l2-normalized `[k_heads][head_dim]` buffers
    /// (tiled across v heads by `v_head % k_heads`); `conv_activated` is the
    /// `SiLU`'d `[q(2048) | k(2048) | v(6144)]` conv output with the v slice at
    /// `v_offset`; `decay`/`beta` are the per-v-head scalars.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    #[allow(
        clippy::too_many_arguments,
        reason = "the launch mirrors the kernel's fixed model geometry"
    )]
    pub fn gdn_state_update(
        &self,
        matrix: &mut CudaSlice<f32>,
        q_normed: &CudaSlice<f32>,
        k_normed: &CudaSlice<f32>,
        conv_activated: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        v_heads: usize,
        k_heads: usize,
        head_dim: usize,
        v_offset: usize,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != matrix.context().as_ref()
            || context.as_ref() != q_normed.context().as_ref()
            || context.as_ref() != k_normed.context().as_ref()
            || context.as_ref() != conv_activated.context().as_ref()
            || context.as_ref() != decay.context().as_ref()
            || context.as_ref() != beta.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if v_heads == 0 || k_heads == 0 || head_dim == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if !v_heads.is_multiple_of(k_heads) {
            return Err(CudaModelKernelError::ShapeOverflow);
        }
        let state_len = v_heads * head_dim * head_dim;
        if matrix.len() != state_len
            || q_normed.len() != k_heads * head_dim
            || k_normed.len() != k_heads * head_dim
            || decay.len() != v_heads
            || beta.len() != v_heads
            || output.len() != v_heads * head_dim
            || conv_activated.len() < v_offset + v_heads * head_dim
        {
            return Err(CudaModelKernelError::InputLength {
                expected: state_len,
                actual: matrix.len(),
            });
        }
        let v_heads_u32 =
            u32::try_from(v_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let k_heads_u32 =
            u32::try_from(k_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let v_offset_u32 =
            u32::try_from(v_offset).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let total = v_heads
            .checked_mul(head_dim)
            .ok_or(CudaModelKernelError::ShapeOverflow)?;
        let total_u32 = u32::try_from(total).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (total_u32.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, geometry is validated, and
        // the launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.gdn_state_update)
                .arg(&mut *matrix)
                .arg(q_normed)
                .arg(k_normed)
                .arg(conv_activated)
                .arg(decay)
                .arg(beta)
                .arg(&mut *output)
                .arg(&v_heads_u32)
                .arg(&k_heads_u32)
                .arg(&head_dim_u32)
                .arg(&v_offset_u32)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Per-head gated norm `rms(input) * ssm_norm (raw) * silu(z_gate)` over
    /// `[v_heads][head_dim]`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    #[allow(
        clippy::too_many_arguments,
        reason = "the launch mirrors the kernel's fixed model geometry"
    )]
    pub fn gdn_gated_norm(
        &self,
        input: &CudaSlice<f32>,
        z_gate: &CudaSlice<f32>,
        ssm_norm: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        v_heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != input.context().as_ref()
            || context.as_ref() != z_gate.context().as_ref()
            || context.as_ref() != ssm_norm.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if v_heads == 0 || head_dim == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if input.len() != v_heads * head_dim
            || z_gate.len() != v_heads * head_dim
            || ssm_norm.len() != head_dim
            || output.len() != v_heads * head_dim
        {
            return Err(CudaModelKernelError::InputLength {
                expected: v_heads * head_dim,
                actual: input.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let v_heads_u32 =
            u32::try_from(v_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (v_heads_u32.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, geometry is validated, and the
        // launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.gdn_gated_norm)
                .arg(input)
                .arg(z_gate)
                .arg(ssm_norm)
                .arg(output)
                .arg(&v_heads_u32)
                .arg(&head_dim_u32)
                .arg(&epsilon)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Per-head `RMSNorm` over the first half of each interleaved
    /// `[q | gate]` slice from the joint `attn_q` projection, packing the
    /// normalized query into a contiguous `[q_heads][head_dim]` output.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    pub fn q_gate_norm(
        &self,
        q_gate: &CudaSlice<f32>,
        q_norm_weight: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        q_heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != q_gate.context().as_ref()
            || context.as_ref() != q_norm_weight.context().as_ref()
            || context.as_ref() != output.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if q_heads == 0 || head_dim == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if q_gate.len() != q_heads * 2 * head_dim
            || q_norm_weight.len() != head_dim
            || output.len() != q_heads * head_dim
        {
            return Err(CudaModelKernelError::InputLength {
                expected: q_heads * 2 * head_dim,
                actual: q_gate.len(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaModelKernelError::InvalidEpsilon);
        }
        let q_heads_u32 =
            u32::try_from(q_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (q_heads_u32.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, geometry is validated, and the
        // launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.q_gate_norm)
                .arg(q_gate)
                .arg(q_norm_weight)
                .arg(output)
                .arg(&q_heads_u32)
                .arg(&head_dim_u32)
                .arg(&epsilon)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Convert one token's F32 K/V to F16 and append it into the
    /// `[token][kv_head][head_dim]` cache at `token_index`.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when contexts, geometry, or launch
    /// arguments are invalid.
    #[allow(
        clippy::too_many_arguments,
        reason = "the launch mirrors the kernel's fixed model geometry"
    )]
    pub fn kv_append_f16(
        &self,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
        cache_keys: &mut CudaSlice<u16>,
        cache_values: &mut CudaSlice<u16>,
        token_index: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), CudaModelKernelError> {
        let context = self.stream.context();
        if context.as_ref() != keys.context().as_ref()
            || context.as_ref() != values.context().as_ref()
            || context.as_ref() != cache_keys.context().as_ref()
            || context.as_ref() != cache_values.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        let total = kv_heads
            .checked_mul(head_dim)
            .ok_or(CudaModelKernelError::ShapeOverflow)?;
        if total == 0 {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if keys.len() != total || values.len() != total {
            return Err(CudaModelKernelError::InputLength {
                expected: total,
                actual: keys.len().min(values.len()),
            });
        }
        let capacity = token_index
            .checked_add(1)
            .and_then(|tokens| tokens.checked_mul(total))
            .ok_or(CudaModelKernelError::ShapeOverflow)?;
        if cache_keys.len() < capacity || cache_values.len() < capacity {
            return Err(CudaModelKernelError::OutputLength {
                expected: capacity,
                actual: cache_keys.len().min(cache_values.len()),
            });
        }
        let token_index_u32 =
            u32::try_from(token_index).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let kv_heads_u32 =
            u32::try_from(kv_heads).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let total_u32 = u32::try_from(total).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (total_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: cudarc allocated all slices, geometry is validated, and the
        // launch keeps all pointers alive on the same stream.
        unsafe {
            self.stream
                .launch_builder(&self.kv_append_f16)
                .arg(keys)
                .arg(values)
                .arg(&mut *cache_keys)
                .arg(&mut *cache_values)
                .arg(&token_index_u32)
                .arg(&kv_heads_u32)
                .arg(&head_dim_u32)
                .launch(config)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}
