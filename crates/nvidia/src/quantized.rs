use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::cuda::CudaQuantizedWeight;

const Q8_0_VALUE_TYPE: u32 = 8;
const Q3_K_VALUE_TYPE: u32 = 11;
const Q4_K_VALUE_TYPE: u32 = 12;
const Q5_K_VALUE_TYPE: u32 = 13;
const Q6_K_VALUE_TYPE: u32 = 14;
const Q8_0_BLOCK_ELEMENTS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q3_K_BLOCK_ELEMENTS: usize = 256;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q4_K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_ELEMENTS: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_ELEMENTS: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;

const Q_K_GEMV_SOURCE: &str = r#"
extern "C" __device__ __forceinline__ float decode_f16(unsigned short bits) {
    const int sign = (bits & 0x8000u) != 0u ? -1 : 1;
    const int exponent = (bits >> 10u) & 0x1fu;
    const int fraction = bits & 0x03ffu;
    if (exponent == 0) {
        return (float)sign * ((float)fraction / 1024.0f) * 0.00006103515625f;
    }
    if (exponent == 31) {
        if (fraction == 0) {
            return sign > 0 ? __int_as_float(0x7f800000) : __int_as_float(0xff800000);
        }
        return __int_as_float(0x7fc00000);
    }
    float scale = 1.0f;
    int shift = exponent - 15;
    if (shift > 0) {
        for (int i = 0; i < shift; ++i) {
            scale *= 2.0f;
        }
    } else {
        for (int i = 0; i > shift; --i) {
            scale *= 0.5f;
        }
    }
    return (float)sign * (1.0f + (float)fraction / 1024.0f) * scale;
}

extern "C" __device__ __forceinline__ int scale_value(const unsigned char* block, int group) {
    if (group < 4) {
        return (int)(block[4 + group] & 0x3fu);
    }
    const int index = group - 4;
    return (int)((block[12 + index] & 0x0fu) | ((block[4 + index] >> 2u) & 0x30u));
}

extern "C" __device__ __forceinline__ int minimum_value(const unsigned char* block, int group) {
    if (group < 4) {
        return (int)(block[8 + group] & 0x3fu);
    }
    const int index = group - 4;
    return (int)((block[12 + index] >> 4u) | ((block[8 + index] >> 2u) & 0x30u));
}

extern "C" __global__ void q8_0_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 32;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 34;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        for (int local = 0; local < 32; ++local) {
            const int quantized = (int)((signed char)block[2 + local]);
            accumulator += d * (float)quantized * input[block_index * 32 + local];
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q3_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 110;
        const float d = decode_f16((unsigned short)block[108] | ((unsigned short)block[109] << 8u));
        const unsigned char* high_bits = block;
        const unsigned char* low_bits = block + 32;
        const unsigned char* packed_scales = block + 96;
        for (int group = 0; group < 16; ++group) {
            const int index = group < 8 ? group : group - 8;
            const int scale_bits = group < 8
                ? (int)((packed_scales[index] & 0x0fu)
                    | (((packed_scales[8 + index % 4] >> ((index / 4) * 2)) & 0x03u) << 4u))
                : (int)((packed_scales[index] >> 4u)
                    | (((packed_scales[8 + index % 4] >> ((group / 4) * 2)) & 0x03u) << 4u));
            const int group_scale = scale_bits - 32;
            const int chunk = group / 8;
            const int variant = (group / 2) & 3;
            const int half = group & 1;
            const int low_offset = chunk * 32 + half * 16;
            const int high_offset = half * 16;
            for (int local = 0; local < 16; ++local) {
                const int low = (int)((low_bits[low_offset + local] >> (variant * 2)) & 0x03u);
                const int high = ((int)(high_bits[high_offset + local] >> (group / 2)) & 1) ^ 1;
                const int quantized = low - high * 4;
                accumulator += d * (float)group_scale * (float)quantized
                    * input[block_index * 256 + group * 16 + local];
            }
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q6_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 210;
        const unsigned char* low_bits = block;
        const unsigned char* high_bits = block + 128;
        const unsigned char* scales = block + 192;
        const float d = decode_f16((unsigned short)block[208] | ((unsigned short)block[209] << 8u));
        for (int group = 0; group < 8; ++group) {
            const int chunk = group / 4;
            const int variant = group & 3;
            const int low_offset = chunk * 64 + (variant & 1) * 32;
            const int low_shift = variant < 2 ? 0 : 4;
            const int high_offset = chunk * 32;
            const int high_shift = variant;
            for (int half = 0; half < 2; ++half) {
                const int group_scale = (int)(signed char)scales[group * 2 + half];
                for (int local = 0; local < 16; ++local) {
                    const int position = half * 16 + local;
                    const int low = (int)((low_bits[low_offset + position] >> low_shift) & 0x0fu);
                    const int high = (int)((high_bits[high_offset + position] >> (high_shift * 2)) & 0x03u);
                    const int quantized = (low | (high << 4)) - 32;
                    accumulator += d * (float)group_scale * (float)quantized
                        * input[block_index * 256 + group * 32 + half * 16 + local];
                }
            }
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q4_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 144;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        for (int local = 0; local < 256; ++local) {
            const int group = local / 32;
            const int index = local & 31;
            const int data_offset = 16 + (group / 2) * 32;
            const int shift = (group & 1) * 4;
            const int quantized = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const float value =
                d * (float)scale_value(block, group) * (float)quantized
                - min * (float)minimum_value(block, group);
            accumulator += value * input[block_index * 256 + local];
        }
    }
    output[output_index] = accumulator;
}

extern "C" __global__ void q5_k_gemv(
    const unsigned char* weights,
    const float* input,
    float* output,
    int input_size,
    int output_size
) {
    const int output_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= output_size) {
        return;
    }

    const int blocks_per_output = input_size / 256;
    float accumulator = 0.0f;
    for (int block_index = 0; block_index < blocks_per_output; ++block_index) {
        const unsigned char* block =
            weights + (output_index * blocks_per_output + block_index) * 176;
        const float d = decode_f16((unsigned short)block[0] | ((unsigned short)block[1] << 8u));
        const float min = decode_f16((unsigned short)block[2] | ((unsigned short)block[3] << 8u));
        for (int local = 0; local < 256; ++local) {
            const int group = local / 32;
            const int index = local & 31;
            const int data_offset = 48 + (group / 2) * 32;
            const int shift = (group & 1) * 4;
            const int low = (int)((block[data_offset + index] >> shift) & 0x0fu);
            const int high = (int)((block[16 + index] >> group) & 1u);
            const int quantized = low | (high << 4);
            const float value =
                d * (float)scale_value(block, group) * (float)quantized
                - min * (float)minimum_value(block, group);
            accumulator += value * input[block_index * 256 + local];
        }
    }
    output[output_index] = accumulator;
}
"#;

#[derive(Debug)]
pub enum CudaQuantizedKernelError {
    Driver(String),
    Nvrtc(String),
    InvalidWeight(String),
    UnsupportedValueType { expected: u32, actual: u32 },
    ShapeOverflow,
    InputLength { expected: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
    ContextMismatch,
}

impl fmt::Display for CudaQuantizedKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
            Self::Nvrtc(message) => write!(f, "NVRTC compilation error: {message}"),
            Self::InvalidWeight(reason) => write!(f, "invalid quantized weight: {reason}"),
            Self::UnsupportedValueType { expected, actual } => write!(
                f,
                "quantized GEMV requires GGML value type {expected}, got {actual}"
            ),
            Self::ShapeOverflow => {
                f.write_str("quantized GEMV shape does not fit CUDA launch arguments")
            }
            Self::InputLength { expected, actual } => {
                write!(
                    f,
                    "quantized GEMV input has {actual} values, expected {expected}"
                )
            }
            Self::OutputLength { expected, actual } => {
                write!(
                    f,
                    "quantized GEMV output has {actual} values, expected {expected}"
                )
            }
            Self::ContextMismatch => {
                f.write_str("quantized GEMV buffers and stream belong to different CUDA contexts")
            }
        }
    }
}

impl std::error::Error for CudaQuantizedKernelError {}

/// Shared launch and validation state for the block-quantized GEMV kernels.
///
/// The kernels follow GGML's column-major tensor ordering: the first tensor
/// dimension is the contiguous input (`K`) count and the second is the output
/// (`N`) count. They deliberately favor a simple one-thread-per-output-column
/// implementation; a measured tiled kernel can replace it without changing
/// the encoded-weight or model-facing boundary.
struct CudaQuantizedGemv {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    value_type: u32,
    block_elements: usize,
    block_bytes: usize,
    label: &'static str,
}

impl CudaQuantizedGemv {
    fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        function_name: &'static str,
        value_type: u32,
        block_elements: usize,
        block_bytes: usize,
        label: &'static str,
    ) -> Result<Self, CudaQuantizedKernelError> {
        if context.as_ref() != stream.context().as_ref() {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let ptx = compile_ptx(Q_K_GEMV_SOURCE)
            .map_err(|error| CudaQuantizedKernelError::Nvrtc(error.to_string()))?;
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let kernel = module
            .load_function(function_name)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        Ok(Self {
            stream,
            kernel,
            value_type,
            block_elements,
            block_bytes,
            label,
        })
    }

    /// Execute one block-quantized matrix-vector product into a caller-owned
    /// output.
    ///
    /// The launch is asynchronous with respect to the host. A subsequent copy
    /// or stream synchronization observes completion. This method does not
    /// dequantize or copy the encoded weight through host memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight type, shape, encoded
    /// length, or device vector lengths are invalid, or when launch fails.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if self.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.stream.context().as_ref() != input.context().as_ref()
            || self.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != self.value_type {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: self.value_type,
                actual: weight.value_type(),
            });
        }
        let dimensions = weight.spec().dimensions();
        if dimensions.len() != 2 {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "expected rank-2 tensor, got rank {}",
                dimensions.len()
            )));
        }
        let input_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let output_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if input_size == 0 || output_size == 0 || !input_size.is_multiple_of(self.block_elements) {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "{} shape is {input_size}x{output_size}; input size must be a positive multiple of {}",
                self.label, self.block_elements
            )));
        }
        let blocks = input_size
            .checked_div(self.block_elements)
            .and_then(|value| value.checked_mul(output_size))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_bytes = blocks
            .checked_mul(self.block_bytes)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if weight.encoded_bytes() != expected_bytes {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "encoded length is {}, expected {expected_bytes}",
                weight.encoded_bytes()
            )));
        }
        if blocks > (i32::MAX as usize) / self.block_bytes {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if input_size > i32::MAX as usize || output_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if input.len() != input_size {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: input_size,
                actual: input.len(),
            });
        }
        if output.len() != output_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: output_size,
                actual: output.len(),
            });
        }
        let input_size =
            u32::try_from(input_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let output_size =
            u32::try_from(output_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (output_size.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `Q3_K` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ3KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ3KGemv {
    /// Compile and load the `Q3_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q3_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q3_k_gemv",
                Q3_K_VALUE_TYPE,
                Q3_K_BLOCK_ELEMENTS,
                Q3_K_BLOCK_BYTES,
                "Q3_K",
            )?,
        })
    }

    /// Execute one `Q3_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }
}

/// A correctness-oriented `Q8_0` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ8_0Gemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ8_0Gemv {
    /// Compile and load the `Q8_0` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q8_0` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q8_0_gemv",
                Q8_0_VALUE_TYPE,
                Q8_0_BLOCK_ELEMENTS,
                Q8_0_BLOCK_BYTES,
                "Q8_0",
            )?,
        })
    }

    /// Execute one `Q8_0` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }
}

/// A correctness-oriented `Q6_K` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ6KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ6KGemv {
    /// Compile and load the `Q6_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q6_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q6_k_gemv",
                Q6_K_VALUE_TYPE,
                Q6_K_BLOCK_ELEMENTS,
                Q6_K_BLOCK_BYTES,
                "Q6_K",
            )?,
        })
    }

    /// Execute one `Q6_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }
}

/// A correctness-oriented `Q4_K` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaQ4KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ4KGemv {
    /// Compile and load the `Q4_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q4_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q4_k_gemv",
                Q4_K_VALUE_TYPE,
                Q4_K_BLOCK_ELEMENTS,
                Q4_K_BLOCK_BYTES,
                "Q4_K",
            )?,
        })
    }

    /// Execute one `Q4_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }
}

/// A correctness-oriented `Q5_K` matrix-vector kernel.
///
/// It shares the GGML `[K,N]` contract and launch validation with
/// [`CudaQ4KGemv`], but decodes the 5-bit high-bit plane and 176-byte block
/// layout used by GGML value type 13.
pub struct CudaQ5KGemv {
    inner: CudaQuantizedGemv,
}

impl CudaQ5KGemv {
    /// Compile and load the `Q5_K` kernel on a new CUDA device context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module loading
    /// fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q5_K` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        Ok(Self {
            inner: CudaQuantizedGemv::from_context(
                context,
                stream,
                "q5_k_gemv",
                Q5_K_VALUE_TYPE,
                Q5_K_BLOCK_ELEMENTS,
                Q5_K_BLOCK_BYTES,
                "Q5_K",
            )?,
        })
    }

    /// Execute one `Q5_K` matrix-vector product into a caller-owned output.
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes, contexts,
    /// or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute(weight, input, output)
    }
}
