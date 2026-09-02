use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::cuda::CudaQuantizedWeight;

const Q4_K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

const Q4_K_GEMV_SOURCE: &str = r#"
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
            Self::InvalidWeight(reason) => write!(f, "invalid Q4_K weight: {reason}"),
            Self::UnsupportedValueType { expected, actual } => write!(
                f,
                "Q4_K GEMV requires GGML value type {expected}, got {actual}"
            ),
            Self::ShapeOverflow => {
                f.write_str("Q4_K GEMV shape does not fit CUDA launch arguments")
            }
            Self::InputLength { expected, actual } => {
                write!(
                    f,
                    "Q4_K GEMV input has {actual} values, expected {expected}"
                )
            }
            Self::OutputLength { expected, actual } => {
                write!(
                    f,
                    "Q4_K GEMV output has {actual} values, expected {expected}"
                )
            }
            Self::ContextMismatch => {
                f.write_str("Q4_K GEMV buffers and stream belong to different CUDA contexts")
            }
        }
    }
}

impl std::error::Error for CudaQuantizedKernelError {}

/// A first correctness-oriented `Q4_K` matrix-vector kernel.
///
/// The kernel follows GGML's column-major tensor ordering: the first tensor
/// dimension is the contiguous input (`K`) count and the second is the output
/// (`N`) count. It deliberately favors a simple one-thread-per-output-column
/// implementation; a measured tiled kernel can replace it without changing
/// the encoded-weight or model-facing boundary.
pub struct CudaQ4KGemv {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

impl CudaQ4KGemv {
    /// Compile and load the kernel on a new CUDA device context.
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

    /// Compile and load the kernel on an existing context/stream pair. Weight
    /// buffers and input/output vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        if context.as_ref() != stream.context().as_ref() {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let ptx = compile_ptx(Q4_K_GEMV_SOURCE)
            .map_err(|error| CudaQuantizedKernelError::Nvrtc(error.to_string()))?;
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let kernel = module
            .load_function("q4_k_gemv")
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        Ok(Self { stream, kernel })
    }

    /// Execute one `Q4_K` matrix-vector product into a caller-owned output.
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
        if weight.value_type() != 12 {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: 12,
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
        if input_size == 0 || output_size == 0 || !input_size.is_multiple_of(Q4_K_BLOCK_ELEMENTS) {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "shape is {input_size}x{output_size}; input size must be a positive multiple of {Q4_K_BLOCK_ELEMENTS}"
            )));
        }
        let blocks = input_size
            .checked_div(Q4_K_BLOCK_ELEMENTS)
            .and_then(|value| value.checked_mul(output_size))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_bytes = blocks
            .checked_mul(Q4_K_BLOCK_BYTES)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if weight.encoded_bytes() != expected_bytes {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "encoded length is {}, expected {expected_bytes}",
                weight.encoded_bytes()
            )));
        }
        if blocks > (i32::MAX as usize) / Q4_K_BLOCK_BYTES {
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
