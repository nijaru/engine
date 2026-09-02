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
    float best_value = -INFINITY;
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
"#;

/// Reference-oriented elementwise operations used by Qwen-family model
/// regions. These kernels intentionally prioritize a clear numerical contract
/// over throughput; measured tiled/fused replacements can preserve the API.
pub struct CudaQwen35Ops {
    stream: Arc<CudaStream>,
    argmax: CudaFunction,
    rms_norm: CudaFunction,
    silu_mul: CudaFunction,
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
        Ok(Self {
            stream,
            argmax,
            rms_norm,
            silu_mul,
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
}
