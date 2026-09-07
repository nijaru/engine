//! Experimental activation packing for integer-dot kernels.
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::CudaQuantizedKernelError;

/// Packs contiguous activations into GGML CUDA `Q8_1` blocks without host copies.
///
/// Each 32-element block occupies nine `u32` words: half scale and half input
/// sum, followed by 32 signed bytes. Rows must contain whole blocks. Inputs
/// must be finite with scale and sum representable in half precision. This
/// lossy primitive is experimental and does not change default model execution.
pub struct CudaQ8_1Quantizer {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

impl CudaQ8_1Quantizer {
    /// Compile on the stream's context.
    ///
    /// # Errors
    /// Returns an error if compilation or CUDA module loading fails.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaQuantizedKernelError> {
        let ptx = compile_ptx(include_str!("activation.cu"))
            .map_err(|e| CudaQuantizedKernelError::Nvrtc(e.to_string()))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| CudaQuantizedKernelError::Driver(e.to_string()))?;
        let kernel = module
            .load_function("quantize_q8_1")
            .map_err(|e| CudaQuantizedKernelError::Driver(e.to_string()))?;
        Ok(Self { stream, kernel })
    }

    /// Enqueue packing into caller-owned storage; no allocation or host wait.
    ///
    /// # Errors
    /// Rejects empty or partial blocks, wrong output sizes, foreign contexts,
    /// oversized launch geometry, and CUDA launch failures.
    pub fn execute(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<u32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if input.context().as_ref() != self.stream.context().as_ref()
            || output.context().as_ref() != self.stream.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if input.is_empty() || !input.len().is_multiple_of(32) {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: input.len().max(1).div_ceil(32) * 32,
                actual: input.len(),
            });
        }
        let blocks = input.len() / 32;
        let expected = blocks * 9;
        if output.len() != expected {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        // Device indexing uses u32 for both input elements and output words.
        let _ = u32::try_from(input.len()).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let blocks = u32::try_from(blocks).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (blocks.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // Shapes and contexts above establish valid accesses; cudarc tracks
        // buffer usage on this stream through completion.
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(input)
                .arg(output)
                .arg(&blocks)
                .launch(config)
        }
        .map_err(|e| CudaQuantizedKernelError::Driver(e.to_string()))?;
        Ok(())
    }
}
