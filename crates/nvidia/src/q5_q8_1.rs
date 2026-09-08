use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use super::{
    CudaQuantizedKernelError, MAX_BATCH_MEMBERS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS,
    Q5_K_VALUE_TYPE, validate_quantized_geometry,
};
use crate::cuda::CudaQuantizedWeight;

const SOURCE: &str = concat!(
    include_str!("kernels/quantized/common.cu"),
    include_str!("kernels/quantized/q5_k_q8_1.cu"),
);
static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();

/// Experimental integer-dot `Q5_K` GEMV consuming packed `Q8_1` activations.
///
/// Activations use [`crate::CudaQ8_1Quantizer`]'s nine-word blocks. The stored
/// half scale and half original-input sum are both used; arithmetic differs
/// from multiplying fully dequantized activations because of that sum term.
/// Packing is lossy. This kernel is opt-in and does not change model defaults.
/// Requires a GPU supporting the compiled compute-75 target or newer.
pub struct CudaQ5KQ8_1Gemv {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    batch_kernel: CudaFunction,
}

impl CudaQ5KQ8_1Gemv {
    /// Compile and load the experimental kernel on the stream's context.
    ///
    /// # Errors
    /// Returns an error when NVRTC or CUDA module loading fails.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaQuantizedKernelError> {
        let ptx = PTX
            .get_or_init(|| {
                compile_ptx_with_opts(
                    format!("#define MAX_BATCH_MEMBERS {MAX_BATCH_MEMBERS}\n{SOURCE}"),
                    CompileOptions {
                        arch: Some("compute_75"),
                        ..CompileOptions::default()
                    },
                )
                .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| CudaQuantizedKernelError::Nvrtc(error.clone()))?
            .clone();
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let kernel = module
            .load_function("q5_k_q8_1_gemv")
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let batch_kernel = module
            .load_function("q5_k_q8_1_gemv_batch")
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        Ok(Self {
            stream,
            kernel,
            batch_kernel,
        })
    }

    /// Enqueue one matrix-vector product into caller-owned output storage.
    ///
    /// `input` must contain exactly `K / 32 * 9` words produced by the `Q8_1`
    /// quantizer from finite, half-representable scale/sum inputs. No allocation
    /// or host synchronization occurs during this call.
    ///
    /// # Errors
    /// Rejects foreign contexts, malformed `Q5_K` weights, incorrect packed
    /// input/output lengths, overflowing geometry, and launch failures.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        let context = self.stream.context().as_ref();
        if context != weight.encoded_data().context().as_ref()
            || context != input.context().as_ref()
            || context != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let (input_size, output_size) = validate_quantized_geometry(
            weight,
            Q5_K_VALUE_TYPE,
            Q5_K_BLOCK_ELEMENTS,
            Q5_K_BLOCK_BYTES,
            "Q5_K",
        )?;
        let expected_input = input_size / 32 * 9;
        if input.len() != expected_input {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: expected_input,
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
            grid_dim: (output_size.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: shared geometry validation bounds every weight load, input
        // storage contains complete Q8_1 blocks, and each row owns one output.
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
        }
        .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        Ok(())
    }

    /// Enqueue one weights-read-once batched matrix product over `members`
    /// batch-major packed activation rows into a batch-major output.
    ///
    /// Each member contributes one `[K / 32 * 9]` packed row produced by the
    /// `Q8_1` quantizer and owns one `[N]` output row. One warp decodes each
    /// weight element once and accumulates it against every member. No
    /// allocation or host synchronization occurs during this call.
    ///
    /// # Errors
    /// Rejects foreign contexts, malformed `Q5_K` weights, member counts
    /// outside `1..=8`, incorrect packed input/output lengths, overflowing
    /// geometry, and launch failures.
    pub fn execute_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        if members == 0 {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: 1,
                actual: 0,
            });
        }
        if members > MAX_BATCH_MEMBERS {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "Q5_K integer-dot batch variant supports at most {MAX_BATCH_MEMBERS} members"
            )));
        }
        let context = self.stream.context().as_ref();
        if context != weight.encoded_data().context().as_ref()
            || context != input.context().as_ref()
            || context != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let (input_size, output_size) = validate_quantized_geometry(
            weight,
            Q5_K_VALUE_TYPE,
            Q5_K_BLOCK_ELEMENTS,
            Q5_K_BLOCK_BYTES,
            "Q5_K",
        )?;
        let expected_input = input_size
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(9))
            .and_then(|per_member| per_member.checked_mul(members))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_output = output_size
            .checked_mul(members)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if input.len() != expected_input {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: expected_input,
                actual: input.len(),
            });
        }
        if output.len() != expected_output {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: expected_output,
                actual: output.len(),
            });
        }
        let input_size =
            u32::try_from(input_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let output_size =
            u32::try_from(output_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let members_u32 =
            u32::try_from(members).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (output_size.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: shared geometry validation bounds every weight load, each
        // member's input storage contains complete Q8_1 blocks, and each
        // member owns one disjoint output row.
        unsafe {
            self.stream
                .launch_builder(&self.batch_kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&input_size)
                .arg(&output_size)
                .arg(&members_u32)
                .launch(config)
        }
        .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        Ok(())
    }
}
