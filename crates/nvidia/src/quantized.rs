use std::fmt;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{Ptx, compile_ptx};

use crate::cuda::CudaQuantizedWeight;

/// Maximum row count supported by the weights-read-once CUDA kernels.
pub const MAX_BATCH_MEMBERS: usize = 8;

const Q8_0_VALUE_TYPE: u32 = 8;
const Q3_K_VALUE_TYPE: u32 = 11;
const Q4_K_VALUE_TYPE: u32 = 12;
const Q5_K_VALUE_TYPE: u32 = 13;
const Q6_K_VALUE_TYPE: u32 = 14;
const IQ4_NL_VALUE_TYPE: u32 = 20;
const IQ3_S_VALUE_TYPE: u32 = 21;
const IQ4_XS_VALUE_TYPE: u32 = 23;
const Q8_0_BLOCK_ELEMENTS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const IQ4_NL_BLOCK_ELEMENTS: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ3_S_BLOCK_ELEMENTS: usize = 256;
const IQ3_S_BLOCK_BYTES: usize = 110;
const Q3_K_BLOCK_ELEMENTS: usize = 256;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q4_K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_ELEMENTS: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_ELEMENTS: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const IQ4_XS_BLOCK_ELEMENTS: usize = 256;
const IQ4_XS_BLOCK_BYTES: usize = 136;

const IQ3_GRID_HEX: &[u8] = include_bytes!("iq3_grid.hex");
const IQ3_GRID_VALUES: [u8; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

fn iq3_hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        b'A'..=b'F' => value - b'A' + 10,
        _ => 0,
    }
}

fn iq3_grid_values() -> Vec<u8> {
    let mut values = Vec::with_capacity(512 * 4);
    for code in 0..512 {
        for lane in 0..4 {
            let value_index = code * 4 + lane;
            let packed_index = value_index / 2;
            let high = iq3_hex_nibble(IQ3_GRID_HEX[packed_index * 2]);
            let low = iq3_hex_nibble(IQ3_GRID_HEX[packed_index * 2 + 1]);
            let packed = (high << 4) | low;
            let grid_index = usize::from((packed >> ((value_index % 2) * 4)) & 0x07);
            values.push(IQ3_GRID_VALUES[grid_index]);
        }
    }
    values
}

static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();

const Q_K_GEMV_SOURCE: &str = concat!(
    include_str!("kernels/quantized/common.cu"),
    include_str!("kernels/quantized/q4_k.cu"),
    include_str!("kernels/quantized/iq3_s.cu"),
    include_str!("kernels/quantized/q8_0.cu"),
    include_str!("kernels/quantized/iq4_nl.cu"),
    include_str!("kernels/quantized/iq4_xs.cu"),
    include_str!("kernels/quantized/q3_k.cu"),
    include_str!("kernels/quantized/q6_k.cu"),
    include_str!("kernels/quantized/q5_k.cu"),
);

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
    warp_kernel: Option<CudaFunction>,
    batch_kernel: Option<CudaFunction>,
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
        // Fixed source and default NVRTC options produce context-independent PTX.
        // Device modules and stream-bound launch handles remain context-owned.
        let ptx = PTX
            .get_or_init(|| {
                compile_ptx(format!(
                    "#define MAX_BATCH_MEMBERS {MAX_BATCH_MEMBERS}\n{Q_K_GEMV_SOURCE}"
                ))
                .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| CudaQuantizedKernelError::Nvrtc(error.clone()))?
            .clone();
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let kernel = module
            .load_function(function_name)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        // Embedding lookups keep the scalar kernel only; GEMV families also
        // compile their warp-cooperative and batched variants for measured
        // selection.
        let warp_name: Option<&'static str> = match function_name {
            "q8_0_gemv" => Some("q8_0_gemv_warp"),
            "iq4_nl_gemv" => Some("iq4_nl_gemv_warp"),
            "iq4_xs_gemv" => Some("iq4_xs_gemv_warp"),
            "q3_k_gemv" => Some("q3_k_gemv_warp"),
            "q6_k_gemv" => Some("q6_k_gemv_warp"),
            "q4_k_gemv" => Some("q4_k_gemv_warp"),
            "q5_k_gemv" => Some("q5_k_gemv_warp"),
            "iq3_s_gemv" => Some("iq3_s_gemv_warp"),
            _ => None,
        };
        let batch_name: Option<&'static str> = match function_name {
            "q8_0_gemv" => Some("q8_0_gemv_warp_batch"),
            "iq4_nl_gemv" => Some("iq4_nl_gemv_warp_batch"),
            "iq4_xs_gemv" => Some("iq4_xs_gemv_warp_batch"),
            "q3_k_gemv" => Some("q3_k_gemv_warp_batch"),
            "q6_k_gemv" => Some("q6_k_gemv_warp_batch"),
            "q4_k_gemv" => Some("q4_k_gemv_warp_batch"),
            "q5_k_gemv" => Some("q5_k_gemv_warp_batch"),
            "iq3_s_gemv" => Some("iq3_s_gemv_warp_batch"),
            "q4_k_embedding" => Some("q4_k_embedding_batch"),
            _ => None,
        };
        let warp_kernel = warp_name
            .map(|name| {
                module
                    .load_function(name)
                    .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))
            })
            .transpose()?;
        let batch_kernel = batch_name
            .map(|name| {
                module
                    .load_function(name)
                    .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))
            })
            .transpose()?;
        Ok(Self {
            stream,
            kernel,
            warp_kernel,
            batch_kernel,
            value_type,
            block_elements,
            block_bytes,
            label,
        })
    }

    /// Validate the quantized weight geometry shared by every variant:
    /// rank-2, block-aligned input extent, exact encoded length.
    fn validate_geometry(
        &self,
        weight: &CudaQuantizedWeight,
    ) -> Result<(usize, usize), CudaQuantizedKernelError> {
        validate_quantized_geometry(
            weight,
            self.value_type,
            self.block_elements,
            self.block_bytes,
            self.label,
        )
    }

    /// Validate a batch-major launch over `members` concurrent inputs and
    /// outputs: each member contributes one `[input_size]` input row and one
    /// `[output_size]` output row.
    fn validate_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        members: usize,
    ) -> Result<(u32, u32, u32, LaunchConfig), CudaQuantizedKernelError> {
        if self.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.stream.context().as_ref() != input.context().as_ref()
            || self.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let (input_size, output_size) = self.validate_geometry(weight)?;
        let expected_input = input_size
            .checked_mul(members)
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
        let members_u32 =
            u32::try_from(members).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        // One warp per weight row (not per (row, member)): the kernel
        // decodes each weight element once and accumulates all members.
        let total_warps =
            u32::try_from(output_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let input_size =
            u32::try_from(input_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let output_size =
            u32::try_from(output_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (total_warps.div_ceil(4), 1, 1),
            block_dim: (4 * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        Ok((input_size, output_size, members_u32, config))
    }

    fn validate(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
    ) -> Result<(u32, u32, LaunchConfig), CudaQuantizedKernelError> {
        if self.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.stream.context().as_ref() != input.context().as_ref()
            || self.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        let (input_size, output_size) = self.validate_geometry(weight)?;
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
        Ok((input_size, output_size, config))
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
        let (input_size, output_size, config) = self.validate(weight, input, output)?;
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

    /// Execute the warp-cooperative variant when one was compiled for this
    /// kernel family. One warp computes each output row: lanes split each
    /// quantization block into consecutive element runs so both weight-byte
    /// and input reads coalesce, and a shuffle reduction produces the row
    /// result. The scalar kernel remains the correctness oracle.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no warp
    /// variant or validation/launch fails.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        const WARPS_PER_BLOCK: u32 = 4;
        let warp_kernel = self.warp_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight(format!(
                "{} has no warp-cooperative variant",
                self.label
            ))
        })?;
        let (input_size, output_size, _) = self.validate(weight, input, output)?;
        let blocks = output_size.div_ceil(WARPS_PER_BLOCK);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: same slices/validation as the scalar path; the warp kernel
        // writes only rows [0, output_size) once each from lane 0.
        unsafe {
            self.stream
                .launch_builder(warp_kernel)
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

    /// Execute the batched warp-cooperative variant over `members`
    /// concurrent inputs/outputs laid out batch-major. The batch-1 warp path
    /// remains the correctness oracle.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    fn execute_warp_batch_inner(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        let batch_kernel = self.batch_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight(format!(
                "{} has no batched warp variant",
                self.label
            ))
        })?;
        if members == 0 {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: 1,
                actual: 0,
            });
        }
        if members > MAX_BATCH_MEMBERS {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "{} weights-read-once batch variant supports at most {MAX_BATCH_MEMBERS} members",
                self.label
            )));
        }
        let (input_size, output_size, members, config) =
            self.validate_batch(weight, input, output, members)?;
        // Safety: same slices/validation as the warp path; the batch kernel
        // writes only [member][row] outputs once each from lane 0.
        unsafe {
            self.stream
                .launch_builder(batch_kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&input_size)
                .arg(&output_size)
                .arg(&members)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `IQ4_NL` matrix-vector kernel.
///
/// The first tensor dimension is the contiguous input (`K`) count and the
/// second is the output (`N`) count, matching GGML's column-major ordering.
/// The kernel keeps encoded weights on the device and does not route them
/// through a host dequantization buffer.
pub struct CudaIq4NlGemv {
    inner: CudaQuantizedGemv,
}

impl CudaIq4NlGemv {
    /// Compile and load the `IQ4_NL` kernel on a new CUDA device context.
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

    /// Compile and load the `IQ4_NL` kernel on an existing context/stream pair.
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
                "iq4_nl_gemv",
                IQ4_NL_VALUE_TYPE,
                IQ4_NL_BLOCK_ELEMENTS,
                IQ4_NL_BLOCK_BYTES,
                "IQ4_NL",
            )?,
        })
    }

    /// Execute one `IQ4_NL` matrix-vector product into a caller-owned output.
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
    }
}

/// A correctness-oriented `IQ3_S` matrix-vector kernel.
///
/// The `IQ3_S` grid is kept as a small device-resident lookup table rather than
/// making the optional CUDA crate depend on the GGUF reader. The table is
/// generated from the canonical GGML packed mapping at construction time.
pub struct CudaIq3SGemv {
    inner: CudaQuantizedGemv,
    grid: CudaSlice<u8>,
}

impl CudaIq3SGemv {
    /// Compile and load the `IQ3_S` kernel on a new CUDA device context.
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

    /// Compile and load the `IQ3_S` kernel on an existing context/stream pair.
    /// Weight buffers and vectors must be allocated from this context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC, module loading, or the
    /// lookup-table upload fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        let grid = stream
            .clone_htod(&iq3_grid_values())
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let inner = CudaQuantizedGemv::from_context(
            context,
            stream,
            "iq3_s_gemv",
            IQ3_S_VALUE_TYPE,
            IQ3_S_BLOCK_ELEMENTS,
            IQ3_S_BLOCK_BYTES,
            "IQ3_S",
        )?;
        Ok(Self { inner, grid })
    }

    /// Execute one `IQ3_S` matrix-vector product into a caller-owned output.
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
        let (input_size, output_size, config) = self.inner.validate(weight, input, output)?;
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(&self.inner.kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&self.grid)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Execute the warp-cooperative `IQ3_S` `GEMV` variant into a
    /// caller-owned output. The launch is asynchronous with respect to the
    /// host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, shapes,
    /// contexts, or launch arguments are invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        const WARPS_PER_BLOCK: u32 = 4;
        let (input_size, output_size, _) = self.inner.validate(weight, input, output)?;
        let warp_kernel = self.inner.warp_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight(
                "IQ3_S has no warp-cooperative variant".to_owned(),
            )
        })?;
        let blocks = output_size.div_ceil(WARPS_PER_BLOCK);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: same slices/validation as the scalar path; the warp kernel
        // writes only rows [0, output_size) once each from lane 0.
        unsafe {
            self.inner
                .stream
                .launch_builder(warp_kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&self.grid)
                .arg(&input_size)
                .arg(&output_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Execute the batched warp-cooperative `IQ3_S` variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        let batch_kernel = self.inner.batch_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight("IQ3_S has no batched warp variant".to_owned())
        })?;
        if members == 0 {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: 1,
                actual: 0,
            });
        }
        let (input_size, output_size, members_u32, config) =
            self.inner.validate_batch(weight, input, output, members)?;
        // Safety: same slices/validation as the warp path; the batch kernel
        // writes only [member][row] outputs once each from lane 0.
        unsafe {
            self.inner
                .stream
                .launch_builder(batch_kernel)
                .arg(weight.encoded_data())
                .arg(input)
                .arg(output)
                .arg(&self.grid)
                .arg(&input_size)
                .arg(&output_size)
                .arg(&members_u32)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `IQ3_S` embedding lookup kernel.
///
/// GGML stores an embedding matrix as `[hidden, vocabulary]`, so each token is
/// one contiguous column of `IQ3_S` blocks. This kernel gathers one such column
/// directly into a device F32 vector.
pub struct CudaIq3SEmbedding {
    inner: CudaQuantizedGemv,
    grid: CudaSlice<u8>,
}

impl CudaIq3SEmbedding {
    /// Compile and load the `IQ3_S` embedding kernel on a new CUDA context.
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

    /// Compile and load the `IQ3_S` embedding kernel on an existing context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or the lookup
    /// table upload fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        let grid = stream
            .clone_htod(&iq3_grid_values())
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let inner = CudaQuantizedGemv::from_context(
            context,
            stream,
            "iq3_s_embedding",
            IQ3_S_VALUE_TYPE,
            IQ3_S_BLOCK_ELEMENTS,
            IQ3_S_BLOCK_BYTES,
            "IQ3_S embedding",
        )?;
        Ok(Self { inner, grid })
    }

    /// Gather one token embedding into a caller-owned device vector.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, token index,
    /// output shape, contexts, or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        token_index: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if self.inner.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.inner.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != IQ3_S_VALUE_TYPE {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: IQ3_S_VALUE_TYPE,
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
        let hidden_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if hidden_size == 0
            || vocabulary_size == 0
            || !hidden_size.is_multiple_of(IQ3_S_BLOCK_ELEMENTS)
        {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "IQ3_S embedding shape is {hidden_size}x{vocabulary_size}; hidden size must be a positive multiple of {IQ3_S_BLOCK_ELEMENTS}"
            )));
        }
        let blocks = hidden_size
            .checked_div(IQ3_S_BLOCK_ELEMENTS)
            .and_then(|value| value.checked_mul(vocabulary_size))
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let expected_bytes = blocks
            .checked_mul(IQ3_S_BLOCK_BYTES)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        if weight.encoded_bytes() != expected_bytes {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "encoded length is {}, expected {expected_bytes}",
                weight.encoded_bytes()
            )));
        }
        if u64::from(token_index) >= dimensions[1] {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "token index {token_index} is outside vocabulary size {vocabulary_size}"
            )));
        }
        if hidden_size > i32::MAX as usize || vocabulary_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if output.len() != hidden_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: hidden_size,
                actual: output.len(),
            });
        }
        let hidden_size =
            u32::try_from(hidden_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            u32::try_from(vocabulary_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (hidden_size.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(&self.inner.kernel)
                .arg(weight.encoded_data())
                .arg(&token_index)
                .arg(output)
                .arg(&self.grid)
                .arg(&hidden_size)
                .arg(&vocabulary_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `Q4_K` embedding lookup kernel.
///
/// `token_embd.weight` in the pinned artifact is `Q4_K` `[hidden, vocab]`;
/// this gathers one vocabulary row into a caller-owned device vector.
pub struct CudaQ4KEmbedding {
    inner: CudaQuantizedGemv,
}

impl CudaQ4KEmbedding {
    /// Compile and load the `Q4_K` embedding kernel on a new CUDA context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when CUDA, NVRTC, or module
    /// loading fails.
    pub fn new(device_index: usize) -> Result<Self, CudaQuantizedKernelError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        Self::from_context(&context, stream)
    }

    /// Compile and load the `Q4_K` embedding kernel on an existing context.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when NVRTC or module loading
    /// fails.
    pub fn from_context(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaQuantizedKernelError> {
        let inner = CudaQuantizedGemv::from_context(
            context,
            stream,
            "q4_k_embedding",
            Q4_K_VALUE_TYPE,
            Q4_K_BLOCK_ELEMENTS,
            Q4_K_BLOCK_BYTES,
            "Q4_K embedding",
        )?;
        Ok(Self { inner })
    }

    /// Gather one token embedding into a caller-owned device vector.
    ///
    /// The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight, token index,
    /// output shape, contexts, or launch arguments are invalid.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        token_index: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if self.inner.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.inner.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != Q4_K_VALUE_TYPE {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: Q4_K_VALUE_TYPE,
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
        let hidden_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if hidden_size == 0
            || vocabulary_size == 0
            || !hidden_size.is_multiple_of(Q4_K_BLOCK_ELEMENTS)
        {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "Q4_K embedding shape is {hidden_size}x{vocabulary_size}; hidden size must be a positive multiple of {Q4_K_BLOCK_ELEMENTS}"
            )));
        }
        let blocks = hidden_size
            .checked_div(Q4_K_BLOCK_ELEMENTS)
            .and_then(|value| value.checked_mul(vocabulary_size))
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
        if u64::from(token_index) >= dimensions[1] {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "token index {token_index} is outside vocabulary size {vocabulary_size}"
            )));
        }
        if hidden_size > i32::MAX as usize || vocabulary_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        if output.len() != hidden_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: hidden_size,
                actual: output.len(),
            });
        }
        let hidden_size =
            u32::try_from(hidden_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            u32::try_from(vocabulary_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (hidden_size.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(&self.inner.kernel)
                .arg(weight.encoded_data())
                .arg(&token_index)
                .arg(output)
                .arg(&hidden_size)
                .arg(&vocabulary_size)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Look up `members` tokens from a device `[members]` index array into a
    /// batch-major `[members][hidden]` output in one launch. The per-element
    /// decode math is identical to the single-token lookup.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the weight geometry, token
    /// indices, or output length is invalid.
    pub fn execute_batch(
        &self,
        weight: &CudaQuantizedWeight,
        token_indices: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        if self.inner.stream.context().as_ref() != weight.encoded_data().context().as_ref()
            || self.inner.stream.context().as_ref() != token_indices.context().as_ref()
            || self.inner.stream.context().as_ref() != output.context().as_ref()
        {
            return Err(CudaQuantizedKernelError::ContextMismatch);
        }
        if weight.value_type() != Q4_K_VALUE_TYPE {
            return Err(CudaQuantizedKernelError::UnsupportedValueType {
                expected: Q4_K_VALUE_TYPE,
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
        let hidden_size =
            usize::try_from(dimensions[0]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            usize::try_from(dimensions[1]).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        if hidden_size == 0
            || vocabulary_size == 0
            || !hidden_size.is_multiple_of(Q4_K_BLOCK_ELEMENTS)
        {
            return Err(CudaQuantizedKernelError::InvalidWeight(format!(
                "Q4_K embedding shape is {hidden_size}x{vocabulary_size}; hidden size must be a positive multiple of {Q4_K_BLOCK_ELEMENTS}"
            )));
        }
        let blocks = hidden_size
            .checked_div(Q4_K_BLOCK_ELEMENTS)
            .and_then(|value| value.checked_mul(vocabulary_size))
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
        if hidden_size > i32::MAX as usize || vocabulary_size > i32::MAX as usize {
            return Err(CudaQuantizedKernelError::ShapeOverflow);
        }
        let members = token_indices.len();
        if members == 0 {
            return Err(CudaQuantizedKernelError::InputLength {
                expected: 1,
                actual: 0,
            });
        }
        if output.len() != members * hidden_size {
            return Err(CudaQuantizedKernelError::OutputLength {
                expected: members * hidden_size,
                actual: output.len(),
            });
        }
        let hidden_size =
            u32::try_from(hidden_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let vocabulary_size =
            u32::try_from(vocabulary_size).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let members =
            u32::try_from(members).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let total = (members as usize)
            .checked_mul(hidden_size as usize)
            .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
        let total = u32::try_from(total).map_err(|_| CudaQuantizedKernelError::ShapeOverflow)?;
        let config = LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_kernel = self.inner.batch_kernel.as_ref().ok_or_else(|| {
            CudaQuantizedKernelError::InvalidWeight(
                "Q4_K embedding has no batched variant".to_owned(),
            )
        })?;
        // Safety: the device slices are allocated by cudarc, remain alive for
        // the launch, and have lengths checked against the kernel's shape.
        unsafe {
            self.inner
                .stream
                .launch_builder(batch_kernel)
                .arg(weight.encoded_data())
                .arg(token_indices)
                .arg(output)
                .arg(&hidden_size)
                .arg(&vocabulary_size)
                .arg(&members)
                .launch(config)
                .map_err(|error| CudaQuantizedKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

/// A correctness-oriented `IQ4_XS` matrix-vector kernel.
///
/// It shares the GGML `[K,N]` contract and launch validation with
/// [`CudaIq4NlGemv`], but decodes per-group scales and the 136-byte block
/// layout used by GGML value type 23.
pub struct CudaIq4XsGemv {
    inner: CudaQuantizedGemv,
}

impl CudaIq4XsGemv {
    /// Compile and load the `IQ4_XS` kernel on a new CUDA device context.
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

    /// Compile and load the `IQ4_XS` kernel on an existing context/stream pair.
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
                "iq4_xs_gemv",
                IQ4_XS_VALUE_TYPE,
                IQ4_XS_BLOCK_ELEMENTS,
                IQ4_XS_BLOCK_BYTES,
                "IQ4_XS",
            )?,
        })
    }

    /// Execute one `IQ4_XS` matrix-vector product into a caller-owned output.
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
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

    /// Execute the warp-cooperative `GEMV` variant into a caller-owned
    /// output. The launch is asynchronous with respect to the host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family lacks a warp
    /// variant or the weight, shapes, contexts, or launch arguments are
    /// invalid.
    pub fn execute_warp(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner.execute_warp(weight, input, output)
    }

    /// Execute the batched warp-cooperative variant over `members`
    /// batch-major input/output rows.
    ///
    /// # Errors
    ///
    /// Returns [`CudaQuantizedKernelError`] when the family has no batched
    /// variant or validation/launch fails.
    pub fn execute_warp_batch(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        members: usize,
    ) -> Result<(), CudaQuantizedKernelError> {
        self.inner
            .execute_warp_batch_inner(weight, input, output, members)
    }
}

fn validate_quantized_geometry(
    weight: &CudaQuantizedWeight,
    value_type: u32,
    block_elements: usize,
    block_bytes: usize,
    label: &str,
) -> Result<(usize, usize), CudaQuantizedKernelError> {
    if weight.value_type() != value_type {
        return Err(CudaQuantizedKernelError::UnsupportedValueType {
            expected: value_type,
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
    if input_size == 0 || output_size == 0 || !input_size.is_multiple_of(block_elements) {
        return Err(CudaQuantizedKernelError::InvalidWeight(format!(
            "{label} shape is {input_size}x{output_size}; input size must be a positive multiple of {block_elements}"
        )));
    }
    let blocks = input_size
        .checked_div(block_elements)
        .and_then(|value| value.checked_mul(output_size))
        .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
    let expected_bytes = blocks
        .checked_mul(block_bytes)
        .ok_or(CudaQuantizedKernelError::ShapeOverflow)?;
    if weight.encoded_bytes() != expected_bytes {
        return Err(CudaQuantizedKernelError::InvalidWeight(format!(
            "encoded length is {}, expected {expected_bytes}",
            weight.encoded_bytes()
        )));
    }
    if blocks > (i32::MAX as usize) / block_bytes {
        return Err(CudaQuantizedKernelError::ShapeOverflow);
    }
    if input_size > i32::MAX as usize || output_size > i32::MAX as usize {
        return Err(CudaQuantizedKernelError::ShapeOverflow);
    }
    Ok((input_size, output_size))
}

#[path = "q4_q8_1.rs"]
mod q4_q8_1;
pub use q4_q8_1::CudaQ4KQ8_1Gemv;
