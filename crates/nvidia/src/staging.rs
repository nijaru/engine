//! Full-model weight staging for the pinned Qwen3.8-27B GGUF artifact.
//!
//! Owns the device-resident encoded weights for the text-only language
//! path: one `CudaWeightStore` upload per tensor with duplicate validation
//! and a pre-flight byte estimate, plus quantized GEMV kernel selection by
//! value type for the dispatcher. No host F32 copies exist on this path.
//! The tensor source is a trait so `engine-gguf` remains optional; the
//! GGUF implementation of that trait lives in the integration tests and the
//! future server wiring, not in this crate.

use std::io::Read;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::cuda::{CudaQuantizedWeight, CudaWeightError, CudaWeightStore};
use crate::quantized::{
    CudaIq3SGemv, CudaIq4NlGemv, CudaIq4XsGemv, CudaQ3KGemv, CudaQ4KGemv, CudaQ5KGemv, CudaQ6KGemv,
    CudaQ8_0Gemv, CudaQuantizedKernelError,
};

/// Errors returned while staging the full model onto the device.
#[derive(Debug)]
pub enum CudaWeightStagingError {
    Weight(CudaWeightError),
    Kernel(CudaQuantizedKernelError),
    /// A required tensor is missing from the source.
    MissingTensor(String),
    /// The staged weights exceed the caller-supplied device byte budget.
    BudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
    },
}

impl std::fmt::Display for CudaWeightStagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Weight(error) => write!(f, "weight staging error: {error}"),
            Self::Kernel(error) => write!(f, "quantized kernel error: {error}"),
            Self::MissingTensor(name) => write!(f, "required tensor {name:?} is missing"),
            Self::BudgetExceeded {
                required_bytes,
                budget_bytes,
            } => write!(
                f,
                "staged weights need {required_bytes} bytes, exceeding the {budget_bytes}-byte device budget"
            ),
        }
    }
}

impl std::error::Error for CudaWeightStagingError {}

impl From<CudaWeightError> for CudaWeightStagingError {
    fn from(error: CudaWeightError) -> Self {
        Self::Weight(error)
    }
}

impl From<CudaQuantizedKernelError> for CudaWeightStagingError {
    fn from(error: CudaQuantizedKernelError) -> Self {
        Self::Kernel(error)
    }
}

/// One tensor in the staging set: its descriptor, GGUF value type,
/// encoded byte length, and a reader over the encoded payload.
pub struct StagedTensorSource<'a> {
    /// Logical tensor descriptor carrying the real dimensions.
    pub spec: engine_core::WeightTensorSpec,
    /// GGUF value type (12 = `Q4_K`, 14 = `Q6_K`, ...).
    pub value_type: u32,
    /// Encoded payload length in bytes.
    pub encoded_bytes: u64,
    /// Reader over the encoded payload.
    pub reader: Box<dyn Read + 'a>,
}

/// The per-value-type GEMV kernel entry compiled once during staging.
///
/// Each variant owns its NVRTC module; the dispatcher picks by value type.
pub enum QwenGemvKernel {
    Q3K(CudaQ3KGemv),
    Q4K(CudaQ4KGemv),
    Q5K(CudaQ5KGemv),
    Q6K(CudaQ6KGemv),
    Q8_0(CudaQ8_0Gemv),
    Iq4Nl(CudaIq4NlGemv),
    Iq3S(CudaIq3SGemv),
    Iq4Xs(CudaIq4XsGemv),
}

impl QwenGemvKernel {
    /// Run the GEMV for one tensor.
    ///
    /// # Errors
    ///
    /// Returns the kernel's error when shape or launch validation fails.
    pub fn execute(
        &self,
        weight: &CudaQuantizedWeight,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaQuantizedKernelError> {
        match self {
            Self::Q3K(kernel) => kernel.execute(weight, input, output),
            Self::Q4K(kernel) => kernel.execute(weight, input, output),
            Self::Q5K(kernel) => kernel.execute(weight, input, output),
            Self::Q6K(kernel) => kernel.execute(weight, input, output),
            Self::Q8_0(kernel) => kernel.execute(weight, input, output),
            Self::Iq4Nl(kernel) => kernel.execute(weight, input, output),
            Self::Iq3S(kernel) => kernel.execute(weight, input, output),
            Self::Iq4Xs(kernel) => kernel.execute(weight, input, output),
        }
    }

    #[must_use]
    pub const fn value_type(&self) -> u32 {
        match self {
            Self::Q3K(_) => 11,
            Self::Q4K(_) => 12,
            Self::Q5K(_) => 13,
            Self::Q6K(_) => 14,
            Self::Q8_0(_) => 8,
            Self::Iq4Nl(_) => 20,
            Self::Iq3S(_) => 21,
            Self::Iq4Xs(_) => 23,
        }
    }
}

/// Full text-path weight staging for the pinned Qwen3.8-27B artifact.
///
/// `stage` uploads every supplied tensor as opaque encoded bytes and
/// compiles one GEMV kernel per encountered value type. The caller supplies
/// the tensor set, so MTP (`blk.64.*`) and vision tensors stay excluded by
/// construction.
pub struct CudaQwen35Weights {
    store: CudaWeightStore,
    gemv: Vec<QwenGemvKernel>,
}

impl CudaQwen35Weights {
    /// Stage the supplied tensors, first checking that their encoded bytes
    /// fit within `budget_bytes`.
    ///
    /// # Errors
    ///
    /// Returns an error when the budget check, upload, or kernel
    /// compilation fails.
    pub fn stage(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        budget_bytes: u64,
        tensors: Vec<StagedTensorSource<'_>>,
    ) -> Result<Self, CudaWeightStagingError> {
        let required_bytes = tensors
            .iter()
            .map(|tensor| tensor.encoded_bytes)
            .try_fold(0_u64, u64::checked_add)
            .ok_or(CudaWeightStagingError::BudgetExceeded {
                required_bytes: u64::MAX,
                budget_bytes,
            })?;
        if required_bytes > budget_bytes {
            return Err(CudaWeightStagingError::BudgetExceeded {
                required_bytes,
                budget_bytes,
            });
        }
        let mut store = CudaWeightStore::new(stream.clone());
        let mut seen_types: Vec<u32> = Vec::new();
        for tensor in tensors {
            let mut reader = tensor.reader;
            store
                .materialize_quantized(
                    tensor.spec,
                    tensor.value_type,
                    tensor.encoded_bytes,
                    &mut reader,
                )
                .map_err(CudaWeightStagingError::Weight)?;
            if !seen_types.contains(&tensor.value_type) {
                seen_types.push(tensor.value_type);
            }
        }
        let mut gemv = Vec::new();
        for value_type in seen_types {
            gemv.push(match value_type {
                8 => QwenGemvKernel::Q8_0(CudaQ8_0Gemv::from_context(context, stream.clone())?),
                11 => QwenGemvKernel::Q3K(CudaQ3KGemv::from_context(context, stream.clone())?),
                12 => QwenGemvKernel::Q4K(CudaQ4KGemv::from_context(context, stream.clone())?),
                13 => QwenGemvKernel::Q5K(CudaQ5KGemv::from_context(context, stream.clone())?),
                14 => QwenGemvKernel::Q6K(CudaQ6KGemv::from_context(context, stream.clone())?),
                20 => QwenGemvKernel::Iq4Nl(CudaIq4NlGemv::from_context(context, stream.clone())?),
                21 => QwenGemvKernel::Iq3S(CudaIq3SGemv::from_context(context, stream.clone())?),
                23 => QwenGemvKernel::Iq4Xs(CudaIq4XsGemv::from_context(context, stream.clone())?),
                other => {
                    return Err(CudaWeightStagingError::Kernel(
                        CudaQuantizedKernelError::UnsupportedValueType {
                            expected: 0,
                            actual: other,
                        },
                    ));
                }
            });
        }
        Ok(Self { store, gemv })
    }

    /// Look up the staged encoded weight for a tensor name.
    #[must_use]
    pub fn quantized_tensor(&self, name: &str) -> Option<&CudaQuantizedWeight> {
        self.store.quantized_tensor(name)
    }

    /// Look up the compiled GEMV kernel for a value type.
    #[must_use]
    pub fn gemv_for(&self, value_type: u32) -> Option<&QwenGemvKernel> {
        self.gemv
            .iter()
            .find(|kernel| kernel.value_type() == value_type)
    }
}
