use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use engine_core::{
    BackendError, DataType, ExecutionMetrics, ExecutionPlan, ExecutionSegment, F32BlockStream,
    HybridStateSet, NvidiaDispatcher, WeightBinding, WeightTensorSpec,
};

const INPUT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
// Column-major 2x4 matrix. Its rows are [1, 2, 1, 2] and [2, 1, 2, 1].
const WEIGHTS: [f32; 8] = [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0];
const EXPECTED_OUTPUT: [f32; 2] = [16.0, 14.0];

#[derive(Debug)]
pub enum CudaRuntimeError {
    Driver(String),
    Blas(String),
    Weight(String),
    InvalidOutput(String),
}

impl fmt::Display for CudaRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
            Self::Blas(message) => write!(f, "cuBLAS error: {message}"),
            Self::Weight(message) => write!(f, "CUDA weight error: {message}"),
            Self::InvalidOutput(message) => write!(f, "CUDA reference output error: {message}"),
        }
    }
}

impl std::error::Error for CudaRuntimeError {}

#[derive(Debug)]
pub enum CudaWeightError {
    Driver(String),
    Source(String),
    UnsupportedDtype {
        name: String,
        dtype: DataType,
    },
    DuplicateTensor(String),
    EmptyBlock(String),
    BlockOverflow {
        name: String,
        offset: usize,
        block_elements: usize,
        capacity: usize,
    },
    ElementCountMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    MissingTensor(String),
    BindingMismatch(String),
}

impl fmt::Display for CudaWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
            Self::Source(message) => write!(f, "weight source error: {message}"),
            Self::UnsupportedDtype { name, dtype } => {
                write!(
                    f,
                    "tensor {name:?} has unsupported materialized dtype {dtype}"
                )
            }
            Self::DuplicateTensor(name) => {
                write!(f, "CUDA weight store already contains tensor {name:?}")
            }
            Self::EmptyBlock(name) => write!(f, "tensor {name:?} source returned an empty block"),
            Self::BlockOverflow {
                name,
                offset,
                block_elements,
                capacity,
            } => write!(
                f,
                "tensor {name:?} block at offset {offset} with {block_elements} elements exceeds capacity {capacity}"
            ),
            Self::ElementCountMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} source produced {actual} elements, expected {expected}"
            ),
            Self::MissingTensor(name) => write!(f, "CUDA weight store is missing tensor {name:?}"),
            Self::BindingMismatch(reason) => write!(f, "CUDA weight binding mismatch: {reason}"),
        }
    }
}

impl std::error::Error for CudaWeightError {}

/// One F32 tensor materialized in backend-owned CUDA memory.
pub struct CudaF32Weight {
    spec: WeightTensorSpec,
    data: CudaSlice<f32>,
}

impl CudaF32Weight {
    #[must_use]
    pub fn spec(&self) -> &WeightTensorSpec {
        &self.spec
    }
}

/// Backend-owned store for canonical F32 tensor materialization.
///
/// This is deliberately a bring-up path. It consumes bounded decoded blocks
/// and uploads them into one contiguous allocation; it is not a replacement
/// for the eventual mixed GGUF quantized kernels.
pub struct CudaWeightStore {
    stream: Arc<CudaStream>,
    tensors: BTreeMap<String, CudaF32Weight>,
}

impl CudaWeightStore {
    #[must_use]
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            tensors: BTreeMap::new(),
        }
    }

    /// Materialize a bounded F32 source into a device allocation.
    ///
    /// The allocation is published only after the source has produced exactly
    /// the declared number of elements. A failed upload drops the private
    /// allocation and leaves the store unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CudaWeightError`] when the source shape/dtype is unsupported,
    /// a block does not fit, the copy fails, or the source is incomplete.
    pub fn materialize_f32<S>(
        &mut self,
        source: &mut S,
    ) -> Result<WeightTensorSpec, CudaWeightError>
    where
        S: F32BlockStream,
        S::Error: fmt::Display,
    {
        let spec = source.spec().clone();
        if spec.dtype() != DataType::F32 {
            return Err(CudaWeightError::UnsupportedDtype {
                name: spec.name().to_owned(),
                dtype: spec.dtype(),
            });
        }
        if self.tensors.contains_key(spec.name()) {
            return Err(CudaWeightError::DuplicateTensor(spec.name().to_owned()));
        }
        let capacity = usize::try_from(spec.element_count()).map_err(|_| {
            CudaWeightError::BindingMismatch(format!(
                "tensor {:?} element count does not fit this host",
                spec.name()
            ))
        })?;
        let mut device = unsafe { self.stream.alloc::<f32>(capacity) }
            .map_err(|error| CudaWeightError::Driver(error.to_string()))?;
        let mut offset = 0_usize;
        while let Some(block) = source
            .next_block()
            .map_err(|error| CudaWeightError::Source(error.to_string()))?
        {
            if block.is_empty() {
                return Err(CudaWeightError::EmptyBlock(spec.name().to_owned()));
            }
            let end =
                offset
                    .checked_add(block.len())
                    .ok_or_else(|| CudaWeightError::BlockOverflow {
                        name: spec.name().to_owned(),
                        offset,
                        block_elements: block.len(),
                        capacity,
                    })?;
            if end > capacity {
                return Err(CudaWeightError::BlockOverflow {
                    name: spec.name().to_owned(),
                    offset,
                    block_elements: block.len(),
                    capacity,
                });
            }
            let mut destination = device.try_slice_mut(offset..end).ok_or_else(|| {
                CudaWeightError::BlockOverflow {
                    name: spec.name().to_owned(),
                    offset,
                    block_elements: block.len(),
                    capacity,
                }
            })?;
            self.stream
                .memcpy_htod(&block, &mut destination)
                .map_err(|error| CudaWeightError::Driver(error.to_string()))?;
            // Blocks are ordinary pageable Vec allocations. Synchronize before
            // dropping each block so an asynchronous H2D copy cannot outlive
            // its host source. A pinned-buffer pipeline can replace this in a
            // measured loading optimization without changing the boundary.
            self.stream
                .synchronize()
                .map_err(|error| CudaWeightError::Driver(error.to_string()))?;
            offset = end;
        }
        if offset != capacity {
            return Err(CudaWeightError::ElementCountMismatch {
                name: spec.name().to_owned(),
                expected: capacity,
                actual: offset,
            });
        }
        self.tensors.insert(
            spec.name().to_owned(),
            CudaF32Weight {
                spec: spec.clone(),
                data: device,
            },
        );
        Ok(spec)
    }

    /// Validate that every logical tensor in a plan binding has an identical
    /// materialized tensor in this store.
    ///
    /// # Errors
    ///
    /// Returns [`CudaWeightError`] when a tensor is missing or its descriptor
    /// differs from the prepared binding.
    pub fn validate_binding(&self, binding: &WeightBinding) -> Result<(), CudaWeightError> {
        for spec in binding.tensors() {
            let Some(weight) = self.tensors.get(spec.name()) else {
                return Err(CudaWeightError::MissingTensor(spec.name().to_owned()));
            };
            if weight.spec() != spec {
                return Err(CudaWeightError::BindingMismatch(format!(
                    "tensor {:?} descriptor differs from materialized storage",
                    spec.name()
                )));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&CudaF32Weight> {
        self.tensors.get(name)
    }

    /// Copy one materialized tensor back to the host for correctness checks or
    /// diagnostics. Production execution should keep this off the hot path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaWeightError::MissingTensor`] when the name is not bound or
    /// a device-to-host copy fails.
    pub fn copy_to_host(&self, name: &str) -> Result<Vec<f32>, CudaWeightError> {
        let weight = self
            .tensor(name)
            .ok_or_else(|| CudaWeightError::MissingTensor(name.to_owned()))?;
        self.stream
            .clone_dtoh(&weight.data)
            .map_err(|error| CudaWeightError::Driver(error.to_string()))
    }
}

/// A concrete CUDA dispatcher for a tiny stateless F32 linear layer.
///
/// This is a backend conformance path, not a Qwen3.8 implementation. It owns
/// the CUDA context, stream, cuBLAS handle, and device buffers. The fixed data
/// makes the path deterministic while the core dispatcher contract still
/// receives the real execution plan, weight binding, and segment.
pub struct CudaReferenceDispatcher {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    weight_store: CudaWeightStore,
    reference_weight_name: String,
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    last_output: Option<Vec<f32>>,
}

impl CudaReferenceDispatcher {
    /// Create the CUDA context, stream, cuBLAS handle, and reference buffers.
    /// The fixed F32 matrix is routed through the same source/materialization
    /// boundary used by format adapters.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRuntimeError`] when CUDA or cuBLAS initialization fails.
    pub fn new(device_index: usize) -> Result<Self, CudaRuntimeError> {
        let spec = WeightTensorSpec::new("reference.weight", vec![2, 4], DataType::F32)
            .map_err(|error| CudaRuntimeError::Weight(error.to_string()))?;
        let mut source = SingleF32BlockStream {
            spec,
            block: Some(WEIGHTS.to_vec()),
        };
        Self::from_f32_source(device_index, &mut source)
    }

    /// Create the reference dispatcher from a real bounded F32 source.
    ///
    /// This method is useful for validating a format adapter independently of
    /// the Qwen model. It still executes only the fixed 2x4 reference GEMV.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRuntimeError`] when source materialization or CUDA setup
    /// fails.
    pub fn from_f32_source<S>(device_index: usize, source: &mut S) -> Result<Self, CudaRuntimeError>
    where
        S: F32BlockStream,
        S::Error: fmt::Display,
    {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaRuntimeError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|error| CudaRuntimeError::Blas(error.to_string()))?;
        let mut weight_store = CudaWeightStore::new(stream.clone());
        let spec = weight_store
            .materialize_f32(source)
            .map_err(|error| CudaRuntimeError::Weight(error.to_string()))?;
        let input = stream
            .clone_htod(&INPUT)
            .map_err(|error| CudaRuntimeError::Driver(error.to_string()))?;
        let output = stream
            .alloc_zeros::<f32>(EXPECTED_OUTPUT.len())
            .map_err(|error| CudaRuntimeError::Driver(error.to_string()))?;
        Ok(Self {
            _context: context,
            stream,
            blas,
            weight_store,
            reference_weight_name: spec.name().to_owned(),
            input,
            output,
            last_output: None,
        })
    }

    #[must_use]
    pub fn last_output(&self) -> Option<&[f32]> {
        self.last_output.as_deref()
    }

    #[must_use]
    pub fn expected_output(&self) -> &[f32] {
        &EXPECTED_OUTPUT
    }

    #[must_use]
    pub fn weight_spec(&self) -> Option<&WeightTensorSpec> {
        self.weight_store
            .tensor(&self.reference_weight_name)
            .map(CudaF32Weight::spec)
    }

    /// Copy the reference tensor back to the host for a materialization
    /// correctness check. This is not part of the execution hot path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRuntimeError::Weight`] when the tensor is unavailable or
    /// the device-to-host copy fails.
    pub fn copy_weight_to_host(&self) -> Result<Vec<f32>, CudaRuntimeError> {
        self.weight_store
            .copy_to_host(&self.reference_weight_name)
            .map_err(|error| CudaRuntimeError::Weight(error.to_string()))
    }

    fn run_linear_layer(
        &mut self,
        binding: &WeightBinding,
    ) -> Result<ExecutionMetrics, CudaRuntimeError> {
        if binding.tensor(&self.reference_weight_name).is_none() {
            return Err(CudaRuntimeError::Weight(format!(
                "weight binding does not include reference tensor {:?}",
                self.reference_weight_name
            )));
        }
        self.weight_store
            .validate_binding(binding)
            .map_err(|error| CudaRuntimeError::Weight(error.to_string()))?;
        let weight = self
            .weight_store
            .tensor(&self.reference_weight_name)
            .ok_or_else(|| {
                CudaRuntimeError::Weight(format!(
                    "reference tensor {:?} is not materialized",
                    self.reference_weight_name
                ))
            })?;
        let started = Instant::now();
        // cuBLAS uses column-major matrices. The reference weight is a 2x4
        // matrix, so the output is a two-element vector with m=2 and n=4.
        // Safety: the device slices remain alive for the duration of the call;
        // their lengths are 8, 4, and 2 elements, matching the 2x4 GEMV
        // configuration and its leading dimensions.
        unsafe {
            self.blas
                .gemv(
                    GemvConfig {
                        trans: cublasOperation_t::CUBLAS_OP_N,
                        m: 2,
                        n: 4,
                        alpha: 1.0,
                        lda: 2,
                        incx: 1,
                        beta: 0.0,
                        incy: 1,
                    },
                    &weight.data,
                    &self.input,
                    &mut self.output,
                )
                .map_err(|error| CudaRuntimeError::Blas(error.to_string()))?;
        }
        let output = self
            .stream
            .clone_dtoh(&self.output)
            .map_err(|error| CudaRuntimeError::Driver(error.to_string()))?;
        if output.len() != EXPECTED_OUTPUT.len()
            || output
                .iter()
                .zip(EXPECTED_OUTPUT)
                .any(|(actual, expected)| (actual - expected).abs() > 1e-5)
        {
            return Err(CudaRuntimeError::InvalidOutput(format!(
                "got {output:?}, expected {EXPECTED_OUTPUT:?}"
            )));
        }
        self.last_output = Some(output);
        let elapsed_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(ExecutionMetrics::new(elapsed_nanos, 0, 0))
    }
}

impl NvidiaDispatcher for CudaReferenceDispatcher {
    fn dispatch(
        &mut self,
        _plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        weights: &WeightBinding,
        _state: &mut HybridStateSet,
    ) -> Result<ExecutionMetrics, BackendError> {
        if segment.batch_size() != 1 || segment.token_count() != 1 {
            return Err(BackendError::ExecutionFailed(
                "CUDA reference dispatcher only accepts batch=1 token=1".to_owned(),
            ));
        }
        self.run_linear_layer(weights)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
    }
}

struct SingleF32BlockStream {
    spec: WeightTensorSpec,
    block: Option<Vec<f32>>,
}

impl F32BlockStream for SingleF32BlockStream {
    type Error = std::convert::Infallible;

    fn spec(&self) -> &WeightTensorSpec {
        &self.spec
    }

    fn next_block(&mut self) -> Result<Option<Vec<f32>>, Self::Error> {
        Ok(self.block.take())
    }
}
