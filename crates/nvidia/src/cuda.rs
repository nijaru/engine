use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use engine_core::{
    BackendError, ExecutionMetrics, ExecutionPlan, ExecutionSegment, HybridStateSet,
    NvidiaDispatcher,
};

const INPUT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
// Column-major 2x4 matrix. Its rows are [1, 2, 1, 2] and [2, 1, 2, 1].
const WEIGHTS: [f32; 8] = [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0];
const EXPECTED_OUTPUT: [f32; 2] = [16.0, 14.0];

#[derive(Debug)]
pub enum CudaRuntimeError {
    Driver(String),
    Blas(String),
    InvalidOutput(String),
}

impl fmt::Display for CudaRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
            Self::Blas(message) => write!(f, "cuBLAS error: {message}"),
            Self::InvalidOutput(message) => write!(f, "CUDA reference output error: {message}"),
        }
    }
}

impl std::error::Error for CudaRuntimeError {}

/// A concrete CUDA dispatcher for a tiny stateless F32 linear layer.
///
/// This is a backend conformance path, not a Qwen3.8 implementation. It owns
/// the CUDA context, stream, cuBLAS handle, and device buffers. The fixed data
/// makes the path deterministic while the core dispatcher contract still
/// receives the real execution plan and segment.
pub struct CudaReferenceDispatcher {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    weights: CudaSlice<f32>,
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    last_output: Option<Vec<f32>>,
}

impl CudaReferenceDispatcher {
    /// Create the CUDA context, stream, cuBLAS handle, and reference buffers.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRuntimeError`] when CUDA or cuBLAS initialization fails.
    pub fn new(device_index: usize) -> Result<Self, CudaRuntimeError> {
        let context = CudaContext::new(device_index)
            .map_err(|error| CudaRuntimeError::Driver(error.to_string()))?;
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|error| CudaRuntimeError::Blas(error.to_string()))?;
        let weights = stream
            .clone_htod(&WEIGHTS)
            .map_err(|error| CudaRuntimeError::Driver(error.to_string()))?;
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
            weights,
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

    fn run_linear_layer(&mut self) -> Result<ExecutionMetrics, CudaRuntimeError> {
        let started = Instant::now();
        // cuBLAS uses column-major matrices. WEIGHTS is a 2x4 matrix, so the
        // output is a two-element vector with m=2 and n=4.
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
                    &self.weights,
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
        _state: &mut HybridStateSet,
    ) -> Result<ExecutionMetrics, BackendError> {
        if segment.batch_size() != 1 || segment.token_count() != 1 {
            return Err(BackendError::ExecutionFailed(
                "CUDA reference dispatcher only accepts batch=1 token=1".to_owned(),
            ));
        }
        self.run_linear_layer()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
    }
}
