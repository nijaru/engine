//! Compute-backend capability and execution boundary.

use std::fmt;

use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionPlan, ExecutionSegment, PlanError,
};
use crate::state::{InferenceStateSet, StateLocation};
use crate::tensor::{DataType, Quantization};

pub use crate::device::DeviceId;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BackendId(String);

impl BackendId {
    /// # Errors
    ///
    /// Returns [`BackendIdError`] when `value` is empty or whitespace-only.
    pub fn new(value: impl Into<String>) -> Result<Self, BackendIdError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(BackendIdError);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendIdError;

impl fmt::Display for BackendIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("backend identity must not be empty")
    }
}

impl std::error::Error for BackendIdError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendSubmissionId(u64);

impl BackendSubmissionId {
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendFeatures {
    data_types: Vec<DataType>,
    quantizations: Vec<Quantization>,
    graph_capture: bool,
    streams: bool,
}

impl BackendFeatures {
    #[must_use]
    pub fn new(
        data_types: Vec<DataType>,
        quantizations: Vec<Quantization>,
        graph_capture: bool,
        streams: bool,
    ) -> Self {
        Self {
            data_types,
            quantizations,
            graph_capture,
            streams,
        }
    }

    #[must_use]
    pub fn supports_dtype(&self, dtype: DataType) -> bool {
        self.data_types.contains(&dtype)
    }

    #[must_use]
    pub fn supports_quantization(&self, quantization: Quantization) -> bool {
        self.quantizations.contains(&quantization)
    }

    #[must_use]
    pub const fn supports_graph_capture(&self) -> bool {
        self.graph_capture
    }

    #[must_use]
    pub const fn supports_streams(&self) -> bool {
        self.streams
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    backend: BackendId,
    device: DeviceId,
    kind: BackendKind,
    memory_bytes: u64,
    features: BackendFeatures,
}

impl BackendCapabilities {
    #[must_use]
    pub fn new(
        backend: BackendId,
        device: DeviceId,
        kind: BackendKind,
        memory_bytes: u64,
        features: BackendFeatures,
    ) -> Self {
        Self {
            backend,
            device,
            kind,
            memory_bytes,
            features,
        }
    }

    #[must_use]
    pub fn backend(&self) -> &BackendId {
        &self.backend
    }

    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    #[must_use]
    pub const fn kind(&self) -> BackendKind {
        self.kind
    }

    #[must_use]
    pub const fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    #[must_use]
    pub fn supports_dtype(&self, dtype: DataType) -> bool {
        self.features.supports_dtype(dtype)
    }

    #[must_use]
    pub fn supports_quantization(&self, quantization: Quantization) -> bool {
        self.features.supports_quantization(quantization)
    }

    #[must_use]
    pub const fn supports_graph_capture(&self) -> bool {
        self.features.supports_graph_capture()
    }

    #[must_use]
    pub const fn supports_streams(&self) -> bool {
        self.features.supports_streams()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendError {
    Unsupported(&'static str),
    PlanMismatch,
    InvalidPlan(PlanError),
    StateCountMismatch,
    StateMismatch(&'static str),
    UnknownSubmission(BackendSubmissionId),
    ExecutionFailed(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(capability) => write!(f, "backend does not support {capability}"),
            Self::PlanMismatch => f.write_str("execution plan targets a different backend/device"),
            Self::InvalidPlan(error) => write!(f, "invalid execution plan: {error}"),
            Self::StateCountMismatch => {
                f.write_str("execution batch and inference-state counts differ")
            }
            Self::StateMismatch(reason) => write!(f, "state does not satisfy execution: {reason}"),
            Self::UnknownSubmission(id) => write!(f, "backend submission {} is unknown", id.get()),
            Self::ExecutionFailed(reason) => write!(f, "execution failed: {reason}"),
        }
    }
}

impl std::error::Error for BackendError {}

pub trait ComputeBackend: Send {
    fn capabilities(&self) -> &BackendCapabilities;

    /// # Errors
    ///
    /// Returns an error when the plan is not prepared for this backend or a
    /// state set does not satisfy its request segment.
    fn validate_execution(
        &self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: &[InferenceStateSet],
    ) -> Result<(), BackendError> {
        if plan.backend() != self.capabilities().backend()
            || plan.device() != self.capabilities().device()
        {
            return Err(BackendError::PlanMismatch);
        }
        plan.validate_batch(batch)
            .map_err(BackendError::InvalidPlan)?;
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        for (segment, state) in batch.segments().iter().zip(states) {
            self.validate_segment_state(segment, state)?;
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error when state does not satisfy one request segment.
    fn validate_segment_state(
        &self,
        segment: &ExecutionSegment,
        state: &InferenceStateSet,
    ) -> Result<(), BackendError> {
        if segment
            .state_requirements()
            .iter()
            .any(|requirement| !state.contains(*requirement))
        {
            return Err(BackendError::StateMismatch(
                "segment state requirement is missing or has the wrong layout",
            ));
        }
        if state
            .token_position()
            .is_some_and(|position| position != segment.state_position())
        {
            return Err(BackendError::StateMismatch(
                "state is at a different prefix position",
            ));
        }
        if state.location() == Some(StateLocation::Device(self.capabilities().device())) {
            return Ok(());
        }
        if matches!(state.location(), Some(StateLocation::Device(_))) {
            return Err(BackendError::StateMismatch(
                "state is resident on a different device",
            ));
        }
        Ok(())
    }

    /// Release backend-owned physical resources associated with one logical
    /// inference-state set. Stateless backends may keep the default no-op.
    ///
    /// This is a lifecycle operation, not model execution. Callers must not
    /// release state while a submission still owns it.
    ///
    /// # Errors
    ///
    /// Returns a backend error when physical state cannot be released safely.
    fn release_inference_state(&mut self, _state: &InferenceStateSet) -> Result<(), BackendError> {
        Ok(())
    }

    /// Submit a multi-request batch without requiring host synchronization
    /// with its completion.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the batch cannot be submitted.
    fn submit(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<BackendSubmissionId, BackendError>;

    /// Poll one submitted batch. Completion is consumed exactly once.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the submission is unknown or completion
    /// handling fails.
    fn poll(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<ExecutionBatchEvent>, BackendError>;

    /// # Errors
    ///
    /// Returns a backend error from completion polling.
    fn wait(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<ExecutionBatchEvent, BackendError> {
        loop {
            if let Some(event) = self.poll(submission)? {
                return Ok(event);
            }
            std::thread::yield_now();
        }
    }

    /// # Errors
    ///
    /// Returns a backend error from submission or completion.
    fn execute(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<ExecutionBatchEvent, BackendError> {
        let submission = self.submit(plan, batch, states)?;
        self.wait(submission)
    }
}
