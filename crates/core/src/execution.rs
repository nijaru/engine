//! Narrow inference execution descriptions and prepared plans.

use std::collections::HashSet;
use std::fmt;

use crate::backend::BackendId;
use crate::device::DeviceId;
use crate::model::{ModelId, ModelRegionId};
use crate::policy::PolicyVersion;
use crate::qualification::{ExecutionVariant, QualificationStatus};
use crate::request::RequestId;
use crate::residency::ModelResidencyPlan;
use crate::state::StateRequirement;
use crate::weights::WeightBinding;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    SpecDraft,
    SpecVerify,
    Encoder,
    MoEExpert,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionSegment {
    request: RequestId,
    phase: ExecutionPhase,
    token_count: u32,
    state_position: u32,
    state_requirements: Vec<StateRequirement>,
}

impl ExecutionSegment {
    /// # Errors
    ///
    /// Returns [`PlanError::InvalidSegmentBatchSize`] unless `batch_size` is
    /// one or [`PlanError::ZeroWork`] when `token_count` is zero. Multi-request
    /// batching is represented by [`ExecutionBatch`].
    pub fn new(
        request: RequestId,
        phase: ExecutionPhase,
        batch_size: u32,
        token_count: u32,
        state_position: u32,
        state_requirements: Vec<StateRequirement>,
    ) -> Result<Self, PlanError> {
        if batch_size != 1 {
            return Err(PlanError::InvalidSegmentBatchSize);
        }
        if token_count == 0 {
            return Err(PlanError::ZeroWork);
        }
        Ok(Self {
            request,
            phase,
            token_count,
            state_position,
            state_requirements,
        })
    }

    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn phase(&self) -> ExecutionPhase {
        self.phase
    }

    #[must_use]
    pub const fn batch_size(&self) -> u32 {
        1
    }

    #[must_use]
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    #[must_use]
    pub const fn state_position(&self) -> u32 {
        self.state_position
    }

    #[must_use]
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.state_requirements
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBatch {
    segments: Vec<ExecutionSegment>,
}

impl ExecutionBatch {
    /// # Errors
    ///
    /// Returns [`PlanError::EmptyBatch`] for no work or
    /// [`PlanError::DuplicateBatchRequest`] when a request appears more than
    /// once in the same submission.
    pub fn new(segments: Vec<ExecutionSegment>) -> Result<Self, PlanError> {
        if segments.is_empty() {
            return Err(PlanError::EmptyBatch);
        }
        let mut requests = HashSet::with_capacity(segments.len());
        for segment in &segments {
            if !requests.insert(segment.request()) {
                return Err(PlanError::DuplicateBatchRequest);
            }
        }
        Ok(Self { segments })
    }

    #[must_use]
    pub fn segments(&self) -> &[ExecutionSegment] {
        &self.segments
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    #[must_use]
    pub fn total_tokens(&self) -> Option<u32> {
        self.segments.iter().try_fold(0_u32, |total, segment| {
            total.checked_add(segment.token_count())
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionStage {
    region: ModelRegionId,
    phase: ExecutionPhase,
}

impl ExecutionStage {
    #[must_use]
    pub const fn new(region: ModelRegionId, phase: ExecutionPhase) -> Self {
        Self { region, phase }
    }

    #[must_use]
    pub const fn region(self) -> ModelRegionId {
        self.region
    }

    #[must_use]
    pub const fn phase(self) -> ExecutionPhase {
        self.phase
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlan {
    model: ModelId,
    backend: BackendId,
    device: DeviceId,
    policy_version: PolicyVersion,
    stages: Vec<ExecutionStage>,
    state_requirements: Vec<StateRequirement>,
    weights: WeightBinding,
    residency: ModelResidencyPlan,
    variant: ExecutionVariant,
}

impl ExecutionPlan {
    /// # Errors
    ///
    /// Returns [`PlanError::EmptyStages`] when the plan has no stages or a
    /// binding does not match the model/device.
    pub fn new(
        model: ModelId,
        backend: BackendId,
        device: DeviceId,
        policy_version: PolicyVersion,
        stages: Vec<ExecutionStage>,
        state_requirements: Vec<StateRequirement>,
        weights: WeightBinding,
    ) -> Result<Self, PlanError> {
        if stages.is_empty() {
            return Err(PlanError::EmptyStages);
        }
        if weights.model() != &model || weights.device() != device {
            return Err(PlanError::WeightBindingMismatch);
        }
        let residency = ModelResidencyPlan::single_device(model.clone(), device);
        Ok(Self {
            model,
            backend,
            device,
            policy_version,
            stages,
            state_requirements,
            weights,
            residency,
            variant: ExecutionVariant::baseline(),
        })
    }

    /// # Errors
    ///
    /// Returns [`PlanError::ResidencyModelMismatch`] when the residency plan
    /// describes a different model.
    pub fn with_residency(mut self, residency: ModelResidencyPlan) -> Result<Self, PlanError> {
        if residency.model() != &self.model {
            return Err(PlanError::ResidencyModelMismatch);
        }
        self.residency = residency;
        Ok(self)
    }

    /// # Errors
    ///
    /// Returns [`PlanError::IncompatibleVariant`] for a variant that has been
    /// explicitly qualified as incompatible.
    pub fn with_variant(mut self, variant: ExecutionVariant) -> Result<Self, PlanError> {
        if variant.status() == QualificationStatus::Incompatible {
            return Err(PlanError::IncompatibleVariant);
        }
        self.variant = variant;
        Ok(self)
    }

    /// Check that a request segment belongs to this prepared plan.
    ///
    /// # Errors
    ///
    /// Returns a [`PlanError`] when the phase or state dependency is not in the
    /// plan.
    pub fn validate_segment(&self, segment: &ExecutionSegment) -> Result<(), PlanError> {
        if !self
            .stages
            .iter()
            .any(|stage| stage.phase() == segment.phase())
        {
            return Err(PlanError::SegmentPhaseMismatch);
        }
        if segment
            .state_requirements()
            .iter()
            .any(|requirement| !self.state_requirements.contains(requirement))
        {
            return Err(PlanError::SegmentStateUndeclared);
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns the first segment validation error in the batch.
    pub fn validate_batch(&self, batch: &ExecutionBatch) -> Result<(), PlanError> {
        for segment in batch.segments() {
            self.validate_segment(segment)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn model(&self) -> &ModelId {
        &self.model
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
    pub const fn policy_version(&self) -> PolicyVersion {
        self.policy_version
    }

    #[must_use]
    pub fn stages(&self) -> &[ExecutionStage] {
        &self.stages
    }

    #[must_use]
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.state_requirements
    }

    #[must_use]
    pub fn weights(&self) -> &WeightBinding {
        &self.weights
    }

    #[must_use]
    pub const fn residency(&self) -> &ModelResidencyPlan {
        &self.residency
    }

    #[must_use]
    pub const fn variant(&self) -> &ExecutionVariant {
        &self.variant
    }

    #[must_use]
    pub fn requires_kv_state(&self) -> bool {
        self.state_requirements
            .iter()
            .any(|requirement| requirement.is_kv())
    }

    #[must_use]
    pub fn requires_recurrent_state(&self) -> bool {
        self.state_requirements
            .iter()
            .any(|requirement| requirement.is_recurrent())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionMetrics {
    elapsed_nanos: u64,
    state_load_bytes: u64,
    state_store_bytes: u64,
}

impl ExecutionMetrics {
    #[must_use]
    pub const fn new(elapsed_nanos: u64, state_load_bytes: u64, state_store_bytes: u64) -> Self {
        Self {
            elapsed_nanos,
            state_load_bytes,
            state_store_bytes,
        }
    }

    #[must_use]
    pub const fn elapsed_nanos(self) -> u64 {
        self.elapsed_nanos
    }

    #[must_use]
    pub const fn state_load_bytes(self) -> u64 {
        self.state_load_bytes
    }

    #[must_use]
    pub const fn state_store_bytes(self) -> u64 {
        self.state_store_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionEvent {
    request: RequestId,
    policy_version: PolicyVersion,
    phase: ExecutionPhase,
    token_count: u32,
    metrics: ExecutionMetrics,
}

impl ExecutionEvent {
    #[must_use]
    pub const fn new(
        request: RequestId,
        policy_version: PolicyVersion,
        phase: ExecutionPhase,
        token_count: u32,
        metrics: ExecutionMetrics,
    ) -> Option<Self> {
        if token_count == 0 {
            None
        } else {
            Some(Self {
                request,
                policy_version,
                phase,
                token_count,
                metrics,
            })
        }
    }

    #[must_use]
    pub const fn request(self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn policy_version(self) -> PolicyVersion {
        self.policy_version
    }

    #[must_use]
    pub const fn phase(self) -> ExecutionPhase {
        self.phase
    }

    #[must_use]
    pub const fn token_count(self) -> u32 {
        self.token_count
    }

    #[must_use]
    pub const fn metrics(self) -> ExecutionMetrics {
        self.metrics
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBatchEvent {
    events: Vec<ExecutionEvent>,
}

impl ExecutionBatchEvent {
    /// # Errors
    ///
    /// Returns [`PlanError::EmptyBatch`] when the completion has no events or
    /// [`PlanError::DuplicateBatchRequest`] when a request occurs twice.
    pub fn new(events: Vec<ExecutionEvent>) -> Result<Self, PlanError> {
        if events.is_empty() {
            return Err(PlanError::EmptyBatch);
        }
        let mut requests = HashSet::with_capacity(events.len());
        for event in &events {
            if !requests.insert(event.request()) {
                return Err(PlanError::DuplicateBatchRequest);
            }
        }
        Ok(Self { events })
    }

    #[must_use]
    pub fn events(&self) -> &[ExecutionEvent] {
        &self.events
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PlanError {
    EmptyStages,
    EmptyBatch,
    DuplicateBatchRequest,
    InvalidSegmentBatchSize,
    ZeroWork,
    SegmentPhaseMismatch,
    SegmentStateUndeclared,
    WeightBindingMismatch,
    ResidencyModelMismatch,
    IncompatibleVariant,
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStages => f.write_str("execution plan must contain at least one stage"),
            Self::EmptyBatch => f.write_str("execution batch must contain at least one segment"),
            Self::DuplicateBatchRequest => {
                f.write_str("execution batch contains duplicate request work")
            }
            Self::InvalidSegmentBatchSize => {
                f.write_str("an execution segment describes exactly one request")
            }
            Self::ZeroWork => f.write_str("execution segment must contain non-zero work"),
            Self::SegmentPhaseMismatch => {
                f.write_str("execution segment phase is not in the prepared plan")
            }
            Self::SegmentStateUndeclared => {
                f.write_str("execution segment requires undeclared model state")
            }
            Self::WeightBindingMismatch => {
                f.write_str("weight binding does not match the execution model/device")
            }
            Self::ResidencyModelMismatch => {
                f.write_str("model residency plan describes a different model")
            }
            Self::IncompatibleVariant => {
                f.write_str("execution variant is qualified as incompatible")
            }
        }
    }
}

impl std::error::Error for PlanError {}
