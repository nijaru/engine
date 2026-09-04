//! Stateful execution orchestration shared by local and serving frontends.

use std::fmt;

use crate::backend::{BackendError, BackendSubmissionId, ComputeBackend};
use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionPlan, ExecutionSegment, PlanError,
};
use crate::model::{ModelError, ModelProvider};
use crate::policy::PolicyVersion;
use crate::state::{InferenceStateSet, StateError, StateManager};

pub struct ExecutionRuntime<P, B, S> {
    provider: P,
    backend: B,
    state_manager: S,
}

#[derive(Debug)]
pub struct RuntimeSubmission {
    backend_submission: BackendSubmissionId,
    batch: ExecutionBatch,
    policy_version: PolicyVersion,
    states: Option<Vec<InferenceStateSet>>,
    next_positions: Vec<u32>,
}

impl RuntimeSubmission {
    #[must_use]
    pub const fn backend_submission(&self) -> BackendSubmissionId {
        self.backend_submission
    }

    #[must_use]
    pub const fn batch(&self) -> &ExecutionBatch {
        &self.batch
    }

    #[must_use]
    pub const fn policy_version(&self) -> PolicyVersion {
        self.policy_version
    }

    /// Recover uncommitted logical state after a terminal submission failure.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::SubmissionConsumed`] after completion or an
    /// earlier recovery already consumed the state.
    pub fn take_uncommitted_states(&mut self) -> Result<Vec<InferenceStateSet>, RuntimeError> {
        self.states.take().ok_or(RuntimeError::SubmissionConsumed)
    }
}

#[derive(Debug)]
pub struct RuntimeSubmitError {
    error: RuntimeError,
    states: Vec<InferenceStateSet>,
}

impl RuntimeSubmitError {
    fn new(error: RuntimeError, states: Vec<InferenceStateSet>) -> Self {
        Self { error, states }
    }

    #[must_use]
    pub const fn error(&self) -> &RuntimeError {
        &self.error
    }

    #[must_use]
    pub fn states(&self) -> &[InferenceStateSet] {
        &self.states
    }

    #[must_use]
    pub fn into_parts(self) -> (RuntimeError, Vec<InferenceStateSet>) {
        (self.error, self.states)
    }

    #[must_use]
    pub fn into_error(self) -> RuntimeError {
        self.error
    }
}

impl fmt::Display for RuntimeSubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime submission failed: {}", self.error)
    }
}

impl std::error::Error for RuntimeSubmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[derive(Debug)]
pub struct CompletedExecutionBatch {
    event: ExecutionBatchEvent,
    states: Vec<InferenceStateSet>,
}

impl CompletedExecutionBatch {
    #[must_use]
    pub const fn event(&self) -> &ExecutionBatchEvent {
        &self.event
    }

    #[must_use]
    pub fn states(&self) -> &[InferenceStateSet] {
        &self.states
    }

    #[must_use]
    pub fn into_parts(self) -> (ExecutionBatchEvent, Vec<InferenceStateSet>) {
        (self.event, self.states)
    }
}

#[derive(Debug)]
pub struct CompletedExecution {
    event: ExecutionEvent,
    state: InferenceStateSet,
}

impl CompletedExecution {
    #[must_use]
    pub const fn event(&self) -> ExecutionEvent {
        self.event
    }

    #[must_use]
    pub const fn state(&self) -> &InferenceStateSet {
        &self.state
    }

    #[must_use]
    pub fn into_parts(self) -> (ExecutionEvent, InferenceStateSet) {
        (self.event, self.state)
    }
}

impl<P, B, S> ExecutionRuntime<P, B, S>
where
    P: ModelProvider,
    B: ComputeBackend,
    S: StateManager,
{
    #[must_use]
    pub const fn new(provider: P, backend: B, state_manager: S) -> Self {
        Self {
            provider,
            backend,
            state_manager,
        }
    }

    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    #[must_use]
    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[must_use]
    pub const fn state_manager(&self) -> &S {
        &self.state_manager
    }

    #[must_use]
    pub const fn state_manager_mut(&mut self) -> &mut S {
        &mut self.state_manager
    }

    /// Release every logical state allocation owned by one finished request.
    ///
    /// Physical backend state remains a backend concern; this closes the core
    /// allocation lifecycle so request reclamation cannot leak `StateManager`
    /// capacity.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::State`] when a state handle is invalid or its
    /// manager cannot release it.
    pub fn release_state_set(&mut self, state: &InferenceStateSet) -> Result<(), RuntimeError> {
        for value in state.states() {
            self.state_manager.release(value.handle().clone())?;
        }
        Ok(())
    }

    /// Submit a batch while preserving caller-owned logical state if model,
    /// plan, arithmetic, or backend submission validation fails.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeSubmitError`] with the uncommitted states when the
    /// backend never takes submission ownership.
    pub fn submit_batch(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        mut states: Vec<InferenceStateSet>,
    ) -> Result<RuntimeSubmission, RuntimeSubmitError> {
        if let Err(error) = self.provider.validate_plan(plan) {
            return Err(RuntimeSubmitError::new(RuntimeError::Model(error), states));
        }
        if batch.len() != states.len() {
            return Err(RuntimeSubmitError::new(
                RuntimeError::StateCountMismatch,
                states,
            ));
        }

        let mut next_positions = Vec::with_capacity(batch.len());
        for segment in batch.segments() {
            let Some(position) = segment.state_position().checked_add(segment.token_count()) else {
                return Err(RuntimeSubmitError::new(
                    RuntimeError::PositionOverflow,
                    states,
                ));
            };
            next_positions.push(position);
        }

        let backend_submission = match self.backend.submit(plan, batch, &mut states) {
            Ok(submission) => submission,
            Err(error) => {
                return Err(RuntimeSubmitError::new(
                    RuntimeError::Backend(error),
                    states,
                ));
            }
        };
        Ok(RuntimeSubmission {
            backend_submission,
            batch: batch.clone(),
            policy_version: plan.policy_version(),
            states: Some(states),
            next_positions,
        })
    }

    /// # Errors
    ///
    /// Returns [`RuntimeSubmitError`] from batch construction or submission,
    /// preserving the supplied state.
    pub fn submit_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        state: InferenceStateSet,
    ) -> Result<RuntimeSubmission, RuntimeSubmitError> {
        let states = vec![state];
        let batch = match ExecutionBatch::new(vec![segment.clone()]) {
            Ok(batch) => batch,
            Err(error) => {
                return Err(RuntimeSubmitError::new(RuntimeError::Plan(error), states));
            }
        };
        self.submit_batch(plan, &batch, states)
    }

    /// Poll a submitted batch. Logical state positions are committed only
    /// after the backend completion is visible and matches the submitted work.
    /// A backend polling error is terminal: the submission retains its states
    /// so the serving owner can recover them with `take_uncommitted_states`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when completion polling, identity validation,
    /// or state commitment fails.
    pub fn poll_submission(
        &mut self,
        submission: &mut RuntimeSubmission,
    ) -> Result<Option<CompletedExecutionBatch>, RuntimeError> {
        let Some(event) = self.backend.poll(submission.backend_submission)? else {
            return Ok(None);
        };
        Self::validate_completion(submission, &event)?;
        let states = submission
            .states
            .take()
            .ok_or(RuntimeError::SubmissionConsumed)?;
        let committed = states
            .into_iter()
            .zip(submission.next_positions.iter().copied())
            .map(|(state, position)| self.state_manager.commit(state, position))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(CompletedExecutionBatch {
            event,
            states: committed,
        }))
    }

    /// # Errors
    ///
    /// Returns [`RuntimeError`] from completion or state commitment.
    pub fn wait_submission(
        &mut self,
        mut submission: RuntimeSubmission,
    ) -> Result<CompletedExecutionBatch, RuntimeError> {
        loop {
            if let Some(completed) = self.poll_submission(&mut submission)? {
                return Ok(completed);
            }
            std::thread::yield_now();
        }
    }

    /// # Errors
    ///
    /// Returns [`RuntimeError`] from submission, completion, or state commit.
    pub fn execute_batch(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: Vec<InferenceStateSet>,
    ) -> Result<CompletedExecutionBatch, RuntimeError> {
        let submission = self
            .submit_batch(plan, batch, states)
            .map_err(RuntimeSubmitError::into_error)?;
        self.wait_submission(submission)
    }

    /// # Errors
    ///
    /// Returns [`RuntimeError`] from submission, completion, or state commit.
    pub fn execute_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        state: InferenceStateSet,
    ) -> Result<CompletedExecution, RuntimeError> {
        let submission = self
            .submit_segment(plan, segment, state)
            .map_err(RuntimeSubmitError::into_error)?;
        let completed = self.wait_submission(submission)?;
        let (batch_event, mut states) = completed.into_parts();
        let mut events = batch_event.events().iter().copied();
        let event = events.next().ok_or(RuntimeError::CompletionMismatch)?;
        if events.next().is_some() || states.len() != 1 {
            return Err(RuntimeError::CompletionMismatch);
        }
        let state = states.pop().ok_or(RuntimeError::CompletionMismatch)?;
        Ok(CompletedExecution { event, state })
    }

    fn validate_completion(
        submission: &RuntimeSubmission,
        event: &ExecutionBatchEvent,
    ) -> Result<(), RuntimeError> {
        if event.len() != submission.batch.len() {
            return Err(RuntimeError::CompletionMismatch);
        }
        for (segment, completed) in submission.batch.segments().iter().zip(event.events()) {
            if segment.request() != completed.request()
                || segment.phase() != completed.phase()
                || segment.token_count() != completed.token_count()
                || completed.policy_version() != submission.policy_version
            {
                return Err(RuntimeError::CompletionMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    Model(ModelError),
    Backend(BackendError),
    State(StateError),
    Plan(PlanError),
    StateCountMismatch,
    PositionOverflow,
    SubmissionConsumed,
    CompletionMismatch,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "model validation failed: {error}"),
            Self::Backend(error) => write!(f, "backend execution failed: {error}"),
            Self::State(error) => write!(f, "state commitment failed: {error}"),
            Self::Plan(error) => write!(f, "execution batch is invalid: {error}"),
            Self::StateCountMismatch => {
                f.write_str("execution batch and inference-state counts differ")
            }
            Self::PositionOverflow => f.write_str("execution position overflowed"),
            Self::SubmissionConsumed => f.write_str("runtime submission was already consumed"),
            Self::CompletionMismatch => {
                f.write_str("backend completion does not match submitted execution batch")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ModelError> for RuntimeError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<BackendError> for RuntimeError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<StateError> for RuntimeError {
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}

impl From<PlanError> for RuntimeError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendCapabilities, BackendFeatures, BackendId, BackendKind};
    use crate::device::DeviceId;
    use crate::execution::{ExecutionMetrics, ExecutionOutcome, ExecutionPhase, ExecutionStage};
    use crate::model::{
        ModelCapabilities, ModelDescription, ModelId, ModelProvider, ModelRegion, ModelRegionId,
        ModelRegionKind, WeightDescription,
    };
    use crate::nvidia::{NvidiaBackend, NvidiaDispatcher};
    use crate::state::{
        InferenceStateSet, KvStateSpec, LogicalStateManager, StateLocation, StateManager,
        StateRequirement,
    };
    use crate::tensor::{DataType, Quantization, WeightFormat};
    use crate::weights::WeightBinding;

    struct TestProvider {
        description: ModelDescription,
    }

    impl ModelProvider for TestProvider {
        fn description(&self) -> &ModelDescription {
            &self.description
        }
    }

    struct TestDispatcher;

    impl NvidiaDispatcher for TestDispatcher {
        fn dispatch(
            &mut self,
            _plan: &ExecutionPlan,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            Ok(ExecutionOutcome::new(ExecutionMetrics::new(20, 0, 0)))
        }
    }

    #[test]
    fn execution_commits_state_only_after_completion() {
        let device = DeviceId::new(0);
        let model = ModelId::new("test-model").expect("model ID");
        let requirement = StateRequirement::FullAttentionKv(
            KvStateSpec::new(1, 1, 2, 2, DataType::F16).expect("KV spec"),
        );
        let description = ModelDescription::new(
            model.clone(),
            "test-hybrid",
            vec![ModelRegion::new(
                ModelRegionId::new(0),
                ModelRegionKind::FullAttention,
            )],
            vec![requirement],
            ModelCapabilities::new(None, false),
            WeightDescription::new(WeightFormat::Gguf, Quantization::GgufQ4Km),
        )
        .expect("model description");
        let backend_id = BackendId::new("cuda").expect("backend ID");
        let policy = PolicyVersion::new(1).expect("policy version");
        let plan = ExecutionPlan::new(
            model.clone(),
            backend_id.clone(),
            device,
            policy,
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            vec![requirement],
            WeightBinding::empty(model.clone(), device),
        )
        .expect("plan");
        let segment = ExecutionSegment::new(
            crate::request::RequestId::new(1).expect("request ID"),
            ExecutionPhase::Decode,
            1,
            1,
            0,
            vec![requirement],
        )
        .expect("segment");
        let capabilities = BackendCapabilities::new(
            backend_id,
            device,
            BackendKind::Cuda,
            1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let backend = NvidiaBackend::new(capabilities, TestDispatcher).expect("backend");
        let mut state_manager = LogicalStateManager::new(device, 1024, 0);
        let kv_spec = match requirement {
            StateRequirement::FullAttentionKv(spec) => spec,
            StateRequirement::Recurrent(_) => unreachable!(),
        };
        let kv = state_manager
            .allocate_kv(kv_spec, StateLocation::Device(device))
            .expect("state allocation");
        let state = InferenceStateSet::try_new(Some(kv), None).expect("state set");
        let mut runtime =
            ExecutionRuntime::new(TestProvider { description }, backend, state_manager);

        let submission = runtime
            .submit_segment(&plan, &segment, state)
            .expect("submission");
        assert_eq!(submission.policy_version(), policy);
        assert_eq!(
            runtime
                .state_manager()
                .used_bytes(StateLocation::Device(device)),
            Some(16)
        );
        let completed = runtime.wait_submission(submission).expect("completion");
        assert_eq!(completed.event().events()[0].metrics().elapsed_nanos(), 20);
        assert_eq!(completed.states()[0].token_position(), Some(1));
    }
}
