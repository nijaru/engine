//! Stateful execution orchestration shared by local and serving frontends.

use std::fmt;

use crate::backend::{BackendError, BackendSubmissionId, ComputeBackend};
use crate::execution::{ExecutionEvent, ExecutionPlan, ExecutionSegment};
use crate::model::{ModelError, ModelProvider};
use crate::state::{InferenceStateSet, StateError, StateManager};

pub struct ExecutionRuntime<P, B, S> {
    provider: P,
    backend: B,
    state_manager: S,
}

#[derive(Debug)]
pub struct RuntimeSubmission {
    backend_submission: BackendSubmissionId,
    state: Option<InferenceStateSet>,
    next_position: u32,
}

impl RuntimeSubmission {
    #[must_use]
    pub const fn backend_submission(&self) -> BackendSubmissionId {
        self.backend_submission
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

    /// Submit one execution segment without committing its logical prefix
    /// transition until the backend reports completion.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model validation, position arithmetic, or
    /// backend submission fails.
    pub fn submit_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        mut state: InferenceStateSet,
    ) -> Result<RuntimeSubmission, RuntimeError> {
        self.provider.validate_plan(plan)?;
        let next_position = segment
            .state_position()
            .checked_add(segment.token_count())
            .ok_or(RuntimeError::PositionOverflow)?;
        let backend_submission = self.backend.submit(plan, segment, &mut state)?;
        Ok(RuntimeSubmission {
            backend_submission,
            state: Some(state),
            next_position,
        })
    }

    /// Poll a submitted segment. The logical state manager commits the new
    /// prefix position only after device/backend completion is visible.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when completion polling or state commitment
    /// fails, or when a completed submission is consumed twice.
    pub fn poll_submission(
        &mut self,
        submission: &mut RuntimeSubmission,
    ) -> Result<Option<CompletedExecution>, RuntimeError> {
        let Some(event) = self.backend.poll(submission.backend_submission)? else {
            return Ok(None);
        };
        let state = submission
            .state
            .take()
            .ok_or(RuntimeError::SubmissionConsumed)?;
        let state = self.state_manager.commit(state, submission.next_position)?;
        Ok(Some(CompletedExecution { event, state }))
    }

    /// Wait for a submitted segment and commit its logical state transition.
    /// This is useful for direct/local inference and tests; serving loops can
    /// poll multiple submissions while preparing later work.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] from completion or state commitment.
    pub fn wait_submission(
        &mut self,
        mut submission: RuntimeSubmission,
    ) -> Result<CompletedExecution, RuntimeError> {
        loop {
            if let Some(completed) = self.poll_submission(&mut submission)? {
                return Ok(completed);
            }
            std::thread::yield_now();
        }
    }

    /// Submit and wait synchronously. This preserves the simple direct
    /// inference path without making synchronous execution the serving-loop
    /// architecture.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] from submission, completion, or state commit.
    pub fn execute_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        state: InferenceStateSet,
    ) -> Result<(ExecutionEvent, InferenceStateSet), RuntimeError> {
        let submission = self.submit_segment(plan, segment, state)?;
        Ok(self.wait_submission(submission)?.into_parts())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    Model(ModelError),
    Backend(BackendError),
    State(StateError),
    PositionOverflow,
    SubmissionConsumed,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "model validation failed: {error}"),
            Self::Backend(error) => write!(f, "backend execution failed: {error}"),
            Self::State(error) => write!(f, "state commitment failed: {error}"),
            Self::PositionOverflow => f.write_str("execution position overflowed"),
            Self::SubmissionConsumed => f.write_str("runtime submission was already consumed"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendCapabilities, BackendFeatures, BackendId, BackendKind};
    use crate::device::DeviceId;
    use crate::execution::{ExecutionMetrics, ExecutionPhase, ExecutionStage};
    use crate::model::{
        ModelCapabilities, ModelDescription, ModelId, ModelProvider, ModelRegion, ModelRegionId,
        ModelRegionKind, WeightDescription,
    };
    use crate::nvidia::{NvidiaBackend, NvidiaDispatcher};
    use crate::policy::PolicyVersion;
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
        ) -> Result<ExecutionMetrics, BackendError> {
            Ok(ExecutionMetrics::new(20, 0, 0))
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
        let plan = ExecutionPlan::new(
            model.clone(),
            backend_id.clone(),
            device,
            PolicyVersion::new(1).expect("policy version"),
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
        assert_eq!(
            runtime.state_manager().used_bytes(StateLocation::Device(device)),
            Some(8)
        );
        let completed = runtime.wait_submission(submission).expect("completion");
        assert_eq!(completed.event().metrics().elapsed_nanos(), 20);
        assert_eq!(completed.state().token_position(), Some(1));
    }
}
