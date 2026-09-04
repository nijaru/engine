//! Stateful execution orchestration shared by local and serving frontends.

use std::fmt;

use crate::backend::{BackendError, ComputeBackend};
use crate::execution::{ExecutionEvent, ExecutionPlan, ExecutionSegment};
use crate::model::{ModelError, ModelProvider};
use crate::state::{InferenceStateSet, StateError, StateManager};

pub struct ExecutionRuntime<P, B, S> {
    provider: P,
    backend: B,
    state_manager: S,
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
    pub const fn provider(&self) -> &P { &self.provider }

    #[must_use]
    pub const fn backend(&self) -> &B { &self.backend }

    #[must_use]
    pub const fn backend_mut(&mut self) -> &mut B { &mut self.backend }

    #[must_use]
    pub const fn state_manager(&self) -> &S { &self.state_manager }

    #[must_use]
    pub const fn state_manager_mut(&mut self) -> &mut S { &mut self.state_manager }

    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model validation, backend dispatch, state
    /// commitment, or position arithmetic fails.
    pub fn execute_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        mut state: InferenceStateSet,
    ) -> Result<(ExecutionEvent, InferenceStateSet), RuntimeError> {
        self.provider.validate_plan(plan)?;
        let next_position = segment
            .state_position()
            .checked_add(segment.token_count())
            .ok_or(RuntimeError::PositionOverflow)?;
        let event = self.backend.execute(plan, segment, &mut state)?;
        let committed = self.state_manager.commit(state, next_position)?;
        Ok((event, committed))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    Model(ModelError),
    Backend(BackendError),
    State(StateError),
    PositionOverflow,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "model validation failed: {error}"),
            Self::Backend(error) => write!(f, "backend execution failed: {error}"),
            Self::State(error) => write!(f, "state commitment failed: {error}"),
            Self::PositionOverflow => f.write_str("execution position overflowed"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ModelError> for RuntimeError {
    fn from(error: ModelError) -> Self { Self::Model(error) }
}

impl From<BackendError> for RuntimeError {
    fn from(error: BackendError) -> Self { Self::Backend(error) }
}

impl From<StateError> for RuntimeError {
    fn from(error: StateError) -> Self { Self::State(error) }
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
    use crate::weights::{WeightBinding, WeightDescription as _};

    struct TestProvider {
        description: ModelDescription,
    }

    impl ModelProvider for TestProvider {
        fn description(&self) -> &ModelDescription { &self.description }
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
    fn execution_commits_state_after_backend_success() {
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
            crate::model::WeightDescription::new(WeightFormat::Gguf, Quantization::GgufQ4Km),
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

        let (event, committed) = runtime
            .execute_segment(&plan, &segment, state)
            .expect("execution");
        assert_eq!(event.metrics().elapsed_nanos(), 20);
        assert_eq!(committed.token_position(), Some(1));
    }
}
