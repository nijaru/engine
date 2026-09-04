//! NVIDIA backend adapter without a CUDA dependency in the core crate.

use std::collections::HashMap;

use crate::backend::{
    BackendCapabilities, BackendError, BackendKind, BackendSubmissionId, ComputeBackend,
};
use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionPlan,
    ExecutionSegment,
};
use crate::state::InferenceStateSet;
use crate::weights::WeightBinding;

pub trait NvidiaDispatcher: Send {
    /// # Errors
    ///
    /// Returns a backend error when CUDA dispatch or state mutation fails.
    fn dispatch(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        weights: &WeightBinding,
        state: &mut InferenceStateSet,
    ) -> Result<ExecutionMetrics, BackendError>;
}

pub struct NvidiaBackend<D> {
    capabilities: BackendCapabilities,
    dispatcher: D,
    next_submission: u64,
    completed: HashMap<BackendSubmissionId, ExecutionBatchEvent>,
}

impl<D> NvidiaBackend<D> {
    /// # Errors
    ///
    /// Returns [`BackendError::Unsupported`] when the supplied capabilities do
    /// not describe a CUDA backend.
    pub fn new(capabilities: BackendCapabilities, dispatcher: D) -> Result<Self, BackendError> {
        if capabilities.kind() != BackendKind::Cuda {
            return Err(BackendError::Unsupported("CUDA backend capabilities"));
        }
        Ok(Self {
            capabilities,
            dispatcher,
            next_submission: 1,
            completed: HashMap::new(),
        })
    }

    #[must_use]
    pub fn dispatcher(&self) -> &D {
        &self.dispatcher
    }

    #[must_use]
    pub fn dispatcher_mut(&mut self) -> &mut D {
        &mut self.dispatcher
    }

    #[must_use]
    pub fn into_dispatcher(self) -> D {
        self.dispatcher
    }

    fn allocate_submission(&mut self) -> Result<BackendSubmissionId, BackendError> {
        let id = BackendSubmissionId::new(self.next_submission).ok_or_else(|| {
            BackendError::ExecutionFailed("submission identity overflowed".to_owned())
        })?;
        self.next_submission = self.next_submission.checked_add(1).ok_or_else(|| {
            BackendError::ExecutionFailed("submission identity overflowed".to_owned())
        })?;
        Ok(id)
    }
}

impl<D: NvidiaDispatcher> ComputeBackend for NvidiaBackend<D> {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn submit(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<BackendSubmissionId, BackendError> {
        self.validate_execution(plan, batch, states)?;
        let mut events = Vec::with_capacity(batch.len());
        for (segment, state) in batch.segments().iter().zip(states) {
            let metrics = self
                .dispatcher
                .dispatch(plan, segment, plan.weights(), state)?;
            let event = ExecutionEvent::new(
                segment.request(),
                plan.policy_version(),
                segment.phase(),
                segment.token_count(),
                metrics,
            )
            .ok_or_else(|| {
                BackendError::ExecutionFailed("segment contained no tokens".to_owned())
            })?;
            events.push(event);
        }
        let event = ExecutionBatchEvent::new(events).map_err(BackendError::InvalidPlan)?;
        let submission = self.allocate_submission()?;
        self.completed.insert(submission, event);
        Ok(submission)
    }

    fn poll(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<ExecutionBatchEvent>, BackendError> {
        self.completed
            .remove(&submission)
            .map(Some)
            .ok_or(BackendError::UnknownSubmission(submission))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendFeatures, BackendId};
    use crate::device::DeviceId;
    use crate::execution::{ExecutionPhase, ExecutionStage};
    use crate::model::{ModelId, ModelRegionId};
    use crate::policy::PolicyVersion;
    use crate::request::RequestId;
    use crate::state::{
        InferenceStateSet, KvStateSpec, LogicalStateManager, StateLocation, StateManager,
        StateRequirement,
    };
    use crate::tensor::{DataType, Quantization};

    struct TestDispatcher;

    impl NvidiaDispatcher for TestDispatcher {
        fn dispatch(
            &mut self,
            _plan: &ExecutionPlan,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionMetrics, BackendError> {
            Ok(ExecutionMetrics::new(12, 4, 8))
        }
    }

    #[test]
    fn backend_dispatches_only_after_capability_and_state_validation() {
        let device = DeviceId::new(0);
        let capabilities = BackendCapabilities::new(
            BackendId::new("cuda").expect("backend ID"),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(
                vec![DataType::F16],
                vec![Quantization::GgufQ4Km],
                false,
                true,
            ),
        );
        let mut backend = NvidiaBackend::new(capabilities, TestDispatcher).expect("CUDA backend");
        let model = ModelId::new("test-model").expect("model ID");
        let policy = PolicyVersion::new(1).expect("policy version");
        let requirement = StateRequirement::FullAttentionKv(
            KvStateSpec::new(1, 1, 2, 1, DataType::F16).expect("KV spec"),
        );
        let plan = ExecutionPlan::new(
            model.clone(),
            BackendId::new("cuda").expect("backend ID"),
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
        let request = RequestId::new(1).expect("request ID");
        let segment =
            ExecutionSegment::new(request, ExecutionPhase::Decode, 1, 1, 0, vec![requirement])
                .expect("segment");
        let batch = ExecutionBatch::new(vec![segment]).expect("batch");
        let spec = match requirement {
            StateRequirement::FullAttentionKv(spec) => spec,
            StateRequirement::Recurrent(_) => unreachable!(),
        };
        let mut manager = LogicalStateManager::new(device, 1024, 0);
        let kv = manager
            .allocate_kv(spec, StateLocation::Device(device))
            .expect("state allocation");
        let state = InferenceStateSet::try_new(Some(kv), None).expect("state set");
        let submission = backend.submit(&plan, &batch, &mut [state]).expect("submit");
        let event = backend.wait(submission).expect("completion");
        assert_eq!(event.events()[0].metrics().elapsed_nanos(), 12);
        assert_eq!(event.events()[0].policy_version(), policy);
    }
}
