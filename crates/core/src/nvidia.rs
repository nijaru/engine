//! NVIDIA backend adapter without a CUDA dependency in the core crate.
//!
//! A platform crate supplies [`NvidiaDispatcher`] with actual CUDA kernels or
//! runtime calls. This adapter owns capability validation and event creation;
//! it never reports successful work without a dispatcher-provided metric.

use crate::backend::{BackendCapabilities, BackendError, BackendKind, ComputeBackend};
use crate::execution::{ExecutionEvent, ExecutionMetrics, ExecutionPlan, ExecutionSegment};
use crate::state::HybridStateSet;
use crate::weights::WeightBinding;

pub trait NvidiaDispatcher: Send {
    /// Dispatch one already validated segment and return measured execution
    /// metrics. The dispatcher may update backend-owned state through `state`.
    ///
    /// # Errors
    ///
    /// Returns a backend error when CUDA dispatch or state mutation fails.
    fn dispatch(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        weights: &WeightBinding,
        state: &mut HybridStateSet,
    ) -> Result<ExecutionMetrics, BackendError>;
}

/// Capability-checked NVIDIA execution adapter. CUDA initialization and kernel
/// ownership stay in the injected dispatcher so `engine-core` remains
/// dependency-free and testable on non-NVIDIA hosts.
pub struct NvidiaBackend<D> {
    capabilities: BackendCapabilities,
    dispatcher: D,
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
}

impl<D: NvidiaDispatcher> ComputeBackend for NvidiaBackend<D> {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn execute(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        state: &mut HybridStateSet,
    ) -> Result<ExecutionEvent, BackendError> {
        self.validate_execution(plan, segment, state)?;
        let metrics = self
            .dispatcher
            .dispatch(plan, segment, plan.weights(), state)?;
        ExecutionEvent::new(
            segment.request(),
            plan.policy_version(),
            segment.phase(),
            segment.token_count(),
            metrics,
        )
        .ok_or_else(|| BackendError::ExecutionFailed("segment contained no tokens".to_owned()))
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
        HybridStateSet, KvStateSpec, LogicalStateManager, StateLocation, StateManager,
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
            _state: &mut HybridStateSet,
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
        let spec = match requirement {
            StateRequirement::FullAttentionKv(spec) => spec,
            // The fixture builder above only produces the KV requirement.
            StateRequirement::Recurrent(_) => {
                unimplemented!("test fixtures only exercise the full-attention KV requirement")
            }
        };
        let mut manager = LogicalStateManager::new(device, 1024, 0);
        let kv = manager
            .allocate_kv(spec, StateLocation::Device(device))
            .expect("state allocation");
        let mut state = HybridStateSet::try_new(Some(kv), None).expect("state set");
        let event = backend
            .execute(&plan, &segment, &mut state)
            .expect("dispatch");
        assert_eq!(event.metrics().elapsed_nanos(), 12);
        assert_eq!(event.policy_version(), policy);
    }
}
