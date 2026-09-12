#[cfg(test)]
mod tests {
    use engine_core::backend::{BackendCapabilities, BackendFeatures, BackendId, BackendKind};
    use engine_core::device::DeviceId;
    use engine_core::execution::{
        ExecutionMetrics, ExecutionOutcome, ExecutionPhase, ExecutionStage,
    };
    use engine_core::model::{
        ModelCapabilities, ModelDescription, ModelId, ModelProvider, ModelRegion, ModelRegionId,
        ModelRegionKind, WeightDescription,
    };
    use engine_core::state::{
        InferenceStateSet, KvStateSpec, LogicalStateManager, StateLocation, StateManager,
        StateRequirement,
    };
    use engine_core::tensor::{DataType, Quantization, WeightFormat};
    use engine_core::weights::WeightBinding;
    use engine_core::*;
    use engine_nvidia::{NvidiaBackend, NvidiaDispatcher};

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
            engine_core::request::RequestId::new(1).expect("request ID"),
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
