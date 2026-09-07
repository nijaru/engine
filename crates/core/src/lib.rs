//! Core types and interfaces for Engine.
//!
//! These interfaces describe inference semantics and execution ownership without
//! committing the core to a checkpoint format, device backend, or scheduler
//! implementation. Model state remains typed without baking one model family's
//! state bundle into the runtime boundary.

pub mod backend;
pub mod device;
pub mod execution;
pub mod model;
pub mod nvidia;
pub mod policy;
pub mod qualification;
pub mod readiness;
pub mod request;
pub mod residency;
pub mod runtime;
pub mod scheduler;
pub mod serving;
pub mod serving_runtime;
pub mod state;
pub mod tensor;
pub mod weights;

pub use backend::{
    BackendCapabilities, BackendError, BackendFeatures, BackendId, BackendKind,
    BackendSubmissionId, ComputeBackend,
};
pub use device::DeviceId;
pub use execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionOutcome,
    ExecutionPhase, ExecutionPlan, ExecutionSegment, ExecutionStage, ExecutionTokenInput,
    PlanError,
};
pub use model::{
    FileModelProvider, FileWeightLoader, ModelCapabilities, ModelDescription, ModelError, ModelId,
    ModelLoadError, ModelProvider, ModelRegion, ModelRegionId, ModelRegionKind, MtpCapability,
    WeightArtifact, WeightDescription, WeightLoader, WeightSource,
};
pub use nvidia::{NvidiaBackend, NvidiaDispatcher};
pub use policy::{
    PolicyError, PolicySnapshot, PolicyVersion, SpeculationPolicy, StateTierPreference,
};
pub use qualification::{
    ExecutionVariant, ExecutionVariantId, ExecutionVariantIdError, QualificationError,
    QualificationEvidence, QualificationScope, QualificationStatus,
};
pub use readiness::{ReadinessError, ReadinessState, RuntimeReadiness};
pub use request::{
    PromptFormat, PromptPolicy, RequestError, RequestId, RequestSemantics, RequestSpec,
    SamplingError, SamplingParams, SpecialTokenPolicy, ThinkingMode,
};
pub use residency::{
    ModelResidencyPlan, ModelResourceId, ResidencyError, ResidencyLocation, ResidencyOverride,
};
pub use runtime::{
    CompletedExecution, CompletedExecutionBatch, ExecutionRuntime, RuntimeError, RuntimeStateError,
    RuntimeSubmission,
};
pub use scheduler::{
    ScheduledWork, SchedulerConfig, SchedulerCounts, SchedulerError, ServingScheduler,
};
pub use serving::{
    ActiveRequestSlot, AdmissionError, RequestLifecycle, RequestProgress, RequestSlotError,
    RequestSlotId, RequestSlots,
};
pub use serving_runtime::{GeneratedToken, ServingIteration, ServingRuntime, ServingRuntimeError};
pub use state::{
    ConvolutionStateShape, InferenceState, InferenceStateSet, KvState, KvStateSpec,
    LogicalStateManager, RecurrentMatrixShape, RecurrentState, RecurrentStateSpec, StateError,
    StateHandle, StateId, StateLocation, StateManager, StateRequirement, StateSpecError,
};
pub use tensor::{DataType, Quantization, WeightFormat};
pub use weights::{
    F32BlockStream, WeightBinding, WeightBindingError, WeightSpecError, WeightTensorSpec,
};

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticProvider {
        description: ModelDescription,
    }

    impl ModelProvider for StaticProvider {
        fn description(&self) -> &ModelDescription {
            &self.description
        }
    }

    fn qwen_description() -> ModelDescription {
        let model = ModelId::new("Qwen/Qwen3.8-27B").expect("valid model identity");
        let kv = KvStateSpec::new(16, 4, 256, 16, DataType::F16).expect("valid KV spec");
        let matrix = RecurrentMatrixShape::new(48, 128, 128).expect("valid matrix shape");
        let convolution =
            ConvolutionStateShape::new(10_240, 3).expect("valid convolution state shape");
        let recurrent =
            RecurrentStateSpec::new(48, matrix, convolution, DataType::F16, DataType::F32)
                .expect("valid recurrent spec");
        let mtp = MtpCapability::new(1, 3);
        ModelDescription::new(
            model,
            "qwen3.8-hybrid",
            vec![
                ModelRegion::new(ModelRegionId::new(0), ModelRegionKind::RecurrentAttention),
                ModelRegion::new(ModelRegionId::new(1), ModelRegionKind::FullAttention),
                ModelRegion::new(ModelRegionId::new(2), ModelRegionKind::FeedForward),
            ],
            vec![
                StateRequirement::Recurrent(recurrent),
                StateRequirement::FullAttentionKv(kv),
            ],
            ModelCapabilities::new(mtp, true),
            WeightDescription::new(WeightFormat::Gguf, Quantization::GgufQ4Km),
        )
        .expect("valid model description")
    }

    #[test]
    fn hybrid_description_and_plan_keep_state_families_distinct() {
        let description = qwen_description();
        let backend = BackendId::new("cuda").expect("valid backend identity");
        let version = PolicyVersion::new(1).expect("valid policy version");
        let plan = ExecutionPlan::new(
            description.id().clone(),
            backend,
            DeviceId::new(0),
            version,
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            description.state_requirements().to_vec(),
            WeightBinding::empty(description.id().clone(), DeviceId::new(0)),
        )
        .expect("valid plan");

        assert!(plan.requires_kv_state());
        assert!(plan.requires_recurrent_state());
        assert_eq!(plan.variant().status(), QualificationStatus::Experimental);
        assert!(plan.variant().evidence().is_none());
        assert_eq!(
            plan.residency().default_location(),
            ResidencyLocation::Device(DeviceId::new(0))
        );
        assert_eq!(
            description
                .capabilities()
                .mtp()
                .expect("MTP")
                .draft_layers(),
            1
        );
        assert!(description.capabilities().has_vision_encoder());

        let segment = ExecutionSegment::new(
            RequestId::new(1).expect("valid request ID"),
            ExecutionPhase::Decode,
            1,
            1,
            32,
            description.state_requirements().to_vec(),
        )
        .expect("valid segment");
        assert!(plan.validate_segment(&segment).is_ok());
        let wrong_phase = ExecutionSegment::new(
            segment.request(),
            ExecutionPhase::Encoder,
            1,
            1,
            32,
            Vec::new(),
        )
        .expect("valid segment");
        assert_eq!(
            plan.validate_segment(&wrong_phase),
            Err(PlanError::SegmentPhaseMismatch)
        );
    }

    #[test]
    fn provider_rejects_unknown_regions_before_execution() {
        let description = qwen_description();
        let plan = ExecutionPlan::new(
            description.id().clone(),
            BackendId::new("cuda").expect("valid backend identity"),
            DeviceId::new(0),
            PolicyVersion::new(1).expect("valid policy version"),
            vec![ExecutionStage::new(
                ModelRegionId::new(99),
                ExecutionPhase::Decode,
            )],
            description.state_requirements().to_vec(),
            WeightBinding::empty(description.id().clone(), DeviceId::new(0)),
        )
        .expect("valid plan shape");

        let provider = StaticProvider { description };
        assert_eq!(
            provider.validate_plan(&plan),
            Err(ModelError::UnknownRegion(ModelRegionId::new(99)))
        );
    }

    #[test]
    fn typed_state_handles_reject_cross_family_construction() {
        let spec = KvStateSpec::new(16, 4, 256, 16, DataType::F16).expect("valid KV spec");
        let requirement = StateRequirement::FullAttentionKv(spec);
        let handle = StateHandle::new(
            StateId::new(1).expect("valid state ID"),
            requirement,
            StateLocation::Device(DeviceId::new(0)),
            32,
        );
        assert!(KvState::new(handle, spec).is_some());

        let matrix = RecurrentMatrixShape::new(48, 128, 128).expect("valid matrix shape");
        let convolution =
            ConvolutionStateShape::new(10_240, 3).expect("valid convolution state shape");
        let recurrent =
            RecurrentStateSpec::new(48, matrix, convolution, DataType::F16, DataType::F32)
                .expect("valid recurrent spec");
        let wrong_family_handle = StateHandle::new(
            StateId::new(2).expect("valid state ID"),
            requirement,
            StateLocation::Device(DeviceId::new(0)),
            32,
        );
        assert!(RecurrentState::new(wrong_family_handle, recurrent).is_none());

        let recurrent_handle = StateHandle::new(
            StateId::new(3).expect("valid state ID"),
            StateRequirement::Recurrent(recurrent),
            StateLocation::Device(DeviceId::new(0)),
            31,
        );
        let recurrent_state =
            RecurrentState::new(recurrent_handle, recurrent).expect("valid recurrent state");
        let kv_handle = StateHandle::new(
            StateId::new(4).expect("valid state ID"),
            requirement,
            StateLocation::Device(DeviceId::new(0)),
            32,
        );
        let kv_state = KvState::new(kv_handle, spec).expect("valid KV state");
        assert_eq!(
            InferenceStateSet::try_new(Some(kv_state), Some(recurrent_state)),
            Err(StateError::PositionMismatch)
        );
    }

    #[test]
    fn request_semantics_do_not_contain_performance_policy() {
        let sampling = SamplingParams::greedy(Some(42));
        let semantics = RequestSemantics::new(128, sampling, ThinkingMode::Off)
            .expect("valid request semantics");
        let request = RequestSpec::new(
            RequestId::new(1).expect("valid request ID"),
            ModelId::new("Qwen/Qwen3.8-27B").expect("valid model identity"),
            semantics,
        );

        assert_eq!(request.semantics().sampling().seed(), Some(42));
        assert_eq!(request.semantics().thinking(), ThinkingMode::Off);
        assert_eq!(request.semantics().max_output_tokens(), 128);
        assert_eq!(
            request.semantics().prompt_policy(),
            PromptPolicy::plain_text()
        );
        let chat_request = request.clone();
        let chat_semantics =
            chat_request
                .semantics()
                .clone()
                .with_prompt_policy(PromptPolicy::new(
                    PromptFormat::EmbeddedChatTemplate,
                    SpecialTokenPolicy::AddBosAndEos,
                ));
        assert_eq!(
            chat_semantics.prompt_policy().format(),
            PromptFormat::EmbeddedChatTemplate
        );
        assert_eq!(
            chat_semantics.prompt_policy().special_tokens(),
            SpecialTokenPolicy::AddBosAndEos
        );
    }

    #[test]
    fn policy_snapshot_is_versioned_and_validated() {
        let snapshot = PolicySnapshot::new(
            PolicyVersion::new(7).expect("valid policy version"),
            8,
            2048,
            StateTierPreference::Automatic,
            SpeculationPolicy::native_mtp(3).expect("valid MTP budget"),
        )
        .expect("valid policy snapshot");

        assert_eq!(snapshot.version().get(), 7);
        assert_eq!(snapshot.max_batch_tokens(), 2048);
        assert_eq!(
            snapshot.speculation(),
            SpeculationPolicy::NativeMtp {
                max_draft_tokens: 3
            }
        );
        assert!(
            snapshot
                .validate_for(qwen_description().capabilities())
                .is_ok()
        );
        assert_eq!(
            snapshot.validate_for(ModelCapabilities::new(None, false)),
            Err(policy::PolicyError::SpeculationUnavailable)
        );
        assert!(
            PolicySnapshot::new(
                PolicyVersion::new(8).expect("valid policy version"),
                0,
                2048,
                StateTierPreference::Device,
                SpeculationPolicy::Disabled,
            )
            .is_err()
        );
    }

    #[test]
    fn logical_state_manager_accounts_for_typed_allocations() {
        let device = DeviceId::new(0);
        let spec = KvStateSpec::new(1, 1, 2, 4, DataType::F16).expect("KV spec");
        assert_eq!(spec.byte_size(), Some(32));
        let mut manager = LogicalStateManager::new(device, 32, 0);
        let state = manager
            .allocate_kv(spec, StateLocation::Device(device))
            .expect("first allocation");
        assert_eq!(manager.used_bytes(StateLocation::Device(device)), Some(32));
        assert!(matches!(
            manager.allocate_kv(spec, StateLocation::Device(device)),
            Err(StateError::CapacityExceeded { .. })
        ));

        let state_set = InferenceStateSet::try_new(Some(state), None).expect("state set");
        let mut committed = state_set;
        manager.commit(&mut committed, 4).expect("commit");
        assert_eq!(committed.token_position(), Some(4));
        manager
            .release(committed.kv().expect("KV state").handle().clone())
            .expect("release");
        assert_eq!(manager.used_bytes(StateLocation::Device(device)), Some(0));
    }

    #[test]
    fn file_provider_verifies_artifact_metadata_without_reading_weights() {
        let path = std::env::temp_dir().join(format!(
            "engine-core-weight-{}-{}",
            std::process::id(),
            RequestId::new(7).expect("request ID").get()
        ));
        std::fs::write(&path, b"weight metadata probe").expect("write fixture");
        let description = qwen_description();
        let loader = FileWeightLoader::new(
            &path,
            WeightDescription::new(WeightFormat::Gguf, Quantization::GgufQ4Km),
        );
        let provider = FileModelProvider::load(description, &loader).expect("load provider");
        assert_eq!(provider.artifact().byte_len(), 21);
        assert_eq!(provider.artifact().source().as_path(), path);
        assert_eq!(provider.description().architecture(), "qwen3.8-hybrid");
        std::fs::remove_file(path).expect("remove fixture");
    }
}
