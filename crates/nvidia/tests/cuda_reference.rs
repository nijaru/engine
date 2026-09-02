#![cfg(feature = "cuda")]

use engine_core::{
    BackendCapabilities, BackendFeatures, BackendId, BackendKind, DeviceId, ExecutionPhase,
    ExecutionPlan, ExecutionRuntime, ExecutionSegment, ExecutionStage, HybridStateSet,
    LogicalStateManager, ModelCapabilities, ModelDescription, ModelId, ModelProvider, ModelRegion,
    ModelRegionId, ModelRegionKind, NvidiaBackend, PolicyVersion, Quantization, WeightDescription,
    WeightFormat,
};
use engine_nvidia::CudaReferenceDispatcher;

struct ReferenceProvider {
    description: ModelDescription,
}

impl ModelProvider for ReferenceProvider {
    fn description(&self) -> &ModelDescription {
        &self.description
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_reference_linear_layer_through_core_runtime() {
    let device = DeviceId::new(0);
    let model = ModelId::new("engine/reference-linear").expect("model ID");
    let description = ModelDescription::new(
        model.clone(),
        "reference-linear",
        vec![ModelRegion::new(
            ModelRegionId::new(0),
            ModelRegionKind::OutputProjection,
        )],
        Vec::new(),
        ModelCapabilities::new(None, false),
        WeightDescription::new(
            WeightFormat::Vendor("reference".to_owned()),
            Quantization::None,
        ),
    )
    .expect("model description");
    let backend_id = BackendId::new("cuda-reference").expect("backend ID");
    let capabilities = BackendCapabilities::new(
        backend_id.clone(),
        device,
        BackendKind::Cuda,
        1 << 30,
        BackendFeatures::new(
            vec![engine_core::DataType::F32],
            vec![Quantization::None],
            false,
            true,
        ),
    );
    let backend = NvidiaBackend::new(
        capabilities,
        CudaReferenceDispatcher::new(device.get() as usize).expect("CUDA reference dispatcher"),
    )
    .expect("CUDA backend");
    let state_manager = LogicalStateManager::new(device, 0, 0);
    let mut runtime =
        ExecutionRuntime::new(ReferenceProvider { description }, backend, state_manager);
    let policy = PolicyVersion::new(1).expect("policy version");
    let plan = ExecutionPlan::new(
        model,
        backend_id,
        device,
        policy,
        vec![ExecutionStage::new(
            ModelRegionId::new(0),
            ExecutionPhase::Decode,
        )],
        Vec::new(),
    )
    .expect("execution plan");
    let segment = ExecutionSegment::new(
        engine_core::RequestId::new(1).expect("request ID"),
        ExecutionPhase::Decode,
        1,
        1,
        0,
        Vec::new(),
    )
    .expect("execution segment");
    let (event, state) = runtime
        .execute_segment(
            &plan,
            &segment,
            HybridStateSet::try_new(None, None).expect("state"),
        )
        .expect("reference execution");

    assert_eq!(event.phase(), ExecutionPhase::Decode);
    assert_eq!(event.token_count(), 1);
    assert_eq!(event.policy_version(), policy);
    assert_eq!(state.token_position(), Some(1));
    assert_eq!(
        runtime.backend().dispatcher().last_output(),
        Some([16.0, 14.0].as_slice())
    );
    assert!(event.metrics().elapsed_nanos() > 0);
}
