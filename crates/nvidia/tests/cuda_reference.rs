#![cfg(feature = "cuda")]

use engine_core::{
    BackendCapabilities, BackendFeatures, BackendId, BackendKind, DeviceId, ExecutionPhase,
    ExecutionPlan, ExecutionRuntime, ExecutionSegment, ExecutionStage, HybridStateSet,
    LogicalStateManager, ModelCapabilities, ModelDescription, ModelId, ModelProvider, ModelRegion,
    ModelRegionId, ModelRegionKind, NvidiaBackend, PolicyVersion, Quantization, WeightBinding,
    WeightDescription, WeightFormat,
};
use engine_gguf::{GgufFile, Qwen35ModelProvider};
use engine_nvidia::CudaReferenceDispatcher;

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend(value.as_bytes());
}

fn reference_gguf() -> Vec<u8> {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 0x4655_4747);
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 1);
    push_u64(&mut bytes, 1);
    push_string(&mut bytes, "general.alignment");
    push_u32(&mut bytes, 4);
    push_u32(&mut bytes, 32);
    push_string(&mut bytes, "reference.weight");
    push_u32(&mut bytes, 2);
    push_u64(&mut bytes, 2);
    push_u64(&mut bytes, 4);
    push_u32(&mut bytes, 0);
    push_u64(&mut bytes, 0);
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    for value in [1.0_f32, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0] {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

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
    let dispatcher =
        CudaReferenceDispatcher::new(device.get() as usize).expect("CUDA reference dispatcher");
    let weights = WeightBinding::new(
        model.clone(),
        device,
        vec![
            dispatcher
                .weight_spec()
                .expect("reference weight spec")
                .clone(),
        ],
    )
    .expect("reference weight binding");
    let backend = NvidiaBackend::new(capabilities, dispatcher).expect("CUDA backend");
    let state_manager = LogicalStateManager::new(device, 0, 0);
    let mut runtime =
        ExecutionRuntime::new(ReferenceProvider { description }, backend, state_manager);
    let policy = PolicyVersion::new(1).expect("policy version");
    let plan = ExecutionPlan::new(
        model.clone(),
        backend_id,
        device,
        policy,
        vec![ExecutionStage::new(
            ModelRegionId::new(0),
            ExecutionPhase::Decode,
        )],
        Vec::new(),
        weights,
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
    assert_eq!(state.token_position(), None);
    assert_eq!(
        runtime.backend().dispatcher().last_output(),
        Some([16.0, 14.0].as_slice())
    );
    assert!(event.metrics().elapsed_nanos() > 0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn materializes_a_bounded_gguf_tensor_before_reference_execution() {
    let path = std::env::temp_dir().join(format!(
        "engine-nvidia-reference-{}-{}.gguf",
        std::process::id(),
        2
    ));
    std::fs::write(&path, reference_gguf()).expect("write GGUF fixture");
    let file = GgufFile::open(&path).expect("open GGUF fixture");
    let mut source = file
        .open_tensor("reference.weight")
        .expect("open reference tensor");
    assert_eq!(source.value_type(), 0);
    assert_eq!(source.spec().element_count(), 8);
    assert_eq!(source.remaining(), 32);
    let spec = source.spec().clone();
    let dispatcher =
        CudaReferenceDispatcher::from_f32_source(0, &mut source).expect("materialize GGUF tensor");
    assert_eq!(dispatcher.weight_spec(), Some(&spec));

    let device = DeviceId::new(0);
    let model = ModelId::new("engine/reference-gguf").expect("model ID");
    let description = ModelDescription::new(
        model.clone(),
        "reference-gguf",
        vec![ModelRegion::new(
            ModelRegionId::new(0),
            ModelRegionKind::OutputProjection,
        )],
        Vec::new(),
        ModelCapabilities::new(None, false),
        WeightDescription::new(WeightFormat::Gguf, Quantization::None),
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
    let weights = WeightBinding::new(model.clone(), device, vec![spec]).expect("weight binding");
    let backend = NvidiaBackend::new(capabilities, dispatcher).expect("CUDA backend");
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
        weights,
    )
    .expect("execution plan");
    let segment = ExecutionSegment::new(
        engine_core::RequestId::new(2).expect("request ID"),
        ExecutionPhase::Decode,
        1,
        1,
        0,
        Vec::new(),
    )
    .expect("execution segment");
    let (event, _) = runtime
        .execute_segment(
            &plan,
            &segment,
            HybridStateSet::try_new(None, None).expect("state"),
        )
        .expect("reference execution");

    assert_eq!(
        runtime.backend().dispatcher().last_output(),
        Some([16.0, 14.0].as_slice())
    );
    assert!(event.metrics().elapsed_nanos() > 0);
    std::fs::remove_file(path).expect("remove GGUF fixture");
}

#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn materializes_a_pinned_qwen_scalar_tensor_without_claiming_model_execution() {
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");
    let mut source = provider
        .open_tensor("blk.0.ssm_a")
        .expect("open pinned scalar tensor");
    assert_eq!(source.value_type(), 0);
    assert_eq!(source.spec().dimensions(), &[48]);
    let spec = source.spec().clone();
    let mut expected = Vec::new();
    while let Some(block) = source
        .read_dequantized_block()
        .expect("decode pinned scalar block")
    {
        expected.extend(block);
    }
    assert_eq!(expected.len(), 48);

    let mut materialized_source = provider
        .open_tensor("blk.0.ssm_a")
        .expect("reopen pinned scalar tensor");
    let dispatcher = CudaReferenceDispatcher::from_f32_source(0, &mut materialized_source)
        .expect("materialize pinned scalar tensor");
    let actual = dispatcher
        .copy_weight_to_host()
        .expect("copy pinned scalar tensor to host");
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );
    assert_eq!(dispatcher.weight_spec(), Some(&spec));
}
