#![cfg(feature = "cuda")]

use std::io::{Cursor, Read};

use cudarc::driver::CudaContext;
use engine_core::{
    BackendCapabilities, BackendFeatures, BackendId, BackendKind, ConvolutionStateShape, DeviceId,
    ExecutionPhase, ExecutionPlan, ExecutionRuntime, ExecutionSegment, ExecutionStage,
    HybridStateSet, LogicalStateManager, ModelCapabilities, ModelDescription, ModelId,
    ModelProvider, ModelRegion, ModelRegionId, ModelRegionKind, NvidiaBackend, PolicyVersion,
    Quantization, RecurrentMatrixShape, RecurrentStateSpec, StateLocation, StateManager,
    WeightBinding, WeightDescription, WeightFormat, WeightTensorSpec,
};
use engine_gguf::{GgufFile, Qwen35LayerKind, Qwen35ModelProvider};
use engine_nvidia::{
    CudaHybridState, CudaQ4KGemv, CudaReferenceDispatcher, CudaStateError, CudaWeightStore,
};

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
    assert_eq!(
        provider.layer_kind(0).expect("recurrent layer kind"),
        Qwen35LayerKind::Recurrent
    );
    assert_eq!(
        provider.layer_kind(3).expect("full-attention layer kind"),
        Qwen35LayerKind::FullAttention
    );
    assert_eq!(
        provider
            .layer_weight_binding(DeviceId::new(0), 0)
            .expect("bind recurrent layer")
            .tensors()
            .len(),
        14
    );
    assert_eq!(
        provider
            .layer_weight_binding(DeviceId::new(0), 3)
            .expect("bind full-attention layer")
            .tensors()
            .len(),
        11
    );
    let binding = provider
        .weight_binding(DeviceId::new(0), &["blk.0.ssm_a"])
        .expect("bind pinned scalar tensor");
    assert_eq!(binding.tensors().len(), 1);
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

#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn materializes_a_pinned_qwen_quantized_block_without_host_dequantization() {
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");
    let mut source = provider
        .open_tensor("blk.0.attn_qkv.weight")
        .expect("open pinned quantized tensor");
    assert_eq!(source.value_type(), 13); // Q5_K
    let mut first_block = [0_u8; 144];
    source
        .read_exact(&mut first_block)
        .expect("read first encoded Q5_K block");
    let spec = WeightTensorSpec::new(
        "blk.0.attn_qkv.weight.first_block",
        vec![256],
        engine_core::DataType::F32,
    )
    .expect("first-block descriptor");
    let mut encoded = Cursor::new(first_block.to_vec());

    let mut dispatcher = CudaReferenceDispatcher::new(0).expect("CUDA reference dispatcher");
    let materialized = dispatcher
        .materialize_quantized(spec.clone(), 13, first_block.len() as u64, &mut encoded)
        .expect("materialize opaque quantized block");
    assert_eq!(materialized, spec);
    let actual = dispatcher
        .copy_quantized_to_host(spec.name())
        .expect("copy opaque quantized block to host");
    assert_eq!(actual, first_block);
}

fn q4_k_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 144];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
    for index in 0..4 {
        let index = u8::try_from(index).expect("Q4_K scale index");
        block[4 + usize::from(index)] = (1 + seed + index) | ((index & 3) << 6);
        block[8 + usize::from(index)] = (1 + index) | (((index + 1) & 3) << 6);
        block[12 + usize::from(index)] = (5 + seed + index) | ((2 + index) << 4);
    }
    for (index, value) in block[16..].iter_mut().enumerate() {
        let pattern =
            u8::try_from((index + usize::from(seed) * 5) % 16).expect("Q4_K fixture index");
        let low = pattern.wrapping_mul(3) & 0x0f;
        let high = 15_u8.wrapping_sub(pattern.wrapping_mul(3)) & 0x0f;
        *value = low | (high << 4);
    }
    block
}

fn q4_k_fixture() -> Vec<u8> {
    (0_u8..4).flat_map(q4_k_fixture_block).collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_q4_k_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = q4_k_fixture();
    let spec = WeightTensorSpec::new("q4_k.fixture", vec![512, 2], engine_core::DataType::F32)
        .expect("Q4_K fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            12,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload Q4_K fixture");
    let weight = store
        .quantized_tensor("q4_k.fixture")
        .expect("Q4_K fixture weight");
    let input = (0..512)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("Q4_K input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(2).expect("allocate output");
    let kernel = CudaQ4KGemv::from_context(&context, stream.clone()).expect("compile Q4_K");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute Q4_K GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..2)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 144;
                    engine_gguf::dequantize_block(12, &encoded[start..start + 144])
                        .expect("decode Q4_K fixture")
                        .into_iter()
                        .zip(&input[block_index * 256..(block_index + 1) * 256])
                        .map(|(weight, input)| weight * input)
                })
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-3);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn allocates_distinct_physical_hybrid_state_buffers() {
    let device = DeviceId::new(0);
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let kv_spec =
        engine_core::KvStateSpec::new(1, 2, 4, 4, engine_core::DataType::F16).expect("KV spec");
    let recurrent_spec = RecurrentStateSpec::new(
        1,
        RecurrentMatrixShape::new(1, 2, 2, 2).expect("matrix shape"),
        ConvolutionStateShape::new(3, 2).expect("convolution shape"),
        engine_core::DataType::F32,
        engine_core::DataType::F32,
    )
    .expect("recurrent spec");
    let mut manager = LogicalStateManager::new(device, 1024, 0);
    let kv = manager
        .allocate_kv(kv_spec, StateLocation::Device(device))
        .expect("KV allocation");
    let recurrent = manager
        .allocate_recurrent(recurrent_spec, StateLocation::Device(device))
        .expect("recurrent allocation");
    let core_state = HybridStateSet::try_new(Some(kv), Some(recurrent)).expect("hybrid state");

    let mut physical =
        CudaHybridState::from_state_set(stream, &core_state).expect("physical hybrid state");
    assert_eq!(physical.kv().expect("KV state").keys().len(), 32);
    assert_eq!(
        physical.kv().expect("KV state").keys().dtype(),
        engine_core::DataType::F16
    );
    assert_eq!(physical.kv().expect("KV state").values().len(), 32);
    assert_eq!(physical.kv().expect("KV state").byte_size(), Some(128));
    assert_eq!(
        physical
            .recurrent()
            .expect("recurrent state")
            .matrix()
            .len(),
        8
    );
    assert_eq!(
        physical
            .recurrent()
            .expect("recurrent state")
            .convolution()
            .len(),
        6
    );
    assert_eq!(
        physical.recurrent().expect("recurrent state").byte_size(),
        Some(56)
    );
    assert_eq!(physical.byte_size(), Some(184));
    physical.zero().expect("zero physical state");
    physical.advance_to(2).expect("advance state");
    assert!(matches!(
        physical.advance_to(1),
        Err(CudaStateError::PositionRegression { .. })
    ));
}
