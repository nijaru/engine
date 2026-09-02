#![cfg(feature = "cuda")]

use std::io::{Cursor, Read};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
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
    CudaHybridState, CudaIq3SEmbedding, CudaIq3SGemv, CudaIq4NlGemv, CudaIq4XsGemv, CudaQ3KGemv,
    CudaQ4KGemv, CudaQ5KGemv, CudaQ6KGemv, CudaQ8_0Gemv, CudaQwen35Ops, CudaReferenceDispatcher,
    CudaStateBuffer, CudaStateError, CudaWeightStore,
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
    let mut first_block = [0_u8; 176];
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

fn q5_k_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = q4_k_fixture_block(seed);
    block.resize(176, 0);
    block[16..48].fill(0x55_u8.rotate_left(u32::from(seed)));
    block[48..]
        .iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let pattern =
                u8::try_from((index + usize::from(seed) * 7) % 16).expect("Q5_K fixture index");
            let low = pattern.wrapping_mul(5) & 0x0f;
            let high = 15_u8.wrapping_sub(pattern.wrapping_mul(5)) & 0x0f;
            *value = low | (high << 4);
        });
    block
}

fn q5_k_fixture() -> Vec<u8> {
    (0_u8..4).flat_map(q5_k_fixture_block).collect()
}

fn q3_k_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 110];
    for (index, value) in block[..32].iter_mut().enumerate() {
        let pattern =
            u8::try_from((index + usize::from(seed) * 3) % 256).expect("Q3_K high-bit index");
        *value = pattern.rotate_left(1);
    }
    for (index, value) in block[32..96].iter_mut().enumerate() {
        let pattern =
            u8::try_from((index * 5 + usize::from(seed) * 7) % 256).expect("Q3_K low-bit index");
        *value = pattern;
    }
    for (index, value) in block[96..104].iter_mut().enumerate() {
        let index = u8::try_from(index).expect("Q3_K scale index");
        *value = ((index + seed) & 0x0f) | (((index * 3 + seed) & 0x0f) << 4);
    }
    for (index, value) in block[104..108].iter_mut().enumerate() {
        let index = u8::try_from(index).expect("Q3_K packed scale index");
        *value = ((index + seed) & 3)
            | (((index + 1 + seed) & 3) << 2)
            | (((index + 2 + seed) & 3) << 4)
            | (((index + 3 + seed) & 3) << 6);
    }
    block[108..110].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block
}

fn q3_k_fixture() -> Vec<u8> {
    (0_u8..4).flat_map(q3_k_fixture_block).collect()
}

fn q6_k_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 210];
    for (index, value) in block[..128].iter_mut().enumerate() {
        *value =
            u8::try_from((index * 7 + usize::from(seed) * 11) % 256).expect("Q6_K low-bit index");
    }
    for (index, value) in block[128..192].iter_mut().enumerate() {
        *value =
            u8::try_from((index * 13 + usize::from(seed) * 5) % 256).expect("Q6_K high-bit index");
    }
    for (index, value) in block[192..208].iter_mut().enumerate() {
        let pattern = u8::try_from((index + usize::from(seed) * 3) % 17).expect("Q6_K scale index");
        let scale = i8::try_from(pattern).expect("Q6_K scale") - 8;
        *value = scale.to_ne_bytes()[0];
    }
    block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block
}

fn q6_k_fixture() -> Vec<u8> {
    (0_u8..4).flat_map(q6_k_fixture_block).collect()
}

fn q8_0_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 34];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for index in 0..32 {
        let pattern =
            u8::try_from((index + usize::from(seed) * 3) % 31).expect("Q8_0 fixture index");
        let quantized = i8::try_from(pattern).expect("Q8_0 value") - 15;
        block[2 + index] = quantized.to_ne_bytes()[0];
    }
    block
}

fn q8_0_fixture() -> Vec<u8> {
    (0_u8..6).flat_map(q8_0_fixture_block).collect()
}

fn iq3_s_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 110];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for (index, value) in block[2..66].iter_mut().enumerate() {
        let index = u8::try_from(index).expect("IQ3_S low-code index");
        *value = index.wrapping_mul(37).wrapping_add(seed.wrapping_mul(13));
    }
    for (index, value) in block[66..74].iter_mut().enumerate() {
        let index = u8::try_from(index).expect("IQ3_S high-code index");
        *value = 0x55_u8.rotate_left(u32::from(index % 8)) ^ seed.wrapping_mul(17);
    }
    for (index, value) in block[74..106].iter_mut().enumerate() {
        let index = u8::try_from(index).expect("IQ3_S sign index");
        *value = 0x33_u8.rotate_left(u32::from(index % 8)) ^ seed.wrapping_mul(29);
    }
    for (index, value) in block[106..110].iter_mut().enumerate() {
        let index = u8::try_from(index).expect("IQ3_S scale index");
        *value = index.wrapping_mul(5).wrapping_add(seed.wrapping_mul(3));
    }
    block
}

fn iq3_s_fixture() -> Vec<u8> {
    (0_u8..6).flat_map(iq3_s_fixture_block).collect()
}

fn iq4_nl_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 18];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for (index, value) in block[2..].iter_mut().enumerate() {
        let low = u8::try_from((index + usize::from(seed)) % 16).expect("IQ4_NL low index");
        let high = u8::try_from((15 + usize::from(seed) - index) % 16).expect("IQ4_NL high index");
        *value = low | (high << 4);
    }
    block
}

fn iq4_nl_fixture() -> Vec<u8> {
    (0_u8..6).flat_map(iq4_nl_fixture_block).collect()
}

fn iq4_xs_fixture_block(seed: u8) -> Vec<u8> {
    let mut block = vec![0_u8; 136];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut high_scales = 0_u16;
    for group in 0..8 {
        let group_bits = u16::try_from(group).expect("IQ4_XS group");
        high_scales |= ((group_bits + u16::from(seed)) & 3) << (group_bits * 2);
        let low = u8::try_from((group + usize::from(seed)) % 16).expect("IQ4_XS low scale");
        let byte = &mut block[4 + group / 2];
        if group % 2 == 0 {
            *byte = (*byte & 0xf0) | low;
        } else {
            *byte = (*byte & 0x0f) | (low << 4);
        }
    }
    block[2..4].copy_from_slice(&high_scales.to_le_bytes());
    for (index, value) in block[8..].iter_mut().enumerate() {
        let low = u8::try_from((index + usize::from(seed) * 5) % 16).expect("IQ4_XS low index");
        let high = u8::try_from((index * 7 + usize::from(seed)) % 16).expect("IQ4_XS high index");
        *value = low | (high << 4);
    }
    block
}

fn iq4_xs_fixture() -> Vec<u8> {
    (0_u8..4).flat_map(iq4_xs_fixture_block).collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_iq3_s_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = iq3_s_fixture();
    let spec = WeightTensorSpec::new("iq3_s.fixture", vec![512, 3], engine_core::DataType::F32)
        .expect("IQ3_S fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            21,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload IQ3_S fixture");
    let weight = store
        .quantized_tensor("iq3_s.fixture")
        .expect("IQ3_S fixture weight");
    let input = (0..512)
        .map(|index| {
            let pattern = u8::try_from(index % 19).expect("IQ3_S input index");
            f32::from(pattern) * 0.0625 - 0.5
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(3).expect("allocate output");
    let kernel = CudaIq3SGemv::from_context(&context, stream.clone()).expect("compile IQ3_S");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute IQ3_S GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..3)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 110;
                    engine_gguf::dequantize_block(21, &encoded[start..start + 110])
                        .expect("decode IQ3_S fixture")
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
fn executes_iq3_s_embedding_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = iq3_s_fixture();
    let spec = WeightTensorSpec::new("iq3_s.embedding", vec![512, 3], engine_core::DataType::F32)
        .expect("IQ3_S embedding fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            21,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload IQ3_S embedding fixture");
    let weight = store
        .quantized_tensor("iq3_s.embedding")
        .expect("IQ3_S embedding weight");
    let mut output = stream
        .alloc_zeros::<f32>(512)
        .expect("allocate embedding output");
    let kernel =
        CudaIq3SEmbedding::from_context(&context, stream.clone()).expect("compile IQ3_S embedding");
    kernel
        .execute(weight, 1, &mut output)
        .expect("execute IQ3_S embedding");
    let actual = stream.clone_dtoh(&output).expect("download embedding");
    let expected = (0..2)
        .flat_map(|block_index| {
            let start = (2 + block_index) * 110;
            engine_gguf::dequantize_block(21, &encoded[start..start + 110])
                .expect("decode IQ3_S embedding fixture")
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-3);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_iq4_nl_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = iq4_nl_fixture();
    let spec = WeightTensorSpec::new("iq4_nl.fixture", vec![64, 3], engine_core::DataType::F32)
        .expect("IQ4_NL fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            20,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload IQ4_NL fixture");
    let weight = store
        .quantized_tensor("iq4_nl.fixture")
        .expect("IQ4_NL fixture weight");
    let input = (0..64)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("IQ4_NL input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(3).expect("allocate output");
    let kernel = CudaIq4NlGemv::from_context(&context, stream.clone()).expect("compile IQ4_NL");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute IQ4_NL GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..3)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 18;
                    engine_gguf::dequantize_block(20, &encoded[start..start + 18])
                        .expect("decode IQ4_NL fixture")
                        .into_iter()
                        .zip(&input[block_index * 32..(block_index + 1) * 32])
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
fn executes_iq4_xs_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = iq4_xs_fixture();
    let spec = WeightTensorSpec::new("iq4_xs.fixture", vec![512, 2], engine_core::DataType::F32)
        .expect("IQ4_XS fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            23,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload IQ4_XS fixture");
    let weight = store
        .quantized_tensor("iq4_xs.fixture")
        .expect("IQ4_XS fixture weight");
    let input = (0..512)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("IQ4_XS input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(2).expect("allocate output");
    let kernel = CudaIq4XsGemv::from_context(&context, stream.clone()).expect("compile IQ4_XS");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute IQ4_XS GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..2)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 136;
                    engine_gguf::dequantize_block(23, &encoded[start..start + 136])
                        .expect("decode IQ4_XS fixture")
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
fn executes_q3_k_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = q3_k_fixture();
    let spec = WeightTensorSpec::new("q3_k.fixture", vec![512, 2], engine_core::DataType::F32)
        .expect("Q3_K fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            11,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload Q3_K fixture");
    let weight = store
        .quantized_tensor("q3_k.fixture")
        .expect("Q3_K fixture weight");
    let input = (0..512)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("Q3_K input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(2).expect("allocate output");
    let kernel = CudaQ3KGemv::from_context(&context, stream.clone()).expect("compile Q3_K");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute Q3_K GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..2)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 110;
                    engine_gguf::dequantize_block(11, &encoded[start..start + 110])
                        .expect("decode Q3_K fixture")
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
fn executes_q6_k_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = q6_k_fixture();
    let spec = WeightTensorSpec::new("q6_k.fixture", vec![512, 2], engine_core::DataType::F32)
        .expect("Q6_K fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            14,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload Q6_K fixture");
    let weight = store
        .quantized_tensor("q6_k.fixture")
        .expect("Q6_K fixture weight");
    let input = (0..512)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("Q6_K input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(2).expect("allocate output");
    let kernel = CudaQ6KGemv::from_context(&context, stream.clone()).expect("compile Q6_K");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute Q6_K GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..2)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 210;
                    engine_gguf::dequantize_block(14, &encoded[start..start + 210])
                        .expect("decode Q6_K fixture")
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
fn executes_q8_0_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = q8_0_fixture();
    let spec = WeightTensorSpec::new("q8_0.fixture", vec![64, 3], engine_core::DataType::F32)
        .expect("Q8_0 fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            8,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload Q8_0 fixture");
    let weight = store
        .quantized_tensor("q8_0.fixture")
        .expect("Q8_0 fixture weight");
    let input = (0..64)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("Q8_0 input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(3).expect("allocate output");
    let kernel = CudaQ8_0Gemv::from_context(&context, stream.clone()).expect("compile Q8_0");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute Q8_0 GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..3)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 34;
                    engine_gguf::dequantize_block(8, &encoded[start..start + 34])
                        .expect("decode Q8_0 fixture")
                        .into_iter()
                        .zip(&input[block_index * 32..(block_index + 1) * 32])
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
fn executes_q5_k_gemv_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let encoded = q5_k_fixture();
    let spec = WeightTensorSpec::new("q5_k.fixture", vec![512, 2], engine_core::DataType::F32)
        .expect("Q5_K fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            13,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload Q5_K fixture");
    let weight = store
        .quantized_tensor("q5_k.fixture")
        .expect("Q5_K fixture weight");
    let input = (0..512)
        .map(|index| {
            let pattern = u8::try_from(index % 17).expect("Q5_K input index");
            f32::from(pattern) * 0.125 - 1.0
        })
        .collect::<Vec<_>>();
    let input_device = stream.clone_htod(&input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(2).expect("allocate output");
    let kernel = CudaQ5KGemv::from_context(&context, stream.clone()).expect("compile Q5_K");
    kernel
        .execute(weight, &input_device, &mut output)
        .expect("execute Q5_K GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");
    let expected = (0..2)
        .map(|output_index| {
            (0..2)
                .flat_map(|block_index| {
                    let start = (output_index * 2 + block_index) * 176;
                    engine_gguf::dequantize_block(13, &encoded[start..start + 176])
                        .expect("decode Q5_K fixture")
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
fn executes_qwen_elementwise_ops_against_host_equations() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    let input = vec![1.0_f32, -2.0, 3.0, -4.0];
    let weights = vec![1.0_f32, 0.5, 2.0, -1.0];
    let input_device = stream.clone_htod(&input).expect("upload RMSNorm input");
    let weights_device = stream.clone_htod(&weights).expect("upload RMSNorm weights");
    let mut normalized = stream
        .alloc_zeros::<f32>(input.len())
        .expect("allocate RMSNorm output");
    ops.rms_norm(&input_device, &weights_device, &mut normalized, 1e-5)
        .expect("execute RMSNorm");
    let actual = stream
        .clone_dtoh(&normalized)
        .expect("download RMSNorm output");
    let inverse_norm = (input.iter().map(|value| value * value).sum::<f32>() / 4.0 + 1e-5)
        .sqrt()
        .recip();
    for ((actual, input), weight) in actual.iter().zip(&input).zip(&weights) {
        let expected = input * inverse_norm * weight;
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    let gate = (0..513)
        .map(|index| f32::from(u16::try_from(index).expect("SiLU index fits")) * 0.01 - 2.5)
        .collect::<Vec<_>>();
    let up = (0..513)
        .map(|index| 1.0 - f32::from(u16::try_from(index).expect("SiLU index fits")) * 0.002)
        .collect::<Vec<_>>();
    let gate_device = stream.clone_htod(&gate).expect("upload SiLU gate");
    let up_device = stream.clone_htod(&up).expect("upload SiLU up");
    let mut fused = stream
        .alloc_zeros::<f32>(gate.len())
        .expect("allocate SiLU output");
    ops.silu_mul(&gate_device, &up_device, &mut fused)
        .expect("execute SiLU multiplication");
    let actual = stream.clone_dtoh(&fused).expect("download SiLU output");
    for ((actual, gate), up) in actual.iter().zip(&gate).zip(&up) {
        let expected = gate / (1.0 + (-gate).exp()) * up;
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    let mut logits = vec![-10.0_f32; 513];
    logits[401] = 7.0;
    logits[402] = 7.0;
    let logits_device = stream.clone_htod(&logits).expect("upload logits");
    assert_eq!(
        ops.argmax(&logits_device).expect("select greedy token"),
        401
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn allocates_distinct_physical_hybrid_state_buffers() {
    let (kv_spec, recurrent_spec) = hybrid_state_fixture();
    let device = DeviceId::new(0);
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let mut manager = LogicalStateManager::new(device, 1024, 0);
    let kv = manager
        .allocate_kv(kv_spec, StateLocation::Device(device))
        .expect("KV allocation");
    let recurrent = manager
        .allocate_recurrent(recurrent_spec, StateLocation::Device(device))
        .expect("recurrent allocation");
    let core_state = HybridStateSet::try_new(Some(kv), Some(recurrent)).expect("hybrid state");

    let mut physical = CudaHybridState::from_state_set(stream.clone(), &core_state)
        .expect("physical hybrid state");
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

    write_and_validate_physical_state(&mut physical, &stream, kv_spec, recurrent_spec);
}

/// Exercise typed physical state writes and validation errors after the
/// allocation test has built the fixture state.
fn write_and_validate_physical_state(
    physical: &mut CudaHybridState,
    stream: &Arc<CudaStream>,
    kv_spec: engine_core::KvStateSpec,
    recurrent_spec: RecurrentStateSpec,
) {
    let keys = vec![1_u16, 2, 3, 4, 5, 6, 7, 8];
    let values = vec![9_u16, 10, 11, 12, 13, 14, 15, 16];
    physical
        .write_kv_token(0, 1, &keys, &values)
        .expect("write KV token");
    let matrix = vec![0.25_f32; 8];
    let convolution = vec![0.5_f32; 6];
    physical
        .write_recurrent_layer(0, &matrix, &convolution)
        .expect("write recurrent layer");
    physical.zero().expect("zero after writes");
    let kv = physical.kv().expect("KV state after zero");
    let CudaStateBuffer::F16(keys_slice) = kv.keys() else {
        panic!("expected F16 KV keys");
    };
    let keys = stream.clone_dtoh(keys_slice).expect("download zeroed keys");
    assert!(keys.iter().all(|&key| key == 0));
    assert!(matches!(
        physical.write_kv_token(1, 0, &keys, &keys),
        Err(CudaStateError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        physical.write_kv_token(0, 4, &keys, &keys),
        Err(CudaStateError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        physical.write_recurrent_layer(1, &matrix, &convolution),
        Err(CudaStateError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        physical.write_recurrent_layer(0, &matrix, &[]),
        Err(CudaStateError::ShapeMismatch { .. })
    ));
    let mut kv_only =
        CudaHybridState::from_specs(stream.clone(), Some(kv_spec), None).expect("KV-only state");
    assert!(matches!(
        kv_only.write_recurrent_layer(0, &matrix, &convolution),
        Err(CudaStateError::MissingFamily { .. })
    ));
    let mut recurrent_only =
        CudaHybridState::from_specs(stream.clone(), None, Some(recurrent_spec))
            .expect("recurrent-only");
    assert!(matches!(
        recurrent_only.write_kv_token(0, 0, &keys, &keys),
        Err(CudaStateError::MissingFamily { .. })
    ));
}

fn hybrid_state_fixture() -> (engine_core::KvStateSpec, RecurrentStateSpec) {
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
    (kv_spec, recurrent_spec)
}

/// Dequantize one entire pinned tensor to host F32 in GGUF flat order
/// (column-major over [ne0, ne1]: element (i0, i1) at i0 + i1*ne0).
fn pinned_tensor_f32(provider: &Qwen35ModelProvider, name: &str) -> Vec<f32> {
    let mut reader = provider.open_tensor(name).expect("open pinned tensor");
    let mut values = Vec::new();
    while let Some(block) = reader
        .read_dequantized_block()
        .expect("decode pinned tensor block")
    {
        values.extend(block);
    }
    let expected =
        usize::try_from(reader.spec().element_count()).expect("tensor element count fits the host");
    assert_eq!(values.len(), expected, "tensor {name} decoded length");
    values
}

/// ggml `mul_mat` GEMV against a GGUF `[in_dim, out_dim]` tensor stored flat
/// column-major over `ne0 = in_dim`: column `n` is `values[n*in_dim..(n+1)*in_dim]`.
fn gguf_gemv(weight: &[f32], in_dim: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    (0..weight.len() / in_dim)
        .map(|n| {
            let column = &weight[n * in_dim..(n + 1) * in_dim];
            x.iter().zip(column).map(|(xv, wv)| xv * wv).sum()
        })
        .collect()
}

/// Qwen3.8-27B GDN layer geometry, verified against the pinned artifact.
const GDN: (usize, usize, usize, usize) = (16, 48, 128, 4);
const GDN_K_HEADS: usize = GDN.0;
const GDN_V_HEADS: usize = GDN.1;
const GDN_HEAD_DIM: usize = GDN.2;
const GDN_D_CONV: usize = GDN.3;
const GDN_HEAD_DIM_U16: u16 = 128;
const GDN_K_OFFSET: usize = 2048;
const GDN_V_OFFSET: usize = 4096;
const GDN_QKV_DIM: usize = 10240;
const GDN_INNER: usize = 6144;

/// Per-head l2 normalization with an eps floor on the norm.
fn l2_normalize(values: &[f32], eps: f32) -> Vec<f32> {
    let sum: f32 = values.iter().map(|x| x * x).sum();
    let scale = 1.0 / sum.sqrt().max(eps);
    values.iter().map(|x| x * scale).collect()
}

/// softplus with the ggml-compatible large-input shortcut.
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

/// Column-oriented recurrent state update for one V head:
/// `sk = S^T k`, `d = (v - sk) * beta`, `S += k (outer) d`, `o = S^T q`.
/// `S` is row-major `[head_dim][head_dim]`. Names match the pinned equations.
#[allow(clippy::many_single_char_names)]
fn gdn_state_step(
    state: &mut [f32],
    k: &[f32],
    v: &[f32],
    q: &[f32],
    decay: f32,
    beta: f32,
) -> Vec<f32> {
    let head_dim = GDN_HEAD_DIM;
    for state_elem in state.iter_mut() {
        *state_elem *= decay;
    }
    let mut sk = vec![0.0_f32; head_dim];
    for (row, state_row) in state.chunks_exact(head_dim).enumerate() {
        for (col, state_elem) in state_row.iter().enumerate() {
            sk[col] += state_elem * k[row];
        }
    }
    let d: Vec<f32> = (0..head_dim).map(|col| (v[col] - sk[col]) * beta).collect();
    for (row, state_row) in state.chunks_exact_mut(head_dim).enumerate() {
        for (state_elem, d_elem) in state_row.iter_mut().zip(&d) {
            *state_elem += k[row] * d_elem;
        }
    }
    (0..head_dim)
        .map(|col| {
            (0..head_dim)
                .map(|row| state[row * head_dim + col] * q[row])
                .sum()
        })
        .collect()
}

/// Causal depth-4 convolution over `[history | new]` per channel, then `SiLU`.
/// The GGUF conv weight stores element `(tap, channel)` at
/// `tap + channel * d_conv`. Advances `conv` in place: drop the oldest
/// input, append the new token's projection. Returns the activated output.
fn gdn_conv_step(qkv_mixed: &[f32], ssm_conv1d: &[f32], conv: &mut [f32]) -> Vec<f32> {
    let history_len = GDN_D_CONV - 1;
    let conv_out: Vec<f32> = qkv_mixed
        .iter()
        .enumerate()
        .map(|(channel, new_input)| {
            (0..GDN_D_CONV)
                .map(|tap| {
                    let input = if tap < history_len {
                        conv[channel * history_len + tap]
                    } else {
                        *new_input
                    };
                    input * ssm_conv1d[tap + channel * GDN_D_CONV]
                })
                .sum::<f32>()
        })
        .collect();
    for (channel, new_input) in qkv_mixed.iter().enumerate() {
        let base = channel * history_len;
        conv[base] = conv[base + 1];
        conv[base + 1] = conv[base + 2];
        conv[base + 2] = *new_input;
    }
    conv_out.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

/// One host-reference Gated-DeltaNet autoregressive decode step, using the
/// equations pinned from llama.cpp `cc83d7b48` (see
/// `ai/research/qwen35-forward-semantics-llama-cpp-2026-09-02.md`).
///
/// `matrix` is all V-head states `[heads][head_dim][head_dim]`; `conv` is the
/// per-channel history `[channels][d_conv-1]` in oldest-first order. Returns
/// the layer output `[5120]`. Names mirror the pinned equations.
// The argument list mirrors the layer's weight/state set; a struct would
// obscure the reference equations under test.
#[allow(clippy::too_many_arguments)]
fn host_gdn_ar_step(
    hidden: &[f32],
    attn_qkv: &[f32],
    attn_gate: &[f32],
    ssm_beta: &[f32],
    ssm_alpha: &[f32],
    ssm_dt_bias: &[f32],
    ssm_a: &[f32],
    ssm_conv1d: &[f32],
    ssm_norm: &[f32],
    ssm_out: &[f32],
    matrix: &mut [f32],
    conv: &mut [f32],
    eps: f32,
) -> Vec<f32> {
    const N_EMBD: usize = 5120;
    assert_eq!(hidden.len(), N_EMBD);

    let qkv_mixed = gguf_gemv(attn_qkv, N_EMBD, hidden);
    let z_gate = gguf_gemv(attn_gate, N_EMBD, hidden);
    let beta: Vec<f32> = gguf_gemv(ssm_beta, N_EMBD, hidden)
        .iter()
        .map(|raw| 1.0 / (1.0 + (-raw).exp()))
        .collect();
    let gate: Vec<f32> = gguf_gemv(ssm_alpha, N_EMBD, hidden)
        .iter()
        .zip(ssm_dt_bias)
        .zip(ssm_a)
        .map(|((raw, dt), a_precomputed)| softplus(raw + dt) * a_precomputed)
        .collect();

    let conv_act = gdn_conv_step(&qkv_mixed, ssm_conv1d, conv);

    // Tile normalized k-heads across v-heads (tiled GGUF V ordering).
    let mut q48 = vec![0.0_f32; GDN_V_HEADS * GDN_HEAD_DIM];
    let mut k48 = vec![0.0_f32; GDN_V_HEADS * GDN_HEAD_DIM];
    for head in 0..GDN_K_HEADS {
        let qh = l2_normalize(
            &conv_act[head * GDN_HEAD_DIM..(head + 1) * GDN_HEAD_DIM],
            eps,
        );
        let kh = l2_normalize(
            &conv_act[GDN_K_OFFSET + head * GDN_HEAD_DIM..GDN_K_OFFSET + (head + 1) * GDN_HEAD_DIM],
            eps,
        );
        for rep in 0..(GDN_V_HEADS / GDN_K_HEADS) {
            let dst = (rep * GDN_K_HEADS + head) * GDN_HEAD_DIM;
            q48[dst..dst + GDN_HEAD_DIM].copy_from_slice(&qh);
            k48[dst..dst + GDN_HEAD_DIM].copy_from_slice(&kh);
        }
    }

    let head_dim = GDN_HEAD_DIM;
    let mut o = vec![0.0_f32; GDN_INNER];
    for head in 0..GDN_V_HEADS {
        let state = &mut matrix[head * head_dim * head_dim..(head + 1) * head_dim * head_dim];
        let k = &k48[head * head_dim..(head + 1) * head_dim];
        let v = &conv_act[GDN_V_OFFSET + head * head_dim..GDN_V_OFFSET + (head + 1) * head_dim];
        let q = &q48[head * head_dim..(head + 1) * head_dim];
        let out_head = gdn_state_step(state, k, v, q, gate[head].exp(), beta[head]);
        o[head * head_dim..(head + 1) * head_dim].copy_from_slice(&out_head);
    }

    // Gated norm: `rms(o) * ssm_norm * silu(z)`, per head; ssm_norm is NOT +1'd.
    let head_dim_f = f32::from(GDN_HEAD_DIM_U16);
    let mut gated = vec![0.0_f32; GDN_INNER];
    for head in 0..GDN_V_HEADS {
        let o_head = &o[head * head_dim..(head + 1) * head_dim];
        let sum: f32 = o_head.iter().map(|x| x * x).sum();
        let inv = (sum / head_dim_f + eps).sqrt().recip();
        for i in 0..head_dim {
            let zv = z_gate[head * head_dim + i];
            let silu = zv / (1.0 + (-zv).exp());
            gated[head * head_dim + i] = o_head[i] * inv * ssm_norm[i] * silu;
        }
    }

    gguf_gemv(ssm_out, GDN_INNER, &gated)
}

/// Fixed pseudo-hidden-state from an xorshift stream in [-1, 1).
fn deterministic_hidden_state() -> Vec<f32> {
    let mut hidden = vec![0.0_f32; 5120];
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    for value in &mut hidden {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let mantissa = u32::try_from(seed >> 40).expect("shifted seed fits u32");
        *value = f32::from_bits(mantissa & 0x007F_FFFF) / 16_777_216.0 - 1.0;
    }
    hidden
}

#[test]
#[ignore = "requires the pinned Qwen GGUF"]
fn host_reference_gdn_ar_step_is_deterministic_and_self_consistent() {
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");
    let prefix = "blk.0.";
    let names = [
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_beta.weight",
        "ssm_alpha.weight",
        "ssm_dt.bias",
        "ssm_a",
        "ssm_conv1d.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
    ];
    let tensors: Vec<Vec<f32>> = names
        .iter()
        .map(|suffix| pinned_tensor_f32(&provider, &format!("{prefix}{suffix}")))
        .collect();
    let (attn_qkv, attn_gate, ssm_beta, ssm_alpha) =
        (&tensors[0], &tensors[1], &tensors[2], &tensors[3]);
    let (ssm_dt_bias, ssm_a, ssm_conv1d, ssm_norm, ssm_out) = (
        &tensors[4],
        &tensors[5],
        &tensors[6],
        &tensors[7],
        &tensors[8],
    );

    // Deterministic pseudo-hidden-state (xorshift), fixed across runs.
    let hidden = deterministic_hidden_state();

    let initial = |scale: f32| vec![scale; GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM];
    let conv_initial = |scale: f32| vec![scale; GDN_QKV_DIM * (GDN_D_CONV - 1)];

    // Two independent runs from identical initial state must agree bit-for-bit
    // and produce finite output.
    let mut matrix_a = initial(0.25);
    let mut conv_a = conv_initial(0.125);
    let mut matrix_b = initial(0.25);
    let mut conv_b = conv_initial(0.125);
    let out_a = host_gdn_ar_step(
        &hidden,
        attn_qkv,
        attn_gate,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm,
        ssm_out,
        &mut matrix_a,
        &mut conv_a,
        1.0e-6,
    );
    let out_b = host_gdn_ar_step(
        &hidden,
        attn_qkv,
        attn_gate,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm,
        ssm_out,
        &mut matrix_b,
        &mut conv_b,
        1.0e-6,
    );
    assert_eq!(out_a.len(), 5120);
    assert!(out_a.iter().all(|v| v.is_finite()));
    assert!(
        out_a
            .iter()
            .zip(&out_b)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        matrix_a
            .iter()
            .zip(&matrix_b)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
#[ignore = "requires the pinned Qwen GGUF"]
fn host_reference_gdn_ar_step_evolves_and_depends_on_state() {
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");
    let prefix = "blk.0.";
    let names = [
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_beta.weight",
        "ssm_alpha.weight",
        "ssm_dt.bias",
        "ssm_a",
        "ssm_conv1d.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
    ];
    let tensors: Vec<Vec<f32>> = names
        .iter()
        .map(|suffix| pinned_tensor_f32(&provider, &format!("{prefix}{suffix}")))
        .collect();
    let (attn_qkv, attn_gate, ssm_beta, ssm_alpha) =
        (&tensors[0], &tensors[1], &tensors[2], &tensors[3]);
    let (ssm_dt_bias, ssm_a, ssm_conv1d, ssm_norm, ssm_out) = (
        &tensors[4],
        &tensors[5],
        &tensors[6],
        &tensors[7],
        &tensors[8],
    );
    let hidden = deterministic_hidden_state();
    let initial = |scale: f32| vec![scale; GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM];
    let conv_initial = |scale: f32| vec![scale; GDN_QKV_DIM * (GDN_D_CONV - 1)];

    let mut matrix = initial(0.25);
    let mut conv = conv_initial(0.125);
    let _first = host_gdn_ar_step(
        &hidden,
        attn_qkv,
        attn_gate,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm,
        ssm_out,
        &mut matrix,
        &mut conv,
        1.0e-6,
    );

    // State must actually change (decay + outer-product update applied).
    assert!(
        matrix
            .iter()
            .zip(initial(0.25))
            .any(|(a, init)| (a - init).abs() > 1.0e-6)
    );
    assert!(
        conv.iter()
            .zip(conv_initial(0.125))
            .any(|(c, init)| (c - init).abs() > 1.0e-6)
    );

    // A second step from the evolved state must differ from a fresh state at
    // the same input (state dependence).
    let mut matrix_fresh = initial(0.25);
    let mut conv_fresh = conv_initial(0.125);
    let out_second = host_gdn_ar_step(
        &hidden,
        attn_qkv,
        attn_gate,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm,
        ssm_out,
        &mut matrix,
        &mut conv,
        1.0e-6,
    );
    let out_fresh = host_gdn_ar_step(
        &hidden,
        attn_qkv,
        attn_gate,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm,
        ssm_out,
        &mut matrix_fresh,
        &mut conv_fresh,
        1.0e-6,
    );
    assert!(
        out_second
            .iter()
            .zip(&out_fresh)
            .any(|(a, b)| (a - b).abs() > 1.0e-4)
    );
}
