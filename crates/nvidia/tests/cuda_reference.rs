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
    ATTN_HEAD_DIM, ATTN_Q_HEADS, AttnLayerWeights, CudaHybridState, CudaIq3SEmbedding,
    CudaIq3SGemv, CudaIq4NlGemv, CudaIq4XsGemv, CudaQ3KGemv, CudaQ4KEmbedding, CudaQ4KGemv,
    CudaQ5KGemv, CudaQ6KGemv, CudaQ8_0Gemv, CudaQwen35Ops, CudaQwen35Weights,
    CudaReferenceDispatcher, CudaStateBuffer, CudaStateError, CudaWeightStagingError,
    CudaWeightStore, FfnLayerWeights, GDN_D_CONV, GDN_HEAD_DIM, GDN_QKV_DIM, GDN_V_HEADS,
    GdnLayerWeights, N_EMBD, StagedTensorSource, host_ffn_step, host_full_attn_ar_step_traced,
    host_gdn_ar_step, host_gdn_ar_step_traced,
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
fn executes_l2_norm_against_host_equations() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    // l2_norm: eps floor on the norm, matching ggml l2_norm_f32.
    let l2_input = vec![3.0_f32, 4.0, 0.0, -12.0];
    let l2_device = stream.clone_htod(&l2_input).expect("upload l2 input");
    let mut l2_out = stream
        .alloc_zeros::<f32>(l2_input.len())
        .expect("allocate l2 output");
    ops.l2_norm(&l2_device, &mut l2_out, 1e-5)
        .expect("execute l2 normalization");
    let actual = stream.clone_dtoh(&l2_out).expect("download l2 output");
    let sum_squares = l2_input.iter().map(|value| value * value).sum::<f32>();
    let scale = 1.0_f32 / sum_squares.sqrt().max(1e-5_f32);
    for (actual, input) in actual.iter().zip(&l2_input) {
        let expected = input * scale;
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    // eps floor: a zero vector must normalize to zero, not NaN.
    let zeros = vec![0.0_f32; 128];
    let zeros_device = stream.clone_htod(&zeros).expect("upload zero l2 input");
    let mut zeros_out = stream
        .alloc_zeros::<f32>(zeros.len())
        .expect("allocate zero l2 output");
    ops.l2_norm(&zeros_device, &mut zeros_out, 1e-5)
        .expect("execute l2 normalization on zeros");
    let actual = stream
        .clone_dtoh(&zeros_out)
        .expect("download zero l2 output");
    assert!(actual.iter().all(|value| value.abs() < 1e-7));
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_gdn_scalar_gate_against_host_equations() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    // gdn_scalar_gate: decay/beta equations against the host reference.
    // alpha/beta_raw come from the ssm_alpha/ssm_beta projections; a is
    // ssm_a = -exp(A_log); dt_bias does not touch beta.
    let heads = 16;
    let alpha: Vec<f32> = (0..heads)
        .map(|index| -0.5 + f32::from(u16::try_from(index).expect("head index fits")) * 0.07)
        .collect();
    let beta_raw: Vec<f32> = (0..heads)
        .map(|index| f32::from(u16::try_from(index).expect("head index fits")) * 0.11 - 0.8)
        .collect();
    let dt_bias: Vec<f32> = (0..heads)
        .map(|index| -0.3 + f32::from(u16::try_from(index).expect("head index fits")) * 0.05)
        .collect();
    let a: Vec<f32> = (0..heads)
        .map(|index| -1.0 - f32::from(u16::try_from(index).expect("head index fits")) * 0.1)
        .collect();
    let alpha_device = stream.clone_htod(&alpha).expect("upload alpha");
    let beta_raw_device = stream.clone_htod(&beta_raw).expect("upload beta raw");
    let dt_bias_device = stream.clone_htod(&dt_bias).expect("upload dt bias");
    let a_device = stream.clone_htod(&a).expect("upload a");
    let mut decay = stream.alloc_zeros::<f32>(heads).expect("allocate decay");
    let mut beta = stream.alloc_zeros::<f32>(heads).expect("allocate beta");
    ops.gdn_scalar_gate(
        &alpha_device,
        &beta_raw_device,
        &dt_bias_device,
        &a_device,
        &mut decay,
        &mut beta,
    )
    .expect("execute GDN scalar gate");
    let decay_actual = stream.clone_dtoh(&decay).expect("download decay");
    let beta_actual = stream.clone_dtoh(&beta).expect("download beta");
    for head in 0..heads {
        let softplus_arg = alpha[head] + dt_bias[head];
        let softplus_value = if softplus_arg > 20.0 {
            softplus_arg
        } else {
            (1.0 + softplus_arg.exp()).ln()
        };
        let expected_decay = (a[head] * softplus_value).exp();
        let expected_beta = 1.0 / (1.0 + (-beta_raw[head]).exp());
        assert!(
            (decay_actual[head] - expected_decay).abs() < 1e-5,
            "decay {}: {} != {}",
            head,
            decay_actual[head],
            expected_decay
        );
        assert!((beta_actual[head] - expected_beta).abs() < 1e-5);
    }
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
    let weights = GdnLayerWeights {
        attn_qkv: attn_qkv.clone(),
        attn_gate: attn_gate.clone(),
        ssm_beta: ssm_beta.clone(),
        ssm_alpha: ssm_alpha.clone(),
        ssm_dt_bias: ssm_dt_bias.clone(),
        ssm_a: ssm_a.clone(),
        ssm_conv1d: ssm_conv1d.clone(),
        ssm_norm: ssm_norm.clone(),
        ssm_out: ssm_out.clone(),
    };
    let out_a = host_gdn_ar_step(&weights, &hidden, &mut matrix_a, &mut conv_a, 1.0e-6);
    let out_b = host_gdn_ar_step(&weights, &hidden, &mut matrix_b, &mut conv_b, 1.0e-6);
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
    let weights = GdnLayerWeights {
        attn_qkv: attn_qkv.clone(),
        attn_gate: attn_gate.clone(),
        ssm_beta: ssm_beta.clone(),
        ssm_alpha: ssm_alpha.clone(),
        ssm_dt_bias: ssm_dt_bias.clone(),
        ssm_a: ssm_a.clone(),
        ssm_conv1d: ssm_conv1d.clone(),
        ssm_norm: ssm_norm.clone(),
        ssm_out: ssm_out.clone(),
    };
    let hidden = deterministic_hidden_state();
    let initial = |scale: f32| vec![scale; GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM];
    let conv_initial = |scale: f32| vec![scale; GDN_QKV_DIM * (GDN_D_CONV - 1)];

    let mut matrix = initial(0.25);
    let mut conv = conv_initial(0.125);
    let _first = host_gdn_ar_step(&weights, &hidden, &mut matrix, &mut conv, 1.0e-6);

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
    let out_second = host_gdn_ar_step(&weights, &hidden, &mut matrix, &mut conv, 1.0e-6);
    let out_fresh = host_gdn_ar_step(
        &weights,
        &hidden,
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

/// Golden layer-0 GDN sums from genuine llama.cpp build 10684 inference on
/// the single-token prompt "Hi" (token 12675, zero state), captured by
/// `scripts/llama-gdn-capture.sh` on 2026-09-02. Keyed by ggml tensor name.
/// llama.cpp quantizes activations to Q8 inside `mul_mat`, so Engine's F32
/// reference differs by bounded activation-quantization noise, not
/// semantics; structural mistakes (orientation/scale/tiling) diverge O(1).
const LLAMA_GDN_CAPTURE_SUMS: &[(&str, &str)] = &[
    ("model.input_embed", "-1.055925"),
    ("attn_norm-0", "-65.716560"),
    ("conv_states-0", "0.000000"),
    ("linear_attn_qkv_mixed-0", "214.836609"),
    ("conv_output_raw-0", "361.088226"),
    ("conv_output_silu-0", "349.484833"),
    ("q_conv_predelta-0", "22.867672"),
    ("k_conv_predelta-0", "17.095335"),
    ("v_conv_predelta-0", "312.238159"),
    ("gate-0", "-29.768518"),
    ("beta_sigmoid-0", "35.756680"),
    ("attn_output-0", "4.304079"),
    ("z-0", "46.594196"),
    ("final_output-0", "7.890058"),
    ("linear_attn_out-0", "9.584183"),
];

/// Parsed golden sums (values are printed with 6 decimals by `llama-debug`;
/// parsing keeps the transcribed text verbatim and comparable at runtime).
fn llama_capture_sum(name: &str) -> Option<f64> {
    capture_sum(LLAMA_GDN_CAPTURE_SUMS, name)
}

/// Relative tolerance for sum comparisons against llama.cpp Q8-activation
/// matmuls: 1% absorbs rounding of the printed 6-decimal sums and Q8
/// activation quantization noise on large tensors.
fn sum_close(actual: f64, expected: f64, elements: usize) -> bool {
    // Elementwise ops match tightly. Matmuls quantize activations to Q8 inside
    // ggml, so the F32 reference carries bounded noise roughly proportional to
    // the element count; a near-cancelling sum with large L1 amplifies it. The
    // bound therefore scales with element count, capped by a 1% relative term.
    let relative = (actual - expected).abs() / actual.abs().max(expected.abs()).max(1.0);
    let absolute = f64::from(u32::try_from(elements).expect("elements fit u32")) * 0.004;
    relative < 0.01 || (actual - expected).abs() < absolute
}

fn pinned_embedding_row(provider: &Qwen35ModelProvider, token: usize) -> Vec<f32> {
    let mut reader = provider
        .open_tensor("token_embd.weight")
        .expect("open token embedding");
    let mut row = Vec::new();
    let mut block = Vec::new();
    let mut emitted = 0_usize;
    while emitted < (token + 1) * N_EMBD {
        let Some(next) = reader
            .read_dequantized_block()
            .expect("decode token embedding block")
        else {
            break;
        };
        block.extend(next);
        while block.len() >= N_EMBD {
            let take = N_EMBD.min(block.len());
            if emitted >= token * N_EMBD {
                row.extend_from_slice(&block[..take]);
            }
            emitted += take;
            block.drain(..take);
        }
    }
    assert_eq!(row.len(), N_EMBD, "embedding row gather");
    row
}

fn pinned_norm_weights(provider: &Qwen35ModelProvider, name: &str) -> Vec<f32> {
    pinned_tensor_f32(provider, name)
}

#[test]
#[ignore = "requires the pinned Qwen GGUF"]
fn host_reference_gdn_step_matches_llama_debug_capture() {
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");
    let hi_token = 12_675_usize;

    // Embedding gather for the single prompt token.
    let embed = pinned_embedding_row(&provider, hi_token);
    let embed_sum: f64 = embed.iter().map(|v| f64::from(*v)).sum();

    // attn_norm: RMSNorm over the embedding with raw (already +1'd) weights.
    let attn_norm_w = pinned_norm_weights(&provider, "blk.0.attn_norm.weight");
    let attn_norm = host_rms_norm(&embed, &attn_norm_w, 1.0e-6);
    let attn_norm_sum: f64 = attn_norm.iter().map(|v| f64::from(*v)).sum();

    // Full traced GDN step from zero state.
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
    let weights = GdnLayerWeights {
        attn_qkv: tensors[0].clone(),
        attn_gate: tensors[1].clone(),
        ssm_beta: tensors[2].clone(),
        ssm_alpha: tensors[3].clone(),
        ssm_dt_bias: tensors[4].clone(),
        ssm_a: tensors[5].clone(),
        ssm_conv1d: tensors[6].clone(),
        ssm_norm: tensors[7].clone(),
        ssm_out: tensors[8].clone(),
    };
    let mut matrix = vec![0.0_f32; GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM];
    let mut conv = vec![0.0_f32; GDN_QKV_DIM * (GDN_D_CONV - 1)];
    let trace = host_gdn_ar_step_traced(&weights, &attn_norm, &mut matrix, &mut conv, 1.0e-6);

    let sum = |values: &[f32]| -> f64 { values.iter().map(|v| f64::from(*v)).sum() };
    // llama.cpp's q/k tensors are pre-tile [128, 16]; the engine trace holds
    // tiled 48-head copies (3 identical repetitions), so compare the first 16
    // heads only.
    let checks: Vec<(&str, f64, usize)> = vec![
        ("model.input_embed", embed_sum, embed.len()),
        ("attn_norm-0", attn_norm_sum, attn_norm.len()),
        (
            "linear_attn_qkv_mixed-0",
            sum(&trace.qkv_mixed),
            trace.qkv_mixed.len(),
        ),
        (
            "conv_output_raw-0",
            sum(&trace.conv_raw),
            trace.conv_raw.len(),
        ),
        (
            "conv_output_silu-0",
            sum(&trace.conv_act),
            trace.conv_act.len(),
        ),
        ("q_conv_predelta-0", sum(&trace.q_tiled[..2048]), 2048),
        ("k_conv_predelta-0", sum(&trace.k_tiled[..2048]), 2048),
        ("v_conv_predelta-0", sum(&trace.conv_act[4096..]), 6144),
        ("gate-0", sum(&trace.gate), trace.gate.len()),
        ("beta_sigmoid-0", sum(&trace.beta), trace.beta.len()),
        ("attn_output-0", sum(&trace.attn_out), trace.attn_out.len()),
        ("z-0", sum(&trace.z_gate), trace.z_gate.len()),
        ("final_output-0", sum(&trace.gated), trace.gated.len()),
        ("linear_attn_out-0", sum(&trace.out), trace.out.len()),
    ];
    let mut failures = Vec::new();
    for (name, actual, elements) in checks {
        let Some(expected) = llama_capture_sum(name) else {
            continue;
        };
        if !sum_close(actual, expected, elements) {
            failures.push(format!(
                "{name}: engine sum {actual:.4} vs llama.cpp {expected:.4}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "GDN parity failures:\n  {}",
        failures.join("\n  ")
    );
}

/// Golden layer-3 full-attention sums from genuine llama.cpp build 10684
/// inference, captured by `scripts/llama-attn-capture.sh` on 2026-09-02
/// (1-token run: prompt "Hi" = token 12675, position 0, rope identity).
const LLAMA_ATTN_CAPTURE_1TOK: &[(&str, &str)] = &[
    ("model.input_embed", "-1.055925"),
    ("attn_norm-0", "-65.716560"),
    ("l_out-2", "6.607050"),
    ("attn_norm-3", "9.732841"),
    ("Qcur_full-3", "-27808.511719"),
    ("Qcur_normed-3", "122.420403"),
    ("Qcur-3", "122.420403"),
    ("Vcur-3", "-4.807240"),
    ("Kcur-3-raw", "22.731773"),
    ("Kcur_normed-3", "12.335545"),
    ("Kcur-3", "12.335545"),
    ("attn_pregate-3", "-28.811741"),
    ("gate_reshaped-3", "-27890.394531"),
    ("gate_sigmoid-3", "94.876389"),
    ("attn_gated-3", "-2.508305"),
    ("attn_output-3", "6.240145"),
    ("attn_residual-3", "12.847154"),
    ("attn_post_norm-3", "-34.353695"),
    ("ffn_out-3", "-0.134532"),
    ("l_out-3", "12.712581"),
];

/// Golden layer-3 sums from the 2-token run (prompt "Hi there" = tokens
/// [12675, 1017]): a genuine prefill batch — GDN layers take the chunked
/// path, attention runs over a 2-entry causal KV cache, and rope rotates at
/// position 1. Tensor sums cover both tokens.
const LLAMA_ATTN_CAPTURE_2TOK: &[(&str, &str)] = &[
    ("model.input_embed", "-1.024361"),
    ("attn_norm-0", "-63.792130"),
    ("l_out-2", "22.629915"),
    ("attn_norm-3", "53.849220"),
    ("Qcur_full-3", "-56920.371094"),
    ("Qcur_normed-3", "230.419525"),
    ("Qcur-3", "243.807587"),
    ("Vcur-3", "38.204910"),
    ("Kcur-3-raw", "35.833229"),
    ("Kcur_normed-3", "13.997391"),
    ("Kcur-3", "9.906808"),
    ("attn_pregate-3", "-27.824806"),
    ("gate_reshaped-3", "-57094.621094"),
    ("gate_sigmoid-3", "193.474472"),
    ("attn_gated-3", "-2.569808"),
    ("attn_output-3", "19.881910"),
    ("attn_residual-3", "42.511784"),
    ("attn_post_norm-3", "-45.576557"),
    ("ffn_out-3", "0.789787"),
    ("l_out-3", "43.301579"),
];

fn capture_sum(table: &[(&str, &str)], name: &str) -> Option<f64> {
    table
        .iter()
        .find(|(capture_name, _)| *capture_name == name)
        .and_then(|(_, text)| text.parse::<f64>().ok())
}

/// Deterministic pseudo-random f32 fixtures in 0.125 steps within [-4, 4):
/// every value is exactly representable in F16, so F16 cache rounding is
/// lossless and host replays can compare against the F32 equations.
fn fixture_quantized_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let steps = u16::try_from((*state >> 16) & 0x3f).expect("steps fit u16");
    f32::from(steps) * 0.125 - 4.0
}

/// Exact F16 bit patterns for `fixture_quantized_f32` values without an
/// f16 dependency: sign, bias-15 exponent, 10-bit fraction. The 2^-3 step
/// size keeps at least 7 trailing zero fraction bits, so the conversion is
/// exact.
fn fixture_f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let sign = u16::from(value.is_sign_negative()) << 15;
    let magnitude = value.abs();
    // magnitude is a multiple of 0.125 in [0, 4); normalize to [1, 2).
    let mut exponent = 0_i32;
    let mut mantissa = magnitude;
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    // mantissa in [1, 2): 10-bit fraction = (mantissa - 1) * 2^10.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "fixtures are exact multiples of 2^-3"
    )]
    let fraction = ((mantissa - 1.0) * 1024.0) as i32;
    sign | u16::try_from((exponent + 15) * 1024 + fraction).expect("f16 bits fit u16")
}

use engine_nvidia::rms_norm_raw as host_rms_norm;

fn load_gdn_layer(provider: &Qwen35ModelProvider, layer: usize) -> GdnLayerWeights {
    let prefix = format!("blk.{layer}.");
    GdnLayerWeights {
        attn_qkv: pinned_tensor_f32(provider, &format!("{prefix}attn_qkv.weight")),
        attn_gate: pinned_tensor_f32(provider, &format!("{prefix}attn_gate.weight")),
        ssm_beta: pinned_tensor_f32(provider, &format!("{prefix}ssm_beta.weight")),
        ssm_alpha: pinned_tensor_f32(provider, &format!("{prefix}ssm_alpha.weight")),
        ssm_dt_bias: pinned_tensor_f32(provider, &format!("{prefix}ssm_dt.bias")),
        ssm_a: pinned_tensor_f32(provider, &format!("{prefix}ssm_a")),
        ssm_conv1d: pinned_tensor_f32(provider, &format!("{prefix}ssm_conv1d.weight")),
        ssm_norm: pinned_tensor_f32(provider, &format!("{prefix}ssm_norm.weight")),
        ssm_out: pinned_tensor_f32(provider, &format!("{prefix}ssm_out.weight")),
    }
}

fn load_attn_layer(provider: &Qwen35ModelProvider, layer: usize) -> AttnLayerWeights {
    let prefix = format!("blk.{layer}.");
    AttnLayerWeights {
        attn_q: pinned_tensor_f32(provider, &format!("{prefix}attn_q.weight")),
        attn_k: pinned_tensor_f32(provider, &format!("{prefix}attn_k.weight")),
        attn_v: pinned_tensor_f32(provider, &format!("{prefix}attn_v.weight")),
        attn_q_norm: pinned_tensor_f32(provider, &format!("{prefix}attn_q_norm.weight")),
        attn_k_norm: pinned_tensor_f32(provider, &format!("{prefix}attn_k_norm.weight")),
        attn_output: pinned_tensor_f32(provider, &format!("{prefix}attn_output.weight")),
    }
}

fn load_ffn_layer(provider: &Qwen35ModelProvider, layer: usize) -> FfnLayerWeights {
    let prefix = format!("blk.{layer}.");
    FfnLayerWeights {
        ffn_gate: pinned_tensor_f32(provider, &format!("{prefix}ffn_gate.weight")),
        ffn_up: pinned_tensor_f32(provider, &format!("{prefix}ffn_up.weight")),
        ffn_down: pinned_tensor_f32(provider, &format!("{prefix}ffn_down.weight")),
    }
}

/// The `gate_reshaped` slice of `Qcur_full`: the gate half of each per-head
/// `[q(256) | gate(256)]` block.
fn gate_half(q_gate: &[f32]) -> Vec<f32> {
    let mut gate = Vec::with_capacity(ATTN_Q_HEADS * ATTN_HEAD_DIM);
    for head in 0..ATTN_Q_HEADS {
        let base = head * 2 * ATTN_HEAD_DIM;
        gate.extend_from_slice(&q_gate[base + ATTN_HEAD_DIM..base + 2 * ATTN_HEAD_DIM]);
    }
    gate
}

#[test]
#[ignore = "requires the pinned Qwen GGUF"]
// One layer-major replay through four layers; splitting it would hide the
// cross-layer state evolution that this gate exists to validate.
#[allow(clippy::too_many_lines)]
fn host_reference_full_attn_matches_llama_debug_capture() {
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");
    let eps = 1.0e-6_f32;
    let tokens = [12_675_usize, 1017];

    // Embedding rows for both prompt tokens.
    let embeds = tokens
        .iter()
        .map(|token| pinned_embedding_row(&provider, *token))
        .collect::<Vec<_>>();

    // Layer-major replay: process both tokens per layer, freeing each layer's
    // weights before loading the next. State (GDN matrix/conv, attention KV)
    // is shared across the two tokens and evolves sequentially.
    let mut x_cur = embeds.clone();
    let mut attn_norm_sums = [0.0_f64; 2];
    let mut layer3_norm_sums = [0.0_f64; 2];
    let mut l_out2_sums = [0.0_f64; 2];
    let mut traces = Vec::new();
    let mut post_norm3_sums = [0.0_f64; 2];
    let mut ffn_out3_sums = [0.0_f64; 2];
    let mut l_out3_sums = [0.0_f64; 2];

    let mut kv_keys = Vec::new();
    let mut kv_values = Vec::new();

    for layer in 0..=3_usize {
        let attn_norm_w = pinned_tensor_f32(&provider, &format!("blk.{layer}.attn_norm.weight"));
        let post_norm_w = pinned_tensor_f32(
            &provider,
            &format!("blk.{layer}.post_attention_norm.weight"),
        );
        let ffn_w = load_ffn_layer(&provider, layer);

        if layer < 3 {
            // Gated-DeltaNet layer with FFN/residual wrapper. Each recurrent
            // layer owns its state; a fresh context starts every layer at zero.
            let gdn_w = load_gdn_layer(&provider, layer);
            let mut gdn_matrix = vec![0.0_f32; GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM];
            let mut gdn_conv = vec![0.0_f32; GDN_QKV_DIM * (GDN_D_CONV - 1)];
            for (token_index, x) in x_cur.iter_mut().enumerate() {
                let normalized = host_rms_norm(x, &attn_norm_w, eps);
                if layer == 0 {
                    attn_norm_sums[token_index] = normalized.iter().map(|v| f64::from(*v)).sum();
                }
                let attn_out =
                    host_gdn_ar_step(&gdn_w, &normalized, &mut gdn_matrix, &mut gdn_conv, eps);
                for (x_elem, out_elem) in x.iter_mut().zip(&attn_out) {
                    *x_elem += out_elem;
                }
                let post = host_rms_norm(x, &post_norm_w, eps);
                let ffn_out = host_ffn_step(&ffn_w, &post);
                for (x_elem, out_elem) in x.iter_mut().zip(&ffn_out) {
                    *x_elem += out_elem;
                }
            }
        } else {
            // Full-attention layer 3: traced, with the residual/FFN wrapper.
            let attn_w = load_attn_layer(&provider, layer);
            for (token_index, x) in x_cur.iter_mut().enumerate() {
                let normalized = host_rms_norm(x, &attn_norm_w, eps);
                layer3_norm_sums[token_index] = normalized.iter().map(|v| f64::from(*v)).sum();
                let trace = host_full_attn_ar_step_traced(
                    &attn_w,
                    &normalized,
                    &mut kv_keys,
                    &mut kv_values,
                    token_index,
                    eps,
                );
                for (x_elem, out_elem) in x.iter_mut().zip(&trace.out) {
                    *x_elem += out_elem;
                }
                let post = host_rms_norm(x, &post_norm_w, eps);
                let ffn_out = host_ffn_step(&ffn_w, &post);
                post_norm3_sums[token_index] = post.iter().map(|v| f64::from(*v)).sum();
                ffn_out3_sums[token_index] = ffn_out.iter().map(|v| f64::from(*v)).sum();
                for (x_elem, out_elem) in x.iter_mut().zip(&ffn_out) {
                    *x_elem += out_elem;
                }
                l_out3_sums[token_index] = x.iter().map(|v| f64::from(*v)).sum();
                traces.push(trace);
            }
        }

        if layer == 2 {
            for (token_index, x) in x_cur.iter().enumerate() {
                l_out2_sums[token_index] = x.iter().map(|v| f64::from(*v)).sum();
            }
        }
    }

    let sum = |values: &[f32]| -> f64 { values.iter().map(|v| f64::from(*v)).sum() };
    let trace_t0 = &traces[0];
    let trace_t1 = &traces[1];

    // Token-0-only values (rope identity at position 0).
    let first_token: Vec<(&str, f64, usize)> = vec![
        ("model.input_embed", sum(&embeds[0]), N_EMBD),
        ("attn_norm-0", attn_norm_sums[0], N_EMBD),
        ("l_out-2", l_out2_sums[0], N_EMBD),
        ("Qcur_full-3", sum(&trace_t0.q_gate), 12_288),
        ("Qcur_normed-3", sum(&trace_t0.q_normed), 6144),
        ("Qcur-3", sum(&trace_t0.q_rope), 6144),
        ("Vcur-3", sum(&trace_t0.v), 1024),
        ("Kcur-3-raw", sum(&trace_t0.k_raw), 1024),
        ("Kcur_normed-3", sum(&trace_t0.k_normed), 1024),
        ("Kcur-3", sum(&trace_t0.k_rope), 1024),
        ("attn_pregate-3", sum(&trace_t0.attn_pregate), 6144),
        ("gate_reshaped-3", sum(&gate_half(&trace_t0.q_gate)), 6144),
        ("gate_sigmoid-3", sum(&trace_t0.gate_sigmoid), 6144),
        ("attn_gated-3", sum(&trace_t0.attn_gated), 6144),
        ("attn_output-3", sum(&trace_t0.out), N_EMBD),
        (
            "attn_residual-3",
            l_out2_sums[0] + sum(&trace_t0.out),
            N_EMBD,
        ),
        ("attn_post_norm-3", post_norm3_sums[0], N_EMBD),
        ("ffn_out-3", ffn_out3_sums[0], N_EMBD),
        ("l_out-3", l_out3_sums[0], N_EMBD),
    ];

    // Both tokens combined (2-token prefill; rope rotates at position 1).
    let combined = |a: &[f32], other: &[f32]| -> f64 { sum(a) + sum(other) };
    let both_tokens: Vec<(&str, f64, usize)> = vec![
        (
            "model.input_embed",
            combined(&embeds[0], &embeds[1]),
            2 * N_EMBD,
        ),
        (
            "attn_norm-0",
            attn_norm_sums[0] + attn_norm_sums[1],
            2 * N_EMBD,
        ),
        ("l_out-2", l_out2_sums[0] + l_out2_sums[1], 2 * N_EMBD),
        (
            "attn_norm-3",
            layer3_norm_sums[0] + layer3_norm_sums[1],
            2 * N_EMBD,
        ),
        (
            "Qcur_full-3",
            combined(&trace_t0.q_gate, &trace_t1.q_gate),
            24_576,
        ),
        (
            "Qcur_normed-3",
            combined(&trace_t0.q_normed, &trace_t1.q_normed),
            12_288,
        ),
        (
            "Qcur-3",
            combined(&trace_t0.q_rope, &trace_t1.q_rope),
            12_288,
        ),
        ("Vcur-3", combined(&trace_t0.v, &trace_t1.v), 2048),
        (
            "Kcur-3-raw",
            combined(&trace_t0.k_raw, &trace_t1.k_raw),
            2048,
        ),
        (
            "Kcur_normed-3",
            combined(&trace_t0.k_normed, &trace_t1.k_normed),
            2048,
        ),
        ("Kcur-3", combined(&trace_t0.k_rope, &trace_t1.k_rope), 2048),
        (
            "attn_pregate-3",
            combined(&trace_t0.attn_pregate, &trace_t1.attn_pregate),
            12_288,
        ),
        (
            "gate_reshaped-3",
            combined(&gate_half(&trace_t0.q_gate), &gate_half(&trace_t1.q_gate)),
            12_288,
        ),
        (
            "gate_sigmoid-3",
            combined(&trace_t0.gate_sigmoid, &trace_t1.gate_sigmoid),
            12_288,
        ),
        (
            "attn_gated-3",
            combined(&trace_t0.attn_gated, &trace_t1.attn_gated),
            12_288,
        ),
        (
            "attn_output-3",
            combined(&trace_t0.out, &trace_t1.out),
            2 * N_EMBD,
        ),
        (
            "attn_residual-3",
            l_out2_sums[0] + sum(&trace_t0.out) + l_out2_sums[1] + sum(&trace_t1.out),
            2 * N_EMBD,
        ),
        (
            "attn_post_norm-3",
            post_norm3_sums[0] + post_norm3_sums[1],
            2 * N_EMBD,
        ),
        ("ffn_out-3", ffn_out3_sums[0] + ffn_out3_sums[1], 2 * N_EMBD),
        ("l_out-3", l_out3_sums[0] + l_out3_sums[1], 2 * N_EMBD),
    ];

    let mut failures = Vec::new();
    for (name, actual, elements) in first_token {
        if let Some(expected) = capture_sum(LLAMA_ATTN_CAPTURE_1TOK, name)
            .filter(|expected| !sum_close(actual, *expected, elements))
        {
            failures.push(format!(
                "{name} (1tok): engine {actual:.4} vs llama.cpp {expected:.4}"
            ));
        }
    }
    for (name, actual, elements) in both_tokens {
        if let Some(expected) = capture_sum(LLAMA_ATTN_CAPTURE_2TOK, name)
            .filter(|expected| !sum_close(actual, *expected, elements))
        {
            failures.push(format!(
                "{name} (2tok): engine {actual:.4} vs llama.cpp {expected:.4}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "attention parity failures:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_rope_sigmoid_against_host_equations() {
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const ROT_DIMS: usize = 64;
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    // rope_neox: NEOX half-split pairs over the first 64 dims of each
    // 256-dim head, matching the pinned reference equations.
    let base = 1.0e7_f32;
    let mut host = vec![0.0_f32; HEADS * HEAD_DIM];
    for (index, value) in host.iter_mut().enumerate() {
        let value_u16 = u16::try_from(index % 97).expect("index fits u16");
        *value = f32::from(value_u16) * 0.03 - 1.4;
    }
    let mut device = stream.clone_htod(&host).expect("upload rope input");
    let position = 3_u64;
    ops.rope_neox(&mut device, position, HEADS, HEAD_DIM, ROT_DIMS, base)
        .expect("execute rope");
    let actual = stream.clone_dtoh(&device).expect("download rope output");
    // Host replay with the pinned equations.
    let mut expected = host.clone();
    for head in 0..HEADS {
        for pair in 0..ROT_DIMS / 2 {
            let exponent = -(2.0 * f32::from(u16::try_from(pair).expect("pair fits u16")))
                / f32::from(u16::try_from(ROT_DIMS).expect("rot dims fit u16"));
            let theta = f32::from(u16::try_from(position).expect("position fits u16"))
                * base.powf(exponent);
            let (sin, cos) = theta.sin_cos();
            let stride = ROT_DIMS / 2;
            let x0 = expected[head * HEAD_DIM + pair];
            let x1 = expected[head * HEAD_DIM + stride + pair];
            expected[head * HEAD_DIM + pair] = x0 * cos - x1 * sin;
            expected[head * HEAD_DIM + stride + pair] = x0 * sin + x1 * cos;
        }
    }
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "rope[{index}]: {actual} != {expected}"
        );
    }

    // sigmoid_inplace: elementwise host replay.
    let mut sigmoid_input = vec![0.0_f32; 300];
    for (index, value) in sigmoid_input.iter_mut().enumerate() {
        let index_f = f32::from(u16::try_from(index).expect("index fits u16"));
        *value = index_f * 0.02 - 3.0;
    }
    let mut sigmoid_device = stream
        .clone_htod(&sigmoid_input)
        .expect("upload sigmoid input");
    ops.sigmoid_inplace(&mut sigmoid_device)
        .expect("execute sigmoid");
    let actual = stream
        .clone_dtoh(&sigmoid_device)
        .expect("download sigmoid output");
    for (index, (actual, expected)) in actual.iter().zip(&sigmoid_input).enumerate() {
        let expected = 1.0 / (1.0 + (-expected).exp());
        assert!((actual - expected).abs() < 1e-5, "sigmoid[{index}]");
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_attn_score_gqa_against_host_equations() {
    const Q_HEADS: usize = 6;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 32;
    const TOKENS: usize = 3;
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    // Small geometry exercising block-mapped GQA: 6 q heads over 2 kv
    // heads, head_dim 32, three cached tokens. Fixtures use 0.125 steps in
    // [-4, 4): exactly representable in F16, so the F16 cache rounding is
    // lossless and the host replay compares against the F32 equations.
    let mut state = 12345_u32;
    let q: Vec<f32> = (0..Q_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let gate_raw: Vec<f32> = (0..Q_HEADS * 2 * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let keys_f32: Vec<f32> = (0..TOKENS * KV_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let values_f32: Vec<f32> = (0..TOKENS * KV_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let keys: Vec<u16> = keys_f32
        .iter()
        .map(|&value| fixture_f16_bits(value))
        .collect();
    let values: Vec<u16> = values_f32
        .iter()
        .map(|&value| fixture_f16_bits(value))
        .collect();

    let q_device = stream.clone_htod(&q).expect("upload q");
    let keys_device = stream.clone_htod(&keys).expect("upload keys");
    let values_device = stream.clone_htod(&values).expect("upload values");
    let gate_device = stream.clone_htod(&gate_raw).expect("upload gate");
    let mut scores = stream
        .alloc_zeros::<f32>(Q_HEADS * TOKENS)
        .expect("allocate scores");
    let mut output = stream
        .alloc_zeros::<f32>(Q_HEADS * HEAD_DIM)
        .expect("allocate output");
    ops.attn_score_gqa(
        &q_device,
        &keys_device,
        &values_device,
        &gate_device,
        &mut scores,
        &mut output,
        TOKENS,
        TOKENS,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
    )
    .expect("execute attention");
    let actual = stream.clone_dtoh(&output).expect("download output");

    // Host replay with the pinned equations, including F16 cache rounding.
    let q_per_kv = Q_HEADS / KV_HEADS;
    let scale = 1.0 / f32::sqrt(f32::from(u16::try_from(HEAD_DIM).expect("dim fits u16")));
    for q_head in 0..Q_HEADS {
        let kv_head = q_head / q_per_kv;
        let q_vec = &q[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
        let mut head_scores = [0.0_f32; TOKENS];
        for (token, score) in head_scores.iter_mut().enumerate() {
            let start = (token * KV_HEADS + kv_head) * HEAD_DIM;
            let key = &keys_f32[start..start + HEAD_DIM];
            *score = q_vec.iter().zip(key).map(|(qv, kv)| qv * kv).sum::<f32>() * scale;
        }
        let max = head_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = head_scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        for dim in 0..HEAD_DIM {
            let mut out = 0.0_f32;
            for (token, exp) in exps.iter().enumerate() {
                let value_start = (token * KV_HEADS + kv_head) * HEAD_DIM;
                let value = &values_f32[value_start..value_start + HEAD_DIM];
                out += (exp / total) * value[dim];
            }
            let gate = gate_raw[q_head * 2 * HEAD_DIM + HEAD_DIM + dim];
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            let expected = out * sigmoid;
            assert!(
                (actual[q_head * HEAD_DIM + dim] - expected).abs() < 1e-3,
                "attn[head {q_head}][dim {dim}]: {} != {expected}",
                actual[q_head * HEAD_DIM + dim]
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_q4_k_embedding_against_the_gguf_decoder() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    // Two tokens x two 256-element blocks: [512, 2] embedding fixture.
    let encoded: Vec<u8> = (0_u8..4).flat_map(q4_k_fixture_block).collect();
    let spec = WeightTensorSpec::new("q4_k.embedding", vec![512, 2], engine_core::DataType::F32)
        .expect("Q4_K embedding fixture spec");
    let mut store = CudaWeightStore::new(stream.clone());
    store
        .materialize_quantized(
            spec,
            12,
            encoded.len() as u64,
            &mut Cursor::new(encoded.clone()),
        )
        .expect("upload Q4_K embedding fixture");
    let weight = store
        .quantized_tensor("q4_k.embedding")
        .expect("Q4_K embedding weight");
    let kernel =
        CudaQ4KEmbedding::from_context(&context, stream.clone()).expect("compile Q4_K embedding");

    for token in [0_u32, 1] {
        let mut output = stream
            .alloc_zeros::<f32>(512)
            .expect("allocate embedding output");
        kernel
            .execute(weight, token, &mut output)
            .expect("execute Q4_K embedding");
        let actual = stream.clone_dtoh(&output).expect("download embedding");
        let expected = (0..2)
            .flat_map(|block_index| {
                let start =
                    (usize::try_from(token).expect("token fits usize") * 2 + block_index) * 144;
                engine_gguf::dequantize_block(12, &encoded[start..start + 144])
                    .expect("decode Q4_K embedding fixture")
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_gdn_conv_silu_against_host_equations() {
    const CHANNELS: usize = 512;
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    let mut state = 777_u32;
    let input: Vec<f32> = (0..CHANNELS)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let weight: Vec<f32> = (0..CHANNELS * 4)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let history: Vec<f32> = (0..CHANNELS * 3)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();

    let input_device = stream.clone_htod(&input).expect("upload conv input");
    let weight_device = stream.clone_htod(&weight).expect("upload conv weight");
    let mut history_device = stream.clone_htod(&history).expect("upload conv history");
    let mut output_device = stream
        .alloc_zeros::<f32>(CHANNELS)
        .expect("allocate conv output");
    ops.gdn_conv_silu(
        &input_device,
        &weight_device,
        &mut history_device,
        &mut output_device,
    )
    .expect("execute conv");
    let actual = stream
        .clone_dtoh(&output_device)
        .expect("download conv output");
    let actual_history = stream
        .clone_dtoh(&history_device)
        .expect("download conv history");

    // Host replay: tap weights sit at tap + channel*4; history is
    // oldest-first per channel.
    for channel in 0..CHANNELS {
        let base = channel * 3;
        let expected = history[base] * weight[channel * 4]
            + history[base + 1] * weight[channel * 4 + 1]
            + history[base + 2] * weight[channel * 4 + 2]
            + input[channel] * weight[channel * 4 + 3];
        let expected = expected / (1.0 + (-expected).exp());
        assert!(
            (actual[channel] - expected).abs() < 1e-4,
            "conv[{channel}]: {} != {expected}",
            actual[channel]
        );
        // Advanced history: drop oldest, append new input.
        assert!((actual_history[base] - history[base + 1]).abs() < 1e-6);
        assert!((actual_history[base + 1] - history[base + 2]).abs() < 1e-6);
        assert!((actual_history[base + 2] - input[channel]).abs() < 1e-6);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_head_norms_against_host_equations() {
    const HEADS: usize = 16;
    const HEAD_DIM: usize = 128;
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    let mut state = 4242_u32;
    let l2_input: Vec<f32> = (0..HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let rms_input: Vec<f32> = (0..HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let norm_weight: Vec<f32> = (0..HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let eps = 1e-5_f32;

    // l2_norm_heads (in place).
    let mut l2_device = stream.clone_htod(&l2_input).expect("upload l2 heads");
    ops.l2_norm_heads(&mut l2_device, HEADS, HEAD_DIM, eps)
        .expect("execute l2 heads");
    let actual = stream.clone_dtoh(&l2_device).expect("download l2 heads");
    for head in 0..HEADS {
        let head_values = &l2_input[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let sum: f32 = head_values.iter().map(|x| x * x).sum();
        let scale = 1.0 / sum.sqrt().max(eps);
        for (index, value) in head_values.iter().enumerate() {
            let expected = value * scale;
            let index = head * HEAD_DIM + index;
            assert!(
                (actual[index] - expected).abs() < 1e-4,
                "l2[{index}]: {} != {expected}",
                actual[index]
            );
        }
    }

    // strided_rms_norm (per-head, shared weights).
    let input_device = stream.clone_htod(&rms_input).expect("upload rms heads");
    let weight_device = stream.clone_htod(&norm_weight).expect("upload rms weight");
    let mut out_device = stream
        .alloc_zeros::<f32>(HEADS * HEAD_DIM)
        .expect("allocate rms output");
    ops.strided_rms_norm(
        &input_device,
        &weight_device,
        &mut out_device,
        HEADS,
        HEAD_DIM,
        1e-6,
    )
    .expect("execute strided rms");
    let actual = stream.clone_dtoh(&out_device).expect("download rms output");
    for head in 0..HEADS {
        let head_values = &rms_input[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let sum: f32 = head_values.iter().map(|x| x * x).sum();
        let count = f32::from(u16::try_from(HEAD_DIM).expect("head dim fits u16"));
        let inv = (sum / count + 1e-6).sqrt().recip();
        for (index, value) in head_values.iter().enumerate() {
            let expected = value * inv * norm_weight[index];
            let index = head * HEAD_DIM + index;
            assert!(
                (actual[index] - expected).abs() < 1e-4,
                "rms[{index}]: {} != {expected}",
                actual[index]
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
#[allow(
    clippy::too_many_lines,
    reason = "one contiguous pinned-geometry replay"
)]
fn executes_gdn_state_update_against_host_equations() {
    // Full pinned geometry: 48 v heads, 16 k heads, 128 head_dim, v at 4096.
    const V_HEADS: usize = 48;
    const K_HEADS: usize = 16;
    const HEAD_DIM: usize = 128;
    const V_OFFSET: usize = 4096;
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    let mut state = 99_u32;
    let mut matrix: Vec<f32> = (0..V_HEADS * HEAD_DIM * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state) * 0.05)
        .collect();
    let conv_activated: Vec<f32> = (0..V_OFFSET + V_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let q_normed: Vec<f32> = (0..K_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let k_normed: Vec<f32> = (0..K_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let decay: Vec<f32> = (0..V_HEADS)
        .map(|index| {
            let index = u16::try_from(index).expect("index fits u16");
            f32::from(index) * 0.01 + 0.5
        })
        .collect();
    let beta: Vec<f32> = (0..V_HEADS)
        .map(|index| {
            let index = u16::try_from(index).expect("index fits u16");
            f32::from(index) * 0.005 + 0.1
        })
        .collect();

    let mut matrix_device = stream.clone_htod(&matrix).expect("upload state matrix");
    let q_device = stream.clone_htod(&q_normed).expect("upload q");
    let k_device = stream.clone_htod(&k_normed).expect("upload k");
    let conv_device = stream.clone_htod(&conv_activated).expect("upload conv");
    let decay_device = stream.clone_htod(&decay).expect("upload decay");
    let beta_device = stream.clone_htod(&beta).expect("upload beta");
    let mut output_device = stream
        .alloc_zeros::<f32>(V_HEADS * HEAD_DIM)
        .expect("allocate output");
    ops.gdn_state_update(
        &mut matrix_device,
        &q_device,
        &k_device,
        &conv_device,
        &decay_device,
        &beta_device,
        &mut output_device,
        V_HEADS,
        K_HEADS,
        HEAD_DIM,
        V_OFFSET,
    )
    .expect("execute state update");
    let actual_out = stream.clone_dtoh(&output_device).expect("download output");
    let actual_matrix = stream.clone_dtoh(&matrix_device).expect("download matrix");

    // Host replay with the pinned equations.
    let scale = 1.0 / f32::sqrt(f32::from(u16::try_from(HEAD_DIM).expect("dim fits u16")));
    for v_head in 0..V_HEADS {
        let k_head = v_head % K_HEADS;
        let state_offset = v_head * HEAD_DIM * HEAD_DIM;
        let q = &q_normed[k_head * HEAD_DIM..(k_head + 1) * HEAD_DIM];
        let k = &k_normed[k_head * HEAD_DIM..(k_head + 1) * HEAD_DIM];
        let v = &conv_activated[V_OFFSET + v_head * HEAD_DIM..V_OFFSET + (v_head + 1) * HEAD_DIM];
        // Decay, sk, d, outer add, then output per column.
        for row in 0..HEAD_DIM {
            for col in 0..HEAD_DIM {
                matrix[state_offset + row * HEAD_DIM + col] *= decay[v_head];
            }
        }
        let mut sk = vec![0.0_f32; HEAD_DIM];
        for row in 0..HEAD_DIM {
            for col in 0..HEAD_DIM {
                sk[col] += matrix[state_offset + row * HEAD_DIM + col] * k[row];
            }
        }
        let d: Vec<f32> = (0..HEAD_DIM)
            .map(|col| (v[col] - sk[col]) * beta[v_head])
            .collect();
        for row in 0..HEAD_DIM {
            for col in 0..HEAD_DIM {
                matrix[state_offset + row * HEAD_DIM + col] += k[row] * d[col];
            }
        }
        for col in 0..HEAD_DIM {
            let out: f32 = (0..HEAD_DIM)
                .map(|row| matrix[state_offset + row * HEAD_DIM + col] * q[row])
                .sum::<f32>()
                * scale;
            assert!(
                (actual_out[v_head * HEAD_DIM + col] - out).abs() < 1e-2,
                "out[{v_head}][{col}]: {} != {out}",
                actual_out[v_head * HEAD_DIM + col]
            );
        }
    }
    // The persisted matrix must equal the host replay elementwise.
    for (index, (actual, expected)) in actual_matrix.iter().zip(&matrix).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-2,
            "matrix[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn executes_gdn_gated_norm_and_residual_against_host_equations() {
    const V_HEADS: usize = 48;
    const HEAD_DIM: usize = 128;
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");

    let mut state = 555_u32;
    let input: Vec<f32> = (0..V_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let z_gate: Vec<f32> = (0..V_HEADS * HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let ssm_norm: Vec<f32> = (0..HEAD_DIM)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let eps = 1e-5_f32;

    let input_device = stream.clone_htod(&input).expect("upload gated input");
    let z_device = stream.clone_htod(&z_gate).expect("upload z gate");
    let norm_device = stream.clone_htod(&ssm_norm).expect("upload ssm norm");
    let mut out_device = stream
        .alloc_zeros::<f32>(V_HEADS * HEAD_DIM)
        .expect("allocate gated output");
    ops.gdn_gated_norm(
        &input_device,
        &z_device,
        &norm_device,
        &mut out_device,
        V_HEADS,
        HEAD_DIM,
        eps,
    )
    .expect("execute gated norm");
    let actual = stream
        .clone_dtoh(&out_device)
        .expect("download gated output");

    // Host replay: rms(o) * ssm_norm (raw, NOT +1'd) * silu(z) per head.
    let head_dim_f = f32::from(u16::try_from(HEAD_DIM).expect("head dim fits u16"));
    for head in 0..V_HEADS {
        let head_input = &input[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let sum: f32 = head_input.iter().map(|x| x * x).sum();
        let inv = (sum / head_dim_f + eps).sqrt().recip();
        for index in 0..HEAD_DIM {
            let zv = z_gate[head * HEAD_DIM + index];
            let silu = zv / (1.0 + (-zv).exp());
            let expected = head_input[index] * inv * ssm_norm[index] * silu;
            let actual = actual[head * HEAD_DIM + index];
            assert!(
                (actual - expected).abs() < 1e-4,
                "gated[{head}][{index}]: {actual} != {expected}"
            );
        }
    }

    // residual_add on a distinct small buffer.
    let acc: Vec<f32> = (0..100)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let inc: Vec<f32> = (0..100)
        .map(|_| fixture_quantized_f32(&mut state))
        .collect();
    let mut acc_device = stream.clone_htod(&acc).expect("upload accumulator");
    let inc_device = stream.clone_htod(&inc).expect("upload increment");
    ops.residual_add(&mut acc_device, &inc_device)
        .expect("execute residual add");
    let actual = stream
        .clone_dtoh(&acc_device)
        .expect("download accumulator");
    for (index, (actual, (a, i))) in actual.iter().zip(acc.iter().zip(&inc)).enumerate() {
        let expected = a + i;
        assert!((actual - expected).abs() < 1e-5, "residual[{index}]");
    }
}

#[test]
#[ignore = "requires the pinned Qwen GGUF"]
fn stages_qwen_tensors_with_budget_validation_and_gemv_lookup() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let provider =
        Qwen35ModelProvider::open("/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
            .expect("open pinned Qwen GGUF");

    let collect = |names: &[&str]| {
        names
            .iter()
            .map(|name| {
                let reader = provider.open_tensor(name).expect("open tensor");
                let spec = reader.spec().clone();
                let value_type = reader.value_type();
                let encoded_bytes = reader.remaining();
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(reader),
                }
            })
            .collect::<Vec<_>>()
    };

    // Budget below the requirement must be rejected before any upload.
    let norm_tensors = collect(&["output_norm.weight", "blk.0.ssm_dt.bias"]);
    let total: u64 = norm_tensors.iter().map(|tensor| tensor.encoded_bytes).sum();
    let undersized = match CudaQwen35Weights::stage(&context, &stream, total - 1, norm_tensors) {
        Err(error) => error,
        Ok(_) => panic!("undersized budget must be rejected before upload"),
    };
    assert!(matches!(
        undersized,
        CudaWeightStagingError::BudgetExceeded { .. }
    ));

    // A Q5_K tensor stages, finds its kernel family, and dispatches a real
    // GEMV against the staged device weights.
    let q5_tensors = collect(&["blk.0.attn_qkv.weight"]);
    let q5_bytes: u64 = q5_tensors.iter().map(|tensor| tensor.encoded_bytes).sum();
    let staged = CudaQwen35Weights::stage(&context, &stream, q5_bytes + 1, q5_tensors)
        .expect("stage Q5_K tensor");
    let weight = staged
        .quantized_tensor("blk.0.attn_qkv.weight")
        .expect("staged Q5_K weight");
    assert_eq!(weight.value_type(), 13);
    let kernel = staged.gemv_for(13).expect("Q5_K kernel compiled");

    let mut host_input = vec![0.0_f32; 5120];
    let mut state = 321_u32;
    for value in &mut host_input {
        *value = fixture_quantized_f32(&mut state);
    }
    let input = stream.clone_htod(&host_input).expect("upload input");
    let mut output = stream.alloc_zeros::<f32>(10240).expect("allocate output");
    kernel
        .execute(weight, &input, &mut output)
        .expect("execute staged GEMV");
    let actual = stream.clone_dtoh(&output).expect("download output");

    // Host replay against the GGUF decoder: column-major [K, N] contract.
    let mut reader = provider
        .open_tensor("blk.0.attn_qkv.weight")
        .expect("reopen tensor");
    let mut columns: Vec<Vec<f32>> = Vec::new();
    let mut block = Vec::new();
    while let Some(next) = reader.read_dequantized_block().expect("decode block") {
        block.extend(next);
        while block.len() >= 5120 {
            let take = 5120.min(block.len());
            let column: Vec<f32> = block[..take].to_vec();
            // Columns arrive element-major over K; accumulate per output.
            if columns.is_empty() {
                columns.push(Vec::new());
            }
            block.drain(..take);
        }
    }
    // Simpler: decode the full tensor into flat row-major [N][K] via the
    // gguf_gemv convention (column n = values[n*5120..(n+1)*5120]) and
    // compare the first outputs directly.
    let mut reader = provider
        .open_tensor("blk.0.attn_qkv.weight")
        .expect("reopen tensor again");
    let mut flat = Vec::new();
    while let Some(next) = reader.read_dequantized_block().expect("decode block") {
        flat.extend(next);
    }
    for (n, actual) in actual.iter().enumerate().take(256) {
        let expected: f32 = (0..5120).map(|k| flat[n * 5120 + k] * host_input[k]).sum();
        assert!(
            (actual - expected).abs() < 1.0,
            "gemv[{n}]: {actual} != {expected}"
        );
    }
}
