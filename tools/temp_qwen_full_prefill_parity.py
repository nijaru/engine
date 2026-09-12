from pathlib import Path

path = Path('crates/nvidia/tests/cuda_reference.rs')
text = path.read_text()
anchor = '''/// Bisect the batched divergence by real-weight prefix: plans [R], [R,R],
'''
if anchor not in text:
    raise SystemExit('prefill parity insertion anchor not found')

addition = r'''#[test]
#[ignore = "requires the pinned Qwen GGUF and an idle CUDA device with enough memory"]
#[allow(
    clippy::too_many_lines,
    reason = "one full-model hardware qualification gate for same-sequence prefill"
)]
fn same_sequence_prefill_chunk_matches_batch1_full_model() {
    use engine_core::{
        ConvolutionStateShape, DataType, DeviceId, KvStateSpec, RecurrentMatrixShape,
        RecurrentStateSpec,
    };
    use engine_nvidia::{
        CudaHybridState, CudaQwen35BatchDecode, CudaQwen35Decode, CudaQwen35Weights,
        QwenLayerKind, StagedTensorSource,
    };
    use std::sync::Arc;

    const GGUF: &str = "/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf";
    const EPS: f32 = 1.0e-6;
    const TOKENS: [u32; 8] = [12_675, 1017, 760, 6511, 314, 9338, 369, 42];
    const NEXT_TOKEN: u32 = 17;
    const LAYERS: u32 = 64;
    const TOLERANCE: f32 = 5.0e-3;

    let provider = QwenGguf::open(GGUF).expect("open pinned Qwen GGUF");
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let device = DeviceId::new(0);

    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    for layer in 0..LAYERS {
        let binding = provider
            .layer_weight_binding(device, layer)
            .expect("layer binding");
        for spec in binding.tensors() {
            names.push(spec.name().to_owned());
        }
    }
    let tensors: Vec<StagedTensorSource> = names
        .iter()
        .map(|name| {
            let reader = provider.open_tensor(name).expect("open tensor");
            let spec = reader.spec().clone();
            let value_type = reader.value_type();
            let encoded_bytes = reader.remaining();
            if matches!(value_type, 0 | 1) {
                let blocks = engine_nvidia::wrap_f32_stream(reader);
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(std::io::empty()),
                    f32_blocks: Some(Box::new(blocks)),
                }
            } else {
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(reader),
                    f32_blocks: None,
                }
            }
        })
        .collect();
    let staged = Arc::new(
        CudaQwen35Weights::stage(&context, &stream, 20_u64 << 30, tensors)
            .expect("stage the full text path"),
    );
    let layer_kinds = (0..LAYERS)
        .map(|layer| match provider.layer_kind(layer).expect("layer kind") {
            engine_qwen::QwenLayerKind::Recurrent => QwenLayerKind::Recurrent,
            engine_qwen::QwenLayerKind::FullAttention => QwenLayerKind::FullAttention,
        })
        .collect::<Vec<_>>();

    let kv_spec = KvStateSpec::new(16, 4, 256, 16, DataType::F16).expect("KV spec");
    let recurrent_spec = RecurrentStateSpec::new(
        48,
        RecurrentMatrixShape::new(48, 128, 128).expect("matrix shape"),
        ConvolutionStateShape::new(10_240, 3).expect("convolution shape"),
        DataType::F32,
        DataType::F32,
    )
    .expect("recurrent spec");
    let fresh_state = || {
        let mut state =
            CudaHybridState::from_specs(stream.clone(), Some(kv_spec), Some(recurrent_spec))
                .expect("physical hybrid state");
        state.zero().expect("zero state");
        state
    };

    let mut oracle = CudaQwen35Decode::new(
        &context,
        stream.clone(),
        Arc::clone(&staged),
        layer_kinds.clone(),
        EPS,
    )
    .expect("batch-1 oracle");
    let mut oracle_state = fresh_state();
    let mut oracle_hidden = Vec::with_capacity(TOKENS.len());
    for (position, &token) in TOKENS.iter().enumerate() {
        let position = u32::try_from(position).expect("position fits u32");
        oracle
            .prefill_step(&mut oracle_state, token, position)
            .expect("oracle prefill");
        oracle_state
            .advance_to(position + 1)
            .expect("oracle advance");
        oracle_hidden.push(oracle.copy_hidden().expect("oracle hidden"));
    }

    let mut candidate_single = CudaQwen35Decode::new(
        &context,
        stream.clone(),
        Arc::clone(&staged),
        layer_kinds,
        EPS,
    )
    .expect("candidate single executor");
    let mut chunk = CudaQwen35BatchDecode::from_decode(&candidate_single, TOKENS.len())
        .expect("chunk executor");
    let mut chunk_state = fresh_state();
    chunk
        .prefill_chunk(&mut chunk_state, &TOKENS, 0)
        .expect("full-model same-sequence prefill chunk");
    chunk_state
        .advance_to(u32::try_from(TOKENS.len()).expect("chunk length fits u32"))
        .expect("chunk advance");

    for (member, expected) in oracle_hidden.iter().enumerate() {
        let actual = chunk.copy_hidden_member(member).expect("chunk hidden row");
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < TOLERANCE,
            "full-model chunk residual diverged at prompt row {member}: max abs {max_abs}"
        );
    }

    let next_position = u32::try_from(TOKENS.len()).expect("chunk length fits u32");
    oracle
        .prefill_step(&mut oracle_state, NEXT_TOKEN, next_position)
        .expect("oracle continuation");
    candidate_single
        .prefill_step(&mut chunk_state, NEXT_TOKEN, next_position)
        .expect("chunk continuation");
    let oracle_next = oracle.copy_hidden().expect("oracle continuation hidden");
    let chunk_next = candidate_single
        .copy_hidden()
        .expect("chunk continuation hidden");
    let max_abs = oracle_next
        .iter()
        .zip(&chunk_next)
        .map(|(oracle, chunk)| (oracle - chunk).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs < TOLERANCE,
        "full-model chunk continuation state diverged: max abs {max_abs}"
    );
}

'''
path.write_text(text.replace(anchor, addition + anchor, 1))
