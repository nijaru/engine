//! Batch-1 greedy decode timing over the pinned Qwen3.8-27B GGUF text path.
//!
//! Stages the full text path onto the device, prefills the raw prompt
//! "The capital of France is", then greedily decodes `N` tokens while
//! timing prefill and decode separately. Correctness gating lives in the
//! `decodes_greedy_tokens_matching_llama_server` test; this example only
//! measures and reports host-driven, batch-1, unoptimized throughput with
//! explicit caveats.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- --tokens=64
//! ```

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaContext;
use engine_core::{
    ConvolutionStateShape, DataType, DeviceId, KvStateSpec, RecurrentMatrixShape,
    RecurrentStateSpec,
};
use engine_gguf::Qwen35ModelProvider;
use engine_nvidia::{
    CudaHybridState, CudaQwen35Decode, CudaQwen35Weights, QwenLayerKind, StagedTensorSource,
};

const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
const EPS: f32 = 1.0e-6;

#[allow(clippy::too_many_lines, reason = "one linear bench script")]
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let token_count: u32 = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--tokens="))
        .map_or(64, |value| {
            value.parse().expect("--tokens expects a number")
        });
    let model_path = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--model=").map(str::to_owned))
        .or_else(|| std::env::var("ENGINE_QWEN_GGUF").ok())
        .expect("set ENGINE_QWEN_GGUF or pass --model=/path/to/model.gguf");

    let provider = Qwen35ModelProvider::open(&model_path).expect("open Qwen GGUF");
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();

    let stage_start = Instant::now();
    let mut names: Vec<String> = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    let device = DeviceId::new(0);
    for layer in 0..64_u32 {
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
    let stage_seconds = stage_start.elapsed().as_secs_f64();
    println!(
        "staged {} tensors in {stage_seconds:.2} s (host reads + device upload, one-time)",
        3 + 48 * 14 + 16 * 11
    );

    let layer_kinds = (0..64_u32)
        .map(
            |layer| match provider.layer_kind(layer).expect("layer kind") {
                engine_gguf::Qwen35LayerKind::Recurrent => QwenLayerKind::Recurrent,
                engine_gguf::Qwen35LayerKind::FullAttention => QwenLayerKind::FullAttention,
            },
        )
        .collect::<Vec<_>>();

    let kv_spec = KvStateSpec::new(16, 4, 256, 512, DataType::F16).expect("KV spec");
    let recurrent_spec = RecurrentStateSpec::new(
        48,
        RecurrentMatrixShape::new(48, 128, 128).expect("matrix shape"),
        ConvolutionStateShape::new(10_240, 3).expect("convolution shape"),
        DataType::F32,
        DataType::F32,
    )
    .expect("recurrent spec");
    let mut state =
        CudaHybridState::from_specs(stream.clone(), Some(kv_spec), Some(recurrent_spec))
            .expect("physical hybrid state");
    state.zero().expect("zero state");

    let mut executor = CudaQwen35Decode::new(&context, stream.clone(), staged, layer_kinds, EPS)
        .expect("build decode executor");

    let prefill_start = Instant::now();
    let mut chosen = 0_u32;
    for (position, token) in PROMPT.iter().enumerate() {
        chosen = executor
            .decode_step(
                &mut state,
                *token,
                u32::try_from(position).expect("fits u32"),
            )
            .expect("prefill step");
    }
    stream.synchronize().expect("sync after prefill");
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    println!(
        "prefill {} tokens in {prefill_seconds:.3} s ({:.3} s/token, batch-1 AR loop)",
        PROMPT.len(),
        prefill_seconds / f64::from(u32::try_from(PROMPT.len()).expect("fits u32"))
    );

    let tokens_f64 = f64::from(token_count);
    let decode_start = Instant::now();
    let mut next_token = chosen;
    let first_decode_position = u32::try_from(PROMPT.len()).expect("prompt length fits u32");
    for step in 0..token_count {
        let position = first_decode_position
            .checked_add(step)
            .expect("decode position fits u32");
        next_token = executor
            .decode_step(&mut state, next_token, position)
            .expect("decode step");
    }
    stream.synchronize().expect("sync after decode");
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    println!(
        "decode {} tokens in {decode_seconds:.3} s ({:.3} s/token = {:.2} tok/s, host-driven batch-1, unoptimized)",
        token_count,
        decode_seconds / tokens_f64,
        tokens_f64 / decode_seconds
    );
    println!(
        "caveat: per-step argmax synchronization and ~{} kernel launches per step dominate; \
         this is a correctness-path measurement, not a serving-throughput claim",
        per_step_launches()
    );
}

fn per_step_launches() -> usize {
    let gdn_layer = 10;
    let attn_layer = 11;
    let ffn = 5;
    48 * (gdn_layer + ffn) + 16 * (attn_layer + ffn) + 4
}
