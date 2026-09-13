//! Batch-1 greedy decode and experimental prompt-prefill timing over the pinned
//! Qwen3.8-27B GGUF text path.
//!
//! Stages the full text path onto the device, prefills a raw token prompt, then
//! greedily decodes `N` tokens while timing prefill and decode separately.
//! `--prefill-chunk=N` selects the same-sequence chunk executor that serving also
//! uses by default, so the two modes can be compared on one build; `--tokens` and
//! `--prompt-tokens` build a repeated prompt for timing.
//!
//! `--prompt-fixture=<path>` reads a reference fixture's prompt tokens instead, and
//! `--logit-margins=N` prints the first `N` steps' top-five log-probabilities and
//! top-1/top-2 margin, which is what a comparison against another engine's recorded
//! log-probabilities needs: agreeing on the winning token says nothing about how
//! close the runners-up were.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo run --release -p engine-nvidia --features cuda --example qwen_decode_bench -- \
//!   --tokens=64 --prompt-tokens=257 --prefill-chunk=8
//! ```

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaContext;
use engine_core::{
    ConvolutionStateShape, DataType, DeviceId, KvStateSpec, RecurrentMatrixShape,
    RecurrentStateSpec,
};
use engine_nvidia::{
    CudaHybridState, CudaQwen35BatchDecode, CudaQwen35Decode, CudaQwen35Weights, QwenLayerKind,
    StagedTensorSource,
};
use engine_qwen::QwenGguf;

const BASE_PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
const EPS: f32 = 1.0e-6;
const KV_CAPACITY: usize = 512;

#[allow(clippy::too_many_lines, reason = "one linear bench script")]
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let token_count: u32 = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--tokens="))
        .map_or(64, |value| {
            value.parse().expect("--tokens expects a number")
        });
    assert!(token_count > 0, "--tokens must be nonzero");
    let prompt_token_count = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--prompt-tokens="))
        .map_or(BASE_PROMPT.len(), |value| {
            value.parse().expect("--prompt-tokens expects a number")
        });
    let decode_token_count = usize::try_from(token_count).expect("token count fits usize");
    let total_token_count = prompt_token_count
        .checked_add(decode_token_count)
        .expect("prompt plus decode token count fits usize");
    assert!(
        prompt_token_count > 0 && total_token_count <= KV_CAPACITY,
        "--prompt-tokens + --tokens must fit the {KV_CAPACITY}-token benchmark KV cache"
    );
    let prefill_chunk = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--prefill-chunk="))
        .map(|value| {
            value
                .parse::<usize>()
                .expect("--prefill-chunk expects a number")
        });
    // Number of generation steps whose top-k log-probabilities are printed. Used
    // with `--prompt-fixture` to compare greedy margins against a reference
    // engine that recorded the same prompt, which token agreement alone cannot do.
    let margin_steps = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--logit-margins="))
        .map_or(0, |value| {
            value
                .parse::<usize>()
                .expect("--logit-margins expects a number")
        });
    let prompt = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--prompt-fixture="))
        .map_or_else(
            || {
                if prompt_token_count == BASE_PROMPT.len() {
                    BASE_PROMPT.to_vec()
                } else {
                    BASE_PROMPT
                        .iter()
                        .copied()
                        .cycle()
                        .take(prompt_token_count)
                        .collect::<Vec<_>>()
                }
            },
            |path| {
                let text = std::fs::read_to_string(path).expect("read prompt fixture");
                let mut lines = text.lines();
                lines.next().expect("fixture identity line");
                lines
                    .next()
                    .expect("fixture prompt line")
                    .split_whitespace()
                    .map(|token| token.parse::<u32>().expect("fixture token"))
                    .collect::<Vec<_>>()
            },
        );
    assert!(
        prompt.len() + decode_token_count <= KV_CAPACITY,
        "fixture prompt plus --tokens must fit the {KV_CAPACITY}-token benchmark KV cache"
    );
    let model_path = args
        .iter()
        .find_map(|argument| argument.strip_prefix("--model=").map(str::to_owned))
        .or_else(|| std::env::var("ENGINE_QWEN_GGUF").ok())
        .expect("set ENGINE_QWEN_GGUF or pass --model=/path/to/model.gguf");

    let provider = QwenGguf::open(&model_path).expect("open Qwen GGUF");
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
                engine_qwen::QwenLayerKind::Recurrent => QwenLayerKind::Recurrent,
                engine_qwen::QwenLayerKind::FullAttention => QwenLayerKind::FullAttention,
            },
        )
        .collect::<Vec<_>>();

    let kv_spec = KvStateSpec::new(
        16,
        4,
        256,
        u32::try_from(KV_CAPACITY).expect("KV capacity fits u32"),
        DataType::F16,
    )
    .expect("KV spec");
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
    let mut chunk_executor = prefill_chunk.map(|members| {
        CudaQwen35BatchDecode::from_decode(&executor, members)
            .unwrap_or_else(|error| panic!("invalid --prefill-chunk={members}: {error}"))
    });

    let prefill_start = Instant::now();
    let last_prompt_index = prompt.len() - 1;
    let mut position = 0_usize;
    if let Some(chunk) = chunk_executor.as_mut() {
        let members = chunk.members();
        while position + members <= last_prompt_index {
            let start = u32::try_from(position).expect("prefill position fits u32");
            chunk
                .prefill_chunk(&mut state, &prompt[position..position + members], start)
                .expect("chunked prefill");
            position += members;
        }
    }
    while position < last_prompt_index {
        executor
            .prefill_step(
                &mut state,
                prompt[position],
                u32::try_from(position).expect("prefill position fits u32"),
            )
            .expect("serial prefill step");
        position += 1;
    }
    let final_position = u32::try_from(last_prompt_index).expect("prompt position fits u32");
    let mut chosen = executor
        .decode_step(&mut state, prompt[last_prompt_index], final_position)
        .expect("final prefill step");
    stream.synchronize().expect("sync after prefill");
    if margin_steps > 0 {
        print_logit_margins(0, chosen, &executor.copy_logits().expect("logits readback"));
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let prefill_mode = prefill_chunk.map_or_else(
        || "serial batch-1 AR loop".to_owned(),
        |members| format!("same-sequence chunks of {members}"),
    );
    println!(
        "prefill {} tokens in {prefill_seconds:.3} s ({:.3} s/token, {prefill_mode})",
        prompt.len(),
        prefill_seconds / f64::from(u32::try_from(prompt.len()).expect("fits u32"))
    );

    let tokens_f64 = f64::from(token_count);
    let decode_start = Instant::now();
    let first_decode_position = u32::try_from(prompt.len()).expect("prompt length fits u32");
    for step in 0..token_count {
        let decode_position = first_decode_position
            .checked_add(step)
            .expect("decode position fits u32");
        chosen = executor
            .decode_step(&mut state, chosen, decode_position)
            .expect("decode step");
        let generation_step = usize::try_from(step).expect("step fits usize");
        if generation_step + 1 < margin_steps {
            print_logit_margins(
                generation_step + 1,
                chosen,
                &executor.copy_logits().expect("logits readback"),
            );
        }
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
        "caveat: decode still synchronizes per step for the argmax and issues ~{} kernel launches per step, so this is a local measurement, not a serving-throughput claim",
        per_step_launches()
    );
}

/// Print the top five log-probabilities and the top-1/top-2 margin for one
/// generation step, in the same shape as a reference engine's `n_probs` output
/// so the two can be compared directly.
fn print_logit_margins(index: usize, chosen: u32, logits: &[f32]) {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f32>();
    let log_normalizer = maximum + sum.ln();
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let top = ranked
        .iter()
        .take(5)
        .map(|(token, logit)| format!("{token}:{:.4}", logit - log_normalizer))
        .collect::<Vec<_>>()
        .join(" ");
    let margin = ranked[0].1 - ranked[1].1;
    println!("margins[{index}] chosen={chosen} gap={margin:.4} {top}");
}

fn per_step_launches() -> usize {
    let gdn_layer = 10;
    let attn_layer = 11;
    let ffn = 5;
    48 * (gdn_layer + ffn) + 16 * (attn_layer + ffn) + 4
}
