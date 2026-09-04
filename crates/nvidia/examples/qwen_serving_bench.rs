//! Multi-request serving-runtime benchmark for the pinned Qwen3.8-27B GGUF path.
//!
//! This exercises Engine's real scheduler/runtime/backend boundary rather than
//! calling the batch-1 decoder directly. The current CUDA dispatcher preserves
//! a scheduler batch but executes its requests sequentially, so this benchmark
//! is a qualification baseline for Phase 4C, not a native-batching claim.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo run --release -p engine-nvidia --features cuda --example qwen_serving_bench -- \
//!   --concurrency=4 --tokens=32
//! ```

use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use engine_core::{
    BackendCapabilities, BackendFeatures, BackendId, BackendKind, DataType, DeviceId,
    ExecutionPhase, ExecutionPlan, ExecutionRuntime, ExecutionStage, InferenceState,
    InferenceStateSet, LogicalStateManager, ModelProvider, NvidiaBackend, PolicySnapshot,
    PolicyVersion, Quantization, RequestId, RequestSemantics, RequestSpec, SamplingParams,
    SchedulerConfig, ServingRuntime, SpeculationPolicy, StateLocation, StateManager,
    StateRequirement, StateTierPreference, ThinkingMode, WeightBinding,
};
use engine_gguf::{Qwen35LayerKind, Qwen35ModelProvider};
use engine_nvidia::{
    CudaQwen35Decode, CudaQwen35ServingDispatcher, CudaQwen35Weights, QwenLayerKind,
    StagedTensorSource, wrap_f32_stream,
};

const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
const DEFAULT_CONCURRENCY: usize = 4;
const DEFAULT_OUTPUT_TOKENS: u32 = 32;
const PREFILL_CHUNK_TOKENS: u32 = 16;
const WEIGHT_BUDGET_BYTES: u64 = 20_u64 << 30;
const EPSILON: f32 = 1.0e-6;

#[allow(
    clippy::too_many_lines,
    reason = "one explicit serving benchmark composition root"
)]
fn main() {
    if let Err(error) = run() {
        eprintln!("qwen_serving_bench: {error}");
        std::process::exit(2);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one explicit serving benchmark composition root"
)]
fn run() -> Result<(), String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let concurrency = parse_usize(&arguments, "--concurrency=", DEFAULT_CONCURRENCY)?;
    let output_tokens = parse_u32(&arguments, "--tokens=", DEFAULT_OUTPUT_TOKENS)?;
    if concurrency == 0 || output_tokens == 0 {
        return Err("concurrency and token count must be greater than zero".to_owned());
    }
    let model_path = arguments
        .iter()
        .find_map(|argument| argument.strip_prefix("--model=").map(str::to_owned))
        .or_else(|| std::env::var("ENGINE_QWEN_GGUF").ok())
        .ok_or_else(|| "set ENGINE_QWEN_GGUF or pass --model=/path/to/model.gguf".to_owned())?;

    let state_tokens = u32::try_from(PROMPT.len())
        .map_err(|_| "prompt length does not fit the runtime".to_owned())?
        .checked_add(output_tokens)
        .ok_or_else(|| "prompt plus output budget overflowed".to_owned())?;
    let provider = Qwen35ModelProvider::open_with_kv_block_tokens(&model_path, state_tokens)
        .map_err(|error| error.to_string())?;
    if u64::from(state_tokens) > provider.config().context_length() {
        return Err("prompt plus output budget exceeds the model context".to_owned());
    }

    let device = DeviceId::new(0);
    let context = CudaContext::new(0).map_err(|error| error.to_string())?;
    let stream = context.default_stream();
    let stage_started = Instant::now();
    let staged = Arc::new(stage_weights(&provider, device, &context, &stream)?);
    let stage_elapsed = stage_started.elapsed();
    let layer_kinds = qwen_layer_kinds(&provider)?;
    let executor = CudaQwen35Decode::new(&context, stream.clone(), staged, layer_kinds, EPSILON)
        .map_err(|error| error.to_string())?;
    let dispatcher = CudaQwen35ServingDispatcher::new(executor, stream);

    let description = provider.description();
    let model = description.id().clone();
    let state_requirements = description.state_requirements().to_vec();
    let execution_stages = [ExecutionPhase::Prefill, ExecutionPhase::Decode]
        .into_iter()
        .flat_map(|phase| {
            description
                .regions()
                .iter()
                .map(move |region| ExecutionStage::new(region.id(), phase))
        })
        .collect::<Vec<_>>();
    let per_request_state_bytes = state_requirements
        .iter()
        .try_fold(0_u64, |total, requirement| {
            total.checked_add(requirement.byte_size()?)
        })
        .ok_or_else(|| "inference-state size overflowed".to_owned())?;
    let state_capacity = per_request_state_bytes
        .checked_mul(u64::try_from(concurrency).map_err(|_| "concurrency is too large".to_owned())?)
        .ok_or_else(|| "aggregate inference-state size overflowed".to_owned())?;
    let memory_budget = WEIGHT_BUDGET_BYTES
        .checked_add(state_capacity)
        .ok_or_else(|| "CUDA memory budget overflowed".to_owned())?;

    let backend_id = BackendId::new("cuda").map_err(|error| error.to_string())?;
    let capabilities = BackendCapabilities::new(
        backend_id.clone(),
        device,
        BackendKind::Cuda,
        memory_budget,
        BackendFeatures::new(
            vec![DataType::F16, DataType::F32],
            vec![Quantization::GgufQ4Km],
            false,
            true,
        ),
    );
    let backend =
        NvidiaBackend::new(capabilities, dispatcher).map_err(|error| error.to_string())?;
    let policy_version =
        PolicyVersion::new(1).ok_or_else(|| "invalid policy version".to_owned())?;
    let batch_size =
        u32::try_from(concurrency).map_err(|_| "concurrency exceeds u32".to_owned())?;
    let batch_tokens = batch_size
        .checked_mul(PREFILL_CHUNK_TOKENS)
        .ok_or_else(|| "batch token budget overflowed".to_owned())?;
    let policy = PolicySnapshot::new(
        policy_version,
        batch_size,
        batch_tokens,
        StateTierPreference::Device,
        SpeculationPolicy::Disabled,
    )
    .map_err(|error| error.to_string())?;
    let scheduler = engine_core::ServingScheduler::new(
        policy,
        SchedulerConfig::new(concurrency, 0, PREFILL_CHUNK_TOKENS)
            .map_err(|error| error.to_string())?,
    );
    let plan = ExecutionPlan::new(
        model.clone(),
        backend_id,
        device,
        policy_version,
        execution_stages,
        state_requirements.clone(),
        WeightBinding::empty(model.clone(), device),
    )
    .map_err(|error| error.to_string())?;

    let mut state_manager = LogicalStateManager::new(device, state_capacity, 0);
    let states = (0..concurrency)
        .map(|_| allocate_state(&mut state_manager, &state_requirements, device))
        .collect::<Result<Vec<_>, _>>()?;
    let runtime = ExecutionRuntime::new(provider, backend, state_manager);
    let mut serving =
        ServingRuntime::new(scheduler, runtime, plan).map_err(|error| error.to_string())?;
    let prompt: Arc<[u32]> = Arc::from(PROMPT);
    for (index, state) in states.into_iter().enumerate() {
        let request_id = request_id(index)?;
        let semantics = RequestSemantics::new(
            output_tokens,
            SamplingParams::greedy(None),
            ThinkingMode::Off,
        )
        .map_err(|error| error.to_string())?;
        serving
            .admit(
                RequestSpec::new(request_id, model.clone(), semantics),
                state,
                prompt.clone(),
            )
            .map_err(|error| error.to_string())?;
    }

    let benchmark_started = Instant::now();
    let mut first_token = vec![None; concurrency];
    let mut previous_token = vec![None; concurrency];
    let mut inter_token = Vec::new();
    let mut generated_tokens = 0_u64;
    let mut remaining = concurrency;

    while remaining > 0 {
        let completed = serving
            .poll_completions()
            .map_err(|error| error.to_string())?;
        while let Some(generated) = serving.pop_generated_token() {
            let now = benchmark_started.elapsed();
            let index = request_index(generated.request(), concurrency)?;
            if first_token[index].is_none() {
                first_token[index] = Some(now);
            }
            if let Some(previous) = previous_token[index].replace(now) {
                inter_token.push(now.saturating_sub(previous));
            }
            generated_tokens = generated_tokens
                .checked_add(1)
                .ok_or_else(|| "generated-token counter overflowed".to_owned())?;
        }
        while serving.scheduler().counts().terminal() > 0 {
            serving
                .reclaim_next()
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "terminal request was not reclaimable".to_owned())?;
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        let submitted = serving
            .submit_ready_batch()
            .map_err(|error| error.to_string())?;
        if completed == 0 && submitted.is_none() {
            if serving.submission_count() == 0 {
                return Err(
                    "serving benchmark stalled without runnable or in-flight work".to_owned(),
                );
            }
            std::thread::yield_now();
        }
    }

    let elapsed = benchmark_started.elapsed();
    let expected_tokens = u64::try_from(concurrency)
        .ok()
        .and_then(|value| value.checked_mul(u64::from(output_tokens)))
        .ok_or_else(|| "expected-token count overflowed".to_owned())?;
    if generated_tokens != expected_tokens {
        return Err(format!(
            "serving path emitted {generated_tokens} tokens, expected {expected_tokens}"
        ));
    }

    let ttft = first_token
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "at least one request never produced a first token".to_owned())?;
    let (ttft_mean, ttft_max) = duration_mean_max(&ttft);
    let (itl_mean, itl_max) = duration_mean_max(&inter_token);
    println!("Qwen3.8 serving-runtime baseline");
    println!("  concurrency: {concurrency}");
    println!("  output tokens/request: {output_tokens}");
    println!("  staged weights: {:.2} s", stage_elapsed.as_secs_f64());
    println!("  request-state bytes: {per_request_state_bytes} each, {state_capacity} aggregate");
    println!("  elapsed: {:.3} s", elapsed.as_secs_f64());
    let generated_tokens_f64 = f64::from(
        u32::try_from(generated_tokens)
            .map_err(|_| "generated-token count exceeds benchmark reporting range".to_owned())?,
    );
    println!(
        "  aggregate throughput: {:.2} tok/s",
        generated_tokens_f64 / elapsed.as_secs_f64()
    );
    println!(
        "  observed TTFT: mean {:.3} s, max {:.3} s",
        ttft_mean.as_secs_f64(),
        ttft_max.as_secs_f64()
    );
    if !inter_token.is_empty() {
        println!(
            "  observed ITL: mean {:.3} ms, max {:.3} ms",
            itl_mean.as_secs_f64() * 1000.0,
            itl_max.as_secs_f64() * 1000.0
        );
    }
    println!(
        "  caveat: the current CUDA serving dispatcher executes members of each scheduler batch sequentially; this measures the Phase-4C baseline before native batching/async completion"
    );
    Ok(())
}

fn parse_usize(arguments: &[String], prefix: &str, default: usize) -> Result<usize, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix(prefix))
        .map_or(Ok(default), |value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
}

fn parse_u32(arguments: &[String], prefix: &str, default: u32) -> Result<u32, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix(prefix))
        .map_or(Ok(default), |value| {
            value
                .parse::<u32>()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
}

fn request_id(index: usize) -> Result<RequestId, String> {
    let value = u64::try_from(index)
        .map_err(|_| "request index is too large".to_owned())?
        .checked_add(1)
        .ok_or_else(|| "request identity overflowed".to_owned())?;
    RequestId::new(value).ok_or_else(|| "invalid request identity".to_owned())
}

fn request_index(request: RequestId, concurrency: usize) -> Result<usize, String> {
    let index = request
        .get()
        .checked_sub(1)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "invalid request identity in generated output".to_owned())?;
    if index >= concurrency {
        return Err("generated output belongs to an unknown request".to_owned());
    }
    Ok(index)
}

fn duration_mean_max(values: &[Duration]) -> (Duration, Duration) {
    if values.is_empty() {
        return (Duration::ZERO, Duration::ZERO);
    }
    let total = values.iter().map(Duration::as_secs_f64).sum::<f64>();
    let count = u32::try_from(values.len()).expect("benchmark sample count fits u32");
    let mean = Duration::from_secs_f64(total / f64::from(count));
    let max = values.iter().copied().max().unwrap_or(Duration::ZERO);
    (mean, max)
}

fn stage_weights(
    provider: &Qwen35ModelProvider,
    device: DeviceId,
    context: &Arc<CudaContext>,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<CudaQwen35Weights, String> {
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    let layer_count = provider
        .config()
        .language_layer_count()
        .ok_or_else(|| "Qwen language layer count is invalid".to_owned())?;
    for layer in 0..layer_count {
        let binding = provider
            .layer_weight_binding(device, layer)
            .map_err(|error| error.to_string())?;
        names.extend(
            binding
                .tensors()
                .iter()
                .map(|tensor| tensor.name().to_owned()),
        );
    }
    let tensors = names
        .iter()
        .map(|name| {
            let reader = provider
                .open_tensor(name)
                .map_err(|error| error.to_string())?;
            let spec = reader.spec().clone();
            let value_type = reader.value_type();
            let encoded_bytes = reader.remaining();
            Ok(if matches!(value_type, 0 | 1) {
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(io::empty()),
                    f32_blocks: Some(Box::new(wrap_f32_stream(reader))),
                }
            } else {
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(reader),
                    f32_blocks: None,
                }
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    CudaQwen35Weights::stage(context, stream, WEIGHT_BUDGET_BYTES, tensors)
        .map_err(|error| error.to_string())
}

fn qwen_layer_kinds(provider: &Qwen35ModelProvider) -> Result<Vec<QwenLayerKind>, String> {
    let layer_count = provider
        .config()
        .language_layer_count()
        .ok_or_else(|| "Qwen language layer count is invalid".to_owned())?;
    (0..layer_count)
        .map(|layer| {
            provider
                .layer_kind(layer)
                .map(|kind| match kind {
                    Qwen35LayerKind::Recurrent => QwenLayerKind::Recurrent,
                    Qwen35LayerKind::FullAttention => QwenLayerKind::FullAttention,
                })
                .map_err(|error| error.to_string())
        })
        .collect()
}

fn allocate_state(
    manager: &mut LogicalStateManager,
    requirements: &[StateRequirement],
    device: DeviceId,
) -> Result<InferenceStateSet, String> {
    let location = StateLocation::Device(device);
    let states = requirements
        .iter()
        .map(|requirement| match *requirement {
            StateRequirement::FullAttentionKv(spec) => manager
                .allocate_kv(spec, location)
                .map(InferenceState::from)
                .map_err(|error| error.to_string()),
            StateRequirement::Recurrent(spec) => manager
                .allocate_recurrent(spec, location)
                .map(InferenceState::from)
                .map_err(|error| error.to_string()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    InferenceStateSet::new(states).map_err(|error| error.to_string())
}
