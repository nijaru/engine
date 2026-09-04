use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use engine_core::{
    BackendCapabilities, BackendFeatures, BackendId, BackendKind, DataType, DeviceId,
    ExecutionPhase, ExecutionPlan, ExecutionRuntime, ExecutionStage, InferenceState,
    InferenceStateSet, LogicalStateManager, ModelProvider, NvidiaBackend, PolicySnapshot,
    PolicyVersion, PromptFormat, PromptPolicy, Quantization, RequestId, RequestSemantics,
    RequestSpec, SamplingParams, SchedulerConfig, ServingRuntime, SpecialTokenPolicy,
    SpeculationPolicy, StateLocation, StateManager, StateRequirement, StateTierPreference,
    ThinkingMode, WeightBinding,
};
use engine_gguf::{
    ChatMessage, ChatTemplateOptions, GgufFile, Qwen35LayerKind, Qwen35ModelProvider,
};
use engine_nvidia::{
    CudaQwen35Decode, CudaQwen35ServingDispatcher, CudaQwen35Weights, QwenLayerKind,
    StagedTensorSource, wrap_f32_stream,
};

const DEFAULT_MAX_TOKENS: u32 = 32;
const DEFAULT_PREFILL_CHUNK_TOKENS: u32 = 16;
const WEIGHT_BUDGET_BYTES: u64 = 20_u64 << 30;
const EPSILON: f32 = 1.0e-6;

pub const USAGE: &str = "engine-server local --model <model.gguf> --prompt <text> [--max-tokens <n>] [--device <ordinal>]";

#[derive(Debug)]
struct LocalOptions {
    model: PathBuf,
    prompt: String,
    max_tokens: u32,
    device: u16,
}

#[allow(
    clippy::too_many_lines,
    reason = "direct CLI composition root wires the complete runtime explicitly"
)]
pub fn run(arguments: &[String]) -> Result<(), String> {
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        println!("{USAGE}");
        return Ok(());
    }
    let options = parse_options(arguments)?;
    let tokenizer_file =
        GgufFile::open(options.model.clone()).map_err(|error| error.to_string())?;
    let tokenizer = tokenizer_file
        .tokenizer()
        .map_err(|error| error.to_string())?;
    let prompt_tokens = tokenizer
        .encode_chat(
            &[ChatMessage::new("user", options.prompt)],
            ChatTemplateOptions::new(true, false),
        )
        .map_err(|error| error.to_string())?;
    let prompt_len = u32::try_from(prompt_tokens.len())
        .map_err(|_| "encoded prompt is too large for the current runtime".to_owned())?;
    let state_tokens = prompt_len
        .checked_add(options.max_tokens)
        .ok_or_else(|| "prompt plus output budget exceeds the current state index".to_owned())?;
    drop(tokenizer_file);

    let provider = Qwen35ModelProvider::open_with_kv_block_tokens(options.model, state_tokens)
        .map_err(|error| error.to_string())?;
    if u64::from(state_tokens) > provider.config().context_length() {
        return Err(format!(
            "prompt plus output budget ({state_tokens} tokens) exceeds model context ({})",
            provider.config().context_length()
        ));
    }

    let device = DeviceId::new(options.device);
    let context =
        CudaContext::new(usize::from(options.device)).map_err(|error| error.to_string())?;
    let stream = context.default_stream();
    let staged = Arc::new(stage_weights(&provider, device, &context, &stream)?);
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
    let state_capacity = state_requirements
        .iter()
        .try_fold(0_u64, |total, requirement| {
            total
                .checked_add(
                    requirement
                        .byte_size()
                        .ok_or_else(|| "inference-state size overflowed".to_owned())?,
                )
                .ok_or_else(|| "inference-state capacity overflowed".to_owned())
        })?;
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
    let version = PolicyVersion::new(1).ok_or_else(|| "invalid policy version".to_owned())?;
    let policy = PolicySnapshot::new(
        version,
        1,
        DEFAULT_PREFILL_CHUNK_TOKENS,
        StateTierPreference::Device,
        SpeculationPolicy::Disabled,
    )
    .map_err(|error| error.to_string())?;
    let scheduler = engine_core::ServingScheduler::new(
        policy,
        SchedulerConfig::new(1, 0, DEFAULT_PREFILL_CHUNK_TOKENS)
            .map_err(|error| error.to_string())?,
    );
    let plan = ExecutionPlan::new(
        model.clone(),
        backend_id,
        device,
        version,
        execution_stages,
        state_requirements.clone(),
        WeightBinding::empty(model.clone(), device),
    )
    .map_err(|error| error.to_string())?;

    let mut state_manager = LogicalStateManager::new(device, state_capacity, 0);
    let state = allocate_state(&mut state_manager, &state_requirements, device)?;
    let runtime = ExecutionRuntime::new(provider, backend, state_manager);
    let mut serving =
        ServingRuntime::new(scheduler, runtime, plan).map_err(|error| error.to_string())?;
    let request_id = RequestId::new(1).ok_or_else(|| "invalid request identity".to_owned())?;
    let semantics = RequestSemantics::new(
        options.max_tokens,
        SamplingParams::greedy(None),
        ThinkingMode::Off,
    )
    .map_err(|error| error.to_string())?
    .with_prompt_policy(PromptPolicy::new(
        PromptFormat::EmbeddedChatTemplate,
        SpecialTokenPolicy::None,
    ));
    let request = RequestSpec::new(request_id, model, semantics);
    serving
        .admit(request, state, Arc::from(prompt_tokens))
        .map_err(|error| error.to_string())?;

    let output_tokens = generate(&mut serving, request_id, tokenizer.eos_token_id())?;
    let text = tokenizer
        .decode(&output_tokens)
        .map_err(|error| error.to_string())?;
    println!("{text}");
    Ok(())
}

fn parse_options(arguments: &[String]) -> Result<LocalOptions, String> {
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut device = 0_u16;
    let mut index = 0;
    while index < arguments.len() {
        let name = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {name}; usage: {USAGE}"))?;
        match name.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value.clone()),
            "--max-tokens" => {
                max_tokens = value
                    .parse::<u32>()
                    .map_err(|_| "--max-tokens expects a positive integer".to_owned())?;
                if max_tokens == 0 {
                    return Err("--max-tokens must be greater than zero".to_owned());
                }
            }
            "--device" => {
                device = value
                    .parse::<u16>()
                    .map_err(|_| "--device expects a non-negative device ordinal".to_owned())?;
            }
            other => return Err(format!("unknown local option {other:?}; usage: {USAGE}")),
        }
        index += 2;
    }
    Ok(LocalOptions {
        model: model.ok_or_else(|| format!("--model is required; usage: {USAGE}"))?,
        prompt: prompt.ok_or_else(|| format!("--prompt is required; usage: {USAGE}"))?,
        max_tokens,
        device,
    })
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
    eprintln!(
        "staging {} Qwen tensors on CUDA device {}",
        names.len(),
        device.get()
    );
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
                let blocks = wrap_f32_stream(reader);
                StagedTensorSource {
                    spec,
                    value_type,
                    encoded_bytes,
                    reader: Box::new(io::empty()),
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

fn generate<P, B, S>(
    serving: &mut ServingRuntime<P, B, S>,
    request: RequestId,
    eos_token: u32,
) -> Result<Vec<u32>, String>
where
    P: ModelProvider,
    B: engine_core::ComputeBackend,
    S: StateManager,
{
    let mut output = Vec::new();
    loop {
        let completed = serving
            .poll_completions()
            .map_err(|error| error.to_string())?;
        let mut reached_eos = false;
        while let Some(generated) = serving.pop_generated_token() {
            if generated.request() != request {
                return Err("direct inference received output for another request".to_owned());
            }
            if generated.token() == eos_token {
                reached_eos = true;
            } else if !reached_eos {
                output.push(generated.token());
            }
        }
        if reached_eos && serving.scheduler().counts().terminal() == 0 {
            serving.finish(request).map_err(|error| error.to_string())?;
        }
        if serving.scheduler().counts().terminal() > 0 {
            break;
        }
        let submitted = serving
            .submit_ready_batch()
            .map_err(|error| error.to_string())?;
        if completed == 0 && submitted.is_none() {
            if serving.submission_count() == 0 {
                return Err(
                    "direct inference runtime stalled with no runnable or in-flight work"
                        .to_owned(),
                );
            }
            std::thread::yield_now();
        }
    }
    let reclaimed = serving
        .reclaim_next()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "terminal request was not reclaimable".to_owned())?;
    if reclaimed.request().id() != request {
        return Err("direct inference reclaimed the wrong request".to_owned());
    }
    Ok(output)
}
