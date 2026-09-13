use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{QwenGguf, QwenLayerKind as GgufQwenLayerKind};
use cudarc::driver::{CudaContext, CudaStream};
use engine_core::{
    BackendCapabilities, BackendFeatures, BackendId, BackendKind, DataType, DeviceId,
    ExecutionPhase, ExecutionPlan, ExecutionStage, ModelProvider, PolicyVersion, WeightBinding,
};
use engine_nvidia::NvidiaBackend;
use engine_nvidia::{
    CudaQwen35Decode, CudaQwen35ServingDispatcher, CudaQwen35Weights, QwenLayerKind,
    StagedTensorSource, wrap_f32_stream,
};
use ribn::{ExecutionError, ExecutorInfo, GenerationLimits};

use crate::execution::{QwenExecution, model_error};
use crate::state;

/// Same-sequence prefill chunk size this backend uses when a caller enables
/// prefill chunking without naming a size.
///
/// Eight is the largest qualified lane ([`engine_nvidia::MAX_BATCH_MEMBERS`]
/// bounds it) and the size the prefill qualification measured, so it is both
/// the fastest measured choice and the one with parity evidence.
pub const DEFAULT_PREFILL_CHUNK_MEMBERS: usize = 8;

/// Preparation limits, not live request scheduling policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenLoadOptions {
    pub device: u16,
    pub context_tokens: u32,
    pub max_sequences: usize,
    /// Enable same-sequence prefill chunking at this many tokens per chunk.
    /// `None` keeps the serial prefill path.
    pub prefill_chunk_members: Option<usize>,
    /// Optional upper bound for staged weights. Automatic mode uses observed
    /// free device memory minus request-state reservations and headroom.
    pub weight_budget_bytes: Option<u64>,
    /// Explicit reserve for preparation/workspace allocations and other users.
    /// This is not a proof that every future CUDA allocation will succeed.
    pub headroom_bytes: u64,
}

impl Default for QwenLoadOptions {
    fn default() -> Self {
        Self {
            device: 0,
            context_tokens: 4096,
            max_sequences: 1,
            prefill_chunk_members: None,
            weight_budget_bytes: None,
            headroom_bytes: 512 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryReport {
    pub free_before_preparation_bytes: u64,
    pub weight_budget_bytes: u64,
    pub reserved_sequence_bytes: u64,
    pub free_after_preparation_bytes: u64,
}

pub(crate) type Backend = NvidiaBackend<CudaQwen35ServingDispatcher>;

/// Keep the device owner separate so uncertain teardown can retain it whole.
pub(crate) struct Resources {
    pub(crate) execution: QwenExecution<Backend>,
    pub(crate) stream: Arc<CudaStream>,
}

pub(crate) fn load(
    path: PathBuf,
    options: QwenLoadOptions,
) -> Result<(Resources, MemoryReport), ExecutionError> {
    if options.context_tokens == 0 || options.max_sequences == 0 {
        return Err(ExecutionError::new(
            "Qwen context and sequence limits must be nonzero",
        ));
    }
    let provider =
        QwenGguf::open_with_kv_block_tokens(path, options.context_tokens).map_err(model_error)?;
    if u64::from(options.context_tokens) > provider.config().context_length() {
        return Err(ExecutionError::new(
            "requested context exceeds the Qwen artifact limit",
        ));
    }
    let vocabulary_size = u32::try_from(
        provider
            .file()
            .tokenizer()
            .map_err(model_error)?
            .tokens()
            .len(),
    )
    .map_err(model_error)?;
    let device = DeviceId::new(options.device);
    let reserved_sequence_bytes = state::capacity(
        provider.description().state_requirements(),
        options.max_sequences,
    )
    .ok_or_else(|| ExecutionError::new("Qwen state reservation overflow"))?;
    let context = CudaContext::new(usize::from(options.device)).map_err(model_error)?;
    let stream = context.default_stream();
    let free_before_preparation_bytes =
        u64::try_from(context.mem_get_info().map_err(model_error)?.0).map_err(model_error)?;
    let available = free_before_preparation_bytes.checked_sub(reserved_sequence_bytes)
        .and_then(|bytes| bytes.checked_sub(options.headroom_bytes))
        .ok_or_else(|| ExecutionError::new(format!("Qwen state requires {reserved_sequence_bytes} bytes plus {} bytes headroom, but only {free_before_preparation_bytes} bytes are free", options.headroom_bytes)))?;
    let weight_budget_bytes = options.weight_budget_bytes.unwrap_or(available);
    if weight_budget_bytes == 0 || weight_budget_bytes > available {
        return Err(ExecutionError::new(format!(
            "weight budget {weight_budget_bytes} exceeds the available {available} bytes after state/headroom reservations"
        )));
    }
    let staged = Arc::new(stage_weights(
        &provider,
        device,
        &context,
        &stream,
        weight_budget_bytes,
    )?);
    let layer_kinds = layer_kinds(&provider)?;
    let executor = CudaQwen35Decode::new(&context, stream.clone(), staged, layer_kinds, 1.0e-6)
        .map_err(model_error)?;
    let dispatcher =
        CudaQwen35ServingDispatcher::new(&context, executor, stream.clone(), options.max_sequences)
            .map_err(model_error)?;
    // The chunk lane is part of preparation, so its scratch is allocated
    // before the post-preparation memory check below accounts for it.
    let dispatcher = match options.prefill_chunk_members {
        Some(members) => dispatcher
            .with_prefill_chunk(members)
            .map_err(model_error)?,
        None => dispatcher,
    };
    // Preparation is outside the scheduler. Publish no ready owner until all
    // uploads and preparation work have completed.
    stream.synchronize().map_err(model_error)?;
    let free_after_preparation_bytes =
        u64::try_from(context.mem_get_info().map_err(model_error)?.0).map_err(model_error)?;
    if free_after_preparation_bytes < reserved_sequence_bytes {
        return Err(ExecutionError::new(format!(
            "prepared Qwen leaves {free_after_preparation_bytes} free bytes, below its {reserved_sequence_bytes}-byte sequence reservation"
        )));
    }
    let (backend, plan) = prepare_execution(&provider, &context, dispatcher, device)?;
    let info = ExecutorInfo {
        name: provider.description().id().to_string(),
        limits: GenerationLimits {
            context_tokens: options.context_tokens,
            max_sequences: options.max_sequences,
            max_batch_tokens: 128,
            max_decode_tokens: 1,
        },
    };
    let execution = QwenExecution::new(
        backend,
        info,
        plan,
        state::manager(device, reserved_sequence_bytes),
        vocabulary_size,
    );
    Ok((
        Resources { execution, stream },
        MemoryReport {
            free_before_preparation_bytes,
            weight_budget_bytes,
            reserved_sequence_bytes,
            free_after_preparation_bytes,
        },
    ))
}

fn prepare_execution(
    provider: &QwenGguf,
    context: &Arc<CudaContext>,
    dispatcher: CudaQwen35ServingDispatcher,
    device: DeviceId,
) -> Result<(Backend, ExecutionPlan), ExecutionError> {
    let description = provider.description();
    let backend_id = BackendId::new("cuda").map_err(model_error)?;
    let capabilities = BackendCapabilities::new(
        backend_id.clone(),
        device,
        BackendKind::Cuda,
        u64::try_from(context.total_mem().map_err(model_error)?).map_err(model_error)?,
        BackendFeatures::new(
            vec![DataType::F16, DataType::F32],
            vec![description.weights().quantization()],
            false,
            true,
        ),
    );
    let backend = NvidiaBackend::new(capabilities, dispatcher).map_err(model_error)?;
    let model = description.id().clone();
    let stages = [ExecutionPhase::Prefill, ExecutionPhase::Decode]
        .into_iter()
        .flat_map(|phase| {
            description
                .regions()
                .iter()
                .map(move |region| ExecutionStage::new(region.id(), phase))
        })
        .collect();
    let plan = ExecutionPlan::new(
        model.clone(),
        backend_id,
        device,
        PolicyVersion::new(1).expect("constant version"),
        stages,
        description.state_requirements().to_vec(),
        WeightBinding::empty(model.clone(), device),
    )
    .map_err(model_error)?;
    provider.validate_plan(&plan).map_err(model_error)?;
    Ok((backend, plan))
}

fn stage_weights(
    provider: &QwenGguf,
    device: DeviceId,
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    budget: u64,
) -> Result<CudaQwen35Weights, ExecutionError> {
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    let count = provider
        .config()
        .language_layer_count()
        .ok_or_else(|| ExecutionError::new("invalid Qwen language layer count"))?;
    for layer in 0..count {
        let binding = provider
            .layer_weight_binding(device, layer)
            .map_err(model_error)?;
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
            let reader = provider.open_tensor(name).map_err(model_error)?;
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
        .collect::<Result<Vec<_>, ExecutionError>>()?;
    CudaQwen35Weights::stage(context, stream, budget, tensors).map_err(model_error)
}

fn layer_kinds(provider: &QwenGguf) -> Result<Vec<QwenLayerKind>, ExecutionError> {
    let count = provider
        .config()
        .language_layer_count()
        .ok_or_else(|| ExecutionError::new("invalid Qwen language layer count"))?;
    (0..count)
        .map(|layer| {
            provider
                .layer_kind(layer)
                .map(|kind| match kind {
                    GgufQwenLayerKind::Recurrent => QwenLayerKind::Recurrent,
                    GgufQwenLayerKind::FullAttention => QwenLayerKind::FullAttention,
                })
                .map_err(model_error)
        })
        .collect()
}
