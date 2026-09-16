//! GGUF/CUDA assembly for the text facade.
//!
//! Loading is deliberately separate from facade behavior so the preprocessing,
//! delivery and batching paths are exercisable without a device.

#[cfg(test)]
mod overhead;

use std::path::PathBuf;
use std::sync::Arc;

use engine_gguf::{GgufError, GgufFile, MetadataValue};
use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::Engine;
use ribn::driver::{Driver, DriverConfig};

use crate::error::TextError;
use crate::model::{TextConfig, TextOwner};
use crate::process::{GgufProcessor, ProcessorLimits};

pub use engine_qwen::MemoryReport;

/// Model loading and text-layer limits for the currently supported path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadOptions {
    pub device: u16,
    pub context_tokens: u32,
    pub max_sequences: usize,
    /// Same-sequence prefill chunk size, or `None` for the serial prefill path.
    pub prefill_chunk_members: Option<usize>,
    pub weight_budget_bytes: Option<u64>,
    /// Continuation bytes the prepared execution may charge in total. `None` keeps the
    /// conservative context-sized reservation; a request is charged for the tokens it
    /// can reach either way.
    pub continuation_capacity_bytes: Option<u64>,
    pub headroom_bytes: u64,
    /// Text-layer assembly settings.
    pub text: TextConfig,
    /// Text preprocessing bounds. `max_prompt_tokens` is additionally clamped to
    /// the configured context length.
    pub limits: ProcessorLimits,
}

impl Default for LoadOptions {
    fn default() -> Self {
        let qwen = QwenLoadOptions::default();
        Self {
            device: qwen.device,
            context_tokens: qwen.context_tokens,
            max_sequences: qwen.max_sequences,
            prefill_chunk_members: qwen.prefill_chunk_members,
            weight_budget_bytes: qwen.weight_budget_bytes,
            continuation_capacity_bytes: qwen.continuation_capacity_bytes,
            headroom_bytes: qwen.headroom_bytes,
            text: TextConfig::default(),
            limits: ProcessorLimits::default(),
        }
    }
}

impl From<LoadOptions> for QwenLoadOptions {
    fn from(value: LoadOptions) -> Self {
        Self {
            device: value.device,
            context_tokens: value.context_tokens,
            max_sequences: value.max_sequences,
            prefill_chunk_members: value.prefill_chunk_members,
            weight_budget_bytes: value.weight_budget_bytes,
            continuation_capacity_bytes: value.continuation_capacity_bytes,
            headroom_bytes: value.headroom_bytes,
        }
    }
}

impl TextOwner {
    /// Load a supported local GGUF model and prepare its CUDA executor.
    ///
    /// Returns the lifecycle owner and the prepared memory report.
    ///
    /// # Errors
    /// Returns an explicit unsupported-architecture error instead of inferring
    /// execution support from the existence of GGUF metadata.
    pub fn load(
        path: impl Into<PathBuf>,
        options: LoadOptions,
    ) -> Result<(Self, MemoryReport), TextError> {
        let path = path.into();
        let file = GgufFile::open(&path).map_err(TextError::Processor)?;
        let architecture = file
            .metadata("general.architecture")
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| {
                TextError::Processor(GgufError::MissingMetadata(
                    "general.architecture".to_owned(),
                ))
            })?;
        if architecture != "qwen35" {
            return Err(TextError::Processor(GgufError::UnsupportedArchitecture(
                format!("{architecture}; this build executes qwen35 GGUF only"),
            )));
        }
        let tokenizer = file.tokenizer().map_err(TextError::Processor)?;
        drop(file);
        let limits = ProcessorLimits {
            max_prompt_tokens: options
                .limits
                .max_prompt_tokens
                .min(options.context_tokens as usize),
            ..options.limits
        };
        if !limits.is_valid() {
            return Err(TextError::InvalidInput(
                "text preprocessing bounds and context length must be nonzero",
            ));
        }
        let processor = Arc::new(GgufProcessor::new(tokenizer, limits));
        let executor = QwenCuda::load_gguf(path, options.into()).map_err(TextError::Execution)?;
        let memory = executor.memory_report();
        let engine = Engine::with_defaults(executor).map_err(TextError::Engine)?;
        let driver = DriverConfig::for_engine(&engine);
        let (shutdown, handle) = Driver::spawn(engine, driver).map_err(TextError::Owner)?;
        let owner = Self::new(processor, shutdown, handle, options.text)?;
        Ok((owner, memory))
    }
}
