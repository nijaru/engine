//! Artifact-independent Qwen hybrid model dimensions.
//!
//! Checkpoint adapters normalize into this structure. Validation establishes
//! model geometry, not support by any particular device executor.

use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenConfig {
    pub context_length: u64,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub block_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub full_attention_interval: u32,
    pub ssm_group_count: u32,
    pub ssm_inner_size: u32,
    pub ssm_state_size: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_conv_kernel: u32,
    pub nextn_predict_layers: u32,
}

impl QwenConfig {
    #[must_use]
    pub const fn context_length(&self) -> u64 {
        self.context_length
    }

    #[must_use]
    pub const fn embedding_length(&self) -> u32 {
        self.embedding_length
    }

    #[must_use]
    pub const fn feed_forward_length(&self) -> u32 {
        self.feed_forward_length
    }

    #[must_use]
    pub const fn block_count(&self) -> u32 {
        self.block_count
    }

    #[must_use]
    pub const fn attention_heads(&self) -> u32 {
        self.attention_heads
    }

    #[must_use]
    pub const fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    #[must_use]
    pub const fn key_length(&self) -> u32 {
        self.key_length
    }

    #[must_use]
    pub const fn value_length(&self) -> u32 {
        self.value_length
    }

    #[must_use]
    pub const fn full_attention_interval(&self) -> u32 {
        self.full_attention_interval
    }

    #[must_use]
    pub const fn ssm_group_count(&self) -> u32 {
        self.ssm_group_count
    }

    #[must_use]
    pub const fn ssm_inner_size(&self) -> u32 {
        self.ssm_inner_size
    }

    #[must_use]
    pub const fn ssm_state_size(&self) -> u32 {
        self.ssm_state_size
    }

    #[must_use]
    pub const fn ssm_time_step_rank(&self) -> u32 {
        self.ssm_time_step_rank
    }

    #[must_use]
    pub const fn ssm_conv_kernel(&self) -> u32 {
        self.ssm_conv_kernel
    }

    #[must_use]
    pub const fn nextn_predict_layers(&self) -> u32 {
        self.nextn_predict_layers
    }

    #[must_use]
    pub const fn language_layer_count(&self) -> Option<u32> {
        self.block_count.checked_sub(self.nextn_predict_layers)
    }

    /// Classify a language layer independently of serialized checkpoint names.
    ///
    /// # Errors
    /// Rejects invalid geometry and indices outside the language model.
    pub fn layer_kind(&self, layer: u32) -> Result<QwenLayerKind, ConfigError> {
        self.validate()?;
        if layer
            >= self
                .language_layer_count()
                .ok_or(ConfigError("invalid language layer count"))?
        {
            return Err(ConfigError("Qwen language layer index is out of range"));
        }
        Ok(
            if (layer + 1).is_multiple_of(self.full_attention_interval) {
                QwenLayerKind::FullAttention
            } else {
                QwenLayerKind::Recurrent
            },
        )
    }

    /// # Errors
    ///
    /// Returns [`ConfigError`] when a required Qwen3.8
    /// hybrid dimension is zero or inconsistent.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let dimensions = [
            self.context_length,
            u64::from(self.embedding_length),
            u64::from(self.feed_forward_length),
            u64::from(self.block_count),
            u64::from(self.attention_heads),
            u64::from(self.kv_heads),
            u64::from(self.key_length),
            u64::from(self.value_length),
            u64::from(self.full_attention_interval),
            u64::from(self.ssm_group_count),
            u64::from(self.ssm_inner_size),
            u64::from(self.ssm_state_size),
            u64::from(self.ssm_time_step_rank),
            u64::from(self.ssm_conv_kernel),
        ];
        if dimensions.contains(&0) {
            return Err(ConfigError("Qwen3.8 dimensions must be non-zero"));
        }
        let language_layers = self
            .language_layer_count()
            .ok_or(ConfigError("MTP layer count exceeds total block count"))?;
        if self.full_attention_interval > language_layers
            || language_layers % self.full_attention_interval != 0
            || !self.attention_heads.is_multiple_of(self.kv_heads)
            || self.ssm_time_step_rank.checked_mul(self.ssm_state_size) != Some(self.ssm_inner_size)
        {
            return Err(ConfigError("Qwen3.8 hybrid dimensions are inconsistent"));
        }
        Ok(())
    }
}

/// Invalid model geometry, independent of a checkpoint container or backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfigError(pub(crate) &'static str);

impl ConfigError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        self.0
    }
}
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ConfigError {}

/// Language-layer structure, independent of artifact and device representation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QwenLayerKind {
    Recurrent,
    FullAttention,
}
