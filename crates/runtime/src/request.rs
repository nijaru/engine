use std::sync::Arc;

use crate::ExecutionError;

/// User-visible identity. A sequence has a separate, model-facing identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(pub(crate) u64);

impl RequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Output semantics, independent of scheduling and kernel selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        }
    }
}

impl Sampling {
    pub(crate) fn validate(self) -> Result<(), ExecutionError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ExecutionError::new(
                "temperature must be finite and nonnegative",
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(ExecutionError::new("top_p must be in (0, 1]"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationOptions {
    pub max_output_tokens: u32,
    pub sampling: Sampling,
    /// Stop token IDs are model/frontend semantics, not backend kernel policy.
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: 32,
            sampling: Sampling::default(),
            stop_tokens: Vec::new(),
        }
    }
}

/// Already encoded input. Chat templates and modality preprocessing stay above
/// this token-generation interface, outside the scheduler.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenRequest {
    pub tokens: Arc<[u32]>,
    pub options: GenerationOptions,
}

impl TokenRequest {
    #[must_use]
    pub fn new(tokens: impl Into<Arc<[u32]>>, options: GenerationOptions) -> Self {
        Self {
            tokens: tokens.into(),
            options,
        }
    }
}

/// Exact committed token accounting for a terminal request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    #[must_use]
    pub const fn total_tokens(self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
    Cancelled,
    Failed(ExecutionError),
}

/// Each request emits ordered tokens followed by exactly one terminal event.
/// An event remains deliverable after its physical sequence has been released.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    Token {
        request: RequestId,
        token: u32,
    },
    Finished {
        request: RequestId,
        reason: FinishReason,
        usage: Usage,
    },
}

impl Event {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        match self {
            Self::Token { request, .. } | Self::Finished { request, .. } => *request,
        }
    }
}
