//! Request identity and semantic parameters.

use std::fmt;

use crate::model::ModelId;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ThinkingMode {
    On,
    Off,
    Automatic,
}

/// Controls whether a tokenizer adds model boundary tokens to a plain-text
/// prompt. Special-token parsing remains an explicit adapter operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpecialTokenPolicy {
    None,
    AddBos,
    AddEos,
    AddBosAndEos,
}

/// Selects the prompt representation requested by the caller.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PromptFormat {
    PlainText,
    EmbeddedChatTemplate,
}

/// Semantic prompt encoding policy. It is part of request identity because
/// changing it can change the token sequence and model output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PromptPolicy {
    format: PromptFormat,
    special_tokens: SpecialTokenPolicy,
}

impl PromptPolicy {
    #[must_use]
    pub const fn new(format: PromptFormat, special_tokens: SpecialTokenPolicy) -> Self {
        Self {
            format,
            special_tokens,
        }
    }

    #[must_use]
    pub const fn plain_text() -> Self {
        Self::new(PromptFormat::PlainText, SpecialTokenPolicy::None)
    }

    #[must_use]
    pub const fn format(self) -> PromptFormat {
        self.format
    }

    #[must_use]
    pub const fn special_tokens(self) -> SpecialTokenPolicy {
        self.special_tokens
    }
}

/// Sampling settings are semantic: changing them can change model output and
/// therefore they do not belong in a performance-policy snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    seed: Option<u64>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
}

impl SamplingParams {
    /// # Errors
    ///
    /// Returns [`SamplingError`] when temperature or top-p is outside its
    /// valid range.
    pub fn new(
        seed: Option<u64>,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> Result<Self, SamplingError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature);
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplingError::InvalidTopP);
        }
        Ok(Self {
            seed,
            temperature,
            top_p,
            top_k,
        })
    }

    #[must_use]
    pub fn greedy(seed: Option<u64>) -> Self {
        Self {
            seed,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        }
    }

    #[must_use]
    pub const fn seed(self) -> Option<u64> {
        self.seed
    }

    #[must_use]
    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    #[must_use]
    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    #[must_use]
    pub const fn top_k(self) -> u32 {
        self.top_k
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplingError {
    InvalidTemperature,
    InvalidTopP,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature => f.write_str("temperature must be finite and non-negative"),
            Self::InvalidTopP => f.write_str("top_p must be finite and in the range (0, 1]"),
        }
    }
}

impl std::error::Error for SamplingError {}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestSemantics {
    max_output_tokens: u32,
    sampling: SamplingParams,
    thinking: ThinkingMode,
    prompt_policy: PromptPolicy,
}

impl RequestSemantics {
    /// # Errors
    ///
    /// Returns [`RequestError::ZeroOutputBudget`] when no output is allowed.
    pub fn new(
        max_output_tokens: u32,
        sampling: SamplingParams,
        thinking: ThinkingMode,
    ) -> Result<Self, RequestError> {
        if max_output_tokens == 0 {
            return Err(RequestError::ZeroOutputBudget);
        }
        Ok(Self {
            max_output_tokens,
            sampling,
            thinking,
            prompt_policy: PromptPolicy::plain_text(),
        })
    }

    #[must_use]
    pub const fn with_prompt_policy(mut self, prompt_policy: PromptPolicy) -> Self {
        self.prompt_policy = prompt_policy;
        self
    }

    #[must_use]
    pub const fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }

    #[must_use]
    pub const fn sampling(&self) -> SamplingParams {
        self.sampling
    }

    #[must_use]
    pub const fn thinking(&self) -> ThinkingMode {
        self.thinking
    }

    #[must_use]
    pub const fn prompt_policy(&self) -> PromptPolicy {
        self.prompt_policy
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RequestError {
    ZeroOutputBudget,
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroOutputBudget => f.write_str("max_output_tokens must be greater than zero"),
        }
    }
}

impl std::error::Error for RequestError {}

/// A request contains semantic identity and behavior only. Queueing, batching,
/// placement, and speculation budgets are supplied by runtime policy.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestSpec {
    id: RequestId,
    model: ModelId,
    semantics: RequestSemantics,
}

impl RequestSpec {
    #[must_use]
    pub const fn new(id: RequestId, model: ModelId, semantics: RequestSemantics) -> Self {
        Self {
            id,
            model,
            semantics,
        }
    }

    #[must_use]
    pub const fn id(&self) -> RequestId {
        self.id
    }

    #[must_use]
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    #[must_use]
    pub const fn semantics(&self) -> &RequestSemantics {
        &self.semantics
    }
}
