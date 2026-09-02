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

    /// Select one token from a logits vector using the request's sampling
    /// parameters. The caller supplies one uniform random draw so the core
    /// remains dependency-free and the runtime can own RNG state.
    ///
    /// For greedy sampling (`temperature == 0`) `random_unit` is ignored.
    /// Ties are resolved by the lowest token ID. For stochastic sampling,
    /// `seed()` identifies the caller's RNG stream but does not itself mutate
    /// or generate randomness.
    ///
    /// # Errors
    ///
    /// Returns [`SamplingError`] when logits are empty/non-finite or the random
    /// draw is outside `[0, 1)`.
    pub fn select_token(self, logits: &[f32], random_unit: f32) -> Result<u32, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(SamplingError::NonFiniteLogit);
        }
        if self.temperature == 0.0 {
            let mut best_index = 0;
            let mut best_value = logits[0];
            for (index, &value) in logits.iter().enumerate().skip(1) {
                if value > best_value {
                    best_index = index;
                    best_value = value;
                }
            }
            return u32::try_from(best_index).map_err(|_| SamplingError::TokenIndexOverflow);
        }
        if !random_unit.is_finite() || !(0.0..1.0).contains(&random_unit) {
            return Err(SamplingError::InvalidRandomUnit);
        }
        let mut candidates = logits
            .iter()
            .enumerate()
            .map(|(index, &logit)| {
                u32::try_from(index)
                    .map(|token| (token, logit / self.temperature))
                    .map_err(|_| SamplingError::TokenIndexOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        candidates.sort_by(|(left_id, left), (right_id, right)| {
            right
                .partial_cmp(left)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_id.cmp(right_id))
        });
        if self.top_k > 0 {
            candidates.truncate(usize::try_from(self.top_k).unwrap_or(usize::MAX));
        }
        let Some((_, max_logit)) = candidates.first().copied() else {
            return Err(SamplingError::EmptyLogits);
        };
        let mut weighted = candidates
            .into_iter()
            .map(|(token, logit)| (token, (logit - max_logit).exp()))
            .collect::<Vec<_>>();
        let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
        let mut cumulative = 0.0;
        let mut cutoff = weighted.len();
        for (index, (_, weight)) in weighted.iter().enumerate() {
            cumulative += *weight / total;
            if cumulative >= self.top_p {
                cutoff = index + 1;
                break;
            }
        }
        weighted.truncate(cutoff);
        let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
        let target = random_unit * total;
        let mut cumulative = 0.0;
        for (token, weight) in weighted {
            cumulative += weight;
            if cumulative > target {
                return Ok(token);
            }
        }
        Err(SamplingError::InvalidRandomUnit)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplingError {
    InvalidTemperature,
    InvalidTopP,
    EmptyLogits,
    NonFiniteLogit,
    InvalidRandomUnit,
    TokenIndexOverflow,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature => f.write_str("temperature must be finite and non-negative"),
            Self::InvalidTopP => f.write_str("top_p must be finite and in the range (0, 1]"),
            Self::EmptyLogits => f.write_str("logits must contain at least one value"),
            Self::NonFiniteLogit => f.write_str("logits must contain only finite values"),
            Self::InvalidRandomUnit => {
                f.write_str("random draw must be finite and in the range [0, 1)")
            }
            Self::TokenIndexOverflow => f.write_str("token index does not fit in u32"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_selection_uses_lowest_id_for_ties() {
        let params = SamplingParams::greedy(Some(7));
        assert_eq!(params.select_token(&[1.0, 3.0, 3.0], 0.75), Ok(1));
    }

    #[test]
    fn stochastic_selection_applies_top_k_and_top_p() {
        let params = SamplingParams::new(Some(11), 1.0, 0.9, 0).expect("sampling parameters");
        assert_eq!(params.select_token(&[0.0, -1.0, -2.0], 0.0), Ok(0));
        assert_eq!(params.select_token(&[0.0, -1.0, -2.0], 0.8), Ok(1));

        let top_k = SamplingParams::new(None, 1.0, 1.0, 1).expect("top-k parameters");
        assert_eq!(top_k.select_token(&[-3.0, -1.0, -2.0], 0.99), Ok(1));
    }

    #[test]
    fn sampling_rejects_invalid_logits_and_random_draws() {
        let params = SamplingParams::new(None, 1.0, 1.0, 0).expect("sampling parameters");
        assert_eq!(
            params.select_token(&[], 0.0),
            Err(SamplingError::EmptyLogits)
        );
        assert_eq!(
            params.select_token(&[f32::NAN], 0.0),
            Err(SamplingError::NonFiniteLogit)
        );
        assert_eq!(
            params.select_token(&[1.0], 1.0),
            Err(SamplingError::InvalidRandomUnit)
        );
    }
}
