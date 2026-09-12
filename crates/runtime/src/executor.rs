use std::fmt;

use crate::{RequestId, TokenRequest};

/// Engine-issued logical identity, not an allocation pointer or a cache layout.
/// Models must not infer state contents from this value or from its prefix.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SequenceId(pub(crate) u64);

impl SequenceId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SubmissionId(u64);

impl SubmissionId {
    /// The prepared model issues IDs unique among its live submissions.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Only distinctions used by the token scheduler belong here. Attention,
/// expert, encoder, and draft/verification algorithms are model internals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepKind {
    Prefill,
    Decode,
}

/// Bounds for scheduling, not a checklist of model architectures or features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationLimits {
    pub context_tokens: u32,
    pub max_sequences: usize,
    pub max_batch_tokens: u32,
    /// Maximum committed advancement a decode item can request. A model may
    /// return fewer tokens, but at least one, without exposing its draft method.
    pub max_decode_tokens: u32,
}

impl GenerationLimits {
    pub(crate) fn validate(self) -> Result<(), ExecutionError> {
        if self.context_tokens == 0
            || self.max_sequences == 0
            || self.max_batch_tokens == 0
            || self.max_decode_tokens == 0
        {
            return Err(ExecutionError::new("prepared model limits must be nonzero"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorInfo {
    /// Display identity only. Persistent cache/qualification keys must be
    /// established from the exact prepared implementation and artifact.
    pub name: String,
    pub limits: GenerationLimits,
}

/// A scalar scheduling record. It owns no physical state or transient tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchItem {
    pub sequence: SequenceId,
    pub kind: StepKind,
    pub prefix: u32,
    /// Exact prefill input length; maximum committed decode advancement.
    pub token_budget: u32,
    /// Zero for intermediate prefill, one for final prefill, and at most the
    /// remaining generation budget for decode. Not the number of draft tokens.
    pub output_budget: u32,
}

/// Completed work for one sequence. The prefix counts consumed model inputs;
/// sampled output is tracked separately. Rejected speculative work must not
/// appear in either committed field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepCompletion {
    pub sequence: SequenceId,
    pub prefix: u32,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Admission {
    Ready,
    /// No sequence resources were retained. The engine may retry admission.
    Deferred,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionError(String);

impl ExecutionError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ExecutionError {}

/// A loaded and prepared model/backend combination. Concrete implementations
/// retain strongly typed model state; the common runtime sees only sequences.
/// Registration is ordinary Rust composition, not a dynamic-library ABI.
///
/// The engine currently keeps one batch in flight. Implementations can use any
/// internal streams, devices, or stages, but must preserve the batch contract.
/// A method returning an error never transfers cleanup ownership to nobody:
/// the implementation must keep uncertain resources alive until `synchronize`
/// establishes completion, and must reject unsafe release.
pub trait GenerationExecutor: Send {
    /// Immutable scheduling metadata, resolved before the engine is created.
    fn info(&self) -> &ExecutorInfo;

    /// Validate request semantics and reserve the complete continuation bundle.
    /// `request_id` is stable for the runtime request and may correlate model-
    /// prepared inputs or tracing; `sequence` identifies executor continuation
    /// ownership.
    /// at prefix zero. `Deferred` or an error must retain no admission resources.
    /// Preparation, restore, fork, and migration need their own concrete proofs;
    /// an arbitrary nonzero logical prefix is not a restoration API.
    ///
    /// # Errors
    /// Returns an unsupported-input or resource error before execution.
    fn admit(
        &mut self,
        request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError>;

    /// Queue an accepted batch. Copy/retain everything needed after this call;
    /// references to the borrowed batch must not escape it. Logical prefixes
    /// advance only after a matching successful completion is returned.
    ///
    /// # Errors
    /// A submission error faults the engine. Retain resources if completion of
    /// partially queued work is uncertain; never silently retry mutated state.
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError>;

    /// Observe completion once, in batch order. Partial results stay internal.
    ///
    /// # Errors
    /// A terminal device/completion error faults the engine. Ownership remains
    /// with the prepared model until safe release or teardown.
    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError>;

    /// Free one completed sequence. Successful release must be idempotent.
    ///
    /// # Errors
    /// An unsuccessful release must retain enough ownership for a later retry.
    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError>;

    /// Establish completion of ALL queued work, including partial submissions
    /// and faulted batches. Used for explicit shutdown and defensive Drop.
    /// This may block; normal serving uses `poll`, not per-step synchronization.
    /// Shutdown may consume pending results to return resource leases; callers
    /// must not expect to resume polling those submissions after shutdown.
    ///
    /// # Errors
    /// On uncertain completion return an error and retain physical resources.
    fn synchronize(&mut self) -> Result<(), ExecutionError>;
}
