use crate::{EngineError, ExecutorInfo};

/// Resource bounds for the engine, fixed for its lifetime. Model-owned device
/// and host allocation limits are resolved by the prepared implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineConfig {
    pub max_active_requests: usize,
    /// Additional capacity beyond active requests. The total resident request
    /// bound is active plus queued; requests may wait before the first step.
    pub max_queued_requests: usize,
    pub max_queued_input_tokens: u64,
    pub max_buffered_events: usize,
    /// One slow consumer cannot consume more than this share of the aggregate
    /// event capacity. Includes reserved in-flight output and terminal events.
    pub max_events_per_request: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_active_requests: 8,
            max_queued_requests: 64,
            max_queued_input_tokens: 1_048_576,
            max_buffered_events: 1024,
            max_events_per_request: 64,
        }
    }
}

/// Live scheduling policy. Updating it does not change request semantics or
/// invalidate in-flight work; completions are checked against their saved batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulePolicy {
    pub max_batch_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub decode_tokens: u32,
    /// Force admitted prefill after this many decode-only submissions.
    /// This is a step bound, not a wall-clock latency guarantee.
    pub max_decode_only_steps: u32,
}

impl Default for SchedulePolicy {
    fn default() -> Self {
        Self {
            max_batch_tokens: 128,
            prefill_chunk_tokens: 16,
            decode_tokens: 1,
            max_decode_only_steps: 8,
        }
    }
}

pub(crate) fn validate_policy(
    policy: SchedulePolicy,
    info: &ExecutorInfo,
) -> Result<(), EngineError> {
    if policy.max_batch_tokens == 0
        || policy.prefill_chunk_tokens == 0
        || policy.decode_tokens == 0
        || policy.max_decode_only_steps == 0
        || policy.max_batch_tokens > info.limits.max_batch_tokens
        || policy.decode_tokens > info.limits.max_decode_tokens
    {
        return Err(EngineError::InvalidConfig);
    }
    Ok(())
}
