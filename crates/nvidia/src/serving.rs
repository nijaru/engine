use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaStream;
use engine_core::{
    BackendError, ExecutionBatch, ExecutionMetrics, ExecutionOutcome, ExecutionPhase,
    ExecutionPlan, ExecutionSegment, InferenceStateSet, NvidiaDispatcher, WeightBinding,
};

use crate::decode::CudaQwen35Decode;
use crate::state::CudaStateRegistry;

/// Correctness-first Qwen3.8 CUDA serving dispatcher.
///
/// The scheduler submits a whole multi-request batch. This implementation
/// initially executes its entries sequentially with one batch-1 Qwen executor;
/// that is intentional. It validates persistent state and serving semantics
/// before true batched kernels or overlapping streams are introduced.
///
/// The dispatcher is still the synchronous compatibility path. A successful
/// dispatch therefore means all queued CUDA work for the submitted batch has
/// completed. Output-producing decode steps already synchronize through token
/// readback; output-free prefill requires an explicit batch-boundary stream
/// synchronization so core does not commit/reclaim logical state while device
/// kernels still own it.
pub struct CudaQwen35ServingDispatcher {
    executor: CudaQwen35Decode,
    states: CudaStateRegistry,
    stream: Arc<CudaStream>,
}

impl CudaQwen35ServingDispatcher {
    #[must_use]
    pub fn new(executor: CudaQwen35Decode, stream: Arc<CudaStream>) -> Self {
        Self {
            executor,
            states: CudaStateRegistry::new(stream.clone()),
            stream,
        }
    }

    #[must_use]
    pub const fn state_registry(&self) -> &CudaStateRegistry {
        &self.states
    }

    fn synchronize_completion(&self) -> Result<(), BackendError> {
        self.stream
            .synchronize()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
    }

    fn dispatch_segment(
        &mut self,
        segment: &ExecutionSegment,
        state: &InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        if let Some(sampling) = segment.sampling()
            && sampling.temperature() != 0.0
        {
            return Err(BackendError::Unsupported(
                "non-greedy Qwen3.8 CUDA sampling on the correctness path",
            ));
        }

        let physical = self
            .states
            .get_or_create(state)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        let started = Instant::now();
        let mut output_token = None;

        match segment.phase() {
            ExecutionPhase::Prefill => {
                let tokens = segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::prompt_slice)
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed(
                            "Qwen prefill segment requires prompt token input".to_owned(),
                        )
                    })?;
                for (offset, &token) in tokens.iter().enumerate() {
                    let offset = u32::try_from(offset).map_err(|_| {
                        BackendError::ExecutionFailed("prefill offset overflowed".to_owned())
                    })?;
                    let position =
                        segment
                            .state_position()
                            .checked_add(offset)
                            .ok_or_else(|| {
                                BackendError::ExecutionFailed(
                                    "prefill position overflowed".to_owned(),
                                )
                            })?;
                    let requests_output =
                        segment.requests_sampling() && offset + 1 == segment.token_count();
                    if requests_output {
                        let chosen = self
                            .executor
                            .decode_step(physical, token, position)
                            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                        output_token = Some(chosen);
                    } else {
                        self.executor
                            .prefill_step(physical, token, position)
                            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                    }
                    let next = position.checked_add(1).ok_or_else(|| {
                        BackendError::ExecutionFailed("prefill position overflowed".to_owned())
                    })?;
                    physical
                        .advance_to(next)
                        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                }
            }
            ExecutionPhase::Decode => {
                let token = segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::decode_token)
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed(
                            "Qwen decode segment requires one decode token".to_owned(),
                        )
                    })?;
                let chosen = self
                    .executor
                    .decode_step(physical, token, segment.state_position())
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                let next = segment.state_position().checked_add(1).ok_or_else(|| {
                    BackendError::ExecutionFailed("decode position overflowed".to_owned())
                })?;
                physical
                    .advance_to(next)
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                if segment.requests_sampling() {
                    output_token = Some(chosen);
                }
            }
            ExecutionPhase::SpecDraft
            | ExecutionPhase::SpecVerify
            | ExecutionPhase::Encoder
            | ExecutionPhase::MoEExpert => {
                return Err(BackendError::Unsupported(
                    "this Qwen3.8 CUDA serving execution phase",
                ));
            }
        }

        let elapsed_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let outcome = ExecutionOutcome::new(ExecutionMetrics::new(elapsed_nanos, 0, 0));
        Ok(match output_token {
            Some(token) => outcome.with_output_token(token),
            None => outcome,
        })
    }

    fn finish_synchronous<T>(
        &self,
        result: Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        match (result, self.synchronize_completion()) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(sync_error)) => Err(sync_error),
            (Err(error), Err(sync_error)) => Err(BackendError::ExecutionFailed(format!(
                "{error}; CUDA completion synchronization also failed: {sync_error}"
            ))),
        }
    }
}

impl NvidiaDispatcher for CudaQwen35ServingDispatcher {
    fn dispatch(
        &mut self,
        _plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        _weights: &WeightBinding,
        state: &mut InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        let result = self.dispatch_segment(segment, state);
        self.finish_synchronous(result)
    }

    fn dispatch_batch(
        &mut self,
        _plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        _weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Vec<ExecutionOutcome>, BackendError> {
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        let result = batch
            .segments()
            .iter()
            .zip(states)
            .map(|(segment, state)| self.dispatch_segment(segment, state))
            .collect();
        self.finish_synchronous(result)
    }

    fn release_inference_state(&mut self, state: &InferenceStateSet) -> Result<(), BackendError> {
        self.states.release(state);
        Ok(())
    }
}
