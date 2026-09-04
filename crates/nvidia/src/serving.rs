use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, PinnedHostSlice};
use engine_core::{
    BackendError, BackendSubmissionId, ExecutionBatch, ExecutionMetrics, ExecutionOutcome,
    ExecutionPhase, ExecutionPlan, ExecutionSegment, InferenceStateSet, NvidiaDispatcher,
    WeightBinding,
};

use crate::decode::CudaQwen35Decode;
use crate::state::{CudaHybridState, CudaStateKey, CudaStateRegistry};

/// Which dispatcher-owned pinned output slot backs one scheduler row.
#[derive(Clone, Copy)]
struct PinnedSlotLease {
    index: usize,
}

/// Retained resources for one asynchronously in-flight scheduler batch.
struct PendingSubmission {
    /// Recorded after every kernel and pinned copy of this submission, in
    /// stream order.
    completion: CudaEvent,
    /// One leased output slot per row, in scheduler order. Rows that did not
    /// request sampling hold `None`, keeping row order aligned with batch
    /// order.
    outputs: Vec<Option<PinnedSlotLease>>,
    /// Per-row execution wall time captured at submit.
    elapsed_nanos: Vec<u64>,
    /// Logical state keys whose physical registry entries this submission's
    /// queued work still accesses. Retained until completion so an early
    /// `release_inference_state` cannot free device state under live work.
    state_keys: Vec<CudaStateKey>,
}

/// Row results collected while queueing one scheduler batch.
struct QueuedRows {
    outputs: Vec<Option<PinnedSlotLease>>,
    elapsed_nanos: Vec<u64>,
    state_keys: Vec<CudaStateKey>,
}

/// One-stream asynchronous Qwen3.8 CUDA serving dispatcher.
///
/// The scheduler submits a whole multi-request batch. Rows still execute
/// sequentially through the batch-1 executor on the single serving stream,
/// but the host is no longer blocked on the result: rows that request
/// sampling copy their greedy selection into dispatcher-owned pinned host
/// slots, one completion event is recorded per submission, and outcomes are
/// published later through [`NvidiaDispatcher::poll_batch`] once the event
/// completes.
///
/// Scratch reuse stays safe because all launches are stream-ordered: a later
/// row or submission cannot observe an earlier row's scratch or state until
/// the stream has finished that work.
///
/// Error contract, matching the `NvidiaDispatcher` seam:
/// - a `submit_batch` error means no asynchronous ownership was retained: the
///   stream is flushed so partial launches cannot touch request state or
///   scratch after the error returns, and every leased slot is recycled;
/// - `poll_batch` returns `Ok(None)` while the completion event is pending.
///   A device fault observed through the event is terminal: the pending
///   record (event, leases, retained state keys) is dropped before the error
///   is returned, ending this dispatcher's access to those resources.
pub struct CudaQwen35ServingDispatcher {
    executor: CudaQwen35Decode,
    states: CudaStateRegistry,
    stream: Arc<CudaStream>,
    /// Reusable pinned one-`u32` output slots.
    pinned_outputs: Vec<PinnedHostSlice<u32>>,
    free_outputs: Vec<usize>,
    pending: HashMap<BackendSubmissionId, PendingSubmission>,
    /// State keys released while still referenced by pending submissions.
    /// Their physical entries stay alive until the referencing submission
    /// completes, then they are released for real.
    deferred_releases: Vec<CudaStateKey>,
}

/// Run one scheduler row's model steps.
///
/// `step` performs the one model step that produces the row's output token
/// (the final prefill token or the decode token): eager callers read the
/// token back immediately; asynchronous callers stream-order the copy into a
/// pinned slot and return `None`. All other prefill tokens run through the
/// output-free `prefill_step`.
///
/// Returns the row's sampled token when the step produced one, plus the
/// wall-clock time spent queueing the row.
fn run_row(
    executor: &mut CudaQwen35Decode,
    registry: &mut CudaStateRegistry,
    segment: &ExecutionSegment,
    state: &InferenceStateSet,
    mut step: impl FnMut(
        &mut CudaQwen35Decode,
        &mut CudaHybridState,
        u32,
        u32,
    ) -> Result<Option<u32>, BackendError>,
) -> Result<(Option<u32>, u64), BackendError> {
    if let Some(sampling) = segment.sampling()
        && sampling.temperature() != 0.0
    {
        return Err(BackendError::Unsupported(
            "non-greedy Qwen3.8 CUDA sampling on the correctness path",
        ));
    }

    let physical = registry
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
                let position = segment
                    .state_position()
                    .checked_add(offset)
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed("prefill position overflowed".to_owned())
                    })?;
                let requests_output =
                    segment.requests_sampling() && offset + 1 == segment.token_count();
                if requests_output {
                    output_token = step(executor, physical, token, position)?;
                } else {
                    executor
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
            output_token = step(executor, physical, token, segment.state_position())?;
            let next = segment.state_position().checked_add(1).ok_or_else(|| {
                BackendError::ExecutionFailed("decode position overflowed".to_owned())
            })?;
            physical
                .advance_to(next)
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
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
    Ok((output_token, elapsed_nanos))
}

impl CudaQwen35ServingDispatcher {
    /// Build the dispatcher with a pool of `max_pending_rows` pinned output
    /// slots. The pool bounds the number of sampling rows that can be
    /// simultaneously in flight across all live submissions.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when the pinned pool cannot
    /// be allocated or is empty.
    pub fn new(
        context: &Arc<CudaContext>,
        executor: CudaQwen35Decode,
        stream: Arc<CudaStream>,
        max_pending_rows: usize,
    ) -> Result<Self, BackendError> {
        if max_pending_rows == 0 {
            return Err(BackendError::ExecutionFailed(
                "the serving dispatcher requires at least one pinned output slot".to_owned(),
            ));
        }
        let mut pinned_outputs = Vec::with_capacity(max_pending_rows);
        for _ in 0..max_pending_rows {
            // Safety: a slot is written only by the stream-ordered copy of
            // the submission that leased it, and is read only after that
            // submission's completion event covers the copy. Slots in the
            // free pool are never read.
            let slot = unsafe { context.alloc_pinned::<u32>(1) }
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            pinned_outputs.push(slot);
        }
        Ok(Self {
            executor,
            states: CudaStateRegistry::new(stream.clone()),
            stream,
            pinned_outputs,
            free_outputs: (0..max_pending_rows).rev().collect(),
            pending: HashMap::new(),
            deferred_releases: Vec::new(),
        })
    }

    #[must_use]
    pub const fn state_registry(&self) -> &CudaStateRegistry {
        &self.states
    }

    #[must_use]
    pub fn pending_submissions(&self) -> usize {
        self.pending.len()
    }

    fn lease_output(&mut self) -> Result<PinnedSlotLease, BackendError> {
        let index = self.free_outputs.pop().ok_or_else(|| {
            BackendError::ExecutionFailed(
                "pinned Qwen output slots exhausted; too many concurrent pending rows".to_owned(),
            )
        })?;
        Ok(PinnedSlotLease { index })
    }

    fn recycle_output(&mut self, lease: PinnedSlotLease) {
        self.free_outputs.push(lease.index);
    }

    /// Query one submission's completion event. `is_complete` swallows the
    /// distinction between "not ready" and "device fault", so the raw query
    /// re-distinguishes them: only `NOT_READY` maps to pending.
    fn completion_is_ready(event: &CudaEvent) -> Result<bool, BackendError> {
        if event.is_complete() {
            return Ok(true);
        }
        match unsafe { cudarc::driver::result::event::query(event.cu_event()) } {
            Ok(()) => Ok(true),
            Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                Ok(false)
            }
            Err(error) => Err(BackendError::ExecutionFailed(error.to_string())),
        }
    }

    /// Eager per-row execution: run the row through the blocking decode step
    /// and synchronize the stream, so no device work outlives the call.
    fn dispatch_segment_eager(
        &mut self,
        segment: &ExecutionSegment,
        state: &InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        let Self {
            executor,
            states: registry,
            ..
        } = self;
        let (output_token, elapsed_nanos) = run_row(
            executor,
            registry,
            segment,
            state,
            |executor, physical, token, position| {
                executor
                    .decode_step(physical, token, position)
                    .map(Some)
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
            },
        )?;
        self.stream
            .synchronize()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        let outcome = ExecutionOutcome::new(ExecutionMetrics::new(elapsed_nanos, 0, 0));
        Ok(match output_token {
            Some(token) => outcome.with_output_token(token),
            None => outcome,
        })
    }

    /// Queue every row of one scheduler batch without blocking the host.
    ///
    /// On success, all row work and pinned output copies are enqueued on the
    /// serving stream. On failure the stream is flushed and every leased slot
    /// is recycled before the error is returned, leaving no retained
    /// asynchronous ownership.
    fn enqueue_batch(
        &mut self,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<QueuedRows, BackendError> {
        let row_count = batch.len();
        let mut outputs = Vec::with_capacity(row_count);
        let mut elapsed_nanos = Vec::with_capacity(row_count);
        let mut state_keys = Vec::with_capacity(row_count);
        let mut leased: Vec<PinnedSlotLease> = Vec::new();

        let enqueue_result = (|| -> Result<(), BackendError> {
            for (segment, state) in batch.segments().iter().zip(states.iter_mut()) {
                let lease = if segment.requests_sampling() {
                    let lease = self.lease_output()?;
                    leased.push(lease);
                    Some(lease)
                } else {
                    None
                };
                // Disjoint field borrows: the executor, the state registry,
                // and the leased pinned slot are independent of each other,
                // so one row can hold all three at once.
                let Self {
                    executor,
                    states: registry,
                    pinned_outputs,
                    ..
                } = self;
                let pinned = lease
                    .map(|lease| {
                        pinned_outputs.get_mut(lease.index).ok_or_else(|| {
                            BackendError::ExecutionFailed(
                                "pinned Qwen output slot was out of range".to_owned(),
                            )
                        })
                    })
                    .transpose()?;
                let (_, row_elapsed) = if let Some(pinned) = pinned {
                    run_row(
                        executor,
                        registry,
                        segment,
                        state,
                        |executor, physical, token, position| {
                            executor
                                .decode_step_into_pinned(physical, token, position, pinned)
                                .map(|()| None)
                                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
                        },
                    )?
                } else {
                    run_row(
                        executor,
                        registry,
                        segment,
                        state,
                        |executor, physical, token, position| {
                            executor
                                .prefill_step(physical, token, position)
                                .map(|()| None)
                                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
                        },
                    )?
                };
                elapsed_nanos.push(row_elapsed);
                outputs.push(lease);
                state_keys.push(CudaStateKey::from_state_set(state));
            }
            Ok(())
        })();

        if let Err(error) = enqueue_result {
            self.stream.synchronize().map_err(|sync_error| {
                BackendError::ExecutionFailed(format!(
                    "{error}; CUDA flush after enqueue failure also failed: {sync_error}"
                ))
            })?;
            for lease in leased {
                self.recycle_output(lease);
            }
            return Err(error);
        }

        Ok(QueuedRows {
            outputs,
            elapsed_nanos,
            state_keys,
        })
    }

    /// Read one leased pinned output slot. Safe only after the owning
    /// submission's completion event covers the stream-ordered copy into the
    /// slot; `as_slice` additionally synchronizes the slot's own event.
    fn read_pinned_output(&self, lease: PinnedSlotLease) -> Result<u32, BackendError> {
        let slot = self
            .pinned_outputs
            .get(lease.index)
            .ok_or_else(|| {
                BackendError::ExecutionFailed("pinned output lease was out of range".to_owned())
            })?
            .as_slice()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        slot.first()
            .copied()
            .ok_or_else(|| BackendError::ExecutionFailed("pinned output slot was empty".to_owned()))
    }

    /// Release every deferred state key no longer referenced by any pending
    /// submission.
    fn drain_deferred_releases(&mut self) {
        let deferred = std::mem::take(&mut self.deferred_releases);
        for key in deferred {
            let still_referenced = self
                .pending
                .values()
                .any(|pending| pending.state_keys.contains(&key));
            if still_referenced {
                self.deferred_releases.push(key);
            } else {
                self.states.release_key(&key);
            }
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
        self.dispatch_segment_eager(segment, state)
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
        let outcomes = batch
            .segments()
            .iter()
            .zip(states.iter_mut())
            .map(|(segment, state)| self.dispatch_segment_eager(segment, state))
            .collect::<Result<Vec<_>, _>>();
        let sync = self
            .stream
            .synchronize()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()));
        match (outcomes, sync) {
            (Ok(values), Ok(())) => Ok(values),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(sync_error)) => Err(sync_error),
            (Err(error), Err(sync_error)) => Err(BackendError::ExecutionFailed(format!(
                "{error}; CUDA completion synchronization also failed: {sync_error}"
            ))),
        }
    }

    fn submit_batch(
        &mut self,
        submission: BackendSubmissionId,
        _plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        _weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        if self.pending.contains_key(&submission) {
            return Err(BackendError::ExecutionFailed(format!(
                "duplicate asynchronous submission {submission:?}"
            )));
        }

        let rows = self.enqueue_batch(batch, states)?;

        // One completion event for the whole submission: every kernel and
        // pinned copy enqueued above precedes it in stream order.
        let completion = match self.executor.record_completion_event() {
            Ok(event) => event,
            Err(error) => {
                // The queued work cannot be orphaned behind a failed event:
                // flush and recycle before reporting the enqueue failure.
                let flush = self.stream.synchronize().map_err(|sync_error| {
                    BackendError::ExecutionFailed(format!(
                        "event recording failed: {error}; CUDA flush also failed: {sync_error}"
                    ))
                });
                for lease in rows.outputs.iter().filter_map(|lease| *lease) {
                    self.recycle_output(lease);
                }
                flush?;
                return Err(BackendError::ExecutionFailed(error.to_string()));
            }
        };

        self.pending.insert(
            submission,
            PendingSubmission {
                completion,
                outputs: rows.outputs,
                elapsed_nanos: rows.elapsed_nanos,
                state_keys: rows.state_keys,
            },
        );
        Ok(None)
    }

    fn poll_batch(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        let ready = {
            let Some(pending) = self.pending.get(&submission) else {
                return Err(BackendError::UnknownSubmission(submission));
            };
            Self::completion_is_ready(&pending.completion)?
        };
        if !ready {
            return Ok(None);
        }

        // The completion event covers every kernel and pinned copy of this
        // submission; taking the record ends async ownership of its leases
        // and retained state keys even on the error paths below. The leases
        // are recycled before outcome construction so a read failure cannot
        // strand pool slots.
        let pending = self
            .pending
            .remove(&submission)
            .expect("presence was checked above");
        for lease in pending.outputs.iter().filter_map(|lease| *lease) {
            self.recycle_output(lease);
        }

        let mut outcomes = Vec::with_capacity(pending.outputs.len());
        for (row_elapsed, lease) in pending.elapsed_nanos.iter().zip(pending.outputs.iter()) {
            let outcome = ExecutionOutcome::new(ExecutionMetrics::new(*row_elapsed, 0, 0));
            let outcome = match lease {
                Some(lease) => {
                    let token = self.read_pinned_output(*lease)?;
                    outcome.with_output_token(token)
                }
                None => outcome,
            };
            outcomes.push(outcome);
        }
        drop(pending);
        self.drain_deferred_releases();

        Ok(Some(outcomes))
    }

    fn release_inference_state(&mut self, state: &InferenceStateSet) -> Result<(), BackendError> {
        // Physical release while queued device work still accesses this
        // state is deferred to the referencing submission's completion.
        let key = CudaStateKey::from_state_set(state);
        let referenced = self
            .pending
            .values()
            .any(|pending| pending.state_keys.contains(&key));
        if referenced {
            self.deferred_releases.push(key);
        } else {
            self.states.release_key(&key);
        }
        Ok(())
    }
}
