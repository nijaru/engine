use std::collections::HashMap;
use std::sync::Arc;

use engine_core::{
    BackendSubmissionId, ComputeBackend, ExecutionBatch, ExecutionBatchEvent, ExecutionPhase,
    ExecutionPlan, ExecutionSegment, ExecutionTokenInput, InferenceStateSet, LogicalStateManager,
    RequestId, SamplingParams, StateManager,
};
use ribn::{
    Admission, BatchItem, ExecutionError, ExecutorInfo, SequenceId, StepCompletion, StepKind,
    SubmissionId, TokenRequest,
};

use crate::state;

struct Sequence {
    state: Option<InferenceStateSet>,
    prompt: Option<Arc<[u32]>>,
    prompt_len: u32,
    next: Option<u32>,
    sampling: SamplingParams,
}

struct Pending {
    id: BackendSubmissionId,
    items: Vec<BatchItem>,
    states: Vec<InferenceStateSet>,
    batch: ExecutionBatch,
}

/// Transitional Qwen execution adapter, not another request scheduler.
/// Generic backend injection lets host tests exercise the real lease protocol.
pub(crate) struct QwenExecution<B> {
    pub(crate) backend: B,
    info: ExecutorInfo,
    plan: ExecutionPlan,
    manager: LogicalStateManager,
    sequences: HashMap<SequenceId, Sequence>,
    pending: Option<Pending>,
    faulted: bool,
    vocabulary_size: u32,
}

impl<B: ComputeBackend> QwenExecution<B> {
    pub(crate) fn new(
        backend: B,
        info: ExecutorInfo,
        plan: ExecutionPlan,
        manager: LogicalStateManager,
        vocabulary_size: u32,
    ) -> Self {
        Self {
            backend,
            plan,
            manager,
            sequences: HashMap::with_capacity(info.limits.max_sequences),
            info,
            pending: None,
            faulted: false,
            vocabulary_size,
        }
    }

    pub(crate) fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    pub(crate) fn admit(
        &mut self,
        id: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        if self.faulted {
            return Err(ExecutionError::new("Qwen execution is faulted"));
        }
        if self.sequences.contains_key(&id) {
            return Err(ExecutionError::new("duplicate sequence"));
        }
        let total = u32::try_from(request.tokens.len())
            .ok()
            .and_then(|tokens| tokens.checked_add(request.options.max_output_tokens));
        if request.tokens.is_empty()
            || request.options.max_output_tokens == 0
            || total.is_none_or(|tokens| tokens > self.info.limits.context_tokens)
            || request
                .tokens
                .iter()
                .chain(&request.options.stop_tokens)
                .any(|&token| token >= self.vocabulary_size)
        {
            return Err(ExecutionError::new(
                "Qwen input/output limits or token IDs are invalid",
            ));
        }
        if request.options.sampling.temperature != 0.0 {
            return Err(ExecutionError::new(
                "the experimental Qwen adapter currently supports greedy sampling only",
            ));
        }
        let sampling = SamplingParams::new(
            request.options.sampling.seed,
            request.options.sampling.temperature,
            request.options.sampling.top_p,
            request.options.sampling.top_k,
        )
        .map_err(model_error)?;
        if self.sequences.len() >= self.info.limits.max_sequences {
            return Ok(Admission::Deferred);
        }
        let state = state::allocate(&mut self.manager, self.plan.state_requirements())
            .map_err(model_error)?;
        self.sequences.insert(
            id,
            Sequence {
                state: Some(state),
                prompt: Some(Arc::clone(&request.tokens)),
                prompt_len: u32::try_from(request.tokens.len()).expect("validated prompt length"),
                next: None,
                sampling,
            },
        );
        Ok(Admission::Ready)
    }

    fn segment(&self, item: &BatchItem) -> Result<ExecutionSegment, ExecutionError> {
        let sequence = self
            .sequences
            .get(&item.sequence)
            .ok_or_else(|| ExecutionError::new("unknown Qwen sequence"))?;
        let state = sequence
            .state
            .as_ref()
            .ok_or_else(|| ExecutionError::new("Qwen sequence is in flight"))?;
        let end = item
            .prefix
            .checked_add(item.token_budget)
            .ok_or_else(|| ExecutionError::new("Qwen prefix overflow"))?;
        if item.token_budget == 0
            || state.token_position() != Some(item.prefix)
            || end > self.info.limits.context_tokens
        {
            return Err(ExecutionError::new(
                "Qwen work does not match the sequence prefix/capacity",
            ));
        }
        let (phase, input) = match item.kind {
            StepKind::Prefill => {
                if end > sequence.prompt_len
                    || item.output_budget != u32::from(end == sequence.prompt_len)
                {
                    return Err(ExecutionError::new("Qwen prefill boundary/output mismatch"));
                }
                (
                    ExecutionPhase::Prefill,
                    ExecutionTokenInput::prompt(
                        Arc::clone(sequence.prompt.as_ref().ok_or_else(|| {
                            ExecutionError::new("Qwen prompt was already consumed")
                        })?),
                        item.prefix,
                        item.token_budget,
                    )
                    .map_err(model_error)?,
                )
            }
            StepKind::Decode => {
                if item.token_budget != 1
                    || item.output_budget != 1
                    || item.prefix < sequence.prompt_len
                {
                    return Err(ExecutionError::new(
                        "Qwen adapter only supports one-token ordinary decode",
                    ));
                }
                (
                    ExecutionPhase::Decode,
                    ExecutionTokenInput::decode(
                        sequence
                            .next
                            .ok_or_else(|| ExecutionError::new("Qwen decode has no next token"))?,
                    ),
                )
            }
        };
        let segment = ExecutionSegment::new_shared(
            RequestId::new(item.sequence.get()).expect("engine sequence ID"),
            phase,
            1,
            item.token_budget,
            item.prefix,
            self.plan.shared_state_requirements(),
        )
        .map_err(model_error)?
        .with_token_input(input)
        .map_err(model_error)?;
        Ok(if item.output_budget > 0 {
            segment.with_sampling(sequence.sampling)
        } else {
            segment
        })
    }

    pub(crate) fn submit(&mut self, items: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        if self.faulted || self.pending.is_some() {
            return Err(ExecutionError::new("Qwen execution is faulted or busy"));
        }
        let tokens = items
            .iter()
            .try_fold(0_u32, |sum, item| sum.checked_add(item.token_budget));
        if items.len() > self.info.limits.max_sequences
            || tokens.is_none_or(|tokens| tokens > self.info.limits.max_batch_tokens)
        {
            return Err(ExecutionError::new("Qwen batch exceeds prepared limits"));
        }
        let segments = items
            .iter()
            .map(|item| self.segment(item))
            .collect::<Result<Vec<_>, _>>()?;
        // Includes duplicate detection, before taking a single sequence lease.
        let batch = ExecutionBatch::new(segments).map_err(model_error)?;
        let mut states = items
            .iter()
            .map(|item| {
                self.sequences
                    .get_mut(&item.sequence)
                    .expect("validated sequence")
                    .state
                    .take()
                    .expect("validated lease")
            })
            .collect::<Vec<_>>();
        match self.backend.submit(&self.plan, &batch, &mut states) {
            Ok(id) => {
                self.pending = Some(Pending {
                    id,
                    items: items.to_vec(),
                    states,
                    batch,
                });
                Ok(SubmissionId::new(id.get()))
            }
            Err(error) => {
                self.restore(items, states);
                self.faulted = true;
                Err(model_error(error))
            }
        }
    }

    fn restore(&mut self, items: &[BatchItem], states: Vec<InferenceStateSet>) {
        for (item, state) in items.iter().zip(states) {
            let sequence = self
                .sequences
                .get_mut(&item.sequence)
                .expect("pending sequence owner");
            debug_assert!(sequence.state.is_none());
            sequence.state = Some(state);
        }
    }

    pub(crate) fn poll(
        &mut self,
        id: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| ExecutionError::new("unknown Qwen submission"))?;
        if pending.id.get() != id.get() {
            return Err(ExecutionError::new("Qwen submission identity mismatch"));
        }
        let event = match self.backend.poll(pending.id) {
            Ok(None) => return Ok(None),
            Ok(Some(event)) => event,
            Err(error) => {
                let pending = self.pending.take().expect("known submission");
                self.restore(&pending.items, pending.states);
                self.faulted = true;
                return Err(model_error(error));
            }
        };
        let mut pending = self.pending.take().expect("known submission");
        let result = self.commit(&mut pending, &event);
        if result.is_err() {
            self.faulted = true;
        }
        self.restore(&pending.items, pending.states);
        result.map(Some)
    }

    fn commit(
        &mut self,
        pending: &mut Pending,
        event: &ExecutionBatchEvent,
    ) -> Result<Vec<StepCompletion>, ExecutionError> {
        if event.len() != pending.items.len() {
            return Err(ExecutionError::new("Qwen completion row count mismatch"));
        }
        for (segment, result) in pending.batch.segments().iter().zip(event.events()) {
            if result.request() != segment.request()
                || result.phase() != segment.phase()
                || result.token_count() != segment.token_count()
                || result.policy_version() != self.plan.policy_version()
                || result.output_token().is_some() != segment.requests_sampling()
                || result
                    .output_token()
                    .is_some_and(|token| token >= self.vocabulary_size)
            {
                return Err(ExecutionError::new(
                    "Qwen completion does not match submitted work",
                ));
            }
        }
        let positions = pending
            .items
            .iter()
            .map(|item| item.prefix + item.token_budget)
            .collect::<Vec<_>>();
        self.manager
            .commit_batch(&mut pending.states, &positions)
            .map_err(model_error)?;
        Ok(pending
            .items
            .iter()
            .zip(event.events())
            .zip(positions)
            .map(|((item, result), prefix)| {
                let sequence = self
                    .sequences
                    .get_mut(&item.sequence)
                    .expect("pending sequence owner");
                if let Some(token) = result.output_token() {
                    sequence.next = Some(token);
                }
                if prefix >= sequence.prompt_len {
                    sequence.prompt = None;
                }
                StepCompletion {
                    sequence: item.sequence,
                    prefix,
                    tokens: result.output_token().into_iter().collect(),
                }
            })
            .collect())
    }

    pub(crate) fn release(&mut self, id: SequenceId) -> Result<(), ExecutionError> {
        let Some(sequence) = self.sequences.get(&id) else {
            return Ok(());
        };
        let state = sequence
            .state
            .as_ref()
            .ok_or_else(|| ExecutionError::new("cannot release in-flight Qwen state"))?;
        self.backend
            .release_inference_state(state)
            .map_err(model_error)?;
        self.manager.release_set(state).map_err(model_error)?;
        self.sequences.remove(&id);
        Ok(())
    }

    /// After the CUDA owner establishes a stream barrier, collect pending
    /// outcomes to return leases. Shutdown may discard their unreported output.
    pub(crate) fn drain_after_barrier(&mut self) -> Result<(), ExecutionError> {
        if let Some(pending) = self.pending.as_ref() {
            let id = SubmissionId::new(pending.id.get());
            if self.poll(id)?.is_none() {
                return Err(ExecutionError::new(
                    "Qwen submission remained pending after synchronization",
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn model_error(error: impl std::fmt::Display) -> ExecutionError {
    ExecutionError::new(error.to_string())
}

#[cfg(test)]
mod tests;
