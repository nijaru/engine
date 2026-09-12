use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Admission, BatchItem, Event, FinishReason, ModelError, ModelInfo, PreparedModel, RequestId,
    SequenceId, StepCompletion, StepKind, SubmissionId, TokenRequest,
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_active_requests: 8,
            max_queued_requests: 64,
            max_queued_input_tokens: 1_048_576,
            max_buffered_events: 1024,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StepStatus {
    pub submitted: bool,
    pub completed: bool,
    pub output_blocked: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineStatus {
    pub requests: usize,
    /// Includes terminal sequences whose resource release has not succeeded.
    pub active_sequences: usize,
    pub waiting: usize,
    pub buffered_events: usize,
    pub in_flight: bool,
    pub faulted: bool,
    pub closed: bool,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Idle,
    InFlight,
    Cancelling,
}

struct Sequence {
    request: RequestId,
    id: SequenceId,
    input: Option<TokenRequest>,
    prompt_tokens: u32,
    options: crate::GenerationOptions,
    prefix: u32,
    generated: u32,
    admitted: bool,
    work: WorkState,
    terminal: Option<FinishReason>,
    notified: bool,
}

/// One model-neutral scheduling and request owner. Physical sequence resources
/// belong to the prepared model, including resources retained after a fault.
///
/// Call [`Self::shutdown`] for explicit error reporting. Defensive Drop attempts
/// synchronization; if completion cannot be established, it intentionally
/// retains the model rather than freeing device-visible memory prematurely.
pub struct Engine {
    model: Option<Box<dyn PreparedModel>>,
    info: ModelInfo,
    config: EngineConfig,
    policy: SchedulePolicy,
    slots: Vec<Option<Sequence>>,
    free: Vec<usize>,
    requests: HashMap<RequestId, usize>,
    waiting: VecDeque<usize>,
    prefill: VecDeque<usize>,
    decode: VecDeque<usize>,
    terminal: VecDeque<usize>,
    active: usize,
    queued_input_tokens: u64,
    events: VecDeque<Event>,
    batch: Vec<BatchItem>,
    batch_slots: Vec<usize>,
    pending: Option<SubmissionId>,
    reserved_events: usize,
    decode_only_steps: u32,
    fault: Option<ModelError>,
    uncertain: bool,
    closed: bool,
}

impl Engine {
    /// # Errors
    /// Rejects invalid or unsupported engine/model scheduling limits.
    pub fn new(
        model: impl PreparedModel + 'static,
        config: EngineConfig,
        policy: SchedulePolicy,
    ) -> Result<Self, EngineError> {
        let info = model.info().clone();
        info.limits.validate()?;
        validate_policy(policy, &info)?;
        let capacity = config
            .max_active_requests
            .checked_add(config.max_queued_requests)
            .ok_or(EngineError::InvalidConfig)?;
        if config.max_active_requests == 0
            || config.max_active_requests > info.limits.max_sequences
            || config.max_buffered_events < 2
            || config.max_queued_input_tokens == 0
        {
            return Err(EngineError::InvalidConfig);
        }
        Ok(Self {
            model: Some(Box::new(model)),
            info,
            config,
            policy,
            slots: Vec::with_capacity(capacity),
            free: Vec::with_capacity(capacity),
            requests: HashMap::with_capacity(capacity),
            waiting: VecDeque::with_capacity(capacity),
            prefill: VecDeque::with_capacity(config.max_active_requests),
            decode: VecDeque::with_capacity(config.max_active_requests),
            terminal: VecDeque::with_capacity(capacity),
            active: 0,
            queued_input_tokens: 0,
            events: VecDeque::with_capacity(config.max_buffered_events),
            batch: Vec::with_capacity(config.max_active_requests),
            batch_slots: Vec::with_capacity(config.max_active_requests),
            pending: None,
            reserved_events: 0,
            decode_only_steps: 0,
            fault: None,
            uncertain: false,
            closed: false,
        })
    }

    #[must_use]
    pub const fn info(&self) -> &ModelInfo {
        &self.info
    }

    #[must_use]
    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            requests: self.requests.len(),
            active_sequences: self.active,
            waiting: self.waiting.len(),
            buffered_events: self.events.len(),
            in_flight: self.pending.is_some(),
            faulted: self.fault.is_some(),
            closed: self.closed,
        }
    }

    #[must_use]
    pub fn committed_prefix(&self, request: RequestId) -> Option<u32> {
        self.requests
            .get(&request)
            .and_then(|&index| self.slots[index].as_ref())
            .map(|sequence| sequence.prefix)
    }

    /// # Errors
    /// Rejects invalid limits before changing the live policy. In-flight
    /// submissions continue under the immutable budgets they were given.
    pub fn set_policy(&mut self, policy: SchedulePolicy) -> Result<(), EngineError> {
        validate_policy(policy, &self.info)?;
        self.policy = policy;
        Ok(())
    }

    /// Queue encoded input without allocating model state for a waiting request.
    ///
    /// # Errors
    /// Rejects invalid input, exhausted queue bounds, or a closed/faulted engine.
    pub fn enqueue(&mut self, input: TokenRequest) -> Result<RequestId, EngineError> {
        self.check_open()?;
        input.options.sampling.validate()?;
        let prompt_tokens =
            u32::try_from(input.tokens.len()).map_err(|_| EngineError::InvalidRequest)?;
        let total = prompt_tokens
            .checked_add(input.options.max_output_tokens)
            .ok_or(EngineError::InvalidRequest)?;
        if prompt_tokens == 0
            || input.options.max_output_tokens == 0
            || total > self.info.limits.context_tokens
        {
            return Err(EngineError::InvalidRequest);
        }
        let queued = self
            .queued_input_tokens
            .checked_add(u64::from(prompt_tokens))
            .ok_or(EngineError::QueueFull)?;
        if self.requests.len() >= self.config.max_active_requests + self.config.max_queued_requests
            || queued > self.config.max_queued_input_tokens
        {
            return Err(EngineError::QueueFull);
        }
        let request = RequestId(next_id()?);
        let id = SequenceId(next_id()?);
        let index = self.free.pop().unwrap_or_else(|| {
            self.slots.push(None);
            self.slots.len() - 1
        });
        self.slots[index] = Some(Sequence {
            request,
            id,
            options: input.options.clone(),
            input: Some(input),
            prompt_tokens,
            prefix: 0,
            generated: 0,
            admitted: false,
            work: WorkState::Idle,
            terminal: None,
            notified: false,
        });
        self.queued_input_tokens = queued;
        self.requests.insert(request, index);
        self.waiting.push_back(index);
        Ok(request)
    }

    /// # Errors
    /// Returns an error for an unknown/reclaimed request. In-flight cancellation
    /// records intent only; its state stays owned until completion is observed.
    ///
    /// # Panics
    /// Panics if an internal sequence or model-ownership invariant is broken.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), EngineError> {
        let index = *self
            .requests
            .get(&request)
            .ok_or(EngineError::UnknownRequest(request))?;
        let sequence = self.slots[index]
            .as_mut()
            .expect("request index has a slot");
        if sequence.terminal.is_some() {
            return Ok(());
        }
        if sequence.work == WorkState::Idle {
            self.waiting.retain(|&slot| slot != index);
            self.prefill.retain(|&slot| slot != index);
            self.decode.retain(|&slot| slot != index);
            self.terminate(index, FinishReason::Cancelled);
        } else {
            sequence.work = WorkState::Cancelling;
        }
        Ok(())
    }

    pub fn pop_event(&mut self) -> Option<Event> {
        self.events.pop_front()
    }

    /// Observe one completion, reclaim terminal work, admit waiting requests,
    /// and submit one batch. Pending output reserves queue capacity so unrelated
    /// cancellations cannot consume the space promised to an in-flight batch.
    ///
    /// # Errors
    /// Reports model/ownership failures without dropping the cleanup owner.
    ///
    /// # Panics
    /// Panics if an internal sequence or model-ownership invariant is broken.
    pub fn step(&mut self) -> Result<StepStatus, EngineError> {
        let completed = self.poll_completion()?;
        self.flush_terminals()?;
        if let Some(error) = &self.fault {
            return Err(EngineError::Faulted(error.clone()));
        }
        if self.closed || self.pending.is_some() {
            return Ok(StepStatus {
                completed,
                ..StepStatus::default()
            });
        }
        self.admit_waiters();
        self.flush_terminals()?;
        self.build_batch();
        if self.batch.is_empty() {
            return Ok(StepStatus {
                completed,
                output_blocked: !self.prefill.is_empty() || !self.decode.is_empty(),
                ..StepStatus::default()
            });
        }
        match self
            .model
            .as_mut()
            .expect("engine owns model")
            .submit(&self.batch)
        {
            Ok(submission) => self.pending = Some(submission),
            Err(error) => {
                self.fault_all(&error);
                self.flush_terminals()?;
                return Err(EngineError::Faulted(error));
            }
        }
        Ok(StepStatus {
            submitted: true,
            completed,
            output_blocked: false,
        })
    }

    /// Stop new work, establish completion, and release every model sequence.
    /// Existing events remain readable. Errors leave this engine as the retry
    /// owner; call shutdown again rather than manufacturing a fresh state lease.
    ///
    /// # Errors
    /// Reports synchronization or resource-release failure.
    ///
    /// # Panics
    /// Panics if an internal sequence or model-ownership invariant is broken.
    pub fn shutdown(&mut self) -> Result<(), EngineError> {
        self.closed = true;
        self.model
            .as_mut()
            .expect("engine owns model")
            .synchronize()?;
        self.uncertain = false;
        self.pending = None;
        self.reserved_events = 0;
        self.batch.clear();
        self.batch_slots.clear();
        self.waiting.clear();
        self.prefill.clear();
        self.decode.clear();
        for index in 0..self.slots.len() {
            if self.slots[index].is_none() {
                continue;
            }
            self.terminate(index, FinishReason::Cancelled);
            self.release(index)?;
        }
        self.flush_terminals()
    }

    fn check_open(&self) -> Result<(), EngineError> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        if let Some(error) = &self.fault {
            return Err(EngineError::Faulted(error.clone()));
        }
        Ok(())
    }

    fn admit_waiters(&mut self) {
        for _ in 0..self.waiting.len() {
            if self.active >= self.config.max_active_requests {
                break;
            }
            let index = self
                .waiting
                .pop_front()
                .expect("waiting queue was nonempty");
            let sequence = self.slots[index].as_ref().expect("waiting slot exists");
            let input = sequence.input.as_ref().expect("waiting input exists");
            let admitted = self
                .model
                .as_mut()
                .expect("engine owns model")
                .admit(sequence.id, input);
            match admitted {
                Ok(Admission::Ready) => {
                    let sequence = self.slots[index].as_mut().expect("waiting slot exists");
                    sequence.admitted = true;
                    sequence.input = None;
                    self.queued_input_tokens -= u64::from(sequence.prompt_tokens);
                    self.active += 1;
                    self.prefill.push_back(index);
                }
                Ok(Admission::Deferred) => self.waiting.push_back(index),
                Err(error) => self.terminate(index, FinishReason::Failed(error)),
            }
        }
    }

    fn build_batch(&mut self) {
        self.batch.clear();
        self.batch_slots.clear();
        let mut budget = self.policy.max_batch_tokens;
        let mut credits = self.config.max_buffered_events - self.events.len();
        let force_prefill =
            !self.prefill.is_empty() && self.decode_only_steps >= self.policy.max_decode_only_steps;
        if force_prefill {
            // One chunk satisfies the fairness debt without monopolizing a batch.
            self.schedule_one(StepKind::Prefill, &mut budget, &mut credits);
        }
        while self.schedule_one(StepKind::Decode, &mut budget, &mut credits) {}
        while self.schedule_one(StepKind::Prefill, &mut budget, &mut credits) {}
        self.reserved_events = self.config.max_buffered_events - self.events.len() - credits;
        if self.batch.iter().any(|item| item.kind == StepKind::Prefill) {
            self.decode_only_steps = 0;
        } else if !self.batch.is_empty() && !self.prefill.is_empty() {
            self.decode_only_steps = self.decode_only_steps.saturating_add(1);
        }
    }

    fn schedule_one(&mut self, kind: StepKind, budget: &mut u32, credits: &mut usize) -> bool {
        if self.batch.len() >= self.config.max_active_requests || *budget == 0 || *credits == 0 {
            return false;
        }
        let queue = match kind {
            StepKind::Prefill => &mut self.prefill,
            StepKind::Decode => &mut self.decode,
        };
        let Some(&index) = queue.front() else {
            return false;
        };
        let sequence = self.slots[index].as_mut().expect("runnable slot exists");
        let (tokens, outputs) = match kind {
            StepKind::Prefill => {
                let tokens = (sequence.prompt_tokens - sequence.prefix)
                    .min(self.policy.prefill_chunk_tokens)
                    .min(*budget);
                let outputs = u32::from(sequence.prefix + tokens == sequence.prompt_tokens);
                (tokens, outputs)
            }
            StepKind::Decode => {
                let output_credits = u32::try_from(credits.saturating_sub(1)).unwrap_or(u32::MAX);
                let tokens = self
                    .policy
                    .decode_tokens
                    .min(sequence.options.max_output_tokens - sequence.generated)
                    .min(*budget)
                    .min(output_credits);
                (tokens, tokens)
            }
        };
        if tokens == 0 || u64::from(outputs) + 1 > *credits as u64 {
            return false;
        }
        queue.pop_front();
        sequence.work = WorkState::InFlight;
        self.batch.push(BatchItem {
            sequence: sequence.id,
            kind,
            prefix: sequence.prefix,
            token_budget: tokens,
            output_budget: outputs,
        });
        self.batch_slots.push(index);
        *budget -= tokens;
        *credits -= outputs as usize + 1;
        true
    }

    fn poll_completion(&mut self) -> Result<bool, EngineError> {
        let Some(submission) = self.pending else {
            return Ok(false);
        };
        let completion = self
            .model
            .as_mut()
            .expect("engine owns model")
            .poll(submission);
        let rows = match completion {
            Ok(None) => return Ok(false),
            Ok(Some(rows)) => rows,
            Err(error) => {
                self.fault_all(&error);
                self.flush_terminals()?;
                return Err(EngineError::Faulted(error));
            }
        };
        if !self.valid_completion(&rows) {
            let error = ModelError::new(
                "completion does not match its submitted sequence, prefix, or output budget",
            );
            self.fault_all(&error);
            self.flush_terminals()?;
            return Err(EngineError::Faulted(error));
        }
        self.pending = None;
        self.reserved_events = 0;
        // Every row was checked before any logical prefix or output is changed.
        for (row_index, row) in rows.into_iter().enumerate() {
            let index = self.batch_slots[row_index];
            self.commit_row(index, row);
        }
        self.batch.clear();
        self.batch_slots.clear();
        Ok(true)
    }

    fn valid_completion(&self, rows: &[StepCompletion]) -> bool {
        rows.len() == self.batch.len()
            && rows
                .iter()
                .zip(&self.batch)
                .zip(&self.batch_slots)
                .all(|((row, item), &index)| {
                    let Some(sequence) = self.slots[index].as_ref() else {
                        return false;
                    };
                    if sequence.work == WorkState::Idle
                        || sequence.id != row.sequence
                        || row.sequence != item.sequence
                        || sequence.prefix != item.prefix
                    {
                        return false;
                    }
                    let Some(advance) = row.prefix.checked_sub(item.prefix) else {
                        return false;
                    };
                    let Ok(outputs) = u32::try_from(row.tokens.len()) else {
                        return false;
                    };
                    match item.kind {
                        StepKind::Prefill => {
                            advance == item.token_budget && outputs == item.output_budget
                        }
                        StepKind::Decode => {
                            advance > 0
                                && advance <= item.token_budget
                                && outputs == advance
                                && outputs <= item.output_budget
                        }
                    }
                })
    }

    fn commit_row(&mut self, index: usize, row: StepCompletion) {
        let sequence = self.slots[index]
            .as_mut()
            .expect("validated completion slot");
        let cancelled = sequence.work == WorkState::Cancelling;
        sequence.work = WorkState::Idle;
        sequence.prefix = row.prefix;
        if cancelled {
            self.terminate(index, FinishReason::Cancelled);
            return;
        }
        let mut reason = None;
        for token in row.tokens {
            sequence.generated += 1;
            if sequence.options.stop_tokens.contains(&token) {
                reason = Some(FinishReason::Stop);
                break;
            }
            self.events.push_back(Event::Token {
                request: sequence.request,
                token,
            });
            if sequence.generated == sequence.options.max_output_tokens {
                reason = Some(FinishReason::Length);
                break;
            }
        }
        if let Some(reason) = reason {
            self.terminate(index, reason);
        } else if sequence.prefix < sequence.prompt_tokens {
            self.prefill.push_back(index);
        } else {
            self.decode.push_back(index);
        }
    }

    fn terminate(&mut self, index: usize, reason: FinishReason) {
        let sequence = self.slots[index].as_mut().expect("terminal slot exists");
        if sequence.terminal.is_some() {
            return;
        }
        if sequence.input.take().is_some() {
            self.queued_input_tokens -= u64::from(sequence.prompt_tokens);
        }
        sequence.terminal = Some(reason);
        sequence.work = WorkState::Idle;
        self.terminal.push_back(index);
    }

    fn release(&mut self, index: usize) -> Result<(), EngineError> {
        let sequence = self.slots[index].as_mut().expect("release slot exists");
        if sequence.admitted {
            self.model
                .as_mut()
                .expect("engine owns model")
                .release(sequence.id)?;
            sequence.admitted = false;
            self.active -= 1;
        }
        Ok(())
    }

    fn flush_terminals(&mut self) -> Result<(), EngineError> {
        for _ in 0..self.terminal.len() {
            let index = self
                .terminal
                .pop_front()
                .expect("terminal queue was nonempty");
            let sequence = self.slots[index].as_mut().expect("terminal slot exists");
            if !sequence.notified
                && self.events.len() + self.reserved_events < self.config.max_buffered_events
            {
                self.events.push_back(Event::Finished {
                    request: sequence.request,
                    reason: sequence
                        .terminal
                        .as_ref()
                        .expect("terminal reason exists")
                        .clone(),
                });
                sequence.notified = true;
            }
            if !self.uncertain
                && let Err(error) = self.release(index)
            {
                self.terminal.push_front(index);
                return Err(error);
            }
            let sequence = self.slots[index].as_ref().expect("terminal slot exists");
            if sequence.notified && !sequence.admitted {
                self.requests.remove(&sequence.request);
                self.slots[index] = None;
                self.free.push(index);
            } else {
                self.terminal.push_back(index);
            }
        }
        Ok(())
    }

    fn fault_all(&mut self, error: &ModelError) {
        self.fault = Some(error.clone());
        self.uncertain = true;
        self.pending = None;
        self.reserved_events = 0;
        self.waiting.clear();
        self.prefill.clear();
        self.decode.clear();
        self.batch.clear();
        self.batch_slots.clear();
        for index in 0..self.slots.len() {
            if self.slots[index].is_some() {
                self.terminate(index, FinishReason::Failed(error.clone()));
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.model.is_some() && self.shutdown().is_err() {
            // A driver error cannot prove that device work stopped. Keep the
            // single physical owner alive rather than causing a use-after-free.
            if let Some(model) = self.model.take() {
                std::mem::forget(model);
            }
        }
    }
}

fn next_id() -> Result<u64, EngineError> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| EngineError::IdentityExhausted)
}

fn validate_policy(policy: SchedulePolicy, info: &ModelInfo) -> Result<(), EngineError> {
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineError {
    InvalidConfig,
    InvalidRequest,
    QueueFull,
    UnknownRequest(RequestId),
    IdentityExhausted,
    Closed,
    Model(ModelError),
    Faulted(ModelError),
}

impl From<ModelError> for EngineError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => {
                f.write_str("engine/policy limits are invalid or exceed the prepared model")
            }
            Self::InvalidRequest => {
                f.write_str("input/output token limits are invalid or exceed the model context")
            }
            Self::QueueFull => f.write_str("request queue or queued-input token budget is full"),
            Self::UnknownRequest(request) => {
                write!(f, "request {} is unknown or reclaimed", request.get())
            }
            Self::IdentityExhausted => f.write_str("runtime identity space is exhausted"),
            Self::Closed => f.write_str("engine is closed"),
            Self::Model(error) => error.fmt(f),
            Self::Faulted(error) => write!(
                f,
                "engine is faulted; shutdown retains cleanup ownership: {error}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}
