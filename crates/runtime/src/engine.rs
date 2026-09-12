use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Admission, BatchItem, EngineConfig, EngineError, Event, ExecutionError, ExecutorInfo,
    FinishReason, GenerationExecutor, RequestId, SchedulePolicy, SequenceId, SubmissionId,
    TokenRequest, Usage,
};

use crate::config::validate_policy;
use crate::output::{Output, OutputId};

mod completion;
mod scheduling;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

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
    output: OutputId,
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
    model: Option<Box<dyn GenerationExecutor>>,
    info: ExecutorInfo,
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
    queued_prompt_tokens: u64,
    output: Output,
    batch: Vec<BatchItem>,
    batch_slots: Vec<usize>,
    pending: Option<SubmissionId>,
    decode_only_steps: u32,
    fault: Option<ExecutionError>,
    uncertain: bool,
    closed: bool,
}

impl Engine {
    /// # Errors
    /// Rejects invalid or unsupported engine/model scheduling limits.
    pub fn new(
        model: impl GenerationExecutor + 'static,
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
            || config.max_events_per_request < 2
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
            queued_prompt_tokens: 0,
            output: Output::new(
                config.max_buffered_events,
                config.max_events_per_request,
                capacity,
            ),
            batch: Vec::with_capacity(config.max_active_requests),
            batch_slots: Vec::with_capacity(config.max_active_requests),
            pending: None,
            decode_only_steps: 0,
            fault: None,
            uncertain: false,
            closed: false,
        })
    }

    /// Construct model-compatible limits without forcing callers to repeat the
    /// prepared model's sequence and token limits. Explicit `new` stays strict.
    ///
    /// # Errors
    /// Reports invalid prepared model limits.
    pub fn with_defaults(model: impl GenerationExecutor + 'static) -> Result<Self, EngineError> {
        let limits = model.info().limits;
        let config = EngineConfig {
            max_active_requests: EngineConfig::default()
                .max_active_requests
                .min(limits.max_sequences),
            ..EngineConfig::default()
        };
        let policy = SchedulePolicy {
            max_batch_tokens: SchedulePolicy::default()
                .max_batch_tokens
                .min(limits.max_batch_tokens),
            ..SchedulePolicy::default()
        };
        Self::new(model, config, policy)
    }

    #[must_use]
    pub const fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    #[must_use]
    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            requests: self.requests.len(),
            active_sequences: self.active,
            waiting: self.waiting.len(),
            buffered_events: self.output.len(),
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
            .queued_prompt_tokens
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
        let output = self.output.register(request);
        self.slots[index] = Some(Sequence {
            output,
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
        self.queued_prompt_tokens = queued;
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

    /// Drain ready requests round-robin, preserving each request's event order.
    pub fn pop_event(&mut self) -> Option<Event> {
        self.output.pop()
    }

    /// Drain only this request. Other clients' output remains bounded and does
    /// not need to be copied into an unbounded frontend-side demultiplexer.
    pub fn pop_event_for(&mut self, request: RequestId) -> Option<Event> {
        self.output.pop_for(request)
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
        self.release_output_reservations();
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
                    self.queued_prompt_tokens -= u64::from(sequence.prompt_tokens);
                    self.active += 1;
                    self.prefill.push_back(index);
                }
                Ok(Admission::Deferred) => self.waiting.push_back(index),
                Err(error) => self.terminate(index, FinishReason::Failed(error)),
            }
        }
    }

    fn terminate(&mut self, index: usize, reason: FinishReason) {
        let sequence = self.slots[index].as_mut().expect("terminal slot exists");
        if sequence.terminal.is_some() {
            return;
        }
        if sequence.input.take().is_some() {
            self.queued_prompt_tokens -= u64::from(sequence.prompt_tokens);
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
            if !sequence.notified && self.output.credits(sequence.output) > 0 {
                self.output.push_to(
                    sequence.output,
                    Event::Finished {
                        request: sequence.request,
                        reason: sequence
                            .terminal
                            .as_ref()
                            .expect("terminal reason exists")
                            .clone(),
                        usage: Usage {
                            prompt_tokens: sequence.prompt_tokens,
                            completion_tokens: sequence.generated,
                        },
                    },
                );
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

    fn fault_all(&mut self, error: &ExecutionError) {
        self.fault = Some(error.clone());
        self.uncertain = true;
        self.pending = None;
        self.release_output_reservations();
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
