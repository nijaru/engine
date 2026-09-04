//! Deterministic request scheduling and admission for the serving runtime.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use crate::backend::BackendSubmissionId;
use crate::execution::{ExecutionEvent, ExecutionPhase};
use crate::policy::PolicySnapshot;
use crate::request::{RequestId, RequestSpec};
use crate::serving::{ActiveRequestSlot, RequestLifecycle, RequestSlotError, RequestSlotId, RequestSlots};
use crate::state::InferenceStateSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SchedulerConfig {
    max_active_requests: usize,
    max_queued_requests: usize,
    prefill_chunk_tokens: u32,
}

impl SchedulerConfig {
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidConfig`] when no request can become
    /// active or the prefill chunk budget is zero.
    pub const fn new(
        max_active_requests: usize,
        max_queued_requests: usize,
        prefill_chunk_tokens: u32,
    ) -> Result<Self, SchedulerError> {
        if max_active_requests == 0 || prefill_chunk_tokens == 0 {
            return Err(SchedulerError::InvalidConfig);
        }
        if max_active_requests.checked_add(max_queued_requests).is_none() {
            return Err(SchedulerError::InvalidConfig);
        }
        Ok(Self {
            max_active_requests,
            max_queued_requests,
            prefill_chunk_tokens,
        })
    }

    #[must_use]
    pub const fn max_active_requests(self) -> usize {
        self.max_active_requests
    }

    #[must_use]
    pub const fn max_queued_requests(self) -> usize {
        self.max_queued_requests
    }

    #[must_use]
    pub const fn prefill_chunk_tokens(self) -> u32 {
        self.prefill_chunk_tokens
    }

    #[must_use]
    pub const fn max_resident_requests(self) -> usize {
        self.max_active_requests + self.max_queued_requests
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScheduledWork {
    slot: RequestSlotId,
    request: RequestId,
    phase: ExecutionPhase,
    token_count: u32,
    state_position: u32,
}

impl ScheduledWork {
    #[must_use]
    pub const fn slot(self) -> RequestSlotId {
        self.slot
    }

    #[must_use]
    pub const fn request(self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn phase(self) -> ExecutionPhase {
        self.phase
    }

    #[must_use]
    pub const fn token_count(self) -> u32 {
        self.token_count
    }

    #[must_use]
    pub const fn state_position(self) -> u32 {
        self.state_position
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SchedulerCounts {
    waiting: usize,
    runnable: usize,
    in_flight: usize,
    terminal: usize,
}

impl SchedulerCounts {
    #[must_use]
    pub const fn waiting(self) -> usize {
        self.waiting
    }

    #[must_use]
    pub const fn runnable(self) -> usize {
        self.runnable
    }

    #[must_use]
    pub const fn in_flight(self) -> usize {
        self.in_flight
    }

    #[must_use]
    pub const fn terminal(self) -> usize {
        self.terminal
    }
}

#[derive(Debug)]
pub struct ServingScheduler {
    policy: PolicySnapshot,
    config: SchedulerConfig,
    slots: RequestSlots,
    waiting: VecDeque<RequestSlotId>,
    runnable: VecDeque<RequestSlotId>,
    in_flight: HashMap<BackendSubmissionId, RequestSlotId>,
    terminal: VecDeque<RequestSlotId>,
    requests: HashMap<RequestId, RequestSlotId>,
}

impl ServingScheduler {
    #[must_use]
    pub fn new(policy: PolicySnapshot, config: SchedulerConfig) -> Self {
        Self {
            policy,
            config,
            slots: RequestSlots::default(),
            waiting: VecDeque::new(),
            runnable: VecDeque::new(),
            in_flight: HashMap::new(),
            terminal: VecDeque::new(),
            requests: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn policy(&self) -> PolicySnapshot {
        self.policy
    }

    #[must_use]
    pub const fn config(&self) -> SchedulerConfig {
        self.config
    }

    #[must_use]
    pub const fn slots(&self) -> &RequestSlots {
        &self.slots
    }

    #[must_use]
    pub fn counts(&self) -> SchedulerCounts {
        SchedulerCounts {
            waiting: self.waiting.len(),
            runnable: self.runnable.len(),
            in_flight: self.in_flight.len(),
            terminal: self.terminal.len(),
        }
    }

    #[must_use]
    pub fn slot_for_request(&self, request: RequestId) -> Option<RequestSlotId> {
        self.requests.get(&request).copied()
    }

    /// Admit one request into scheduler-owned persistent state.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::Backpressure`] when resident capacity is
    /// exhausted, or a slot error when the request cannot be inserted.
    pub fn admit(
        &mut self,
        request: RequestSpec,
        state: InferenceStateSet,
        prompt_tokens: u32,
    ) -> Result<RequestSlotId, SchedulerError> {
        if self.requests.contains_key(&request.id()) {
            return Err(SchedulerError::DuplicateRequest(request.id()));
        }
        if self.slots.len() >= self.config.max_resident_requests() {
            return Err(SchedulerError::Backpressure);
        }

        let request_id = request.id();
        let id = self.slots.insert(request, state, prompt_tokens)?;
        if self.active_count() < self.config.max_active_requests() {
            self.slots
                .get_mut(id)
                .ok_or(SchedulerError::Invariant("new request slot disappeared"))?
                .make_runnable()?;
            self.runnable.push_back(id);
        } else {
            if self.waiting.len() >= self.config.max_queued_requests() {
                return Err(SchedulerError::Backpressure);
            }
            self.waiting.push_back(id);
        }
        self.requests.insert(request_id, id);
        Ok(id)
    }

    /// Select one deterministic scheduling iteration. Decode work is selected
    /// first; remaining sequence/token budget is filled with chunked prefill.
    /// Selection does not mutate request lifecycle until submission begins.
    ///
    /// # Errors
    ///
    /// Returns a scheduler invariant error if queue state and persistent slots
    /// disagree.
    pub fn schedule(&self) -> Result<Vec<ScheduledWork>, SchedulerError> {
        let sequence_limit = self
            .config
            .max_active_requests()
            .min(usize::try_from(self.policy.max_batch_size()).unwrap_or(usize::MAX));
        let mut token_budget = self.policy.max_batch_tokens();
        let mut work = Vec::with_capacity(sequence_limit);

        for &id in &self.runnable {
            if work.len() >= sequence_limit || token_budget == 0 {
                break;
            }
            let slot = self.runnable_slot(id)?;
            if self.phase_for(slot)? != ExecutionPhase::Decode {
                continue;
            }
            work.push(self.make_work(slot, ExecutionPhase::Decode, 1)?);
            token_budget -= 1;
        }

        for &id in &self.runnable {
            if work.len() >= sequence_limit || token_budget == 0 {
                break;
            }
            let slot = self.runnable_slot(id)?;
            if self.phase_for(slot)? != ExecutionPhase::Prefill {
                continue;
            }
            let remaining = slot
                .progress()
                .prompt_tokens()
                .checked_sub(slot.progress().prompt_processed())
                .ok_or(SchedulerError::Invariant("prompt progress exceeded prompt length"))?;
            let token_count = remaining
                .min(self.config.prefill_chunk_tokens())
                .min(token_budget);
            if token_count == 0 {
                continue;
            }
            work.push(self.make_work(slot, ExecutionPhase::Prefill, token_count)?);
            token_budget -= token_count;
        }

        Ok(work)
    }

    /// Associate selected work with the backend submission that now owns its
    /// inference state.
    ///
    /// # Errors
    ///
    /// Returns an error for stale work, duplicate submission identity, or a
    /// lifecycle/queue mismatch.
    pub fn begin_submission(
        &mut self,
        work: ScheduledWork,
        submission: BackendSubmissionId,
    ) -> Result<(), SchedulerError> {
        if self.in_flight.contains_key(&submission) {
            return Err(SchedulerError::DuplicateSubmission(submission));
        }
        let queue_index = self
            .runnable
            .iter()
            .position(|candidate| *candidate == work.slot())
            .ok_or(SchedulerError::StaleWork)?;
        {
            let slot = self
                .slots
                .get(work.slot())
                .ok_or(SchedulerError::StaleWork)?;
            if slot.request().id() != work.request()
                || slot.lifecycle() != RequestLifecycle::Runnable
                || self.phase_for(slot)? != work.phase()
                || self.state_position(slot)? != work.state_position()
            {
                return Err(SchedulerError::StaleWork);
            }
            match work.phase() {
                ExecutionPhase::Decode if work.token_count() != 1 => {
                    return Err(SchedulerError::StaleWork);
                }
                ExecutionPhase::Prefill => {
                    let remaining = slot
                        .progress()
                        .prompt_tokens()
                        .checked_sub(slot.progress().prompt_processed())
                        .ok_or(SchedulerError::StaleWork)?;
                    if work.token_count() == 0
                        || work.token_count() > remaining
                        || work.token_count() > self.config.prefill_chunk_tokens()
                    {
                        return Err(SchedulerError::StaleWork);
                    }
                }
                ExecutionPhase::SpecDraft
                | ExecutionPhase::SpecVerify
                | ExecutionPhase::Encoder
                | ExecutionPhase::MoEExpert => return Err(SchedulerError::StaleWork),
                ExecutionPhase::Decode => {}
            }
        }
        self.slots
            .get_mut(work.slot())
            .ok_or(SchedulerError::StaleWork)?
            .begin_submission(submission)?;
        self.runnable.remove(queue_index);
        self.in_flight.insert(submission, work.slot());
        Ok(())
    }

    /// Apply a backend completion and return the request to the runnable queue
    /// or its terminal queue.
    ///
    /// # Errors
    ///
    /// Returns an error when the submission/request identity is stale or the
    /// completed progress is invalid.
    pub fn complete_submission(
        &mut self,
        submission: BackendSubmissionId,
        event: ExecutionEvent,
        state: InferenceStateSet,
    ) -> Result<(), SchedulerError> {
        let id = self
            .in_flight
            .get(&submission)
            .copied()
            .ok_or(SchedulerError::UnknownSubmission(submission))?;
        {
            let slot = self
                .slots
                .get(id)
                .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?;
            if slot.request().id() != event.request() {
                return Err(SchedulerError::CompletionRequestMismatch);
            }
        }
        self.slots
            .get_mut(id)
            .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?
            .complete_step(submission, event.phase(), event.token_count(), state)?;
        self.in_flight.remove(&submission);

        let lifecycle = self
            .slots
            .get(id)
            .ok_or(SchedulerError::Invariant("completed slot disappeared"))?
            .lifecycle();
        match lifecycle {
            RequestLifecycle::Runnable => {
                let reached_output_limit = {
                    let slot = self
                        .slots
                        .get(id)
                        .ok_or(SchedulerError::Invariant("completed slot disappeared"))?;
                    slot.progress().generated_tokens() >= slot.request().semantics().max_output_tokens()
                };
                if reached_output_limit {
                    self.slots
                        .get_mut(id)
                        .ok_or(SchedulerError::Invariant("completed slot disappeared"))?
                        .mark_completed()?;
                    self.terminal.push_back(id);
                    self.fill_runnable()?;
                } else {
                    self.runnable.push_back(id);
                }
            }
            RequestLifecycle::Cancelled | RequestLifecycle::Failed | RequestLifecycle::Completed => {
                self.terminal.push_back(id);
                self.fill_runnable()?;
            }
            RequestLifecycle::Waiting
            | RequestLifecycle::InFlight(_)
            | RequestLifecycle::Cancelling(_) => {
                return Err(SchedulerError::Invariant(
                    "completion left request in a non-completable lifecycle",
                ));
            }
        }
        Ok(())
    }

    /// Mark an in-flight request failed after the backend reports release of
    /// submission ownership.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown submission or lifecycle mismatch.
    pub fn fail_submission(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<(), SchedulerError> {
        let id = self
            .in_flight
            .get(&submission)
            .copied()
            .ok_or(SchedulerError::UnknownSubmission(submission))?;
        self.slots
            .get_mut(id)
            .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?
            .fail_submission(submission)?;
        self.in_flight.remove(&submission);
        self.terminal.push_back(id);
        self.fill_runnable()?;
        Ok(())
    }

    /// Request cancellation by semantic request identity.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or terminal requests.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), SchedulerError> {
        let id = self
            .requests
            .get(&request)
            .copied()
            .ok_or(SchedulerError::UnknownRequest(request))?;
        let before = self
            .slots
            .get(id)
            .ok_or(SchedulerError::Invariant("request index points to no slot"))?
            .lifecycle();
        self.slots
            .get_mut(id)
            .ok_or(SchedulerError::Invariant("request index points to no slot"))?
            .request_cancel()?;
        let after = self
            .slots
            .get(id)
            .ok_or(SchedulerError::Invariant("cancelled slot disappeared"))?
            .lifecycle();

        if after == RequestLifecycle::Cancelled {
            match before {
                RequestLifecycle::Waiting => remove_id(&mut self.waiting, id)?,
                RequestLifecycle::Runnable => remove_id(&mut self.runnable, id)?,
                _ => {
                    return Err(SchedulerError::Invariant(
                        "immediate cancellation came from invalid lifecycle",
                    ));
                }
            }
            self.terminal.push_back(id);
            self.fill_runnable()?;
        }
        Ok(())
    }

    /// Mark a runnable request complete for a semantic stop condition such as
    /// EOS before its maximum output budget.
    ///
    /// # Errors
    ///
    /// Returns an error unless the request is currently runnable.
    pub fn finish(&mut self, request: RequestId) -> Result<(), SchedulerError> {
        let id = self
            .requests
            .get(&request)
            .copied()
            .ok_or(SchedulerError::UnknownRequest(request))?;
        let queue_index = self
            .runnable
            .iter()
            .position(|candidate| *candidate == id)
            .ok_or(SchedulerError::InvalidLifecycle)?;
        self.slots
            .get_mut(id)
            .ok_or(SchedulerError::Invariant("request index points to no slot"))?
            .mark_completed()?;
        self.runnable.remove(queue_index);
        self.terminal.push_back(id);
        self.fill_runnable()?;
        Ok(())
    }

    /// Reclaim one terminal slot. Terminal slots remain resident until this is
    /// called so state/resource destruction can be handled explicitly.
    ///
    /// # Errors
    ///
    /// Returns a slot error if terminal bookkeeping and persistent state
    /// disagree.
    pub fn reclaim_next(&mut self) -> Result<Option<ActiveRequestSlot>, SchedulerError> {
        let Some(id) = self.terminal.pop_front() else {
            return Ok(None);
        };
        let request = self
            .slots
            .get(id)
            .ok_or(SchedulerError::Invariant("terminal slot disappeared"))?
            .request()
            .id();
        let slot = self.slots.remove(id)?;
        self.requests.remove(&request);
        Ok(Some(slot))
    }

    fn active_count(&self) -> usize {
        self.runnable.len() + self.in_flight.len()
    }

    fn fill_runnable(&mut self) -> Result<(), SchedulerError> {
        while self.active_count() < self.config.max_active_requests() {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            self.slots
                .get_mut(id)
                .ok_or(SchedulerError::Invariant("waiting slot disappeared"))?
                .make_runnable()?;
            self.runnable.push_back(id);
        }
        Ok(())
    }

    fn runnable_slot(&self, id: RequestSlotId) -> Result<&ActiveRequestSlot, SchedulerError> {
        let slot = self
            .slots
            .get(id)
            .ok_or(SchedulerError::Invariant("runnable slot disappeared"))?;
        if slot.lifecycle() != RequestLifecycle::Runnable {
            return Err(SchedulerError::Invariant(
                "runnable queue contains non-runnable request",
            ));
        }
        Ok(slot)
    }

    fn phase_for(&self, slot: &ActiveRequestSlot) -> Result<ExecutionPhase, SchedulerError> {
        if slot.progress().prompt_processed() < slot.progress().prompt_tokens() {
            return Ok(ExecutionPhase::Prefill);
        }
        if slot.progress().generated_tokens() < slot.request().semantics().max_output_tokens() {
            return Ok(ExecutionPhase::Decode);
        }
        Err(SchedulerError::Invariant(
            "runnable request has no remaining semantic work",
        ))
    }

    fn state_position(&self, slot: &ActiveRequestSlot) -> Result<u32, SchedulerError> {
        if let Some(position) = slot.state().token_position() {
            return Ok(position);
        }
        slot.progress()
            .prompt_processed()
            .checked_add(slot.progress().generated_tokens())
            .ok_or(SchedulerError::PositionOverflow)
    }

    fn make_work(
        &self,
        slot: &ActiveRequestSlot,
        phase: ExecutionPhase,
        token_count: u32,
    ) -> Result<ScheduledWork, SchedulerError> {
        Ok(ScheduledWork {
            slot: slot.id(),
            request: slot.request().id(),
            phase,
            token_count,
            state_position: self.state_position(slot)?,
        })
    }
}

fn remove_id(queue: &mut VecDeque<RequestSlotId>, id: RequestSlotId) -> Result<(), SchedulerError> {
    let index = queue
        .iter()
        .position(|candidate| *candidate == id)
        .ok_or(SchedulerError::Invariant("request was absent from expected queue"))?;
    queue.remove(index);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SchedulerError {
    InvalidConfig,
    Backpressure,
    DuplicateRequest(RequestId),
    UnknownRequest(RequestId),
    UnknownSubmission(BackendSubmissionId),
    DuplicateSubmission(BackendSubmissionId),
    CompletionRequestMismatch,
    InvalidLifecycle,
    StaleWork,
    PositionOverflow,
    Invariant(&'static str),
    Slot(RequestSlotError),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("scheduler configuration is invalid"),
            Self::Backpressure => f.write_str("scheduler admission capacity is exhausted"),
            Self::DuplicateRequest(id) => write!(f, "request {} is already admitted", id.get()),
            Self::UnknownRequest(id) => write!(f, "request {} is unknown", id.get()),
            Self::UnknownSubmission(id) => write!(f, "submission {} is unknown", id.get()),
            Self::DuplicateSubmission(id) => write!(f, "submission {} is already in flight", id.get()),
            Self::CompletionRequestMismatch => {
                f.write_str("backend completion belongs to a different request")
            }
            Self::InvalidLifecycle => f.write_str("request lifecycle does not permit this operation"),
            Self::StaleWork => f.write_str("scheduled work is stale or no longer runnable"),
            Self::PositionOverflow => f.write_str("request state position overflowed"),
            Self::Invariant(reason) => write!(f, "scheduler invariant failed: {reason}"),
            Self::Slot(error) => write!(f, "request slot error: {error}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<RequestSlotError> for SchedulerError {
    fn from(error: RequestSlotError) -> Self {
        Self::Slot(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionMetrics;
    use crate::model::ModelId;
    use crate::policy::{PolicyVersion, SpeculationPolicy, StateTierPreference};
    use crate::request::{RequestSemantics, SamplingParams, ThinkingMode};

    fn policy(max_batch_size: u32, max_batch_tokens: u32) -> PolicySnapshot {
        PolicySnapshot::new(
            PolicyVersion::new(1).expect("policy version"),
            max_batch_size,
            max_batch_tokens,
            StateTierPreference::Automatic,
            SpeculationPolicy::Disabled,
        )
        .expect("policy")
    }

    fn request(id: u64, max_output_tokens: u32) -> RequestSpec {
        RequestSpec::new(
            RequestId::new(id).expect("request ID"),
            ModelId::new("test/model").expect("model ID"),
            RequestSemantics::new(
                max_output_tokens,
                SamplingParams::greedy(Some(id)),
                ThinkingMode::Off,
            )
            .expect("semantics"),
        )
    }

    fn state() -> InferenceStateSet {
        InferenceStateSet::new(Vec::new()).expect("empty state")
    }

    fn event(work: ScheduledWork) -> ExecutionEvent {
        ExecutionEvent::new(
            work.request(),
            PolicyVersion::new(1).expect("policy version"),
            work.phase(),
            work.token_count(),
            ExecutionMetrics::new(1, 0, 0),
        )
        .expect("event")
    }

    #[test]
    fn admission_is_bounded_and_promotes_oldest_waiter() {
        let config = SchedulerConfig::new(1, 1, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(1, 4), config);
        scheduler.admit(request(1, 2), state(), 0).expect("first");
        scheduler.admit(request(2, 2), state(), 0).expect("second");
        assert_eq!(scheduler.counts().runnable(), 1);
        assert_eq!(scheduler.counts().waiting(), 1);
        assert_eq!(
            scheduler.admit(request(3, 2), state(), 0),
            Err(SchedulerError::Backpressure)
        );

        scheduler
            .cancel(RequestId::new(1).expect("request ID"))
            .expect("cancel");
        assert_eq!(scheduler.counts().terminal(), 1);
        assert_eq!(scheduler.counts().waiting(), 0);
        assert_eq!(scheduler.counts().runnable(), 1);
        let work = scheduler.schedule().expect("schedule");
        assert_eq!(work[0].request(), RequestId::new(2).expect("request ID"));
    }

    #[test]
    fn decode_precedes_chunked_prefill_under_shared_budget() {
        let config = SchedulerConfig::new(3, 0, 3).expect("config");
        let mut scheduler = ServingScheduler::new(policy(3, 4), config);
        scheduler.admit(request(1, 4), state(), 0).expect("decode");
        scheduler.admit(request(2, 4), state(), 10).expect("prefill");
        scheduler.admit(request(3, 4), state(), 10).expect("prefill");

        let work = scheduler.schedule().expect("schedule");
        assert_eq!(work.len(), 2);
        assert_eq!(work[0].request(), RequestId::new(1).expect("request ID"));
        assert_eq!(work[0].phase(), ExecutionPhase::Decode);
        assert_eq!(work[0].token_count(), 1);
        assert_eq!(work[1].request(), RequestId::new(2).expect("request ID"));
        assert_eq!(work[1].phase(), ExecutionPhase::Prefill);
        assert_eq!(work[1].token_count(), 3);
    }

    #[test]
    fn completion_rotates_runnable_requests_and_finishes_output_budget() {
        let config = SchedulerConfig::new(2, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(2, 2), config);
        scheduler.admit(request(1, 1), state(), 0).expect("first");
        scheduler.admit(request(2, 2), state(), 0).expect("second");
        let work = scheduler.schedule().expect("schedule");
        let first = work[0];
        let submission = BackendSubmissionId::new(1).expect("submission");
        scheduler
            .begin_submission(first, submission)
            .expect("begin submission");
        scheduler
            .complete_submission(submission, event(first), state())
            .expect("complete");
        assert_eq!(scheduler.counts().terminal(), 1);
        let next = scheduler.schedule().expect("next schedule");
        assert_eq!(next[0].request(), RequestId::new(2).expect("request ID"));
    }

    #[test]
    fn in_flight_cancel_waits_for_backend_completion_before_reclaim() {
        let config = SchedulerConfig::new(1, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(1, 1), config);
        let request_id = RequestId::new(1).expect("request ID");
        scheduler.admit(request(1, 4), state(), 0).expect("admit");
        let work = scheduler.schedule().expect("schedule")[0];
        let submission = BackendSubmissionId::new(1).expect("submission");
        scheduler
            .begin_submission(work, submission)
            .expect("begin submission");
        scheduler.cancel(request_id).expect("cancel");
        assert_eq!(scheduler.counts().in_flight(), 1);
        assert_eq!(scheduler.counts().terminal(), 0);
        assert!(scheduler.reclaim_next().expect("reclaim").is_none());

        scheduler
            .complete_submission(submission, event(work), state())
            .expect("completion");
        assert_eq!(scheduler.counts().in_flight(), 0);
        assert_eq!(scheduler.counts().terminal(), 1);
        let reclaimed = scheduler
            .reclaim_next()
            .expect("reclaim")
            .expect("terminal request");
        assert_eq!(reclaimed.request().id(), request_id);
    }
}
