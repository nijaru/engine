//! Deterministic request scheduling and admission for the serving runtime.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::backend::BackendSubmissionId;
use crate::execution::{ExecutionBatchEvent, ExecutionPhase};
use crate::policy::PolicySnapshot;
use crate::request::{RequestId, RequestSpec};
use crate::serving::{
    ActiveRequestSlot, RequestLifecycle, RequestSlotError, RequestSlotId, RequestSlots,
};
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
        if max_active_requests
            .checked_add(max_queued_requests)
            .is_none()
        {
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
    sample_output: bool,
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

    #[must_use]
    pub const fn requests_output(self) -> bool {
        self.sample_output
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SchedulerCounts {
    waiting: usize,
    runnable: usize,
    prepared: usize,
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
    pub const fn prepared(self) -> usize {
        self.prepared
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
    prepared: HashMap<RequestSlotId, ScheduledWork>,
    in_flight: HashMap<BackendSubmissionId, Vec<ScheduledWork>>,
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
            prepared: HashMap::new(),
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
            prepared: self.prepared.len(),
            in_flight: self.in_flight_request_count(),
            terminal: self.terminal.len(),
        }
    }

    #[must_use]
    pub fn slot_for_request(&self, request: RequestId) -> Option<RequestSlotId> {
        self.requests.get(&request).copied()
    }

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
            self.waiting.push_back(id);
        }
        self.requests.insert(request_id, id);
        Ok(id)
    }

    /// Select one deterministic scheduling iteration. Decode work is selected
    /// first; remaining sequence/token budget is filled with chunked prefill.
    ///
    /// # Errors
    ///
    /// Returns an invariant error if queue state and persistent slots disagree.
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
            if Self::phase_for(slot)? != ExecutionPhase::Decode {
                continue;
            }
            work.push(Self::make_work(slot, ExecutionPhase::Decode, 1)?);
            token_budget -= 1;
        }

        for &id in &self.runnable {
            if work.len() >= sequence_limit || token_budget == 0 {
                break;
            }
            let slot = self.runnable_slot(id)?;
            if Self::phase_for(slot)? != ExecutionPhase::Prefill {
                continue;
            }
            let remaining = slot
                .progress()
                .prompt_tokens()
                .checked_sub(slot.progress().prompt_processed())
                .ok_or(SchedulerError::Invariant(
                    "prompt progress exceeded prompt length",
                ))?;
            let token_count = remaining
                .min(self.config.prefill_chunk_tokens())
                .min(token_budget);
            if token_count == 0 {
                continue;
            }
            work.push(Self::make_work(slot, ExecutionPhase::Prefill, token_count)?);
            token_budget -= token_count;
        }

        Ok(work)
    }

    /// Transfer state for a selected batch out of request slots before backend
    /// submission. All work is validated before any slot is mutated.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::StaleWork`] when selection is no longer valid.
    pub fn prepare_submission(
        &mut self,
        work: &[ScheduledWork],
    ) -> Result<Vec<InferenceStateSet>, SchedulerError> {
        if work.is_empty() {
            return Err(SchedulerError::EmptyWork);
        }
        self.validate_work_set(work)?;

        let mut states = Vec::with_capacity(work.len());
        for item in work {
            remove_id(&mut self.runnable, item.slot())?;
            let state = self
                .slots
                .get_mut(item.slot())
                .ok_or(SchedulerError::StaleWork)?
                .prepare_submission()?;
            self.prepared.insert(item.slot(), *item);
            states.push(state);
        }
        Ok(states)
    }

    /// Associate a prepared batch with one backend submission identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown prepared item, duplicate submission, or
    /// lifecycle mismatch.
    pub fn confirm_submission(
        &mut self,
        work: Vec<ScheduledWork>,
        submission: BackendSubmissionId,
    ) -> Result<(), SchedulerError> {
        if work.is_empty() {
            return Err(SchedulerError::EmptyWork);
        }
        if self.in_flight.contains_key(&submission) {
            return Err(SchedulerError::DuplicateSubmission(submission));
        }
        self.validate_prepared(&work)?;

        for item in &work {
            self.slots
                .get_mut(item.slot())
                .ok_or(SchedulerError::StaleWork)?
                .confirm_submission(submission)?;
            self.prepared.remove(&item.slot());
        }
        self.in_flight.insert(submission, work);
        Ok(())
    }

    /// Restore a prepared batch after backend submission failed before taking
    /// ownership of its state.
    ///
    /// # Errors
    ///
    /// Returns an error when work/state identity no longer matches the prepared
    /// scheduler state.
    pub fn fail_prepared(
        &mut self,
        work: &[ScheduledWork],
        states: Vec<InferenceStateSet>,
    ) -> Result<(), SchedulerError> {
        if work.len() != states.len() || work.is_empty() {
            return Err(SchedulerError::StateCountMismatch);
        }
        self.validate_prepared(work)?;

        for (item, state) in work.iter().zip(states) {
            self.slots
                .get_mut(item.slot())
                .ok_or(SchedulerError::StaleWork)?
                .fail_prepared(state)?;
            self.prepared.remove(&item.slot());
            self.terminal.push_back(item.slot());
        }
        self.fill_runnable()?;
        Ok(())
    }

    /// Apply one completed backend batch and restore scheduler-owned state.
    ///
    /// # Errors
    ///
    /// Returns an error when completion identity, state count, or lifecycle no
    /// longer matches the submitted batch.
    pub fn complete_submission(
        &mut self,
        submission: BackendSubmissionId,
        event: &ExecutionBatchEvent,
        states: Vec<InferenceStateSet>,
    ) -> Result<(), SchedulerError> {
        let work = self
            .in_flight
            .get(&submission)
            .cloned()
            .ok_or(SchedulerError::UnknownSubmission(submission))?;
        self.validate_completion(submission, &work, event, &states)?;

        for ((item, completed), state) in work.iter().zip(event.events()).zip(states) {
            self.slots
                .get_mut(item.slot())
                .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?
                .complete_step(
                    submission,
                    completed.phase(),
                    completed.token_count(),
                    completed.output_token(),
                    state,
                )?;

            let lifecycle = self
                .slots
                .get(item.slot())
                .ok_or(SchedulerError::Invariant("completed slot disappeared"))?
                .lifecycle();
            match lifecycle {
                RequestLifecycle::Runnable => {
                    let reached_output_limit = {
                        let slot = self
                            .slots
                            .get(item.slot())
                            .ok_or(SchedulerError::Invariant("completed slot disappeared"))?;
                        slot.progress().generated_tokens()
                            >= slot.request().semantics().max_output_tokens()
                    };
                    if reached_output_limit {
                        self.slots
                            .get_mut(item.slot())
                            .ok_or(SchedulerError::Invariant("completed slot disappeared"))?
                            .mark_completed()?;
                        self.terminal.push_back(item.slot());
                    } else {
                        self.runnable.push_back(item.slot());
                    }
                }
                RequestLifecycle::Cancelled
                | RequestLifecycle::Failed
                | RequestLifecycle::Completed => self.terminal.push_back(item.slot()),
                RequestLifecycle::Waiting
                | RequestLifecycle::Submitting
                | RequestLifecycle::InFlight(_)
                | RequestLifecycle::Cancelling(_) => {
                    return Err(SchedulerError::Invariant(
                        "completion left request in a non-completable lifecycle",
                    ));
                }
            }
        }

        self.in_flight.remove(&submission);
        self.fill_runnable()?;
        Ok(())
    }

    /// Restore state and terminalize every request in a backend submission that
    /// failed after taking ownership.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown submission or state-count mismatch.
    pub fn fail_submission(
        &mut self,
        submission: BackendSubmissionId,
        states: Vec<InferenceStateSet>,
    ) -> Result<(), SchedulerError> {
        let work = self
            .in_flight
            .get(&submission)
            .cloned()
            .ok_or(SchedulerError::UnknownSubmission(submission))?;
        if work.len() != states.len() {
            return Err(SchedulerError::StateCountMismatch);
        }
        for item in &work {
            let lifecycle = self
                .slots
                .get(item.slot())
                .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?
                .lifecycle();
            if !matches!(
                lifecycle,
                RequestLifecycle::InFlight(actual) | RequestLifecycle::Cancelling(actual)
                    if actual == submission
            ) {
                return Err(SchedulerError::SubmissionMismatch);
            }
        }

        for (item, state) in work.iter().zip(states) {
            self.slots
                .get_mut(item.slot())
                .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?
                .fail_submission(submission, state)?;
            self.terminal.push_back(item.slot());
        }
        self.in_flight.remove(&submission);
        self.fill_runnable()?;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error for unknown, terminal, or transiently submitting
    /// requests.
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
        if before == RequestLifecycle::Submitting {
            return Err(SchedulerError::InvalidLifecycle);
        }
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

    fn validate_work_set(&self, work: &[ScheduledWork]) -> Result<(), SchedulerError> {
        let mut seen = HashSet::with_capacity(work.len());
        for item in work {
            if !seen.insert(item.slot()) {
                return Err(SchedulerError::StaleWork);
            }
            if !self
                .runnable
                .iter()
                .any(|candidate| *candidate == item.slot())
            {
                return Err(SchedulerError::StaleWork);
            }
            self.validate_work(*item)?;
        }
        Ok(())
    }

    fn validate_work(&self, work: ScheduledWork) -> Result<(), SchedulerError> {
        let slot = self
            .slots
            .get(work.slot())
            .ok_or(SchedulerError::StaleWork)?;
        if slot.request().id() != work.request()
            || slot.lifecycle() != RequestLifecycle::Runnable
            || slot.state().is_none()
            || Self::phase_for(slot)? != work.phase()
            || Self::state_position(slot)? != work.state_position()
        {
            return Err(SchedulerError::StaleWork);
        }
        match work.phase() {
            ExecutionPhase::Decode if work.token_count() != 1 => Err(SchedulerError::StaleWork),
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
                    Err(SchedulerError::StaleWork)
                } else {
                    Ok(())
                }
            }
            ExecutionPhase::SpecDraft
            | ExecutionPhase::SpecVerify
            | ExecutionPhase::Encoder
            | ExecutionPhase::MoEExpert => Err(SchedulerError::StaleWork),
            ExecutionPhase::Decode => Ok(()),
        }
    }

    fn validate_prepared(&self, work: &[ScheduledWork]) -> Result<(), SchedulerError> {
        let mut seen = HashSet::with_capacity(work.len());
        for item in work {
            if !seen.insert(item.slot())
                || self.prepared.get(&item.slot()) != Some(item)
                || self
                    .slots
                    .get(item.slot())
                    .is_none_or(|slot| slot.lifecycle() != RequestLifecycle::Submitting)
            {
                return Err(SchedulerError::StaleWork);
            }
        }
        Ok(())
    }

    fn validate_completion(
        &self,
        submission: BackendSubmissionId,
        work: &[ScheduledWork],
        event: &ExecutionBatchEvent,
        states: &[InferenceStateSet],
    ) -> Result<(), SchedulerError> {
        if work.len() != event.len() || work.len() != states.len() {
            return Err(SchedulerError::CompletionMismatch);
        }
        for ((item, completed), state) in work.iter().zip(event.events()).zip(states) {
            if completed.request() != item.request()
                || completed.phase() != item.phase()
                || completed.token_count() != item.token_count()
                || completed.policy_version() != self.policy.version()
                || completed.output_token().is_some() != item.requests_output()
            {
                return Err(SchedulerError::CompletionMismatch);
            }
            let expected_position = item
                .state_position()
                .checked_add(item.token_count())
                .ok_or(SchedulerError::PositionOverflow)?;
            if state
                .token_position()
                .is_some_and(|position| position != expected_position)
            {
                return Err(SchedulerError::CompletionMismatch);
            }
            let lifecycle = self
                .slots
                .get(item.slot())
                .ok_or(SchedulerError::Invariant("in-flight slot disappeared"))?
                .lifecycle();
            if !matches!(
                lifecycle,
                RequestLifecycle::InFlight(actual) | RequestLifecycle::Cancelling(actual)
                    if actual == submission
            ) {
                return Err(SchedulerError::SubmissionMismatch);
            }
        }
        Ok(())
    }

    fn active_count(&self) -> usize {
        self.runnable.len() + self.prepared.len() + self.in_flight_request_count()
    }

    fn in_flight_request_count(&self) -> usize {
        self.in_flight.values().map(Vec::len).sum()
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
        if slot.lifecycle() != RequestLifecycle::Runnable || slot.state().is_none() {
            return Err(SchedulerError::Invariant(
                "runnable queue contains request without slot-owned state",
            ));
        }
        Ok(slot)
    }

    fn phase_for(slot: &ActiveRequestSlot) -> Result<ExecutionPhase, SchedulerError> {
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

    fn state_position(slot: &ActiveRequestSlot) -> Result<u32, SchedulerError> {
        let state = slot.state().ok_or(SchedulerError::Invariant(
            "runnable request does not own inference state",
        ))?;
        if let Some(position) = state.token_position() {
            return Ok(position);
        }
        slot.progress()
            .prompt_processed()
            .checked_add(slot.progress().decode_processed())
            .ok_or(SchedulerError::PositionOverflow)
    }

    fn make_work(
        slot: &ActiveRequestSlot,
        phase: ExecutionPhase,
        token_count: u32,
    ) -> Result<ScheduledWork, SchedulerError> {
        let sample_output = match phase {
            ExecutionPhase::Decode => true,
            ExecutionPhase::Prefill => {
                slot.progress()
                    .prompt_processed()
                    .checked_add(token_count)
                    .ok_or(SchedulerError::PositionOverflow)?
                    == slot.progress().prompt_tokens()
            }
            ExecutionPhase::SpecDraft
            | ExecutionPhase::SpecVerify
            | ExecutionPhase::Encoder
            | ExecutionPhase::MoEExpert => false,
        };
        Ok(ScheduledWork {
            slot: slot.id(),
            request: slot.request().id(),
            phase,
            token_count,
            state_position: Self::state_position(slot)?,
            sample_output,
        })
    }
}

fn remove_id(queue: &mut VecDeque<RequestSlotId>, id: RequestSlotId) -> Result<(), SchedulerError> {
    let index =
        queue
            .iter()
            .position(|candidate| *candidate == id)
            .ok_or(SchedulerError::Invariant(
                "request was absent from expected queue",
            ))?;
    queue.remove(index);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SchedulerError {
    InvalidConfig,
    EmptyWork,
    Backpressure,
    DuplicateRequest(RequestId),
    UnknownRequest(RequestId),
    UnknownSubmission(BackendSubmissionId),
    DuplicateSubmission(BackendSubmissionId),
    StateCountMismatch,
    SubmissionMismatch,
    CompletionMismatch,
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
            Self::EmptyWork => f.write_str("scheduler submission must contain work"),
            Self::Backpressure => f.write_str("scheduler admission capacity is exhausted"),
            Self::DuplicateRequest(id) => write!(f, "request {} is already admitted", id.get()),
            Self::UnknownRequest(id) => write!(f, "request {} is unknown", id.get()),
            Self::UnknownSubmission(id) => write!(f, "submission {} is unknown", id.get()),
            Self::DuplicateSubmission(id) => {
                write!(f, "submission {} is already in flight", id.get())
            }
            Self::StateCountMismatch => {
                f.write_str("scheduler work and inference-state counts differ")
            }
            Self::SubmissionMismatch => {
                f.write_str("request lifecycle does not match backend submission ownership")
            }
            Self::CompletionMismatch => {
                f.write_str("backend completion does not match scheduled work")
            }
            Self::InvalidLifecycle => {
                f.write_str("request lifecycle does not permit this operation")
            }
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
    use crate::execution::{ExecutionEvent, ExecutionMetrics};
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

    fn batch_event(work: &[ScheduledWork]) -> ExecutionBatchEvent {
        ExecutionBatchEvent::new(
            work.iter()
                .map(|item| {
                    let event = ExecutionEvent::new(
                        item.request(),
                        PolicyVersion::new(1).expect("policy version"),
                        item.phase(),
                        item.token_count(),
                        ExecutionMetrics::new(1, 0, 0),
                    )
                    .expect("event");
                    if item.requests_output() {
                        event.with_output_token(7)
                    } else {
                        event
                    }
                })
                .collect(),
        )
        .expect("batch event")
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
        scheduler
            .admit(request(2, 4), state(), 10)
            .expect("prefill");
        scheduler
            .admit(request(3, 4), state(), 10)
            .expect("prefill");

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
    fn one_submission_owns_multiple_request_slots() {
        let config = SchedulerConfig::new(2, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(2, 2), config);
        scheduler.admit(request(1, 2), state(), 0).expect("first");
        scheduler.admit(request(2, 2), state(), 0).expect("second");
        let work = scheduler.schedule().expect("schedule");
        let states = scheduler.prepare_submission(&work).expect("prepare");
        assert_eq!(scheduler.counts().prepared(), 2);
        assert_eq!(states.len(), 2);
        let submission = BackendSubmissionId::new(1).expect("submission");
        scheduler
            .confirm_submission(work.clone(), submission)
            .expect("confirm");
        assert_eq!(scheduler.counts().prepared(), 0);
        assert_eq!(scheduler.counts().in_flight(), 2);

        scheduler
            .complete_submission(submission, &batch_event(&work), vec![state(), state()])
            .expect("complete");
        assert_eq!(scheduler.counts().in_flight(), 0);
        assert_eq!(scheduler.counts().runnable(), 2);
    }

    #[test]
    fn cancelling_one_request_in_batch_does_not_cancel_its_peer() {
        let config = SchedulerConfig::new(2, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(2, 2), config);
        let first_id = RequestId::new(1).expect("request ID");
        scheduler.admit(request(1, 2), state(), 0).expect("first");
        scheduler.admit(request(2, 2), state(), 0).expect("second");
        let work = scheduler.schedule().expect("schedule");
        let _states = scheduler.prepare_submission(&work).expect("prepare");
        let submission = BackendSubmissionId::new(1).expect("submission");
        scheduler
            .confirm_submission(work.clone(), submission)
            .expect("confirm");
        scheduler.cancel(first_id).expect("cancel first");

        scheduler
            .complete_submission(submission, &batch_event(&work), vec![state(), state()])
            .expect("complete");
        assert_eq!(scheduler.counts().terminal(), 1);
        assert_eq!(scheduler.counts().runnable(), 1);
    }

    #[test]
    fn failed_prepare_restores_state_and_terminalizes_requests() {
        let config = SchedulerConfig::new(2, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(2, 2), config);
        scheduler.admit(request(1, 2), state(), 0).expect("first");
        scheduler.admit(request(2, 2), state(), 0).expect("second");
        let work = scheduler.schedule().expect("schedule");
        let states = scheduler.prepare_submission(&work).expect("prepare");
        scheduler
            .fail_prepared(&work, states)
            .expect("fail prepared");
        assert_eq!(scheduler.counts().prepared(), 0);
        assert_eq!(scheduler.counts().terminal(), 2);
        assert!(
            scheduler
                .reclaim_next()
                .expect("reclaim")
                .expect("terminal")
                .state()
                .is_some()
        );
    }

    #[test]
    fn final_prefill_output_does_not_advance_decode_input_position() {
        let config = SchedulerConfig::new(1, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(1, 4), config);
        let request_id = RequestId::new(1).expect("request ID");
        scheduler.admit(request(1, 2), state(), 3).expect("admit");

        let prefill = scheduler.schedule().expect("schedule prefill");
        assert_eq!(prefill.len(), 1);
        assert_eq!(prefill[0].phase(), ExecutionPhase::Prefill);
        assert_eq!(prefill[0].token_count(), 3);
        assert!(prefill[0].requests_output());
        let _states = scheduler.prepare_submission(&prefill).expect("prepare");
        let submission = BackendSubmissionId::new(1).expect("submission");
        scheduler
            .confirm_submission(prefill.clone(), submission)
            .expect("confirm");
        scheduler
            .complete_submission(submission, &batch_event(&prefill), vec![state()])
            .expect("complete prefill");

        let slot_id = scheduler.slot_for_request(request_id).expect("slot");
        let slot = scheduler.slots().get(slot_id).expect("request");
        assert_eq!(slot.progress().prompt_processed(), 3);
        assert_eq!(slot.progress().decode_processed(), 0);
        assert_eq!(slot.progress().generated_tokens(), 1);

        let decode = scheduler.schedule().expect("schedule decode");
        assert_eq!(decode.len(), 1);
        assert_eq!(decode[0].phase(), ExecutionPhase::Decode);
        assert_eq!(decode[0].state_position(), 3);
        assert!(decode[0].requests_output());
    }

    #[test]
    fn mismatched_completion_does_not_mutate_in_flight_slots() {
        let config = SchedulerConfig::new(1, 0, 4).expect("config");
        let mut scheduler = ServingScheduler::new(policy(1, 4), config);
        scheduler.admit(request(1, 2), state(), 0).expect("admit");
        let work = scheduler.schedule().expect("schedule");
        let _states = scheduler.prepare_submission(&work).expect("prepare");
        let submission = BackendSubmissionId::new(1).expect("submission");
        scheduler
            .confirm_submission(work.clone(), submission)
            .expect("confirm");
        let wrong = ExecutionBatchEvent::new(vec![
            ExecutionEvent::new(
                work[0].request(),
                PolicyVersion::new(1).expect("policy version"),
                ExecutionPhase::Prefill,
                1,
                ExecutionMetrics::new(1, 0, 0),
            )
            .expect("event"),
        ])
        .expect("batch event");
        assert_eq!(
            scheduler.complete_submission(submission, &wrong, vec![state()]),
            Err(SchedulerError::CompletionMismatch)
        );
        assert_eq!(scheduler.counts().in_flight(), 1);
    }
}
