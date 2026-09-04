//! Persistent request-slot bookkeeping for the serving runtime.

use std::fmt;

use crate::backend::BackendSubmissionId;
use crate::execution::ExecutionPhase;
use crate::request::{RequestId, RequestSpec};
use crate::state::InferenceStateSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestSlotId {
    index: u32,
    generation: u32,
}

impl RequestSlotId {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RequestLifecycle {
    Waiting,
    Runnable,
    Submitting,
    InFlight(BackendSubmissionId),
    Cancelling(BackendSubmissionId),
    Completed,
    Cancelled,
    Failed,
}

impl RequestLifecycle {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct RequestProgress {
    prompt_tokens: u32,
    prompt_processed: u32,
    generated_tokens: u32,
}

impl RequestProgress {
    #[must_use]
    pub const fn new(prompt_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            prompt_processed: 0,
            generated_tokens: 0,
        }
    }

    #[must_use]
    pub const fn prompt_tokens(self) -> u32 {
        self.prompt_tokens
    }

    #[must_use]
    pub const fn prompt_processed(self) -> u32 {
        self.prompt_processed
    }

    #[must_use]
    pub const fn generated_tokens(self) -> u32 {
        self.generated_tokens
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::ProgressOverflow`] when counters overflow or
    /// prefill work exceeds the declared prompt length.
    pub fn record(
        &mut self,
        phase: ExecutionPhase,
        token_count: u32,
    ) -> Result<(), RequestSlotError> {
        match phase {
            ExecutionPhase::Prefill => {
                let next = self
                    .prompt_processed
                    .checked_add(token_count)
                    .ok_or(RequestSlotError::ProgressOverflow)?;
                if next > self.prompt_tokens {
                    return Err(RequestSlotError::ProgressOverflow);
                }
                self.prompt_processed = next;
            }
            ExecutionPhase::Decode => {
                self.generated_tokens = self
                    .generated_tokens
                    .checked_add(token_count)
                    .ok_or(RequestSlotError::ProgressOverflow)?;
            }
            ExecutionPhase::SpecDraft
            | ExecutionPhase::SpecVerify
            | ExecutionPhase::Encoder
            | ExecutionPhase::MoEExpert => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActiveRequestSlot {
    id: RequestSlotId,
    request: RequestSpec,
    state: Option<InferenceStateSet>,
    lifecycle: RequestLifecycle,
    progress: RequestProgress,
}

impl ActiveRequestSlot {
    #[must_use]
    pub const fn id(&self) -> RequestSlotId {
        self.id
    }

    #[must_use]
    pub const fn request(&self) -> &RequestSpec {
        &self.request
    }

    #[must_use]
    pub const fn state(&self) -> Option<&InferenceStateSet> {
        self.state.as_ref()
    }

    #[must_use]
    pub fn state_mut(&mut self) -> Option<&mut InferenceStateSet> {
        self.state.as_mut()
    }

    #[must_use]
    pub const fn lifecycle(&self) -> RequestLifecycle {
        self.lifecycle
    }

    #[must_use]
    pub const fn progress(&self) -> RequestProgress {
        self.progress
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] unless the request is
    /// currently waiting.
    pub fn make_runnable(&mut self) -> Result<(), RequestSlotError> {
        if self.lifecycle != RequestLifecycle::Waiting || self.state.is_none() {
            return Err(RequestSlotError::InvalidTransition);
        }
        self.lifecycle = RequestLifecycle::Runnable;
        Ok(())
    }

    /// Transfer logical state out of the slot before backend submission.
    ///
    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] unless the request is
    /// runnable and owns its state.
    pub fn prepare_submission(&mut self) -> Result<InferenceStateSet, RequestSlotError> {
        if self.lifecycle != RequestLifecycle::Runnable {
            return Err(RequestSlotError::InvalidTransition);
        }
        let state = self.state.take().ok_or(RequestSlotError::StateUnavailable)?;
        self.lifecycle = RequestLifecycle::Submitting;
        Ok(state)
    }

    /// Associate a successfully submitted request with backend ownership.
    ///
    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] unless the request is
    /// prepared and its logical state is outside the slot.
    pub fn confirm_submission(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<(), RequestSlotError> {
        if self.lifecycle != RequestLifecycle::Submitting || self.state.is_some() {
            return Err(RequestSlotError::InvalidTransition);
        }
        self.lifecycle = RequestLifecycle::InFlight(submission);
        Ok(())
    }

    /// Restore state after submission failed before backend ownership existed.
    ///
    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] unless the request is
    /// prepared and has no state in the slot.
    pub fn fail_prepared(&mut self, state: InferenceStateSet) -> Result<(), RequestSlotError> {
        if self.lifecycle != RequestLifecycle::Submitting || self.state.is_some() {
            return Err(RequestSlotError::InvalidTransition);
        }
        self.state = Some(state);
        self.lifecycle = RequestLifecycle::Failed;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] for terminal or
    /// transiently submitting requests.
    pub fn request_cancel(&mut self) -> Result<(), RequestSlotError> {
        self.lifecycle = match self.lifecycle {
            RequestLifecycle::Waiting | RequestLifecycle::Runnable => RequestLifecycle::Cancelled,
            RequestLifecycle::InFlight(submission) | RequestLifecycle::Cancelling(submission) => {
                RequestLifecycle::Cancelling(submission)
            }
            RequestLifecycle::Submitting
            | RequestLifecycle::Completed
            | RequestLifecycle::Cancelled
            | RequestLifecycle::Failed => {
                return Err(RequestSlotError::InvalidTransition);
            }
        };
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] unless the request is
    /// runnable and owns its state.
    pub fn mark_completed(&mut self) -> Result<(), RequestSlotError> {
        if self.lifecycle != RequestLifecycle::Runnable || self.state.is_none() {
            return Err(RequestSlotError::InvalidTransition);
        }
        self.lifecycle = RequestLifecycle::Completed;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] while a backend
    /// submission still owns request state or after terminal completion.
    pub fn mark_failed(&mut self) -> Result<(), RequestSlotError> {
        match self.lifecycle {
            RequestLifecycle::Waiting | RequestLifecycle::Runnable if self.state.is_some() => {
                self.lifecycle = RequestLifecycle::Failed;
                Ok(())
            }
            RequestLifecycle::Waiting
            | RequestLifecycle::Runnable
            | RequestLifecycle::Submitting
            | RequestLifecycle::InFlight(_)
            | RequestLifecycle::Cancelling(_)
            | RequestLifecycle::Completed
            | RequestLifecycle::Cancelled
            | RequestLifecycle::Failed => Err(RequestSlotError::InvalidTransition),
        }
    }

    /// Restore state after a backend submission reports terminal failure.
    ///
    /// # Errors
    ///
    /// Returns [`RequestSlotError::SubmissionMismatch`] when a different
    /// submission owns the slot.
    pub fn fail_submission(
        &mut self,
        submission: BackendSubmissionId,
        state: InferenceStateSet,
    ) -> Result<(), RequestSlotError> {
        match self.lifecycle {
            RequestLifecycle::InFlight(actual) | RequestLifecycle::Cancelling(actual)
                if actual == submission && self.state.is_none() =>
            {
                self.state = Some(state);
                self.lifecycle = RequestLifecycle::Failed;
                Ok(())
            }
            _ => Err(RequestSlotError::SubmissionMismatch),
        }
    }

    /// Apply one completed backend step and restore scheduler-owned state.
    ///
    /// # Errors
    ///
    /// Returns [`RequestSlotError`] when the completed work does not match the
    /// owning submission or progress cannot be advanced safely.
    pub fn complete_step(
        &mut self,
        submission: BackendSubmissionId,
        phase: ExecutionPhase,
        token_count: u32,
        state: InferenceStateSet,
    ) -> Result<(), RequestSlotError> {
        if self.state.is_some() {
            return Err(RequestSlotError::StateUnavailable);
        }
        match self.lifecycle {
            RequestLifecycle::InFlight(actual) if actual == submission => {
                self.progress.record(phase, token_count)?;
                self.state = Some(state);
                self.lifecycle = RequestLifecycle::Runnable;
                Ok(())
            }
            RequestLifecycle::Cancelling(actual) if actual == submission => {
                self.state = Some(state);
                self.lifecycle = RequestLifecycle::Cancelled;
                Ok(())
            }
            _ => Err(RequestSlotError::SubmissionMismatch),
        }
    }
}

#[derive(Debug, Default)]
pub struct RequestSlots {
    slots: Vec<SlotCell>,
    free: Vec<u32>,
}

#[derive(Debug)]
struct SlotCell {
    generation: u32,
    value: Option<ActiveRequestSlot>,
}

impl RequestSlots {
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.value.is_some())
            .count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::DuplicateRequest`] when the request is
    /// already active or [`RequestSlotError::SlotOverflow`] when the table
    /// cannot represent another stable slot.
    pub fn insert(
        &mut self,
        request: RequestSpec,
        state: InferenceStateSet,
        prompt_tokens: u32,
    ) -> Result<RequestSlotId, RequestSlotError> {
        if self
            .slots
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .any(|slot| slot.request().id() == request.id())
        {
            return Err(RequestSlotError::DuplicateRequest(request.id()));
        }

        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            let index =
                u32::try_from(self.slots.len()).map_err(|_| RequestSlotError::SlotOverflow)?;
            self.slots.push(SlotCell {
                generation: 0,
                value: None,
            });
            index
        };
        let cell = self
            .slots
            .get_mut(usize::try_from(index).map_err(|_| RequestSlotError::SlotOverflow)?)
            .ok_or(RequestSlotError::UnknownSlot)?;
        cell.generation = cell
            .generation
            .checked_add(1)
            .ok_or(RequestSlotError::SlotOverflow)?;
        let id = RequestSlotId {
            index,
            generation: cell.generation,
        };
        cell.value = Some(ActiveRequestSlot {
            id,
            request,
            state: Some(state),
            lifecycle: RequestLifecycle::Waiting,
            progress: RequestProgress::new(prompt_tokens),
        });
        Ok(id)
    }

    #[must_use]
    pub fn get(&self, id: RequestSlotId) -> Option<&ActiveRequestSlot> {
        let cell = self.slots.get(usize::try_from(id.index).ok()?)?;
        if cell.generation != id.generation {
            return None;
        }
        cell.value.as_ref()
    }

    #[must_use]
    pub fn get_mut(&mut self, id: RequestSlotId) -> Option<&mut ActiveRequestSlot> {
        let cell = self.slots.get_mut(usize::try_from(id.index).ok()?)?;
        if cell.generation != id.generation {
            return None;
        }
        cell.value.as_mut()
    }

    /// # Errors
    ///
    /// Returns [`RequestSlotError::UnknownSlot`] for a stale/unknown slot or
    /// [`RequestSlotError::RequestStillActive`] when removal is attempted
    /// before the request reaches a terminal state.
    pub fn remove(&mut self, id: RequestSlotId) -> Result<ActiveRequestSlot, RequestSlotError> {
        let cell = self
            .slots
            .get_mut(usize::try_from(id.index).map_err(|_| RequestSlotError::UnknownSlot)?)
            .ok_or(RequestSlotError::UnknownSlot)?;
        if cell.generation != id.generation {
            return Err(RequestSlotError::UnknownSlot);
        }
        let value = cell.value.as_ref().ok_or(RequestSlotError::UnknownSlot)?;
        if !value.lifecycle().is_terminal() {
            return Err(RequestSlotError::RequestStillActive);
        }
        if value.state().is_none() {
            return Err(RequestSlotError::StateUnavailable);
        }
        let value = cell.value.take().ok_or(RequestSlotError::UnknownSlot)?;
        self.free.push(id.index);
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RequestSlotError {
    DuplicateRequest(RequestId),
    UnknownSlot,
    SlotOverflow,
    InvalidTransition,
    SubmissionMismatch,
    StateUnavailable,
    ProgressOverflow,
    RequestStillActive,
}

impl fmt::Display for RequestSlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRequest(id) => write!(f, "request {} is already active", id.get()),
            Self::UnknownSlot => f.write_str("request slot is unknown or stale"),
            Self::SlotOverflow => f.write_str("request slot table overflowed"),
            Self::InvalidTransition => f.write_str("request lifecycle transition is invalid"),
            Self::SubmissionMismatch => {
                f.write_str("completed submission does not own this request")
            }
            Self::StateUnavailable => f.write_str("request inference state is not slot-owned"),
            Self::ProgressOverflow => f.write_str("request progress is invalid or overflowed"),
            Self::RequestStillActive => f.write_str("request slot cannot be removed while active"),
        }
    }
}

impl std::error::Error for RequestSlotError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_work_does_not_claim_committed_output() {
        let mut progress = RequestProgress::new(8);
        progress
            .record(ExecutionPhase::Prefill, 8)
            .expect("prefill");
        progress
            .record(ExecutionPhase::SpecDraft, 4)
            .expect("draft");
        progress
            .record(ExecutionPhase::SpecVerify, 4)
            .expect("verify");
        assert_eq!(progress.generated_tokens(), 0);
        progress.record(ExecutionPhase::Decode, 1).expect("decode");
        assert_eq!(progress.generated_tokens(), 1);
    }
}
