//! Stateful execution orchestration shared by local and serving frontends.

use std::fmt;

use crate::backend::{BackendError, BackendSubmissionId, ComputeBackend};
use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionPlan, ExecutionSegment, PlanError,
};
use crate::model::{ModelError, ModelProvider};
use crate::policy::PolicyVersion;
use crate::state::{InferenceStateSet, StateError, StateManager};

pub struct ExecutionRuntime<P, B, S> {
    provider: P,
    backend: B,
    state_manager: S,
}

#[derive(Debug)]
pub struct RuntimeSubmission {
    backend_submission: BackendSubmissionId,
    batch: ExecutionBatch,
    policy_version: PolicyVersion,
    states: Option<Vec<InferenceStateSet>>,
    next_positions: Vec<u32>,
}

impl RuntimeSubmission {
    #[must_use]
    pub const fn backend_submission(&self) -> BackendSubmissionId {
        self.backend_submission
    }

    #[must_use]
    pub const fn batch(&self) -> &ExecutionBatch {
        &self.batch
    }

    #[must_use]
    pub const fn policy_version(&self) -> PolicyVersion {
        self.policy_version
    }

    /// Recover uncommitted logical state after a terminal submission failure.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::SubmissionConsumed`] after completion or an
    /// earlier recovery already consumed the state.
    pub fn take_uncommitted_states(&mut self) -> Result<Vec<InferenceStateSet>, RuntimeError> {
        self.states.take().ok_or(RuntimeError::SubmissionConsumed)
    }
}

#[derive(Debug)]
pub struct RuntimeStateError {
    error: RuntimeError,
    states: Vec<InferenceStateSet>,
}

impl RuntimeStateError {
    fn new(error: RuntimeError, states: Vec<InferenceStateSet>) -> Self {
        Self { error, states }
    }

    #[must_use]
    pub const fn error(&self) -> &RuntimeError {
        &self.error
    }

    #[must_use]
    pub fn states(&self) -> &[InferenceStateSet] {
        &self.states
    }

    #[must_use]
    pub fn into_parts(self) -> (RuntimeError, Vec<InferenceStateSet>) {
        (self.error, self.states)
    }
}

impl fmt::Display for RuntimeStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime execution failed: {}", self.error)
    }
}

impl std::error::Error for RuntimeStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[derive(Debug)]
pub struct CompletedExecutionBatch {
    event: ExecutionBatchEvent,
    states: Vec<InferenceStateSet>,
}

impl CompletedExecutionBatch {
    #[must_use]
    pub const fn event(&self) -> &ExecutionBatchEvent {
        &self.event
    }

    #[must_use]
    pub fn states(&self) -> &[InferenceStateSet] {
        &self.states
    }

    #[must_use]
    pub fn into_parts(self) -> (ExecutionBatchEvent, Vec<InferenceStateSet>) {
        (self.event, self.states)
    }
}

#[derive(Debug)]
pub struct CompletedExecution {
    event: ExecutionEvent,
    state: InferenceStateSet,
}

impl CompletedExecution {
    #[must_use]
    pub const fn event(&self) -> ExecutionEvent {
        self.event
    }

    #[must_use]
    pub const fn state(&self) -> &InferenceStateSet {
        &self.state
    }

    #[must_use]
    pub fn into_parts(self) -> (ExecutionEvent, InferenceStateSet) {
        (self.event, self.state)
    }
}

impl<P, B, S> ExecutionRuntime<P, B, S>
where
    P: ModelProvider,
    B: ComputeBackend,
    S: StateManager,
{
    #[must_use]
    pub const fn new(provider: P, backend: B, state_manager: S) -> Self {
        Self {
            provider,
            backend,
            state_manager,
        }
    }

    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    #[must_use]
    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[must_use]
    pub const fn state_manager(&self) -> &S {
        &self.state_manager
    }

    #[must_use]
    pub const fn state_manager_mut(&mut self) -> &mut S {
        &mut self.state_manager
    }

    /// Release backend-owned physical state, then release the logical state
    /// identities and capacity owned by one finished request.
    ///
    /// Keeping this ordering preserves the logical handles needed by a backend
    /// to find its physical allocations. If physical release fails, logical
    /// identity is retained rather than silently orphaning device memory.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Backend`] when physical state release fails or
    /// [`RuntimeError::State`] when a logical state handle cannot be released.
    pub fn release_state_set(&mut self, state: &InferenceStateSet) -> Result<(), RuntimeError> {
        self.backend
            .release_inference_state(state)
            .map_err(RuntimeError::Backend)?;
        self.state_manager.release_set(state)?;
        Ok(())
    }

    /// Submit a batch while preserving caller-owned logical state if model,
    /// plan, arithmetic, or backend submission validation fails.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeStateError`] with the uncommitted states when the
    /// backend never takes submission ownership.
    pub fn submit_batch(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        mut states: Vec<InferenceStateSet>,
    ) -> Result<RuntimeSubmission, RuntimeStateError> {
        if let Err(error) = self.provider.validate_plan(plan) {
            return Err(RuntimeStateError::new(RuntimeError::Model(error), states));
        }
        if batch.len() != states.len() {
            return Err(RuntimeStateError::new(
                RuntimeError::StateCountMismatch,
                states,
            ));
        }

        for state in &states {
            if let Err(error) = self.state_manager.validate(state) {
                return Err(RuntimeStateError::new(RuntimeError::State(error), states));
            }
        }

        let mut next_positions = Vec::with_capacity(batch.len());
        for segment in batch.segments() {
            let Some(position) = segment.state_position().checked_add(segment.token_count()) else {
                return Err(RuntimeStateError::new(
                    RuntimeError::PositionOverflow,
                    states,
                ));
            };
            next_positions.push(position);
        }

        let backend_submission = match self.backend.submit(plan, batch, &mut states) {
            Ok(submission) => submission,
            Err(error) => {
                return Err(RuntimeStateError::new(RuntimeError::Backend(error), states));
            }
        };
        Ok(RuntimeSubmission {
            backend_submission,
            batch: batch.clone(),
            policy_version: plan.policy_version(),
            states: Some(states),
            next_positions,
        })
    }

    /// # Errors
    ///
    /// Returns [`RuntimeStateError`] from batch construction or submission,
    /// preserving the supplied state.
    pub fn submit_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        state: InferenceStateSet,
    ) -> Result<RuntimeSubmission, RuntimeStateError> {
        let states = vec![state];
        let batch = match ExecutionBatch::new(vec![segment.clone()]) {
            Ok(batch) => batch,
            Err(error) => {
                return Err(RuntimeStateError::new(RuntimeError::Plan(error), states));
            }
        };
        self.submit_batch(plan, &batch, states)
    }

    /// Poll a submitted batch. Logical state positions are committed only
    /// after the backend completion is visible and matches the submitted work.
    /// A backend polling error is terminal: the submission retains its states
    /// so the serving owner can recover them with `take_uncommitted_states`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when completion polling, identity validation,
    /// or state commitment fails.
    pub fn poll_submission(
        &mut self,
        submission: &mut RuntimeSubmission,
    ) -> Result<Option<CompletedExecutionBatch>, RuntimeError> {
        let Some(event) = self.backend.poll(submission.backend_submission)? else {
            return Ok(None);
        };
        Self::validate_completion(submission, &event)?;
        let states = submission
            .states
            .as_mut()
            .ok_or(RuntimeError::SubmissionConsumed)?;
        self.state_manager
            .commit_batch(states, &submission.next_positions)?;
        let committed = submission
            .states
            .take()
            .ok_or(RuntimeError::SubmissionConsumed)?;
        Ok(Some(CompletedExecutionBatch {
            event,
            states: committed,
        }))
    }

    /// # Errors
    ///
    /// Returns [`RuntimeError`] from completion or state commitment.
    pub fn wait_submission(
        &mut self,
        mut submission: RuntimeSubmission,
    ) -> Result<CompletedExecutionBatch, RuntimeStateError> {
        loop {
            match self.poll_submission(&mut submission) {
                Ok(Some(completed)) => return Ok(completed),
                Ok(None) => std::thread::yield_now(),
                Err(error) => {
                    return Err(RuntimeStateError::new(
                        error,
                        submission.states.take().unwrap_or_default(),
                    ));
                }
            }
        }
    }

    /// Execute a batch while retaining allocation ownership on every failure.
    ///
    /// # Errors
    ///
    /// Returns the runtime error and states for terminal cleanup.
    pub fn execute_batch(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: Vec<InferenceStateSet>,
    ) -> Result<CompletedExecutionBatch, RuntimeStateError> {
        let submission = self.submit_batch(plan, batch, states)?;
        self.wait_submission(submission)
    }

    /// Execute one segment while retaining allocation ownership on failure.
    ///
    /// # Errors
    ///
    /// Returns the runtime error and states for terminal cleanup.
    pub fn execute_segment(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        state: InferenceStateSet,
    ) -> Result<CompletedExecution, RuntimeStateError> {
        let submission = self.submit_segment(plan, segment, state)?;
        let completed = self.wait_submission(submission)?;
        let (batch_event, mut states) = completed.into_parts();
        let mut events = batch_event.events().iter().copied();
        let Some(event) = events.next() else {
            return Err(RuntimeStateError::new(
                RuntimeError::CompletionMismatch,
                states,
            ));
        };
        if events.next().is_some() || states.len() != 1 {
            return Err(RuntimeStateError::new(
                RuntimeError::CompletionMismatch,
                states,
            ));
        }
        let Some(state) = states.pop() else {
            return Err(RuntimeStateError::new(
                RuntimeError::CompletionMismatch,
                states,
            ));
        };
        Ok(CompletedExecution { event, state })
    }

    fn validate_completion(
        submission: &RuntimeSubmission,
        event: &ExecutionBatchEvent,
    ) -> Result<(), RuntimeError> {
        if event.len() != submission.batch.len() {
            return Err(RuntimeError::CompletionMismatch);
        }
        for (segment, completed) in submission.batch.segments().iter().zip(event.events()) {
            if segment.request() != completed.request()
                || segment.phase() != completed.phase()
                || segment.token_count() != completed.token_count()
                || completed.policy_version() != submission.policy_version
                || completed.output_token().is_some() != segment.requests_sampling()
            {
                return Err(RuntimeError::CompletionMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    Model(ModelError),
    Backend(BackendError),
    State(StateError),
    Plan(PlanError),
    StateCountMismatch,
    PositionOverflow,
    SubmissionConsumed,
    CompletionMismatch,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "model validation failed: {error}"),
            Self::Backend(error) => write!(f, "backend execution failed: {error}"),
            Self::State(error) => write!(f, "state commitment failed: {error}"),
            Self::Plan(error) => write!(f, "execution batch is invalid: {error}"),
            Self::StateCountMismatch => {
                f.write_str("execution batch and inference-state counts differ")
            }
            Self::PositionOverflow => f.write_str("execution position overflowed"),
            Self::SubmissionConsumed => f.write_str("runtime submission was already consumed"),
            Self::CompletionMismatch => {
                f.write_str("backend completion does not match submitted execution batch")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ModelError> for RuntimeError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<BackendError> for RuntimeError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<StateError> for RuntimeError {
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}

impl From<PlanError> for RuntimeError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}
