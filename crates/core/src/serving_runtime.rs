//! Coordination between persistent request scheduling and execution ownership.

use std::collections::HashMap;
use std::fmt;

use crate::backend::{BackendSubmissionId, ComputeBackend};
use crate::execution::{ExecutionBatch, ExecutionPlan, ExecutionSegment, PlanError};
use crate::model::ModelProvider;
use crate::request::{RequestId, RequestSpec};
use crate::runtime::{ExecutionRuntime, RuntimeError, RuntimeSubmission};
use crate::scheduler::{ScheduledWork, SchedulerError, ServingScheduler};
use crate::serving::ActiveRequestSlot;
use crate::state::{InferenceStateSet, StateManager};

pub struct ServingRuntime<P, B, S> {
    scheduler: ServingScheduler,
    runtime: ExecutionRuntime<P, B, S>,
    plan: ExecutionPlan,
    submissions: HashMap<BackendSubmissionId, RuntimeSubmission>,
}

impl<P, B, S> ServingRuntime<P, B, S>
where
    P: ModelProvider,
    B: ComputeBackend,
    S: StateManager,
{
    /// # Errors
    ///
    /// Returns [`ServingRuntimeError::PolicyMismatch`] when the scheduler and
    /// prepared execution plan do not use the same policy snapshot version.
    pub fn new(
        scheduler: ServingScheduler,
        runtime: ExecutionRuntime<P, B, S>,
        plan: ExecutionPlan,
    ) -> Result<Self, ServingRuntimeError> {
        if scheduler.policy().version() != plan.policy_version() {
            return Err(ServingRuntimeError::PolicyMismatch);
        }
        Ok(Self {
            scheduler,
            runtime,
            plan,
            submissions: HashMap::new(),
        })
    }

    #[must_use]
    pub const fn scheduler(&self) -> &ServingScheduler {
        &self.scheduler
    }

    #[must_use]
    pub const fn runtime(&self) -> &ExecutionRuntime<P, B, S> {
        &self.runtime
    }

    #[must_use]
    pub const fn runtime_mut(&mut self) -> &mut ExecutionRuntime<P, B, S> {
        &mut self.runtime
    }

    #[must_use]
    pub const fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    #[must_use]
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    /// # Errors
    ///
    /// Returns a scheduler error when admission is backpressured or invalid.
    pub fn admit(
        &mut self,
        request: RequestSpec,
        state: InferenceStateSet,
        prompt_tokens: u32,
    ) -> Result<crate::serving::RequestSlotId, ServingRuntimeError> {
        Ok(self.scheduler.admit(request, state, prompt_tokens)?)
    }

    /// # Errors
    ///
    /// Returns a scheduler error for unknown or non-cancellable requests.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), ServingRuntimeError> {
        Ok(self.scheduler.cancel(request)?)
    }

    /// # Errors
    ///
    /// Returns a scheduler error unless the request is runnable.
    pub fn finish(&mut self, request: RequestId) -> Result<(), ServingRuntimeError> {
        Ok(self.scheduler.finish(request)?)
    }

    /// # Errors
    ///
    /// Returns a scheduler error when terminal bookkeeping is inconsistent.
    pub fn reclaim_next(&mut self) -> Result<Option<ActiveRequestSlot>, ServingRuntimeError> {
        Ok(self.scheduler.reclaim_next()?)
    }

    /// Schedule and submit one multi-request batch if runnable work exists.
    /// Submission failure restores state to the scheduler before returning.
    ///
    /// # Errors
    ///
    /// Returns a plan, scheduler, or runtime submission error.
    pub fn submit_ready_batch(
        &mut self,
    ) -> Result<Option<BackendSubmissionId>, ServingRuntimeError> {
        let work = self.scheduler.schedule()?;
        if work.is_empty() {
            return Ok(None);
        }
        let batch = self.execution_batch(&work)?;
        let states = self.scheduler.prepare_submission(&work)?;
        let submission = match self.runtime.submit_batch(&self.plan, &batch, states) {
            Ok(submission) => submission,
            Err(error) => {
                let (runtime_error, states) = error.into_parts();
                self.scheduler.fail_prepared(&work, states)?;
                return Err(ServingRuntimeError::Runtime(runtime_error));
            }
        };
        let id = submission.backend_submission();
        if self.submissions.contains_key(&id) {
            return Err(ServingRuntimeError::SubmissionCollision(id));
        }
        self.submissions.insert(id, submission);
        self.scheduler.confirm_submission(work, id)?;
        Ok(Some(id))
    }

    /// Poll every live backend submission once. Completed state is restored to
    /// request slots before this method reports the completion count.
    ///
    /// # Errors
    ///
    /// Returns a runtime or scheduler error. When runtime polling fails before
    /// logical state is consumed, the affected requests are terminalized with
    /// their recovered state.
    pub fn poll_completions(&mut self) -> Result<usize, ServingRuntimeError> {
        let ids = self.submissions.keys().copied().collect::<Vec<_>>();
        let mut completed_count = 0;

        for id in ids {
            let mut submission = self
                .submissions
                .remove(&id)
                .ok_or(ServingRuntimeError::SubmissionMissing(id))?;
            match self.runtime.poll_submission(&mut submission) {
                Ok(None) => {
                    self.submissions.insert(id, submission);
                }
                Ok(Some(completed)) => {
                    let (event, states) = completed.into_parts();
                    self.scheduler.complete_submission(id, &event, states)?;
                    completed_count += 1;
                }
                Err(error) => match submission.take_uncommitted_states() {
                    Ok(states) => {
                        self.scheduler.fail_submission(id, states)?;
                        return Err(ServingRuntimeError::Runtime(error));
                    }
                    Err(_) => return Err(ServingRuntimeError::Runtime(error)),
                },
            }
        }
        Ok(completed_count)
    }

    /// Poll completions first, then submit the next ready batch.
    ///
    /// # Errors
    ///
    /// Returns any completion or submission error from the serving loop.
    pub fn drive_once(&mut self) -> Result<ServingIteration, ServingRuntimeError> {
        let completed_batches = self.poll_completions()?;
        let submitted = self.submit_ready_batch()?;
        Ok(ServingIteration {
            completed_batches,
            submitted,
        })
    }

    fn execution_batch(
        &self,
        work: &[ScheduledWork],
    ) -> Result<ExecutionBatch, ServingRuntimeError> {
        let segments = work
            .iter()
            .map(|item| {
                let mut segment = ExecutionSegment::new(
                    item.request(),
                    item.phase(),
                    1,
                    item.token_count(),
                    item.state_position(),
                    self.plan.state_requirements().to_vec(),
                )?;
                if item.requests_output() {
                    let slot = self
                        .scheduler
                        .slots()
                        .get(item.slot())
                        .ok_or(SchedulerError::StaleWork)?;
                    segment = segment.with_sampling(slot.request().semantics().sampling());
                }
                Ok(segment)
            })
            .collect::<Result<Vec<_>, ServingRuntimeError>>()?;
        Ok(ExecutionBatch::new(segments)?)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ServingIteration {
    completed_batches: usize,
    submitted: Option<BackendSubmissionId>,
}

impl ServingIteration {
    #[must_use]
    pub const fn completed_batches(self) -> usize {
        self.completed_batches
    }

    #[must_use]
    pub const fn submitted(self) -> Option<BackendSubmissionId> {
        self.submitted
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServingRuntimeError {
    PolicyMismatch,
    SubmissionCollision(BackendSubmissionId),
    SubmissionMissing(BackendSubmissionId),
    Plan(PlanError),
    Scheduler(SchedulerError),
    Runtime(RuntimeError),
}

impl fmt::Display for ServingRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyMismatch => {
                f.write_str("scheduler and execution plan use different policy versions")
            }
            Self::SubmissionCollision(id) => {
                write!(f, "backend reused live submission identity {}", id.get())
            }
            Self::SubmissionMissing(id) => {
                write!(f, "serving submission {} disappeared", id.get())
            }
            Self::Plan(error) => write!(f, "serving execution batch is invalid: {error}"),
            Self::Scheduler(error) => write!(f, "serving scheduler failed: {error}"),
            Self::Runtime(error) => write!(f, "serving execution failed: {error}"),
        }
    }
}

impl std::error::Error for ServingRuntimeError {}

impl From<PlanError> for ServingRuntimeError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<SchedulerError> for ServingRuntimeError {
    fn from(error: SchedulerError) -> Self {
        Self::Scheduler(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        BackendCapabilities, BackendError, BackendFeatures, BackendId, BackendKind,
    };
    use crate::device::DeviceId;
    use crate::execution::{
        ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionPhase, ExecutionStage,
    };
    use crate::model::{
        ModelCapabilities, ModelDescription, ModelId, ModelRegion, ModelRegionId, ModelRegionKind,
        WeightDescription,
    };
    use crate::policy::{PolicySnapshot, PolicyVersion, SpeculationPolicy, StateTierPreference};
    use crate::request::{RequestSemantics, SamplingParams, ThinkingMode};
    use crate::scheduler::SchedulerConfig;
    use crate::state::LogicalStateManager;
    use crate::tensor::{Quantization, WeightFormat};
    use crate::weights::WeightBinding;

    struct TestProvider {
        description: ModelDescription,
    }

    impl ModelProvider for TestProvider {
        fn description(&self) -> &ModelDescription {
            &self.description
        }
    }

    struct DelayedBackend {
        capabilities: BackendCapabilities,
        next_submission: u64,
        pending: HashMap<BackendSubmissionId, (u8, ExecutionBatchEvent)>,
        fail_submit: bool,
    }

    impl DelayedBackend {
        fn new(capabilities: BackendCapabilities, fail_submit: bool) -> Self {
            Self {
                capabilities,
                next_submission: 1,
                pending: HashMap::new(),
                fail_submit,
            }
        }
    }

    impl ComputeBackend for DelayedBackend {
        fn capabilities(&self) -> &BackendCapabilities {
            &self.capabilities
        }

        fn submit(
            &mut self,
            plan: &ExecutionPlan,
            batch: &ExecutionBatch,
            states: &mut [InferenceStateSet],
        ) -> Result<BackendSubmissionId, BackendError> {
            self.validate_execution(plan, batch, states)?;
            if self.fail_submit {
                return Err(BackendError::ExecutionFailed(
                    "test submit failure".to_owned(),
                ));
            }
            let id = BackendSubmissionId::new(self.next_submission)
                .ok_or_else(|| BackendError::ExecutionFailed("submission overflow".to_owned()))?;
            self.next_submission = self
                .next_submission
                .checked_add(1)
                .ok_or_else(|| BackendError::ExecutionFailed("submission overflow".to_owned()))?;
            let events = batch
                .segments()
                .iter()
                .map(|segment| {
                    let event = ExecutionEvent::new(
                        segment.request(),
                        plan.policy_version(),
                        segment.phase(),
                        segment.token_count(),
                        ExecutionMetrics::new(10, 0, 0),
                    )
                    .expect("non-zero segment");
                    if segment.requests_sampling() {
                        event.with_output_token(7)
                    } else {
                        event
                    }
                })
                .collect();
            let event = ExecutionBatchEvent::new(events).expect("batch event");
            self.pending.insert(id, (1, event));
            Ok(id)
        }

        fn poll(
            &mut self,
            submission: BackendSubmissionId,
        ) -> Result<Option<ExecutionBatchEvent>, BackendError> {
            let Some((remaining, _)) = self.pending.get_mut(&submission) else {
                return Err(BackendError::UnknownSubmission(submission));
            };
            if *remaining > 0 {
                *remaining -= 1;
                return Ok(None);
            }
            let (_, event) = self
                .pending
                .remove(&submission)
                .ok_or(BackendError::UnknownSubmission(submission))?;
            Ok(Some(event))
        }
    }

    fn fixture(
        fail_submit: bool,
    ) -> ServingRuntime<TestProvider, DelayedBackend, LogicalStateManager> {
        let device = DeviceId::new(0);
        let model = ModelId::new("test/model").expect("model ID");
        let backend_id = BackendId::new("test-backend").expect("backend ID");
        let version = PolicyVersion::new(1).expect("policy version");
        let description = ModelDescription::new(
            model.clone(),
            "test",
            vec![ModelRegion::new(
                ModelRegionId::new(0),
                ModelRegionKind::FullAttention,
            )],
            Vec::new(),
            ModelCapabilities::new(None, false),
            WeightDescription::new(WeightFormat::Gguf, Quantization::None),
        )
        .expect("description");
        let plan = ExecutionPlan::new(
            model,
            backend_id.clone(),
            device,
            version,
            vec![
                ExecutionStage::new(ModelRegionId::new(0), ExecutionPhase::Prefill),
                ExecutionStage::new(ModelRegionId::new(0), ExecutionPhase::Decode),
            ],
            Vec::new(),
            WeightBinding::empty(ModelId::new("test/model").expect("model ID"), device),
        )
        .expect("plan");
        let policy = PolicySnapshot::new(
            version,
            2,
            4,
            StateTierPreference::Automatic,
            SpeculationPolicy::Disabled,
        )
        .expect("policy");
        let scheduler = ServingScheduler::new(
            policy,
            SchedulerConfig::new(2, 2, 3).expect("scheduler config"),
        );
        let capabilities = BackendCapabilities::new(
            backend_id,
            device,
            BackendKind::Cpu,
            1024,
            BackendFeatures::new(Vec::new(), Vec::new(), false, true),
        );
        let runtime = ExecutionRuntime::new(
            TestProvider { description },
            DelayedBackend::new(capabilities, fail_submit),
            LogicalStateManager::new(device, 0, 0),
        );
        ServingRuntime::new(scheduler, runtime, plan).expect("serving runtime")
    }

    fn request(id: u64) -> RequestSpec {
        RequestSpec::new(
            RequestId::new(id).expect("request ID"),
            ModelId::new("test/model").expect("model ID"),
            RequestSemantics::new(2, SamplingParams::greedy(Some(id)), ThinkingMode::Off)
                .expect("semantics"),
        )
    }

    fn state() -> InferenceStateSet {
        InferenceStateSet::new(Vec::new()).expect("state")
    }

    #[test]
    fn one_runtime_submission_owns_multiple_slots_until_completion() {
        let mut serving = fixture(false);
        let first = RequestId::new(1).expect("request ID");
        let second = RequestId::new(2).expect("request ID");
        serving.admit(request(1), state(), 0).expect("first");
        serving.admit(request(2), state(), 0).expect("second");

        let submission = serving
            .submit_ready_batch()
            .expect("submit")
            .expect("submission");
        assert_eq!(serving.submission_count(), 1);
        assert_eq!(serving.scheduler().counts().in_flight(), 2);
        for request in [first, second] {
            let slot = serving.scheduler().slot_for_request(request).expect("slot");
            assert!(
                serving
                    .scheduler()
                    .slots()
                    .get(slot)
                    .expect("request")
                    .state()
                    .is_none()
            );
        }

        assert_eq!(serving.poll_completions().expect("pending poll"), 0);
        assert_eq!(serving.submission_count(), 1);
        assert_eq!(serving.poll_completions().expect("completion poll"), 1);
        assert_eq!(serving.submission_count(), 0);
        assert_eq!(serving.scheduler().counts().runnable(), 2);
        assert_eq!(serving.scheduler().counts().in_flight(), 0);
        assert!(submission.get() > 0);
    }

    #[test]
    fn cancelling_one_request_does_not_cancel_peer_in_same_submission() {
        let mut serving = fixture(false);
        let first = RequestId::new(1).expect("request ID");
        serving.admit(request(1), state(), 0).expect("first");
        serving.admit(request(2), state(), 0).expect("second");
        serving.submit_ready_batch().expect("submit");
        serving.cancel(first).expect("cancel");
        serving.poll_completions().expect("pending poll");
        serving.poll_completions().expect("completion poll");
        assert_eq!(serving.scheduler().counts().terminal(), 1);
        assert_eq!(serving.scheduler().counts().runnable(), 1);
    }

    #[test]
    fn failed_backend_submit_restores_states_before_terminalizing() {
        let mut serving = fixture(true);
        serving.admit(request(1), state(), 0).expect("first");
        serving.admit(request(2), state(), 0).expect("second");
        assert!(matches!(
            serving.submit_ready_batch(),
            Err(ServingRuntimeError::Runtime(RuntimeError::Backend(_)))
        ));
        assert_eq!(serving.scheduler().counts().prepared(), 0);
        assert_eq!(serving.scheduler().counts().terminal(), 2);
        assert!(
            serving
                .reclaim_next()
                .expect("reclaim")
                .expect("terminal request")
                .state()
                .is_some()
        );
    }
}
