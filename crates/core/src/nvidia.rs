//! NVIDIA backend adapter without a CUDA dependency in the core crate.

use std::collections::HashMap;

use crate::backend::{
    BackendCapabilities, BackendError, BackendKind, BackendSubmissionId, ComputeBackend,
};
use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionOutcome, ExecutionPlan,
    ExecutionSegment,
};
use crate::policy::PolicyVersion;
use crate::state::InferenceStateSet;
use crate::weights::WeightBinding;

pub trait NvidiaDispatcher: Send {
    /// Dispatch one request segment.
    ///
    /// This remains the compatibility primitive for simple/reference
    /// dispatchers. Production dispatchers can override [`Self::dispatch_batch`]
    /// to consume the scheduler's whole multi-request batch in one backend
    /// operation.
    ///
    /// # Errors
    ///
    /// Returns a backend error when CUDA dispatch or state mutation fails.
    fn dispatch(
        &mut self,
        plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        weights: &WeightBinding,
        state: &mut InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError>;

    /// Release dispatcher-owned physical state associated with one logical
    /// inference-state set. Stateless/reference dispatchers use the no-op.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the dispatcher cannot release the state.
    fn release_inference_state(&mut self, _state: &InferenceStateSet) -> Result<(), BackendError> {
        Ok(())
    }

    /// Dispatch one scheduler-selected multi-request batch synchronously.
    ///
    /// The default implementation preserves reference/single-request
    /// dispatchers by visiting each segment in order. CUDA implementations
    /// can override this to consume the whole batch in one backend operation.
    /// Async-capable dispatchers should instead override [`Self::submit_batch`]
    /// and [`Self::poll_batch`].
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::StateCountMismatch`] when request/state counts
    /// differ, or propagates a segment dispatch failure. A synchronous
    /// implementation must not return while device work can still access the
    /// supplied request state.
    fn dispatch_batch(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Vec<ExecutionOutcome>, BackendError> {
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        batch
            .segments()
            .iter()
            .zip(states)
            .map(|(segment, state)| self.dispatch(plan, segment, weights, state))
            .collect()
    }

    /// Submit one scheduler batch under the backend-owned submission identity.
    ///
    /// The compatibility implementation executes immediately and returns its
    /// outcomes. A genuinely asynchronous dispatcher returns `Ok(None)` after
    /// queueing device work and later exposes outcomes from [`Self::poll_batch`].
    /// The submission ID is stable across both calls and can key CUDA events,
    /// pinned output slots, or other backend-local completion state.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the batch cannot be accepted. Returning an
    /// error means the dispatcher did not retain asynchronous ownership: no
    /// queued work may still access the supplied request state or transient
    /// submission resources after this method returns.
    fn submit_batch(
        &mut self,
        _submission: BackendSubmissionId,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        self.dispatch_batch(plan, batch, weights, states).map(Some)
    }

    /// Poll outcomes for a batch previously accepted asynchronously.
    ///
    /// Synchronous dispatchers never reach this method. Async dispatchers
    /// return `Ok(None)` while device work is pending and `Ok(Some(...))` once
    /// all outcomes for the scheduler batch are ready.
    ///
    /// # Errors
    ///
    /// Returns a backend error for terminal completion failure. An error is a
    /// terminal result for this submission: before returning it, the dispatcher
    /// must have relinquished all asynchronous access to request state and
    /// submission-local resources so the runtime may safely reclaim them.
    fn poll_batch(
        &mut self,
        _submission: BackendSubmissionId,
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        Err(BackendError::Unsupported(
            "asynchronous NVIDIA dispatcher completion",
        ))
    }
}

#[derive(Clone, Debug)]
struct PendingNvidiaSubmission {
    batch: ExecutionBatch,
    policy_version: PolicyVersion,
}

pub struct NvidiaBackend<D> {
    capabilities: BackendCapabilities,
    dispatcher: D,
    next_submission: u64,
    completed: HashMap<BackendSubmissionId, ExecutionBatchEvent>,
    pending: HashMap<BackendSubmissionId, PendingNvidiaSubmission>,
}

impl<D> NvidiaBackend<D> {
    /// # Errors
    ///
    /// Returns [`BackendError::Unsupported`] when the supplied capabilities do
    /// not describe a CUDA backend.
    pub fn new(capabilities: BackendCapabilities, dispatcher: D) -> Result<Self, BackendError> {
        if capabilities.kind() != BackendKind::Cuda {
            return Err(BackendError::Unsupported("CUDA backend capabilities"));
        }
        Ok(Self {
            capabilities,
            dispatcher,
            next_submission: 1,
            completed: HashMap::new(),
            pending: HashMap::new(),
        })
    }

    #[must_use]
    pub fn dispatcher(&self) -> &D {
        &self.dispatcher
    }

    #[must_use]
    pub fn dispatcher_mut(&mut self) -> &mut D {
        &mut self.dispatcher
    }

    #[must_use]
    pub fn into_dispatcher(self) -> D {
        self.dispatcher
    }

    fn allocate_submission(&mut self) -> Result<BackendSubmissionId, BackendError> {
        let id = BackendSubmissionId::new(self.next_submission).ok_or_else(|| {
            BackendError::ExecutionFailed("submission identity overflowed".to_owned())
        })?;
        self.next_submission = self.next_submission.checked_add(1).ok_or_else(|| {
            BackendError::ExecutionFailed("submission identity overflowed".to_owned())
        })?;
        Ok(id)
    }
}

fn completion_event(
    batch: &ExecutionBatch,
    policy_version: PolicyVersion,
    outcomes: Vec<ExecutionOutcome>,
) -> Result<ExecutionBatchEvent, BackendError> {
    if outcomes.len() != batch.len() {
        return Err(BackendError::ExecutionFailed(format!(
            "NVIDIA dispatcher returned {} outcomes for {} request segments",
            outcomes.len(),
            batch.len()
        )));
    }

    let events = batch
        .segments()
        .iter()
        .zip(outcomes)
        .map(|(segment, outcome)| {
            let output_token = outcome.output_token();
            if segment.requests_sampling() != output_token.is_some() {
                return Err(BackendError::ExecutionFailed(format!(
                    "NVIDIA dispatcher output-token presence did not match sampling for request {}",
                    segment.request().get()
                )));
            }
            let event = ExecutionEvent::new(
                segment.request(),
                policy_version,
                segment.phase(),
                segment.token_count(),
                outcome.metrics(),
            )
            .ok_or_else(|| {
                BackendError::ExecutionFailed("segment contained no tokens".to_owned())
            })?;
            Ok(match output_token {
                Some(token) => event.with_output_token(token),
                None => event,
            })
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    ExecutionBatchEvent::new(events).map_err(BackendError::InvalidPlan)
}

impl<D: NvidiaDispatcher> ComputeBackend for NvidiaBackend<D> {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn release_inference_state(&mut self, state: &InferenceStateSet) -> Result<(), BackendError> {
        self.dispatcher.release_inference_state(state)
    }

    fn submit(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<BackendSubmissionId, BackendError> {
        self.validate_execution(plan, batch, states)?;
        let submission = self.allocate_submission()?;
        let result =
            self.dispatcher
                .submit_batch(submission, plan, batch, plan.weights(), states)?;
        match result {
            Some(outcomes) => {
                let event = completion_event(batch, plan.policy_version(), outcomes)?;
                self.completed.insert(submission, event);
            }
            None => {
                self.pending.insert(
                    submission,
                    PendingNvidiaSubmission {
                        batch: batch.clone(),
                        policy_version: plan.policy_version(),
                    },
                );
            }
        }
        Ok(submission)
    }

    fn poll(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<ExecutionBatchEvent>, BackendError> {
        if let Some(event) = self.completed.remove(&submission) {
            return Ok(Some(event));
        }
        if !self.pending.contains_key(&submission) {
            return Err(BackendError::UnknownSubmission(submission));
        }
        let outcomes = match self.dispatcher.poll_batch(submission) {
            Ok(None) => return Ok(None),
            Ok(Some(outcomes)) => outcomes,
            Err(error) => {
                self.pending.remove(&submission);
                return Err(error);
            }
        };
        let pending = self
            .pending
            .remove(&submission)
            .ok_or(BackendError::UnknownSubmission(submission))?;
        completion_event(&pending.batch, pending.policy_version, outcomes).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendFeatures, BackendId};
    use crate::device::DeviceId;
    use crate::execution::{ExecutionMetrics, ExecutionPhase, ExecutionStage};
    use crate::model::{ModelId, ModelRegionId};
    use crate::policy::PolicyVersion;
    use crate::request::{RequestId, SamplingParams};
    use crate::state::{
        InferenceStateSet, KvStateSpec, LogicalStateManager, StateLocation, StateManager,
        StateRequirement,
    };
    use crate::tensor::{DataType, Quantization};
    use std::collections::HashSet;

    struct TestDispatcher;

    impl NvidiaDispatcher for TestDispatcher {
        fn dispatch(
            &mut self,
            _plan: &ExecutionPlan,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            Ok(ExecutionOutcome::new(ExecutionMetrics::new(12, 4, 8)))
        }
    }

    #[derive(Default)]
    struct BatchAwareDispatcher {
        batches: usize,
        segments: usize,
        releases: usize,
    }

    impl NvidiaDispatcher for BatchAwareDispatcher {
        fn dispatch(
            &mut self,
            _plan: &ExecutionPlan,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            self.segments += 1;
            Ok(ExecutionOutcome::new(ExecutionMetrics::new(99, 0, 0)))
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
            self.batches += 1;
            Ok((0..batch.len())
                .map(|_| ExecutionOutcome::new(ExecutionMetrics::new(7, 0, 0)))
                .collect())
        }

        fn release_inference_state(
            &mut self,
            _state: &InferenceStateSet,
        ) -> Result<(), BackendError> {
            self.releases += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    struct AsyncDispatcher {
        submits: usize,
        polls: usize,
        pending: HashMap<BackendSubmissionId, usize>,
    }

    impl NvidiaDispatcher for AsyncDispatcher {
        fn dispatch(
            &mut self,
            _plan: &ExecutionPlan,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            Err(BackendError::ExecutionFailed(
                "async dispatcher should not use synchronous dispatch".to_owned(),
            ))
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
            self.submits += 1;
            self.pending.insert(submission, batch.len());
            Ok(None)
        }

        fn poll_batch(
            &mut self,
            submission: BackendSubmissionId,
        ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
            self.polls += 1;
            let count = self
                .pending
                .remove(&submission)
                .ok_or(BackendError::UnknownSubmission(submission))?;
            Ok(Some(
                (0..count)
                    .map(|_| ExecutionOutcome::new(ExecutionMetrics::new(5, 0, 0)))
                    .collect(),
            ))
        }
    }

    #[derive(Default)]
    struct FailingAsyncDispatcher {
        pending: HashSet<BackendSubmissionId>,
    }

    impl NvidiaDispatcher for FailingAsyncDispatcher {
        fn dispatch(
            &mut self,
            _plan: &ExecutionPlan,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            unreachable!("failing async dispatcher does not synchronously dispatch")
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
            self.pending.insert(submission);
            Ok(None)
        }

        fn poll_batch(
            &mut self,
            submission: BackendSubmissionId,
        ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
            self.pending
                .remove(&submission)
                .ok_or(BackendError::UnknownSubmission(submission))?;
            Err(BackendError::ExecutionFailed(
                "terminal async failure".to_owned(),
            ))
        }
    }

    fn empty_decode_fixture<D: NvidiaDispatcher>(
        dispatcher: D,
    ) -> (
        NvidiaBackend<D>,
        ExecutionPlan,
        ExecutionBatch,
        Vec<InferenceStateSet>,
    ) {
        let device = DeviceId::new(0);
        let backend_id = BackendId::new("cuda").expect("backend ID");
        let capabilities = BackendCapabilities::new(
            backend_id.clone(),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let backend = NvidiaBackend::new(capabilities, dispatcher).expect("CUDA backend");
        let model = ModelId::new("test-model").expect("model ID");
        let plan = ExecutionPlan::new(
            model.clone(),
            backend_id,
            device,
            PolicyVersion::new(1).expect("policy version"),
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            Vec::new(),
            WeightBinding::empty(model, device),
        )
        .expect("plan");
        let batch = ExecutionBatch::new(vec![
            ExecutionSegment::new(
                RequestId::new(1).expect("request ID"),
                ExecutionPhase::Decode,
                1,
                1,
                0,
                Vec::new(),
            )
            .expect("segment"),
        ])
        .expect("batch");
        let states = vec![InferenceStateSet::new(Vec::new()).expect("state")];
        (backend, plan, batch, states)
    }

    #[test]
    fn backend_dispatches_only_after_capability_and_state_validation() {
        let device = DeviceId::new(0);
        let capabilities = BackendCapabilities::new(
            BackendId::new("cuda").expect("backend ID"),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(
                vec![DataType::F16],
                vec![Quantization::GgufQ4Km],
                false,
                true,
            ),
        );
        let mut backend = NvidiaBackend::new(capabilities, TestDispatcher).expect("CUDA backend");
        let model = ModelId::new("test-model").expect("model ID");
        let policy = PolicyVersion::new(1).expect("policy version");
        let requirement = StateRequirement::FullAttentionKv(
            KvStateSpec::new(1, 1, 2, 1, DataType::F16).expect("KV spec"),
        );
        let plan = ExecutionPlan::new(
            model.clone(),
            BackendId::new("cuda").expect("backend ID"),
            device,
            policy,
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            vec![requirement],
            WeightBinding::empty(model.clone(), device),
        )
        .expect("plan");
        let request = RequestId::new(1).expect("request ID");
        let segment =
            ExecutionSegment::new(request, ExecutionPhase::Decode, 1, 1, 0, vec![requirement])
                .expect("segment");
        let batch = ExecutionBatch::new(vec![segment]).expect("batch");
        let spec = match requirement {
            StateRequirement::FullAttentionKv(spec) => spec,
            StateRequirement::Recurrent(_) => unreachable!(),
        };
        let mut manager = LogicalStateManager::new(device, 1024, 0);
        let kv = manager
            .allocate_kv(spec, StateLocation::Device(device))
            .expect("state allocation");
        let state = InferenceStateSet::try_new(Some(kv), None).expect("state set");
        let submission = backend.submit(&plan, &batch, &mut [state]).expect("submit");
        let event = backend.wait(submission).expect("completion");
        assert_eq!(event.events()[0].metrics().elapsed_nanos(), 12);
        assert_eq!(event.events()[0].policy_version(), policy);
    }

    #[test]
    fn backend_rejects_missing_requested_output_token() {
        let device = DeviceId::new(0);
        let backend_id = BackendId::new("cuda").expect("backend ID");
        let capabilities = BackendCapabilities::new(
            backend_id.clone(),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let mut backend = NvidiaBackend::new(capabilities, TestDispatcher).expect("CUDA backend");
        let model = ModelId::new("test-model").expect("model ID");
        let plan = ExecutionPlan::new(
            model.clone(),
            backend_id,
            device,
            PolicyVersion::new(1).expect("policy version"),
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            Vec::new(),
            WeightBinding::empty(model, device),
        )
        .expect("plan");
        let segment = ExecutionSegment::new(
            RequestId::new(1).expect("request ID"),
            ExecutionPhase::Decode,
            1,
            1,
            0,
            Vec::new(),
        )
        .expect("segment")
        .with_sampling(SamplingParams::greedy(None));
        let batch = ExecutionBatch::new(vec![segment]).expect("batch");
        let mut states = vec![InferenceStateSet::new(Vec::new()).expect("state")];

        assert!(matches!(
            backend.submit(&plan, &batch, &mut states),
            Err(BackendError::ExecutionFailed(_))
        ));
    }

    #[test]
    fn backend_forwards_state_release_to_the_dispatcher() {
        let device = DeviceId::new(0);
        let capabilities = BackendCapabilities::new(
            BackendId::new("cuda").expect("backend ID"),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let mut backend = NvidiaBackend::new(capabilities, BatchAwareDispatcher::default())
            .expect("CUDA backend");
        let state = InferenceStateSet::new(Vec::new()).expect("state");

        backend
            .release_inference_state(&state)
            .expect("release state");
        assert_eq!(backend.dispatcher().releases, 1);
    }

    #[test]
    fn backend_forwards_the_whole_scheduler_batch_to_the_dispatcher() {
        let device = DeviceId::new(0);
        let backend_id = BackendId::new("cuda").expect("backend ID");
        let capabilities = BackendCapabilities::new(
            backend_id.clone(),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let mut backend = NvidiaBackend::new(capabilities, BatchAwareDispatcher::default())
            .expect("CUDA backend");
        let model = ModelId::new("test-model").expect("model ID");
        let policy = PolicyVersion::new(1).expect("policy version");
        let plan = ExecutionPlan::new(
            model.clone(),
            backend_id,
            device,
            policy,
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            Vec::new(),
            WeightBinding::empty(model, device),
        )
        .expect("plan");
        let batch = ExecutionBatch::new(vec![
            ExecutionSegment::new(
                RequestId::new(1).expect("request ID"),
                ExecutionPhase::Decode,
                1,
                1,
                0,
                Vec::new(),
            )
            .expect("segment"),
            ExecutionSegment::new(
                RequestId::new(2).expect("request ID"),
                ExecutionPhase::Decode,
                1,
                1,
                0,
                Vec::new(),
            )
            .expect("segment"),
        ])
        .expect("batch");
        let empty_state = || InferenceStateSet::new(Vec::new()).expect("state");
        let mut states = vec![empty_state(), empty_state()];

        let submission = backend.submit(&plan, &batch, &mut states).expect("submit");
        let completed = backend.wait(submission).expect("completion");

        assert_eq!(completed.len(), 2);
        assert!(
            completed
                .events()
                .iter()
                .all(|event| event.metrics().elapsed_nanos() == 7)
        );
        assert_eq!(backend.dispatcher().batches, 1);
        assert_eq!(backend.dispatcher().segments, 0);
    }

    #[test]
    fn backend_supports_dispatcher_owned_async_completion() {
        let (mut backend, plan, batch, mut states) =
            empty_decode_fixture(AsyncDispatcher::default());

        let submission = backend.submit(&plan, &batch, &mut states).expect("submit");
        assert_eq!(backend.dispatcher().submits, 1);
        assert_eq!(backend.dispatcher().polls, 0);

        let completed = backend
            .poll(submission)
            .expect("poll async dispatcher")
            .expect("completion");
        assert_eq!(completed.events()[0].metrics().elapsed_nanos(), 5);
        assert_eq!(backend.dispatcher().polls, 1);
        assert!(backend.pending.is_empty());
    }

    #[test]
    fn terminal_async_failure_releases_backend_pending_metadata() {
        let (mut backend, plan, batch, mut states) =
            empty_decode_fixture(FailingAsyncDispatcher::default());
        let submission = backend.submit(&plan, &batch, &mut states).expect("submit");
        assert_eq!(backend.pending.len(), 1);

        assert!(matches!(
            backend.poll(submission),
            Err(BackendError::ExecutionFailed(_))
        ));
        assert!(backend.pending.is_empty());
        assert!(backend.dispatcher().pending.is_empty());
        assert!(matches!(
            backend.poll(submission),
            Err(BackendError::UnknownSubmission(actual)) if actual == submission
        ));
    }
}
