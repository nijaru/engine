use super::*;
use crate::backend::{BackendCapabilities, BackendError, BackendFeatures, BackendId, BackendKind};
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
    submitted_inputs: Vec<Vec<ExecutionTokenInput>>,
    fail_submit: bool,
    fail_release: bool,
}

impl DelayedBackend {
    fn new(capabilities: BackendCapabilities, fail_submit: bool) -> Self {
        Self {
            capabilities,
            next_submission: 1,
            pending: HashMap::new(),
            submitted_inputs: Vec::new(),
            fail_submit,
            fail_release: false,
        }
    }
}

impl ComputeBackend for DelayedBackend {
    fn release_inference_state(&mut self, _state: &InferenceStateSet) -> Result<(), BackendError> {
        if self.fail_release {
            return Err(BackendError::ExecutionFailed(
                "injected physical release failure".into(),
            ));
        }
        Ok(())
    }

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
        let inputs = batch
            .segments()
            .iter()
            .map(|segment| {
                segment.token_input().cloned().ok_or_else(|| {
                    BackendError::ExecutionFailed(
                        "test serving segment lacked token input".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.submitted_inputs.push(inputs);
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

fn fixture(fail_submit: bool) -> ServingRuntime<TestProvider, DelayedBackend, FaultManager> {
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
        FaultManager {
            inner: LogicalStateManager::new(device, 1024, 0),
            fail_commit: false,
            fail_release: false,
        },
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

fn prompt(tokens: &[u32]) -> Arc<[u32]> {
    Arc::from(tokens)
}

#[test]
fn one_runtime_submission_owns_multiple_slots_until_completion() {
    let mut serving = fixture(false);
    let first = RequestId::new(1).expect("request ID");
    let second = RequestId::new(2).expect("request ID");
    serving
        .admit(request(1), state(), prompt(&[11]))
        .expect("first");
    serving
        .admit(request(2), state(), prompt(&[12]))
        .expect("second");

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
fn sampled_prefill_token_becomes_the_next_decode_input() {
    let mut serving = fixture(false);
    serving
        .admit(request(1), state(), prompt(&[11, 12]))
        .expect("request");

    serving.submit_ready_batch().expect("prefill submit");
    assert_eq!(
        serving.runtime().backend().submitted_inputs[0][0].prompt_slice(),
        Some(&[11, 12][..])
    );
    assert_eq!(serving.poll_completions().expect("pending poll"), 0);
    assert_eq!(serving.poll_completions().expect("prefill completion"), 1);
    assert_eq!(serving.generated_token_count(), 1);
    assert_eq!(
        serving.pop_generated_token(),
        Some(GeneratedToken::new(
            RequestId::new(1).expect("request ID"),
            7
        ))
    );
    assert_eq!(serving.generated_token_count(), 0);

    serving.submit_ready_batch().expect("decode submit");
    assert_eq!(
        serving.runtime().backend().submitted_inputs[1][0].decode_token(),
        Some(7)
    );
}

#[test]
fn cancelling_one_request_does_not_cancel_peer_in_same_submission() {
    let mut serving = fixture(false);
    let first = RequestId::new(1).expect("request ID");
    serving
        .admit(request(1), state(), prompt(&[11]))
        .expect("first");
    serving
        .admit(request(2), state(), prompt(&[12]))
        .expect("second");
    serving.submit_ready_batch().expect("submit");
    serving.cancel(first).expect("cancel");
    serving.poll_completions().expect("pending poll");
    serving.poll_completions().expect("completion poll");
    assert_eq!(serving.scheduler().counts().terminal(), 1);
    assert_eq!(serving.scheduler().counts().runnable(), 1);
    assert_eq!(serving.generated_token_count(), 1);
    assert_eq!(
        serving.pop_generated_token(),
        Some(GeneratedToken::new(
            RequestId::new(2).expect("request ID"),
            7
        ))
    );
    assert_eq!(serving.generated_token_count(), 0);
}

#[test]
fn terminal_reclaim_releases_logical_state_capacity() {
    let mut serving = fixture(false);
    let device = DeviceId::new(0);
    let spec =
        crate::state::KvStateSpec::new(1, 1, 2, 4, crate::tensor::DataType::F16).expect("KV spec");
    let kv = serving
        .runtime_mut()
        .state_manager_mut()
        .allocate_kv(spec, crate::state::StateLocation::Device(device))
        .expect("KV allocation");
    let state = InferenceStateSet::try_new(Some(kv), None).expect("state set");
    assert!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(device))
            .is_some_and(|bytes| bytes > 0)
    );

    let request_id = RequestId::new(1).expect("request ID");
    serving
        .admit(request(1), state, prompt(&[11]))
        .expect("admit");
    serving.cancel(request_id).expect("cancel");
    let reclaimed = serving
        .reclaim_next()
        .expect("reclaim")
        .expect("terminal request");

    assert!(reclaimed.state().is_none());
    assert_eq!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(device)),
        Some(0)
    );
}

#[test]
fn failed_backend_submit_restores_states_before_terminalizing() {
    let mut serving = fixture(true);
    serving
        .admit(request(1), state(), prompt(&[11]))
        .expect("first");
    serving
        .admit(request(2), state(), prompt(&[12]))
        .expect("second");
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
            .is_none()
    );
}

#[derive(Debug)]
struct FaultManager {
    inner: LogicalStateManager,
    fail_commit: bool,
    fail_release: bool,
}
impl FaultManager {
    fn used_bytes(&self, location: crate::state::StateLocation) -> Option<u64> {
        self.inner.used_bytes(location)
    }
}
impl StateManager for FaultManager {
    fn validate(&self, state: &InferenceStateSet) -> Result<(), crate::state::StateError> {
        self.inner.validate(state)
    }
    fn allocate_kv(
        &mut self,
        spec: crate::state::KvStateSpec,
        location: crate::state::StateLocation,
    ) -> Result<crate::state::KvState, crate::state::StateError> {
        self.inner.allocate_kv(spec, location)
    }
    fn allocate_recurrent(
        &mut self,
        spec: crate::state::RecurrentStateSpec,
        location: crate::state::StateLocation,
    ) -> Result<crate::state::RecurrentState, crate::state::StateError> {
        self.inner.allocate_recurrent(spec, location)
    }
    fn commit_batch(
        &mut self,
        states: &mut [InferenceStateSet],
        positions: &[u32],
    ) -> Result<(), crate::state::StateError> {
        if self.fail_commit {
            return Err(crate::state::StateError::PositionRegression);
        }
        self.inner.commit_batch(states, positions)
    }
    fn release_set(&mut self, state: &InferenceStateSet) -> Result<(), crate::state::StateError> {
        if self.fail_release {
            return Err(crate::state::StateError::UnsupportedLocation);
        }
        self.inner.release_set(state)
    }
    fn release(
        &mut self,
        handle: crate::state::StateHandle,
    ) -> Result<(), crate::state::StateError> {
        self.inner.release(handle)
    }
}

fn allocated_state(
    serving: &mut ServingRuntime<TestProvider, DelayedBackend, FaultManager>,
) -> InferenceStateSet {
    let spec =
        crate::state::KvStateSpec::new(1, 1, 2, 4, crate::tensor::DataType::F16).expect("spec");
    let kv = serving
        .runtime_mut()
        .state_manager_mut()
        .allocate_kv(spec, crate::state::StateLocation::Device(DeviceId::new(0)))
        .expect("allocation");
    InferenceStateSet::try_new(Some(kv), None).expect("state")
}

#[test]
fn missing_sampled_output_terminalizes_and_releases_request() {
    let mut serving = fixture(false);
    let request_id = RequestId::new(1).expect("request ID");
    let state = allocated_state(&mut serving);
    serving
        .admit(request(1), state, prompt(&[11]))
        .expect("admit");
    let id = serving.submit_ready_batch().expect("submit").expect("id");
    let event = ExecutionEvent::new(
        request_id,
        serving.plan().policy_version(),
        ExecutionPhase::Prefill,
        1,
        ExecutionMetrics::new(10, 0, 0),
    )
    .expect("event");
    serving.runtime_mut().backend_mut().pending.insert(
        id,
        (0, ExecutionBatchEvent::new(vec![event]).expect("batch")),
    );
    assert!(matches!(
        serving.poll_completions(),
        Err(ServingRuntimeError::Runtime(
            RuntimeError::CompletionMismatch
        ))
    ));
    assert_eq!(serving.submission_count(), 0);
    assert_eq!(serving.scheduler().counts().in_flight(), 0);
    assert_eq!(serving.scheduler().counts().terminal(), 1);
    assert_eq!(serving.generated_token_count(), 0);
    assert!(serving.reclaim_next().expect("reclaim").is_some());
    assert_eq!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(DeviceId::new(0))),
        Some(0)
    );
}

#[test]
fn batch_commit_failure_recovers_every_state_for_reclamation() {
    let mut serving = fixture(false);
    for id in [1, 2] {
        let state = allocated_state(&mut serving);
        serving
            .admit(request(id), state, prompt(&[11]))
            .expect("admit");
    }
    serving.submit_ready_batch().expect("submit");
    serving.runtime_mut().state_manager_mut().fail_commit = true;
    serving.poll_completions().expect("pending");
    assert!(matches!(
        serving.poll_completions(),
        Err(ServingRuntimeError::Runtime(RuntimeError::State(_)))
    ));
    assert_eq!(serving.scheduler().counts().in_flight(), 0);
    assert_eq!(serving.scheduler().counts().terminal(), 2);
    for _ in 0..2 {
        assert!(serving.reclaim_next().expect("reclaim").is_some());
    }
    assert_eq!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(DeviceId::new(0))),
        Some(0)
    );
}

#[test]
fn failed_release_retains_terminal_owner_until_both_release_layers_succeed() {
    let mut serving = fixture(false);
    let state = allocated_state(&mut serving);
    serving
        .admit(request(1), state, prompt(&[11]))
        .expect("admit");
    serving
        .cancel(RequestId::new(1).expect("id"))
        .expect("cancel");
    serving.runtime_mut().backend_mut().fail_release = true;
    assert!(serving.reclaim_next().is_err());
    assert_eq!(serving.scheduler().counts().terminal(), 1);
    serving.runtime_mut().backend_mut().fail_release = false;
    serving.runtime_mut().state_manager_mut().fail_release = true;
    assert!(serving.reclaim_next().is_err());
    assert_eq!(serving.scheduler().counts().terminal(), 1);
    assert!(
        serving
            .scheduler()
            .slot_for_request(RequestId::new(1).expect("id"))
            .is_some()
    );
    serving.runtime_mut().state_manager_mut().fail_release = false;
    assert!(serving.reclaim_next().expect("retry release").is_some());
    assert_eq!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(DeviceId::new(0))),
        Some(0)
    );
}

#[test]
fn admission_rejection_returns_allocations_for_release() {
    let mut serving = fixture(false);
    for id in 1..=4 {
        serving
            .admit(request(id), state(), prompt(&[11]))
            .expect("admit");
    }
    let state = allocated_state(&mut serving);
    let rejected = serving
        .admit(request(5), state, prompt(&[11]))
        .expect_err("backpressure");
    let (error, state) = rejected.into_parts();
    assert!(matches!(
        error,
        ServingRuntimeError::Scheduler(SchedulerError::Backpressure)
    ));
    serving
        .runtime_mut()
        .release_state_set(&state)
        .expect("release rejected state");
    assert_eq!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(DeviceId::new(0))),
        Some(0)
    );
}

#[test]
fn synchronous_completion_failure_returns_state_for_cleanup() {
    let mut serving = fixture(false);
    let state = allocated_state(&mut serving);
    let plan = serving.plan().clone();
    let segment = ExecutionSegment::new_shared(
        RequestId::new(1).expect("id"),
        ExecutionPhase::Prefill,
        1,
        1,
        0,
        plan.shared_state_requirements(),
    )
    .expect("segment")
    .with_token_input(ExecutionTokenInput::prompt(prompt(&[11]), 0, 1).expect("input"))
    .expect("segment input")
    .with_sampling(SamplingParams::greedy(None));
    serving.runtime_mut().state_manager_mut().fail_commit = true;
    let error = serving
        .runtime_mut()
        .execute_segment(&plan, &segment, state)
        .expect_err("commit failure");
    let (error, states) = error.into_parts();
    assert!(matches!(error, RuntimeError::State(_)));
    assert_eq!(states.len(), 1);
    serving
        .runtime_mut()
        .release_state_set(&states[0])
        .expect("release returned state");
    assert_eq!(
        serving
            .runtime()
            .state_manager()
            .used_bytes(crate::state::StateLocation::Device(DeviceId::new(0))),
        Some(0)
    );
}
