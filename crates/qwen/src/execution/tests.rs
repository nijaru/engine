use std::sync::{Arc, Mutex};

use engine_core::{
    BackendCapabilities, BackendError, BackendFeatures, BackendId, BackendKind, ComputeBackend,
    DataType, DeviceId, ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics,
    ExecutionPhase, ExecutionPlan, ExecutionStage, InferenceStateSet, KvStateSpec, ModelId,
    ModelRegionId, PolicyVersion, StateRequirement, WeightBinding,
};
use ribn::{
    Engine, EngineConfig, Event, GenerationOptions, ModelLimits, PreparedModel, SchedulePolicy,
};

use super::*;

#[derive(Default)]
struct Control {
    pending_polls: usize,
    bad_token: bool,
    fail_submit: bool,
    fail_release: bool,
    releases: usize,
    admissions: Vec<SequenceId>,
    batches: Vec<Vec<(ExecutionPhase, u32, u32)>>,
}

struct Backend {
    caps: BackendCapabilities,
    pending: Option<ExecutionBatchEvent>,
    control: Arc<Mutex<Control>>,
    id: u64,
}

impl ComputeBackend for Backend {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.caps
    }
    fn submit(
        &mut self,
        plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<BackendSubmissionId, BackendError> {
        self.validate_execution(plan, batch, states)?;
        let mut control = self.control.lock().unwrap();
        control.batches.push(
            batch
                .segments()
                .iter()
                .map(|row| (row.phase(), row.state_position(), row.token_count()))
                .collect(),
        );
        if control.fail_submit {
            return Err(BackendError::ExecutionFailed(
                "injected submission failure".into(),
            ));
        }
        assert!(self.pending.is_none());
        self.id += 1;
        let events = batch
            .segments()
            .iter()
            .map(|row| {
                let result = ExecutionEvent::new(
                    row.request(),
                    plan.policy_version(),
                    row.phase(),
                    row.token_count(),
                    ExecutionMetrics::new(1, 0, 0),
                )
                .unwrap();
                if row.requests_sampling() {
                    result.with_output_token(if control.bad_token { 100 } else { 7 })
                } else {
                    result
                }
            })
            .collect();
        self.pending = Some(ExecutionBatchEvent::new(events).unwrap());
        Ok(BackendSubmissionId::new(self.id).unwrap())
    }
    fn poll(
        &mut self,
        _: BackendSubmissionId,
    ) -> Result<Option<ExecutionBatchEvent>, BackendError> {
        let mut control = self.control.lock().unwrap();
        if control.pending_polls > 0 {
            control.pending_polls -= 1;
            return Ok(None);
        }
        Ok(self.pending.take())
    }
    fn release_inference_state(&mut self, _: &InferenceStateSet) -> Result<(), BackendError> {
        assert!(self.pending.is_none(), "release precedes barrier");
        let mut control = self.control.lock().unwrap();
        if control.fail_release {
            return Err(BackendError::ExecutionFailed(
                "injected release failure".into(),
            ));
        }
        control.releases += 1;
        Ok(())
    }
}

struct Model(QwenExecution<Backend>);
impl PreparedModel for Model {
    fn info(&self) -> &ModelInfo {
        self.0.info()
    }
    fn admit(&mut self, id: SequenceId, request: &TokenRequest) -> Result<Admission, ModelError> {
        self.0.backend.control.lock().unwrap().admissions.push(id);
        self.0.admit(id, request)
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ModelError> {
        self.0.submit(batch)
    }
    fn poll(&mut self, id: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ModelError> {
        self.0.poll(id)
    }
    fn release(&mut self, id: SequenceId) -> Result<(), ModelError> {
        self.0.release(id)
    }
    fn synchronize(&mut self) -> Result<(), ModelError> {
        self.0.backend.control.lock().unwrap().pending_polls = 0;
        self.0.drain_after_barrier()
    }
}

fn model(control: Arc<Mutex<Control>>) -> Model {
    let device = DeviceId::new(0);
    let model = ModelId::new("Qwen adapter fixture").unwrap();
    let backend = BackendId::new("host-fixture").unwrap();
    let requirements = vec![StateRequirement::FullAttentionKv(
        KvStateSpec::new(1, 1, 2, 64, DataType::F16).unwrap(),
    )];
    let capacity = state::capacity(&requirements, 2).unwrap();
    let plan = ExecutionPlan::new(
        model.clone(),
        backend.clone(),
        device,
        PolicyVersion::new(1).unwrap(),
        vec![
            ExecutionStage::new(ModelRegionId::new(0), ExecutionPhase::Prefill),
            ExecutionStage::new(ModelRegionId::new(0), ExecutionPhase::Decode),
        ],
        requirements,
        WeightBinding::empty(model, device),
    )
    .unwrap();
    let info = ModelInfo {
        name: "Qwen adapter fixture".into(),
        limits: ModelLimits {
            context_tokens: 64,
            max_sequences: 2,
            max_batch_tokens: 8,
            max_decode_tokens: 1,
        },
    };
    Model(QwenExecution::new(
        Backend {
            caps: BackendCapabilities::new(
                backend,
                device,
                BackendKind::Cpu,
                capacity,
                BackendFeatures::new(vec![DataType::F16], vec![], false, false),
            ),
            pending: None,
            control,
            id: 0,
        },
        info,
        plan,
        state::manager(device, capacity),
        100,
    ))
}

fn engine(control: Arc<Mutex<Control>>) -> Engine {
    Engine::new(
        model(control),
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 128,
            max_buffered_events: 32,
        },
        SchedulePolicy {
            max_batch_tokens: 8,
            prefill_chunk_tokens: 2,
            ..SchedulePolicy::default()
        },
    )
    .unwrap()
}

fn request(tokens: Vec<u32>) -> TokenRequest {
    TokenRequest::new(
        tokens,
        GenerationOptions {
            max_output_tokens: 3,
            ..GenerationOptions::default()
        },
    )
}

#[test]
fn qwen_bridge_preserves_chunk_boundaries_completion_and_cancellation() {
    let control = Arc::new(Mutex::new(Control {
        pending_polls: 1,
        ..Control::default()
    }));
    let mut engine = engine(control.clone());
    let first = engine.enqueue(request(vec![1, 2, 3])).unwrap();
    let peer = engine.enqueue(request(vec![4, 5, 6])).unwrap();
    engine.step().unwrap();
    assert_eq!(engine.committed_prefix(peer), Some(0));
    engine.cancel(first).unwrap();
    engine.step().unwrap();
    assert!(engine.pop_event().is_none());
    let mut events = Vec::new();
    for _ in 0..16 {
        engine.step().unwrap();
        while let Some(event) = engine.pop_event() {
            events.push(event);
        }
        if engine.status().requests == 0 {
            break;
        }
    }
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { request, .. } if *request == first))
            .count(),
        0
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { request, token: 7 } if *request == peer))
            .count(),
        3
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Finished { .. }))
            .count(),
        2
    );
    let control = control.lock().unwrap();
    assert_eq!(control.releases, 2);
    assert!(
        control
            .batches
            .iter()
            .flatten()
            .any(|row| *row == (ExecutionPhase::Prefill, 2, 1))
    );
    assert!(
        control
            .batches
            .iter()
            .flatten()
            .any(|row| *row == (ExecutionPhase::Decode, 3, 1))
    );
}

#[test]
fn adapter_rejects_unsupported_semantics_before_allocating() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut engine = engine(control.clone());
    let mut input = request(vec![1]);
    input.options.sampling.temperature = 1.0;
    engine.enqueue(input).unwrap();
    engine.enqueue(request(vec![100])).unwrap();
    engine.step().unwrap();
    assert_eq!(engine.status().requests, 0);
    assert_eq!(control.lock().unwrap().releases, 0);
    for _ in 0..2 {
        assert!(matches!(
            engine.pop_event(),
            Some(Event::Finished {
                reason: ribn::FinishReason::Failed(_),
                ..
            })
        ));
    }
}

#[test]
fn adapter_returns_leases_on_submission_and_completion_failure() {
    for submit in [false, true] {
        let control = Arc::new(Mutex::new(Control {
            fail_submit: submit,
            bad_token: !submit,
            ..Control::default()
        }));
        let mut engine = engine(control.clone());
        let id = engine.enqueue(request(vec![1])).unwrap();
        let first = engine.step();
        assert!(if submit {
            first.is_err()
        } else {
            first.is_ok() && engine.step().is_err()
        });
        assert_eq!(engine.committed_prefix(id), Some(0));
        assert_eq!(control.lock().unwrap().releases, 0);
        engine.shutdown().unwrap();
        assert_eq!(control.lock().unwrap().releases, 1);
    }
}

#[test]
fn adapter_release_failure_is_retryable_after_output_completion() {
    let control = Arc::new(Mutex::new(Control {
        fail_release: true,
        ..Control::default()
    }));
    let mut engine = engine(control.clone());
    let mut input = request(vec![1]);
    input.options.max_output_tokens = 1;
    engine.enqueue(input).unwrap();
    engine.step().unwrap();
    assert!(engine.step().is_err());
    assert_eq!(engine.status().active_sequences, 1);
    control.lock().unwrap().fail_release = false;
    engine.step().unwrap();
    assert_eq!(engine.status().requests, 0);
    assert_eq!(control.lock().unwrap().releases, 1);
}

#[test]
fn adapter_shutdown_drains_a_pending_batch_before_release() {
    let control = Arc::new(Mutex::new(Control {
        pending_polls: 10,
        ..Control::default()
    }));
    let mut engine = engine(control.clone());
    engine.enqueue(request(vec![1])).unwrap();
    engine.step().unwrap();
    engine.shutdown().unwrap();
    assert_eq!(engine.status().requests, 0);
    assert_eq!(control.lock().unwrap().releases, 1);
}
