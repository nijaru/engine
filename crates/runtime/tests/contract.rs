//! Out-of-tree model implementations exercise only Ribn's public contract.
//! These are lifecycle/extension tests, not numerical model qualification.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, EngineError, Event, ExecutionError, ExecutorInfo,
    FinishReason, GenerationExecutor, GenerationLimits, GenerationOptions, RequestId,
    SchedulePolicy, SequenceId, StepCompletion, SubmissionId, TokenRequest, Usage,
};

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent fault-injection switches"
)]
#[derive(Default)]
struct Control {
    admitted: Vec<SequenceId>,
    released: Vec<SequenceId>,
    batches: Vec<Vec<BatchItem>>,
    defer: bool,
    reject_admission: bool,
    fail_submit: bool,
    fail_poll: bool,
    fail_release: usize,
    fail_sync: bool,
    malformed: bool,
    synchronizations: usize,
    drops: usize,
}

type Shared = Arc<Mutex<Control>>;

#[derive(Default)]
struct DenseState([u8; 16]);

struct HybridState {
    matrix: [[f32; 3]; 3],
    host_index: Vec<u32>,
}
impl Default for HybridState {
    fn default() -> Self {
        Self {
            matrix: [[0.0; 3]; 3],
            host_index: vec![1, 2],
        }
    }
}

trait PrivateState: Default + Send {
    fn update(&mut self);
}
impl PrivateState for DenseState {
    fn update(&mut self) {
        self.0[0] = 1;
    }
}
impl PrivateState for HybridState {
    fn update(&mut self) {
        if self.host_index.first() == Some(&1) {
            self.matrix[0][0] += 1.0;
        }
    }
}

struct Model<S> {
    info: ExecutorInfo,
    control: Shared,
    states: HashMap<SequenceId, S>,
    pending: Option<Vec<BatchItem>>,
    delay: bool,
    next_submission: u64,
}

impl<S> Model<S> {
    fn new(control: Shared) -> Self {
        Self {
            info: ExecutorInfo {
                name: std::any::type_name::<S>().to_owned(),
                limits: GenerationLimits {
                    context_tokens: 1024,
                    max_sequences: 8,
                    max_batch_tokens: 128,
                    max_decode_tokens: 4,
                },
            },
            control,
            states: HashMap::new(),
            pending: None,
            delay: false,
            next_submission: 0,
        }
    }
}

impl<S: PrivateState> GenerationExecutor for Model<S> {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _request_id: RequestId,
        sequence: SequenceId,
        _request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let mut control = self.control.lock().unwrap();
        if control.reject_admission {
            return Err(ExecutionError::new("unsupported request"));
        }
        if control.defer {
            return Ok(Admission::Deferred);
        }
        assert!(self.states.insert(sequence, S::default()).is_none());
        control.admitted.push(sequence);
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let mut control = self.control.lock().unwrap();
        if control.fail_submit {
            return Err(ExecutionError::new("partial submission fault"));
        }
        assert!(self.pending.is_none());
        for item in batch {
            assert!(self.states.contains_key(&item.sequence));
        }
        control.batches.push(batch.to_vec());
        self.pending = Some(batch.to_vec());
        self.delay = true;
        self.next_submission += 1;
        Ok(SubmissionId::new(self.next_submission))
    }

    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        assert_eq!(submission.get(), self.next_submission);
        let control = self.control.lock().unwrap();
        if control.fail_poll {
            return Err(ExecutionError::new("device completion uncertain"));
        }
        if self.delay {
            self.delay = false;
            return Ok(None);
        }
        let batch = self.pending.take().unwrap();
        let mut rows = batch
            .iter()
            .map(|item| {
                self.states.get_mut(&item.sequence).unwrap().update();
                StepCompletion {
                    sequence: item.sequence,
                    prefix: item.prefix + item.token_budget,
                    tokens: (0..item.output_budget)
                        .map(|i| 100 + item.prefix + i)
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        if control.malformed {
            rows.last_mut().unwrap().prefix += 1;
        }
        Ok(Some(rows))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        let mut control = self.control.lock().unwrap();
        if control.fail_release > 0 {
            control.fail_release -= 1;
            return Err(ExecutionError::new("release must be retried"));
        }
        assert!(
            self.pending
                .as_ref()
                .is_none_or(|batch| batch.iter().all(|item| item.sequence != sequence))
        );
        if self.states.remove(&sequence).is_some() {
            control.released.push(sequence);
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        let mut control = self.control.lock().unwrap();
        control.synchronizations += 1;
        if control.fail_sync {
            return Err(ExecutionError::new("cannot prove device completion"));
        }
        self.pending = None;
        Ok(())
    }
}

impl<S> Drop for Model<S> {
    fn drop(&mut self) {
        assert!(
            self.pending.is_none(),
            "device-visible resources dropped before synchronization"
        );
        self.control.lock().unwrap().drops += 1;
    }
}

fn config() -> EngineConfig {
    EngineConfig {
        max_active_requests: 2,
        max_queued_requests: 4,
        max_queued_input_tokens: 4096,
        max_buffered_events: 32,
        max_events_per_request: 64,
    }
}

fn request(prompt: usize, output: u32) -> TokenRequest {
    TokenRequest::new(
        vec![1; prompt],
        GenerationOptions {
            max_output_tokens: output,
            ..GenerationOptions::default()
        },
    )
}

fn engine<S: PrivateState + 'static>(control: &Shared) -> Engine {
    Engine::new(
        Model::<S>::new(control.clone()),
        config(),
        SchedulePolicy::default(),
    )
    .unwrap()
}

fn drain(engine: &mut Engine) -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..2000 {
        engine.step().unwrap();
        while let Some(event) = engine.pop_event() {
            events.push(event);
        }
        if engine.status().requests == 0 {
            return events;
        }
    }
    panic!("runtime did not complete");
}

#[test]
fn distinct_private_state_layouts_need_no_runtime_changes() {
    let dense = Arc::new(Mutex::new(Control::default()));
    let hybrid = Arc::new(Mutex::new(Control::default()));
    let mut first = engine::<DenseState>(&dense);
    let mut second = engine::<HybridState>(&hybrid);
    first.enqueue(request(5, 3)).unwrap();
    second.enqueue(request(5, 3)).unwrap();
    for runtime in [&mut first, &mut second] {
        let events = drain(runtime);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Token { .. }))
                .count(),
            3
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finished {
                reason: FinishReason::Length,
                ..
            })
        ));
    }
    assert_ne!(
        dense.lock().unwrap().admitted[0],
        hybrid.lock().unwrap().admitted[0]
    );
    assert_eq!(dense.lock().unwrap().released.len(), 1);
    assert_eq!(hybrid.lock().unwrap().released.len(), 1);
}

#[test]
fn prefixes_and_output_wait_for_completion_and_cancelled_peers_do_not_escape() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut runtime = engine::<DenseState>(&control);
    let cancelled = runtime.enqueue(request(1, 2)).unwrap();
    let peer = runtime.enqueue(request(1, 2)).unwrap();
    assert!(runtime.step().unwrap().submitted);
    assert_eq!(runtime.committed_prefix(peer), Some(0));
    runtime.cancel(cancelled).unwrap();
    assert!(!runtime.step().unwrap().completed);
    assert!(runtime.pop_event().is_none());
    assert!(control.lock().unwrap().released.is_empty());
    let events = drain(&mut runtime);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::Token { request, .. } if *request == cancelled))
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { request, .. } if *request == peer))
            .count(),
        2
    );
    assert!(events.contains(&Event::Finished {
        request: cancelled,
        reason: FinishReason::Cancelled,
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 0,
        },
    }));
    assert_eq!(control.lock().unwrap().released.len(), 2);
}

#[test]
fn multi_token_completion_and_live_policy_do_not_change_the_request_api() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut runtime = engine::<HybridState>(&control);
    runtime.enqueue(request(1, 6)).unwrap();
    runtime.step().unwrap();
    runtime
        .set_policy(SchedulePolicy {
            decode_tokens: 4,
            ..SchedulePolicy::default()
        })
        .unwrap();
    let events = drain(&mut runtime);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { .. }))
            .count(),
        6
    );
    let control = control.lock().unwrap();
    assert_eq!(control.batches[0][0].output_budget, 1);
    assert_eq!(control.batches[1][0].output_budget, 4);
    assert_eq!(control.batches[2][0].output_budget, 1);
}

#[test]
fn malformed_batch_commits_no_peer_prefix_or_output() {
    let control = Arc::new(Mutex::new(Control {
        malformed: true,
        ..Control::default()
    }));
    let mut runtime = engine::<DenseState>(&control);
    let first = runtime.enqueue(request(1, 2)).unwrap();
    let second = runtime.enqueue(request(1, 2)).unwrap();
    runtime.step().unwrap();
    runtime.step().unwrap();
    assert!(matches!(runtime.step(), Err(EngineError::Faulted(_))));
    assert_eq!(runtime.committed_prefix(first), Some(0));
    assert_eq!(runtime.committed_prefix(second), Some(0));
    while let Some(event) = runtime.pop_event() {
        assert!(matches!(
            event,
            Event::Finished {
                reason: FinishReason::Failed(_),
                ..
            }
        ));
    }
    assert!(control.lock().unwrap().released.is_empty());
    assert!(matches!(
        runtime.enqueue(request(1, 1)),
        Err(EngineError::Faulted(_))
    ));
    runtime.shutdown().unwrap();
    assert_eq!(control.lock().unwrap().released.len(), 2);
}

#[test]
fn waiting_requests_do_not_allocate_model_state_and_input_memory_is_bounded() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut cfg = config();
    cfg.max_active_requests = 1;
    cfg.max_queued_requests = 1;
    cfg.max_queued_input_tokens = 3;
    let mut runtime = Engine::new(
        Model::<DenseState>::new(control.clone()),
        cfg,
        SchedulePolicy::default(),
    )
    .unwrap();
    runtime.enqueue(request(2, 2)).unwrap();
    assert_eq!(runtime.enqueue(request(2, 2)), Err(EngineError::QueueFull));
    assert!(control.lock().unwrap().admitted.is_empty());
    runtime.step().unwrap();
    runtime.enqueue(request(2, 2)).unwrap();
    assert_eq!(runtime.enqueue(request(1, 2)), Err(EngineError::QueueFull));
    assert_eq!(control.lock().unwrap().admitted.len(), 1);
    drain(&mut runtime);
    assert_eq!(control.lock().unwrap().admitted.len(), 2);
}

#[test]
fn output_backpressure_preserves_reserved_completion_credits() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut cfg = config();
    cfg.max_active_requests = 1;
    cfg.max_buffered_events = 2;
    let mut runtime = Engine::new(
        Model::<DenseState>::new(control.clone()),
        cfg,
        SchedulePolicy::default(),
    )
    .unwrap();
    let running = runtime.enqueue(request(1, 3)).unwrap();
    runtime.step().unwrap();
    let waiting = runtime.enqueue(request(1, 3)).unwrap();
    runtime.cancel(waiting).unwrap();
    runtime.step().unwrap();
    assert!(
        runtime.pop_event().is_none(),
        "a cancellation consumed reserved output capacity"
    );
    let step = runtime.step().unwrap();
    assert!(step.output_blocked);
    assert_eq!(runtime.status().buffered_events, 2);
    assert_eq!(control.lock().unwrap().batches.len(), 1);
    let mut events = Vec::new();
    while let Some(event) = runtime.pop_event() {
        events.push(event);
    }
    events.extend(drain(&mut runtime));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { request, .. } if *request == running))
            .count(),
        3
    );
    assert!(events.contains(&Event::Finished {
        request: waiting,
        reason: FinishReason::Cancelled,
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 0,
        },
    }));
}

#[test]
fn failed_release_retains_its_owner_and_terminal_event_is_not_duplicated() {
    let control = Arc::new(Mutex::new(Control {
        fail_release: 1,
        ..Control::default()
    }));
    let mut runtime = engine::<DenseState>(&control);
    runtime.enqueue(request(1, 1)).unwrap();
    runtime.step().unwrap();
    runtime.step().unwrap();
    assert!(matches!(runtime.step(), Err(EngineError::Execution(_))));
    assert_eq!(runtime.status().active_sequences, 1);
    assert_eq!(runtime.status().requests, 1);
    let events = drain(&mut runtime);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Finished { .. }))
            .count(),
        1
    );
    assert_eq!(control.lock().unwrap().released.len(), 1);
}

#[test]
fn partial_submission_and_poll_failures_retain_state_until_shutdown() {
    for fail_submit in [true, false] {
        let control = Arc::new(Mutex::new(Control {
            fail_submit,
            fail_poll: !fail_submit,
            ..Control::default()
        }));
        let mut runtime = engine::<DenseState>(&control);
        runtime.enqueue(request(1, 3)).unwrap();
        if !fail_submit {
            runtime.step().unwrap();
        }
        assert!(matches!(runtime.step(), Err(EngineError::Faulted(_))));
        assert!(control.lock().unwrap().released.is_empty());
        runtime.shutdown().unwrap();
        assert_eq!(control.lock().unwrap().released.len(), 1);
    }
}

#[test]
fn deferred_and_rejected_admission_do_not_own_sequence_allocations() {
    let control = Arc::new(Mutex::new(Control {
        defer: true,
        ..Control::default()
    }));
    let mut runtime = engine::<DenseState>(&control);
    runtime.enqueue(request(1, 1)).unwrap();
    assert!(!runtime.step().unwrap().submitted);
    assert_eq!(runtime.status().active_sequences, 0);
    assert_eq!(runtime.status().waiting, 1);
    control.lock().unwrap().defer = false;
    drain(&mut runtime);
    control.lock().unwrap().reject_admission = true;
    runtime.enqueue(request(1, 1)).unwrap();
    let events = drain(&mut runtime);
    assert!(matches!(
        events.as_slice(),
        [Event::Finished {
            reason: FinishReason::Failed(_),
            ..
        }]
    ));
    assert_eq!(control.lock().unwrap().admitted.len(), 1);
}

#[test]
fn shutdown_failure_keeps_the_engine_available_for_retry() {
    let control = Arc::new(Mutex::new(Control {
        fail_sync: true,
        ..Control::default()
    }));
    let mut runtime = engine::<DenseState>(&control);
    runtime.enqueue(request(1, 1)).unwrap();
    runtime.step().unwrap();
    assert!(runtime.shutdown().is_err());
    assert!(control.lock().unwrap().released.is_empty());
    control.lock().unwrap().fail_sync = false;
    runtime.shutdown().unwrap();
    assert_eq!(control.lock().unwrap().released.len(), 1);
    assert_eq!(runtime.enqueue(request(1, 1)), Err(EngineError::Closed));
}

#[test]
fn dropping_an_in_flight_engine_establishes_completion_before_teardown() {
    let control = Arc::new(Mutex::new(Control::default()));
    {
        let mut runtime = engine::<DenseState>(&control);
        runtime.enqueue(request(1, 1)).unwrap();
        runtime.step().unwrap();
    }
    let control = control.lock().unwrap();
    assert_eq!(control.drops, 1);
    assert_eq!(control.released.len(), 1);
    assert_eq!(control.synchronizations, 1);
}

#[test]
fn uncertain_drop_retains_the_model_instead_of_freeing_live_memory() {
    let control = Arc::new(Mutex::new(Control {
        fail_sync: true,
        ..Control::default()
    }));
    {
        let mut runtime = engine::<DenseState>(&control);
        runtime.enqueue(request(1, 1)).unwrap();
        runtime.step().unwrap();
    }
    let control = control.lock().unwrap();
    assert_eq!(control.drops, 0, "faulted model must remain owned");
    assert!(control.released.is_empty());
}

#[test]
fn decode_cannot_starve_admitted_prefill_under_a_one_token_budget() {
    let control = Arc::new(Mutex::new(Control::default()));
    let policy = SchedulePolicy {
        max_batch_tokens: 1,
        prefill_chunk_tokens: 1,
        max_decode_only_steps: 3,
        ..SchedulePolicy::default()
    };
    let mut runtime =
        Engine::new(Model::<DenseState>::new(control.clone()), config(), policy).unwrap();
    runtime.enqueue(request(1, 20)).unwrap();
    runtime.enqueue(request(8, 1)).unwrap();
    for _ in 0..20 {
        runtime.step().unwrap();
        while runtime.pop_event().is_some() {}
    }
    let control = control.lock().unwrap();
    let second = control.admitted[1];
    let first_progress = control
        .batches
        .iter()
        .position(|batch| batch.iter().any(|item| item.sequence == second))
        .unwrap();
    assert!(
        first_progress <= 4,
        "prefill waited {first_progress} submissions"
    );
}

#[test]
fn stop_and_output_budgets_remain_exact_with_multi_token_completion() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut runtime = engine::<HybridState>(&control);
    runtime
        .set_policy(SchedulePolicy {
            decode_tokens: 4,
            ..SchedulePolicy::default()
        })
        .unwrap();
    let mut input = request(1, 8);
    input.options.stop_tokens.push(102);
    runtime.enqueue(input).unwrap();
    let events = drain(&mut runtime);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { .. }))
            .count(),
        2
    );
    assert!(matches!(
        events.last(),
        Some(Event::Finished {
            reason: FinishReason::Stop,
            ..
        })
    ));
}

#[test]
fn invalid_requests_and_policy_are_rejected_before_model_admission() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut runtime = engine::<DenseState>(&control);
    assert_eq!(
        runtime.enqueue(request(0, 1)),
        Err(EngineError::InvalidRequest)
    );
    assert_eq!(
        runtime.enqueue(request(1024, 1)),
        Err(EngineError::InvalidRequest)
    );
    assert_eq!(
        runtime.enqueue(request(1, 0)),
        Err(EngineError::InvalidRequest)
    );
    let mut bad = request(1, 1);
    bad.options.sampling.temperature = f32::NAN;
    assert!(runtime.enqueue(bad).is_err());
    assert_eq!(
        runtime.set_policy(SchedulePolicy {
            decode_tokens: 5,
            ..SchedulePolicy::default()
        }),
        Err(EngineError::InvalidConfig)
    );
    assert!(control.lock().unwrap().admitted.is_empty());
}

#[test]
fn stalled_consumer_does_not_block_a_peer_with_output_capacity() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut cfg = config();
    cfg.max_buffered_events = 8;
    cfg.max_events_per_request = 2;
    let mut runtime = Engine::new(
        Model::<DenseState>::new(control.clone()),
        cfg,
        SchedulePolicy::default(),
    )
    .unwrap();
    let slow = runtime.enqueue(request(1, 8)).unwrap();
    let peer = runtime.enqueue(request(1, 5)).unwrap();
    let mut peer_tokens = 0;
    let mut peer_finished = false;
    for _ in 0..100 {
        runtime.step().unwrap();
        while let Some(event) = runtime.pop_event_for(peer) {
            match event {
                Event::Token { .. } => peer_tokens += 1,
                Event::Finished { reason, .. } => {
                    assert_eq!(reason, FinishReason::Length);
                    peer_finished = true;
                }
            }
        }
        assert!(runtime.status().buffered_events <= cfg.max_buffered_events);
        if peer_finished {
            break;
        }
    }
    assert!(peer_finished, "ready peer was blocked by another consumer");
    assert_eq!(peer_tokens, 5);
    assert_eq!(runtime.committed_prefix(slow), Some(1));
    assert_eq!(runtime.status().active_sequences, 1);
    runtime.cancel(slow).unwrap();
    runtime.step().unwrap();
    assert!(matches!(
        runtime.pop_event_for(slow),
        Some(Event::Token { .. })
    ));
    assert_eq!(
        runtime.pop_event_for(slow),
        Some(Event::Finished {
            request: slow,
            reason: FinishReason::Cancelled,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
            },
        })
    );
    assert!(runtime.pop_event().is_none());
    assert_eq!(runtime.status().requests, 0);
    assert_eq!(control.lock().unwrap().released.len(), 2);
}

#[test]
fn model_compatible_defaults_work_for_small_executors() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut model = Model::<DenseState>::new(control.clone());
    model.info.limits.max_sequences = 1;
    model.info.limits.max_batch_tokens = 1;
    let mut runtime = Engine::with_defaults(model).unwrap();
    runtime.enqueue(request(3, 2)).unwrap();
    runtime.enqueue(request(2, 2)).unwrap();
    assert_eq!(drain(&mut runtime).len(), 6);
    assert!(
        control
            .lock()
            .unwrap()
            .batches
            .iter()
            .all(|batch| batch.len() == 1 && batch[0].token_budget == 1)
    );
}

#[test]
fn request_mailboxes_survive_slot_reuse_without_cross_delivery() {
    let control = Arc::new(Mutex::new(Control::default()));
    let mut runtime = engine::<DenseState>(&control);
    let mut retained = Vec::new();
    for _ in 0..8 {
        let id = runtime.enqueue(request(1, 1)).unwrap();
        for _ in 0..4 {
            runtime.step().unwrap();
        }
        assert_eq!(runtime.status().requests, 0);
        retained.push(id);
    }
    for id in retained.into_iter().rev() {
        assert!(
            matches!(runtime.pop_event_for(id), Some(Event::Token { request, .. }) if request == id)
        );
        assert_eq!(
            runtime.pop_event_for(id),
            Some(Event::Finished {
                request: id,
                reason: FinishReason::Length,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            })
        );
        assert!(runtime.pop_event_for(id).is_none());
    }
    assert!(runtime.pop_event().is_none());
}
