//! Runtime-owned abandonment. Cancellation preserves output; discard relinquishes
//! present and future delivery while retaining device retirement ownership.
//! Mailboxes can outlive execution slots, so discard cannot rely on cancellation
//! succeeding. These host contract tests do not qualify a device or text decoder.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, EngineError, Event, ExecutionError, ExecutorInfo, FinishReason,
    GenerationExecutor, GenerationLimits, GenerationOptions, RequestId, SequenceId, StepCompletion,
    SubmissionId, TokenRequest,
};

#[derive(Default)]
struct Control {
    batches: Vec<Vec<BatchItem>>,
    released: Vec<SequenceId>,
}

/// Minimal single-batch executor: it completes one step behind submission so a
/// request can be observed while it is in flight.
struct Model {
    info: ExecutorInfo,
    control: Arc<Mutex<Control>>,
    pending: Option<Vec<BatchItem>>,
    next_submission: u64,
}

impl Model {
    fn new(control: Arc<Mutex<Control>>) -> Self {
        Self {
            info: ExecutorInfo {
                name: "abandoned-request-fixture".to_owned(),
                limits: GenerationLimits {
                    context_tokens: 128,
                    max_sequences: 4,
                    max_batch_tokens: 32,
                    max_decode_tokens: 4,
                },
            },
            control,
            pending: None,
            next_submission: 0,
        }
    }
}

impl GenerationExecutor for Model {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _request_id: RequestId,
        _sequence: SequenceId,
        _request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        assert!(self.pending.is_none(), "one batch in flight");
        self.control.lock().unwrap().batches.push(batch.to_vec());
        self.pending = Some(batch.to_vec());
        self.next_submission += 1;
        Ok(SubmissionId::new(self.next_submission))
    }

    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        assert_eq!(submission.get(), self.next_submission);
        let Some(batch) = self.pending.take() else {
            return Ok(None);
        };
        Ok(Some(
            batch
                .iter()
                .map(|item| StepCompletion {
                    sequence: item.sequence,
                    prefix: item.prefix + item.token_budget,
                    tokens: (0..item.output_budget)
                        .map(|i| 200 + item.prefix + i)
                        .collect(),
                })
                .collect(),
        ))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.control.lock().unwrap().released.push(sequence);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.pending = None;
        Ok(())
    }
}

fn engine() -> (Engine, Arc<Mutex<Control>>) {
    let control = Arc::new(Mutex::new(Control::default()));
    let engine = Engine::with_defaults(Model::new(Arc::clone(&control))).expect("engine");
    (engine, control)
}

/// Drive steps until `request` has no buffered event or the runtime stops making
/// progress, returning the events drained for that request.
fn drain(engine: &mut Engine, request: RequestId) -> Vec<Event> {
    let mut drained = Vec::new();
    for _ in 0..64 {
        while let Some(event) = engine.pop_event_for(request) {
            drained.push(event);
        }
        if matches!(drained.last(), Some(Event::Finished { .. })) {
            return drained;
        }
        engine.step().expect("step");
    }
    drained
}

fn options() -> GenerationOptions {
    GenerationOptions::default()
}

#[test]
fn discard_reclaims_a_terminal_mailbox_after_the_execution_slot_is_gone() {
    let (mut engine, _) = engine();
    let request = engine
        .enqueue(TokenRequest::new(vec![1], options()))
        .unwrap();
    engine.cancel(request).unwrap();
    engine.step().unwrap();
    assert_eq!(engine.status().requests, 0);
    assert_eq!(engine.status().buffered_events, 1);
    assert!(matches!(
        engine.cancel(request),
        Err(EngineError::UnknownRequest(_))
    ));
    engine.discard(request);
    engine.discard(request);
    assert_eq!(engine.status().buffered_events, 0);
    assert!(engine.pop_event().is_none());
}

#[test]
fn discard_in_flight_preserves_resources_and_healthy_peer_output() {
    let (mut engine, control) = engine();
    let abandoned = engine
        .enqueue(TokenRequest::new(vec![1], options()))
        .unwrap();
    let peer = engine
        .enqueue(TokenRequest::new(vec![2], options()))
        .unwrap();
    engine.step().unwrap();
    engine.discard(abandoned);
    assert!(control.lock().unwrap().released.is_empty());
    assert_eq!(engine.status().active_sequences, 2);
    let events = drain(&mut engine, peer);
    assert!(matches!(
        events.last(),
        Some(Event::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
    assert!(engine.pop_event_for(abandoned).is_none());
    assert_eq!(engine.status().buffered_events, 0);
    assert_eq!(engine.status().requests, 0);
    assert_eq!(control.lock().unwrap().released.len(), 2);
}

#[test]
fn discard_survives_a_partial_in_flight_completion() {
    let (mut engine, control) = engine();
    // A prompt longer than the prefill chunk settles on a positive partial
    // range, so the in-flight completion is valid but does not finish the prompt.
    let request = engine
        .enqueue(TokenRequest::new(vec![1; 20], options()))
        .unwrap();
    engine.step().unwrap();
    engine.discard(request);
    let status = engine.step().unwrap();
    assert!(status.completed);
    assert!(!status.submitted);
    assert_eq!(engine.status().requests, 0);
    assert_eq!(engine.status().active_sequences, 0);
    assert_eq!(engine.status().buffered_events, 0);
    assert_eq!(control.lock().unwrap().released.len(), 1);
}

#[test]
fn repeated_discard_without_draining_does_not_retain_events() {
    let (mut engine, _) = engine();
    for _ in 0..1024 {
        let request = engine
            .enqueue(TokenRequest::new(vec![1], options()))
            .unwrap();
        engine.discard(request);
        engine.step().unwrap();
        assert_eq!(engine.status().requests, 0);
        assert_eq!(engine.status().buffered_events, 0);
    }
}

/// Cancellation preserves output until the owner drains or discards it.
#[test]
fn cancellation_preserves_its_terminal_event_for_the_owner() {
    let (mut engine, _control) = engine();
    let abandoned = engine
        .enqueue(TokenRequest::new(vec![1, 2, 3], options()))
        .expect("enqueue");

    // Let it run far enough to have produced output, then abandon it the way a
    // dropped stream does: cancel, and do not drain.
    engine.step().expect("first step");
    engine.step().expect("second step");
    engine.cancel(abandoned).expect("cancel");

    // Cancellation is intent: this submission has not completed yet, so its
    // execution slot is still addressable independently of mailbox consumption.
    assert!(
        engine.cancel(abandoned).is_ok(),
        "an undrained cancellation stays addressable"
    );

    let events = drain(&mut engine, abandoned);
    assert!(
        matches!(events.last(), Some(Event::Finished { .. })),
        "draining delivers the terminal event, got {events:?}"
    );
    assert!(
        matches!(
            engine.cancel(abandoned),
            Err(EngineError::UnknownRequest(request)) if request == abandoned
        ),
        "a fully drained request is reclaimed and no longer addressable"
    );
}

/// The global drain is round-robin across requests, so it can hand a caller an
/// event for a request that caller does not own. This is why a batch frontend must
/// drain per request instead of indexing a map keyed by its own requests.
#[test]
fn the_global_drain_can_return_another_callers_event() {
    let (mut engine, _control) = engine();
    let abandoned = engine
        .enqueue(TokenRequest::new(vec![1, 2, 3], options()))
        .expect("abandoned");
    engine.cancel(abandoned).expect("cancel");
    let owned = engine
        .enqueue(TokenRequest::new(vec![4, 5], options()))
        .expect("owned");

    let mut saw_abandoned_globally = false;
    let mut owned_events = Vec::new();
    for _ in 0..64 {
        engine.step().expect("step");
        // The global drain is round-robin over every live request, so it hands
        // back the abandoned request's events while `owned` is still running.
        while !saw_abandoned_globally {
            let Some(event) = engine.pop_event() else {
                break;
            };
            let request = event.request();
            assert!(
                request == abandoned || request == owned,
                "no third request exists, got {request:?}"
            );
            if request == abandoned {
                saw_abandoned_globally = true;
            } else {
                owned_events.push(event);
            }
        }
        while let Some(event) = engine.pop_event_for(owned) {
            owned_events.push(event);
        }
        if saw_abandoned_globally && matches!(owned_events.last(), Some(Event::Finished { .. })) {
            break;
        }
    }

    assert!(
        saw_abandoned_globally,
        "the global drain exposes the abandoned request's terminal event"
    );
    assert!(
        owned_events.iter().all(|event| event.request() == owned),
        "a per-request drain never returns another request's event"
    );
    assert!(matches!(
        owned_events.last(),
        Some(Event::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
}

/// A per-request drain leaves a batch's own requests independent of an abandoned
/// one, which is the property the text facade now relies on.
#[test]
fn a_batch_routing_by_request_id_ignores_an_abandoned_mailbox() {
    let (mut engine, _control) = engine();
    let abandoned = engine
        .enqueue(TokenRequest::new(vec![1, 2, 3], options()))
        .expect("abandoned");
    engine.step().expect("first step");
    engine.cancel(abandoned).expect("cancel");

    let first = engine
        .enqueue(TokenRequest::new(vec![7], options()))
        .expect("first");
    let second = engine
        .enqueue(TokenRequest::new(vec![8, 9], options()))
        .expect("second");
    let owned: HashMap<RequestId, &str> = HashMap::from([(first, "first"), (second, "second")]);

    let mut finished = 0_usize;
    for _ in 0..64 {
        engine.step().expect("step");
        for (&request, &label) in &owned {
            while let Some(event) = engine.pop_event_for(request) {
                assert!(
                    owned.contains_key(&event.request()),
                    "{label} drained {event:?}, which the batch did not submit"
                );
                if matches!(event, Event::Finished { .. }) {
                    finished += 1;
                }
            }
        }
        if finished == owned.len() {
            break;
        }
    }
    assert_eq!(finished, owned.len(), "every owned request completed");
}
