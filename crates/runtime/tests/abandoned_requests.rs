//! Abandoned-request cleanup at the engine boundary.
//!
//! A streaming frontend can drop a request before it finishes. Ribn's contract is
//! that cancellation records intent, not completion, and that a request's mailbox
//! outlives its execution slot until its terminal event is delivered. A frontend
//! that cancels without draining therefore leaves a mailbox holding output
//! capacity, and `flush_terminals` cannot reclaim the request without it.
//!
//! These tests pin the two engine properties the text facade depends on, because
//! the facade cannot test them itself without a device:
//!
//! - a cancelled request's terminal event arrives through `pop_event_for`, and
//!   draining it reclaims both mailbox and request slot;
//! - the global `pop_event` drain can hand back a request another caller owns
//!   (here: an abandoned one), which is why a batch must drain per request rather
//!   than route the global drain through its own request map.
//!
//! They are lifecycle tests over the public contract; they are not numerical
//! model qualification and they do not exercise a real device.

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

/// A cancelled request that its owner never drains keeps a mailbox alive; draining
/// the terminal event reclaims the mailbox and the request slot.
#[test]
fn an_abandoned_request_is_reclaimed_only_after_its_terminal_event_is_drained() {
    let (mut engine, _control) = engine();
    let abandoned = engine
        .enqueue(TokenRequest::new(vec![1, 2, 3], options()))
        .expect("enqueue");

    // Let it run far enough to have produced output, then abandon it the way a
    // dropped stream does: cancel, and do not drain.
    engine.step().expect("first step");
    engine.step().expect("second step");
    engine.cancel(abandoned).expect("cancel");

    // Cancellation is intent: the request is still addressable until its terminal
    // event is delivered, which is what keeps its mailbox owned rather than leaked.
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
