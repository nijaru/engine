//! What a backend may report for one submitted row, and what the engine refuses.
//!
//! A prefill row may accept fewer inputs than its budget, because a backend has to
//! be able to stop before work it cannot do (an encoder item that is not ready, or
//! a step budget smaller than an indivisible item). A row that can do nothing
//! reports `Blocked` instead of overspending or faulting its peers. Everything
//! else stays strict: the engine refuses a completion that claims progress it did
//! not receive, that samples output before the prompt is finished, or that makes an
//! empty successful decode step.
//!
//! These are contract tests over mock executors; they exercise no device or model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, EngineError, Event, ExecutionError, ExecutorInfo,
    FinishReason, GenerationExecutor, GenerationLimits, GenerationOptions, RequestId,
    SchedulePolicy, SequenceId, StepCompletion, StepOutcome, SubmissionId, TokenRequest,
};

/// How a fixture row misbehaves, or `None` for the legal behaviour.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Misbehaviour {
    /// A prefill row that consumes more than the engine allowed.
    PrefillBeyondBudget,
    /// A prefill row that samples before the prompt is finished.
    PrefillOutputEarly,
    /// A decode row that reports no progress but claims success.
    DecodeEmpty,
    /// A decode row whose token count disagrees with its advancement.
    DecodeTokenMismatch,
    /// A blocked outcome naming a sequence that was not submitted.
    BlockedWrongSequence,
    /// A blocked outcome while the row could have progressed.
    UnnecessaryBlock,
}

struct Fixture {
    info: ExecutorInfo,
    /// Prompt length per sequence, recorded at admission so the fixture knows
    /// when a prefill row has finished the prompt and must sample.
    prompt_tokens: HashMap<SequenceId, u32>,
    pending: Option<Vec<BatchItem>>,
    next_submission: u64,
    misbehaviour: Option<Misbehaviour>,
    /// Accept one input per prefill row instead of the whole budget.
    short_prefill: bool,
    submitted_rows: Arc<Mutex<usize>>,
}

impl Fixture {
    fn new(misbehaviour: Option<Misbehaviour>) -> (Self, Arc<Mutex<usize>>) {
        let submitted_rows = Arc::new(Mutex::new(0));
        (
            Self {
                info: ExecutorInfo {
                    name: "progress-negotiation fixture".to_owned(),
                    limits: GenerationLimits {
                        context_tokens: 64,
                        max_sequences: 4,
                        max_batch_tokens: 16,
                        max_decode_tokens: 4,
                    },
                },
                prompt_tokens: HashMap::new(),
                pending: None,
                next_submission: 0,
                misbehaviour,
                short_prefill: false,
                submitted_rows: Arc::clone(&submitted_rows),
            },
            submitted_rows,
        )
    }

    fn new_short_prefill() -> (Self, Arc<Mutex<usize>>) {
        let (mut fixture, submitted) = Self::new(None);
        fixture.short_prefill = true;
        (fixture, submitted)
    }

    /// A legal prefill row: consume the planned range and sample only when that
    /// range finished the prompt.
    fn legal_prefill(&self, item: &BatchItem) -> StepOutcome {
        let accepted = if self.short_prefill {
            item.token_budget.min(1)
        } else {
            item.token_budget
        };
        let prefix = item.prefix + accepted;
        let tokens = if prefix
            == self
                .prompt_tokens
                .get(&item.sequence)
                .copied()
                .unwrap_or_default()
            && item.output_budget == 1
        {
            vec![7]
        } else {
            Vec::new()
        };
        StepOutcome::Progress(StepCompletion {
            sequence: item.sequence,
            prefix,
            tokens,
        })
    }

    fn legal_decode(item: &BatchItem) -> StepOutcome {
        StepOutcome::Progress(StepCompletion {
            sequence: item.sequence,
            prefix: item.prefix + 1,
            tokens: vec![7],
        })
    }
}

impl GenerationExecutor for Fixture {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.prompt_tokens
            .entry(sequence)
            .or_insert_with(|| u32::try_from(request.tokens.len()).unwrap_or(u32::MAX));
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        *self.submitted_rows.lock().unwrap() += batch.len();
        self.pending = Some(batch.to_vec());
        self.next_submission += 1;
        Ok(SubmissionId::new(self.next_submission))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepOutcome>>, ExecutionError> {
        let Some(batch) = self.pending.take() else {
            return Ok(None);
        };
        // Sequence ids are engine-issued, so a test cannot invent one: the wrong
        // sequence for a blocked row is another submitted row's sequence.
        let sequences = batch.iter().map(|item| item.sequence).collect::<Vec<_>>();
        Ok(Some(
            batch
                .iter()
                .enumerate()
                .map(|(index, item)| match (item.kind, self.misbehaviour) {
                    // Misbehaviours target the row kind they are about; the other
                    // kind behaves legally so the run reaches the intended step.
                    (ribn::StepKind::Prefill, Some(Misbehaviour::PrefillBeyondBudget)) => {
                        StepOutcome::Progress(StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + item.token_budget + 1,
                            tokens: Vec::new(),
                        })
                    }
                    (ribn::StepKind::Prefill, Some(Misbehaviour::PrefillOutputEarly)) => {
                        StepOutcome::Progress(StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + 1,
                            tokens: vec![7],
                        })
                    }
                    (_, Some(Misbehaviour::BlockedWrongSequence)) => {
                        StepOutcome::Blocked(sequences[(index + 1) % sequences.len()])
                    }
                    (_, Some(Misbehaviour::UnnecessaryBlock)) => {
                        StepOutcome::Blocked(item.sequence)
                    }
                    (ribn::StepKind::Decode, Some(Misbehaviour::DecodeEmpty)) => {
                        StepOutcome::Progress(StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix,
                            tokens: Vec::new(),
                        })
                    }
                    (ribn::StepKind::Decode, Some(Misbehaviour::DecodeTokenMismatch)) => {
                        StepOutcome::Progress(StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + 2,
                            tokens: vec![7],
                        })
                    }
                    (ribn::StepKind::Prefill, _) => self.legal_prefill(item),
                    (ribn::StepKind::Decode, _) => Self::legal_decode(item),
                })
                .collect(),
        ))
    }

    fn release(&mut self, _: SequenceId) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.pending = None;
        Ok(())
    }
}

fn engine(fixture: Fixture) -> Engine {
    Engine::new(
        fixture,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 4096,
            max_buffered_events: 32,
            max_events_per_request: 64,
        },
        SchedulePolicy {
            max_batch_tokens: 16,
            prefill_chunk_tokens: 4,
            ..SchedulePolicy::default()
        },
    )
    .expect("engine")
}

fn request(prompt: Vec<u32>, output: u32) -> TokenRequest {
    TokenRequest::new(
        prompt,
        GenerationOptions {
            max_output_tokens: output,
            ..GenerationOptions::default()
        },
    )
}

/// Drive steps until the engine faults or the request finishes.
fn run(engine: &mut Engine) -> Result<Vec<Event>, EngineError> {
    let mut events = Vec::new();
    for _ in 0..64 {
        engine.step()?;
        while let Some(event) = engine.pop_event() {
            events.push(event);
        }
        if engine.status().requests == 0 {
            break;
        }
    }
    Ok(events)
}

/// A shortened prefill range is a legal step: the engine commits exactly the
/// prefix the backend reports and keeps the sequence runnable.
#[test]
fn a_shorter_prefill_range_is_committed_as_reported() {
    let (fixture, submitted) = Fixture::new_short_prefill();
    let mut engine = engine(fixture);
    engine
        .enqueue(request(vec![1, 2, 3, 4, 5, 6], 2))
        .expect("enqueue");
    let events = run(&mut engine).expect("legal partial prefill completes");

    assert!(matches!(
        events.last(),
        Some(Event::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
    // One accepted prompt token per prefill step, and the sixth step finishes the
    // prompt and samples the first output token. One decode step then reaches the
    // output limit: seven submissions, each advancing exactly what it reported.
    assert_eq!(*submitted.lock().unwrap(), 7);
}

/// A prefill row cannot consume more than the engine allowed.
#[test]
fn a_prefill_row_cannot_exceed_its_budget() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::PrefillBeyondBudget));
    let mut engine = engine(fixture);
    engine
        .enqueue(request(vec![1, 2, 3, 4, 5, 6], 2))
        .expect("enqueue");
    assert!(matches!(run(&mut engine), Err(EngineError::Faulted(_))));
}

/// Sampling belongs to the row that finished the prompt, not to an intermediate
/// prefill step, even when that step consumed part of the range.
#[test]
fn a_partial_prefill_row_cannot_sample() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::PrefillOutputEarly));
    let mut engine = engine(fixture);
    engine
        .enqueue(request(vec![1, 2, 3, 4, 5, 6], 2))
        .expect("enqueue");
    assert!(matches!(run(&mut engine), Err(EngineError::Faulted(_))));
}

/// An empty successful decode step is refused; a backend with nothing to commit
/// reports `Blocked` instead.
#[test]
fn an_empty_successful_decode_step_is_refused() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::DecodeEmpty));
    let mut engine = engine(fixture);
    // Two prompt tokens, so the first step is prefill and the second is decode.
    engine.enqueue(request(vec![1, 2], 2)).expect("enqueue");
    assert!(matches!(run(&mut engine), Err(EngineError::Faulted(_))));
}

/// A decode row's sampled token count must match its advancement.
#[test]
fn a_decode_row_must_sample_exactly_what_it_advanced() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::DecodeTokenMismatch));
    let mut engine = engine(fixture);
    engine.enqueue(request(vec![1, 2], 2)).expect("enqueue");
    assert!(matches!(run(&mut engine), Err(EngineError::Faulted(_))));
}

/// A blocked outcome has to name the row it belongs to.
#[test]
fn a_blocked_outcome_must_name_its_own_sequence() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::BlockedWrongSequence));
    let mut engine = engine(fixture);
    // Two rows, so the misbehaving one can name its peer.
    engine
        .enqueue(request(vec![1, 2, 3, 4], 2))
        .expect("first enqueue");
    engine
        .enqueue(request(vec![5, 6, 7, 8], 2))
        .expect("second enqueue");
    assert!(matches!(run(&mut engine), Err(EngineError::Faulted(_))));
}

/// Blocking is legal and observable: the engine reports it, keeps the sequence
/// runnable, and does not treat it as a failure.
#[test]
fn a_blocked_row_is_reported_and_stays_runnable() {
    let (fixture, submitted) = Fixture::new(Some(Misbehaviour::UnnecessaryBlock));
    let mut engine = engine(fixture);
    engine
        .enqueue(request(vec![1, 2, 3, 4], 2))
        .expect("enqueue");

    let status = engine.step().expect("step");
    assert_eq!(engine.status().requests, 1, "the request is still live");
    assert!(status.submitted, "the first step submitted the row");

    let mut blocked = false;
    for _ in 0..8 {
        let status = engine.step().expect("retry");
        blocked |= status.blocked > 0 || status.completed;
    }
    assert!(blocked, "the engine reported the blocked submission");
    assert_eq!(engine.status().requests, 1, "a blocked row stays runnable");
    assert!(
        *submitted.lock().unwrap() > 1,
        "the row was resubmitted after blocking"
    );
}
