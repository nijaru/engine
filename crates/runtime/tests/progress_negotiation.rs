//! What a backend may report for one submitted row, and what the engine refuses.
//!
//! A prefill row may accept fewer inputs than its budget, because a backend has to
//! be able to stop before work it cannot do. Every completion must still advance
//! by at least one input: a row that can do nothing is not a completion, and a
//! permanent inability to proceed is admission-time request-local rejection, not a
//! completion-time outcome. Everything else stays strict: the engine refuses a
//! completion that claims progress it did not receive, that samples output before
//! the prompt is finished, or that makes an empty successful decode step.
//!
//! These are contract tests over mock executors; they exercise no device or model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, EngineError, Event, ExecutionError, ExecutorInfo,
    FinishReason, GenerationExecutor, GenerationLimits, GenerationOptions, RequestId,
    SchedulePolicy, SequenceId, StepCompletion, SubmissionId, TokenRequest,
};

/// How a fixture row misbehaves, or `None` for the legal behaviour.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Misbehaviour {
    /// A prefill row that consumes more than the engine allowed.
    PrefillBeyondBudget,
    /// A prefill row that samples before the prompt is finished.
    PrefillOutputEarly,
    /// A prefill row that reports no advancement at all.
    PrefillZeroProgress,
    /// A decode row that reports no progress but claims success.
    DecodeEmpty,
    /// A decode row whose token count disagrees with its advancement.
    DecodeTokenMismatch,
    /// Only the last prefill row in the batch consumes beyond its budget, so the
    /// engine must reject the whole batch without committing its healthy peers.
    LastPrefillBeyondBudget,
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
    fn legal_prefill(&self, item: &BatchItem) -> StepCompletion {
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
        StepCompletion {
            sequence: item.sequence,
            prefix,
            tokens,
        }
    }

    fn legal_decode(item: &BatchItem) -> StepCompletion {
        StepCompletion {
            sequence: item.sequence,
            prefix: item.prefix + 1,
            tokens: vec![7],
        }
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

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let Some(batch) = self.pending.take() else {
            return Ok(None);
        };
        let last = batch.len().saturating_sub(1);
        Ok(Some(
            batch
                .iter()
                .enumerate()
                .map(|(index, item)| match (item.kind, self.misbehaviour) {
                    // Misbehaviours target the row kind they are about; the other
                    // kind behaves legally so the run reaches the intended step.
                    (ribn::StepKind::Prefill, Some(Misbehaviour::PrefillBeyondBudget)) => {
                        StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + item.token_budget + 1,
                            tokens: Vec::new(),
                        }
                    }
                    (ribn::StepKind::Prefill, Some(Misbehaviour::PrefillOutputEarly)) => {
                        StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + 1,
                            tokens: vec![7],
                        }
                    }
                    (ribn::StepKind::Prefill, Some(Misbehaviour::PrefillZeroProgress)) => {
                        StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix,
                            tokens: Vec::new(),
                        }
                    }
                    (ribn::StepKind::Prefill, Some(Misbehaviour::LastPrefillBeyondBudget))
                        if index == last =>
                    {
                        StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + item.token_budget + 1,
                            tokens: Vec::new(),
                        }
                    }
                    (ribn::StepKind::Decode, Some(Misbehaviour::DecodeEmpty)) => StepCompletion {
                        sequence: item.sequence,
                        prefix: item.prefix,
                        tokens: Vec::new(),
                    },
                    (ribn::StepKind::Decode, Some(Misbehaviour::DecodeTokenMismatch)) => {
                        StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + 2,
                            tokens: vec![7],
                        }
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
    engine_with_output_limit(fixture, 32)
}

fn engine_with_output_limit(fixture: Fixture, max_buffered_events: usize) -> Engine {
    Engine::new(
        fixture,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 4096,
            max_buffered_events,
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
    // Final-prefill offers reserve two credits even when the accepted shorter
    // range emits nothing. Every unused credit must return for this pool to live.
    let mut engine = engine_with_output_limit(fixture, 2);
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

/// Positive progress is required: a prefill row that advances nothing is not a
/// completion, so the engine faults instead of resubmitting it forever. This is
/// the zero-prefill regression the roadmap requires.
#[test]
fn a_zero_progress_prefill_row_is_refused() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::PrefillZeroProgress));
    let mut engine = engine(fixture);
    engine
        .enqueue(request(vec![1, 2, 3, 4], 2))
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
/// must not report a completion.
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

/// A malformed row rejects the whole submission: the healthy peer's prefix is not
/// committed either, because the batch is validated before any row commits.
#[test]
fn a_malformed_row_does_not_commit_its_healthy_peer() {
    let (fixture, _) = Fixture::new(Some(Misbehaviour::LastPrefillBeyondBudget));
    let mut engine = engine(fixture);
    let first = engine
        .enqueue(request(vec![1, 2, 3, 4], 2))
        .expect("first enqueue");
    let second = engine
        .enqueue(request(vec![5, 6, 7, 8], 2))
        .expect("second enqueue");
    engine.step().expect("submit both rows");
    assert!(matches!(engine.step(), Err(EngineError::Faulted(_))));
    assert_eq!(engine.committed_prefix(first), Some(0));
    assert_eq!(engine.committed_prefix(second), Some(0));
    while let Some(event) = engine.pop_event() {
        assert!(matches!(
            event,
            Event::Finished {
                reason: FinishReason::Failed(_),
                ..
            }
        ));
    }
}
