//! Encoder-availability pressure through the real admission and completion loop.
//!
//! `multimodal_dependencies.rs` asks whether a *proposed* scheduling decision can
//! exist: given prompt-position dependencies and two encoder budgets, how should one
//! prefill range be chosen? That answers a design question but proves nothing about
//! the engine, because a pure function can express decisions the request contract
//! cannot.
//!
//! This file asks the complementary question through the real `Engine`: which of
//! those decisions can today's admission, submission, and completion contract
//! actually carry, and which cannot? It is test-only and defines no production
//! multimodal API; its value is that the answers are observed from the engine rather
//! than assumed by a model.
//!
//! What it establishes:
//!
//! - `Admission::Deferred` is a working per-request encoder gate. The engine retries
//!   the request every step until the backend admits it, and no prompt token is
//!   committed while it waits.
//! - A backend can perform encoder work inside its own step and reuse the output
//!   across requests, and the engine's loop neither duplicates nor loses that work.
//! - A prefill completion must advance *exactly* the chunk the engine chose
//!   (`crates/runtime/src/engine/completion.rs`), so a backend cannot shorten a step
//!   to stop before an unavailable placeholder. Its only alternatives are to do the
//!   work anyway or to fail the submission, which faults every request in the batch.
//!   Encoder budgets therefore hold only when `SchedulePolicy::prefill_chunk_tokens`
//!   aligns with prompt-item granularity; the third test pins that constraint.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, Event, ExecutionError, ExecutorInfo, FinishReason,
    GenerationExecutor, GenerationLimits, GenerationOptions, RequestId, SchedulePolicy, SequenceId,
    StepCompletion, SubmissionId, TokenRequest,
};

/// Prompt tokens with this value mark a placeholder span that a prompt-position
/// encoder item occupies.
const MEDIA: u32 = 7;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ItemId(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EncoderItem {
    id: ItemId,
    /// Half-open token span the encoder output replaces.
    span: (u32, u32),
    /// Encoder work this item costs, charged against the step's budget.
    compute: u32,
}

impl EncoderItem {
    fn overlaps(self, start: u32, end: u32) -> bool {
        self.span.0 < end && start < self.span.1
    }
}

/// Prompt-position dependencies declared by the prompt itself: each maximal run of
/// placeholder tokens is one item, its span is the run, and its encoder cost is half
/// the width so that wider media cost strictly more.
fn prompt_items(prompt: &[u32]) -> Vec<EncoderItem> {
    let mut items = Vec::new();
    let mut position = 0_usize;
    while position < prompt.len() {
        if prompt[position] != MEDIA {
            position += 1;
            continue;
        }
        let start = position;
        while position < prompt.len() && prompt[position] == MEDIA {
            position += 1;
        }
        let width = u32::try_from(position - start).expect("span fits u32");
        let start = u32::try_from(start).expect("span fits u32");
        items.push(EncoderItem {
            id: ItemId(start),
            span: (start, start + width),
            compute: width.div_ceil(2),
        });
    }
    items
}

#[derive(Default)]
struct Control {
    /// Encoder items whose output an external component has already published.
    published: HashSet<ItemId>,
    /// Encoder work this backend performed, in order. A repeated id means the
    /// output was recomputed instead of reused.
    encoded: Vec<ItemId>,
    /// Whether admission waits for the first item to be published.
    gate_admission: bool,
    /// Encoder compute each submission may spend.
    step_budget: u32,
    /// Submissions that had to spend more than `step_budget`.
    overruns: usize,
    /// Admission attempts the backend refused, and admissions it granted.
    deferred: usize,
    admitted: usize,
    /// Encoder compute spent per submission, for budget assertions.
    spent_per_submission: Vec<u32>,
}

type Shared = Arc<Mutex<Control>>;

struct EncoderModel {
    info: ExecutorInfo,
    control: Shared,
    items: HashMap<SequenceId, Vec<EncoderItem>>,
    pending: Option<Vec<BatchItem>>,
    next_submission: u64,
}

impl EncoderModel {
    fn new(control: Shared) -> Self {
        Self {
            info: ExecutorInfo {
                name: "encoder-dependency model".to_owned(),
                limits: GenerationLimits {
                    context_tokens: 256,
                    max_sequences: 4,
                    max_batch_tokens: 32,
                    max_decode_tokens: 4,
                },
            },
            control,
            items: HashMap::new(),
            pending: None,
            next_submission: 0,
        }
    }

    fn items_for(&self, sequence: SequenceId) -> Vec<EncoderItem> {
        self.items.get(&sequence).cloned().unwrap_or_default()
    }
}

impl GenerationExecutor for EncoderModel {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        // Admission may be retried, so registration must be idempotent.
        self.items
            .entry(sequence)
            .or_insert_with(|| prompt_items(&request.tokens));
        let control = self.control.lock().unwrap();
        if control.gate_admission
            && let Some(first) = self.items_for(sequence).first()
            && !control.published.contains(&first.id)
        {
            drop(control);
            self.control.lock().unwrap().deferred += 1;
            return Ok(Admission::Deferred);
        }
        drop(control);
        self.control.lock().unwrap().admitted += 1;
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        assert!(self.pending.is_none());
        let mut spent = 0_u32;
        {
            let mut control = self.control.lock().unwrap();
            for item in batch {
                for encoder in self.items_for(item.sequence) {
                    if !encoder.overlaps(item.prefix, item.prefix + item.token_budget)
                        || control.published.contains(&encoder.id)
                    {
                        continue;
                    }
                    spent = spent.saturating_add(encoder.compute);
                    control.encoded.push(encoder.id);
                    control.published.insert(encoder.id);
                }
            }
            control.spent_per_submission.push(spent);
            if spent > control.step_budget {
                control.overruns += 1;
            }
        }
        self.pending = Some(batch.to_vec());
        self.next_submission += 1;
        Ok(SubmissionId::new(self.next_submission))
    }

    fn poll(
        &mut self,
        _submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let Some(batch) = self.pending.take() else {
            return Ok(None);
        };
        Ok(Some(
            batch
                .iter()
                .map(|item| match item.kind {
                    ribn::StepKind::Prefill => {
                        // The engine asks for sampling on the chunk that completes
                        // the prompt, and validates the row against that budget.
                        let tokens = if item.output_budget == 1 {
                            vec![100 + item.prefix]
                        } else {
                            Vec::new()
                        };
                        StepCompletion {
                            sequence: item.sequence,
                            prefix: item.prefix + item.token_budget,
                            tokens,
                        }
                    }
                    ribn::StepKind::Decode => StepCompletion {
                        sequence: item.sequence,
                        prefix: item.prefix + 1,
                        tokens: vec![100 + item.prefix],
                    },
                })
                .collect(),
        ))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.items.remove(&sequence);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.pending = None;
        Ok(())
    }
}

fn engine(control: &Shared, policy: SchedulePolicy) -> Engine {
    Engine::new(
        EncoderModel::new(control.clone()),
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 4096,
            max_buffered_events: 32,
            max_events_per_request: 64,
        },
        policy,
    )
    .expect("engine")
}

fn policy(prefill_chunk_tokens: u32) -> SchedulePolicy {
    SchedulePolicy {
        // The executor advertises a 32-token batch limit, and the engine refuses
        // a policy that exceeds it.
        max_batch_tokens: 32,
        prefill_chunk_tokens,
        ..SchedulePolicy::default()
    }
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

fn drain(engine: &mut Engine) -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..4000 {
        engine.step().expect("step");
        while let Some(event) = engine.pop_event() {
            events.push(event);
        }
        if engine.status().requests == 0 {
            return events;
        }
    }
    panic!("runtime did not finish");
}

/// Two placeholder spans separated by real tokens: items at [1,3) and [4,8).
fn two_item_prompt() -> Vec<u32> {
    vec![1, MEDIA, MEDIA, 1, MEDIA, MEDIA, MEDIA, MEDIA, 1, 1, 1, 1]
}

#[test]
fn deferred_admission_is_a_real_encoder_gate_that_the_engine_retries() {
    let control = Arc::new(Mutex::new(Control {
        gate_admission: true,
        step_budget: 8,
        ..Control::default()
    }));
    let mut runtime = engine(&control, policy(4));
    let request = runtime
        .enqueue(request(two_item_prompt(), 2))
        .expect("enqueue");

    // Nothing can advance while the first item is unpublished, no matter how many
    // times the engine evaluates the request.
    for _ in 0..8 {
        runtime.step().expect("step");
    }
    assert!(control.lock().unwrap().deferred > 0);
    assert_eq!(control.lock().unwrap().admitted, 0);
    // A deferred request exists but has committed nothing: the engine reports its
    // prefix as zero rather than pretending progress.
    assert_eq!(runtime.committed_prefix(request), Some(0));
    assert!(!runtime.status().in_flight);
    assert!(control.lock().unwrap().encoded.is_empty());
    assert!(runtime.pop_event().is_none());

    // An external encoder publishing the item is what unblocks admission.
    control.lock().unwrap().published.insert(ItemId(1));
    let events = drain(&mut runtime);
    assert!(matches!(
        events.last(),
        Some(Event::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
    let control = control.lock().unwrap();
    assert_eq!(control.admitted, 1);
    // The published item is not recomputed, and the remaining item is encoded
    // exactly once even though admission was attempted repeatedly.
    assert_eq!(control.encoded, vec![ItemId(4)]);
}

#[test]
fn coupled_encoder_work_is_reused_across_requests_with_the_same_prompt() {
    let control = Arc::new(Mutex::new(Control {
        step_budget: 4,
        ..Control::default()
    }));
    // Chunk 4 matches the item granularity: [1,3) costs 1 and [4,8) costs 2.
    let mut runtime = engine(&control, policy(4));
    runtime
        .enqueue(request(two_item_prompt(), 1))
        .expect("first enqueue");
    runtime
        .enqueue(request(two_item_prompt(), 1))
        .expect("second enqueue");
    drain(&mut runtime);

    let control = control.lock().unwrap();
    assert_eq!(control.admitted, 2, "both requests were admitted");
    assert_eq!(
        control.encoded,
        vec![ItemId(1), ItemId(4)],
        "the second request must reuse encoder output instead of recomputing it"
    );
    assert_eq!(control.overruns, 0);
    assert!(
        control
            .spent_per_submission
            .iter()
            .all(|spent| *spent <= control.step_budget),
        "per-step encoder budget was exceeded: {:?}",
        control.spent_per_submission
    );
}

/// The constraint, pinned rather than assumed: a prefill completion must advance
/// exactly the chunk the policy chose, so a backend cannot stop before an
/// unencoded placeholder. When the chunk spans more encoder work than the step
/// budget allows, the work happens anyway; the alternative is failing the batch,
/// which faults unrelated requests too.
#[test]
fn encoder_budget_holds_only_when_the_policy_chunk_matches_item_granularity() {
    let control = Arc::new(Mutex::new(Control {
        // Room for one item's work, but not for both items at once.
        step_budget: 1,
        ..Control::default()
    }));
    let mut runtime = engine(&control, policy(8));
    let events = {
        runtime
            .enqueue(request(two_item_prompt(), 1))
            .expect("enqueue");
        drain(&mut runtime)
    };

    assert!(matches!(
        events.last(),
        Some(Event::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
    let control = control.lock().unwrap();
    assert_eq!(
        control.overruns, 1,
        "an eight-token chunk spans both items, so one submission must overspend"
    );
    assert_eq!(
        control.spent_per_submission.first().copied(),
        Some(3),
        "the model spent both items' compute in one step because it had no way to refuse"
    );
}
