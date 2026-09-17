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
//! - `Admission::Deferred` is a registered per-request encoder gate. The engine
//!   retries after encoder publication; no prompt token is committed while it waits.
//! - A backend can perform encoder work inside its own step and reuse the output
//!   across requests, and the engine's loop neither duplicates nor loses that work.
//! - A prefill row may accept *fewer* inputs than the engine offered
//!   (`crates/runtime/src/engine/completion.rs`), so a step budget smaller than one
//!   indivisible item no longer forces a per-row overrun. The submission budget is
//!   aggregate: rows share one step budget instead of each resetting it.
//! - An indivisible item that exceeds the whole step budget can never run, so the
//!   backend rejects it at admission. That is request-local: peers keep running.
//!
//! What is still not expressible: a *temporary* inability of a row to make progress
//! (for example, an earlier row consumed the aggregate budget before this row's
//! leading item). There is no completion-time waiting outcome any more, so a backend
//! must not submit a row whose first action is an unaffordable item. The fixture
//! therefore relies on leading plain tokens to guarantee positive progress; tests
//! that would need temporary waiting are documented limitations, not fabricated
//! successes.

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
    readiness: ribn::Readiness,
    /// Encoder compute one whole submission may spend, shared across its rows.
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

/// One submitted row plus the range the backend decided it could afford. The
/// decision is made in `submit`, before any encoder work, and `poll` only
/// reports it: a backend that executes first and negotiates afterwards has
/// already overspent.
struct Planned {
    item: BatchItem,
    /// Inputs this row will consume, starting at `item.prefix`. The fixture only
    /// plans rows that advance, because a completion cannot report zero progress.
    accepted: u32,
    tokens: Vec<u32>,
}

struct EncoderModel {
    info: ExecutorInfo,
    control: Shared,
    items: HashMap<SequenceId, Vec<EncoderItem>>,
    prompt_tokens: HashMap<SequenceId, u32>,
    pending: Option<Vec<Planned>>,
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
            prompt_tokens: HashMap::new(),
            pending: None,
            next_submission: 0,
        }
    }

    /// The longest prefix of this row that fits the shared remaining budget,
    /// consuming prompt tokens up to the first item that cannot be encoded. An
    /// item is indivisible: its span cannot be entered without paying its whole
    /// cost. Admission rejects a permanent impossibility, so any span reached
    /// here fits the whole step budget; the shared budget can still stop a row.
    fn plan_prefill(
        &self,
        item: &BatchItem,
        control: &mut Control,
        budget: &mut u32,
    ) -> (Planned, u32) {
        let limit = item.prefix + item.token_budget;
        let items = self.items_for(item.sequence);
        let mut accepted = item.prefix;
        let mut spent = 0_u32;
        while accepted < limit {
            let blocking = items
                .iter()
                .find(|encoder| encoder.span.0 <= accepted && accepted < encoder.span.1);
            match blocking {
                Some(encoder) if !control.published.contains(&encoder.id) => {
                    if encoder.compute > *budget {
                        // Not affordable inside this submission's shared budget.
                        break;
                    }
                    *budget -= encoder.compute;
                    spent = spent.saturating_add(encoder.compute);
                    control.encoded.push(encoder.id);
                    control.published.insert(encoder.id);
                    control.readiness.publish();
                }
                _ => {}
            }
            accepted += 1;
        }
        assert!(
            accepted > item.prefix,
            "the fixture must not submit a row that cannot advance"
        );
        let tokens = if accepted
            == self
                .prompt_tokens
                .get(&item.sequence)
                .copied()
                .unwrap_or_default()
            && item.output_budget == 1
        {
            vec![100 + item.prefix]
        } else {
            Vec::new()
        };
        (
            Planned {
                item: *item,
                accepted: accepted - item.prefix,
                tokens,
            },
            spent,
        )
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
        // Rejected/deferred admission retains no sequence-owned state.
        let items = prompt_items(&request.tokens);
        let mut control = self.control.lock().unwrap();
        // A permanent impossibility is a request-local rejection, not a wait.
        if items
            .iter()
            .any(|item| item.compute > control.step_budget && !control.published.contains(&item.id))
        {
            return Err(ExecutionError::new(
                "encoder item is larger than the step budget",
            ));
        }
        if control.gate_admission
            && let Some(first) = items.first()
            && !control.published.contains(&first.id)
        {
            control.deferred += 1;
            return Ok(Admission::Deferred(control.readiness.register()));
        }
        control.admitted += 1;
        self.items.insert(sequence, items);
        self.prompt_tokens.insert(
            sequence,
            u32::try_from(request.tokens.len()).expect("validated prompt length"),
        );
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        assert!(self.pending.is_none());
        let planned = {
            let mut control = self.control.lock().unwrap();
            let mut budget = control.step_budget;
            let mut spent = 0_u32;
            let planned = batch
                .iter()
                .map(|item| match item.kind {
                    ribn::StepKind::Prefill => {
                        let (plan, cost) = self.plan_prefill(item, &mut control, &mut budget);
                        spent = spent.saturating_add(cost);
                        plan
                    }
                    ribn::StepKind::Decode => Planned {
                        item: *item,
                        accepted: 1,
                        tokens: vec![100 + item.prefix],
                    },
                })
                .collect::<Vec<_>>();
            control.spent_per_submission.push(spent);
            if spent > control.step_budget {
                control.overruns += 1;
            }
            planned
        };
        self.pending = Some(planned);
        self.next_submission += 1;
        Ok(SubmissionId::new(self.next_submission))
    }

    fn poll(
        &mut self,
        _submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let Some(planned) = self.pending.take() else {
            return Ok(None);
        };
        Ok(Some(
            planned
                .into_iter()
                .map(|plan| StepCompletion {
                    sequence: plan.item.sequence,
                    prefix: plan.item.prefix + plan.accepted,
                    tokens: plan.tokens,
                })
                .collect(),
        ))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.items.remove(&sequence);
        self.prompt_tokens.remove(&sequence);
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
    {
        let mut control = control.lock().unwrap();
        control.published.insert(ItemId(1));
        control.readiness.publish();
    }
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

/// Two distinct prompts each carry one item that costs two units. With a shared
/// three-unit submission budget, the first row's item consumes two units and the
/// second row is shortened before its own item instead of both rows resetting the
/// budget. Both rows still advance, so the completion contract can report them.
#[test]
fn the_step_budget_is_shared_across_the_rows_of_one_submission() {
    let control = Arc::new(Mutex::new(Control {
        step_budget: 3,
        ..Control::default()
    }));
    let mut runtime = engine(&control, policy(8));
    runtime
        .enqueue(request(
            vec![1, 1, MEDIA, MEDIA, MEDIA, MEDIA, 1, 1, 1, 1],
            1,
        ))
        .expect("first enqueue");
    runtime
        .enqueue(request(
            vec![1, 1, 1, MEDIA, MEDIA, MEDIA, MEDIA, 1, 1, 1],
            1,
        ))
        .expect("second enqueue");
    let events = drain(&mut runtime);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                Event::Finished {
                    reason: FinishReason::Length,
                    ..
                }
            ))
            .count(),
        2,
        "both requests finish: {events:?}"
    );
    let control = control.lock().unwrap();
    assert_eq!(
        control.overruns, 0,
        "no submission overspent the shared budget"
    );
    // Item at [2,6) for `first`, item at [3,7) for `second`; the second item is
    // only encoded after the first submission spent the budget on the first.
    assert_eq!(control.encoded, vec![ItemId(2), ItemId(3)]);
    assert!(
        control
            .spent_per_submission
            .iter()
            .all(|spent| *spent <= control.step_budget),
        "shared budget was exceeded: {:?}",
        control.spent_per_submission
    );
    assert_eq!(
        control.spent_per_submission.first().copied(),
        Some(2),
        "the first submission only afforded the first row's item"
    );
}

/// An item larger than the whole step budget can never run, so admission rejects
/// it request-locally. A healthy peer with only affordable items keeps running,
/// and no encoder work is performed for the impossible item.
#[test]
fn an_impossible_item_is_rejected_at_admission_without_harming_its_peer() {
    let control = Arc::new(Mutex::new(Control {
        // Room for a one-unit item but not the wider two-unit item.
        step_budget: 1,
        ..Control::default()
    }));
    let mut runtime = engine(&control, policy(8));
    let impossible = runtime
        .enqueue(request(two_item_prompt(), 1))
        .expect("enqueue impossible");
    let healthy = runtime
        .enqueue(request(vec![1, 1, 1, 1], 1))
        .expect("enqueue healthy");

    let mut impossible_reason = None;
    let mut healthy_finished = false;
    for _ in 0..64 {
        runtime.step().expect("step");
        while let Some(event) = runtime.pop_event() {
            match event.request() {
                request if request == impossible => {
                    if let Event::Finished { reason, .. } = event {
                        impossible_reason = Some(reason);
                    }
                }
                request if request == healthy => {
                    if matches!(
                        event,
                        Event::Finished {
                            reason: FinishReason::Length,
                            ..
                        }
                    ) {
                        healthy_finished = true;
                    }
                }
                _ => {}
            }
        }
        if impossible_reason.is_some() && healthy_finished {
            break;
        }
    }

    let control = control.lock().unwrap();
    assert_eq!(control.overruns, 0);
    assert!(
        control.encoded.is_empty(),
        "no encoder work ran for an impossible item"
    );
    assert!(
        matches!(impossible_reason, Some(FinishReason::Failed(_))),
        "the impossible request is rejected, got {impossible_reason:?}"
    );
    assert!(
        healthy_finished,
        "the healthy peer completes despite the request-local rejection"
    );
}

/// With room for the wider item, the same prompt completes: the engine accepts a
/// shortened range, then the remainder on the next step, and the policy chunk no
/// longer has to align with encoder-item granularity.
#[test]
fn a_shortened_range_lets_a_chunk_spanning_items_complete_within_budget() {
    let control = Arc::new(Mutex::new(Control {
        step_budget: 2,
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
    assert_eq!(control.overruns, 0, "the budget held for every submission");
    assert_eq!(
        control.spent_per_submission.first().copied(),
        Some(1),
        "the first step accepted only up to the item it could afford"
    );
    assert_eq!(
        control.spent_per_submission.get(1).copied(),
        Some(2),
        "the next step afforded the wider item and finished the prompt"
    );
    assert_eq!(control.encoded, vec![ItemId(1), ItemId(4)]);
}
