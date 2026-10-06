//! Real engine and byte-pool control transitions; no GPU or model qualification.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, BatchPreparation, Engine, EngineConfig, EngineError, Event,
    ExecutionError, ExecutorInfo, FinishReason, GenerationExecutor, GenerationLimits,
    GenerationOptions, PreparedRow, Readiness, RequestId, SchedulePolicy, SequenceId,
    StepCompletion, StepKind, SubmissionId, TokenRequest,
};
use ribn_foundation::{BytePool, PoolLease, ReserveError};

#[derive(Clone, Copy, Debug)]
enum Invalid {
    Zero,
    Beyond,
    Output,
    Prefix,
    Order,
    Count,
}

#[derive(Clone, Copy)]
enum Fault {
    Report(Invalid),
    Prepare,
    Abandon,
    Submit,
}

struct Control {
    ready: bool,
    held: bool,
    readiness: Readiness,
    shorten: Option<u32>,
    fault: Option<Fault>,
    fail_sync: bool,
    preparations: usize,
    submissions: Vec<Vec<BatchItem>>,
    abandons: usize,
    releases: usize,
}
impl Default for Control {
    fn default() -> Self {
        Self {
            ready: true,
            held: false,
            readiness: Readiness::default(),
            shorten: None,
            fault: None,
            fail_sync: false,
            preparations: 0,
            submissions: Vec::new(),
            abandons: 0,
            releases: 0,
        }
    }
}

// This host model owns one base byte and one new byte per consumed input. It
// retains both storage and the actual pool lease, including after injected faults.
struct Storage {
    _bytes: Box<[u8]>,
    _lease: PoolLease,
}
impl Storage {
    fn new(lease: PoolLease, bytes: usize) -> Self {
        Self {
            _bytes: vec![0; bytes].into_boxed_slice(),
            _lease: lease,
        }
    }
}
struct Continuation {
    _base: Storage,
    growth: Vec<Storage>,
    prompt: u32,
    token: u32,
}
struct Ready {
    item: BatchItem,
    growth: Storage,
}
struct Model {
    info: ExecutorInfo,
    pool: Arc<BytePool>,
    control: Arc<Mutex<Control>>,
    sequences: HashMap<SequenceId, Continuation>,
    prepared: Option<Vec<Ready>>,
    pending: Option<Vec<Ready>>,
}
impl GenerationExecutor for Model {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _: RequestId,
        id: SequenceId,
        input: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let wait = self.pool.capacity_wait();
        let base = match self.pool.reserve(1) {
            Ok(lease) => Storage::new(lease, 1),
            Err(ReserveError::Exhausted { .. }) => return Ok(Admission::Deferred(wait)),
            Err(error) => return Err(ExecutionError::new(error.to_string())),
        };
        self.sequences.insert(
            id,
            Continuation {
                _base: base,
                growth: Vec::new(),
                prompt: u32::try_from(input.tokens.len()).unwrap(),
                token: input.tokens[0],
            },
        );
        Ok(Admission::Ready)
    }

    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        assert!(self.prepared.is_none() && self.pending.is_none());
        let (shorten, fault, held, dependency) = {
            let mut control = self.control.lock().unwrap();
            control.preparations += 1;
            (
                control.shorten,
                control.fault,
                control.held,
                control.readiness.register(),
            )
        };
        let mut rows = Vec::with_capacity(batch.len());
        let mut accepted = Vec::with_capacity(batch.len());
        for offered in batch {
            let sequence = &self.sequences[&offered.sequence];
            if sequence.token == 77 && held {
                rows.push(PreparedRow::Deferred(dependency.clone()));
                continue;
            }
            if sequence.token == 99 {
                rows.push(PreparedRow::Rejected(ExecutionError::new(
                    "unsupported continuation",
                )));
                continue;
            }
            let wait = self.pool.capacity_wait();
            let available = u32::try_from(self.pool.available()).unwrap();
            if available == 0 {
                rows.push(PreparedRow::Deferred(wait));
                continue;
            }
            let mut item = *offered;
            item.token_budget = item
                .token_budget
                .min(shorten.unwrap_or(u32::MAX))
                .min(available);
            item.output_budget = match item.kind {
                StepKind::Prefill => u32::from(item.prefix + item.token_budget == sequence.prompt),
                StepKind::Decode => item.token_budget,
            };
            match self.pool.reserve(u64::from(item.token_budget)) {
                Ok(lease) => {
                    accepted.push(Ready {
                        item,
                        growth: Storage::new(lease, item.token_budget as usize),
                    });
                    rows.push(PreparedRow::Ready(item));
                }
                Err(ReserveError::Exhausted { .. }) => rows.push(PreparedRow::Deferred(wait)),
                Err(error) => rows.push(PreparedRow::Rejected(ExecutionError::new(
                    error.to_string(),
                ))),
            }
        }
        self.prepared = Some(accepted);
        if matches!(fault, Some(Fault::Prepare)) {
            return Err(ExecutionError::new("uncertain preparation"));
        }
        let invalid = match fault {
            Some(Fault::Report(invalid)) => Some(invalid),
            Some(Fault::Abandon) => Some(Invalid::Zero),
            _ => None,
        };
        if let Some(invalid) = invalid {
            match invalid {
                Invalid::Order => rows.reverse(),
                Invalid::Count => {
                    rows.pop();
                }
                _ => {
                    let PreparedRow::Ready(last) = rows.last_mut().unwrap() else {
                        panic!("ready fault target")
                    };
                    match invalid {
                        Invalid::Zero => last.token_budget = 0,
                        Invalid::Beyond => last.token_budget += 1,
                        Invalid::Output => last.output_budget = 0,
                        Invalid::Prefix => last.prefix += 1,
                        Invalid::Order | Invalid::Count => unreachable!(),
                    }
                }
            }
        }
        Ok(BatchPreparation::Selected(rows))
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        let mut control = self.control.lock().unwrap();
        control.abandons += 1;
        if matches!(control.fault, Some(Fault::Abandon)) {
            return Err(ExecutionError::new("preparation retirement failed"));
        }
        drop(control);
        self.prepared = None;
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let prepared = self.prepared.take().expect("executor owns preparation");
        assert_eq!(prepared.len(), batch.len());
        assert!(
            prepared
                .iter()
                .zip(batch)
                .all(|(row, item)| row.item == *item)
        );
        self.pending = Some(prepared);
        let mut control = self.control.lock().unwrap();
        control.submissions.push(batch.to_vec());
        if matches!(control.fault, Some(Fault::Submit)) {
            return Err(ExecutionError::new("partial enqueue failed"));
        }
        Ok(SubmissionId::new(control.submissions.len() as u64))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        if !self.control.lock().unwrap().ready {
            return Ok(None);
        }
        Ok(self.pending.take().map(|batch| {
            batch
                .into_iter()
                .map(|row| {
                    let sequence = self.sequences.get_mut(&row.item.sequence).unwrap();
                    sequence.growth.push(row.growth);
                    StepCompletion {
                        sequence: row.item.sequence,
                        prefix: row.item.prefix + row.item.token_budget,
                        tokens: vec![sequence.token; row.item.output_budget as usize],
                    }
                })
                .collect()
        }))
    }

    fn release(&mut self, id: SequenceId) -> Result<(), ExecutionError> {
        assert!(
            !self
                .prepared
                .iter()
                .chain(&self.pending)
                .flatten()
                .any(|row| row.item.sequence == id),
            "retiring device-visible continuation"
        );
        if self.sequences.remove(&id).is_some() {
            self.control.lock().unwrap().releases += 1;
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        if self.control.lock().unwrap().fail_sync {
            return Err(ExecutionError::new("completion unknown"));
        }
        self.prepared = None;
        self.pending = None;
        Ok(())
    }
}

fn engine(pool: &Arc<BytePool>, control: &Arc<Mutex<Control>>) -> Engine {
    engine_with_policy(pool, control, 2, 8)
}
fn engine_with_policy(
    pool: &Arc<BytePool>,
    control: &Arc<Mutex<Control>>,
    active: usize,
    budget: u32,
) -> Engine {
    Engine::new(
        Model {
            info: ExecutorInfo {
                name: "host byte-growth model".into(),
                limits: GenerationLimits {
                    context_tokens: 32,
                    max_sequences: active,
                    max_batch_tokens: budget,
                    max_decode_tokens: 4,
                },
            },
            pool: Arc::clone(pool),
            control: Arc::clone(control),
            sequences: HashMap::new(),
            prepared: None,
            pending: None,
        },
        EngineConfig {
            max_active_requests: active,
            max_queued_requests: 1,
            max_queued_input_tokens: 32,
            max_buffered_events: 4,
            max_events_per_request: 4,
        },
        SchedulePolicy {
            max_batch_tokens: budget,
            prefill_chunk_tokens: 8,
            decode_tokens: 4,
            ..SchedulePolicy::default()
        },
    )
    .unwrap()
}
fn request(tokens: Vec<u32>) -> TokenRequest {
    TokenRequest::new(
        tokens,
        GenerationOptions {
            max_output_tokens: 1,
            ..GenerationOptions::default()
        },
    )
}
fn finish(engine: &mut Engine) -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..16 {
        engine.step().unwrap();
        events.extend(std::iter::from_fn(|| engine.pop_event()));
        if engine.status().requests == 0 {
            return events;
        }
    }
    panic!("work did not finish");
}

#[test]
fn all_deferred_offers_do_not_hide_a_runnable_peer_from_the_driver() {
    let pool = BytePool::new(16).shared();
    let control = Arc::new(Mutex::new(Control {
        held: true,
        ..Control::default()
    }));
    let mut engine = engine_with_policy(&pool, &control, 3, 1);
    engine.enqueue(request(vec![77])).unwrap();
    engine.enqueue(request(vec![77])).unwrap();
    let healthy = engine.enqueue(request(vec![10])).unwrap();
    assert!(
        engine.step().unwrap().submitted,
        "ready work must not depend on an unrelated wake"
    );
    assert_eq!(control.lock().unwrap().preparations, 3);
    assert_eq!(engine.status().waiting, 2);
    engine.step().unwrap();
    assert!(matches!(
        engine.pop_event_for(healthy),
        Some(Event::Token { token: 10, .. })
    ));
    engine.shutdown().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
fn aggregate_growth_parks_rows_and_reactivates_after_peer_retirement() {
    for cancel_parked in [false, true] {
        let pool = BytePool::new(6).shared();
        let outside = pool.reserve(3).unwrap();
        let control = Arc::new(Mutex::new(Control {
            ready: false,
            ..Control::default()
        }));
        let mut engine = engine(&pool, &control);
        let first = engine.enqueue(request(vec![10])).unwrap();
        let second = engine.enqueue(request(vec![20, 21])).unwrap();
        assert!(engine.step().unwrap().submitted);
        assert_eq!(engine.status().waiting, 1);
        assert_eq!(pool.granted(), 6);
        assert_eq!(control.lock().unwrap().submissions[0].len(), 1);
        assert_eq!(engine.committed_prefix(first), Some(0));
        assert_eq!(engine.committed_prefix(second), Some(0));
        for _ in 0..3 {
            engine.step().unwrap();
        }
        assert_eq!(control.lock().unwrap().preparations, 1);
        if cancel_parked {
            engine.cancel(second).unwrap();
            engine.step().unwrap();
            assert_eq!(engine.status().waiting, 0);
            assert_eq!(pool.granted(), 5, "only the parked continuation retired");
            assert_eq!(engine.committed_prefix(first), Some(0));
        }
        control.lock().unwrap().ready = true;
        let events = finish(&mut engine);
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Token { .. }))
                .count(),
            if cancel_parked { 1 } else { 2 }
        );
        if cancel_parked {
            assert_eq!(control.lock().unwrap().submissions.len(), 1);
            assert!(events.iter().any(|event| matches!(event, Event::Finished { request, reason: FinishReason::Cancelled, usage } if *request == second && usage.completion_tokens == 0)));
        } else {
            assert_eq!(control.lock().unwrap().submissions[1][0].token_budget, 2);
        }
        assert_eq!(pool.granted(), 3);
        drop(outside);
        assert_eq!(pool.granted(), 0);
    }
}

#[test]
fn shortened_final_prefill_refunds_credit_before_completion() {
    let pool = BytePool::new(32).shared();
    let control = Arc::new(Mutex::new(Control {
        ready: false,
        shorten: Some(1),
        ..Control::default()
    }));
    let mut engine = engine(&pool, &control);
    engine.enqueue(request(vec![10; 3])).unwrap();
    engine.enqueue(request(vec![20; 3])).unwrap();
    engine.step().unwrap();
    assert!(
        control.lock().unwrap().submissions[0]
            .iter()
            .all(|item| item.token_budget == 1 && item.output_budget == 0)
    );
    let cancelled = engine.enqueue(request(vec![30])).unwrap();
    engine.cancel(cancelled).unwrap();
    engine.step().unwrap();
    assert!(matches!(
        engine.pop_event_for(cancelled),
        Some(Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    assert!(engine.status().in_flight);
    control.lock().unwrap().ready = true;
    let events = finish(&mut engine);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Token { .. }))
            .count(),
        2
    );
    assert!(
        events
            .iter()
            .filter_map(|e| match e {
                Event::Finished { usage, .. } => Some(usage),
                Event::Token { .. } => None,
            })
            .all(|usage| usage.prompt_tokens == 3 && usage.completion_tokens == 1)
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
fn shortened_decode_refunds_credit_and_commits_only_accepted_tokens() {
    let pool = BytePool::new(16).shared();
    let control = Arc::new(Mutex::new(Control {
        shorten: Some(1),
        ..Control::default()
    }));
    let mut engine = engine(&pool, &control);
    let active = engine
        .enqueue(TokenRequest::new(
            vec![10],
            GenerationOptions {
                max_output_tokens: 4,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();
    engine.step().unwrap();
    engine.step().unwrap();
    let item = control.lock().unwrap().submissions[1][0];
    assert_eq!(item.kind, StepKind::Decode);
    assert_eq!((item.token_budget, item.output_budget), (1, 1));
    assert_eq!(engine.committed_prefix(active), Some(1));
    control.lock().unwrap().ready = false;
    let cancelled = engine.enqueue(request(vec![30])).unwrap();
    engine.cancel(cancelled).unwrap();
    engine.step().unwrap();
    // One prefill token remains buffered. Shrinking the offered decode from
    // two outputs to one must leave credit for this terminal before completion.
    assert!(matches!(
        engine.pop_event_for(cancelled),
        Some(Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    assert!(engine.status().in_flight);
    control.lock().unwrap().ready = true;
    let events = finish(&mut engine);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Token { .. }))
            .count(),
        4
    );
    assert!(
        matches!(events.last(), Some(Event::Finished { usage, .. }) if usage.completion_tokens == 4)
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
fn all_waiting_abandons_new_work_and_cancellation_cannot_reactivate_it() {
    let pool = BytePool::new(6).shared();
    let outside = pool.reserve(4).unwrap();
    let control = Arc::new(Mutex::new(Control::default()));
    let mut engine = engine(&pool, &control);
    let cancelled = engine.enqueue(request(vec![10])).unwrap();
    let survivor = engine.enqueue(request(vec![20])).unwrap();
    let status = engine.step().unwrap();
    assert!(!status.submitted && !status.output_blocked);
    assert_eq!(engine.status().waiting, 2);
    assert_eq!(control.lock().unwrap().abandons, 1);
    for _ in 0..3 {
        engine.step().unwrap();
    }
    assert_eq!(control.lock().unwrap().preparations, 1);
    engine.cancel(cancelled).unwrap();
    engine.step().unwrap();
    assert!(matches!(
        engine.pop_event_for(cancelled),
        Some(Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    // Cancellation returns its base byte: the one-token survivor is now
    // feasible even before the unrelated external lease returns.
    assert!(engine.status().in_flight);
    drop(outside);
    let events = finish(&mut engine);
    assert!(events.iter().all(|event| event.request() == survivor));
    assert_eq!(control.lock().unwrap().releases, 2);
    assert_eq!(pool.granted(), 0);
}

#[test]
fn permanent_preparation_rejection_does_not_fault_a_healthy_peer() {
    let pool = BytePool::new(16).shared();
    let control = Arc::new(Mutex::new(Control::default()));
    let mut engine = engine(&pool, &control);
    let rejected = engine.enqueue(request(vec![99])).unwrap();
    let healthy = engine.enqueue(request(vec![10])).unwrap();
    assert!(engine.step().unwrap().submitted);
    assert!(!engine.status().faulted);
    assert!(matches!(
        engine.pop_event_for(rejected),
        Some(Event::Finished {
            reason: FinishReason::Failed(_),
            ..
        })
    ));
    assert_eq!(engine.status().active_sequences, 1);
    assert_eq!(control.lock().unwrap().submissions[0].len(), 1);
    assert!(
        finish(&mut engine)
            .iter()
            .all(|event| event.request() == healthy)
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
fn malformed_whole_reports_abandon_only_new_growth_without_logical_commit() {
    for invalid in [
        Invalid::Zero,
        Invalid::Beyond,
        Invalid::Output,
        Invalid::Prefix,
        Invalid::Order,
        Invalid::Count,
    ] {
        let pool = BytePool::new(16).shared();
        let control = Arc::new(Mutex::new(Control {
            fault: Some(Fault::Report(invalid)),
            ..Control::default()
        }));
        let mut engine = engine(&pool, &control);
        let first = engine.enqueue(request(vec![10; 2])).unwrap();
        let second = engine.enqueue(request(vec![20; 2])).unwrap();
        assert!(
            matches!(engine.step(), Err(EngineError::Faulted(_))),
            "{invalid:?}"
        );
        assert_eq!(engine.committed_prefix(first), Some(0));
        assert_eq!(engine.committed_prefix(second), Some(0));
        assert_eq!(control.lock().unwrap().abandons, 1);
        assert!(control.lock().unwrap().submissions.is_empty());
        assert_eq!(pool.granted(), 2, "old continuations stay owned");
        engine.shutdown().unwrap();
        assert_eq!(pool.granted(), 0);
    }
}

#[test]
fn preparation_abort_and_enqueue_faults_retain_old_and_new_storage_until_barrier() {
    for fault in [Fault::Prepare, Fault::Abandon, Fault::Submit] {
        let pool = BytePool::new(16).shared();
        let control = Arc::new(Mutex::new(Control {
            fault: Some(fault),
            fail_sync: true,
            ..Control::default()
        }));
        let mut engine = engine(&pool, &control);
        engine.enqueue(request(vec![10; 2])).unwrap();
        engine.enqueue(request(vec![20; 2])).unwrap();
        assert!(matches!(engine.step(), Err(EngineError::Faulted(_))));
        assert_eq!(pool.granted(), 6);
        assert!(engine.shutdown().is_err());
        assert_eq!(pool.granted(), 6);
        assert_eq!(control.lock().unwrap().releases, 0);
        control.lock().unwrap().fail_sync = false;
        engine.shutdown().unwrap();
        assert_eq!(pool.granted(), 0);
    }
}
