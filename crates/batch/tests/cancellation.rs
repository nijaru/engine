//! Cancellation: intent recorded against a live request, delivered through the
//! ordinary terminal queue, and never a reason for the runtime to release storage or
//! a charge on its own.
//!
//! These tests use the real [`ribn_batch::BatchRuntime`] and
//! [`ribn_foundation::BytePool`]; the device is a fixture, because a host test cannot
//! own device memory.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, BlockReason, CancelOutcome, Job,
    JobOutput, StepOutcome,
};
use ribn_foundation::{BytePool, ParameterVersion};

const ENVELOPE: u64 = 1 << 14;

/// Stands in for the device storage a result holds: what matters here is that the
/// runtime never frees it or its charge, only the executor can.
struct DeviceResult;

/// An encoder whose results are retained until a proven drain, and which records what
/// it was actually asked to run.
struct CancellableEncoder {
    complete: Arc<AtomicBool>,
    batches: Vec<Vec<usize>>,
    quarantined: Vec<JobOutput<DeviceResult>>,
}

impl CancellableEncoder {
    fn new() -> Self {
        Self {
            complete: Arc::new(AtomicBool::new(false)),
            batches: Vec::new(),
            quarantined: Vec::new(),
        }
    }

    fn complete_pending(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }

    fn drain(&mut self) -> usize {
        if !self.complete.load(Ordering::SeqCst) {
            return 0;
        }
        let drained = self.quarantined.len();
        self.quarantined.clear();
        drained
    }

    fn quarantined_charge(&self) -> u64 {
        self.quarantined
            .iter()
            .map(|output| output.lease().bytes())
            .sum()
    }
}

impl BatchExecutor for CancellableEncoder {
    type Input = usize;
    type Output = DeviceResult;
    type Error = Infallible;
    type Constraint = Infallible;

    fn parameter_version(&self) -> ParameterVersion {
        ParameterVersion::new(1)
    }

    fn max_batch_items(&self) -> usize {
        2
    }

    fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
        BatchSelection::Ready {
            items: candidates.len(),
        }
    }

    fn retained_bytes(&self, _input: &Self::Input) -> u64 {
        ENVELOPE
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, ribn_batch::EnqueueError<Self::Error>> {
        self.complete.store(false, Ordering::SeqCst);
        self.batches
            .push(batch.iter().map(|job| *job.input()).collect());
        Ok(batch
            .into_iter()
            .map(|job| {
                let (request, _input, lease) = job.into_parts();
                JobOutput::new(request, DeviceResult, lease)
            })
            .collect())
    }

    fn retire(&mut self, outputs: Vec<JobOutput<Self::Output>>) {
        self.quarantined.extend(outputs);
    }
}

fn config() -> BatchConfig {
    BatchConfig {
        max_waiting_requests: 8,
        max_retained_results: 4,
    }
}

fn runtime(pool: &Arc<BytePool>) -> BatchRuntime<CancellableEncoder> {
    BatchRuntime::new(CancellableEncoder::new(), Arc::clone(pool), config()).expect("runtime")
}

#[test]
fn cancelling_a_waiting_request_reports_cancellation_without_reserving() {
    // A sibling holds the whole pool, so this request can only wait. Nothing is
    // reserved for waiting work, so abandoning it releases nothing.
    let pool = BytePool::new(ENVELOPE).shared();
    let held = pool.reserve(ENVELOPE).expect("sibling lease");
    let mut runtime = runtime(&pool);
    let request = runtime.submit(1).expect("request");
    assert!(matches!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool { .. })
    ));

    assert_eq!(runtime.cancel(request), CancelOutcome::Queued);
    assert_eq!(
        pool.granted(),
        ENVELOPE,
        "the sibling still holds its charge"
    );
    assert_eq!(
        runtime.step().expect("cancelling step"),
        StepOutcome::Cancelled { request }
    );
    assert_eq!(runtime.queued(), 0);
    let entry = runtime.pop_completed().expect("cancellation entry");
    assert_eq!(entry.request(), request);
    assert!(entry.is_cancelled());
    assert_eq!(entry.retained_bytes(), 0);
    assert!(runtime.executor().batches.is_empty());
    drop(held);
    assert_eq!(
        runtime.step().expect("idle step"),
        StepOutcome::Idle,
        "a cancelled request is never executed"
    );
}

#[test]
fn cancelling_a_retained_result_hands_storage_and_charge_to_the_executor() {
    let pool = BytePool::new(2 * ENVELOPE).shared();
    let mut runtime = runtime(&pool);
    let request = runtime.submit(1).expect("request");
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(pool.granted(), ENVELOPE);

    assert_eq!(runtime.cancel(request), CancelOutcome::Discarded);
    // The runtime never releases device storage or its charge itself.
    assert_eq!(pool.granted(), ENVELOPE);
    assert_eq!(runtime.executor().quarantined_charge(), ENVELOPE);
    assert_eq!(runtime.retained_results(), 1);
    let entry = runtime.pop_completed().expect("cancellation entry");
    assert!(entry.is_cancelled());
    assert_eq!(entry.retained_bytes(), 0);
    assert!(entry.output().is_none());

    // Completion still permits the release, so the charge is not lost.
    assert_eq!(runtime.executor_mut().drain(), 0);
    assert_eq!(pool.granted(), ENVELOPE);
    runtime.executor().complete_pending();
    assert_eq!(runtime.executor_mut().drain(), 1);
    assert_eq!(pool.granted(), 0);
}

#[test]
fn cancellation_is_deliverable_while_the_terminal_bound_is_full() {
    let pool = BytePool::new(8 * ENVELOPE).shared();
    let mut runtime = BatchRuntime::new(
        CancellableEncoder::new(),
        Arc::clone(&pool),
        BatchConfig {
            max_waiting_requests: 8,
            max_retained_results: 1,
        },
    )
    .expect("runtime");
    let first = runtime.submit(1).expect("first request");
    let second = runtime.submit(2).expect("second request");
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(runtime.retained_results(), 1);

    // The intent is recorded even though no terminal slot is free.
    assert_eq!(runtime.cancel(second), CancelOutcome::Queued);
    assert_eq!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::RetainedResults)
    );
    assert_eq!(runtime.queued(), 1, "the cancelled request keeps its place");
    assert_eq!(runtime.executor().batches.len(), 1);

    assert_eq!(
        runtime.pop_completed().expect("first entry").request(),
        first
    );
    assert_eq!(
        runtime.step().expect("step after consuming the entry"),
        StepOutcome::Cancelled { request: second }
    );
    assert!(
        runtime
            .pop_completed()
            .expect("cancellation")
            .is_cancelled()
    );
}

#[test]
fn a_cancelled_request_stops_its_accepted_range_and_never_runs() {
    let pool = BytePool::new(8 * ENVELOPE).shared();
    let mut runtime = runtime(&pool);
    let first = runtime.submit(1).expect("first request");
    let cancelled = runtime.submit(2).expect("second request");
    let third = runtime.submit(3).expect("third request");

    // `max_batch_items` is two, so without the cancellation the first step would run
    // the first two requests as one batch.
    assert_eq!(runtime.cancel(cancelled), CancelOutcome::Queued);
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 },
        "the accepted range stops before a cancelled request"
    );
    assert_eq!(runtime.executor().batches, vec![vec![1]]);

    assert_eq!(
        runtime.step().expect("cancelling step"),
        StepOutcome::Cancelled { request: cancelled }
    );
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(runtime.executor().batches, vec![vec![1], vec![3]]);
    let mut requests = Vec::new();
    while let Some(entry) = runtime.pop_completed() {
        requests.push((entry.request(), entry.is_cancelled()));
    }
    assert_eq!(
        requests,
        vec![(first, false), (cancelled, true), (third, false)],
        "every request still reports exactly one terminal outcome, in order"
    );
}

#[test]
fn cancelling_an_unknown_or_consumed_request_reports_unknown() {
    // The runtime issues request identity, so a consumed request is the only id a test
    // can name without fabricating one; an id that was never issued reports the same
    // outcome because it matches no live request.
    let pool = BytePool::new(2 * ENVELOPE).shared();
    let mut runtime = runtime(&pool);
    let request = runtime.submit(1).expect("request");
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 }
    );
    drop(runtime.pop_completed().expect("entry"));
    assert_eq!(runtime.cancel(request), CancelOutcome::Unknown);
    assert_eq!(runtime.cancel(request), CancelOutcome::Unknown);
}
