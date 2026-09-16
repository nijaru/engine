//! Handing off an asynchronously completed, device-resident result, and what an
//! executor owns when a batch fails or cannot be committed.
//!
//! A real encoder returns device storage that is still being written when the runtime
//! hands the result over. The runtime contract must therefore let a consumer (a)
//! observe the producer's completion dependency, (b) keep the result's pool charge
//! alive while it holds the storage, and (c) release that charge only when the storage
//! and a stalled consumer both let go. When the batch instead fails after reaching the
//! device, the executor — not the runtime — keeps the storage and the charge until it
//! can prove completion.
//!
//! These tests use the real [`ribn_batch::BatchRuntime`] and
//! [`ribn_foundation::BytePool`]; the device is a fixture, because a host test cannot
//! own device memory.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, BlockReason, EnqueueError, Job,
    JobOutput, RuntimeError, StepOutcome, Terminal,
};
use ribn_foundation::{BytePool, ParameterVersion};

const ENVELOPE: u64 = 1 << 16;

/// Why this fixture's device refused a batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceError {
    Unavailable,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("fixture device is unavailable")
    }
}

impl Error for DeviceError {}

/// How a fixture run misbehaves, to exercise the ownership rules around failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Misbehaviour {
    /// Refuse the batch before touching the device at all.
    Refuse,
    /// Enqueue this many requests and then fail.
    FailAfter(usize),
    /// Return one result fewer than submitted, keeping the rest.
    ShortReturn,
}

/// A result that is enqueued immediately and completes later.
///
/// `is_complete`/`synchronize`/`read` are the producer completion dependency a
/// consumer must respect; reading before completion is refused rather than racing
/// the producer.
struct DeviceResult {
    bytes: u64,
    complete: Arc<AtomicBool>,
    pooled: Vec<f32>,
}

impl DeviceResult {
    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::SeqCst)
    }

    fn synchronize(&self) {
        while !self.is_complete() {
            std::hint::spin_loop();
        }
    }

    fn read(&self) -> Result<&[f32], &'static str> {
        if self.is_complete() {
            Ok(&self.pooled)
        } else {
            Err("result read before the producer established completion")
        }
    }
}

/// An encoder that enqueues device work without waiting for it.
struct DeferredEncoder {
    complete: Arc<AtomicBool>,
    misbehaviour: Option<Misbehaviour>,
    /// Storage and charge a failed or uncommittable batch left with this executor.
    /// Nothing here is released until a proven drain.
    quarantined: Vec<JobOutput<DeviceResult>>,
}

impl DeferredEncoder {
    fn new() -> Self {
        Self {
            complete: Arc::new(AtomicBool::new(false)),
            misbehaviour: None,
            quarantined: Vec::new(),
        }
    }

    fn misbehaving(misbehaviour: Misbehaviour) -> Self {
        Self {
            misbehaviour: Some(misbehaviour),
            ..Self::new()
        }
    }

    fn complete_pending(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }

    fn result(rows: usize, complete: &Arc<AtomicBool>) -> DeviceResult {
        DeviceResult {
            bytes: ENVELOPE,
            complete: Arc::clone(complete),
            pooled: vec![f32::from(u8::try_from(rows).unwrap_or(0)); 4],
        }
    }

    /// Bytes this executor still owns after a failed or uncommittable batch.
    fn quarantined_bytes(&self) -> u64 {
        self.quarantined
            .iter()
            .map(|output| output.lease().bytes())
            .sum()
    }

    /// Release quarantined work once the device is known to have drained.
    fn drain(&mut self) -> usize {
        if !self.complete.load(Ordering::SeqCst) {
            return 0;
        }
        let drained = self.quarantined.len();
        self.quarantined.clear();
        drained
    }
}

impl BatchExecutor for DeferredEncoder {
    type Input = usize;
    type Output = DeviceResult;
    type Error = DeviceError;
    type Constraint = Infallible;

    fn parameter_version(&self) -> ParameterVersion {
        ParameterVersion::new(1)
    }

    fn max_batch_items(&self) -> usize {
        4
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
    ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>> {
        // Enqueue only: nothing here waits for the device, which is what makes the
        // result asynchronous.
        self.complete.store(false, Ordering::SeqCst);
        if self.misbehaviour == Some(Misbehaviour::Refuse) {
            // Dropping the whole batch releases every reservation: nothing reached
            // the device, so this is a clean refusal.
            return Err(EnqueueError::Refused(DeviceError::Unavailable));
        }
        let mut outputs = Vec::with_capacity(batch.len());
        for (index, job) in batch.into_iter().enumerate() {
            let (request, rows, lease) = job.into_parts();
            if self.misbehaviour == Some(Misbehaviour::FailAfter(index)) {
                // What already reached the device stays owned here, with its charge,
                // until a proven drain; this request and the ones after it never did.
                self.quarantined.append(&mut outputs);
                return Err(EnqueueError::Uncertain(DeviceError::Unavailable));
            }
            outputs.push(JobOutput::new(
                request,
                Self::result(rows, &self.complete),
                lease,
            ));
        }
        if self.misbehaviour == Some(Misbehaviour::ShortReturn) {
            // A contract violation: the runtime cannot map this onto its work, and
            // the result it holds back stays here with its charge.
            let withheld = outputs.pop().expect("the runtime submits at least one job");
            self.quarantined.push(withheld);
        }
        Ok(outputs)
    }

    fn retire(&mut self, outputs: Vec<JobOutput<Self::Output>>) {
        // Storage and charge come back together and wait for a proven drain.
        self.quarantined.extend(outputs);
    }
}

fn config() -> BatchConfig {
    BatchConfig {
        max_waiting_requests: 8,
        max_retained_results: 4,
    }
}

fn runtime(pool: &Arc<BytePool>) -> BatchRuntime<DeferredEncoder> {
    BatchRuntime::new(DeferredEncoder::new(), Arc::clone(pool), config()).expect("runtime")
}

#[test]
fn an_asynchronous_result_is_charged_until_its_consumer_releases_it() {
    let pool = BytePool::new(2 * ENVELOPE).shared();
    let mut runtime = runtime(&pool);
    runtime.submit(1).expect("first request");
    runtime.submit(2).expect("second request");
    runtime.submit(3).expect("third request");

    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 2 },
        "the accepted range is the prefix the pool can cover"
    );
    assert_eq!(pool.granted(), 2 * ENVELOPE);
    assert_eq!(runtime.retained_output_bytes(), 2 * ENVELOPE);
    assert_eq!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: ENVELOPE,
            available: 0
        })
    );

    // Nothing has completed on the device yet, and the consumer must not read.
    let entry = runtime.pop_completed().expect("completed entry");
    let output = entry.output().expect("output");
    assert!(!output.is_complete());
    assert!(output.read().is_err());

    // Taking the result takes its charge: the storage is still readable, so its
    // bytes stay reserved even though the entry left the runtime's terminal queue.
    let (outcome, lease) = entry.into_parts();
    let Terminal::Output(result) = outcome else {
        panic!("this request executed");
    };
    let lease = lease.expect("outputs carry a lease");
    assert_eq!(lease.bytes(), ENVELOPE);
    assert_eq!(result.bytes, lease.bytes(), "the charge covers the storage");
    assert_eq!(pool.granted(), 2 * ENVELOPE);
    assert_eq!(runtime.retained_output_bytes(), ENVELOPE);
    assert_eq!(
        runtime
            .step()
            .expect("step while the caller holds the charge"),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: ENVELOPE,
            available: 0
        }),
        "a popped result keeps its bytes reserved until its owner frees them"
    );

    // The producer completes later; only now can the consumer read.
    runtime.executor().complete_pending();
    result.synchronize();
    assert_eq!(result.read().expect("read"), [1.0, 1.0, 1.0, 1.0]);

    // Storage and charge travel together and are released together.
    drop(result);
    drop(lease);
    assert_eq!(pool.granted(), ENVELOPE, "the second result is still held");
    drop(runtime.pop_completed().expect("second entry"));
    assert_eq!(pool.granted(), 0);
    assert_eq!(
        runtime.step().expect("step after release"),
        StepOutcome::Executed { results: 1 }
    );
}

#[test]
fn a_stalled_consumer_blocks_a_sibling_while_a_healthy_peer_progresses() {
    // One authority, two runtimes: the stalled consumer's charge is what binds the
    // sibling, and a private per-runtime budget would have admitted it.
    let pool = BytePool::new(2 * ENVELOPE).shared();
    let mut stalled = runtime(&pool);
    let mut healthy = runtime(&pool);
    stalled.submit(1).expect("stalled request");
    stalled.submit(2).expect("stalled request");
    assert_eq!(
        stalled.step().expect("stalled step"),
        StepOutcome::Executed { results: 2 }
    );
    assert_eq!(pool.granted(), 2 * ENVELOPE);

    healthy.submit(7).expect("healthy request");
    assert_eq!(
        healthy.step().expect("sibling step"),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: ENVELOPE,
            available: 0
        })
    );

    // The consumer stops draining but keeps holding one result. Releasing exactly
    // one envelope lets the healthy peer run without disturbing the stalled one.
    let entry = stalled.pop_completed().expect("first entry");
    assert!(entry.output().is_some());
    drop(entry);
    assert_eq!(pool.granted(), ENVELOPE);
    assert_eq!(
        healthy.step().expect("sibling step after one release"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(pool.granted(), 2 * ENVELOPE);
}

#[test]
fn a_clean_refusal_releases_every_reservation() {
    let pool = BytePool::new(2 * ENVELOPE).shared();
    let mut runtime = BatchRuntime::new(
        DeferredEncoder::misbehaving(Misbehaviour::Refuse),
        Arc::clone(&pool),
        config(),
    )
    .expect("runtime");
    runtime.submit(1).expect("first request");
    runtime.submit(2).expect("second request");

    let error = runtime.step().expect_err("clean refusal");
    assert!(
        matches!(error, RuntimeError::Executor { .. }),
        "a refusal before device access is not uncertain: {error:?}"
    );
    assert_eq!(
        pool.granted(),
        0,
        "nothing reached the device, so no charge is retained"
    );
    assert_eq!(runtime.executor().quarantined_bytes(), 0);
    assert_eq!(runtime.retained_results(), 0);
}

#[test]
fn a_failure_after_device_access_keeps_the_charge_with_the_executor() {
    let pool = BytePool::new(4 * ENVELOPE).shared();
    let mut runtime = BatchRuntime::new(
        DeferredEncoder::misbehaving(Misbehaviour::FailAfter(2)),
        Arc::clone(&pool),
        config(),
    )
    .expect("runtime");
    for request in 0..4 {
        runtime.submit(request).expect("request");
    }

    let error = runtime.step().expect_err("partial submission");
    match error {
        RuntimeError::Uncertain { requests, .. } => assert_eq!(requests.len(), 4),
        other => panic!("a device-visible failure is uncertain, got {other:?}"),
    }
    // Two requests reached the device. Their storage and charge stay with the
    // executor; the two that never did are released, so a sibling can use them.
    assert_eq!(pool.granted(), 2 * ENVELOPE);
    assert_eq!(runtime.executor().quarantined_bytes(), 2 * ENVELOPE);
    assert_eq!(runtime.retained_results(), 0);
    assert_eq!(pool.available(), 2 * ENVELOPE);

    // Only a proven drain releases the quarantined storage and its charge.
    assert_eq!(
        runtime.executor_mut().drain(),
        0,
        "nothing has completed yet"
    );
    assert_eq!(pool.granted(), 2 * ENVELOPE);
    runtime.executor().complete_pending();
    assert_eq!(
        runtime.executor_mut().drain(),
        2,
        "completion is what permits release"
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
fn output_the_runtime_cannot_commit_is_handed_back_to_the_executor() {
    let pool = BytePool::new(2 * ENVELOPE).shared();
    let mut runtime = BatchRuntime::new(
        DeferredEncoder::misbehaving(Misbehaviour::ShortReturn),
        Arc::clone(&pool),
        config(),
    )
    .expect("runtime");
    runtime.submit(1).expect("first request");
    runtime.submit(2).expect("second request");

    let error = runtime.step().expect_err("malformed completion");
    match error {
        RuntimeError::MalformedCompletion { requests, returned } => {
            assert_eq!(requests.len(), 2);
            assert_eq!(returned, 1, "the executor returned one result too few");
        }
        other => panic!("expected a malformed completion, got {other:?}"),
    }
    // Nothing is committed, and nothing returns to the free list prematurely: the
    // storage and both charges are the executor's again.
    assert_eq!(runtime.retained_results(), 0);
    assert_eq!(runtime.executor().quarantined_bytes(), 2 * ENVELOPE);
    assert_eq!(pool.granted(), 2 * ENVELOPE);
    assert!(pool.reserve(ENVELOPE).is_err());

    runtime.executor().complete_pending();
    assert_eq!(runtime.executor_mut().drain(), 2);
    assert_eq!(pool.granted(), 0);
}
