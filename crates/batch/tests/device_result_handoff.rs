//! Handing off an asynchronously completed, device-resident result.
//!
//! A real encoder returns device storage that is still being written when the
//! runtime hands the result over. The runtime contract must therefore let a
//! consumer (a) observe the producer's completion dependency, (b) keep the result's
//! pool charge alive while it holds the storage, and (c) release that charge only
//! when the storage and a stalled consumer both let go. These tests use the real
//! [`ribn_batch::BatchRuntime`] and [`ribn_foundation::BytePool`]; the device is a
//! fixture, because a host test cannot own device memory.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, BlockReason, Job, JobOutput,
    StepOutcome, Terminal,
};
use ribn_foundation::{BytePool, ParameterVersion};

const ENVELOPE: u64 = 1 << 16;

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
}

impl DeferredEncoder {
    fn new() -> Self {
        Self {
            complete: Arc::new(AtomicBool::new(false)),
        }
    }

    fn complete_pending(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }
}

impl BatchExecutor for DeferredEncoder {
    type Input = usize;
    type Output = DeviceResult;
    type Error = Infallible;
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
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        // Enqueue only: nothing here waits for the device, which is what makes the
        // result asynchronous.
        self.complete.store(false, Ordering::SeqCst);
        Ok(batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                let rows = job.into_input();
                JobOutput::new(
                    request,
                    DeviceResult {
                        bytes: ENVELOPE,
                        complete: Arc::clone(&self.complete),
                        pooled: vec![f32::from(u8::try_from(rows).unwrap_or(0)); 4],
                    },
                )
            })
            .collect())
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
