//! Capacity waiting: registration-and-recheck against the pool's release epoch.
//!
//! The pool publishes an epoch, not a wakeup, so a runtime that parks on capacity
//! must register before attempting and recheck after a bounded park. These tests
//! exercise the protocol on the real [`ribn_batch::BatchRuntime`] and the real
//! [`ribn_foundation::BytePool`]; only the executor is a fixture.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, BlockReason, Job, JobOutput,
    StepOutcome,
};
use ribn_foundation::{BytePool, ParameterVersion};

/// Interval a driver parks for while no capacity notification exists.
const POLL: Duration = Duration::from_millis(5);

/// An executor whose retention is exactly the envelope its input names.
struct Envelope;

impl BatchExecutor for Envelope {
    type Input = u64;
    type Output = u64;
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

    fn retained_bytes(&self, input: &Self::Input) -> u64 {
        *input
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        Ok(batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                JobOutput::new(request, job.into_input())
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

fn runtime(pool: &Arc<BytePool>) -> BatchRuntime<Envelope> {
    BatchRuntime::new(Envelope, Arc::clone(pool), config()).expect("runtime")
}

#[test]
fn a_registration_observes_a_release_no_notification_delivered() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).expect("hold every byte");
    let mut runtime = runtime(&pool);
    runtime.submit(64).expect("request");

    // Register before attempting, exactly as a parking driver does.
    let wait = runtime.capacity_wait();
    assert_eq!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: 64,
            available: 0
        })
    );
    assert!(
        !wait.released(),
        "nothing has been released since the registration"
    );

    // No wakeup exists for this release; only the epoch changes.
    drop(held);
    assert!(wait.released());
    assert_eq!(
        runtime.step().expect("step after the release"),
        StepOutcome::Executed { results: 1 }
    );
}

#[test]
fn a_registration_is_taken_from_this_runtimes_own_pool() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).expect("hold every byte");
    let mut runtime = runtime(&pool);
    runtime.submit(64).expect("request");
    let wait = runtime.capacity_wait();
    assert!(matches!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool { .. })
    ));

    // A different pool's release cannot make this registration look satisfied.
    let other = BytePool::new(64).shared();
    let other_hold = other.reserve(64).expect("hold the other pool");
    drop(other_hold);
    assert!(!wait.released());

    drop(held);
    assert!(wait.released());
}

#[test]
fn a_release_during_a_bounded_park_is_not_lost() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).expect("hold every byte");
    let mut runtime = runtime(&pool);
    runtime.submit(64).expect("request");

    // The producer releases while this thread is parked on a channel that nothing
    // notifies: the release is only observable through the epoch recheck.
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(POLL);
        drop(held);
    });
    let (_parker, park) = std::sync::mpsc::channel::<()>();
    let mut wait = runtime.capacity_wait();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match runtime.step().expect("step") {
            StepOutcome::Executed { results } => {
                assert_eq!(results, 1);
                break;
            }
            StepOutcome::Blocked(BlockReason::Pool { .. }) => {
                if !wait.released() {
                    // Bounded, because the pool sends no notification.
                    let _ = park.recv_timeout(POLL);
                }
                wait = runtime.capacity_wait();
            }
            other => panic!("unexpected step outcome: {other:?}"),
        }
        assert!(
            Instant::now() < deadline,
            "the capacity wait never observed the release"
        );
    }
    releaser.join().expect("releaser");
    assert_eq!(pool.granted(), 64);
}
