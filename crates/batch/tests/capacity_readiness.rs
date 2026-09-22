//! Real byte-pool/batch-runtime readiness, with a bounded caller-owned wake channel.
use std::convert::Infallible;
use std::sync::{Arc, mpsc};
use std::task::{Wake, Waker};
use std::time::Duration;

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, BlockReason, EnqueueError, Job,
    JobOutput, RuntimeError, StepOutcome,
};
use ribn_foundation::{BytePool, ParameterVersion};

struct Notify(mpsc::SyncSender<()>);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}
fn notification() -> (Waker, mpsc::Receiver<()>) {
    let (sender, receiver) = mpsc::sync_channel(1);
    (Waker::from(Arc::new(Notify(sender))), receiver)
}
fn notified(receiver: &mpsc::Receiver<()>) {
    // A test deadline, not periodic capacity polling: one notification must arrive.
    receiver
        .recv_timeout(Duration::from_secs(3))
        .expect("resource notification");
}

/// The fixture executor's retained charge is exactly its input envelope.
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
    ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>> {
        Ok(batch
            .into_iter()
            .map(|job| {
                let (request, input, lease) = job.into_parts();
                JobOutput::new(request, input, lease)
            })
            .collect())
    }
    fn retire(&mut self, _outputs: Vec<JobOutput<Self::Output>>) {
        // No device storage: ordinary drop is safe retirement for this fixture.
    }
}
fn runtime(pool: &Arc<BytePool>) -> BatchRuntime<Envelope> {
    BatchRuntime::new(
        Envelope,
        Arc::clone(pool),
        BatchConfig {
            max_waiting_requests: 8,
            max_retained_results: 4,
        },
    )
    .expect("runtime")
}
fn blocked(runtime: &mut BatchRuntime<Envelope>) {
    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: 64,
            available: 0,
        })
    );
}
fn executed(runtime: &mut BatchRuntime<Envelope>) {
    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Executed { results: 1 }
    );
}

#[test]
fn a_release_before_arming_is_not_lost() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    assert!(!wait.changed());
    drop(held);
    assert!(wait.changed());
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    notified(&receiver);
    executed(&mut runtime);
}

#[test]
fn a_registration_is_taken_from_this_runtimes_own_pool() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    let other = BytePool::new(64).shared();
    drop(other.reserve(64).unwrap());
    assert!(!wait.changed());
    assert!(receiver.try_recv().is_err());
    drop(held);
    notified(&receiver);
    assert!(wait.changed());
}

#[test]
fn a_release_from_another_thread_wakes_without_polling() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    let releaser = std::thread::spawn(move || drop(held));
    notified(&receiver);
    releaser.join().unwrap();
    executed(&mut runtime);
}

#[test]
fn pool_closure_reactivates_a_wait_without_releasing_live_charges() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    pool.close();
    notified(&receiver);
    assert!(wait.changed());
    assert_eq!(pool.granted(), 64);
    assert!(matches!(runtime.step(), Err(RuntimeError::PoolClosed)));
    let mut already_closed = pool.capacity_wait();
    already_closed.wake_on_change(&waker);
    pool.close();
    assert!(
        !already_closed.changed(),
        "repeated closure changes no condition"
    );
    assert!(receiver.try_recv().is_err());
    drop(held);
    assert_eq!(pool.granted(), 0);
}

#[test]
fn cancelling_and_dropping_a_registration_detaches_its_notification() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    let request = runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    assert_eq!(runtime.cancel(request), ribn_batch::CancelOutcome::Queued);
    drop(wait);
    drop(held);
    assert!(receiver.try_recv().is_err());
    assert_eq!(pool.granted(), 0);
}

#[test]
fn dequeued_output_keeps_its_charge_and_only_retirement_wakes_a_sibling() {
    let pool = BytePool::new(64).shared();
    let mut first = runtime(&pool);
    let mut second = runtime(&pool);
    first.submit(64).unwrap();
    executed(&mut first);
    second.submit(64).unwrap();
    let mut wait = second.capacity_wait();
    blocked(&mut second);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    let output = first.pop_completed().unwrap();
    assert_eq!(pool.granted(), 64);
    assert!(!wait.changed());
    assert!(receiver.try_recv().is_err());
    drop(output);
    notified(&receiver);
    executed(&mut second);
}

#[test]
fn stolen_capacity_requires_a_fresh_registration_and_second_publication() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    drop(held);
    notified(&receiver);
    let competitor = pool.reserve(64).unwrap();
    let mut retry = runtime.capacity_wait();
    blocked(&mut runtime);
    retry.wake_on_change(&waker);
    assert!(!retry.changed());
    assert!(receiver.try_recv().is_err());
    drop(competitor);
    notified(&receiver);
    executed(&mut runtime);
}
