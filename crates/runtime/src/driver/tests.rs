use super::*;
use crate::{
    Admission, BatchItem, EngineConfig, ExecutionError, ExecutorInfo, FinishReason,
    GenerationExecutor, GenerationLimits, GenerationOptions, SchedulePolicy, SequenceId,
    StepCompletion, SubmissionId,
};
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Instant;

struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn run<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        let left = deadline
            .checked_duration_since(Instant::now())
            .expect("future lost wakeup or stalled");
        std::thread::park_timeout(left);
    }
}

fn wait(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "worker did not reach expected state"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

struct Control {
    ready: AtomicBool,
    deferred: AtomicBool,
    fail_poll: AtomicBool,
    panic_poll: AtomicBool,
    fail_sync: AtomicBool,
    fail_release: AtomicBool,
    polls: AtomicUsize,
    admitted: AtomicUsize,
    released: AtomicUsize,
    synced: AtomicUsize,
    dropped: AtomicBool,
    panic_gate: Mutex<Option<(Sender<()>, Receiver<()>)>>,
}
impl Default for Control {
    fn default() -> Self {
        Self {
            ready: AtomicBool::new(true),
            deferred: AtomicBool::new(false),
            fail_poll: AtomicBool::new(false),
            panic_poll: AtomicBool::new(false),
            fail_sync: AtomicBool::new(false),
            fail_release: AtomicBool::new(false),
            polls: AtomicUsize::new(0),
            admitted: AtomicUsize::new(0),
            released: AtomicUsize::new(0),
            synced: AtomicUsize::new(0),
            dropped: AtomicBool::new(false),
            panic_gate: Mutex::new(None),
        }
    }
}

struct Model {
    info: ExecutorInfo,
    control: Arc<Control>,
    states: HashMap<SequenceId, u32>,
    pending: Option<Vec<BatchItem>>,
    next: u64,
}
impl GenerationExecutor for Model {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }
    fn admit(
        &mut self,
        _: RequestId,
        sequence: SequenceId,
        input: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        if self.control.deferred.load(Ordering::SeqCst) {
            return Ok(Admission::Deferred);
        }
        if input.tokens[0] == 99 {
            return Err(ExecutionError::new("request-local rejection"));
        }
        self.states.insert(sequence, input.tokens[0]);
        self.control.admitted.fetch_add(1, Ordering::SeqCst);
        Ok(Admission::Ready)
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        assert!(self.pending.is_none());
        self.pending = Some(batch.to_vec());
        self.next += 1;
        Ok(SubmissionId::new(self.next))
    }
    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        self.control.polls.fetch_add(1, Ordering::SeqCst);
        if let Some((entered, resume)) = self.control.panic_gate.lock().unwrap().take() {
            entered.send(()).unwrap();
            resume.recv().unwrap();
        }
        assert!(
            !self.control.panic_poll.swap(false, Ordering::SeqCst),
            "injected worker panic"
        );
        if self.control.fail_poll.load(Ordering::SeqCst) {
            return Err(ExecutionError::new("poll failed"));
        }
        if !self.control.ready.load(Ordering::SeqCst) {
            return Ok(None);
        }
        Ok(self.pending.take().map(|batch| {
            batch
                .into_iter()
                .map(|item| StepCompletion {
                    sequence: item.sequence,
                    prefix: item.prefix + item.token_budget,
                    tokens: vec![self.states[&item.sequence]; item.output_budget as usize],
                })
                .collect()
        }))
    }
    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        assert!(self.pending.is_none(), "premature release");
        if self.control.fail_release.load(Ordering::SeqCst) {
            return Err(ExecutionError::new("release failed"));
        }
        if self.states.remove(&sequence).is_some() {
            self.control.released.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.control.synced.fetch_add(1, Ordering::SeqCst);
        if self.control.fail_sync.load(Ordering::SeqCst) {
            return Err(ExecutionError::new("sync failed"));
        }
        self.pending = None;
        Ok(())
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        self.control.dropped.store(true, Ordering::SeqCst);
    }
}

fn test_engine(control: &Arc<Control>) -> Engine {
    let model = Model {
        info: ExecutorInfo {
            name: "driver fixture".into(),
            limits: GenerationLimits {
                context_tokens: 128,
                max_sequences: 2,
                max_batch_tokens: 8,
                max_decode_tokens: 1,
            },
        },
        control: control.clone(),
        states: HashMap::new(),
        pending: None,
        next: 0,
    };
    Engine::new(
        model,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 256,
            max_buffered_events: 4,
            max_events_per_request: 2,
        },
        SchedulePolicy {
            max_batch_tokens: 8,
            ..SchedulePolicy::default()
        },
    )
    .unwrap()
}

fn setup(control: &Arc<Control>, poll_interval: Duration) -> (DriverOwner, GenerationHandle) {
    Driver::spawn(
        test_engine(control),
        DriverConfig {
            max_requests: 2,
            max_input_bytes: 1024,
            events_per_request: 1,
            poll_interval,
        },
    )
    .unwrap()
}
fn fixture() -> (Arc<Control>, DriverOwner, GenerationHandle) {
    let control = Arc::new(Control::default());
    let (owner, handle) = setup(&control, Duration::from_millis(1));
    (control, owner, handle)
}
fn input(token: u32, output: u32) -> TokenRequest {
    TokenRequest::new(
        vec![token],
        GenerationOptions {
            max_output_tokens: output,
            ..GenerationOptions::default()
        },
    )
}
fn collect(stream: &mut GenerationStream) -> Vec<Event> {
    let mut result = Vec::new();
    while let Some(event) = run(stream.next()) {
        result.push(event.unwrap());
    }
    assert!(matches!(result.last(), Some(Event::Finished { .. })));
    result
}
fn park_hook(handle: &GenerationHandle) -> (Receiver<()>, Sender<()>) {
    let (entered_tx, entered) = flume::bounded(1);
    let (resume, resume_rx) = flume::bounded(1);
    *handle.shared.before_wait.lock().unwrap() = Some((entered_tx, resume_rx));
    handle.shared.notify();
    (entered, resume)
}

#[test]
fn two_callers_get_owned_ordered_streams_and_stalled_peers_do_not_block() {
    let (_, mut owner, handle) = fixture();
    let mut stalled = run(handle.stream(input(11, 8))).unwrap();
    let peer = handle.clone();
    let other = std::thread::spawn(move || {
        let mut stream = peer.stream_blocking(input(22, 8)).unwrap();
        let id = stream.request_id();
        let mut count = 0;
        while let Some(event) = stream.next_blocking() {
            let event = event.unwrap();
            assert_eq!(event.request(), id);
            if let Event::Token { token, .. } = event {
                assert_eq!(token, 22);
                count += 1;
            }
        }
        count
    });
    // A second independent caller can finish while the first has not read once.
    wait(|| other.is_finished());
    assert_eq!(other.join().unwrap(), 8);
    let id = stalled.request_id();
    let events = collect(&mut stalled);
    assert!(events.iter().all(|event| event.request() == id));
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Token { token: 11, .. }))
            .count(),
        8
    );
    owner.shutdown().unwrap();
}

#[test]
fn permits_cover_preparation_and_unconsumed_terminal_output() {
    let (control, mut owner, handle) = fixture();
    let reserved = handle.try_reserve().unwrap();
    let mut stream = run(handle.stream(input(7, 1))).unwrap();
    wait(|| control.released.load(Ordering::SeqCst) == 1);
    assert!(matches!(handle.try_reserve(), Err(DriverError::Overloaded)));
    collect(&mut stream);
    wait(|| handle.try_reserve().is_ok());
    drop(reserved);
    owner.shutdown().unwrap();
}

#[test]
fn encoded_byte_envelope_includes_stop_vector_capacity_and_runtime_copy() {
    let (_, mut owner, handle) = fixture();
    let mut request = input(7, 1);
    request.options.stop_tokens = Vec::with_capacity(1024);
    assert!(matches!(
        run(handle.stream(request)),
        Err(DriverError::InputTooLarge)
    ));
    let a = handle.try_reserve().unwrap();
    let b = handle.try_reserve().unwrap();
    assert!(matches!(handle.try_reserve(), Err(DriverError::Overloaded)));
    drop((a, b));
    owner.shutdown().unwrap();
}

#[test]
fn cancellation_is_delivered_under_saturated_admission_without_early_release() {
    let (control, mut owner, handle) = fixture();
    control.ready.store(false, Ordering::SeqCst);
    let mut stream = run(handle.stream(input(7, 8))).unwrap();
    wait(|| control.polls.load(Ordering::SeqCst) > 0);
    let other_permit = handle.try_reserve().unwrap();
    assert!(matches!(handle.try_reserve(), Err(DriverError::Overloaded)));
    stream.cancel();
    assert_eq!(control.released.load(Ordering::SeqCst), 0);
    control.ready.store(true, Ordering::SeqCst);
    handle.shared.notify();
    let events = collect(&mut stream);
    assert!(matches!(
        events.last(),
        Some(Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    drop(other_permit);
    owner.shutdown().unwrap();
}

#[test]
fn abandoned_in_flight_requests_keep_their_admission_charge_until_retirement() {
    let (control, mut owner, handle) = fixture();
    control.ready.store(false, Ordering::SeqCst);
    let stream = run(handle.stream(input(7, 8))).unwrap();
    wait(|| control.polls.load(Ordering::SeqCst) > 0);
    let spare = handle.try_reserve().unwrap();
    let before = control.polls.load(Ordering::SeqCst);
    drop(stream);
    wait(|| control.polls.load(Ordering::SeqCst) > before + 1);
    assert!(
        matches!(handle.try_reserve(), Err(DriverError::Overloaded)),
        "stream drop must not refund input/stop-list storage still owned by the engine"
    );
    control.ready.store(true, Ordering::SeqCst);
    handle.shared.notify();
    wait(|| handle.permits.len() == 1);
    assert_eq!(control.released.load(Ordering::SeqCst), 1);
    drop(spare);
    owner.shutdown().unwrap();
}

#[test]
fn output_credit_return_between_check_and_park_is_not_lost() {
    let control = Arc::new(Control::default());
    let (mut owner, handle) = setup(&control, Duration::from_secs(60));
    let mut stream = run(handle.stream(input(7, 8))).unwrap();
    wait(|| stream.events.is_full() && control.polls.load(Ordering::SeqCst) >= 2);
    let (entered, resume) = park_hook(&handle);
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(matches!(run(stream.next()), Some(Ok(Event::Token { .. }))));
    // The worker already checked credits but has not entered its blocking wait.
    resume.send(()).unwrap();
    collect(&mut stream);
    run(owner.shutdown_async()).unwrap();
}

#[test]
fn cancelling_submission_futures_before_and_after_enqueue_reclaims_interest() {
    let (control, mut owner, handle) = fixture();
    let (entered, resume) = park_hook(&handle);
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let mut pending = Box::pin(handle.stream(input(7, 8)));
    assert!(
        pending
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(pending);
    resume.send(()).unwrap();
    wait(|| handle.permits.len() == 2);
    assert_eq!(control.admitted.load(Ordering::SeqCst), 0);

    control.ready.store(false, Ordering::SeqCst);
    let (entered, resume) = park_hook(&handle);
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let mut pending = Box::pin(handle.stream(input(8, 8)));
    assert!(
        pending
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    resume.send(()).unwrap();
    wait(|| control.admitted.load(Ordering::SeqCst) == 1);
    drop(pending);
    control.ready.store(true, Ordering::SeqCst);
    handle.shared.notify();
    wait(|| handle.permits.len() == 2 && control.released.load(Ordering::SeqCst) == 1);
    owner.shutdown().unwrap();
}

#[test]
fn cancelling_next_future_loses_no_event_and_drop_wakes_idle_owner() {
    let (control, mut owner, handle) = fixture();
    control.ready.store(false, Ordering::SeqCst);
    let mut stream = run(handle.stream(input(7, 8))).unwrap();
    let mut next = Box::pin(stream.next());
    assert!(
        next.as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(next);
    control.ready.store(true, Ordering::SeqCst);
    handle.shared.notify();
    assert!(matches!(
        run(stream.next()),
        Some(Ok(Event::Token { token: 7, .. }))
    ));
    wait(|| stream.events.is_full());
    drop(stream);
    wait(|| handle.permits.len() == 2 && control.released.load(Ordering::SeqCst) == 1);
    owner.shutdown().unwrap();
}

#[test]
fn explicit_shutdown_failure_retains_worker_for_retry() {
    let (control, mut owner, handle) = fixture();
    control.ready.store(false, Ordering::SeqCst);
    let mut stream = run(handle.stream(input(7, 8))).unwrap();
    wait(|| control.polls.load(Ordering::SeqCst) > 0);
    control.fail_sync.store(true, Ordering::SeqCst);
    assert!(matches!(
        run(owner.shutdown_async()),
        Err(DriverError::Owner(_))
    ));
    assert!(!control.dropped.load(Ordering::SeqCst));
    assert_eq!(control.released.load(Ordering::SeqCst), 0);
    assert!(matches!(run(stream.next()), Some(Err(DriverError::Closed))));
    assert!(run(stream.next()).is_none());
    assert!(matches!(handle.try_reserve(), Err(DriverError::Closed)));
    control.fail_sync.store(false, Ordering::SeqCst);
    run(owner.shutdown_async()).unwrap();
    assert!(control.dropped.load(Ordering::SeqCst));
    assert_eq!(control.released.load(Ordering::SeqCst), 1);
    owner.shutdown().unwrap();
}

#[test]
fn cancelled_shutdown_future_keeps_its_retry_owner() {
    let (control, mut owner, handle) = fixture();
    let (entered, resume) = park_hook(&handle);
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    control.fail_sync.store(true, Ordering::SeqCst);
    let mut pending = Box::pin(owner.shutdown_async());
    assert!(
        pending
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(pending);
    resume.send(()).unwrap();
    wait(|| control.synced.load(Ordering::SeqCst) == 1);
    assert!(!control.dropped.load(Ordering::SeqCst));
    // The next call observes the abandoned attempt, not a fresh attempt that
    // silently overwrites its failure.
    assert!(matches!(
        run(owner.shutdown_async()),
        Err(DriverError::Owner(_))
    ));
    control.fail_sync.store(false, Ordering::SeqCst);
    run(owner.shutdown_async()).unwrap();
    assert!(control.dropped.load(Ordering::SeqCst));
}

#[test]
fn request_rejection_and_owner_fault_have_distinct_scope() {
    let (control, mut owner, handle) = fixture();
    let mut rejected = run(handle.stream(input(99, 1))).unwrap();
    assert!(matches!(
        collect(&mut rejected).last(),
        Some(Event::Finished {
            reason: FinishReason::Failed(_),
            ..
        })
    ));
    let mut healthy = run(handle.stream(input(7, 1))).unwrap();
    collect(&mut healthy);
    control.ready.store(false, Ordering::SeqCst);
    let mut a = run(handle.stream(input(8, 8))).unwrap();
    let mut b = run(handle.stream(input(9, 8))).unwrap();
    control.fail_poll.store(true, Ordering::SeqCst);
    handle.shared.notify();
    for stream in [&mut a, &mut b] {
        assert!(matches!(
            run(stream.next()),
            Some(Err(DriverError::Owner(_)))
        ));
        assert!(run(stream.next()).is_none());
    }
    assert!(matches!(handle.try_reserve(), Err(DriverError::Owner(_))));
    owner.shutdown().unwrap();
}

#[test]
fn failed_release_after_terminal_is_still_an_owner_failure() {
    let (control, mut owner, handle) = fixture();
    control.fail_release.store(true, Ordering::SeqCst);
    let mut stream = run(handle.stream(input(7, 1))).unwrap();
    assert!(matches!(
        run(stream.next()),
        Some(Err(DriverError::Owner(_)))
    ));
    assert!(matches!(owner.shutdown(), Err(DriverError::Owner(_))));
    assert!(!control.dropped.load(Ordering::SeqCst));
    control.fail_release.store(false, Ordering::SeqCst);
    owner.shutdown().unwrap();
}

#[test]
fn worker_panic_notifies_stream_and_explicit_shutdown() {
    let (control, mut owner, handle) = fixture();
    control.panic_poll.store(true, Ordering::SeqCst);
    let mut stream = run(handle.stream(input(7, 8))).unwrap();
    assert!(matches!(
        run(stream.next()),
        Some(Err(DriverError::WorkerPanicked))
    ));
    assert!(matches!(owner.shutdown(), Err(DriverError::WorkerPanicked)));
    assert!(control.dropped.load(Ordering::SeqCst));
    assert_eq!(control.released.load(Ordering::SeqCst), 1);
}

#[test]
fn owner_drop_cleans_up_without_requiring_stream_consumption() {
    let (control, owner, handle) = fixture();
    control.ready.store(false, Ordering::SeqCst);
    let mut stream = run(handle.stream(input(7, 8))).unwrap();
    wait(|| control.polls.load(Ordering::SeqCst) > 0);
    drop(owner);
    assert!(matches!(run(stream.next()), Some(Err(DriverError::Closed))));
    wait(|| control.dropped.load(Ordering::SeqCst));
    assert_eq!(control.released.load(Ordering::SeqCst), 1);
}

#[test]
fn queued_submission_and_shutdown_acknowledgements_are_dropped_on_worker_panic() {
    let (control, mut owner, handle) = fixture();
    let (entered_tx, entered) = flume::bounded(1);
    let (resume, resume_rx) = flume::bounded(1);
    *control.panic_gate.lock().unwrap() = Some((entered_tx, resume_rx));
    control.panic_poll.store(true, Ordering::SeqCst);
    let first = run(handle.stream(input(7, 8))).unwrap();
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    let mut queued = Box::pin(handle.stream(input(8, 8)));
    assert!(
        queued
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    let mut shutdown = Box::pin(owner.shutdown_async());
    assert!(
        shutdown
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    resume.send(()).unwrap();
    assert!(matches!(run(queued), Err(DriverError::WorkerPanicked)));
    assert!(matches!(run(shutdown), Err(DriverError::WorkerPanicked)));
    drop(first);
    assert_eq!(handle.permits.len(), 2);
    assert!(control.dropped.load(Ordering::SeqCst));
}

#[test]
fn cancelling_deferred_admission_wakes_without_a_device_batch() {
    let control = Arc::new(Control::default());
    control.deferred.store(true, Ordering::SeqCst);
    let (mut owner, handle) = setup(&control, Duration::from_secs(60));
    let mut stream = run(handle.stream(input(7, 1))).unwrap();
    stream.cancel();
    assert!(matches!(
        collect(&mut stream).last(),
        Some(Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    assert_eq!(control.admitted.load(Ordering::SeqCst), 0);
    owner.shutdown().unwrap();
}

#[test]
fn incompatible_driver_bounds_and_nonidle_engines_are_rejected() {
    let control = Arc::new(Control::default());
    let valid = DriverConfig {
        max_requests: 2,
        ..DriverConfig::default()
    };
    for config in [
        DriverConfig {
            max_requests: 0,
            ..valid
        },
        DriverConfig {
            max_requests: 3,
            ..valid
        }, // monopolizable global output pool
        DriverConfig {
            max_input_bytes: 0,
            ..valid
        },
        DriverConfig {
            max_input_bytes: usize::MAX,
            ..valid
        },
        DriverConfig {
            events_per_request: 0,
            ..valid
        },
        DriverConfig {
            events_per_request: usize::MAX,
            ..valid
        },
        DriverConfig {
            poll_interval: Duration::ZERO,
            ..valid
        },
    ] {
        assert!(matches!(
            Driver::spawn(test_engine(&control), config),
            Err(DriverError::InvalidConfig)
        ));
    }
    let mut engine = test_engine(&control);
    engine.enqueue(input(7, 1)).unwrap();
    assert!(matches!(
        Driver::spawn(engine, valid),
        Err(DriverError::InvalidConfig)
    ));
}

#[test]
fn enqueue_rejection_returns_its_source_and_refunds_the_permit() {
    use std::error::Error;
    let (_, mut owner, handle) = fixture();
    let Err(error) = run(handle.stream(input(7, 0))) else {
        panic!("zero requested output should be rejected");
    };
    assert!(matches!(
        error,
        DriverError::Enqueue(EngineError::InvalidRequest)
    ));
    assert!(error.source().is_some());
    wait(|| handle.permits.len() == 2);
    owner.shutdown().unwrap();
}

#[test]
fn diagnostic_payload_is_bounded_at_utf8_boundary() {
    let error = ExecutionError::new("é".repeat(10_000));
    let message = error.to_string();
    assert!(message.len() <= 4096);
    assert!(message.ends_with("… [truncated]"));
}
