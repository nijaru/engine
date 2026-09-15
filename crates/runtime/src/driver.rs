//! Owned access to the token runtime, without a second scheduler.
//!
//! Admission is fail-fast. Reserve before preprocessing; a reservation bounds only
//! encoded input once submitted, not caller/tokenizer scratch. See `DriverConfig`.
//!
//! ```no_run
//! # async fn example(engine: ribn::Engine) -> Result<(), Box<dyn std::error::Error>> {
//! use ribn::{TokenRequest, GenerationOptions};
//! use ribn::driver::{Driver, DriverConfig};
//! let (mut owner, handle) = Driver::spawn(engine, DriverConfig::default())?;
//! let peer = handle.clone();
//! let mut first = handle.stream(TokenRequest::new(vec![1], GenerationOptions::default())).await?;
//! let second = peer.stream(TokenRequest::new(vec![2], GenerationOptions::default())).await?;
//! first.cancel(); // does not require an admission permit
//! while let Some(event) = first.next().await { let _ = event?; }
//! drop(second); // abandon delivery; the worker retains device retirement
//! owner.shutdown_async().await?;
//! # Ok(()) }
//! ```

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use flume::{Receiver, Sender};

use crate::{Engine, EngineError, Event, RequestId, TokenRequest};

#[cfg(test)]
mod tests;
mod worker;

/// Bounds for the owned token driver, independent of physical model storage.
#[derive(Clone, Copy, Debug)]
pub struct DriverConfig {
    /// Preparation permits, queued requests and retained streams together.
    pub max_requests: usize,
    /// Per-permit encoded token and stop-list storage, including the runtime copy.
    /// The aggregate payload bound is this value times `max_requests`.
    pub max_input_bytes: usize,
    /// Per-stream events, in addition to the runtime's own bounded mailboxes.
    pub events_per_request: usize,
    /// Fallback for pending device work and legacy deferred admission, not idle spin.
    pub poll_interval: Duration,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            max_requests: 32,
            max_input_bytes: 256 * 1024,
            events_per_request: 8,
            poll_interval: Duration::from_micros(250),
        }
    }
}

impl DriverConfig {
    /// Bounds compatible with `engine`, so a frontend does not have to restate
    /// the runtime's capacity to stay spawnable.
    ///
    /// Permits fit the engine's resident request capacity, and the runtime's
    /// per-request mailbox limit times the permit count fits its global event
    /// budget. The per-stream channel matches that mailbox limit, so a permitted
    /// response is not backpressured merely by driving it through this facade.
    #[must_use]
    pub fn for_engine(engine: &Engine) -> Self {
        let runtime = engine.config();
        let resident = runtime
            .max_active_requests
            .saturating_add(runtime.max_queued_requests);
        let permits = (runtime.max_buffered_events / runtime.max_events_per_request)
            .clamp(1, resident.max(1));
        Self {
            max_requests: permits,
            events_per_request: runtime.max_events_per_request,
            ..Self::default()
        }
    }
}

/// Submission and owner failures retain their runtime error source.
#[derive(Clone, Debug)]
pub enum DriverError {
    InvalidConfig,
    Overloaded,
    InputTooLarge,
    Closed,
    Enqueue(EngineError),
    Owner(EngineError),
    WorkerPanicked,
    Spawn(Arc<std::io::Error>),
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => {
                f.write_str("driver requires an idle open engine and compatible nonzero bounds")
            }
            Self::Overloaded => f.write_str("all request permits are retained"),
            Self::InputTooLarge => {
                f.write_str("encoded request exceeds its retained-byte envelope")
            }
            Self::Closed => f.write_str("execution owner is shut down"),
            Self::Enqueue(error) => write!(f, "request enqueue rejected: {error}"),
            Self::Owner(error) => write!(f, "execution owner failed: {error}"),
            Self::WorkerPanicked => f.write_str("execution worker unwound"),
            Self::Spawn(error) => write!(f, "cannot start execution worker: {error}"),
        }
    }
}

impl std::error::Error for DriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Enqueue(error) | Self::Owner(error) => Some(error),
            Self::Spawn(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

struct Shared {
    failure: Mutex<Option<DriverError>>,
    exit: Mutex<Option<Result<(), DriverError>>>,
    final_stop: AtomicBool,
    wake: Sender<()>,
    #[cfg(test)]
    before_wait: Mutex<Option<(Sender<()>, Receiver<()>)>>,
}

impl Shared {
    fn notify(&self) {
        // Full means a persistent notification is already queued. Disconnection
        // means the owner exited. Neither requires an auxiliary retry list.
        let _ = self.wake.try_send(());
    }

    fn stop(&self, error: DriverError) {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert(error);
        self.notify();
    }

    fn check_open(&self) -> Result<(), DriverError> {
        match &*self
            .failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn error(&self) -> DriverError {
        self.check_open()
            .err()
            .unwrap_or(DriverError::WorkerPanicked)
    }

    fn submit(
        &self,
        sender: &Sender<Submission>,
        submission: Submission,
    ) -> Result<(), DriverError> {
        // Serialize the final open check and queue insertion against worker exit.
        // Flume disconnection alone does not destroy messages retained by senders.
        let failure = self
            .failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(error) = &*failure {
            return Err(error.clone());
        }
        sender
            .try_send(submission)
            .map_err(|_| DriverError::WorkerPanicked)
    }

    fn exit_result(&self) -> Result<(), DriverError> {
        self.exit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .unwrap_or(Err(DriverError::WorkerPanicked))
    }
}

struct Lease(Sender<()>);

impl Drop for Lease {
    fn drop(&mut self) {
        // Exactly one token was removed for this lease. No cloning the lease.
        let _ = self.0.try_send(());
    }
}

struct Interest {
    _lease: Lease,
    cancel: AtomicBool,
    discard: AtomicBool,
}

struct Submission {
    input: TokenRequest,
    interest: Arc<Interest>,
    events: Sender<Event>,
    accepted: Sender<Result<RequestId, DriverError>>,
}

type ShutdownReply = Sender<Result<(), DriverError>>;

/// Starts a worker owning the same `Engine` used by direct embedding.
pub struct Driver;

impl Driver {
    /// Consume an idle engine and start one execution worker.
    ///
    /// Runtime global output capacity must cover each permitted request's mailbox
    /// limit, preventing one stalled stream from monopolizing peer credits.
    ///
    /// # Errors
    /// Invalid bounds/state or OS thread creation failure. The supplied engine is
    /// dropped on error, using its normal completion-safe teardown.
    ///
    /// # Panics
    /// Only if the initial permit-channel capacity invariant is broken.
    pub fn spawn(
        engine: Engine,
        config: DriverConfig,
    ) -> Result<(DriverOwner, GenerationHandle), DriverError> {
        let runtime = engine.config();
        let status = engine.status();
        let requests = runtime
            .max_active_requests
            .checked_add(runtime.max_queued_requests);
        let events = runtime
            .max_events_per_request
            .checked_mul(config.max_requests);
        if config.max_requests == 0
            || config.max_input_bytes == 0
            || config.events_per_request == 0
            || config.poll_interval.is_zero()
            || config
                .max_input_bytes
                .checked_mul(config.max_requests)
                .is_none()
            || config
                .events_per_request
                .checked_mul(config.max_requests)
                .and_then(|count| count.checked_add(runtime.max_buffered_events))
                .is_none()
            || requests.is_none_or(|count| config.max_requests > count)
            || events.is_none_or(|count| count > runtime.max_buffered_events)
            || status.requests != 0
            || status.active_sequences != 0
            || status.buffered_events != 0
            || status.in_flight
            || status.closed
            || status.faulted
        {
            return Err(DriverError::InvalidConfig);
        }
        let (wake, wake_rx) = flume::bounded(1);
        let shared = Arc::new(Shared {
            failure: Mutex::new(None),
            exit: Mutex::new(None),
            final_stop: AtomicBool::new(false),
            wake,
            #[cfg(test)]
            before_wait: Mutex::new(None),
        });
        let (submissions, submission_rx) = flume::bounded(config.max_requests);
        let (permits_tx, permits) = flume::bounded(config.max_requests);
        for _ in 0..config.max_requests {
            permits_tx.try_send(()).expect("initial permit capacity");
        }
        let (shutdown, shutdown_rx) = flume::bounded(1);
        let (done_tx, done) = flume::bounded(1);
        let worker_shared = shared.clone();
        // Completion is reported by Exit after engine teardown, not by detaching
        // a fallible task and assuming that disappearance means success.
        std::thread::Builder::new()
            .name("ribn-execution".into())
            .spawn(move || {
                worker::run(
                    engine,
                    config,
                    worker_shared,
                    &submission_rx,
                    &shutdown_rx,
                    &wake_rx,
                    done_tx,
                );
            })
            .map_err(|error| DriverError::Spawn(Arc::new(error)))?;
        Ok((
            DriverOwner {
                shared: shared.clone(),
                shutdown,
                done,
                finished: false,
                pending: None,
            },
            GenerationHandle {
                shared,
                submissions,
                permits,
                permits_tx,
                config,
            },
        ))
    }
}

/// Unique shutdown/retry owner. Drop asks for final cleanup without joining.
/// Call an explicit shutdown method to observe synchronization/release failures.
pub struct DriverOwner {
    shared: Arc<Shared>,
    shutdown: Sender<ShutdownReply>,
    done: Receiver<()>,
    finished: bool,
    pending: Option<Receiver<Result<(), DriverError>>>,
}

impl DriverOwner {
    /// Whether `handle` came from the same [`Driver::spawn`] call. A frontend uses
    /// this to reject a pair assembled from two different workers, which would
    /// otherwise shut down one worker while another served requests.
    #[must_use]
    pub fn owns(&self, handle: &GenerationHandle) -> bool {
        Arc::ptr_eq(&self.shared, &handle.shared)
    }

    /// Close admission, abandon delivery and establish device completion.
    /// Blocks the calling thread; do not call from an async executor task.
    ///
    /// # Errors
    /// Cleanup failure retains the same worker for another shutdown attempt.
    pub fn shutdown(&mut self) -> Result<(), DriverError> {
        if self.finished {
            return self.shared.exit_result();
        }
        self.start_shutdown();
        if let Some(rx) = &self.pending {
            let result = rx.recv();
            self.pending = None;
            if let Ok(result) = result {
                result?;
            }
        }
        self.finish_blocking()
    }

    fn start_shutdown(&mut self) {
        self.shared.stop(DriverError::Closed);
        if self.pending.is_some() {
            return;
        }
        // An interrupted shutdown keeps its response receiver in this owner.
        // Never queue an unbounded series of abandoned shutdown attempts.
        let exit = self
            .shared
            .exit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if exit.is_some() {
            return;
        }
        let (tx, rx) = flume::bounded(1);
        if self.shutdown.try_send(tx).is_ok() {
            self.pending = Some(rx);
        }
        drop(exit);
        self.shared.notify();
    }

    fn finish_blocking(&mut self) -> Result<(), DriverError> {
        let _ = self.done.recv();
        self.finished = true;
        self.shared.exit_result()
    }

    /// Async counterpart of `shutdown`, independent of the caller's async runtime.
    /// Cancelling this future never reopens admission or loses cleanup ownership.
    /// The next call observes that same attempt before starting another retry.
    ///
    /// # Errors
    /// Cleanup failure retains the same worker for another shutdown attempt.
    pub async fn shutdown_async(&mut self) -> Result<(), DriverError> {
        if self.finished {
            return self.shared.exit_result();
        }
        self.start_shutdown();
        if let Some(rx) = &self.pending {
            let result = rx.recv_async().await;
            self.pending = None;
            if let Ok(result) = result {
                result?;
            }
        }
        let _ = self.done.recv_async().await;
        self.finished = true;
        self.shared.exit_result()
    }
}

impl Drop for DriverOwner {
    fn drop(&mut self) {
        self.shared.stop(DriverError::Closed);
        self.shared.final_stop.store(true, Ordering::SeqCst);
        self.shared.notify();
    }
}

/// Cloneable token submission handle. Holds no mutable model state.
#[derive(Clone)]
pub struct GenerationHandle {
    shared: Arc<Shared>,
    submissions: Sender<Submission>,
    permits: Receiver<()>,
    permits_tx: Sender<()>,
    config: DriverConfig,
}

impl GenerationHandle {
    /// Request permits this handle can hold at once. A frontend uses it to bound
    /// its own admission window and preprocessing queue.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.config.max_requests
    }

    /// Reserve before expensive preprocessing. Does not wait or queue producers.
    ///
    /// # Errors
    /// Returns overload if all permits are retained, or the owner's stopped state.
    pub fn try_reserve(&self) -> Result<RequestPermit, DriverError> {
        self.shared.check_open()?;
        self.permits
            .try_recv()
            .map_err(|_| DriverError::Overloaded)?;
        Ok(RequestPermit {
            handle: self.clone(),
            lease: Lease(self.permits_tx.clone()),
        })
    }

    /// Enqueue encoded input and await its runtime request identity.
    ///
    /// # Errors
    /// Overload, input envelope violation, enqueue rejection or owner failure.
    pub async fn stream(&self, input: TokenRequest) -> Result<GenerationStream, DriverError> {
        self.try_reserve()?.stream(input).await
    }

    /// Blocking counterpart of `stream`. Not for async executor tasks.
    ///
    /// # Errors
    /// Overload, input envelope violation, enqueue rejection or owner failure.
    pub fn stream_blocking(&self, input: TokenRequest) -> Result<GenerationStream, DriverError> {
        self.try_reserve()?.stream_blocking(input)
    }
}

/// One outstanding-request lease, including preparation before token submission.
/// Dropping an unused permit returns capacity. This is not a device reservation.
pub struct RequestPermit {
    handle: GenerationHandle,
    lease: Lease,
}

type PendingStream = (GenerationStream, Receiver<Result<RequestId, DriverError>>);

impl RequestPermit {
    fn start(self, input: TokenRequest) -> Result<PendingStream, DriverError> {
        self.handle.shared.check_open()?;
        let size = input
            .tokens
            .len()
            .checked_add(input.options.stop_tokens.capacity())
            .and_then(|n| n.checked_add(input.options.stop_tokens.len()))
            .and_then(|n| n.checked_mul(size_of::<u32>()));
        if size.is_none_or(|size| size > self.handle.config.max_input_bytes) {
            return Err(DriverError::InputTooLarge);
        }
        let (events, rx) = flume::bounded(self.handle.config.events_per_request);
        let (accepted, ack) = flume::bounded(1);
        let interest = Arc::new(Interest {
            _lease: self.lease,
            cancel: AtomicBool::new(false),
            discard: AtomicBool::new(false),
        });
        let stream = GenerationStream {
            shared: self.handle.shared.clone(),
            interest: Some(interest.clone()),
            events: rx,
            request: None,
            finished: false,
        };
        // A permit covers the command as well as its subsequent lifetime, so the
        // submission channel cannot fill independently of the permit pool.
        self.handle.shared.submit(
            &self.handle.submissions,
            Submission {
                input,
                interest,
                events,
                accepted,
            },
        )?;
        self.handle.shared.notify();
        Ok((stream, ack))
    }

    /// Submit under this permit, then await runtime enqueue (not model admission).
    ///
    /// # Errors
    /// Input envelope violation, enqueue rejection or owner failure.
    pub async fn stream(self, input: TokenRequest) -> Result<GenerationStream, DriverError> {
        let (mut stream, ack) = self.start(input)?;
        stream.request = Some(
            ack.recv_async()
                .await
                .map_err(|_| stream.shared.error())??,
        );
        Ok(stream)
    }

    /// Blocking counterpart of `stream`. Not for async executor tasks.
    ///
    /// # Errors
    /// Input envelope violation, enqueue rejection or owner failure.
    pub fn stream_blocking(self, input: TokenRequest) -> Result<GenerationStream, DriverError> {
        let (mut stream, ack) = self.start(input)?;
        stream.request = Some(ack.recv().map_err(|_| stream.shared.error())??);
        Ok(stream)
    }
}

/// Ordered, owned token stream. Drop abandons delivery without waiting for device
/// completion. Explicit cancellation preserves buffered events and terminal usage.
pub struct GenerationStream {
    shared: Arc<Shared>,
    interest: Option<Arc<Interest>>,
    events: Receiver<Event>,
    request: Option<RequestId>,
    finished: bool,
}

impl GenerationStream {
    /// Identity allocated by the runtime before this stream is returned.
    ///
    /// # Panics
    /// Only if an internal enqueue/acknowledgement invariant is broken.
    #[must_use]
    pub fn request_id(&self) -> RequestId {
        self.request.expect("enqueue acknowledged")
    }

    /// Record cancellation regardless of admission/output saturation. Cancellation
    /// may race with an already-settled terminal; it does not discard buffered events.
    pub fn cancel(&self) {
        if let Some(interest) = &self.interest {
            interest.cancel.store(true, Ordering::SeqCst);
            self.shared.notify();
        }
    }

    /// Await one event. Cancelling this future does not consume an event. A terminal
    /// event or owner error is returned once, followed permanently by `None`.
    pub async fn next(&mut self) -> Option<Result<Event, DriverError>> {
        if self.finished {
            return None;
        }
        let event = self.events.recv_async().await;
        Some(self.received(event))
    }

    /// Blocking counterpart of `next`. Not for async executor tasks.
    pub fn next_blocking(&mut self) -> Option<Result<Event, DriverError>> {
        if self.finished {
            return None;
        }
        let event = self.events.recv();
        Some(self.received(event))
    }

    fn received(&mut self, event: Result<Event, flume::RecvError>) -> Result<Event, DriverError> {
        self.shared.notify();
        let result = event.map_err(|_| self.shared.error());
        if matches!(result, Ok(Event::Finished { .. }) | Err(_)) {
            self.finished = true;
            self.interest = None;
        }
        result
    }
}

impl Drop for GenerationStream {
    fn drop(&mut self) {
        if let Some(interest) = &self.interest {
            interest.discard.store(true, Ordering::SeqCst);
            self.shared.notify();
        }
    }
}
