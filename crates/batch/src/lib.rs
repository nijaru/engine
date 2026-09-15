//! Non-autoregressive dynamic-batch runtime used to pressure-test Ribn's common boundaries.
//!
//! Inputs and outputs are executor-defined. The runtime has no token, prefix, KV,
//! decode, or chat semantics. Its batching policy is intentionally minimal until
//! real encoder/pooling models provide shape and memory evidence.
//!
//! The runtime accounts for two different quantities. Waiting work is bounded by
//! [`BatchConfig::max_waiting_requests`]. Terminal results the caller has not
//! consumed are bounded by [`BatchConfig::max_retained_results`] and by a shared
//! [`BytePool`], because a caller that stops draining results would otherwise grow
//! memory without limit, and because encoder or media outputs can own substantial
//! host and device allocations.
//!
//! The pool is the single byte authority for every runtime drawing on it, so one
//! constraint binds siblings instead of each runtime enforcing a private byte
//! budget. Reserving a request's retained output yields an owning [`PoolLease`]
//! that travels with the entry, so a completed result keeps its charge until the
//! caller takes responsibility for the storage it covers.
//!
//! Admission has three outcomes rather than an integer. The executor reports that
//! the oldest candidates are [`BatchSelection::Ready`], temporarily
//! [`BatchSelection::Blocked`], or permanently [`BatchSelection::Rejected`] under
//! its own limits. Temporary contention is not a runtime invariant failure, and
//! every blocking outcome names the condition that must change before
//! [`BatchRuntime::step`] can progress, so a caller never has to retry blindly.
//!
//! Retained capacity is reserved before a batch executes and released when the
//! caller consumes the result, so an executor cannot overshoot the budget by
//! running first and accounting afterwards. When the pool cannot cover the whole
//! selected batch, the runtime runs the largest prefix it can reserve instead of
//! blocking: batching is an optimization, so a shorter batch produces the same
//! results. `step` reports [`BlockReason::RetainedResults`] or
//! [`BlockReason::Pool`] only when not even one request fits, which is the caller's
//! signal to consume retained entries or wait for capacity to come back.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_foundation::{AllocationId, BytePool, ParameterVersion, PoolLease, ReserveError};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Bounds for waiting work and for terminal results that are still retained.
///
/// The count bound applies to every terminal entry. The byte bound is not here: it
/// belongs to the shared [`BytePool`] the runtime reserves from, so sibling
/// runtimes cannot each spend the same capacity. Rejections hold no bulk payload,
/// so they are bounded by [`Self::max_retained_results`] alone and stay
/// deliverable while the pool is exhausted.
///
/// Both counts are deliberately placeholder defaults. A deployment sizes them from
/// its own admission policy, not from the physical pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchConfig {
    /// Requests that may wait for execution.
    pub max_waiting_requests: usize,
    /// Terminal entries that may be retained before the caller consumes them.
    pub max_retained_results: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_waiting_requests: 1024,
            max_retained_results: 1024,
        }
    }
}

pub struct Job<I> {
    request: RequestId,
    input: I,
}

impl<I> Job<I> {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn input(&self) -> &I {
        &self.input
    }

    #[must_use]
    pub fn into_input(self) -> I {
        self.input
    }
}

pub struct JobOutput<O> {
    request: RequestId,
    output: O,
}

impl<O> JobOutput<O> {
    #[must_use]
    pub fn new(request: RequestId, output: O) -> Self {
        Self { request, output }
    }

    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn output(&self) -> &O {
        &self.output
    }

    #[must_use]
    pub fn into_output(self) -> O {
        self.output
    }
}

/// What the oldest compatible candidates should do next.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchSelection<C> {
    /// Execute the oldest `items` candidates.
    ///
    /// The runtime reserves [`BatchExecutor::retained_bytes`] from the shared pool
    /// for every selected input before it runs them, so an executor that
    /// under-reports its retention is what breaks the bound, not the runtime.
    Ready { items: usize },
    /// The oldest candidates are valid but cannot run yet.
    ///
    /// The queue is left untouched. This is temporary contention, not an
    /// executor invariant failure, and it must not be reported as an error.
    Blocked(C),
    /// The oldest candidate cannot execute under this executor's limits, and no
    /// amount of waiting changes that.
    ///
    /// The runtime fails only that request and keeps later work moving. FIFO
    /// order is preserved, so only the queue head can be rejected.
    Rejected(C),
}

/// Why the runtime stopped short of executing work.
///
/// Every variant names the condition that must change before
/// [`BatchRuntime::step`] can make progress.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockReason<C> {
    /// Not even the head request fits the retained-result bound, so the caller
    /// must consume retained results before more work can run.
    RetainedResults,
    /// The shared pool cannot cover the head request while other leases hold
    /// capacity. The caller must consume retained entries, or wait for a sibling
    /// runtime to release its own, before more work can run.
    ///
    /// Read [`BatchRuntime::pool`]'s epoch when this is reported and retry after it
    /// changes instead of polling capacity blindly.
    Pool {
        /// Bytes the head request needs reserved before it can run.
        requested: u64,
        /// Bytes the pool had available when the reservation failed.
        available: u64,
    },
    /// The executor reports that its own resources are not available yet. The
    /// caller that owns those resources decides what releases them.
    Executor(C),
}

/// Why a request was refused before any work was submitted for it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rejection<C> {
    /// The executor can never run this request under its own limits.
    Executor(C),
    /// This request's own retained output can never fit the shared pool, so
    /// releasing capacity can never admit it. Permanent request-local
    /// infeasibility, not backpressure.
    RetainedOutputTooLarge {
        /// Bytes the request needs reserved.
        requested: u64,
        /// Total bytes the pool can ever grant.
        capacity: u64,
    },
}

/// How one request ended.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Terminal<O, C> {
    /// The request executed. Its output stays retained until consumed.
    Output(O),
    /// The request could not execute.
    Rejected(Rejection<C>),
}

/// One terminal entry awaiting consumption, in completion order.
pub struct Completed<O, C> {
    request: RequestId,
    parameter_version: ParameterVersion,
    outcome: Terminal<O, C>,
    lease: Option<PoolLease>,
}

impl<O, C> Completed<O, C> {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn parameter_version(&self) -> ParameterVersion {
        self.parameter_version
    }

    #[must_use]
    pub const fn outcome(&self) -> &Terminal<O, C> {
        &self.outcome
    }

    #[must_use]
    pub fn into_outcome(self) -> Terminal<O, C> {
        self.outcome
    }

    /// Split into the terminal outcome and the charge covering its storage.
    ///
    /// Handing the result to another owner means handing over the lease too:
    /// dropping it releases pool bytes that owner's storage still needs.
    /// Rejections carry no lease.
    #[must_use]
    pub fn into_parts(self) -> (Terminal<O, C>, Option<PoolLease>) {
        (self.outcome, self.lease)
    }

    /// The reservation covering this entry's storage until it is consumed.
    #[must_use]
    pub const fn lease(&self) -> Option<&PoolLease> {
        self.lease.as_ref()
    }

    /// Identity of the allocation this entry holds, when it produced output.
    #[must_use]
    pub fn allocation(&self) -> Option<AllocationId> {
        self.lease.as_ref().map(PoolLease::allocation)
    }

    /// Bytes this entry retains until the caller consumes it.
    ///
    /// Rejections retain nothing and report zero.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.lease.as_ref().map_or(0, PoolLease::bytes)
    }

    /// The completed output, when this request executed rather than failing.
    #[must_use]
    pub const fn output(&self) -> Option<&O> {
        match &self.outcome {
            Terminal::Output(output) => Some(output),
            Terminal::Rejected(_) => None,
        }
    }
}

/// What one [`BatchRuntime::step`] call did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StepOutcome<C> {
    /// No queued work.
    Idle,
    /// A batch executed and its results are retained until consumed.
    Executed { results: usize },
    /// Nothing executed; the reason names the wakeup condition.
    Blocked(BlockReason<C>),
    /// The head request was rejected, and its terminal entry was delivered.
    Rejected { request: RequestId },
}

/// Coarse execution boundary for non-autoregressive request batching.
/// Concrete executors own tensor shapes, padding/ragged layout, device buffers,
/// and any model-specific batch preparation.
pub trait BatchExecutor {
    type Input;
    type Output;
    type Error: Error + Send + Sync + 'static;

    /// Why this executor cannot run a request, either now ([`BatchSelection::Blocked`])
    /// or under its own limits ([`BatchSelection::Rejected`]).
    type Constraint: Error + Send + Sync + 'static;

    fn parameter_version(&self) -> ParameterVersion;

    fn max_batch_items(&self) -> usize;

    /// Choose what the oldest compatible candidates should do next.
    ///
    /// `candidates` contains at most [`Self::max_batch_items`] requests, all
    /// pinned to the executor's current parameter version. Returning the full
    /// slice is the default. A model can select a shorter nonzero prefix when
    /// shapes, padding, workspace, or another concrete execution constraint makes
    /// the whole candidate set unsuitable.
    ///
    /// This provisional contract preserves FIFO order. Reordering/bucketing is
    /// intentionally deferred until a real workload demonstrates that the
    /// additional scheduling complexity belongs in this runtime.
    fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint>;

    /// Bytes this request's completed output will retain until the caller
    /// consumes it.
    ///
    /// Include every host allocation and device buffer the retained output keeps
    /// alive, not just the bytes a caller is likely to read. The runtime reserves
    /// this amount before executing the batch and releases it when the entry is
    /// consumed, so under-reporting here is what allows the retained budget to be
    /// exceeded. This is a per-request prediction rather than a post-execution
    /// measurement, which is what lets the runtime reserve capacity before it
    /// submits work.
    fn retained_bytes(&self, input: &Self::Input) -> u64;

    /// # Errors
    /// Returns the concrete executor error when the selected batch cannot execute.
    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error>;
}

struct Queued<I> {
    request: RequestId,
    input: I,
    parameter_version: ParameterVersion,
}

/// One executed batch and the reservations that cover its retained outputs.
struct Executed<O> {
    outputs: Vec<JobOutput<O>>,
    leases: Vec<PoolLease>,
}

/// What the queue head resolved to before any execution or reservation happened.
///
/// `Ready` stays separate from `Blocked` so the capacity checks that follow can
/// block on the runtime's own retained budget without confusing it with an
/// executor-reported constraint.
enum Resolved<C> {
    Idle,
    Blocked(BlockReason<C>),
    Rejected {
        request: RequestId,
    },
    /// Per-request retention reservations for the executor's selected prefix.
    Ready {
        items: usize,
    },
}

pub struct BatchRuntime<E: BatchExecutor> {
    executor: E,
    pool: Arc<BytePool>,
    config: BatchConfig,
    queue: VecDeque<Queued<E::Input>>,
    terminal: VecDeque<Completed<E::Output, E::Constraint>>,
}

impl<E: BatchExecutor> BatchRuntime<E> {
    /// Create a runtime that reserves retained output from `pool`.
    ///
    /// The pool is shared, not owned: sibling runtimes drawing on one physical
    /// pool pass the same authority, which is what makes a single constraint bind
    /// all of them.
    ///
    /// # Errors
    /// Rejects a zero bound or an executor that cannot run one item.
    pub fn new(
        executor: E,
        pool: Arc<BytePool>,
        config: BatchConfig,
    ) -> Result<Self, RuntimeError<E::Error>> {
        if config.max_waiting_requests == 0
            || config.max_retained_results == 0
            || executor.max_batch_items() == 0
        {
            return Err(RuntimeError::InvalidConfig);
        }
        Ok(Self {
            executor,
            pool,
            config,
            queue: VecDeque::with_capacity(config.max_waiting_requests),
            terminal: VecDeque::new(),
        })
    }

    /// # Errors
    /// Rejects a closed pool, a full waiting queue or identity exhaustion.
    pub fn submit(&mut self, input: E::Input) -> Result<RequestId, RuntimeError<E::Error>> {
        if self.pool.is_closed() {
            return Err(RuntimeError::PoolClosed);
        }
        if self.queue.len() >= self.config.max_waiting_requests {
            return Err(RuntimeError::WaitingCapacityExhausted);
        }
        let request = RequestId(
            NEXT_REQUEST_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| RuntimeError::IdentityExhausted)?,
        );
        self.queue.push_back(Queued {
            request,
            input,
            parameter_version: self.executor.parameter_version(),
        });
        Ok(request)
    }

    /// Resolve what to do with the queue head, then execute at most one batch.
    ///
    /// A [`StepOutcome::Blocked`] result is a wakeup contract rather than a
    /// failure: retry only after the named condition changes. Consuming retained
    /// entries with [`Self::pop_completed`] releases their reservation, which is
    /// the wakeup for [`BlockReason::RetainedResults`]. A sibling runtime's release
    /// advances the pool epoch, which is the wakeup for [`BlockReason::Pool`].
    ///
    /// # Errors
    /// Rejects a parameter-version change while queued work still targets the
    /// previous version, a malformed executor batch selection, malformed executor
    /// output, an internal queue invariant failure, or an executor failure.
    pub fn step(&mut self) -> Result<StepOutcome<E::Constraint>, RuntimeError<E::Error>> {
        let mut accepted = match self.resolve_head()? {
            Resolved::Idle => return Ok(StepOutcome::Idle),
            Resolved::Blocked(reason) => return Ok(StepOutcome::Blocked(reason)),
            Resolved::Rejected { request } => return Ok(StepOutcome::Rejected { request }),
            Resolved::Ready { items } => items,
        };

        // A rejection occupies the retained-result budget until it is consumed,
        // exactly like an output, so the count bound applies before anything runs.
        let free_slots = self
            .config
            .max_retained_results
            .saturating_sub(self.terminal.len());
        if free_slots == 0 {
            return Ok(StepOutcome::Blocked(BlockReason::RetainedResults));
        }
        accepted = accepted.min(free_slots);

        // Reserve every retained output before submitting work, so a full pool
        // stops execution instead of being discovered after the fact. Batching is
        // an optimization, so the accepted range shrinks to the largest prefix the
        // pool grants rather than blocking on the whole selection.
        let mut leases = Vec::with_capacity(accepted);
        for queued in self.queue.iter().take(accepted) {
            let bytes = self.executor.retained_bytes(&queued.input);
            match self.pool.reserve(bytes) {
                Ok(lease) => leases.push(lease),
                Err(ReserveError::Exhausted {
                    requested,
                    available,
                }) if leases.is_empty() => {
                    return Ok(StepOutcome::Blocked(BlockReason::Pool {
                        requested,
                        available,
                    }));
                }
                // A later oversized item stops the prefix here and becomes the
                // queue head's own rejection once earlier work drains.
                Err(ReserveError::Exhausted { .. } | ReserveError::TooLarge { .. }) => break,
                Err(ReserveError::Closed) => return Err(RuntimeError::PoolClosed),
            }
        }
        if leases.is_empty() {
            return Err(RuntimeError::QueueInvariant);
        }

        let items = leases.len();
        let version = self.executor.parameter_version();
        // Submission may reach the device before failing, so the reservations
        // travel into the error: nothing releases pool bytes covering state the
        // device may still access until completion is known.
        let executed = self.execute_selected(items, leases)?;
        let results = executed.outputs.len();
        self.terminal
            .extend(
                executed
                    .outputs
                    .into_iter()
                    .zip(executed.leases)
                    .map(|(output, lease)| Completed {
                        request: output.request(),
                        parameter_version: version,
                        outcome: Terminal::Output(output.into_output()),
                        lease: Some(lease),
                    }),
            );
        Ok(StepOutcome::Executed { results })
    }

    /// Resolve the queue head without executing anything.
    fn resolve_head(&mut self) -> Result<Resolved<E::Constraint>, RuntimeError<E::Error>> {
        let version = self.executor.parameter_version();
        let head_retention = match self.queue.front() {
            None => return Ok(Resolved::Idle),
            Some(front) => {
                if front.parameter_version != version {
                    return Err(RuntimeError::ParameterVersionChanged {
                        queued: front.parameter_version,
                        current: version,
                    });
                }
                self.executor.retained_bytes(&front.input)
            }
        };
        // Releasing capacity cannot change the pool's total capacity, so a head
        // whose own output can never fit is rejected rather than parked.
        if head_retention > self.pool.capacity() {
            return self.reject_head(Rejection::RetainedOutputTooLarge {
                requested: head_retention,
                capacity: self.pool.capacity(),
            });
        }

        let candidates = self
            .queue
            .iter()
            .take(self.executor.max_batch_items())
            .take_while(|queued| queued.parameter_version == version)
            .map(|queued| &queued.input)
            .collect::<Vec<_>>();
        match self.executor.select_batch(&candidates) {
            BatchSelection::Blocked(constraint) => {
                Ok(Resolved::Blocked(BlockReason::Executor(constraint)))
            }
            BatchSelection::Rejected(constraint) => {
                self.reject_head(Rejection::Executor(constraint))
            }
            BatchSelection::Ready { items } => {
                if items == 0 || items > candidates.len() {
                    return Err(RuntimeError::MalformedBatchSelection {
                        selected: items,
                        candidates: candidates.len(),
                    });
                }
                Ok(Resolved::Ready { items })
            }
        }
    }

    /// Deliver the head's rejection now, or block while no terminal slot is free.
    fn reject_head(
        &mut self,
        rejection: Rejection<E::Constraint>,
    ) -> Result<Resolved<E::Constraint>, RuntimeError<E::Error>> {
        if self.terminal.len() >= self.config.max_retained_results {
            return Ok(Resolved::Blocked(BlockReason::RetainedResults));
        }
        let Some(rejected) = self.queue.pop_front() else {
            return Err(RuntimeError::QueueInvariant);
        };
        self.terminal.push_back(Completed {
            request: rejected.request,
            parameter_version: rejected.parameter_version,
            outcome: Terminal::Rejected(rejection),
            lease: None,
        });
        Ok(Resolved::Rejected {
            request: rejected.request,
        })
    }

    /// Drain `items` queued requests and run them as one batch.
    ///
    /// `leases` are the reservations made for those items. A failure after the
    /// executor was called returns them to the caller instead of dropping them,
    /// because dropping would release pool bytes that may still cover device
    /// storage.
    fn execute_selected(
        &mut self,
        items: usize,
        leases: Vec<PoolLease>,
    ) -> Result<Executed<E::Output>, RuntimeError<E::Error>> {
        let mut queued = Vec::with_capacity(items);
        while queued.len() < items {
            let Some(next) = self.queue.pop_front() else {
                return Err(RuntimeError::QueueInvariant);
            };
            queued.push(next);
        }
        let expected = queued.iter().map(|item| item.request).collect::<Vec<_>>();
        let jobs = queued
            .into_iter()
            .map(|item| Job {
                request: item.request,
                input: item.input,
            })
            .collect();
        let outputs = match self.executor.execute(jobs) {
            Ok(outputs) => outputs,
            Err(source) => {
                return Err(RuntimeError::Executor {
                    requests: expected,
                    source,
                    leases,
                });
            }
        };
        if outputs.len() != expected.len()
            || outputs
                .iter()
                .zip(expected.iter())
                .any(|(output, request)| output.request() != *request)
        {
            return Err(RuntimeError::MalformedCompletion {
                requests: expected,
                leases,
            });
        }
        Ok(Executed { outputs, leases })
    }

    /// Consume the oldest terminal entry and hand over its reservation.
    ///
    /// Dropping the returned entry after popping discards the output and releases
    /// its charge. Handing the entry to another owner transfers the charge with it,
    /// because the storage it covers travels too.
    #[must_use]
    pub fn pop_completed(&mut self) -> Option<Completed<E::Output, E::Constraint>> {
        self.terminal.pop_front()
    }

    #[must_use]
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Terminal entries the caller has not consumed, which is what
    /// `max_retained_results` bounds: a rejection occupies the budget until it is
    /// consumed exactly like an output does.
    #[must_use]
    pub fn retained_results(&self) -> usize {
        self.terminal.len()
    }

    /// Bytes this runtime's retained entries currently hold in the shared pool.
    ///
    /// A sibling runtime's leases are not counted here; read [`Self::pool`] for the
    /// authority's own totals.
    #[must_use]
    pub fn retained_output_bytes(&self) -> u64 {
        self.terminal
            .iter()
            .filter_map(Completed::lease)
            .fold(0_u64, |total, lease| total.saturating_add(lease.bytes()))
    }

    /// The shared byte authority this runtime reserves from.
    #[must_use]
    pub fn pool(&self) -> &Arc<BytePool> {
        &self.pool
    }

    #[must_use]
    pub const fn config(&self) -> BatchConfig {
        self.config
    }

    #[must_use]
    pub fn executor(&self) -> &E {
        &self.executor
    }

    #[must_use]
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }
}

#[derive(Debug)]
pub enum RuntimeError<E> {
    InvalidConfig,
    WaitingCapacityExhausted,
    IdentityExhausted,
    ParameterVersionChanged {
        queued: ParameterVersion,
        current: ParameterVersion,
    },
    MalformedBatchSelection {
        selected: usize,
        candidates: usize,
    },
    QueueInvariant,
    MalformedCompletion {
        requests: Vec<RequestId>,
        /// Reservations covering state the executor may already hold. Keep them
        /// until completion is known; dropping them releases pool bytes that may
        /// still cover device storage.
        leases: Vec<PoolLease>,
    },
    Executor {
        requests: Vec<RequestId>,
        source: E,
        /// Reservations covering state the executor may already hold.
        leases: Vec<PoolLease>,
    },
    /// The shared byte pool is closed, so no new submission is accepted.
    PoolClosed,
}

impl<E: fmt::Display> fmt::Display for RuntimeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("batch runtime configuration is invalid"),
            Self::WaitingCapacityExhausted => {
                f.write_str("batch runtime waiting capacity is exhausted")
            }
            Self::IdentityExhausted => {
                f.write_str("batch runtime request identity space is exhausted")
            }
            Self::ParameterVersionChanged { queued, current } => write!(
                f,
                "queued request targets parameter version {}, but executor now exposes version {}",
                queued.get(),
                current.get()
            ),
            Self::MalformedBatchSelection {
                selected,
                candidates,
            } => write!(
                f,
                "batch executor selected {selected} requests from {candidates} candidates"
            ),
            Self::QueueInvariant => {
                f.write_str("batch runtime queue changed while draining a selected batch")
            }
            Self::MalformedCompletion { requests, .. } => write!(
                f,
                "batch executor returned malformed completion metadata for {} requests",
                requests.len()
            ),
            Self::Executor { source, .. } => source.fmt(f),
            Self::PoolClosed => f.write_str("shared byte pool is closed"),
        }
    }
}

impl<E: Error + 'static> Error for RuntimeError<E> {}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    enum Constraint {
        WorkspaceBusy,
        ExceedsTokenBudget { tokens: usize },
    }

    impl fmt::Display for Constraint {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::WorkspaceBusy => f.write_str("encoder workspace is busy"),
                Self::ExceedsTokenBudget { tokens } => {
                    write!(f, "request of {tokens} tokens exceeds the encoder budget")
                }
            }
        }
    }

    impl Error for Constraint {}

    struct Encoder {
        version: ParameterVersion,
        batches: Vec<Vec<usize>>,
        token_budget: Option<usize>,
        busy: bool,
        bytes_per_token: u64,
        over_select: bool,
        max_batch_items: usize,
    }

    impl Encoder {
        fn new() -> Self {
            Self {
                version: ParameterVersion::new(7),
                batches: Vec::new(),
                token_budget: None,
                busy: false,
                bytes_per_token: 1,
                over_select: false,
                max_batch_items: 2,
            }
        }
    }

    impl BatchExecutor for Encoder {
        type Input = Vec<u32>;
        type Output = u32;
        type Error = Infallible;
        type Constraint = Constraint;

        fn parameter_version(&self) -> ParameterVersion {
            self.version
        }

        fn max_batch_items(&self) -> usize {
            self.max_batch_items
        }

        fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
            let head = candidates
                .first()
                .expect("the runtime never selects from an empty candidate set");
            if self.over_select {
                return BatchSelection::Ready {
                    items: candidates.len() + 1,
                };
            }
            if self.token_budget.is_some_and(|budget| head.len() > budget) {
                return BatchSelection::Rejected(Constraint::ExceedsTokenBudget {
                    tokens: head.len(),
                });
            }
            if self.busy {
                return BatchSelection::Blocked(Constraint::WorkspaceBusy);
            }
            let mut items = 0_usize;
            let mut tokens = 0_usize;
            for candidate in candidates {
                let next = tokens.saturating_add(candidate.len());
                if items > 0 && self.token_budget.is_some_and(|budget| next > budget) {
                    break;
                }
                tokens = next;
                items += 1;
            }
            BatchSelection::Ready { items }
        }

        fn retained_bytes(&self, input: &Self::Input) -> u64 {
            u64::try_from(input.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(self.bytes_per_token)
        }

        fn execute(
            &mut self,
            batch: Vec<Job<Self::Input>>,
        ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
            self.batches
                .push(batch.iter().map(|job| job.input().len()).collect());
            Ok(batch
                .into_iter()
                .map(|job| {
                    let request = job.request();
                    let sum = job.into_input().into_iter().sum();
                    JobOutput::new(request, sum)
                })
                .collect())
        }
    }

    fn pool() -> Arc<BytePool> {
        BytePool::new(1 << 20).shared()
    }

    /// An executor that always fails after the runtime has reserved for it.
    struct FailingEncoder;

    impl BatchExecutor for FailingEncoder {
        type Input = Vec<u32>;
        type Output = u32;
        type Error = Constraint;
        type Constraint = Constraint;

        fn parameter_version(&self) -> ParameterVersion {
            ParameterVersion::new(1)
        }

        fn max_batch_items(&self) -> usize {
            1
        }

        fn select_batch(&self, _candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
            BatchSelection::Ready { items: 1 }
        }

        fn retained_bytes(&self, _input: &Self::Input) -> u64 {
            1
        }

        fn execute(
            &mut self,
            _batch: Vec<Job<Self::Input>>,
        ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
            Err(Constraint::WorkspaceBusy)
        }
    }

    fn config() -> BatchConfig {
        BatchConfig {
            max_waiting_requests: 8,
            max_retained_results: 4,
        }
    }

    fn runtime() -> BatchRuntime<Encoder> {
        BatchRuntime::new(Encoder::new(), pool(), config()).expect("runtime")
    }

    #[test]
    fn encoder_batches_without_token_or_sequence_semantics() {
        let mut runtime = runtime();
        let first = runtime.submit(vec![1, 2]).expect("first");
        let second = runtime.submit(vec![3, 4, 5]).expect("second");
        let third = runtime.submit(vec![6, 7]).expect("third");
        assert_eq!(
            runtime.step().expect("step"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(runtime.queued(), 1);
        assert_eq!(runtime.executor().batches, vec![vec![2, 3]]);
        let first_output = runtime.pop_completed().expect("first output");
        let second_output = runtime.pop_completed().expect("second output");
        assert_eq!(first_output.request(), first);
        assert_eq!(*first_output.output().expect("first output"), 3);
        assert_eq!(first_output.parameter_version(), ParameterVersion::new(7));
        assert_eq!(second_output.request(), second);
        assert_eq!(*second_output.output().expect("second output"), 12);
        assert_eq!(
            runtime.step().expect("second step"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(
            runtime.pop_completed().expect("third output").request(),
            third
        );
    }

    #[test]
    fn queued_work_is_pinned_to_a_parameter_version() {
        let mut runtime = runtime();
        runtime.submit(vec![1]).expect("request");
        runtime.executor_mut().version = ParameterVersion::new(8);
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::ParameterVersionChanged {
                queued,
                current
            }) if queued == ParameterVersion::new(7) && current == ParameterVersion::new(8)
        ));
    }

    #[test]
    fn retained_results_are_bounded_when_the_caller_stops_draining() {
        let mut runtime = BatchRuntime::new(
            Encoder::new(),
            pool(),
            BatchConfig {
                max_retained_results: 2,
                ..config()
            },
        )
        .expect("runtime");
        for index in 0..4_u32 {
            runtime.submit(vec![index]).expect("request");
        }
        assert_eq!(
            runtime.step().expect("first step"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(runtime.retained_results(), 2);
        assert_eq!(runtime.queued(), 2);

        assert_eq!(
            runtime.step().expect("blocked step"),
            StepOutcome::Blocked(BlockReason::RetainedResults)
        );
        assert_eq!(runtime.queued(), 2, "blocked work stays queued");
        assert_eq!(
            runtime.executor().batches.len(),
            1,
            "blocked admission must not execute"
        );

        runtime.pop_completed().expect("release one result");
        assert_eq!(runtime.retained_results(), 1);
        assert_eq!(
            runtime.step().expect("partially drained step"),
            StepOutcome::Executed { results: 1 },
            "the selected batch shrinks to the capacity that is free"
        );
        assert_eq!(runtime.retained_results(), 2);
        assert_eq!(runtime.queued(), 1);
        assert_eq!(
            runtime.step().expect("fully retained step"),
            StepOutcome::Blocked(BlockReason::RetainedResults)
        );
        runtime.pop_completed().expect("release one result");
        runtime.pop_completed().expect("release the other result");
        assert_eq!(runtime.retained_results(), 0);
        assert_eq!(
            runtime.step().expect("step after release"),
            StepOutcome::Executed { results: 1 }
        );
    }

    #[test]
    fn retained_output_bytes_are_reserved_before_execution() {
        let mut executor = Encoder::new();
        executor.bytes_per_token = 1 << 19;
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime =
            BatchRuntime::new(executor, Arc::clone(&pool), config()).expect("runtime");
        runtime.submit(vec![1]).expect("first request");
        runtime.submit(vec![2]).expect("second request");

        assert_eq!(
            runtime.step().expect("first step"),
            StepOutcome::Executed { results: 1 },
            "the executor wanted both requests, but only one reservation fits"
        );
        assert_eq!(runtime.retained_output_bytes(), 1 << 19);
        assert_eq!(pool.granted(), 1 << 19);
        assert_eq!(
            runtime.step().expect("pool-blocked step"),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 1 << 19,
                available: 0
            })
        );
        assert_eq!(runtime.executor().batches.len(), 1);

        // Consuming the entry hands the charge over rather than releasing it:
        // the caller's storage still occupies those bytes.
        let (outcome, lease) = runtime
            .pop_completed()
            .expect("released entry")
            .into_parts();
        assert!(matches!(outcome, Terminal::Output(_)));
        let lease = lease.expect("outputs carry a lease");
        assert_eq!(lease.bytes(), 1 << 19);
        assert_eq!(pool.granted(), 1 << 19);
        assert_eq!(runtime.retained_output_bytes(), 0);
        assert_eq!(
            runtime
                .step()
                .expect("step while the caller holds the charge"),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 1 << 19,
                available: 0
            }),
            "a popped result keeps its bytes reserved until its owner frees them"
        );
        drop(lease);
        assert_eq!(pool.granted(), 0);
        assert_eq!(
            runtime.step().expect("step after release"),
            StepOutcome::Executed { results: 1 }
        );
    }

    #[test]
    fn one_pool_binds_every_runtime_drawing_on_it() {
        let pool = BytePool::new(1 << 19).shared();
        let mut first_executor = Encoder::new();
        first_executor.bytes_per_token = 1 << 19;
        let mut second_executor = Encoder::new();
        second_executor.bytes_per_token = 1 << 19;
        let mut first =
            BatchRuntime::new(first_executor, Arc::clone(&pool), config()).expect("first runtime");
        let mut second = BatchRuntime::new(second_executor, Arc::clone(&pool), config())
            .expect("second runtime");
        first.submit(vec![1]).expect("first request");
        second.submit(vec![1]).expect("second request");
        assert_eq!(
            first.step().expect("first step"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(
            second.step().expect("sibling step"),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 1 << 19,
                available: 0
            }),
            "a private per-runtime budget would have admitted this work"
        );
        let epoch = pool.epoch();
        drop(first.pop_completed().expect("first entry"));
        assert!(pool.epoch() > epoch, "release advances the readiness epoch");
        assert_eq!(
            second.step().expect("sibling step after release"),
            StepOutcome::Executed { results: 1 }
        );
    }

    #[test]
    fn a_head_larger_than_the_pool_is_rejected_not_waited_out() {
        let mut executor = Encoder::new();
        executor.bytes_per_token = 1024;
        let pool = BytePool::new(1024).shared();
        let mut runtime = BatchRuntime::new(executor, pool, config()).expect("runtime");
        let oversized = runtime.submit(vec![0; 2]).expect("oversized request");
        let healthy = runtime.submit(vec![0; 1]).expect("healthy request");
        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert_eq!(
            runtime.pop_completed().expect("rejection").outcome(),
            &Terminal::Rejected(Rejection::RetainedOutputTooLarge {
                requested: 2048,
                capacity: 1024
            })
        );
        assert_eq!(
            runtime.step().expect("healthy peer still runs"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(
            runtime.pop_completed().expect("healthy output").request(),
            healthy
        );
    }

    #[test]
    fn a_closed_pool_refuses_new_work_and_a_failed_submission_keeps_its_charge() {
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime =
            BatchRuntime::new(Encoder::new(), Arc::clone(&pool), config()).expect("runtime");
        pool.close();
        assert!(matches!(
            runtime.submit(vec![1]),
            Err(RuntimeError::PoolClosed)
        ));
        drop(runtime);

        // A submission that fails after reaching the executor must not free its
        // reservation: the executor may still hold state the device can read.
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime = BatchRuntime::new(FailingEncoder, pool, config()).expect("runtime");
        runtime.submit(vec![1]).expect("request");
        let error = runtime.step().expect_err("executor failure");
        match error {
            RuntimeError::Executor { leases, .. } => {
                assert_eq!(leases.len(), 1);
                assert_eq!(leases[0].bytes(), 1);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn blocked_selection_is_not_a_runtime_failure() {
        let mut runtime = runtime();
        runtime.submit(vec![1]).expect("first request");
        runtime.submit(vec![2]).expect("second request");
        runtime.executor_mut().busy = true;

        assert_eq!(
            runtime.step().expect("blocked step"),
            StepOutcome::Blocked(BlockReason::Executor(Constraint::WorkspaceBusy))
        );
        assert_eq!(runtime.queued(), 2);
        assert!(runtime.executor().batches.is_empty());

        runtime.executor_mut().busy = false;
        assert_eq!(
            runtime.step().expect("step after wakeup"),
            StepOutcome::Executed { results: 2 }
        );
    }

    #[test]
    fn oversized_head_is_rejected_without_blocking_later_work() {
        let mut executor = Encoder::new();
        executor.token_budget = Some(2);
        let mut runtime = BatchRuntime::new(executor, pool(), config()).expect("runtime");
        let oversized = runtime.submit(vec![1, 2, 3]).expect("oversized request");
        let first = runtime.submit(vec![4]).expect("first request");
        let second = runtime.submit(vec![5]).expect("second request");

        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert_eq!(runtime.queued(), 2);
        assert!(runtime.executor().batches.is_empty());

        let entry = runtime.pop_completed().expect("rejection entry");
        assert_eq!(entry.request(), oversized);
        assert_eq!(entry.retained_bytes(), 0);
        assert!(matches!(
            entry.outcome(),
            Terminal::Rejected(Rejection::Executor(Constraint::ExceedsTokenBudget {
                tokens: 3
            }))
        ));

        assert_eq!(
            runtime.step().expect("later work still runs"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(
            runtime.pop_completed().expect("first output").request(),
            first
        );
        assert_eq!(
            runtime.pop_completed().expect("second output").request(),
            second
        );
    }

    #[test]
    fn rejection_is_delivered_while_output_bytes_are_exhausted() {
        let mut executor = Encoder::new();
        executor.token_budget = Some(2);
        executor.bytes_per_token = 1 << 19;
        let mut runtime = BatchRuntime::new(executor, pool(), config()).expect("runtime");
        runtime.submit(vec![1]).expect("fitting request");
        assert_eq!(
            runtime.step().expect("first step"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(runtime.retained_output_bytes(), 1 << 19);

        let oversized = runtime.submit(vec![1, 2, 3]).expect("oversized request");
        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert!(
            runtime
                .pop_completed()
                .expect("first entry")
                .output()
                .is_some(),
            "earlier completions are delivered first"
        );
        assert!(matches!(
            runtime.pop_completed().expect("rejection entry").outcome(),
            Terminal::Rejected(_)
        ));
    }

    #[test]
    fn retained_rejections_count_toward_the_retained_bound() {
        // A rejection is a terminal entry the caller has not consumed, so it
        // occupies the retained-result budget exactly like an output. Counting
        // only outputs lets a rejection and a later output sit in `terminal`
        // together under `max_retained_results = 1`.
        let mut executor = Encoder::new();
        executor.token_budget = Some(2);
        let mut runtime = BatchRuntime::new(
            executor,
            pool(),
            BatchConfig {
                max_retained_results: 1,
                ..config()
            },
        )
        .expect("runtime");
        let oversized = runtime.submit(vec![1, 2, 3]).expect("oversized request");
        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert_eq!(runtime.retained_results(), 1);

        runtime.submit(vec![4]).expect("later request");
        assert!(
            matches!(
                runtime.step().expect("step under a full retained budget"),
                StepOutcome::Blocked(BlockReason::RetainedResults)
            ),
            "an unconsumed rejection must block execution, not share the bound"
        );
        assert_eq!(runtime.retained_results(), 1);

        // Consuming the rejection releases the budget for the queued request.
        assert!(matches!(
            runtime.pop_completed().expect("rejection entry").outcome(),
            Terminal::Rejected(_)
        ));
        assert_eq!(
            runtime.step().expect("step after consuming the rejection"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(runtime.retained_results(), 1);
    }

    #[test]
    fn waiting_capacity_is_bounded_independently_of_retained_results() {
        let mut runtime = BatchRuntime::new(
            Encoder::new(),
            pool(),
            BatchConfig {
                max_waiting_requests: 1,
                ..config()
            },
        )
        .expect("runtime");
        runtime.submit(vec![1]).expect("first request");
        assert!(matches!(
            runtime.submit(vec![2]),
            Err(RuntimeError::WaitingCapacityExhausted)
        ));
    }

    #[test]
    fn malformed_selection_is_still_a_runtime_error() {
        let mut runtime = runtime();
        runtime.submit(vec![1]).expect("request");
        runtime.executor_mut().over_select = true;
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::MalformedBatchSelection {
                selected: 2,
                candidates: 1
            })
        ));
    }
}
