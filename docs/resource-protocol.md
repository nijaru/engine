# Resource, submission and snapshot protocol

Status: provisional design contract. None of this is implemented. It records the
contract the runtime needs before dynamic continuation resources, speculation, or a
second non-AR runtime are built on top of the current seams, so that work does not
have to invent it piecemeal later.

This is not a universal framework. Every type below exists because a concrete
failure mode is already visible in the current code or in a pressure test, and each
one is sized to the smallest shape that fixes it.

## What the runtime does today

The AR runtime is honest about ownership and about committed progress, and the gaps
are all in the same place: physical resource demand is invisible to the code that
decides what runs next.

- `GenerationExecutor::admit` validates a request and reserves that sequence's whole
  continuation bundle. `Admission::Deferred` already distinguishes "not now" from
  "invalid".
- `BatchItem::token_budget` is the exact prefill advancement or the maximum
  *committed* decode advancement, and `output_budget` explicitly excludes
  speculative draft work. Both describe committed progress, not physical demand.
- `batch_tokens` (`BatchRuntime`) bounds waiting work and retained results, and
  admits work as ready, blocked, or rejected.
- `Engine` keeps one batch in flight, reserves output capacity before submission,
  treats one executor fault as faulting all in-flight work, and retains physical
  ownership when safe teardown cannot be established.

Those invariants are load-bearing and this document keeps all of them:

- validate a complete batch before changing logical progress;
- reserve output capacity before submission;
- cancellation is intent, not device completion;
- retain physical ownership when safe teardown cannot be established.

## 1. Prepared submission

**Problem.** A step's physical demand is not a function of its committed token
count. Verifying a speculative step needs candidate state and verification inputs;
a chunked prefill needs additional KV pages; a coupled multimodal step needs encoder
output storage plus decoder workspace. The scheduler currently compares token
budgets, so it cannot refuse work it cannot afford, and an executor that runs first
and accounts afterwards can overshoot any bound the caller set.

**Shape.** Split submission into a fallible reservation phase and an ownership
transfer, and let the executor declare what the reservation covers.

```rust
/// Bytes one prepared step has reserved from a pool the executor does not own.
pub struct PoolClaim {
    pool: PoolId,
    bytes: u64,
    /// Temporary claims are released on completion; committed claims belong to
    /// continuation state and are released with the sequence.
    kind: ClaimKind,
}

pub trait PreparedSubmission {
    /// What this prepared step already holds. The caller accounts for these
    /// before it agrees to enqueue.
    fn claims(&self) -> &[PoolClaim];

    /// Give up the reservation without executing. Every temporary claim is
    /// released; committed claims stay with the sequence.
    fn abandon(self) -> Result<(), ExecutionError>;
}

pub trait GenerationExecutor {
    /// Validate and reserve a batch without enqueuing device work. On error
    /// nothing is retained and live sequence state is unchanged.
    fn prepare(&mut self, batch: &[BatchItem])
        -> Result<Self::Prepared, ExecutionError>;

    /// Enqueue a prepared batch. Ownership of its reservations moves here with
    /// the call. A partially enqueued batch must still leave this executor owning
    /// completion for everything the device can observe.
    fn enqueue(&mut self, prepared: Self::Prepared) -> SubmissionId;
}
```

`prepare`/`enqueue` replace `submit`. The reason is not symmetry: a single call
cannot both fail cleanly *and* guarantee that ownership survives a partial enqueue.
After `prepare` returns, the only remaining decisions are the caller's accounting
and the ownership transfer, and neither can fail in a way that loses state.

**Completion reconciles every component.** `StepCompletion` reports a prefix and
sampled tokens; a hybrid sequence also carries recurrent state, convolution history,
and KV rows. Reconciliation therefore commits continuation state per component, not
per scalar position, and a reusable boundary retained after termination must be the
component-wise boundary — not simply the furthest physically computed position.

**Rollback is not a thing.** A failed or abandoned step releases only what it
reserved for that step. Anything the device may already have written stays owned
until completion is established, which is what the current fault model already
requires.

## 2. One accounting authority per shared pool

**Problem.** Independently bounded runtimes do not make a bounded pipeline. Encoder
results, AR continuation state, execution workspace and cached features can all
occupy one device, and two runtimes that each believe they hold some bytes will
happily fill the pool with upstream results while leaving no workspace for the
downstream stage that must consume them.

**Shape.** The pool has one owner that hands out reservations; runtimes state
demand in concrete bytes through their own `PoolClaim`s and never interpret each
other's.

```rust
/// The single accounting authority for one shared pool. External orchestrators
/// still decide physical allocation; this only decides who may consume it.
pub struct PoolBudget { /* capacity, outstanding reservations */ }

impl PoolBudget {
    /// Reserve `bytes`, or report the shortfall without reserving anything.
    pub fn reserve(&self, bytes: u64, kind: ClaimKind) -> Result<PoolReservation, Shortfall>;
    /// Release a reservation. Idempotent, because release retries happen.
    pub fn release(&self, reservation: PoolReservation);
}
```

The scheduler asks an executor for claims, reserves them, and only then permits
`enqueue`. A refused reservation is ordinary backpressure, not an error: the
request waits, and the runtime reports which pool is short so an operator can see
whether the bottleneck is continuation state, workspace, or an upstream backlog.

Model and backend code still calculates demand. What moves into the shared layer is
only the decision to grant or refuse it, and the aggregate view that prevents two
stages from double-spending one pool.

## 3. Completion and access ownership across runtimes

**Problem.** The sequential encoder→decoder handoff proves that admission transfers
ownership of a prepared state. It does not establish that the encoder's writes are
visible to the decoder, that the decoder is the last reader, or that its buffer may
be reused. An `Arc` says who may *access* an allocation, not who may *overwrite* it.
Cancelling after encoder submission cannot reclaim its output, because AR admission
never happened but the device may still be writing.

**Shape.** A handoff carries a completion dependency and an owned access lease
alongside the representation.

```rust
/// Proof that a producer's writes are visible to a consumer.
pub struct CompletionDependency(/* backend-owned */);

/// Exclusive access to a prepared state until it is dropped.
pub struct StateLease { /* opaque */ }

pub struct Handoff<S> {
    state: S,
    /// Which model snapshot and representation `state` belongs to.
    identity: StateIdentity,
    ready: CompletionDependency,
    lease: StateLease,
}
```

Requirements:

- a consumer must not read a state before its completion dependency is satisfied;
- a producer must not reuse or free a state while any access lease is outstanding;
- cancellation does not imply completion, so a cancelled handoff keeps its lease;
- a faulted handoff's state is quarantined with the pool that owns it, not returned
  to a free list;
- the representation stays backend-specific. This layer never learns what a tensor,
  KV block, or recurrent state is, only that some prepared state exists and who may
  touch it.

**Pinned host metadata is not a transfer buffer.** Reusable metadata can live in
pinned host memory, but an asynchronous transfer must read a buffer whose lifetime
the transfer owns. Overwriting shared metadata while a copy is in flight is a data
race that no ownership check catches today, because the buffer is borrowed rather
than owned.

## 4. Owning model snapshot

**Problem.** `ParameterVersion` labels a parameter set consistently, and the batch
runtime detects a version change instead of preserving the previous version for
queued work. A version label does not own a coherent executable snapshot, and "the
versions are equal" is not a compatibility rule.

**Shape.** An admitted request pins an immutable snapshot for its lifetime.

```rust
/// One executable model configuration: weights, adapters, processor
/// configuration, and the prepared resources they require.
pub struct ModelSnapshot { /* version, owned artifacts, prepared resources */ }

impl ModelSnapshot {
    pub fn version(&self) -> ParameterVersion;
    /// May derived state from `self` be reused for a request on `other`?
    pub fn semantic_compatibility(&self, other: &Self) -> Compatibility;
    /// May this allocation, captured graph, or prepared state be used by the
    /// backend execution prepared for `other`?
    pub fn physical_compatibility(&self, other: &Self) -> Compatibility;
}
```

A single coarse snapshot epoch is enough to start; per-parameter multiversioning is
not required. Drain-and-replace is an acceptable update policy before seamless hot
swap is worth building: new admissions pin the new snapshot, existing requests keep
the old one alive until they finish, and the old snapshot is released when its last
request does.

**Semantic and physical compatibility are different questions** and must not collapse
into one equality check:

- *Semantic*: may this prompt prefix, recurrent checkpoint, or encoder result be
  reused without changing the intended computation? If `false` for a component, that
  component is recomputed and the rest may still be reused.
- *Physical*: may this buffer, captured graph, or transferred state be used by this
  prepared backend execution? Equal logical weight versions do not make state
  interchangeable between different quantized materializations, and a captured graph
  has additional address and layout lifetime constraints.

Cache reuse keys are therefore built from snapshot identity plus materialization
identity, not from token content alone. Multi-tenant reuse additionally needs a
trust domain: two tenants may legitimately share tokens whose *state* must not be
shared.

## Acceptance tests

Each one is written to fail against the current code, which is the point.

1. **Reservation under pressure.** With a constrained pool, prepare a multi-token
   step, inject failures before and after enqueue, accept only a prefix, and verify
   the resulting continuation matches a non-speculative reference with no leaked or
   prematurely reused reservation.
2. **Consumer stall.** Stall a consumer while submissions continue, run encoder and
   decoder work against a small shared pool, and verify retained memory plateaus,
   overload is explicit (which pool, how much short), and cancellation still works.
3. **Handoff lifetime.** Run an asynchronous encoder→decoder handoff with
   cancellation at each handoff point, delayed completion, producer failure, and an
   attempted buffer reuse, and verify no reader observes incomplete writes while an
   independent request still completes.
4. **Snapshot transition.** Keep requests running across a snapshot change, alter an
   adapter or processor configuration, and verify reuse happens only for explicitly
   compatible state while old allocations stay alive until their users finish.

## What this does not change

- Models still own their state layouts, kernels, and materialization choices.
- The scheduler stays cheap: expensive planning and profiling do not move onto the
  per-step path, and the pool owner only grants or refuses reservations.
- `ribn-batch` keeps its ready/blocked/rejected admission and its waiting and
  retained bounds; it gains pool claims rather than a second admission model.
- No universal resource vector, tensor abstraction, or dynamic plugin ABI appears
  here. A `PoolClaim` is bytes plus a pool handle, and `Handoff` is generic over a
  backend-specific representation.
