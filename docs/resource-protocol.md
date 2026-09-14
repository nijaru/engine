# Resource, submission and snapshot protocol

Status: accepted direction. Current completion/discard behavior is identified below;
prepared resources, readiness and snapshot replacement are not implemented APIs. The 2026-09-13 review replaces
contradictory claim/reservation sketches with the ownership rules below. The
[roadmap](roadmap.md) owns implementation order and exit evidence.

## Current boundary

At `14d0290`, `GenerationExecutor` combines admission, submission and completion.
Qwen reserves its full continuation capacity at admission and executes the offered
range. `poll` returns one positive `StepCompletion` per row. `None` means submitted
work remains pending, not a settled request waiting for a resource.

The incomplete completion-time `Blocked` enum was removed: it had no readiness source
or parking and could immediately resubmit unchanged work. Positive partial-prefill
completion remains: it reports a contiguous consumed range, not permission to exceed
physical capacity and not a pre-submit reservation. Ordinary resource waiting must
arrive with a real preparation implementation and its readiness source, not another
isolated completion enum. `Admission::Deferred` and device polling still rely on
explicit driver steps; the current synchronous facade is not event-driven.

Keep these invariants throughout replacement:

- validate every submitted row before logical commitment;
- reserve output capacity before launch;
- cancellation records intent; it never proves completion;
- malformed completion or uncertain executor state faults the execution owner,
  including its other live requests; ordinary request rejection is different;
- retain ownership of anything the device may access until completion is known.

## One authority, four ownership states

For each shared physical pool, one allocator/accountant grants reservations. The
model/backend calculates concrete demand and asks that authority during preparation.
The scheduler does not charge the same reservation again.

1. **Candidate:** scheduler policy offers work; no new allocation is owned by the
   candidate. Token counts are policy bounds, not a universal memory cost.
2. **Prepared:** the backend owns newly granted reservations and an accepted range
   for each ready request. Existing continuation remains sequence-owned. Preparation
   may shorten work, report temporary waiting, or reject an impossible request.
3. **Submitted:** the executor consumes the prepared work and owns every resource
   visible to partially or fully enqueued device work, even if enqueue fails.
4. **Settled:** after completion, temporary storage can return to its pool; accepted
   persistent growth transfers to continuation or result ownership. Logical progress
   reflects the actual valid continuation boundary, not speculative physical work.

Demand estimates are plain values. Reservations are owning, non-duplicable leases.
Moving a lease between owners does not reserve again. Physical storage and its
charge live together; popping a result from a queue does not release its charge.
Reference-counted sharing is valid where necessary, but must not duplicate accounting.

Abandoning prepared, unsubmitted work releases **all new** reservations, including
uncommitted persistent growth. It does not release pre-existing continuation. Ordinary
RAII can handle host-only reservation release; fallible device retirement needs an
explicit executor-owned retry/quarantine path. Drop cannot assert device completion.

An enqueue error must distinguish a clean rejection before device access from
uncertain partial submission. Never return ownership to the caller as though nothing
happened after device-visible mutations. A separate `prepare` method alone cannot
make enqueue infallible; the implementation must retain a concrete completion owner.

Do not add public generic `PreparedSubmission`, `PoolClaim` or lease traits until the
real encoder/backend integration establishes their necessary representation.

## Feasible batches and waiting

Preparation considers aggregate submission resources, including shared encoder work,
cache residency, workspace, continuation growth and output capacity. A per-row budget
reset is not a batch bound. Choose the accepted ranges before executing their work.

A prepared report distinguishes:

- **Ready:** positive feasible work with owned reservations;
- **Waiting:** no committed work; a concrete condition can change;
- **Rejected:** permanent request-local infeasibility, with a useful diagnostic.

One oversized indivisible item must not wait forever for a per-step budget that
cannot grow. Do not infer impossibility from an arbitrary retry count.

Waiting requests park outside runnable queues. Readiness uses a generation/epoch or
registration-and-recheck protocol so notification before parking cannot be lost.
Reactivation works with no device batch in flight. Cancellation and shutdown wake
parked owners; output consumption wakes output-blocked work. A new compute-budget
epoch differs from an external encoder dependency becoming ready.

Device completion may initially require timed event polling. A worker should sleep
on commands/readiness while idle and use bounded timed waits while polling such a
backend. Document that fallback rather than claiming fully event-driven execution.
Do not stall healthy requests merely because another request is waiting.

## Output ownership and boundedness

The execution owner owns cancellation and discarded mailboxes. A frontend dropping
a request relinquishes interest once; it does not maintain a second cleanup queue.
Discard must work after an execution slot is freed but its terminal mailbox remains.
It suppresses future delivery without releasing in-flight storage early.

Bound submission count, input bytes/tokens, retained event count/bytes and active
physical storage separately. A bounded channel containing arbitrarily large vectors
is not a memory bound. Public collect helpers intentionally allocate the requested
result; document their limit and offer incremental consumption for large workloads.

Cancellation must remain deliverable when the ordinary submission queue is full.
Use per-request cancellation intent plus a wakeup, or a separately bounded control
path whose capacity follows admitted requests—not an unbounded emergency queue.

## Owned AR driver contract

The first owned-access increment drives the existing token runtime; it is not the
loaded-model or multimodal application API. `Driver::spawn(engine, config)` consumes
an idle, open engine and returns a shutdown owner and cloneable submission handle.
Direct `Engine` use remains channel-free. One worker owns execution, including failed
retirement. Moving an engine requires its existing `GenerationExecutor: Send` contract;
this increment does not support non-Send/thread-affine executor construction.

### Admission and bounds

- Admission is fail-fast. `try_reserve` acquires one request permit or returns typed
  overload; there is no hidden, potentially unbounded queue of waiting producers.
  A frontend must acquire this permit **before** expensive preprocessing. The token
  driver itself accepts already-encoded input, not raw media or preprocessing tasks.
- Permits cover unsubmitted preparation, queued commands, execution and undelivered
  stream output. A permit returns only after runtime retirement and stream delivery
  both relinquish it. A private runtime lifetime guard pins the admission charge even
  after the worker drops a discarded route; there is no frontend retirement retry list.
  Runtime execution-slot counts are not application request counts.
- Each permit has a configured encoded-input byte envelope covering token storage and
  stop-token vector capacity plus the runtime's cloned stop list. The aggregate bound
  is request permits times this envelope. Overflow is rejected. Caller-owned inputs
  before acceptance, allocator overhead and immutable loaded model storage are not
  included. Raw-input/tokenizer scratch bounds remain a frontend obligation.
- Each stream has a bounded channel of token events. Runtime mailboxes remain bounded
  separately: at most the runtime event limit plus permit count times stream capacity
  are buffered. Runtime global event capacity must cover every permitted request's
  per-request mailbox limit, so a stalled peer cannot monopolize delivery credits.
  This conservative configuration can be revisited with shared-credit evidence.
- Tokens/usage are copied values, not shared device storage. Runtime execution-error
  diagnostic strings are bounded to 4096 UTF-8 bytes at construction; overflow ends
  with an explicit truncation marker. This bounds retained diagnostic payload, not
  backend formatting scratch. Collected caller-owned results are deliberately outside
  buffered-delivery accounting; physical output leases belong to a later real encoder.

### Delivery, wakeups and failure

Use Flume 0.12 bounded channels for both blocking and runtime-independent async waits.
One capacity-one wake channel coalesces notifications; commands and persistent atomic
cancellation/discard flags are the authoritative state. The worker drains old wake
notifications **before** inspecting state, never after checking a condition and before
sleeping. Mutators publish state before notifying. This covers arrivals, stream drops,
explicit cancel and output-credit return without a second cancellation queue.

- `stream(...).await` or its blocking counterpart returns after runtime enqueue, not
  model admission or device completion. Rejection before enqueue returns a typed error.
  Dropping the submission future relinquishes its request even across enqueue/ack races.
- An owned stream yields ordered token events and one terminal event or owner error,
  then remains exhausted. Dropping an individual `next` future loses no event.
  Explicit cancel preserves already-buffered output and requests a `Cancelled` terminal;
  it can race with an already-settled terminal. Dropping the stream suppresses delivery.
- The worker never blocks sending output. Only it sends to each stream; it checks channel
  space before draining that runtime mailbox. Consumption wakes the execution owner.
- An idle/output-blocked owner sleeps indefinitely on the wake channel. Pending device
  completion and legacy `Admission::Deferred` use a configured nonzero timed-poll
  fallback. This is not the future resource-readiness protocol. Synchronous backend
  calls remain non-preemptible; wakeups cannot interrupt a blocked driver operation.
- Request-local runtime admission failures remain `FinishReason::Failed`. Driver-level
  enqueue rejection retains its `EngineError` source. Execution-owner failure stops
  admission, preserves events already handed to stream channels, then exposes an
  owner-scoped error. Undelivered runtime-mailbox events are abandoned; no final
  request usage is promised after owner failure. It does not relabel healthy peers
  as individually invalid.

### Shutdown

The non-cloneable shutdown owner has blocking and async shutdown methods. Shutdown
closes admission first, abandons outstanding delivery and synchronizes/releases the
engine. Previously queued stream events remain readable, followed by an owner-closed
error unless a terminal was already delivered. Shutdown is not per-request cancellation.
A failed shutdown reports its source and leaves the same worker owning the engine for
explicit retry. Cancelling the shutdown future does not reopen admission or discard
that retry owner: its next shutdown call observes the same attempt's result before
another retry starts. Successful shutdown reports cleanup success, not recovery from
an earlier execution fault; handles retain that fault as their diagnostic.
Dropping the owner requests final shutdown without joining; the
worker retains resources through cleanup, with the runtime's conservative quarantine
on unresolved completion. Explicit shutdown is required for error reporting.

Worker exit closes queue insertion before explicitly draining queued acknowledgements.
Flume receiver disconnection alone does not drop queued values while senders remain;
otherwise a queued reply sender can keep a waiting client alive forever.
Worker unwind is reported distinctly, never as successful exhaustion. Defensive engine
Drop still establishes completion or retains device-visible ownership. Process abort,
OOM abort and a backend that never returns cannot promise recoverable notification.
No force-kill timeout may release device-visible memory.

## Cross-runtime device ownership

A device-resident result needs its representation, model identity, pool charge and
producer completion dependency. The consumer must wait on that dependency before
reading. Storage cannot be overwritten until all consumer device accesses finish;
an `Arc` alone proves neither completion nor exclusive reuse.

Cancellation before downstream admission leaves the producer/transport owner
responsible for retirement. Successful admission transfers that responsibility to
the consumer. Failed handoffs retain an owner. Unknown completion quarantines the
storage and its charge rather than returning it to a free list.

Async host-to-device transfers likewise own an immutable staging buffer until the
copy completes. Reusable pinned metadata is not automatically safe transfer storage.

Pool boundedness is not progress: upstream results can fill a pool and starve the
consumer workspace needed to release them. Reserve downstream headroom or admit
whole pipeline resource envelopes for the concrete pipeline; prove this with a
constrained-pool test before generalizing it.

## Owning executable snapshot

A snapshot is the owned executable configuration: prepared weights/materializations,
adapters, processor configuration and execution resources. A numeric version is not
an owner. Mutable request state stays in the runtime, not inside a shared mutable
snapshot object protected by one global lock.

Start with one loaded execution owner's lifetime pinning one snapshot. Implement
**drain-and-replace** first: stop new admission, finish/cancel old work, establish
completion, release old resources, then publish the replacement. Concurrent old/new
snapshots are a distinct double-buffered policy requiring measured extra capacity,
not another name for draining.

An accepted request uses one coherent snapshot. Derived-state reuse requires semantic
compatibility (parameters, adapters, processor, numerical policy and trust domain)
and physical compatibility (backend, encoding, layout, addresses where captured).
Do not partially reuse hybrid continuation unless all required components form a
valid boundary. Exact snapshot identity is a safe initial compatibility policy;
relax it only with evidence.

Trainer-to-rollout updates eventually prepare a serving materialization and publish
it at an explicit version boundary. Inference must never observe tensors while an
optimizer mutates them. Sharing logical parameter identity does not require training
master weights and quantized serving weights to share an allocation.

## Acceptance evidence

- Positive partial prefill, final-prefill output rules, credit refunds, malformed
  mixed rows with no logical commit, cancellation during delayed completion.
- Discard before admission, during execution, after terminal publication, on decoder
  errors and on batch failure; stalled and healthy consumers together; bounded memory.
- Multiple callers, queue saturation, cancellation under saturation, owner shutdown,
  lost-wakeup interleavings and worker failure propagation.
- A real asynchronous encoder: aggregate resource pressure, partial enqueue failure,
  abandonment of new persistent growth, consumer stalls, failed handoffs and delayed
  producer/consumer completion without premature reuse or leaked reservations.
- Snapshot draining and replacement, incompatible adapter/processor state, and later
  explicit overlap accounting before supporting concurrent versions.

Host fixtures prove lifecycle transitions, not GPU memory safety or model support.
Run real device qualification for the affected execution path.
