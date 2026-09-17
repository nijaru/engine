# Resource, submission and snapshot protocol

Status: accepted direction. Current completion/discard behavior is identified below;
prepared resources, readiness and snapshot replacement are not implemented APIs. The 2026-09-13 review replaces
contradictory claim/reservation sketches with the ownership rules below. The
[roadmap](roadmap.md) owns implementation order and exit evidence.

## Current boundary

`GenerationExecutor` combines admission, submission and completion. Since roadmap 4a,
Qwen reserves the continuation capacity its own request can reach instead of the
model's whole context (see [AR continuation resources](#ar-continuation-resources));
it still reserves that capacity once, at admission, and executes the offered range.
`poll` returns one positive `StepCompletion` per row. `None` means submitted work
remains pending, not a settled request waiting for a resource.

The incomplete completion-time `Blocked` enum was removed: it had no readiness source
or parking and could immediately resubmit unchanged work. Positive partial-prefill
completion remains: it reports a contiguous consumed range, not permission to exceed
physical capacity and not a pre-submit reservation. Ordinary resource waiting must
arrive with a real preparation implementation and its readiness source, not another
isolated completion enum. Admission waits carry an authority-owned readiness registration. Direct callers
recheck registrations when stepping; the driver uses its bounded timed fallback because
readiness sources do not yet notify its wake channel. Device completion is also polled.

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
path whose capacity follows admitted requests—not an unbounded emergency queue. The
concrete non-AR form (2026-09-15) is intent recorded on the waiting request itself: a
cancelled request keeps its FIFO place and reports cancellation instead of executing,
so the intent occupies no additional structure and cannot be lost to a full terminal
queue. Cancelling a request whose result is already retained gives that result — and
the charge covering it — back to the executor through the same retirement path a
malformed completion uses; the runtime never releases device storage or its charge by
itself.

## Encoder preparation and prepared resources

Status: implemented for the first real encoder (2026-09-15, roadmap slices 3b.2–3b.3).
The concrete representation is deliberately not generic: `ribn-foundation`'s
`BytePool`/`PoolLease`/`AllocationId` is the authority, `ribn-batch` owns reservation,
accepted ranges, capacity registration, result retention and the enqueue-failure
contract, and `engine-bert` supplies the concrete shape decisions
(`crates/bert/src/request.rs`), its device result and its retirement list
(`crates/bert/src/executor.rs`, `crates/bert/src/cuda.rs`). Roadmap 3c qualified the failure
paths on device (constrained pool, cancellation before and after enqueue, a lagging
consumer beside a healthy peer, permanent rejection with peer progress, charges
surviving dequeue) except for one criterion whose precondition does not exist yet:
downstream workspace headroom needs a real downstream stage sharing the pool, which
slice 4 introduces. Two limits are recorded: a request's whole envelope stays charged
until its result is dropped, so completed temporary storage is not released early, and a
release that no drain can prove keeps its storage and its charge until one can.

1. **One authority, owning byte leases.** The first concrete authority is a shared byte
   pool that grants an owning, non-duplicable lease per reservation. Moving a lease
   between owners does not reserve again. Physical storage and its charge live in one
   owner, so a backend that materializes device memory for a lease keeps both together,
   and popping a result from a queue does not release its charge. A lease carries the
   allocation identity that a later derived-state compatibility check needs; a bare byte
   count is not an allocation identity.
2. **Accepted ranges.** Preparation reports the concrete work the executor accepted and
   the reservations it granted for that work. The runtime prepares an executable prefix
   of its queue; it never prepares an arbitrary iterator. Aggregate submission resources
   are budgeted across the whole prepared range, not reset per row.
3. **Waiting versus rejection.** Waiting names both a condition and a readiness source.
   For capacity the source is the pool's allocation epoch; readiness uses
   registration-and-recheck so a release between the capacity check and parking cannot
   be lost. Rejection is permanent request-local infeasibility — an indivisible input
   larger than the pool can ever grant, or an unsupported shape — and is delivered
   without waiting. A retry count never establishes impossibility. Waiting work parks
   outside runnable queues and does not stall healthy requests; while a backend provides
   no completion notification, a bounded timed fallback is explicit, not implied.
4. **Pre-submit abandonment.** Abandoning prepared, unsubmitted work releases all new
   reservations, including uncommitted persistent growth, and never releases
   pre-existing continuation. Host-only reservations may rely on ordinary RAII; fallible
   device retirement needs an executor-owned retry/quarantine path, and Drop cannot
   assert device completion.
5. **Partial enqueue.** An enqueue failure distinguishes clean rejection before device
   access from uncertain partial submission. After any device-visible mutation the
   executor retains completion ownership of everything that may have been enqueued; a
   caller is never given ownership back as though nothing happened. Concretely, a
   granted reservation travels into the submission with the request it covers, so a
   failure returns either "nothing was created, every reservation released" or "the
   executor keeps the storage and the charge". The distinction is part of the reported
   error, an enqueue error carries no reservation, and output the runtime cannot commit
   is handed back to the executor rather than released by the runtime.
6. **Producer/consumer completion ownership.** A device-resident result carries its
   representation, model identity, pool charge and a producer completion dependency. The
   consumer waits on that dependency before reading. [Cross-runtime device
   ownership](#cross-runtime-device-ownership) already owns the reuse, cancellation and
   quarantine rules; preparation adds only the lease and dependency that make them
   checkable.

Do not add public generic `PreparedSubmission`, `PoolClaim` or lease traits yet. The
first representation should be concrete enough to be exercised by a real encoder, and
generalized only when a second consumer demonstrates the same shape.

## AR continuation resources

Status: the declared-shape/concrete-capacity split and capacity backpressure are
implemented for the Qwen path (2026-09-15, roadmap 4a, device-qualified at `323f258`).
Block-table KV addressing is qualified in the backend (2026-10-12, `ee689d7`) but has
no runtime consumer: production Qwen still holds one contiguous allocation per
sequence. Paged block reuse, eviction and preemption are accepted design and remain
unimplemented; this section states the contract they must satisfy.

Continuation is the state a sequence must keep to continue: full-attention KV,
sliding-window KV, recurrent matrices and convolution history, and hybrid combinations
of them. It is owned per sequence until a cache owns it, and it is charged to the same
byte authority as every other device reservation on that device.

1. **A declaration is a shape and a bound.** A model declares each continuation
   component's shape and maximum capacity. A concrete state materializes a capacity
   within that bound, and validation accepts any concrete capacity whose shape matches
   the declaration. Reserving less is the point: capacity that a request cannot reach is
   not reserved. The bound is a hard limit, not a suggestion.
2. **Capacity is a reservation request.** Requested capacity is derived from the request
   (prompt plus output budget, or the accepted range for growth), granted by the
   authority, and charged as granted. Growing continuation is a new reservation
   transaction; it never edits the declaration and never borrows unreserved capacity.
   Insufficient capacity is ordinary backpressure and an indivisible demand larger than
   the authority can ever grant is request-local rejection, delivered without waiting.
   Admission returns `Deferred(ReadinessWait)` registered before checking the blocking
   condition. The runtime retains this registration on the waiting request, outside
   runnable queues, and re-offers it only after the source changes. The capacity authority
   publishes after a successful release, never after commit, failed release or allocation.
   A change permits retry, not allocation: a peer may already have consumed the capacity.
   A retry that still cannot fit registers again. Cancellation and shutdown remove the
   wait with its request; no executor-owned waiter list is needed. The driver rechecks
   epochs on its bounded poll interval when no device work is in flight, so a change
   before parking is not lost. This is registered readiness with timed observation, not
   notification-driven wakeup.
3. **One owner holds storage and its charge.** A backend materializes device storage for
   the lease it was granted, and the same owner releases both together. A lease is never
   duplicated to share storage: storage shared between sequences is owned once (by the
   cache or another owner) and handed out as a reference, so the charge is counted once
   and reference counting does not duplicate accounting.
4. **A sequence's continuation is a bundle with per-component validity.** A prefix is
   reusable only when every component the model requires at that boundary is present and
   valid — for a hybrid model, KV without the matching recurrent checkpoint is not a
   continuation. A partial match is not partially usable; it advances only to the last
   boundary where all required components agree.
5. **Commit reports the valid boundary.** After completion, the runtime advances the
   sequence's committed prefix to what the model actually validated, not to what was
   enqueued. Persistent growth that was charged as prepared work settles into
   continuation ownership at that boundary; temporary work returns to its pool.
6. **Release follows proven completion.** Dropping continuation is logical until the
   device may have finished reading it. A release the backend cannot prove keeps its
   storage and its charge, on the same retirement path as any other uncertain device
   work. Cancellation is intent and does not by itself release continuation.
7. **Eviction and preemption are different owners.** Cache eviction returns
   unreferenced storage to the authority; preemption takes continuation from a live
   sequence. Preemption by recomputation drops that sequence's continuation, returns the
   request to waiting with its tokens, and rebuilds state by replay. Cancellation,
   preemption and eviction must each be observable as a distinct outcome, because they
   imply different work and different accounting.

The engine owns which sequence holds which committed prefix and the policy decisions
(admission, eviction, preemption, priority). The model/backend owns layout, block or
page granularity, checkpoint materialization and kernel-facing indexing. The authority
owns bytes. None of these three charges the same reservation again.

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
  completion and registered admission waits use a configured nonzero timed-poll
  fallback. Unchanged registrations are checked without rerunning model admission;
  resource publication does not yet notify the worker's wake channel. Synchronous backend
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

## Text application facade

Status: implemented; host-qualified, with the CUDA lifecycle, cancellation and
multi-request determinism gates passing on the RTX 4090. The one
accepted deviation from the first draft is that this layer does **not** own an
architecture registry or a second worker: it owns text preprocessing, incremental
decoding and offline batching above the owned token driver. It holds no scheduler,
no second execution loop and no device state.

`TextOwner` is the non-cloneable lifecycle owner; it owns the token driver's
shutdown owner and the preprocessing pool. `TextModel` is a cloneable generation
handle over the same worker. Assembly rejects a shutdown owner and generation handle
that came from different workers, because that pair would shut down one worker while
another served requests. `TextConfig` carries preprocessing worker count and batch
window; `ProcessorLimits` carries the text byte bounds. CUDA/GGUF assembly is
`TextOwner::load`, and it is the only part that needs a device.

Stop IDs come from the artifact, never from a model-name list. Every request carries
the artifact's declared EOS IDs, and a chat request also carries each control token its
**own** rendered prompt used: those are the delimiters that conversation's template
rendered, so a model emitting one is ending or restarting a turn rather than writing
text. Grounding and vision markers are control tokens too, which is why they stop
exactly when a conversation used them and a raw prompt adds nothing. Control tokens
decode to no bytes, so an emitted marker never reaches user-visible output, while
user-defined content markup such as a thinking or tool-call tag stays visible. Every
delivered event carries its token ID, so a structured consumer loses nothing to the
text policy. Caller-supplied stop tokens are added to this set, not replaced by it.

Implementation status and host evidence: [architecture](architecture.md#owned-text-facade)
and [runtime evidence](../benchmarks/runtime-contract.md#owned-text-facade-host-gate-2026-09-14).

### Preprocessing ownership and bounds

The token driver accepts encoded input only. Text preprocessing therefore sits
above it and must be bounded independently:

- A caller acquires a request permit **before** preprocessing. The permit covers
  retained raw/message input, template rendering, encoded tokens and driver
  delivery until retirement. Preprocessing may not run before that reservation.
- Raw prompt/message content, rendered chat-template bytes, encoded prompt tokens
  and the bytes one token decodes to are separate limits. Raw content and rendered
  bytes are rejected while the payload is produced, and decoded bytes are bounded as
  they are decoded. Encoded prompt tokens are bounded by those byte bounds and by an
  explicit token limit; a caller-supplied token vector is checked before admission.
  Exceeding any limit is a request-local rejection with a typed error; it is never a
  device fault.
- The retained-input bound is checked **before** an input is queued, so aggregate
  queued retention is the permit count times that envelope rather than whatever
  callers happen to hold while preprocessing waits.
- Encoded tokens plus the runtime's stop-list copy must still fit the permit's
  encoded-input envelope. The text layer does not enlarge that envelope.
- Preprocessing runs on a fixed, bounded worker pool with a bounded queue. Jobs
  own their permit, so outstanding preprocessing cannot exceed the permit pool
  and the queue is bounded by construction. A dropped submission may let its job
  finish; the permit returns only when that work releases its input.
- Preprocessing must not occupy the execution worker or an async executor thread.
  Cancelling an async submission does not strand state: the permit is released by
  the job, not by the cancelled future.
- A pool that loses its last worker — including through a processor panic — becomes
  observably closed: queued jobs are released and later submissions are refused, so
  no caller waits on a queue nothing will serve. Once closure begins, a worker that
  is about to start a queued job drops it instead of beginning new work.
- Immutable loaded vocabulary, caller-collected results and caller-owned input
  before acceptance are outside buffered application storage. Decoded delta text
  and terminal staging are inside it and are bounded per token and per request.

### Delivery and terminal semantics

- Ordinary token events decode incrementally. A token that does not complete a
  UTF-8 code point yields an empty text delta while retaining its token id; bytes
  that cannot form valid UTF-8 are a request-local decode error.
- A normal or explicitly cancelled terminal flushes an incomplete final code
  point once as a replacement delta before the text terminal event. Flushing is
  terminal bookkeeping, not recovery.
- A decode error abandons only its own request. Peers continue, and the failing
  request yields one typed error instead of a text terminal.
- A terminated stream releases its decode scratch and keeps only its terminal
  outcome until its consumer reads it, so a retained finished stream does not
  accumulate request payload.
- Owner failure preserves events already handed to stream channels, then yields
  one owner error. It does not invent successful usage or perform an ordinary
  terminal flush, and it does not relabel healthy peers as invalid.
- Text cancellation forwards to the driver's intent flag. It does not wait for
  device completion and does not discard buffered output.

### Offline batching

- Batching consumes a caller iterator lazily through a bounded admission window.
  It never collects or prepares an arbitrary iterator before admission. Lookahead
  is bounded by the window; a stalled earliest item bounds how far later items
  advance rather than admitting replacements ahead of the ordered head.
- Results are yielded in input order. Each item settles independently: an
  unencodable input, an admission failure, a decode error or an item-local
  terminal failure affects only that item. There is no batch-wide error variant;
  the collect helper returns ordered per-item outcomes.
- Collect helpers intentionally allocate the requested result. The incremental
  iterator is the bounded interface for large workloads. Dropping either abandons
  live streams without waiting for device completion; the driver retains
  retirement ownership.
- Admission is fail-fast. An item that cannot reserve because other handles hold
  permits reports overload for that item. There is no unbounded waiter queue and
  no silent spin-retry.

### Shutdown

Closing the preprocessing pool precedes driver shutdown; no new text work starts
while cleanup runs. `TextOwner` keeps the driver's retry ownership, so a failed or
cancelled shutdown leaves the same worker available and reports the earlier fault
through the handle. Successful shutdown reports cleanup success, not recovery from
an execution fault.

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
- AR continuation: per-request capacity within the declared bound, capacity exhaustion
  as waitable backpressure rather than a fault or an unbounded deferral, reuse only at a
  boundary whose every component is valid, eviction of unreferenced storage, preemption
  by recomputation, and cancellation that leaves no continuation charge behind.
- Snapshot draining and replacement, incompatible adapter/processor state, and later
  explicit overlap accounting before supporting concurrent versions.

Host fixtures prove lifecycle transitions, not GPU memory safety or model support.
Run real device qualification for the affected execution path.
