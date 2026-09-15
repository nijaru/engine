# Engineering roadmap

Status: ordered implementation and decision gates, updated 2026-09-14.

[Target design](inference-engine-design.md) owns architecture and API semantics;
[resource protocol](resource-protocol.md) owns execution ownership. This document
owns work order and unresolved decisions. Server-first inference is the priority;
training is a future execution system sharing demonstrated lower-level mechanisms.
All v0 interfaces may change. Do not preserve a broken API for compatibility.

## Engineering method

For each slice:

1. Inspect current source and relevant experiments. Research current primary sources
   for consequential uncertain choices; record revision/date and limitations.
2. State the contract in its existing owner: inputs/results, owner at each transition,
   cancellation and error scope, memory bounds, readiness and shutdown. Include the
   direct single-model path and a materially different counterexample.
3. Resolve competing designs with a small experiment when reasoning is insufficient.
   Do not mistake a fixture for model support or a prototype type for a stable API.
4. Write contract tests and implement a complete vertical slice. Replace obsolete
   paths instead of adding wrappers. Keep qualified kernels while replacing control
   plumbing unless kernel changes are independently required.
5. Run host checks and affected device gates, inspect actual user behavior and compare
   performance where the change affects it. Update current-state evidence only then.

Agents may choose local representation and algorithms within the contract. A change
in ownership, failure isolation, observable API semantics, bounds, execution order or
numerical policy requires a design/test update before dependent implementation.
Research does not need to settle every future feature before the first slice starts.
It must settle the slice's contract and show that known future requirements do not
contradict it. Record assumptions and reconsideration triggers, not absolute promises.

## Starting baseline

The historical starting point below explains the repair order. Slice entries record
later changes; [architecture](architecture.md) describes implemented surfaces.

At `2abf382`, Qwen GGUF/CUDA is the only qualified model execution path. The text
facade remains mutable and single-caller. The batch/BERT, topology and composition
paths are design experiments, not production model coverage.

- Progress completion allows partial prefill, but also incorrectly allows zero
  successful prefill. `Blocked` requeues immediately, potentially in the same step;
  there is no parked state or readiness protocol. Remove that incomplete API until
  real preparation supplies ordinary backpressure and reactivation.
- Text cleanup improved but is not complete: unknown execution identity can still
  have an output mailbox; batch/stream error paths can abandon cleanup. Move discard
  responsibility into the runtime rather than extending frontend retry lists.
- Batch terminal-count bounds were fixed with a failing-before regression. Output
  dequeue still does not establish a bound on downstream live physical allocations.
- Qwen reserves full continuation capacity and translates through legacy core types.
  Foundation version/topology metadata does not own executable storage.
- Device gates at `6f0ebbf`: lifecycle 4/4 (76.80 s), exact-token runtime (174.39 s),
  CUDA reference 61/61 (408.84 s). These qualify that revision's tested device paths,
  not missing lifecycle/readiness contracts. See [runtime evidence](../benchmarks/runtime-contract.md).

Detailed historical experiments remain in [execution foundation](execution-foundation.md)
and [pipeline composition](pipeline-composition.md). Performance/numerical evidence
belongs in [benchmarks](../benchmarks/README.md), not repeated milestone tables.

## Ordered vertical slices

### 1. Repair the execution ownership baseline

Partially implemented at `4477a0e`: runtime-owned discard replaces frontend cleanup
lists; batch and stream errors relinquish output interest. Host checks and CUDA text
lifecycle 4/4 passed. A failing-before test also pins cancellation through the legacy
blocked completion branch. See [qualification](../benchmarks/runtime-contract.md#runtime-owned-discard-qualification-2026-09-14).
At `14d0290`, the completion-time blocking API is removed and prefill advancement
must be positive. Completion validation no longer allocates/clones row plans and
retains batch-slot capacity. Zero-progress rejection has a failing-before regression;
partial-prefill runs with a two-credit output pool; mixed malformed rows, cancellation,
failed release and faulted retirement are covered. Host checks pass. Combined device
gates at `14d0290` pass: text lifecycle 4/4 (76.91 s), exact-token runtime (174.22 s),
CUDA reference 61/61 (408.13 s), each with recorded exit 0. The next design gate is
owned concurrent access. Direct host injection into the CUDA-gated text decoder and
broader frontend error-path testing remain part of that gate; current device tests
are narrow, not exhaustive frontend failure coverage.

Decided scope:

- retain positive contiguous partial-prefill completion and whole-batch validation;
- remove incomplete completion-time blocking and redundant completion-plan copies;
- runtime-owned cancel-and-discard, including terminal mailboxes after slot release;
- remove frontend cleanup queues and ensure every batch/stream error relinquishes
  request interest without releasing device-visible storage;
- distinguish request-local rejection from executor corruption. Conservative global
  faulting remains correct for unknown device state, not ordinary resource shortage.

Exit: failing-before regressions for zero progress, discarded terminal mailboxes,
late completion, mixed healthy/abandoned consumers and error cleanup; output-credit
accounting stays bounded; affected device gates pass. Do not build a synthetic
readiness framework merely to retain the old `Blocked` variant.

### 2. Owned concurrent application access

Decided contract: cloneable handle; one execution owner; owned request streams;
bounded admission and output; cancellation under saturated admission; explicit owner
shutdown/failure; one coherent executable lifetime. Direct runtime use remains valid.

First increment at `7005fad`: `ribn::driver` owns the existing token engine on one
worker, with fail-fast preparation permits, encoded-input envelopes, bounded owned
streams, cancellation/credit wakeups and explicit shutdown retry. Flume supplies
blocking and async waits without Tokio. A private runtime lifetime guard holds each
admission charge through retirement, including after stream drop. Exact ownership,
error and cancellation semantics are in the [resource contract](resource-protocol.md#owned-ar-driver-contract).

Nineteen actual-driver host tests pass, including 100 consecutive suite runs and
mutation checks for retirement charges, wakeups and queued acknowledgement teardown.
Required host checks pass. [Direct/driver host cost](../benchmarks/runtime-alignment/README.md#owned-token-driver-7005fad)
is measured; it is not model throughput. The new CUDA driver test is **unrun**:
desktop SSH remains unavailable. On 2026-09-14 Tailscale was restarted and disco
pings succeeded, but system DNS failed and TCP/22 timed out via the tailnet IP.
The cause is unverified; no GPU availability was established. Sync and run the pending
[device gate](../benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad)
before calling the driver GPU-qualified.

#### Audit alignment order (2026-09-14)

The [source-traced assessment](architecture.md#slice-2-boundary-audit-2026-09-14-84a146a)
found text-boundary gaps, not a reason to replace the token driver. Its findings are
resolved by the owned text facade; the [text contract](resource-protocol.md#text-application-facade)
is now implemented and host-qualified.

1. **Done (2026-09-14).** Desktop was reachable and idle; the checkout was pulled and
   the pinned artifact hash re-verified. Serialized gates, all exit 0: runtime 2/2
   (199.11 s, including the stalled/abandoned-peer driver test), text lifecycle 5/5
   (95.83 s), and a new multi-request determinism test 1/1 (19.52 s). The concurrency
   example ran four callers end to end. Evidence:
   [runtime contract](../benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad).
2. **Done.** The text contract landed before dependent code: a bounded preprocessing
   pool with its own shutdown owner, per-item settlement instead of whole-input
   preparation atomicity, UTF-8/terminal precedence, typed error sources, and
   growth-time byte limits for input/rendered/prompt/decoded payloads. Overload stays
   fail-fast per item with no waiter queue and no spin-retry.
3. **Done except device evidence.** The borrowed execution loop is replaced by the
   host-testable owned facade with CUDA-only assembly. Host tests cover decode failure
   with a healthy peer, cancellation, dropped streams and batches, a stalled consumer
   beside a healthy request, buffered owner-failure semantics at the driver level, and
   incomplete UTF-8 on a normal terminal. Driver-level tests retain shutdown retry
   ownership.
4. **Done.** Eager batching and its atomic-preparation test were replaced together. A
   pull-counting iterator proves bounded lookahead, and per-item failures — an invalid
   input, a decode error and saturated admission — leave peers unaffected.
5. **Partially done.** CLI cutover, explicit shutdown and host/CUDA-feature checks are
   complete, and collect-result exclusions are documented. Matched frontend overhead
   and the affected device gates still need a reachable GPU. CLI file/stdin ingestion
   remains a whole read and is not claimed as bounded.

Text facade status: `ribn-text` now provides cloneable `TextModel` handles over one
owned driver, `TextOwner` shutdown, bounded preprocessing, typed `TextError` sources,
owned `TextStream`, and ordered bounded-window `TextBatch`. Twenty-one host tests pass
with a real GGUF tokenizer, the real driver and a scripted fixture device, including
per-item overload under full permit retention, bound-checking at the retention
boundary, owner-failure delivery and a dead preprocessing pool that fails callers
instead of queueing them. Six CUDA-backed tests pass on the device, including
multi-request determinism. Evidence:
[runtime contract](../benchmarks/runtime-contract.md#owned-text-facade-host-gate-2026-09-14).

Remaining before closing this slice:

- measure matched direct-versus-handle frontend overhead on the GPU path; the device
  gates now cover correctness, not comparative cost;
- decide the chat stop-token set. Only the artifact's `eos_token_id` is added today, so
  a Qwen chat turn that ends with `<|im_end|>`-class markers can run to the token limit
  and leak marker text into output. This is pre-existing stop policy, not a slice-2
  regression, and it interacts with the reasoning controls that also remain unexposed;
- bound CLI file/stdin ingestion with a limited read before claiming it is bounded;
- thread-affine non-Send construction, if a real backend requires it, needs a separate
  factory contract;
- re-run the gates once the chat stop policy changes, since it alters termination.

Exit: device-qualified frontend behavior, saturation/race/shutdown evidence on both
paths, no orphan requests, and matched frontend overhead measurements. The loaded owner
pins one actual executable configuration; no metadata-only snapshot wrapper or
unimplemented general architecture registry counts as snapshot ownership.

### 3. Real asynchronous encoder and prepared resources

Use actual encoder device work to determine preparation and readiness representation.
Resolve concrete accepted ranges, one pool authority, waiting versus rejection,
pre-submit abandonment, partial enqueue and producer/consumer completion ownership
according to the resource protocol. Budget resources across the whole submission.

Exit: constrained shared pool, delayed completion, cancellation, failed handoff,
consumer stall and permanent oversized-input rejection with healthy peer progress.
Charges survive dequeue until safe reuse. Prove downstream workspace remains available.
Do not implement a universal cost vector or operation graph as a prerequisite.

### 4. Dynamic hybrid AR and model-local integration

Implement dynamic KV plus recurrent-state ownership, valid prefix reuse, eviction and
preemption before speculative reconciliation. Compare unified token-budget scheduling
with current queues using actual resource costs and mixed-arrival workloads.

A second real decoder and real VLM/processor must integrate with model/processor,
backend, registration and tests—not family branches in cancellation, routing or
protocols. A genuinely new mechanism may require a focused shared-contract change.
A sequential encoder-decoder and small iterative model then test composition beyond
AR without forcing every encoder into its own stage.

Exit: independent numerical references, constrained resources, continuation and
cancellation qualification; documented shared changes and their concrete necessity.
Replace legacy Qwen/core translation as its real consumers migrate. Delete unused
paths immediately; do not wait for every future runtime class to remove dead code.

### 5. Serving qualification and wider systems

Implement a documented protocol subset over the owned application API, not another
execution loop. Test streaming, errors, disconnects, overload, health/readiness,
metrics and security limits. Python in-process use follows the same semantics.

Qualify mixed lengths/arrivals, long context, stalled clients and memory pressure.
Report throughput within explicit latency objectives, latency distributions, host
cost, peak memory and cancellation latency. A benchmark needs pinned workload,
artifact, numerical policy, revisions and repeated matched measurements.

Test a materially different backend before calling device contracts general.
Implement collectives and real sharding before promoting topology metadata into a
distributed runtime. Local single-device execution must not require serialization
or synthetic worker/process layers.

### 6. Training integration, then training execution

First establish trainer-to-rollout version publication and tensor/result interchange.
Start drain-and-replace; overlapping snapshots require explicit extra memory and
compatibility evidence. Never expose optimizer-mutating storage to inference.

A later small forward/backward/update reference test determines shared operator/model
semantics. Training owns gradients, activation lifetimes, optimizers and its scheduler.
Checkpoint/restart and distributed training require their own qualification. Revisit
shared model representation if duplicated semantics becomes a measured maintenance or
correctness problem; do not impose autograd on inference preemptively.

## Independent qualification tracks and blockers

- **GDN scan:** opt-in only. Full-model gate fails `1.10e-2` versus `5.0e-3`; cause
  unestablished. Diagnose captured real state against a higher-precision recurrence.
  Existing baseline reference error grants no tolerance budget. Compare common
  histories; step 19 after step-18 token divergence is not a common-history comparison.
- **Kernel scaling:** retain qualified small-M GEMV; investigate tiled quantized GEMM,
  packed cross-request work and tiled attention. Do not repeat measured losers without
  new evidence: four rows per warp, shared IQ4 codebook, pre-elimination decayed keys.
  Compilation is not GPU qualification; CUDA Rust migration has its own
  [gate](cuda-rust-migration.md), not priority over runtime correctness.
- **Artifact loading:** GGUF lazy readers reduced sampled descriptors 888→38. HF
  shard-cache count is not a hard peak-byte limit; incoming loads and retained clones
  can overlap. Qualify streaming/byte-aware ownership before large-model claims.
- **Research:** [research agenda](research-agenda.md) supplies candidate sources and
  later topics. Only the relevant questions above block a slice; do not turn broad
  research into a second implementation backlog.

## Required checks

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

Capture actual exit codes, including when logging through pipes. CUDA-feature checks
compile otherwise gated frontend code but do not replace serialized device tests.
Follow the [model-integration skill](../.agents/skills/model-integration/SKILL.md)
for model/numerical qualification. Record failed and unrun gates explicitly.
