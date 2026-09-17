# Current implementation

Observed baseline: `0ddb914` (owned driver and text facade device-qualified on the RTX 4090). This is a code map, not a second target design.
[Inference engine design](inference-engine-design.md) owns the target;
[resource protocol](resource-protocol.md) owns contracts;
[roadmap](roadmap.md) owns implementation order and exit evidence.

## Qualified text path

```text
CLI run
  → ribn-text: GGUF tokenizer/chat formatting, bounded preprocessing pool,
    incremental UTF-8 decoding, owned driver streams, ordered bound-window batching
  → ribn::driver: one execution owner, request permits, bounded streams, shutdown
  → ribn Engine: AR scheduling, request slots, bounded output mailboxes
  → engine-qwen: QwenExecution / QwenCuda
  → legacy engine-core batch/state translation
  → engine-nvidia: physical state, CUDA dispatch and kernels
```

`TextOwner::load` selects Qwen GGUF/CUDA and assembles the facade over the owned
token driver. `TextModel` is a cloneable handle over that worker; `TextOwner` is the
single shutdown owner. CUDA assembly is the only device-dependent part:
preprocessing, decoding, batching, cancellation and error scoping are host-tested.
There is no further concurrent text handle and no HTTP server.

The runtime keeps one submission in flight, reserves output credits and validates
all completion rows before logical commitment. `RequestId` and `SequenceId` identify
different owners. Cancellation of in-flight work records intent; failed cleanup
retains an owner. Executor errors conservatively fault all live requests, not just
the submitted batch.

Output mailboxes can outlive execution slots. `Engine::discard` owns abandonment
across those lifetimes; the text facade has no separate cleanup list. Discard suppresses
delivery without driving the runtime or establishing device completion. Completion
returns positive `StepCompletion` rows, including shortened prefill ranges; the
incomplete completion-time `Blocked` API was removed. Validation borrows all rows
before moving them into commitment and retains batch-buffer capacity. Deferred
admission retains a readiness registration on its waiting request; the engine only
retries after the resource owner publishes a change. The driver observes these
registrations with its bounded timed fallback, not resource notifications.

Qwen admission reserves continuation for the request's prompt plus output budget,
not the full configured context. Its logical state authority publishes readiness only
after successful release. Admission checks the whole hybrid bundle before reserving
components so a failed attempt cannot wake itself through rollback. The adapter translates
AR batch records into `engine-core` execution and state-manager types. It is not an
additional scheduler, but it retains legacy coupling and per-step allocation.

## Owned text facade

A request reserves a token permit before preprocessing, is prepared on a fixed,
bounded worker pool, and reaches the runtime as encoded input. Per-token decoding and
UTF-8 handling happen in the consuming stream, so a decode error abandons only its
own request. Text terminals distinguish ordinary completion — including explicit
cancellation, which flushes an incomplete trailing code point once — from owner
failure, which preserves delivered events and reports one owner error without
fabricating a terminal flush or usage. Offline batching pulls lazily through a bounded
admission window and yields ordered per-item outcomes, so there is no batch-wide
error variant and a stalled earliest item bounds lookahead instead of admitting
replacements.

The old facade's whole-input batch preparation, borrowed `&mut TextModel` streaming,
`TextError(String)` source erasure and CUDA-gated facade tests are removed. Text
bounds are `ProcessorLimits` (input bytes, rendered bytes, prompt tokens, decoded
token bytes); the driver keeps its own encoded-input envelope.

Stop IDs come from the artifact: declared EOS IDs for every request, plus each control
token a chat prompt itself used, because those are that conversation's turn delimiters.
Control tokens decode to no bytes while user-defined content markup stays visible, and
`encode_rendered` scans the vocabulary's control and user-defined spellings instead of a
hardcoded marker list. See the [chat stop policy](../benchmarks/runtime-contract.md#chat-stop-policy-2026-09-14).

Boundaries and rationale: [resource contract](resource-protocol.md#text-application-facade).
Host evidence: [runtime contract](../benchmarks/runtime-contract.md#owned-text-facade-host-gate-2026-09-14).
The CUDA-backed lifecycle, cancellation and multi-request determinism tests pass on the
RTX 4090 serially; see the [runtime contract
evidence](../benchmarks/runtime-contract.md#owned-text-facade-host-gate-2026-09-14).

## Owned token access

`Driver::spawn` consumes an idle engine. `DriverOwner` controls shutdown/retry;
cloneable `GenerationHandle`s obtain fail-fast request permits and return owned
`GenerationStream`s. Flume bounded channels support blocking and runtime-independent
async clients. One worker alone mutates the engine; direct embedding remains valid.

Permits count preparation, execution and retained delivery, not just execution slots.
A private sequence lifetime guard retains the permit through failed/in-flight
retirement after abandonment. Stream Drop records intent and wakes the owner; it does
not join or maintain a cleanup retry list. Shutdown failure leaves the same worker
available for retry. Idle/output-blocked workers park; device/deferred-admission polling
uses a timed fallback. The [resource contract](resource-protocol.md#owned-ar-driver-contract)
owns bounds, wakeup ordering, error scope and terminal semantics.

The driver has host lifecycle coverage and passes its Qwen CUDA runtime gate on the
RTX 4090, including the stalled and abandoned peer test. It still does not preprocess
raw input, decode text or supply bounded offline text batching; the text facade above it
does. See [qualification](../benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad).

## Slice-2 boundary audit (2026-09-14, `84a146a`)

Scope: token driver → text processing/delivery → CLI. This is a source-traced
assessment, not renewed GPU qualification or a review of kernel correctness. The
findings below are now **resolved** by the owned text facade; the table is retained
as the evidence trail for why the facade was replaced rather than patched.

Preserve the single worker, runtime-owned discard, retirement-held admission charge,
nonblocking delivery and retryable shutdown. `driver.rs` and `driver/worker.rs`
keep these responsibilities separate without another scheduler. The remaining
application gaps are not a reason to replace that owner.

| Priority / finding | Evidence and consequence | Bounded correction |
| --- | --- | --- |
| High: preprocessing precedes bounds | `text/src/model.rs::stream` calls `encode_input` before enqueue; `gguf/src/tokenizer.rs::encode` allocates per-byte BPE strings, and `render_chat` renders into a complete string. Template fuel is not a declared byte envelope. Token admission cannot bound this earlier work. | Reserve before preprocessing; bound accepted raw storage, rendered bytes, encoded output and processor scratch/concurrency. Enforce limits during growth, not only after allocation. |
| High: bounded batching conflicts with current failure semantics | `TextModel::generate_batch` collects the whole iterator and prepares every input before enqueue. `an_unpreparable_batch_input_fails_before_submitting_anything` explicitly requires all-input preparation before execution. An arbitrary incremental iterator cannot retain that atomicity and also use bounded preparation. | Replace whole-batch atomic preparation with the proposed per-item ordered contract in the target design; migrate the test deliberately. |
| High: decoder failure is batch-wide | `collect_batch` propagates `decode_bytes`/UTF-8 errors with `?`; `generate_batch` then discards every member. Runtime admission rejection already remains per-member, so failure scope differs by layer. | Return a typed per-item decode failure and abandon only that item's token stream; prove healthy peers continue. |
| Medium: text lifecycle and diagnostics remain coupled to loading | `text/src/lib.rs` CUDA-gates all of `model`; `TextError::from_display` erases sources. The two local unit tests exercise only `Utf8Decoder`; actual facade lifecycle tests load CUDA. | Separate CUDA assembly from the real facade and inject processor/executor behavior; preserve typed source chains and test failure/drop/shutdown at that surface. |
| Medium: decoded delivery has no explicit byte policy | `decode_bytes` allocates a vector based on vocabulary spelling; `Utf8Decoder::push` copies into pending storage and an owned string. The token driver's fixed-size events do not bound this added storage. | Validate a model-specific maximum decoded token size or use bounded decoding, and account for retained text deltas and terminal staging. |

Resolution: the facade now reserves before preprocessing, prepares on a bounded pool,
bounds input/rendered/prompt/decoded payloads, settles decode failures per request,
distinguishes ordinary terminals from owner failure, and replaces eager batching with
ordered bounded-window iteration. `TextStream` is owned rather than borrowed.

Other application boundaries (including subsequent CLI ingestion repair):

- CLI `input.rs` now bounds file/stdin ingestion before loading: it reads at most the
  processor input allowance plus one overflow-probe byte and rejects excess or invalid
  UTF-8. Chat reserves room for the user role within that same allowance. Argument
  storage remains caller-owned; this is a payload bound, not an allocator/RSS bound.
- `TextResponse.text` and `generate_batch` results are caller-collected storage and are
  deliberately outside buffered-application accounting, exactly like the driver's
  documented collect limit.
- Quantitative processor peak memory was not measured; the bounds are enforced, not
  profiled.

Audit verification at `84a146a`: dependency-boundary and whitespace checks passed and
all 19 driver unit tests passed; `cargo test -p ribn-text` succeeded with **zero tests**,
which is what confirmed the default-feature facade coverage gap. Kernel/backend
internals, other runtime families and device gates were not audited.

## Packages

| Package/path | Current responsibility |
| --- | --- |
| `ribn`, `crates/runtime` | AR lifecycle, scheduling, output mailboxes, executor contract and owned token driver |
| `ribn-text`, `crates/text` | Shared text preprocessing/decoding, cloneable owned facade, ordered bounded-window batching; only CUDA assembly is device-gated |
| `engine-qwen`, `crates/qwen` | Qwen configuration, GGUF interpretation, execution adapter |
| `engine-nvidia`, `crates/nvidia` | CUDA storage/state, kernels and physical execution, including the encoder primitives |
| `engine-bert`, `crates/bert` | BERT encoder semantics: configuration, parameter mapping, host shape acceptance and device byte envelope, device forward, per-request completion, the batching runtime's executor binding, and the quarantine that keeps storage and its charge together until a drain proves completion |
| `engine-core`, `crates/core` | Legacy runtime, batch/state, weight and device contracts still consumed by production code |
| `engine-gguf`, `crates/gguf` | GGUF metadata/tensor access and tokenizer support |
| `ribn-safetensors`, `crates/safetensors` | Validated artifact tensor views, no model semantics |
| `ribn-hf`, `crates/hf` | Local config and shard resolution/cache, no architecture selection |
| `ribn-foundation`, `crates/foundation` | Provisional parameter/materialization/topology metadata plus the shared byte-pool authority: owning leases, allocation identity and a release epoch for capacity readiness |
| `ribn-batch`, `crates/batch` | Non-AR batching runtime with executor-owned shape constraints, reserving retained output through a shared byte pool instead of a private budget, publishing a pool-epoch capacity registration for callers that park, handing each reservation to the executor that materializes its storage, and delivering cancellation through the ordinary terminal queue |
| `ribn-cli`, `crates/cli` | `inspect`, experimental `run`, legacy `local` comparison frontend |

`tools/check-boundaries.py` checks production dependency direction. Update the checker
when an accepted migration changes that direction; do not freeze transitional crates.

## Evidence and limits

[Execution foundation](execution-foundation.md) retains BERT/HF, topology, artifact
and parameter-version experiments. [Pipeline composition](pipeline-composition.md)
retains sequential handoff and coupled prompt-position counterexamples. These show
which assumptions fail; they do not prove production model coverage or a universal
foundation API.

Device numerical/lifecycle evidence belongs in
[the runtime contract](../benchmarks/runtime-contract.md) and
[Qwen prefill qualification](../benchmarks/qwen-prefill-qualification.md).
Compilation alone does not qualify CUDA execution. The opt-in GDN scan remains
numerically unqualified for promotion.

A second model family now executes on the device: `engine-bert` runs a BERT-style
encoder with masked attention and a pooler, numerically qualified against an
independent Hugging Face reference for its fixture geometry
([evidence](../benchmarks/encoder-qualification.md)). It also runs through
`ribn-batch`: the pool bounds how many requests execute, a device-resident result keeps
its charge after it leaves the runtime, and permanent request-local infeasibility is
rejected while peers progress, a lagging consumer holds its own charge without blocking
a peer, and cancelling a request either runs nothing (while it waits) or returns its
result and charge to the encoder. A submission that fails after reaching the device keeps
its storage and the charge covering it in the encoder's quarantine until a drain proves
completion, and one that never reached the device releases everything. A device that is
simply slow cannot be scheduled deterministically at this geometry, so delayed completion
is qualified by the properties that matter — nothing waits for completion before handing a
result over, and a result is only read after its consumer awaits the dependency — rather
than by timing a stalled kernel.

Not implemented: general architecture resolution, real media processors, cancellation
for asynchronous non-AR device ownership, dynamic hybrid continuation allocation,
executable weight replacement, a second hardware backend or distributed execution.
Metadata that accepts several devices is not distributed execution support.
