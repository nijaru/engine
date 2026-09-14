# Current implementation

Observed baseline: `7005fad` (owned driver host-qualified; device gate pending). This is a code map, not a second target design.
[Inference engine design](inference-engine-design.md) owns the target;
[resource protocol](resource-protocol.md) owns contracts;
[roadmap](roadmap.md) owns implementation order and exit evidence.

## Qualified text path

```text
CLI run
  → ribn-text: GGUF tokenizer/chat formatting, synchronous borrowed stream
  → ribn Engine: AR scheduling, request slots, bounded output mailboxes
  → engine-qwen: QwenExecution / QwenCuda
  → legacy engine-core batch/state translation
  → engine-nvidia: physical state, CUDA dispatch and kernels
```

`TextModel::load` selects Qwen GGUF/CUDA directly. `TextStream` borrows the model
mutably. There is no concurrent text/model handle or HTTP server. The new token-level
`ribn::driver` provides a separate owned access surface over the same engine; it does
not wrap the text facade's execution loop.

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
before moving them into commitment and retains batch-buffer capacity. Genuine
resource parking/reactivation remains unimplemented.

Qwen admission reserves full configured continuation state. The adapter translates
AR batch records into `engine-core` execution and state-manager types. It is not an
additional scheduler, but it retains legacy coupling and per-step allocation.

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

The driver has host lifecycle coverage and a compiled but unrun Qwen CUDA test; it is
not yet GPU-qualified. It does not preprocess raw input, decode text or supply bounded
offline text batching. See [qualification](../benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad).

## Slice-2 boundary audit (2026-09-14, `84a146a`)

Scope: token driver → text processing/delivery → CLI. This is a source-traced
assessment, not renewed GPU qualification or a review of kernel correctness.

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

Current stream completion flushes an incomplete code point as a replacement delta
before `Finished`; batch completion appends that replacement directly. Neither path
has the owned driver's buffered-events-then-owner-error behavior yet. The replacement
must specify that distinction rather than treating owner failure as normal completion.

CLI `run.rs::read_input` also reads a whole file/stdin before loading or admission.
It is caller-owned input today, outside the token-driver bound; any future claim of
bounded CLI ingestion must add a limited read at this boundary. CLI write failure
already drops the borrowed stream and attempts explicit model shutdown; preserve that
cleanup behavior during cutover.

Audit verification: dependency-boundary and whitespace checks passed; all 19 driver
unit tests passed again. `cargo test -p ribn-text --locked` succeeded with **zero tests**,
confirming the default-feature facade coverage gap. Full workspace/clippy and device
gates were not rerun for this documentation-only assessment. Kernel/backend internals,
other runtime families and quantitative processor peak memory were not audited here.

## Packages

| Package/path | Current responsibility |
| --- | --- |
| `ribn`, `crates/runtime` | AR lifecycle, scheduling, output mailboxes, executor contract and owned token driver |
| `ribn-text`, `crates/text` | Shared text processing and current synchronous facade; model module CUDA-gated |
| `engine-qwen`, `crates/qwen` | Qwen configuration, GGUF interpretation, execution adapter |
| `engine-nvidia`, `crates/nvidia` | CUDA storage/state, kernels and physical execution |
| `engine-core`, `crates/core` | Legacy runtime, batch/state, weight and device contracts still consumed by production code |
| `engine-gguf`, `crates/gguf` | GGUF metadata/tensor access and tokenizer support |
| `ribn-safetensors`, `crates/safetensors` | Validated artifact tensor views, no model semantics |
| `ribn-hf`, `crates/hf` | Local config and shard resolution/cache, no architecture selection |
| `ribn-foundation`, `crates/foundation` | Provisional parameter/materialization/topology metadata, not owning device infrastructure |
| `ribn-batch`, `crates/batch` | Non-AR batching experiment with executor-owned shape constraints and retained-result bounds |
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

Not implemented: general architecture resolution, real media processors, asynchronous
non-AR device ownership, dynamic hybrid continuation allocation, executable weight
replacement, a second hardware backend or distributed execution. Metadata that
accepts several devices is not distributed execution support.
