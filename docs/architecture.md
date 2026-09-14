# Current implementation

Observed baseline: `2abf382`. This is a code map, not a second target design.
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
mutably. There is no concurrent owned application handle or HTTP server. Public
operation sketches in the target design are not executable examples.

The runtime keeps one submission in flight, reserves output credits and validates
all completion rows before logical commitment. `RequestId` and `SequenceId` identify
different owners. Cancellation of in-flight work records intent; failed cleanup
retains an owner. Executor errors conservatively fault all live requests, not just
the submitted batch.

Output mailboxes can outlive execution slots. The current text facade's discard
list does not fully cover that distinction or all error exits. Completion-time
`Blocked` requeues immediately and has no readiness protocol. Both mechanisms are
scheduled for replacement, not guarantees to preserve.

Qwen admission reserves full configured continuation state. The adapter translates
AR batch records into `engine-core` execution and state-manager types. It is not an
additional scheduler, but it retains legacy coupling and per-step allocation.

## Packages

| Package/path | Current responsibility |
| --- | --- |
| `ribn`, `crates/runtime` | AR lifecycle, scheduling, output mailboxes, executor contract |
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
