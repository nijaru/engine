# Architecture

Ribn is a Rust-first inference runtime and serving engine. The design is intentionally
provisional where real model, workload, or hardware evidence is still missing.
[Ground-up design](ground-up-design.md) records the current target and external
reference points; [roadmap](roadmap.md) records the next proof gates.

## Current stack

```text
CLI / future HTTP / application
              |
        ribn-text frontend
  raw prompt / chat / token IDs
  tokenizer / template / text decode
              |
          ribn runtime
 request slots / batching / output
              |
      GenerationExecutor
              |
    Qwen execution + NVIDIA
```

These are practical ownership boundaries, not a requirement that every future model
or feature fit a fixed number of crates or traits.

| Package | Current responsibility |
| --- | --- |
| `ribn` (`crates/runtime`) | Token-generation lifecycle, admission, scheduling, completion, bounded output, cancellation, cleanup ownership |
| `ribn-text` (`crates/text`) | Shared text input/output behavior: raw prompts, chat formatting, token IDs, tokenization, incremental decoding, generation/batch results |
| `engine-qwen` | Qwen configuration, GGUF mapping, and the current CUDA generation executor |
| `engine-gguf` | Generic GGUF metadata/tensor reader and embedded tokenizer support |
| `engine-nvidia` | NVIDIA resources, kernels, physical state, and submission mechanics |
| `engine-core` | Legacy execution/state/runtime contracts retained temporarily for comparison and migration |
| `ribn-cli` | `inspect`, experimental `run`, and the legacy `local` correctness path |

Production dependency direction is checked by `tools/check-boundaries.py`. The goal
is to prevent accidental reverse coupling, not to forbid shared code when a real
implementation demonstrates that it belongs at a lower layer.

## Application-facing behavior

The public workflow should look like an inference engine, not like its scheduler.
The shared text frontend currently establishes three distinct inputs:

- **raw prompt** -> tokenize without a chat template;
- **chat messages** -> apply the artifact's supported chat template, then tokenize;
- **token IDs** -> submit directly without text preprocessing.

`TextModel` provides synchronous `generate`, `chat`, `generate_tokens`, `stream`,
and offline `generate_batch` operations over the same runtime. The CLI now uses
this frontend instead of reimplementing Qwen loading and streaming itself. The
current stream borrows the model mutably, so it is not yet the concurrent public
handle needed by an HTTP server or multiple independent Rust callers.

Generation completion carries prompt/completion token accounting with explicitly
documented semantics. Protocol-specific usage, finish-reason, tool, logprob, and
structured-output behavior should be mapped and tested above the scheduler rather
than inferred from similarly named HTTP fields.

The intended CLI remains conventional: local model execution (`ribn run`), model
inspection, and later serving/benchmark/device commands where implemented. Useful
execution controls should be exposed when supported. Internal queues, output-credit
bookkeeping, and model-specific state shapes should not become mandatory application
concepts, but lower-level execution APIs can remain available for integrations that
need them.

## Runtime invariants

`RequestId` is user-visible request identity. `SequenceId` identifies executor-owned
continuation state. Neither is a cache key or proof that continuation contents exist.

The committed prefix counts model inputs consumed. Output progress is separate:
final prefill may produce the first output without consuming it, and speculative
execution may compute more work than it commits. Completion is validated for the
whole submitted batch before any public prefix/output mutation.

Cancellation records intent. In-flight device state is not freed until completion
or a proven synchronization barrier. Failed release retains an owner for retry;
uncertain teardown prefers retaining device-visible resources over premature free.
Those are correctness constraints, not performance-policy choices.

Output has aggregate and per-request bounds. Ready mailboxes use stable internal
slots, and a stalled consumer can be skipped while peers with output/model capacity
continue. The runtime currently keeps one scheduler batch in flight and is driven
by polling. Wakeups, CPU/GPU scheduler overlap, and multiple in-flight batches are
future performance work that needs lifetime and cancellation tests.

## Model, artifact, and backend boundaries

Model geometry and artifact representation are different concerns. `QwenConfig`
does not require GGUF; `QwenGguf` maps GGUF metadata/tensors into Qwen-specific
semantics. The current Qwen CUDA implementation is still specialized and must
reject unsupported geometry rather than treating a parsed config as proof that the
kernels support it.

The scheduler should know enough about resource availability to make good choices,
but it should not own physical KV/GDN/MoE layouts. The current `Ready`/`Deferred`
admission contract is enough for the fixed-reservation path, not a final paging or
preemption API. When dynamic state allocation lands, scheduling and the resource
manager should cooperate around real capacity, reuse, and materialization costs.

Qwen3.8's hybrid continuation state makes a KV-only cache abstraction insufficient.
Any future prefix reuse must establish that *all* required continuation components
match the same prefix boundary. How those components are physically paged or
checkpointed is an implementation decision to validate against the real model.

## Current limitations

The new Qwen/runtime integration remains experimental pending GPU qualification.
The CUDA executor still has fixed-shape assumptions, full-context state reservation,
and limited sampling. There is no HTTP server, concurrent high-level application
handle, prefix cache, dynamic paging/preemption, automatic execution-variant
selection, or second production model/backend path yet.

CUDA Rust remains the NVIDIA kernel direction, but kernel migration and runtime
migration are separate proof gates. See [CUDA Rust migration](cuda-rust-migration.md).
