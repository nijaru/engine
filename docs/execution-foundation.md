# Shared execution foundation

Date: 2026-09-12
Status: design-validation boundary; APIs are provisional

Ribn is an inference engine. The lower-level compute and model infrastructure should,
however, avoid unnecessary inference-only assumptions so that other execution systems
could reuse it later. A future trainer is the main pressure test, not a current Ribn
feature.

The intended split is:

```text
                    model definitions / artifacts
                              |
                    shared model semantics
                 parameters / operators / topology
                              |
                shared execution foundation
     devices / buffers / views / streams / events
     operator implementations / kernel selection
     collectives / communication / topology
     checkpoint + parameter I/O
     parameter identity + versioning
     placement / sharding primitives
                              |
             +----------------+----------------+
             |                                 |
       Ribn inference                    future trainer
             |                                 |
    AR / batch / iterative              autograd / backward
    / session runtimes                  gradients / optimizer
    continuation state                  activation lifetime
    batching / serving                  training scheduler
```

This is not a commitment to build a general tensor framework, compiler, autograd
system, or training runtime. It is a dependency and ownership boundary.

## Invariants

1. Inference-specific policy does not leak below the shared foundation. A device
   buffer, parameter materialization, collective, or kernel must not require
   `RequestId`, KV cache, token scheduling, HTTP, or serving concepts.
2. Training-specific policy does not leak into Ribn inference. Shared tensors and
   parameters do not acquire gradients, autograd tape, optimizer state, or loss
   semantics merely to preserve future optionality.
3. Logical model/parameter identity is separate from physical representation.
4. An execution uses a coherent parameter version. State or prepared execution
   that depends on parameters must not silently survive an incompatible update.
5. Model semantics are separate from deployment topology. The same logical model
   may be prepared for one local device, several local devices, or remote nodes.
6. Backend-specific physical layouts and kernels remain free to specialize.
7. Artifact representation is separate from both model semantics and prepared
   execution. A checkpoint reader may expose names/shapes/bytes without deciding
   what a tensor means or how it is materialized for a backend.

## Logical parameters and materializations

A parameter is not a permanent device pointer or one checkpoint tensor. Conceptually:

```text
logical parameter
  identity
  logical shape
        |
        +-- materialization A: version 7 / bf16 / host / dense
        +-- materialization B: version 7 / q4 packed / cuda:0
        +-- materialization C: version 8 / fp8 / cuda shards 0..7
```

The current `ribn-foundation` prototype therefore separates `ParameterId` and
`ParameterVersion` from `ParameterMaterialization`. A materialization records scalar
type, storage encoding, layout identity, and one or more physical storage/device
parts. The exact types are deliberately incomplete; their job is to pressure-test
the distinction before real quantization, sharding, hot-weight, adapter, and
multi-backend paths determine the final contract.

A future trainer might own FP32 master parameters, optimizer shards, and gradient
state while an inference plan owns quantized or otherwise transformed serving
materializations. Sharing logical identity does not require sharing physical layout.

The SafeTensors/HF package pressure tests reinforce this split: an artifact exposes
parameter names as bytes plus format metadata, the model integration decides each
name's semantic role and logical parameter identity, and execution preparation later
decides the backend-specific materialization. The test-only BERT integration now
exercises this with an actual encoder architecture rather than only a toy embedding
fixture.

## Parameter versioning

The useful invariant is not "loaded weights are immutable forever." It is:

> Work that can affect observable results or reusable state executes against a
> coherent parameter version.

This matters for ordinary inference as well as RL/post-training:

- hot weight replacement;
- trainer-to-rollout synchronization;
- LoRA or adapter changes;
- prefix/continuation cache validity;
- recurrent checkpoints;
- prepared encoder/media state;
- captured or compiled execution variants;
- multi-rank deployments.

The non-AR batch validation runtime pins queued work to the executor's parameter
version and rejects execution after an uncoordinated version change. That is
intentionally strict validation behavior, not yet the final hot-update protocol.
A production design may drain, version-partition, double-buffer, or otherwise
coordinate transitions.

Derived state must carry equivalent compatibility. Sequential encoder->decoder tests
prove in-process handoff identity and cancellation ownership, but a production
encoder-state cache must also reject reuse across incompatible model/parameter/
adapter or processor versions.

## Artifact and package boundaries

Two validation layers exist below model execution:

- `ribn-safetensors` validates SafeTensors bytes and exposes tensor names, shape,
  dtype, and borrowed payload bytes. It does not create `ParameterMaterialization`
  values or assign model meaning.
- `ribn-hf` resolves a local HF-style `config.json` and either a single
  `model.safetensors` or a sharded `model.safetensors.index.json`. It preserves raw
  config metadata and maps parameter names to shard files without choosing a model
  architecture, runtime, processor, backend, or serving operation.

The BERT pressure test now loads a small actual BERT architecture from this boundary
and executes embeddings, multi-head self-attention, residual/LayerNorm, feed-forward
and pooler semantics through `ribn-batch`. That is stronger evidence that format and
package code can stay below model meaning. It is still a reference implementation,
not production BERT support or a general model registry.

It also exposed a concrete loading concern that toy fixtures could hide: repeatedly
calling a per-parameter helper that reopens a SafeTensors shard would reread the same
checkpoint bytes many times. The BERT loader therefore keeps a private artifact cache
and opens each unique shard once. A reusable package/weight-set owner should preserve
that property for real integrations, while model-specific tensor names, expected
shapes and semantic mapping remain above it.

Remote repository IDs/revisions, tokenizer/processor metadata, prepared backend
storage and the general architecture resolver remain future work.

SafeTensors also exposed one useful future-proofing detail: its dtype enum is
non-exhaustive. The adapter therefore preserves unknown future dtypes by name rather
than assuming Ribn's current scalar-type list is complete.

## Resource topology and execution plans

`ResourceTopology` is a small validation representation of nodes, compute devices,
and optional device links. It is not a cluster scheduler. `ExecutionPlan` currently
maps logical stage IDs to one or more devices plus a runtime-class identity and
parameter version.

The important property under test is:

```text
logical model semantics
        +
available resource topology
        +
execution policy
        -> prepared execution plan
```

A local one-device plan should collapse to a direct runtime/device path with no RPC,
serialization, or artificial worker hierarchy. Distributed placement, collectives,
and state transfer are additive when the prepared plan actually needs them.
External systems may allocate machines/devices; Ribn owns inference-local execution
inside those resources.

The current stage representation is intentionally weak and opaque. Do not stabilize
a universal stage graph. Sequential encoder-decoder execution can justify a real
stage boundary, while VLM/omni execution may need tightly coupled scheduling of
encoder items and AR progress inside one composite runtime. See
`pipeline-composition.md`.

## Specialized runtimes above the foundation

The first two runtime families are deliberately different:

- `ribn` (`crates/runtime`) is the existing autoregressive token runtime. Tokens,
  prefill/decode, continuation state, speculation, and AR scheduling belong here.
- `ribn-batch` (`crates/batch`) is a minimal non-AR pressure test. Its executor owns
  input/output types, tensor shapes, padding/ragged layout, device buffers, and can
  shorten the oldest FIFO candidate set when concrete shape/memory/compute
  constraints make the entire candidate batch unsuitable.

The early variable-length reference encoder showed that request count alone is not
enough to form a safe batch. The resulting `select_batch` hook deliberately exposes
no universal cost unit: the executor sees its own inputs and returns a shorter FIFO
prefix. Reordering, length bucketing, heterogeneous batching, and shared cost
metadata remain unresolved until real workloads demonstrate that they belong in the
common runtime.

The BERT architecture pressure test is the first substantially real model-semantic
use of this path. It preserves the same executor-defined request/result boundary and
uses sequence length to constrain batching while running actual BERT attention and
feed-forward structure. No new universal batching abstraction was required. The
next useful pressure comes from attention masks, padding/ragged layouts and real
device memory/compute admission rather than another synthetic cost type.

`ribn-batch` is not yet an embedding API or the final encoder scheduler. In
particular, cancellation, asynchronous device execution, resource admission,
per-request failures, padding/mask policy, and optimized device batching remain
unfinished.

Further runtime families should be introduced only when real execution regimes
justify them. Iterative diffusion/flow work and full-duplex sessions are known
pressure tests, not fixed enum variants.

## Cross-runtime request and state identity

Sequential encoder->decoder validation found one useful AR seam without introducing
a universal pipeline payload: `GenerationExecutor::admit` receives both `RequestId`
and `SequenceId`.

They have different roles:

- `RequestId` is stable for one request within the AR runtime and can correlate
  model-owned prepared inputs, tracing, or other request-scoped state;
- `SequenceId` is the executor continuation identity used for physical/state
  ownership after admission.

Tests run a batch encoder, keep its output in an `Arc`, install prepared states for
two AR requests in reverse order, and prove each decoder request consumes the
correct allocation without serialization or data copying. A missing prepared state
fails only that request.

Cancellation establishes the ownership transition without another generic cleanup
interface. Before AR admission the producer/orchestrator still owns prepared state;
after successful admission the executor's sequence state owns it and the ordinary
`release` path reclaims it after cancellation/completion is established.

This proves a useful **sequential handoff** mechanism. It does not mean every
multimodal model should put an opaque prepared-input handle into `TokenRequest`, nor
that AR `RequestId` is a universal top-level pipeline identity. Top-level
application/orchestrator identity may remain distinct from each specialized runtime's
internal request identity.

A separate VLM pressure test now covers the coupled case at the scheduling boundary.
It models model-prepared feature identity, prompt span, cache readiness, encoder
compute cost and encoder-cache cost. The tests show that prompt progress can cross a
cached feature with no encoder compute, can schedule an encoder item in the same
iteration that consumes its prompt span, must stop before an uncached item when
compute is unavailable, and can fail independently under encoder-cache pressure.
This is evidence that coupled generation needs scheduler/resource cooperation and
that encoder compute and cache are distinct concerns. It is **not** evidence for an
arbitrary resource vector or a finalized production dependency descriptor. A real
VLM integration should determine that seam.

## Operators, kernels, and future training

A small semantic-operation layer remains an experiment. If it proves useful, model
code can describe an operation while preparation chooses an implementation for the
actual backend/hardware. Forward implementations may be shared by inference and a
future trainer; backward kernels would be training-side additions.

The current test chooses a specialized or reference RMSNorm implementation once at
preparation, then executes through a prepared function pointer without repeating
support/registry lookup. This is only evidence for the dispatch pattern; it is not a
production operator registry or IR.

Do not require every optimized kernel to be reusable. Paged decode attention,
quantized decode kernels, fused optimizers, and backward attention naturally serve
different execution systems. Share operation semantics and backend infrastructure
where useful; specialize hot paths freely.

Likewise, device communication primitives and collectives may be shared, while
inference and training use different higher-level parallel strategies.

## Current validation status

Implemented as provisional scaffolding:

- dependency-free `ribn-foundation` parameter/version/materialization metadata;
- node/device/link resource topology;
- a prepared execution-plan representation that can place the same logical model
  locally or across nodes;
- test-only preparation-time semantic RMSNorm implementation selection;
- `ribn-batch`, a non-AR batching runtime with no token/prefix/KV concepts;
- coherent parameter-version checks for queued non-AR work;
- executor-informed variable-length batch sizing without a universal work unit;
- SafeTensors format-level validation/borrowed tensor access;
- local HF-style config plus single/sharded SafeTensors package resolution;
- an actual BERT architecture reference path over that package boundary, including
  embeddings, self-attention, residual/LayerNorm, FFN and pooler execution;
- evidence that real model loading needs each SafeTensors shard owned/opened once
  for repeated tensor access rather than reread per parameter;
- stable AR `RequestId` passed separately from `SequenceId` into executor admission;
- sequential encoder->AR handoff, out-of-order correlation, request-local failure,
  and cancellation ownership before/after admission;
- VLM prompt-position dependency scheduling with separate encoder compute/cache
  pressure as a test-only model of the coupled case.

This validates that the broad boundary is implementable and has forced several
interface changes. It does **not** validate that these exact types are sufficient or
optimal. The next high-value pressure tests are masked/padded/ragged encoder/device
execution, ordinary architecture resolution using the concrete Qwen+BERT evidence,
a genuine encoder-decoder model, a real VLM/processor integration, an iterative
non-AR runtime, more real semantic-op/backend implementations, and a second hardware
backend.
