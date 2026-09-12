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
the distinction before real HF/safetensors, quantization, sharding, hot-weight, and
adapter paths determine the final contract.

A future trainer might own FP32 master parameters, optimizer shards, and gradient
state while an inference plan owns quantized or otherwise transformed serving
materializations. Sharing logical identity does not require sharing physical layout.

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
- captured or compiled execution variants;
- multi-rank deployments.

The first non-AR batch validation runtime pins a queued request to the executor's
parameter version and rejects execution after an uncoordinated version change. That
is intentionally strict validation behavior, not yet the final hot-update protocol.
A production design may drain, version-partition, double-buffer, or otherwise
coordinate transitions.

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

The current stage representation is intentionally weak and opaque. We should not
stabilize a universal stage graph before a VLM/encoder-decoder and an iterative
media pipeline demonstrate what cross-stage dependencies and payloads are actually
needed.

## Specialized runtimes above the foundation

The first two runtime families are now deliberately different:

- `ribn` (`crates/runtime`) is the existing autoregressive token runtime. Tokens,
  prefill/decode, continuation state, speculation, and AR scheduling belong here.
- `ribn-batch` (`crates/batch`) is a minimal non-AR pressure test. Its executor owns
  input/output types, tensor shapes, padding/ragged layout, device buffers, and can
  shorten the oldest FIFO candidate set when concrete shape/memory/compute
  constraints make the entire candidate batch unsuitable.

The reference encoder test uses variable-length token inputs, embedding lookup plus
mean pooling, vector outputs, and a model-specific total-token batch limit. This
immediately showed that request count alone is not enough to form a safe encoder
batch. The resulting `select_batch` hook deliberately exposes no universal cost
unit: the executor sees its own inputs and returns a shorter FIFO prefix. Reordering,
length bucketing, heterogeneous batching, and shared cost metadata remain unresolved
until a real workload demonstrates that they belong in the common runtime.

`ribn-batch` is not yet an embedding API or the final encoder scheduler. In
particular, cancellation, asynchronous device execution, resource admission,
per-request failures, and optimal batching for real model shapes are intentionally
unfinished.

Further runtime families should be introduced only when real execution regimes
justify them. Iterative diffusion/flow work and full-duplex sessions are known
pressure tests, not fixed enum variants.

## Operators, kernels, and future training

A small semantic-operation layer remains an experiment. If it proves useful, model
code can describe an operation while preparation chooses an implementation for the
actual backend/hardware. Forward implementations may be shared by inference and a
future trainer; backward kernels would be training-side additions.

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
- `ribn-batch`, a non-AR batching runtime with no token/prefix/KV concepts;
- coherent parameter-version checks for queued non-AR work;
- a tiny reference encoder/pooling path with variable-length inputs and vector
  outputs;
- executor-informed FIFO batch sizing that arose from the encoder's concrete
  batch-cost constraint rather than a generic work-unit abstraction.

This validates that the boundary is implementable and has already forced one
runtime-interface change. It does **not** validate that these exact types are
sufficient or optimal. The next pressure tests are an artifact-backed small
encoder/pooling model path, a multi-stage multimodal/encoder-decoder path, an
iterative non-AR path, semantic-op dispatch, a modern model-loading/reference path,
and a second hardware backend.
