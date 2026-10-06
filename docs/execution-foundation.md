# Execution foundation: evidence and limits

`ribn-foundation` owns shared byte-pool accounting, readiness notifications, parameter
version labels and artifact scalar metadata. It contains no request, token, KV,
autograd or optimizer policy. Physical storage remains backend-owned.

[Target design](inference-engine-design.md) owns architecture,
[resource protocol](resource-protocol.md) owns execution contracts, and
[roadmap](roadmap.md) owns sequencing. This document records counterexamples that
still matter; it does not prescribe provisional APIs.

## Storage ownership is not metadata

The original parameter/materialization, resource-topology and prepared-placement
experiments described versions, layouts and devices without owning executable
storage. They had no production consumers. Those types and their self-contained
validation tests have been removed rather than promoted into a framework. Core's
unused residency/qualification labels and metadata-only file-provider/loader wrappers
were also removed; actual format readers, Qwen mapping and qualification gates remain.

Keep the distinctions they explored when a real consumer requires them:

- logical parameters are separate from encoding, quantization, placement and storage;
- a version label does not keep the weights that produced derived state alive;
- model topology is separate from deployment topology;
- future training may share storage/completion mechanisms without sharing inference
  scheduling or physical materializations.

The test-only RMSNorm registry was also removed. Selecting a function pointer in a
fixture does not qualify a production operator abstraction. Resolve supported kernels
at preparation where practical, but derive any shared operation interface from real
model/backend consumers.

## Artifact loading

`ribn-safetensors` validates bytes once and retains metadata/payload offsets;
`ribn-hf` resolves local config and shard ownership without choosing model semantics.
`engine-bert` interprets names and shapes above these adapters.

Repeated per-parameter shard reads and header parses were unnecessary loading costs.
`LocalWeightSet` instead opens shards lazily and reuses validated artifacts. Its
configurable LRU bound counts cached shards, **not peak bytes**: an incoming load and
retained artifact clones can overlap eviction. Whole-file owned bytes remain the
storage representation. A byte-aware or streaming loader still needs qualification
before large-model HF loading claims.

GGUF exposed a descriptor-lifetime problem: staging the pinned artifact once held
888 file descriptors; readers opening lazily and closing at payload exhaustion reduced
the sampled peak to 38. See [runtime evidence](../benchmarks/runtime-contract.md).
This belongs to artifact access, not model semantics.

Unknown SafeTensors scalar types are preserved by name. Recognizing a dtype does not
mean a backend can execute it. Remote HF resolution and tokenizer/processor package
loading are not implemented.

## Different execution regimes

The real AR and non-AR runtimes have different units of progress. Encoder inputs and
results are executor-defined; they do not need fake token requests or KV sequences.

Variable-length encoder fixtures showed that request count alone cannot make a safe
batch. `BatchExecutor::select_batch` chooses an executable FIFO prefix using concrete
shape/cost constraints. Padded and ragged layouts can fit different amounts of the
same work; this does not justify a universal cost vector. Impossible inputs reject
locally while later requests progress.

The real BERT device path subsequently established owning leases, completion
dependencies, partial-enqueue retirement and cancellation through `ribn-batch`.
A dequeued result keeps its charge until its owner safely releases storage. See
[encoder qualification](../benchmarks/encoder-qualification.md) for tested geometry
and failure paths. AR still uses a separate logical capacity authority; sharing one
physical pool across AR and encoder execution remains unfinished.

## Composition

[Pipeline composition](pipeline-composition.md) retains the sequential and coupled
counterexamples. They do not establish a universal stage graph, real multimodal model
support or distributed execution. The next composition gate requires actual models,
not more metadata-only fixtures.
