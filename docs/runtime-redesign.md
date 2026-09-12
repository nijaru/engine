# Model-neutral runtime redesign

Date: 2026-09-11 (America/Los_Angeles)
Status: initial implementation; CUDA integration experimental

## Decision

Make the common runtime depend on **execution and ownership behavior**, not a
closed catalogue of model layers, attention mechanisms, or physical state types.
The four working concepts are Engine, prepared model, sequence identity, and
scheduled batch. Keep Rust types inside concrete model/backend implementations.

The acceptance target is that a new text-generation architecture needs its model
implementation, artifact/backend adapters where necessary, registration or
ordinary application composition, and qualification tests—not a new enum arm
through the scheduler. This is a design test, not a promise that all possible
future task semantics fit one immutable API.

## Findings behind the change

The audit of `a4009e2c` found several independent sources of unnecessary coupling:

| Existing surface | Problem | Direction |
| --- | --- | --- |
| `StateRequirement`, `InferenceStateSet` | Core knows KV/recurrent families, storage dtypes, and co-location | Keep concrete bundles under prepared implementations; expose ownership and committed prefix |
| `ModelRegion`, `ExecutionStage` | Local setup manufactures a phase × region list; core mainly validates it | Describe real model execution below the scheduler, not a decorative core graph |
| `ModelCapabilities`, `SpeculationPolicy` | Central MTP/vision choices grow with model mechanisms | Expose actual scheduling bounds and accepted completion behavior |
| `core::NvidiaBackend` | Vendor adapter is exported by nominally neutral core | New runtime has no dependency on it; relocate legacy glue during cutover |
| `server/local.rs` | Frontend constructs model geometry, state, kernels, plan, and runtime | Prepared-model construction owns this work |
| Repeated submission/state maps | Scheduler, runtime, and adapter retain overlapping bookkeeping | One request/scheduling owner; backend retains physical submission resources |
| Per-step static validation | Unchanged model descriptions are repeatedly inspected | Prepare once; retain validation of mutable identity, prefix, budgets, and completion |
| `QualificationScope` strings | Caller-authored identities can omit important dimensions | Canonical structured identity before automatic selection/reuse |

An opaque handle alone would not fix this. Capacity, input support, prefix
advancement, completion, release, and failure ownership still need explicit
contracts. Removing those checks would make the API smaller but less correct.

## Research and its limits

Primary sources were reviewed during this session. The design decisions below
are Ribn's interpretation, not claims that another engine uses this Rust API.

- [vLLM Model Runner V2, versioned design](https://docs.vllm.ai/en/v0.28.0/design/model_runner_v2/)
  separates persistent request rows from scheduled batches and emphasizes
  async-first execution and incremental metadata. This supports retaining request
  state rather than reconstructing it every step. Its GPU implementation and
  reported performance do not establish Ribn's performance.
- [vLLM hybrid cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
  documents different attention/recurrent cache behavior and page-size/padding
  constraints. Ribn therefore shares lifecycle contracts without requiring one
  physical allocator or page shape for every state family.
- [TensorRT-LLM PyTorch architecture](https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html)
  separates the scheduler, model engine, decoder, and resource management. Ribn
  similarly separates request selection from concrete execution/resource work,
  without copying that project's component count or implementation languages.
- [DeepSeek-V4.1-Flash model documentation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  describes causal encoder/decoder asymmetry, shared sparse state, compact
  representations, replay, and model-native speculation. These are boundary
  pressure tests, not implemented Ribn features. Its separately documented input
  encoding also reinforces keeping model input preparation above scheduling.

Qwen3.8-Flash-Next remains the next model target from the direction handoff. The
Qwen blog returned no readable body through the research tool; its detailed
claims were not independently reverified here. Pin the actual artifact, config,
reference implementation, and architecture source before implementing that path.

## Implemented contract

`ribn::PreparedModel` has six methods: immutable information, admission, submission,
completion polling, release, and synchronization. The model owns sequence state
from successful admission through successful release. Rejection/defer retains
nothing. Submission failure retains uncertain resources and faults the Engine.
There is no raw pointer, tensor map, CUDA object, KV spec, or model-family enum
in this interface.

`BatchItem` distinguishes prefill and decode explicitly. A one-token prefill tail
is still prefill. Prefill commits exactly its input chunk, producing one token
only for the final chunk. Decode commits a positive bounded advancement and its
accepted outputs; the implementation may do additional uncommitted draft work.
Partial speculation acceptance need not alter the public request API.

All completion rows are checked before any core prefix/output mutation. The
Engine keeps only one in-flight batch initially. It reuses scheduling buffers,
keeps request state in slots, bounds queued inputs/events, reserves output credits,
and admits prefill after a bounded number of decode-only steps. Live policy is
separate from request semantics and saved in-flight budgets.

Shutdown is a resource operation, not a fake successful completion. Pending
outputs can be discarded during cancellation/shutdown; already committed events
remain available. A failed release retains a retry owner. Uncertain teardown
retains the prepared object rather than dropping allocations still visible to a
device. This must remain true across future futures/async adapters as well.

## Qwen transition

`engine-qwen` implements the new boundary. Its `QwenExecution` privately translates
batch records to existing CUDA execution segments, owns the old logical leases,
and commits them only after validated device completion. It does not contain a
second request scheduler. Host tests inject a backend into this actual adapter
rather than testing an unrelated pseudocode implementation.

`QwenPrepared::load_gguf` assembles loading, state budget, device resources,
kernels, and legacy plan validation. It reports observed free memory and state
reservations instead of treating the previous CLI's fixed weight allowance as
hardware capacity. The adapter currently retains the original Qwen precision,
full-context state reservation, and ordinary greedy decode; it does not implement
new cache formats, speculation kernels, or CUDA Rust migration gate 2.

The `run` command streams decoded token bytes without rebuilding the whole output
string each step. This preserves split UTF-8 bytes across token boundaries; at a
hard output limit the byte stream may end in an incomplete code point. Diagnostics
go to stderr. Output errors still call shutdown. `local` remains unchanged in its
numerical path for comparison. No HTTP service or generic automatic model loader
is being claimed.

## Architecture pressure tests

| Test | Evidence now | Required next evidence |
| --- | --- | --- |
| Different private state layouts | External contract fixtures implement dense and hybrid state without core changes | Second native architecture |
| Multi-token accepted completion | Mock completion and budget/policy tests | Real MTP/other proposer with rollback and numerical parity |
| Qwen hybrid execution | Real adapter compiles; host lease/failure tests pass | Same-artifact GPU replay, concurrency, cancellation |
| Mixed placement and packed state | No layout or placement enum in new runtime | Actual allocator/precision implementation and exact compatibility checks |
| Prefix restore/fork/replay | Fresh admission cannot fabricate a nonzero prefix | Snapshot/materialization ownership, content identity, and completion proof |
| GGUF vs Safetensors | Format not required by `ribn` | Artifact-independent model definition and second loader |
| Metal or another backend | No CUDA dependency in `ribn` | Concrete second hardware implementation |
| Multimodal or other task input | Explicitly outside current token interface | Owned prepared input/output contract; no fake text tokens |

The external fixtures prove an extension boundary, not support for real DeepSeek,
MoE, FP4, host paging, or a second GPU vendor.

## Retirement and naming

After GPU cutover, remove the old request/scheduler runtime and its temporary
Qwen translation. Move surviving model/weight/state helpers to their real owners.
Do not maintain separate old/new feature matrices. The retirement map is:

| Old name/surface | Target |
| --- | --- |
| `ServingRuntime` + `ServingScheduler` | `ribn::Engine` with private scheduling machinery |
| Scheduler-facing `InferenceStateSet` | `SequenceId` and prepared-model-owned state |
| `ExecutionSegment` in common scheduling | `BatchItem` |
| Public region/phase matrix | Concrete model execution code |
| Frontend model assembly | `QwenPrepared::load_gguf`, then proven loader registration |
| `RequestSemantics` for generation | `GenerationOptions`; input encoding above runtime |
| Core NVIDIA adapter | Backend/Qwen internals, then removal of translation |

Do not rename serialized `qwen35.*` GGUF keys to match a product label. Do not
rename all CUDA kernel types while migrating their execution substrate. Extract
shared quantized operations from model-named helpers only when the implementations
actually share a contract.

Before persistent cache or automatic variant selection, define a versioned,
canonical compatibility manifest: artifact bytes/revision, model implementation,
input encoding where relevant, backend/toolchain/runtime and device capabilities,
state encoding/layout, kernel/graph mode, speculation, and distributed layout.
Lossy precision choices need numerical acceptance gates, not merely matching
semantic labels. Display names and in-process sequence IDs remain separate.

## Performance and scope constraints

The synthetic host example measures scheduling/completion/event processing with
an immediate mock, including that mock's completion allocation. It does not
measure GPU inference, end-to-end serving, or a gain over the old implementation.
See [verification](../benchmarks/runtime-contract.md) for reproducible commands.

The next performance audit must separate scheduler CPU time, adapter batch
construction, allocation count, H2D metadata bytes, launches, GPU gaps, prefill,
decode, sampling, preparation, and state memory. Batched/chunked prefill kernels,
integer-dot kernels, CUDA graphs, prefix caching, and multi-batch overlap remain
separate measured work—not benefits already obtained by changing type names.

Known initial limits: one prepared model per Engine, one batch in flight, global
bounded output queue, polling rather than async wakeup integration, no persisted
cache/reload, and an experimental token-generation API. Stabilize task and
capability negotiation before promising a public 1.0 API. No distributed scheduler,
universal model IR, or dynamic Rust plugin ABI is required for this work.
