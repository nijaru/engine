# Ground-up design and alignment

Date: 2026-09-11 (America/Los_Angeles)
Status: target design with an explicit implementation gap map
Baseline reviewed: `4787885949c4f89f6ed62a9757e3daef88fdf667`

## What I would build without the existing code

Ribn's goal is a state-of-the-art inference engine in Rust. Judge progress by
correctness, model support, latency, throughput, memory use, reliability, and
usability on real workloads. The architecture should support those goals with
reusable model and device implementations, not become a separate product or a
universal model compiler.

Keep changing model mechanisms below the request and scheduling interfaces.
Loading should validate model/backend support and prepare execution, but those
internal steps do not define the public workflow. Ordinary users load or serve
a model and submit requests; advanced users can configure and integrate the
supported execution mechanisms directly where useful.

Internally, generation has a prefill/decode lifecycle. Other inference operations
may need different execution contracts when implemented. This is not a mandate
for a user-visible task system or a generic framework before useful model
execution. A new attention mechanism should not change the generation API.

## Public interface direction

Ribn should provide familiar inference-server, command-line, and library
interfaces, with sensible defaults and explicit configuration. Follow established
conventions unless a concrete usability, performance, or correctness benefit
justifies a difference. Rust changes the implementation, not the workflow users
must learn to run a model.

The following are interface targets, not a list of shipped capabilities:

| Surface | Planned experience |
| --- | --- |
| Serving | `ribn serve <model>` with conventional server options and OpenAI-compatible HTTP operations for the supported subset; existing clients should not need a Ribn-specific protocol |
| Local inference | `ribn run <model>` with prompt, file/stdin input, interactive use, and familiar generation options as implemented |
| Rust library | Idiomatic model loading, generation, batching, and streaming, with typed configuration and errors; retain lower-level executor/device integration where useful |
| Configuration | Explicit model source/revision, context, device, memory, cache, batching, quantization/precision, parallelism, and sampling controls as those capabilities are implemented |

Use vLLM and SGLang as references for serving/offline workflows and llama.cpp for
local execution conventions. Review their actual interfaces before choosing exact
flags or signatures. Choose consistent semantics for each surface rather than
combining every option or preserving incompatible quirks. Record a concrete
reason for deliberate deviations; a new internal abstraction is not such a reason.

Normally infer the operation from the requested API/command and model metadata.
Ask for an explicit task selection only when there is a real ambiguity. Model
loading may select a compatible backend by default, but must respect explicit
choices, show the effective configuration, and reject unsupported combinations.
Do not silently change numerical policy or ignore a user override.

Ordinary callers should not need to assemble cache allocations, scheduler queues,
execution plans, or streams. This does not prohibit advanced control over cache
policy, memory placement, batching, device resources, or execution. Keep useful
low-level access with clear ownership and safety contracts. Likewise, expose
ordinary streaming responses or an idiomatic Rust stream rather than requiring
application callers to manage internal mailbox credits.

Document CLI/configuration precedence, option units, defaults, errors, and supported
HTTP behavior. Test compatibility with selected unmodified clients before claiming
it. Show unsupported features explicitly instead of accepting ineffective options.
Existing `run --model ...` examples remain valid until an intentional syntax
migration is implemented and documented. `serve`, positional model syntax for
`run`, interactive input, and the high-level library surface are plans, not new
features introduced by this document.

## Boundaries and the data they own

| Boundary | Owned information and responsibility | Kept out |
| --- | --- | --- |
| High-level API | Inputs, output semantics, cancellation, and streaming responses | Required assembly of model geometry or device allocations; optional lower-level access remains separate |
| Artifact reader | Container metadata, tensor directory, encoded bytes, exact source identity | Request scheduling and architecture execution |
| Model definition | Validated model dimensions, layer semantics, parameter roles, input preparation rules | GGUF field names, CUDA pointers, serving queues |
| Preparation | Resolve definition + artifact + device + options into a supported execution; report identity, limits, costs, readiness | Per-token expensive search or compilation |
| Generation runtime | Sequence identity, committed progress, admission, scheduling, output, failure ownership | KV/GDN/MoE families and physical state formats |
| Model executor | Concrete forward/speculation algorithms, typed continuation bundle, execution-local resource decisions | Fleet allocation and application protocols |
| Device implementation | Allocations, transfers, streams, completion, kernels, vendor primitives | Application request semantics |

These are ownership boundaries, not a mandate for seven traits, crates, threads,
or processes. The code should have fewer modules when there is no independent
lifetime or implementation choice to justify a split.

### Model portability is not just an opaque handle

A coarse executor interface insulates the scheduler, but a model author should
not have to rebuild tokenization, lifecycle, scheduling, diagnostics, sampling,
and every kernel to implement it. Reuse validated model configuration and math;
reuse device primitives when their layouts and numerical contracts really match.
Keep specialized attention, routing, packed projections, and state allocators
replaceable. Do not standardize a lowest-common-denominator operator graph to
make every backend look identical.

The useful change matrix is:

| Change | Expected place to change |
| --- | --- |
| Checkpoint revision with the same architecture | Artifact/configuration and requalification |
| GGUF to another container | Reader and tensor/metadata mapping; not generation lifecycle |
| Another valid model geometry | Model configuration and backend preparation/kernels as needed |
| New attention/state mechanism | Model definition/execution and physical state; not request queues |
| Different GPU vendor | Device implementation and model lowering where required |
| Different speculative proposer | Executor composition and qualification; accepted-result contract remains |
| New input/output task | A task adapter/API, with shared resource and lifecycle rules where applicable |

New kernels are legitimate work. Editing the common request scheduler for each
new attention mechanism is not. Moving code between files does not by itself
establish portability or numerical correctness.

## Preparation and resources

Model identity, artifact identity, prepared execution identity, and sequence ID
are different things. A prepared execution binds exact weight bytes/revision,
model implementation, numerical policy, device capability/runtime, kernel set,
state encoding/layout, capture mode, speculation, and distributed layout where
relevant. Use a canonical, versioned compatibility manifest for automatic
selection, persistent preparation products, and state reuse. A display label or
caller-authored string with missing dimensions is not proof of compatibility.

Preparation reports why it selected an implementation, whether it is experimental
or qualified for the exact scope, required and optional work, estimated versus
actual memory, and readiness. Failure should identify the unsupported geometry,
option, artifact, or resource budget before a long load wherever possible.
Present this as normal loading progress and diagnostics; expose explicit compile
or preparation controls only where they serve a concrete supported use case.

A process-local resource owner can share device/model allocations across task
instances later. Executors must expose actual reservations, capacity pressure,
and materialization cost to policy without exposing model tensor internals.
Keeping state opaque must not turn the scheduler into a blind token counter.
The existing admission result (`Ready`/`Deferred`) is a useful minimum, not the
complete resource/cost interface. Build that richer interface against real paging,
host lookup, or multi-executor contention measurements rather than guessed fields.

Live scheduling policy can change at step boundaries. A different physical layout,
precision, backend, or model requires a prepared replacement and compatible state
migration or draining—not merely changing a live flag. Fleet placement remains
external; a process-local resource budget is not a cluster scheduler.

## Fast path and correctness

Keep active requests in stable slots. Prepare only incremental batch metadata;
separate persistent CPU state from memory borrowed by asynchronous device work.
Do not reuse a staging allocation until completion proves it is no longer read.
Validate immutable model/implementation compatibility during preparation, and
retain per-submission validation of mutable identity, prefix, budgets, and results.

The state model must distinguish allocated capacity, valid continuation contents,
physical work that is in flight, and committed output. A nonzero position is not
proof of retained state. Restoring/forking a prefix needs exact compatible content
and a completed materialization operation. Recurrent state, sparse structures,
shared state, and compressed state need not share one physical allocation shape.

Generation consumes inputs and produces outputs; those counts are not identical
at every lifecycle boundary. Final prefill produces the first token without
consuming it. Speculative execution may compute more work than it commits. Only
accepted progress appears in public completion. Stop handling must not expose
rejected output or silently reuse state beyond a stopped prefix.

Output must have per-request limits as well as an aggregate bound. Reserve credits
before submitting work. One client should not consume the whole output budget
when other admitted clients have credits. Preserve each request's order; a global
arrival order across clients is not a generation semantic. Execution state can
be reclaimed while committed output remains deliverable.

Cancellation is intent, not completion. Failed submission can leave partially
queued work; failed release can leave an allocation owner. Retain a retry or
quarantine owner rather than reconstructing state from counters. Explicit shutdown
must report uncertainty. Conservative retention after a failed device barrier is
preferable to freeing memory that may still be device-visible, but a leak is a
fault containment policy, not successful cleanup.

A ground-up fast path would support asynchronous completion/wakeup integration
and independently schedulable work without unnecessary host/device barriers.
Overlap, multiple batches, and graph execution still require proof of dependency,
cancellation, and buffer lifetimes. They are not obtained merely by naming a
method `submit`.

## Alignment implemented in this pass

| Gap at the reviewed baseline | Change made | What remains |
| --- | --- | --- |
| `PreparedModel` implied universal inference while its contract was token generation | Renamed to `GenerationExecutor`; metadata is `ExecutorInfo`/`GenerationLimits`, errors are `ExecutionError` | Other task interfaces are not implemented |
| Qwen model definition lived in the GGUF reader | `QwenConfig` and layer classification now live in `engine-qwen`, with no normal dependency when its features are disabled | The existing forward executor is still fixed-shape |
| The format reader exported Qwen-specific providers | `QwenGguf` owns metadata/tensor mapping in the model package; the GGUF crate no longer knows the model | Safetensors and normalized weight-binding portability need a real second adapter |
| Neutral core exported NVIDIA submission glue | Moved the adapter and its integration tests into the NVIDIA package | The legacy execution/state types still need staged retirement |
| One shared event queue let a slow consumer stop peers | Per-request mailboxes, aggregate and local credits, request-specific draining, bounded ready-list membership | Aggregate saturation and occupied model capacity still backpressure work; disconnect/wakeup integration is not done |
| Output isolation initially added repeated hashing | Stable mailbox slots for the execution path; public request lookup is separate | No allocation-free or inference speedup claim |
| New engine combined configuration, selection, completion, and output internals in one file | Separate modules with one request owner, not another scheduler abstraction | Lifecycle remains conservative and one batch is in flight |
| Defaults could exceed a small prepared model's limits | `Engine::with_defaults` derives compatible sequence/token limits | General model/variant resolution is not implemented |
| A local CLI was named `engine-server` | Package `ribn-cli`, binary `ribn`, and CPU-only `inspect` | `serve`, broad model auto-detection, packaging and SDK wrappers remain |

Production dependency direction is checked by `tools/check-boundaries.py` in CI.
Format-free configuration tests run without GGUF/CUDA features. Contract tests
cover a stalled client beside a progressing peer, event delivery after execution
slot reuse, mixed draining, credit reservations, and model-compatible defaults.
These are concrete boundary and lifecycle tests, not model-quality tests.

## Remaining distance from the ground-up target

**Execution portability is the largest remaining gap.** The CUDA Qwen body still
uses pinned dimensions and serialized tensor names in parts of its execution
implementation. Introduce a validated backend preparation/profile description,
normalize weight roles, and replace fixed assumptions incrementally. Reuse the
current numerical references. Do not accept arbitrary configurations while
quietly running kernels for the original geometry.

**The compatibility bridge is not the destination.** `QwenCuda` still translates
to legacy segments and state leases; `core` still compiles old runtime code and
Qwen still uses old fixed-context state reservations. Run the GPU cutover gate,
then remove this translation and the old serving runtime. Kernel migration and
request-runtime retirement are different proof gates. Do not add a third runtime
or maintain parallel feature roadmaps.

**Preparation, resource observability, and asynchronous integration remain thin.**
The current code has memory checks, explicit admission, synchronous preparation,
one in-flight batch, and polling. It lacks the full resource report, canonical
qualification/selection manifest, state restore/reuse, wakeup integration, and
multi-executor memory coordination described above. A persistent release failure
can still stall normal driving; retry/quarantine telemetry and independent cleanup
progress need a specific design and fault tests before network serving.

The API is experimental. These changes align the code with the target; they do
not make every item in the target implemented or hardware-qualified.

## External evidence checked for this decision

These primary sources support the engineering pressures, not a claim that Ribn
has matched their performance or implemented their complete architecture:

- [vLLM Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/):
  separates persistent request state from per-step inputs and uses incremental
  device metadata. This supports stable slots and explicit async staging lifetime.
- [TensorRT-LLM PyTorch architecture](https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html):
  separates execution/scheduling and resource managers with prepare/update/free
  operations. This supports exposing resource behavior without a universal tensor
  representation in the request scheduler.
- [vLLM hybrid cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/):
  describes grouping and padding needed by its shared-page allocation design.
  Ribn should not impose a common physical page size on every continuation type.

The boundary decisions and gap map above are this project's design judgments.
