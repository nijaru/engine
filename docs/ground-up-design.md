# Ground-up design and external lessons

Date: 2026-09-11 (America/Los_Angeles)
Status: working design; choices without implementation evidence remain provisional

## Goal

Ribn is trying to be a state-of-the-art inference engine in Rust. The useful
scorecard is correctness, model coverage, latency, throughput, memory efficiency,
reliability, and developer/operator usability on representative workloads.

If rebuilding today, I would still separate request scheduling from model/device
mechanisms, but I would not try to make the common core immune to every future
model change. A good abstraction exposes the information needed by its caller and
keeps the mechanism in its owner. If a new model exposes a real missing common
concept, changing shared code can be the correct design.

The public workflow should remain familiar: load or serve a model, then generate
from prompts/messages/tokens with explicit supported options. There is no need for
an additional user-visible "task defaults" system. Internal preparation can select
and validate an execution implementation without turning that into the normal UX.

## Public interface direction

The following are targets rather than claims about the current feature set:

| Surface | Direction |
| --- | --- |
| Local CLI | `ribn run <model>` with prompt/file/stdin and clear chat-vs-raw behavior; interactive use when implemented |
| Serving | `ribn serve <model>` with a documented, tested compatibility subset and normal streaming/cancellation |
| Rust | Load once, then raw generation, chat, token input, batching, streaming, cancellation, and eventually concurrent callers |
| Python | In-process bindings are worth supporting later in addition to HTTP access |
| Controls | Expose useful context/device/memory/cache/batching/parallelism/precision/sampling options when the implementation actually supports them |

The current shared `ribn-text` layer implements the first reusable slice: raw text,
chat messages, token IDs, incremental UTF-8 decoding, synchronous streaming,
offline batching, and terminal token accounting. The CLI consumes this layer. It
is not yet an async/concurrent server API.

## Internal responsibilities

A useful decomposition is:

- **artifact/model integration:** read weights/config/tokenizer and validate the
  model implementation;
- **execution preparation:** choose/construct a supported device implementation,
  allocate long-lived resources, prepare kernels/graphs when useful;
- **request runtime:** retain request progress, schedule work, apply backpressure,
  commit results, and own cancellation/fault transitions;
- **continuation resources:** allocate/reuse/evict materialized state and expose
  enough capacity/cost information for scheduling;
- **device execution:** perform model computation and own device-visible lifetimes;
- **frontends:** prepare text/chat/token inputs and translate results to CLI, Rust,
  Python, or HTTP semantics.

These are responsibilities, not a mandate for one trait/crate/process per bullet.

## Lessons from current engines

The comparisons below influence Ribn's experiments and roadmap; they are not rules
that Ribn must copy.

### vLLM: persistent request state and token-budget scheduling

vLLM V1 treats scheduled prompt and output work under a token budget and supports
chunked prefill, prefix reuse, and speculative work within that scheduling model.
Its newer model-runner work also emphasizes persistent request rows and incremental
device metadata. Ribn should evaluate whether its current explicit prefill/decode
queues remain optimal once dynamic paging, prefix reuse, and speculation are real.
`StepKind` can remain an execution distinction even if scheduling eventually uses a
more unified budget. Do not rewrite the scheduler from this comparison alone.

References:
- https://docs.vllm.ai/en/latest/usage/v1_guide/
- https://docs.vllm.ai/en/stable/design/model_runner_v2/

### TensorRT-LLM: scheduling cooperates with resource management

TensorRT-LLM separates request scheduling from resource managers that prepare,
update, and free per-request resources. That is a better pressure test for Ribn
than an opaque state handle with no capacity information. The current fixed
`Ready`/`Deferred` admission can evolve when we have a real paged/prefix-cache
implementation: scheduling should see resource availability and cost without
owning physical model layouts.

Reference: https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html

### SGLang: cache-aware scheduling, overlap, and hybrid-state tradeoffs

SGLang combines prefix caching with scheduling and now exposes hierarchical GPU,
host, and external cache tiers. Its Mamba options explicitly trade additional
state buffers for overlap scheduling and branching-point caching. That is directly
relevant to hybrid recurrent/attention models: extra state used to make caching
or overlap possible is a measurable throughput/capacity tradeoff, not a free
abstraction. SGLang's overlap scheduler is also evidence that CPU scheduling and
GPU execution should eventually be pipelined where measurements justify it.

References:
- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx

### mistral.rs: useful Rust precedents and hybrid prefix ownership

mistral.rs is a particularly useful Rust comparison. It uses a model-driving engine
thread, channels for concurrent requests, paged attention, preemption/requeue under
cache pressure, and CUDA-graph fallbacks. Its observability and speculative-decoding
designs also account for recurrent-prefix checkpoints alongside paged KV state for
hybrid models. For Ribn this suggests two concrete experiments:

1. a future concurrent public handle can use a dedicated engine driver rather than
   requiring each caller to hold mutable runtime state and call `step()`;
2. Qwen hybrid prefix reuse should pair full-attention cache pages with recurrent
   state checkpoints/materialization at the same semantic boundary rather than
   pretending KV pages alone define a reusable prefix.

These are design leads to reproduce and measure in Ribn, not proof that the same
thread/channel or cache layout is optimal here.

Reference: https://docs.mistralrs.dev/developer/architecture/

### FlashInfer: ragged/paged metadata and mixed prefill/decode kernels

FlashInfer's serving kernels accept packed ragged batches and page tables instead
of padding every request to a common sequence length. It also exposes mixed
prefill/decode attention paths and reusable workspace/planning objects. This
supports keeping per-request sequence/page metadata compact and persistent and
letting the backend select specialized kernels for the actual batch shape. It does
not imply that Ribn needs FlashInfer's exact page layout or API.

References:
- https://docs.flashinfer.ai/tutorials/kv_layout.html
- https://docs.flashinfer.ai/api/attention.html

### MAX: model integration should inherit the serving stack

MAX's custom-model integration combines model configuration, tokenizer/input
metadata, weight mapping, and execution while retaining shared serving,
continuous-batching, caching, and parallelism infrastructure. Ribn's Qwen package
should similarly contain what is model-specific without requiring a new scheduler
or frontend for each model. Shared kernels/ops should be factored from concrete
implementations when actual reuse appears.

Reference: https://max.modular.com/develop/

### llama.cpp: low-level control and cache/checkpoint features are useful

llama.cpp demonstrates that an accessible low-level model/context/batch/sampler
surface can coexist with convenient CLI/server interfaces. Its server also exposes
continuous batching, prompt/cache reuse, cache precision controls, metrics, and
slot inspection. Ribn should not hide useful execution controls merely to keep a
small facade; high-level convenience and low-level embedding are complementary.

Reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

### TGI: protocol/router and model execution can evolve independently

TGI's router/model-server split is evidence that request protocol/queueing and
model execution can have separate ownership and even separate processes. Ribn
does not need that process split now, but HTTP compatibility should stay outside
model math so deployment architecture can evolve later without rewriting kernels.

Reference: https://huggingface.co/docs/text-generation-inference/architecture

## Design implications for Ribn

The strongest implications are concrete:

- **Do not reserve full maximum-context state per sequence forever.** High
  concurrency will require dynamic/paged allocation or another model-appropriate
  mechanism. Hybrid state may use multiple physical managers.
- **Prefix caching for Qwen hybrid state must cover every continuation component.**
  KV pages plus stale/missing recurrent state are not a valid prefix hit.
- **Scheduling should become resource-aware when dynamic resources exist.** Token
  budget, cache pressure, preemption/recompute, and reuse belong in the policy
  discussion; the scheduler should not need tensor-layout details.
- **Keep host/device work overlap as a performance objective.** A concurrent engine
  driver, persistent device metadata, asynchronous completion, and graph/kernel
  variants should be measured rather than inferred from API shape.
- **Backend execution variants need explicit compatibility.** CUDA graph capture,
  low-precision cache, fused mixed-batch kernels, or speculative paths can be
  faster only for supported shapes/state; maintain a correct fallback and qualify
  variant selection.
- **Model integration should reuse the serving stack.** New architecture code may
  change shared interfaces when a genuine common concept is missing, but should
  not duplicate request scheduling, HTTP semantics, or generic cache bookkeeping.
- **Structured output and tools affect decoding, not just HTTP JSON.** Grammar or
  schema constraints must participate in token selection. Protocol adapters then
  serialize the result.
- **Benchmark direct execution and serving separately.** Track prefill, decode,
  scheduling, memory/cache behavior, TTFT, inter-token latency, throughput, and
  tail latency under concurrency.

## Non-decisions

This review does **not** decide that Ribn needs one universal page size, vLLM's
scheduler, SGLang's radix tree, mistral.rs's thread layout, a general task system,
a model compiler IR, or a plugin ABI. Those choices should be driven by concrete
implementations and measurements.

The largest current implementation gaps remain GPU qualification of the new Qwen
path, fixed-shape CUDA assumptions, dynamic/hybrid state management, concurrent
application driving, and the CUDA Rust kernel migration.
