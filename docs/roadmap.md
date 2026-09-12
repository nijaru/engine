# Roadmap

This is an ordered engineering plan, not release dates. Items below are evidence
gates and implementation targets; they are not architectural requirements when a
better measured design emerges.

## Current status

| Area | Current evidence | Main remaining work |
| --- | --- | --- |
| Existing Qwen GGUF/CUDA path | Same-artifact correctness/performance references and legacy serving path | Keep as oracle through cutover |
| New generation runtime | Host lifecycle/backpressure tests, persistent slots, bounded per-request output, multi-token completion | GPU workload qualification; concurrent driver |
| Shared text frontend | Raw prompt/chat/token input, tokenizer/template reuse, incremental decode, synchronous stream, offline batch, token usage | Concurrent handle, broader generation features, HTTP integration |
| Qwen integration | Artifact-independent config + GGUF adapter + host-tested executor bridge | GPU replay/lifecycle proof; remove legacy bridge after cutover |
| CUDA Rust migration | Resource/toolchain proof gate complete | Representative quantized + recurrent kernels, integration, coverage |
| State/cache | Fixed full-context reservations | Dynamic hybrid state, prefix reuse, pressure/preemption experiments |

## 1. Qualify the new runtime with the existing Qwen kernels

Before deleting the reference serving path, run exact-artifact GPU replay and
lifecycle tests through the new runtime. Cover concurrency 1/2/8/9, long outputs,
mixed prompt lengths, final prefill tails, cancellation in prefill/decode, stop
limits, repeated admission, constrained memory, and fallback batch shapes.

Measure old versus new TTFT, inter-token latency, throughput, CPU scheduling time,
allocations, host/device memory, and preparation. Do not call host compilation or
mock-runtime tests GPU qualification.

## 2. Continue CUDA Rust gate 2 and target the new runtime for integration

Prove representative Q8_1 packing -> quantized integer-dot projection and batched
GDN/recurrent state updates with numerical references, tails, repeated updates,
generated-code inspection, and matched timings. Keep kernel migration separate
from request-runtime retirement.

Once representative kernels are proven, integrate them through the current Qwen
executor/runtime path rather than building another serving stack.

## 3. Finish the reusable application layer and serving driver

The first shared text layer is implemented. Continue from that concrete surface:

- provide a concurrent model handle/driver so independent Rust and server callers
  can submit, stream, and cancel without manually calling `Engine::step()`;
- preserve bounded per-request backpressure when designing channels/wakeups;
- expose raw prompt, chat-message, and token-ID inputs consistently across CLI,
  Rust, and HTTP;
- add request settings incrementally: stop strings, penalties/logit bias,
  logprobs, structured/constrained output, and reasoning/tool formatting only when
  their full execution semantics are implemented;
- keep public request overrides distinct from fully resolved executor settings;
- implement a documented compatibility subset for serving and test unmodified
  clients, including streaming, errors, finish reasons, usage, and disconnect
  cancellation.

`ribn run` now accepts a positional or `--model` path, chat-formatted input by
default, `--raw` for raw completion, prompt/file/piped stdin, context capacity,
and generation length. Interactive terminal chat and `ribn serve` remain future
features.

## 4. Replace fixed per-sequence state reservation with a measured hybrid resource layer

The current Qwen path reserves full configured context state per sequence, which
will limit concurrency. Implement a real dynamic state manager before inventing a
final cache API.

For full-attention state, evaluate paged allocation and compact page-table metadata.
For recurrent state, evaluate checkpoints/materialization appropriate to the actual
Qwen recurrence. A reusable hybrid prefix is valid only when all required state
components correspond to the same semantic prefix. Track their ownership and
completion together even if they use different physical allocators.

Add capacity pressure and resource telemetry. Then test preemption/recompute,
eviction, prefix reuse, and optional host tiers against realistic long-context and
multi-turn workloads. Do not force recurrent state into a KV page abstraction.

## 5. Revisit scheduling with real paging/reuse data

Once dynamic resources exist, compare the current decode-priority/prefill-fairness
policy against alternatives such as a unified scheduled-token budget, cache-aware
priority, and preemption/recompute. Measure TTFT, ITL, throughput, cache hit rate,
GPU utilization, and tail latency across short/long and shared-prefix workloads.

The scheduler should receive enough resource/cost information to make decisions
without owning model tensor layouts. Evolve `Admission::Ready/Deferred` only when
real resource operations demonstrate what additional contract is needed.

## 6. Optimize the host/device pipeline

Profile before choosing work. Candidate optimizations include persistent device
request rows, incremental metadata writes, packed/ragged prefill, mixed
prefill/decode kernels, device-side sampling, fused/vendor kernels, CPU/GPU
scheduler overlap, and multiple in-flight batches.

CUDA graphs should be treated as qualified execution variants with explicit
shape/cache/workspace compatibility and a correct fallback. Do not make graph mode
or an overlap scheme a global assumption.

## 7. Cut over and retire transitional runtime code

After GPU correctness/lifecycle and performance gates pass, make the new path the
supported default, remove the duplicate `local` serving runtime, and remove Qwen's
legacy segment/state translation as its backend adopts the new contracts directly.
Retain independent numerical fixtures and benchmark history.

Normalize remaining parameter roles and shape support before adding a second
checkpoint format. A valid `QwenConfig` is not proof that a backend supports that
geometry.

## 8. Add another architecture/backend as a pressure test

Use a materially different model architecture and, later, another hardware backend
to test what actually belongs in shared code. New attention/state/MoE mechanisms
should reuse request/serving infrastructure where sensible, while shared interfaces
may change when the new implementation exposes a genuine common requirement.

Add speculative decoding only with draft/verify/rollback correctness, stop
semantics, acceptance metrics, resource accounting, and joint variant
qualification. The current multi-token completion contract is necessary but not
sufficient proof.

## 9. Distributed inference when local execution and resource semantics are solid

Ribn may eventually own inference-local tensor/expert/pipeline/sequence parallelism,
communication, and prefill/decode disaggregation inside externally allocated
resources. Fleet placement and datacenter policy remain outside the engine.

The strategic benchmark stays simple: useful model coverage with strong latency,
throughput, memory efficiency, reliability, and integration ergonomics. Cleaner
abstractions matter only insofar as they help achieve those results.
