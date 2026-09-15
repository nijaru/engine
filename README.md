# Ribn

A Rust-first, server-first general model inference engine.

The goal is state-of-the-art inference in idiomatic Rust, initially competitive with
vLLM, SGLang and similar serving engines: performance, broad support for common latest
models and hardware, reliability, and familiar CLI/library/server workflows. The current
implementation is an experimental Qwen GGUF/CUDA **autoregressive** path under
qualification; Qwen and the development RTX 4090 are test/qualification vehicles,
not the architectural scope of the engine.

The target architecture is documented in
[Inference engine design](docs/inference-engine-design.md). In short, the existing
`ribn` token runtime becomes the AR runtime inside a broader loaded-model + shallow
orchestrator design that can also support encoder/pooling, multimodal/omni,
encoder-decoder speech, diffusion/media, and other execution regimes without
forcing them through token-generation contracts.

A provisional [shared execution foundation](docs/execution-foundation.md) now
pressure-tests additional boundaries: logical parameter/version identity is
separate from physical materialization, model semantics are separate from
resource/deployment topology, and backend/operator compatibility can be resolved at
preparation rather than rediscovered in the hot path. These validation types are
intentionally not stable public APIs yet.

[Pipeline composition](docs/pipeline-composition.md) records another important
result: genuinely sequential encoder->decoder models and tightly coupled VLM/omni
execution do not necessarily want the same composition mechanism.

## Current interfaces

Metadata inspection requires no GPU:

```sh
cargo run -p ribn-cli -- inspect /path/model.gguf
```

The experimental CUDA path supports one-shot Qwen text generation. Text is treated
as a user chat message by default; `--raw` performs raw completion instead. The
model can be positional or supplied by the existing `--model` form.

```sh
cargo run -p ribn-cli --features cuda -- run /path/model.gguf \
  --prompt 'Explain a mutex.' --max-tokens 128

cargo run -p ribn-cli --features cuda -- run /path/model.gguf \
  --raw --prompt 'The answer is'
```

File input and piped stdin are supported. `run` checks input before model loading:
up to 256 KiB of UTF-8 for raw completion, or 256 KiB minus the four-byte `user` role
for chat. Oversized input is rejected, not truncated; file/stdin reads consume at most
one extra byte to detect overflow. Interactive terminal chat and an HTTP
`serve` command are not implemented yet. `ribn local` remains the legacy numerical
comparison frontend during migration.

## Library layers

`crates/foundation` (`ribn-foundation`) is a **design-validation** layer for shared
execution infrastructure. It currently models logical parameter versions and
physical materializations, resource topology, and prepared stage placement without
request, token, KV, autograd, or optimizer semantics. A test-only RMSNorm experiment
also checks that semantic/backend compatibility can be resolved at preparation time
without committing Ribn to a general operator IR.

`crates/runtime` (`ribn`) is currently the low-level **AR generation runtime**. It
owns token-generation request lifecycle, scheduling, cancellation, bounded output,
and the current `GenerationExecutor` contract. Stable `RequestId` is now passed to
executor admission separately from internal `SequenceId`, which lets model-owned
request-scoped prepared state be correlated without turning `TokenRequest` into a
generic multimodal payload container. Those AR contracts are still not intended as
universal interfaces for every inference workload.

`ribn::driver` adds a low-level owned token interface: one execution worker, cloneable
handles, bounded request permits and streams, and explicit shutdown/retry. Blocking
and async access use the same engine. The driver is host-tested and GPU-qualified
on the RTX 4090 for the recorded runtime and text gates; it accepts encoded input only
and does not bound raw text preprocessing. See its [contract](docs/resource-protocol.md#owned-ar-driver-contract)
and [qualification](benchmarks/runtime-contract.md#owned-token-driver-host-gate-7005fad).

`crates/batch` (`ribn-batch`) is a second **design-validation** runtime for non-AR
encoder/pooling-style batching. Its inputs and outputs are executor-defined and it
contains no token/prefix/KV concepts. A variable-length reference encoder showed
that request count alone is not enough to form safe batches, so the executor can
shorten the oldest FIFO candidate set using its concrete constraints. SafeTensors
and local HF-style package fixtures now exercise this path from artifact metadata
through model-owned parameter interpretation. There is deliberately no universal
work/cost unit or length-bucketing policy yet.

`crates/safetensors` (`ribn-safetensors`) is a thin artifact adapter. It validates
SafeTensors bytes and exposes names, shapes, dtypes, and borrowed payload bytes; it
does not assign parameter semantics or allocate execution tensors.

`crates/hf` (`ribn-hf`) is a local Hugging Face-style package resolver for
`config.json` and unsharded/sharded SafeTensors weights. It intentionally preserves
raw model metadata and resolves files/parameter ownership without choosing a model
architecture, runtime, backend, or processor implementation.

`crates/text` (`ribn-text`) is the shared text frontend used by the CLI: raw prompt,
chat-message and token-ID input, tokenization/chat-template handling under explicit
byte bounds, a bounded preprocessing pool, incremental UTF-8 decoding, owned
concurrent streaming, and ordered bounded-window offline batching. `TextModel` is a
cloneable handle over an owned token execution worker and `TextOwner` controls
shutdown. Its behavior is host-qualified and the CUDA-backed lifecycle, cancellation and
multi-request determinism gates pass on the RTX 4090; `TextOwner::load` is still
Qwen/GGUF/CUDA assembly. Stop-token and decoding behavior is documented in the
[text contract](docs/resource-protocol.md#text-application-facade).

A sequential encoder->AR pressure test now passes prepared state in-process and
correlates it with the correct AR request even when handoffs are installed out of
order. This validates one staged-execution seam, not a general multimodal solution.
VLM-style prompt-positioned encoder dependencies remain a separate coupled-runtime
pressure test.

The broader design still needs the real loaded-model/model-package boundary, remote
Hugging Face repository/revision resolution, tokenizer/processor package metadata,
typed multimodal inputs/results, a concurrent model handle, and specialized
runtimes for execution regimes that are not AR token generation. Native optimized
execution remains the production target; a compatibility/reference model path is
being evaluated separately for faster model bring-up and correctness comparison.

## Intended use

Ribn is intended to become usable for local/embedded Rust applications, offline
batch inference, Python in-process workloads, OpenAI/Anthropic-compatible serving
where appropriate, embeddings/reranking, multimodal understanding, speech,
image/video generation, adapters/quantization, and eventually multi-device,
multi-node and disaggregated inference. These are design targets rather than claims
about current implementation support.

The public UX should stay conventional: load or serve a model and invoke the
operation you need. Internal AR/encoder/diffusion runtime selection should not turn
into a mandatory user-facing task-default system.

The shared foundation is intentionally reusable below inference policy. That keeps
future training or other compute runtimes from needing to reimplement parameter,
device, operator, placement, and collective infrastructure, without adding autograd
or training semantics to Ribn's inference fast path.

## Development

Use the Rust toolchain pinned in `rust-toolchain.toml`:

```sh
python3 tools/check-boundaries.py
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

CUDA-feature compile/tests run in CPU CI, but numerical/lifecycle qualification
requires an actual compatible NVIDIA GPU.

## Code map

| Location | Current responsibility |
| --- | --- |
| `crates/foundation` (`ribn-foundation`) | Provisional parameter/version/materialization, resource-topology, prepared-placement, and test-only semantic-op validation |
| `crates/runtime` (`ribn`) | AR token-generation lifecycle/scheduling/output/executor contract |
| `crates/batch` (`ribn-batch`) | Provisional non-AR batch/encoder runtime with executor-informed FIFO batch sizing and cross-runtime pressure tests |
| `crates/safetensors` (`ribn-safetensors`) | Generic validated SafeTensors artifact/tensor views |
| `crates/hf` (`ribn-hf`) | Local HF-style config/weight package resolution without model semantics |
| `crates/text` (`ribn-text`) | Current shared text/chat/token input and generation-result frontend |
| `crates/qwen` | Qwen definition, GGUF mapping, and current CUDA AR executor |
| `crates/nvidia` | NVIDIA physical state, resources, and kernels |
| `crates/gguf` | Generic GGUF metadata/tensor reader and current tokenizer support |
| `crates/core` | Legacy execution/state contracts retained for cutover/reference tests |
| `crates/cli` | Current `ribn` command-line frontend |

## Documentation

Implementation references, in order:

1. [Target design](docs/inference-engine-design.md) — priorities, architecture and API semantics.
2. [Execution contracts](docs/resource-protocol.md) — ownership, preparation, waiting and snapshots.
3. [Roadmap](docs/roadmap.md) — ordered slices, open decisions and acceptance evidence.

Supporting references (not competing implementation plans):

- [Current implementation](docs/architecture.md)
- [Foundation experiments](docs/execution-foundation.md)
- [Composition experiments](docs/pipeline-composition.md)
- [Research sources and questions](docs/research-agenda.md)
- [Runtime redesign history](docs/runtime-redesign.md)
- [Runtime verification](benchmarks/runtime-contract.md)
- [Benchmarks](benchmarks/README.md)
- [CUDA Rust migration](docs/cuda-rust-migration.md)

## License

[Apache-2.0](LICENSE)
