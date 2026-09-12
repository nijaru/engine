# Ribn

A Rust-first general model inference runtime and serving engine.

The goal is state-of-the-art inference performance, model coverage, reliability,
and familiar CLI/library/server workflows across modern model classes. The current
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
pressure-tests two additional boundaries: logical parameter/version identity is
separate from physical materialization, and model semantics are separate from
resource/deployment topology. These validation types are intentionally not stable
public APIs yet.

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

File input and piped stdin are supported. Interactive terminal chat and an HTTP
`serve` command are not implemented yet. `ribn local` remains the legacy numerical
comparison frontend during migration.

## Library layers

`crates/foundation` (`ribn-foundation`) is a **design-validation** layer for shared
execution infrastructure. It currently models logical parameter versions and
physical materializations, resource topology, and prepared stage placement without
request, token, KV, autograd, or optimizer semantics.

`crates/runtime` (`ribn`) is currently the low-level **AR generation runtime**. It
owns token-generation request lifecycle, scheduling, cancellation, bounded output,
and the current `GenerationExecutor` contract. Those are not intended as universal
contracts for every inference workload.

`crates/batch` (`ribn-batch`) is a second **design-validation** runtime for non-AR
encoder/pooling-style batching. Its inputs and outputs are executor-defined and it
contains no token/prefix/KV concepts. It is not yet a production embedding runtime;
real model pressure tests must determine shape-, memory-, async-, and
resource-aware batching semantics.

`crates/text` (`ribn-text`) is the current shared text frontend used by the CLI: raw
prompt, chat-message and token-ID input, tokenization/chat-template handling,
incremental UTF-8 decoding, synchronous streaming, offline batching, and terminal
token usage. `TextModel::load` is still Qwen/GGUF/CUDA-specific and the high-level
stream borrows the model mutably; both are transitional.

The broader design will add a loaded-model/model-package boundary, Hugging Face +
safetensors/tokenizer/processor loading, typed multimodal inputs/results, a
concurrent model handle, and specialized runtimes for execution regimes that are
not AR token generation. Native optimized execution remains the production target;
a compatibility/reference model path is being evaluated separately for faster
model bring-up and correctness comparison.

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
| `crates/foundation` (`ribn-foundation`) | Provisional parameter/version/materialization, resource-topology, and prepared-placement validation |
| `crates/runtime` (`ribn`) | AR token-generation lifecycle/scheduling/output/executor contract |
| `crates/batch` (`ribn-batch`) | Provisional non-AR batch/encoder runtime pressure test |
| `crates/text` (`ribn-text`) | Current shared text/chat/token input and generation-result frontend |
| `crates/qwen` | Qwen definition, GGUF mapping, and current CUDA AR executor |
| `crates/nvidia` | NVIDIA physical state, resources, and kernels |
| `crates/gguf` | Generic GGUF metadata/tensor reader and current tokenizer support |
| `crates/core` | Legacy execution/state contracts retained for cutover/reference tests |
| `crates/cli` | Current `ribn` command-line frontend |

## Documentation

- [Inference engine design](docs/inference-engine-design.md)
- [Shared execution foundation](docs/execution-foundation.md)
- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Research agenda](docs/research-agenda.md)
- [Runtime redesign history](docs/runtime-redesign.md)
- [Runtime verification](benchmarks/runtime-contract.md)
- [Benchmarks](benchmarks/README.md)
- [CUDA Rust migration](docs/cuda-rust-migration.md)

## License

[Apache-2.0](LICENSE)
