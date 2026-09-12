from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:80]!r}")
    file.write_text(text.replace(old, new, 1))


replace_once(
    "docs/execution-foundation.md",
    """It also exposed a concrete loading concern that toy fixtures could hide: repeatedly
calling a per-parameter helper that reopens a SafeTensors shard would reread the same
checkpoint bytes many times. The BERT loader therefore keeps a private artifact cache
and opens each unique shard once. A reusable package/weight-set owner should preserve
that property for real integrations, while model-specific tensor names, expected
shapes and semantic mapping remain above it.
""",
    """It also exposed a concrete loading concern that toy fixtures could hide: repeatedly
calling a per-parameter helper that reopens a SafeTensors shard would reread the same
checkpoint bytes many times. `ribn-hf::LocalWeightSet` now owns that reusable behavior:
it opens each resolved shard lazily on first use, reuses it for later tensor views,
and leaves unused shards unopened. Model-specific tensor aliases, expected shapes
and semantic mapping remain above the package layer.
""",
)
replace_once(
    "docs/execution-foundation.md",
    """The BERT architecture pressure test is the first substantially real model-semantic
use of this path. It preserves the same executor-defined request/result boundary and
uses sequence length to constrain batching while running actual BERT attention and
feed-forward structure. No new universal batching abstraction was required. The
next useful pressure comes from attention masks, padding/ragged layouts and real
device memory/compute admission rather than another synthetic cost type.
""",
    """The BERT architecture pressure test is the first substantially real model-semantic
use of this path. It preserves the same executor-defined request/result boundary and
runs actual BERT attention and feed-forward structure. Follow-up tests add attention
masks and compare padded versus ragged batch cost: masked padding is semantically
inert for real tokens, and the concrete executor can shorten a padded FIFO batch even
when the same requests fit a ragged token budget. No new common batching abstraction
was required. The next useful pressure comes from real device memory/compute
admission, asynchronous execution and optimized kernels rather than another synthetic
cost type.
""",
)
replace_once(
    "docs/execution-foundation.md",
    """- evidence that real model loading needs each SafeTensors shard owned/opened once
  for repeated tensor access rather than reread per parameter;
""",
    """- `LocalWeightSet` lazily owns each resolved SafeTensors shard once for repeated
  parameter access while leaving model semantics above the package layer;
- BERT attention-mask semantics and padded-versus-ragged batch-cost pressure tests
  pass without changing the common `ribn-batch` contract;
""",
)
replace_once(
    "docs/execution-foundation.md",
    """The next high-value pressure tests are masked/padded/ragged encoder/device
execution, ordinary architecture resolution using the concrete Qwen+BERT evidence,
a genuine encoder-decoder model, a real VLM/processor integration, an iterative
non-AR runtime, more real semantic-op/backend implementations, and a second hardware
backend.
""",
    """The next high-value pressure tests are real encoder device execution/resource
admission, ordinary architecture resolution once another production model makes that
boundary useful, a genuine encoder-decoder model, a real VLM/processor integration,
an iterative non-AR runtime, more real semantic-op/backend implementations, and a
second hardware backend.
""",
)

replace_once(
    "docs/roadmap.md",
    """| Qwen GGUF/CUDA AR | Legacy same-artifact references, new host lifecycle tests, CUDA-feature compilation | New-path GPU qualification; fixed-shape/model assumptions |
""",
    """| Qwen GGUF/CUDA AR | Legacy same-artifact references, new host lifecycle tests, CUDA-feature compilation; experimental same-sequence multi-token prefill path and parity/benchmark harness compile but are not device-qualified or serving-selected | Run the new prefill parity/timing gates on the 4090; new-path GPU qualification; fixed-shape/model assumptions; true chunked GDN/full-attention prefill after evidence |
""",
)
replace_once(
    "docs/roadmap.md",
    """| Non-AR runtime validation | Generic bounded batch runtime; parameter-version pinning; executor-informed FIFO batch sizing; an actual BERT encoder reference path loads HF/SafeTensors weights and executes embeddings, self-attention, residual/LayerNorm, FFN and pooler semantics | Attention masks/padding/ragged batching, async/cancellation/resource admission, device execution and optimized kernels |
""",
    """| Non-AR runtime validation | Generic bounded batch runtime; parameter-version pinning; executor-informed FIFO batch sizing; actual BERT encoder semantics; attention masks and padded-versus-ragged batch-cost tests pass without a common-runtime change | Async/cancellation/resource admission, device execution and optimized kernels |
""",
)
replace_once(
    "docs/roadmap.md",
    """| Model/artifact separation | `QwenConfig` independent of GGUF; thin SafeTensors artifact adapter; local HF-style config + unsharded/sharded weight-package resolver; Qwen and BERT integrations keep model meaning above artifact parsing | Remote HF repository/revision resolution, tokenizer/processor package integration, architecture resolution/model package, reusable shard opening/materialization path |
""",
    """| Model/artifact separation | `QwenConfig` independent of GGUF; thin SafeTensors artifact adapter; local HF-style config + unsharded/sharded weight-package resolver; `LocalWeightSet` lazily reuses opened shards; Qwen and BERT integrations keep model meaning above artifact parsing | Remote HF repository/revision resolution, tokenizer/processor package integration, architecture resolution when a second production model justifies it, backend materialization path |
""",
)
replace_once(
    "docs/roadmap.md",
    """- pressure-test attention masks, padding/ragged shapes and real device resource
  costs on the encoder path; let that evidence refine non-AR batching/admission
  rather than inventing a universal cost unit;
- establish the general loaded-model/model-package and architecture-resolution
  boundary using the concrete Qwen and BERT integrations. An ordinary central Rust
  enum/registry/factory is acceptable if it is the simplest fit; extensibility does
  not require a plugin ABI or a scheduler that never changes;
""",
    """- pressure-test real device resource costs, asynchronous execution, cancellation
  and failure behavior on the encoder path; masks plus padded/ragged host execution
  already fit executor-owned batch selection without a universal cost unit;
- establish the general loaded-model/model-package and architecture-resolution
  boundary when another production model makes it useful. Qwen is still the only
  production model path and BERT is a pressure-test integration, so a registry now
  would mostly formalize strings rather than remove real duplication. An ordinary
  central Rust enum/registry/factory remains acceptable when evidence justifies it;
""",
)
replace_once(
    "docs/roadmap.md",
    """First make repeated tensor access practical for real checkpoints: one resolved
SafeTensors shard should be opened/owned once and provide many borrowed parameter
views or prepared materializations. The BERT reference path currently proves this
need with a private per-loader artifact cache; promote only the reusable artifact
ownership mechanism, not BERT-specific semantics.
""",
    """Repeated tensor access is now practical without moving model semantics into the
package layer: `LocalWeightSet` lazily opens each resolved SafeTensors shard once and
reuses it for many borrowed tensor views. Backend-specific prepared materialization
and streaming/loading policy remain future work.
""",
)

replace_once(
    "AGENTS.md",
    """- SafeTensors + local HF-style package code keeps artifact semantics separate from
  model semantics;
- an actual BERT architecture reference path executes over that package boundary and
  confirms sequence-length batching without another common scheduling abstraction;
""",
    """- SafeTensors + local HF-style package code keeps artifact semantics separate from
  model semantics, while `LocalWeightSet` lazily reuses each opened shard across
  parameter views;
- an actual BERT architecture reference path executes over that package boundary;
  attention masks and padded-versus-ragged batch-cost tests still fit executor-owned
  `select_batch` without another common scheduling abstraction;
""",
)
replace_once(
    "AGENTS.md",
    """Next evidence-driven pressure points are masked/padded/ragged encoder/device
execution, ordinary architecture resolution using the concrete Qwen+BERT cases, a
genuine encoder-decoder model, and an actual VLM/processor integration. Prefer those
over another synthetic framework layer. The actual VLM should determine whether the
AR scheduler needs explicit per-item dependency descriptors, a model/resource
planner, or another representation.
""",
    """Next evidence-driven pressure points are real encoder device/resource execution,
a genuine encoder-decoder model, and an actual VLM/processor integration. Delay a
central architecture resolver until another production model makes it useful rather
than formalizing the current Qwen-only product path plus test-only BERT. Prefer these
concrete integrations over another synthetic framework layer. The actual VLM should
determine whether the AR scheduler needs explicit per-item dependency descriptors,
a model/resource planner, or another representation.
""",
)
replace_once(
    "AGENTS.md",
    """Optimized variants need correctness qualification for the scope in which they are
automatically selected. Compilation or host tests are not GPU evidence.
""",
    """The NVIDIA package now contains an experimental same-sequence multi-token Qwen
prefill path that batches projection/norm/FFN work while keeping recurrent and KV
state updates causal. It is intentionally unwired from serving. Its ignored hybrid
parity test and opt-in benchmark harness must run on the 4090 before any automatic
selection or performance claim; compilation is not GPU evidence.

Optimized variants need correctness qualification for the scope in which they are
automatically selected. Compilation or host tests are not GPU evidence.
""",
)

replace_once(
    "crates/nvidia/src/decode.rs",
    """//! corresponding parity-tested CUDA kernel. Prefill uses the same
//! state-advancing model body autoregressively, one token at a time, but can
//! skip the output head for intermediate prompt tokens whose successor is
//! already known.
""",
    """//! corresponding parity-tested CUDA kernel. The default serving prefill still uses
//! the state-advancing model body autoregressively, one token at a time. An
//! experimental same-sequence chunk path batches token-independent work while
//! keeping recurrent/KV updates causal; it remains unwired pending device parity
//! and performance qualification.
""",
)
