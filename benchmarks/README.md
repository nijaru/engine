# Engine benchmark methodology

Benchmarking is part of the architecture, not a release-time exercise. Headline Engine claims must be reproducible against current incumbents on matched hardware, model, precision, workload, and endpoint semantics.

## Load generators

Use three complementary surfaces:

1. **AIPerf** as the primary backend-neutral online load generator for OpenAI-compatible endpoints. It supports request/concurrency control, arrival-pattern sweeps, latency/throughput/goodput analysis, raw exports, server/GPU telemetry, custom datasets, trace replay, and multimodal endpoints.
2. **vLLM `bench serve`** as an incumbent-native cross-check for vLLM and OpenAI-compatible serving behavior.
3. **SGLang `sglang.benchmark.serving`** as an incumbent-native cross-check for SGLang and OpenAI-compatible serving behavior.

Do not rely on one benchmark client for all conclusions. If the neutral and incumbent-native tools materially disagree, investigate before publishing a comparison.

Current references:

- https://docs.nvidia.com/aiperf/
- https://docs.vllm.ai/en/latest/cli/bench/serve/
- https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md

## Reproducibility rules

Every recorded run must capture at least:

- Engine/incumbent name, version, and commit/container digest;
- exact model and revision;
- tokenizer and revision;
- dtype/quantization and any model transformation;
- device model, VRAM, driver, CUDA/runtime versions, clocks/power policy when relevant;
- server command/config and environment variables;
- benchmark-client version and exact command/config;
- endpoint type and streaming behavior;
- warmup method;
- input/output length distribution or dataset identity;
- request-rate/concurrency/arrival pattern and seed;
- sampling parameters and EOS behavior;
- benchmark duration/request count;
- any SLO/goodput thresholds;
- raw result artifacts and server logs sufficient to reproduce anomalies.

Do not compare results produced with materially different semantic settings merely because the nominal model name is the same.

## Metrics

Minimum online serving metrics:

- time to first token (TTFT), including p50/p95/p99;
- inter-token latency / time per output token (ITL/TPOT), including tails;
- end-to-end request latency;
- request throughput;
- output-token throughput;
- total-token throughput where meaningful;
- goodput under explicit TTFT/ITL/E2E SLOs;
- error/cancellation rate;
- server CPU utilization and RSS;
- GPU utilization, memory use, and relevant device telemetry;
- scheduler/host overhead when Engine instrumentation can separate it;
- cold and warm startup time for startup comparisons.

Throughput without latency/SLO context is not sufficient for a serving claim.

## Qualification evidence

- [Runtime contract](runtime-contract.md) — runtime, driver and text-facade device gates.
- [Encoder qualification](encoder-qualification.md) — BERT encoder numerical parity with an
  independent Hugging Face reference.
- [Qwen prefill qualification](qwen-prefill-qualification.md) — chunked same-sequence prefill.
- [Runtime alignment](runtime-alignment/README.md) — host and CUDA frontend overhead.
- [Loader memory](loader-memory/README.md) — sharded BERT host-loading peak and bit preservation.

## Initial workload matrix

Start with deterministic synthetic workloads because they isolate engine behavior, then add realistic traces/datasets.

| Workload | Purpose |
|---|---|
| batch-1 / low concurrency | decode latency and host overhead floor |
| short 256 -> 128 | interactive short chat |
| balanced 1k -> 256 | general serving baseline |
| prefill-heavy 4k -> 128 | prompt processing pressure |
| decode-heavy 256 -> 1k | long generation / reasoning pressure |
| long-context 8k+ -> 256 | KV/state and chunked-prefill behavior |
| mixed lengths | scheduler robustness under heterogeneity |
| repeated shared prefix | prefix-cache/reuse behavior |
| request-rate sweep | saturation curve and goodput frontier |
| concurrency sweep | queueing and scheduling behavior |

Add multimodal, MoE, speculation-specific, trace-replay, and distributed cases only when the corresponding Engine capability exists.

## Native Qwen serving sweep

The Qwen-owned `qwen_serving_bench` example exercises the production AR engine with
the pinned Qwen GGUF path. It replaces the legacy core-runtime harness; historical
measurements do not qualify the migrated harness or imply directly comparable timing:

```text
ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  bash benchmarks/run-qwen-serving-sweep.sh
```

Defaults are 32 output tokens at concurrency 1, 2, 4, and 8. Override them with `TOKENS` and `CONCURRENCIES`, for example:

```text
TOKENS=64 CONCURRENCIES="1 4 8" \
  ENGINE_QWEN_GGUF=/path/to/model.gguf \
  bash benchmarks/run-qwen-serving-sweep.sh
```

The script builds `engine-qwen`'s example and records revision, environment and one
raw log per concurrency under `benchmarks/results/`. Compatible multi-row decode uses
backend batch lanes; mixed/single rows stay per-row. Explicit `--gemv=scalar|warp|int-dot`,
`--prefill-chunk`, fixture prompts and six divergence probes remain available.
Preparation timing now includes loading/staging/kernels; execution timing includes
AR admission and event observation. The effective batch-token limit is printed.
This fixed-arrival sweep is not representative online serving evidence, and device
execution of the migrated harness remains pending.

### Mixed-arrival native traces

Replay a finite open-loop workload through the same AR engine:

```sh
WORKLOAD=benchmarks/workloads/qwen-mixed-arrivals.json \
QUEUED_REQUESTS=4 CONCURRENCIES="1 2 4" \
CONTINUATION_CAPACITY_BYTES=524288000 \
TTFT_SLO_MS=5000 ITL_SLO_MS=200 E2E_SLO_MS=10000 \
ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
bash benchmarks/run-qwen-serving-sweep.sh
```

Trace mode defaults to the qualified warp GEMV and production same-sequence prefill
chunking; the old burst sweep retains its scalar-script/serial-prefill baseline.
The example accepts `--prefill-chunk=off` for a serial comparison.
`--grouped-decode` opts into the experimental compatible-row subgroup candidate;
its device and mixed-workload performance gates are pending. Omit it for the qualified
whole-batch policy. JSON records the choice; it does not enable paging or change admission.

A workload is a JSON array. Each row specifies a nondecreasing `arrival_ms`,
positive `output_tokens`, a `prompt_fixture` path relative to the workload file,
and optional `prompt_tokens` selecting a nonempty fixture prefix. Fixture files use
the three-line artifact/prompt/reference format in
[`crates/qwen/tests/fixtures`](../crates/qwen/tests/fixtures/README.md).
The trace supplies request lengths; do not combine it with burst prompt/token flags.

`--concurrency` bounds active requests; `--queue-capacity` bounds additional requests
(default zero). `--continuation-capacity-bytes` optionally constrains the model's private
continuation authority. Rejected arrivals are counted, never delayed or retried.
TTFT and E2E start at the **planned arrival**, so host dispatch lateness is included
rather than erased; submit-lag statistics expose that lateness. A blocking engine step
can delay actual submission, so this is not an independent network load generator.
ITL measures successive host-observed token events. Successful-request distributions
report nearest-rank p50/p95/p99; failures and rejections are separate. SLO goodput counts
completed requests satisfying every supplied TTFT, maximum-ITL and E2E threshold.
A one-token request has no ITL interval.

The script writes JSON with configuration, workload and per-request outcomes alongside
raw logs and environment metadata. Direct invocation uses `--workload=PATH`,
`--result-json=PATH` (a new file, never overwriting inputs/results) and optional
`--ttft-slo-ms`, `--itl-slo-ms`, `--e2e-slo-ms`.
Inputs are preloaded; model load and client-bookkeeping initialization are outside the
arrival timeline. There is no execution warmup. Reservation capacity and device-free
snapshots are **not** a measured physical peak.

The checked-in eight-request trace is a heterogeneous smoke workload, not steady-state
traffic or competitive evidence. It excludes HTTP and tokenization. Use matched endpoint
load generators above for serving comparisons. Its initial device evidence and the
heterogeneous-capacity regression are in the [runtime contract](runtime-contract.md#mixed-arrival-native-qwen-gate).

## Qwen multi-token prefill gate

Before selecting the experimental same-sequence Qwen prefill path in serving, follow
[`qwen-prefill-qualification.md`](qwen-prefill-qualification.md). It records the
four-layer diagnostic and full 64-layer ignored parity gates plus a matched serial
versus chunk-size 2/4/8 timing matrix for the RTX 4090. The qualified scope and selected default are recorded there; new geometry or variants
need their own device gates. CPU CI and CUDA-feature compilation are not performance
or numerical evidence.

## Baseline run shape

For an OpenAI-compatible completion endpoint, a current AIPerf synthetic example is conceptually:

```text
aiperf profile \
  --model <model> \
  --endpoint-type completions \
  --endpoint /v1/completions \
  --streaming \
  --synthetic-input-tokens-mean <isl> \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean <osl> \
  --output-tokens-stddev 0 \
  --url localhost:8000 \
  --request-count <n>
```

Use the installed AIPerf version's `--help` as the authority for exact flags before a run; preserve the version and full command with the result.

Current vLLM cross-check shape:

```text
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --input-len <isl> \
  --output-len <osl> \
  --request-rate <rps> \
  --num-prompts <n>
```

Current SGLang cross-check shape:

```text
python -m sglang.benchmark.serving \
  --backend <backend> \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --max-concurrency <n> \
  --num-prompts <n>
```

Again, preserve exact installed versions and commands rather than treating these examples as frozen APIs.

## Result layout

Store benchmark outputs outside version control by default. A run should be reproducible from a small checked-in manifest plus external/raw artifacts:

```text
results/
  <date>-<engine>-<model>-<workload>/
    manifest.toml
    client/
    server/
    telemetry/
    summary.json
```

`results/` should remain ignored unless a small curated result is intentionally committed for a regression or published benchmark.

## First success criterion

Before adding broad Engine features, establish a repeatable baseline suite that can answer:

> On one model and one RTX 4090, where do current vLLM and SGLang spend time and host resources, and what performance envelope must Engine match or beat to be credible?
