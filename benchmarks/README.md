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
