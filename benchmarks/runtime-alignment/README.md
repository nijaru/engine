# Host runtime and delivery cost

## Owned token driver (`7005fad`)

Measured 2026-09-14 on Apple M3 Max, macOS 26.6.2 (25G83), Rust 1.98.0, release
mode. Same immediate executor, one input token and 512 generated tokens per request,
concurrency 1/8/32. Five direct/driver pairs per concurrency, alternating measurement
order. No explicit warmup; the first direct c1 run is a visible startup outlier.
Raw observations: [owned-driver-7005fad-macos.csv](owned-driver-7005fad-macos.csv).

Timing includes enqueue/acknowledgement and token/terminal consumption. Model creation,
worker spawn call and shutdown are outside timing; first worker scheduling can still
contribute. Direct delivery drains each engine iteration. Driver consumption rotates
through blocking owned streams. Both have four runtime events per request; the driver
adds eight channel slots per stream. No model math or device completion delay runs.

| Concurrency | Direct median elapsed | Driver median elapsed | Direct ns/output | Driver ns/output |
| --- | ---: | ---: | ---: | ---: |
| 1 | 46,166 ns | 325,167 ns | 90.2 | 635.1 |
| 8 | 237,500 ns | 863,125 ns | 58.0 | 210.7 |
| 32 | 811,875 ns | 3,070,958 ns | 49.6 | 187.4 |

The owned path costs more than direct dispatch. These observations expose channel,
notification and worker bookkeeping cost; they establish neither model throughput nor
serving latency. The matched CUDA comparison below covers the real GPU path at
concurrency 1; concurrency above 1 and mixed-arrival workloads remain unmeasured. No
claim of faster execution follows from the async interface.

```sh
cargo run --release -p ribn --example host_overhead --locked -- --driver
```

## Owned text facade over CUDA (`d61b62c`)

Measured 2026-09-15 on the idle RTX 4090, Fedora, Rust 1.98.0, release mode, pinned
artifact SHA-256 `322e194f…23482`, concurrency 1. Five independently loaded direct/handle
pairs per workload, alternating mode order per pair. Model load and shutdown are outside
timing; one warmup per workload precedes each measured sample. Both modes use the same
`GgufProcessor`, the same `Engine::with_defaults` engine configuration, the same sampling
options, and the same incremental UTF-8 decoding work. Raw observations:
[owned-text-facade-cuda-overhead-2026-09-15.csv](owned-text-facade-cuda-overhead-2026-09-15.csv).

```sh
ssh desktop   # idle GPU, one process at a time
RIBN_MODEL=/path/pinned.gguf cargo test --release -p ribn-text --features cuda \
  --locked --offline --lib matched_frontend_overhead -- --ignored --test-threads=1 --nocapture
```

Median of five pairs per mode; `TTFT` is submission-to-first-delivered-text-token and
`elapsed` includes full consumption of the terminal event:

| Workload | Direct TTFT | Handle TTFT | ΔTTFT | Direct elapsed | Handle elapsed | Δelapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw, 9 prompt / 32 output tokens, `Length` | 86.7 ms | 92.1 ms | +5.4 ms (+6.2%) | 938.2 ms | 950.5 ms | +12.3 ms (+1.3%) |
| Chat, 17 prompt / 2 output tokens, `Stop` | 142.3 ms | 150.3 ms | +8.0 ms (+5.6%) | 164.3 ms | 172.8 ms | +8.4 ms (+5.1%) |
| Raw, 347 prompt / 32 output tokens, `Length` | 2550.6 ms | 2562.1 ms | +11.5 ms (+0.4%) | 3649.8 ms | 3665.7 ms | +16.0 ms (+0.4%) |

Every one of the ten loaded configurations produced identical tokens, text, finish
reason and usage per workload, which extends determinism evidence across load
boundaries as well as across requests. Within-mode spread was under 1% of the median
except for the handle path's 2-token chat TTFT (149.6–155.6 ms).

Interpretation for the slice-2 exit: the owned handle costs a fixed **5–12 ms per
request** — channel submission, wakeup and worker scheduling — plus roughly 0.3–0.5 ms
per delivered token of extra delivery work. That is under 0.5% of end-to-end time for a
347-prompt-token request and under 1.5% for a short raw request; only a trivially short
chat request pays a visible ~5% relative cost. The async/bounded-retention guarantees
therefore do not impose a throughput-relevant penalty at concurrency 1.

Limits of this evidence: it is a **single-request host-overhead comparison**, not serving
throughput. It does not cover concurrency above 1, mixed arrivals, or batch windows, and
the direct path polls `Engine::step` with `yield_now` while the handle path parks on
channels, so host CPU usage is deliberately not matched. No SOTA or vLLM/SGLang parity
claim follows from it.

## Historical output-isolation comparison

This is a synthetic CPU scheduling/completion/event benchmark, not model
inference and not hardware qualification. Both versions use the same immediate
mock and include its result-vector allocation. The baseline is the code at
`47878859` (exported through snapshot `a926e181`); the candidate is the ground-up
alignment in this change. Rust 1.98.0, release mode, shared Linux x86-64 sandbox.
Three alternating runs per version, each with 128 warmup and 10,000 measured
iterations. Raw CSVs are beside this note.

Median of each run's median iteration time, nanoseconds:

| Concurrency | Baseline | Aligned runtime | Difference |
| --- | ---: | ---: | ---: |
| 1 | 110 | 140 | +30 |
| 8 | 421 | 601 | +180 |
| 32 | 1923 | 2424 | +501 |
| 128 | 7882 | 10456 | +2574 |

The aligned runtime does more work: per-request limits, independent draining,
ready-mailbox tracking, and aggregate reservation checks. An initial hash-map
mailbox implementation measured roughly 24.5 microseconds at concurrency 128;
replacing hot-path lookups with stable mailbox slots reduced that overhead.
The final candidate is still slower than the original global-queue-only mock.
This is an explicit isolation-versus-bookkeeping tradeoff, not a speedup claim.
Absolute times are noisy on this shared host; controlled-host repeats, allocation
instrumentation, and matched GPU timelines are required for a performance gate.

Reproduce the candidate:

```sh
cargo run -p ribn --release --example host_overhead
```

The fixture does not model stalled clients. Dedicated contract tests establish
peer progress under per-request backpressure; this benchmark establishes neither
real serving throughput nor real-world tail latency.

Candidate Rust-source digest (sorted path bytes followed by file bytes):
`7389636731164804a68506212b08045a8895124a7043dc1d7742319f55ac1eb9`
