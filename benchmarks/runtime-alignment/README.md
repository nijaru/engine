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
serving latency. Real GPU driver performance and mixed-arrival workload comparisons
remain unmeasured. No claim of faster execution follows from the async interface.

```sh
cargo run --release -p ribn --example host_overhead --locked -- --driver
```

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
