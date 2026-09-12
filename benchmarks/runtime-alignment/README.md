# Host cost of output isolation

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
