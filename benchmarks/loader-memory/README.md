# Sharded BERT loader host memory

Measured on Apple M3 Max, macOS 26.6.2 (25G83), Rust 1.98.0, release mode,
2026-10-06. The baseline is `1dadac1`; the candidate removes BERT's duplicate decoded
parameter cache, retains one source shard and evicts old cache entries before opening
incoming shards. No CUDA execution or upload runs.

The synthetic F32 package has 109,482,240 parameters: hidden width 768, 12 layers,
12 heads, intermediate width 3072, vocabulary 30,522 and 512 position embeddings.
Four shards contain embeddings, layers 0–3, layers 4–7 and layers 8–11 plus pooler.
Their files total 437,950,960 bytes. Values are deterministic constants per parameter;
this is a loading-memory workload, not a numerical model reference.

Three baseline processes preceded three candidate processes. Inputs were generated
before measurement; filesystem cache state was not controlled. `/usr/bin/time -l`
measured each process directly, not Cargo. [Raw CSV](2026-10-06-macos.csv) records
peak process RSS and the loader's internal elapsed time.

| Median peak process RSS | Baseline | Candidate |
| --- | ---: | ---: |
| Bytes | 1,549,664,256 | 997,031,936 |
| MiB | 1477.88 | 950.84 |

The candidate's median peak was 35.7% lower in this workload. Every run produced
parameter count 109,482,240 and digest `4ae09df98cbc9b25` over all decoded F32 bits.
The measured process includes a post-load digest pass and retains the returned host
weights. Neither timing nor peak RSS establishes a general speedup, a hard byte bound,
large-model support or GPU/serving performance.

## Reproduce

From the repository root, choose a new fixture directory:

```sh
python3 benchmarks/loader-memory/generate.py /tmp/ribn-loader-fixture
cargo build --release -p engine-bert --example bert_loading_bench --locked
/usr/bin/time -l target/release/examples/bert_loading_bench /tmp/ribn-loader-fixture
```

For the baseline, copy `crates/bert/examples/bert_loading_bench.rs` into a separate
worktree at `1dadac1`, build it there and run against the same fixture. The probe's
FNV-1a digest is a deterministic bit-comparison check, not an artifact security hash.

## Remaining limits

- Each decoded parameter remains in the complete `HostWeights` owner. Device upload
  is still a separate operation over that owner, not incremental streaming.
- SafeTensors files are read into owned bytes; conversion into shared storage can
  temporarily overlap allocations. This change does not eliminate those copies.
- Resident-cache counts exclude caller-held artifact clones and allocator overhead.
- Interleaved shard layouts can require repeated reads under one-shard residency.
  The compact sharded fixture test covers value preservation across those rereads;
  their load-time cost needs workload-specific measurement.
