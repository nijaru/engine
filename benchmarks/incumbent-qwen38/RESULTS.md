# Qwen3.8-27B + DFlash2 on RTX 4090

Date: 2026-08-20

## Hardware and builds

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, Ada SM 8.9
- Driver: 610.57.04
- CUDA: 13.1.80
- Target: `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` (16.46 GB)
- Target KV: `q4_0/q4_0`
- Master llama.cpp: `a298422da`, build 10548, CUDA FA and graphs enabled, `GGML_CUDA_ARCHITECTURES=89`
- DFlash2 llama.cpp worktree: `5ecbe1ac1`, build 10498, PR #27342 head
- Default test context: 16,384; greedy, seed 42, 384 generated tokens
- Server default used 4 slots unless `-np 1` is stated

The DFlash2 PR is still open upstream. Main supports DFlash v1 (`draft-dflash`), but not the DFlash2 graph. DFlash2 therefore requires the PR worktree.

## Baseline and recommended short-context result

Three runs per workload; median decode speed. The normal-chat prompt is a distributed-systems explanation; coding-agent is a retry-decorator patch; hard reasoning is a distributed rate-limiter design.

| setup | normal chat | coding agent | hard reasoning | acceptance | VRAM |
|---|---:|---:|---:|---|---:|
| master plain AR, 4 slots | 49.4 tok/s | 49.6 tok/s | 49.4 tok/s | — | ~15.9 GB target-only* |
| DFlash2 Q8 drafter, n=5, 4 slots | 121.0 | 108.2 | 90.0 | 0.582 / 0.500 / 0.378 | 21.2 GB |
| DFlash2 Q4_K_M drafter, n=5, 4 slots | 123.2 | 110.3 | 92.3 | 0.574 / 0.495 / 0.382 | 21.2 GB |
| DFlash2 Q4_K_M drafter, n=5, `-np 1` | ~90.0 | 109.8 | 92.9 | 0.370 / 0.495 / 0.382 | 18.4 GB |

*AR VRAM was measured with `-np 1`; target-only memory is not directly comparable to the DFlash rows.

`-np 1` does not materially change decode speed in this sample, but frees about 2.8 GB. It is the correct single-user setting and improves context headroom.

## DFlash2 n-max sweep

Q8_0 drafter, 4 slots, two runs per workload. Acceptance is chat / code / reasoning.

| n-max | chat tok/s | code tok/s | reasoning tok/s | acceptance |
|---:|---:|---:|---:|---|
| 3 | 106.7 | 101.1 | 96.7 | .726 / .661 / .617 |
| 4 | 113.6 | 105.4 | 89.8 | .643 / .571 / .440 |
| 5 | 121.0 | 108.2 | 90.0 | .582 / .500 / .378 |
| 6 | 121.1 | 105.3 | 87.2 | .528 / .434 / .326 |
| 7 | 118.8 | 105.0 | 85.9 | .473 / .399 / .294 |
| 8 | 117.3 | 103.5 | 84.7 | .473 / .399 / .294 |

n=5 is the best balanced setting. n=6 wins the short chat prompt but loses on code and hard reasoning. Larger blocks do not help.

## Drafter quant sweep

n=5, 4 slots, two runs per workload.

| drafter | chat | code | reasoning | acceptance chat/code/reason |
|---|---:|---:|---:|---|
| Q4_K_M | 123.2 | 110.3 | 92.3 | .574 / .495 / .382 |
| Q8_0 | 121.0 | 108.2 | 90.0 | .582 / .500 / .378 |
| BF16 | 112.6 | 101.0 | 84.6 | .582 / .494 / .373 |

The Q4 drafter is the practical choice: similar acceptance and lower draft cost. The final 3-run Q4 vs Q8 rerun measured Q4 ahead by about 2% on normal chat, code, and reasoning because of lower draft compute.

## Context-length scaling

Coding-context prompts were 6,766 / 28,126 / 56,292 / 112,992 actual tokens. First run of each fresh server, Q4 drafter n=5.

| filled context | AR tok/s | DFlash2 tok/s | DFlash acceptance | prefill tok/s (DFlash) |
|---:|---:|---:|---:|---:|
| 6.8K | 48.2 | 101.1 | .480 | 2,239 |
| 28.1K | 44.9 | 86.2 | .444 | 2,306 |
| 56.3K | 41.2 | 83.3 | .504 | 2,113 |
| 113.0K | 35.5 | 62.7 | .459 | 1,781 |

DFlash2 keeps a useful advantage at long context, but it is not context-invariant. At ~113K, decode falls to ~63 tok/s and first-turn prefill takes ~63 seconds. Prompt-cache reuse makes subsequent turns cheap to ingest; it does not remove decode slowdown.

## Lookup/context drafting

Main and the DFlash2 branch both accept comma-separated speculative types. A draftless n-gram implementation has precedence over the model drafter. This combination works:

```text
--spec-type draft-dflash,ngram-map-k
```

On a 28K-token repetitive code context:

- n-gram-only: ~50 tok/s; second run acceptance ~0.98, but target verification remained the bottleneck.
- DFlash2 Q4 + ngram-map-k: ~90 tok/s first run, ~96 tok/s cached run.
- DFlash2 alone at comparable context: ~86 tok/s first run.

The combination is worth testing in real Pi sessions because it can help repeated code/tool output, but n-gram-only is not a replacement for DFlash2 here. The synthetic code corpus is not a substitute for a captured agent trace.

## Native MTP

Main can create an MTP draft context directly from the target GGUF; the separately downloaded MTP file was not needed and was removed.

| MTP n-max | chat | code | reasoning |
|---:|---:|---:|---:|
| 1 | 72.6 | 74.5 | 74.3 |
| 2 | 92.5 | 86.9 | 86.2 |

MTP n=2 is slower than DFlash2 Q4 n=5 on these workloads. Keep it as a fallback/comparison, not the everyday configuration.

## KV and GPU observations

- Q8/Q8 KV was tested at 16K. It produced a CUDA memory-fit warning with `-ngl 999`, used ~21.5 GB, and did not show a stable throughput advantage. Keep Q4/Q4 for context headroom.
- A DFlash2 decode sample reached ~95% SM utilization, ~79% memory-controller utilization, and ~426 W on a 450 W limit. The card was not power-limited. The short-context target/drafter path is compute/kernel-limited rather than simply idle or PCIe-limited.
- At 113K context, prefill fell to ~1,781 tok/s and decode to ~63 tok/s. The long-context limit is attention/KV work plus verification cost.

## Recommendation

Everyday single-user Pi preset:

```text
master target: Qwen3.8-27B-UD-Q4_K_M.gguf
DFlash2 PR build: 5ecbe1ac1
DFlash2 drafter: Qwen3.8-27B-DFlash2-Q4_K_M.gguf
--spec-type draft-dflash
--spec-draft-n-max 5
--cache-type-k q4_0 --cache-type-v q4_0
--flash-attn on
--cuda-graphs on
--parallel 1
--n-gpu-layers 999
```

Use the current main build for plain AR, n-gram, and MTP tests. Use the DFlash2 PR build only for DFlash2 arms. Do not chase 300+ tok/s: on this 4090, the tested ordinary-workload range is ~90–123 tok/s at 16K and ~63 tok/s at ~113K context, with hard reasoning at the low end.

## Remaining work

- Capture one real Pi trace with tool output and repeated source fragments; this is the highest-value lookup-drafting validation.
- Test asymmetric K/V (`q8_0/q4_0`) only if extra context capacity is needed; the current Q4/Q4 result is the safer default.
- Avoid downloading larger target quants on the 24 GB card unless quality testing is explicitly prioritized. No target-quant sweep was run by design.
