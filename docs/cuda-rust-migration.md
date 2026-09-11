# CUDA Rust NVIDIA backend migration

## Decision and scope

CUDA Rust is the intended foundation for Engine's NVIDIA backend: Rust-authored device kernels with cuTile for tile-oriented computations and cuda-oxide for explicit SIMT control. This is a migration direction, not a claim of hardware qualification or improved performance. The existing CUDA C++ implementation is the migration oracle, not a permanent second implementation to maintain after equivalent coverage is qualified.

Focus on NVIDIA now. A second hardware backend should test the semantic boundary when it has a real implementation and qualification target; a speculative Metal/AMD framework or backend survey is not a prerequisite.

Vendor libraries remain valid implementation choices. “Rust-first” does not require rewriting cuBLAS or every third-party kernel. Full migration means replacing Engine-owned CUDA C++ kernels and their NVRTC authoring pipeline, and converging on one coherent NVIDIA resource/execution owner. It does not mean replacing the CUDA driver or NVIDIA's compilers with Engine code.

## Boundary review

The current `ComputeBackend` in `crates/core/src/backend.rs` validates semantic plans and typed state, submits batches, polls completion, and releases physical state through an explicit lifecycle operation. No cudarc buffer, stream, event, or CUDA Rust type crosses this interface. Stream and graph support are capability flags, not requirements every backend must implement.

Core currently also contains the NVIDIA-specific `NvidiaDispatcher` and adapter in `crates/core/src/nvidia.rs`. Those names are not a CUDA library dependency: the seam exchanges Engine plans, state, submission identities, and outcomes. Moving that adapter is not needed to adopt CUDA Rust; do not conflate organizational cleanup with the migration.

Keep these ownership boundaries:

- Core owns admission, stable request identities, scheduling, logical state position, cancellation intent, and committed output.
- The NVIDIA backend owns physical KV and recurrent state, model residency, scratch, device allocations, compilation, streams, launches, and completion observation.
- The dispatcher retains resources for in-flight submissions. Completion permits logical commitment; requesting cancellation does not establish device completion.
- Partial enqueue failure must drain successfully before recycling resources. Uncertain completion faults the backend and retains resources rather than reporting successful release.
- CUDA Rust tensors, futures, graphs, and compiler representations remain backend-local. Its execution machinery may implement a submission, but must not become a second request scheduler.

`crates/nvidia/src/serving.rs` already documents and implements the relevant stream ordering, pinned output leases, pending submission records, deferred release, and fault retention. Preserve those contracts even if their physical implementation changes.

## Target shape

```text
Engine request scheduler / typed state / submit-poll contract
                            |
                 NVIDIA serving dispatcher
                            |
          backend-owned CUDA resources and execution
                  /                     \
          cuTile Rust                cuda-oxide SIMT
      tile-oriented kernels       explicit warp/thread kernels
                  \                     /
                     NVIDIA device
```

The two upstream projects share `cuda-core` and `cuda-async`. Assess those as the eventual common substrate, rather than assuming cudarc must stay forever or that two host runtimes must coexist permanently. During migration, borrowed handles and allocation adapters can let new kernels use existing resources. The bridge must have one deallocation owner and explicit access ordering; an `Arc` proves liveness, not freedom from conflicting accesses.

Select Tile versus SIMT per operation. Start by investigating Tile for normalization, attention, and recurrent matrix operations, and SIMT for packed quantization, integer-dot GEMV, and warp-sensitive code. These are hypotheses, not fixed assignments: preserve algorithms where appropriate and change layout or tiling when measurements justify it.

## Migration surface

| Current source | Migration responsibility |
| --- | --- |
| `activation.cu`, `kernels/quantized/*.cu` | Rust device implementations, including packing, decoding, embeddings, scalar/warp/batched projections |
| Embedded CUDA source in `model_ops.rs` | Normalization, reductions, rotary embedding, attention, gates, convolution, KV append, recurrent state updates |
| `activation.rs`, `quantized.rs`, `model_ops.rs` launch wrappers | Compiled kernel preparation, typed arguments, shape/context validation, launch contracts |
| `cuda.rs`, `staging.rs`, `state.rs` | Allocation, transfers, weight and physical-state ownership; migrate only after resource interoperability is proven |
| `decode.rs`, `serving.rs`, `submissions.rs` | Compose kernels under existing model and submission semantics; preserve cancellation and uncertain-failure ownership |
| CUDA-gated server callers and benchmarks | Update device-resource construction and qualification paths with backend changes |

Do not assume Rust removes runtime shape/context checks, every unsafe operation, or launch geometry. Do not rebuild lazy operation graphs or synchronize once per kernel merely because a tutorial does so. Persistent resources and bounded submission work remain requirements.

## Evidence checked

Source review on 2026-09-09 used [cuTile Rust](https://github.com/NVlabs/cutile-rs/tree/2eed75e) and [cuda-oxide](https://github.com/NVlabs/cuda-oxide/tree/26754ae). Pin full revisions for the executable proof; these moving upstream sources are not yet Engine dependencies.

- cuTile documents foreign CUDA context/stream borrowing, external PTX/CUBIN integration, and foreign allocation ownership. Its `cudarc_interop.rs` example uses a stand-in allocation owner, not an actual cudarc integration test. The 2026-09-10 probe verifies the ownership half of this on Engine's actual libraries (see below); the kernel half remains unverified.
- cuTile has attention, RMSNorm, softmax, GEMM, asynchronous execution, graph, and JIT disk-cache examples. Availability is not proof of compatibility with Engine's shapes, state layouts, failure policy, or graph mode.
- cuda-oxide exposes `dp4a` intrinsics, warp operations, shared memory, and barriers. These match mechanisms used by Engine's existing quantized kernels; generated code and arithmetic still require validation.
- cuTile source says sm8x support starts with CUDA 13.2, with CUDA 13.3 recommended. This is more specific than the introductory blog's blanket 13.3 requirement. Check the actual qualification host rather than inferring its toolkit from cudarc's `cuda-13020` feature.
- cuda-oxide source currently pins nightly-2026-08-28 and lists CUDA 13.x host requirements, differing from the blog's older setup. Isolate kernel compilation/toolchain requirements from non-CUDA workspace development where practical; prove the build arrangement rather than assuming a nightly backend invalidates stable core code.

That review is now backed by a hardware probe on the qualification host (below). No Engine kernel has been authored in Rust yet, and nothing has been measured on Qwen3.8-27B Q4 GGUF; published results on other models and GPUs do not transfer to this artifact or device.

## Hardware probe on the qualification host, 2026-09-10

The RTX 4090 host runs Fedora 44 with driver 615.71.09 (CUDA UMD 13.4), an NVIDIA-runfile toolkit 13.1 at `/usr/local/cuda-13.1` (with `/usr/local/cuda` pointing at it), rustup stable plus nightly, and the pinned Qwen3.8-27B Q4 artifact present. Both upstream trees were cloned at the reviewed revisions and checked out there: cuTile Rust `2eed75e`, cuda-oxide `26754ae` — the same commits the table above was reviewed against, so those citations are current rather than stale.

**The SIMT track runs on the existing toolkit.** `cargo oxide doctor` reports every requirement satisfied on CUDA 13.1: `cuda.h`, nvcc 13.1.80, libNVVM 2.0, nvJitLink 13.1, `libdevice.10.bc`, `llc` from the pinned nightly's rustlib, the clang 22 resource directory, the RTX 4090 at compute capability 8.9, and both optional debuggers. `cargo oxide setup` then built and published the `librustc_codegen_cuda.so` backend for revision `26754ae52c`, and `cargo oxide run vecadd` compiled and executed a Rust-authored kernel on the 4090 with all 1024 elements correct. cuda-oxide needs CUDA 13.0+ only; a toolkit upgrade is not on its critical path.

**The tile track needs a newer toolkit before it can run on this GPU.** With `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.1`, the cuTile `hello_world` example generates Tile IR and then fails at `tileiras` discovery with the compiler's own 13.2 floor:

```text
ERROR cuTile requires CUDA 13.2 or newer: the resolved toolkit at /usr/local/cuda-13.1 is CUDA 13.1. Set CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA 13.2+ install (the shared CUDA host-side crates themselves support 13.0+).
```

Three independent checks agree, so this is a real floor rather than a message artifact: `cutile-compiler` sets `MIN_TILE_CUDA_VERSION = 13020` and asserts that exact 13.1 diagnostic in its own tests; the 13.1 `tileiras --help` lists only `sm_100`, `sm_103`, `sm_110`, `sm_120`, and `sm_121` for `--gpu-name`, so it cannot target the 4090's `sm_89`; and the cutile-rs README places `sm_8x` support at CUDA 13.2. `cuda-toolkit-13-3` was therefore installed from the host's already-configured NVIDIA `cuda-fedora44.repo`; it pulls toolkit components only — compiler, libraries, tools, NVML, documentation — and no driver package, so the driver and the 13.1 tree were untouched. CUDA 13.3 is the version cutile-rs recommends and the one `cuda-core`'s default toolkit candidate list names. Its `tileiras` (13.3.36) lists `sm_80` through `sm_121`, including `sm_89`, and the cuTile `hello_world` example then ran on the 4090 and printed `Hello, I am program <0, 0, 0> in a kernel with <1, 1, 1> programs.` One side effect to know about: the package repoints `/usr/local/cuda` from 13.1 to 13.3, so anything that must resolve a specific toolkit should set `CUDA_TOOLKIT_PATH` rather than inherit the symlink.

**One shared substrate, confirmed by dependency direction rather than by convention.** `cuda-bindings`, `cuda-core`, and `cuda-async` are published from NVlabs/cutile-rs at 0.3.1 (crates.io, 2026-09-04); cuda-oxide 0.3.1 depends on those crates.io versions and keeps its SIMT surface in their `simt` modules. cuda-oxide's kernel-authoring crates (`cuda-device`, `cuda-macros`, `cuda-host`, `cuda-artifact-finalizer`) are not published, so any Engine kernel-authoring proof pins the cuda-oxide revision. The shared layer also reuses cuda-oxide's published `oxide-artifacts` 0.2.1 for artifact loading. There is no second host runtime to reconcile and no allocation bridge to invent between the two tracks.

**The resource-owner decision is constrained by what cuda-core can borrow.** Its `runtime` layer exposes deliberately non-owning foreign adoption: `Device::borrow_raw` / `borrow_with_owner`, `Stream::borrow_raw` / `borrow_with_owner`, and module/function `borrow_raw`, documented for external frameworks including cudarc. The `simt` layer that both kernel tracks launch through is different: the only raw-construction path for `DeviceBuffer` is `from_raw_parts`, whose contract transfers ownership and frees the pointer on drop, and `CudaStream` has no foreign-handle constructor at all. So Engine cannot adopt a cudarc allocation into cuda-core without handing over ownership, and should not keep two live wrappers over one allocation. The coherent target is the one the migration already names: make cuda-core the single context/stream/buffer owner, and let any remaining legacy code receive borrowed raw handles until it is retired. Proving exactly that mix — a cuda-oxide kernel launched over memory whose lifetime one owner owns — is the remaining gate-1 deliverable.

## Ordered proof and migration gates

### 1. Reproducible build and resource interoperability

Pin compatible upstream revisions and verify the Linux GPU host, toolkit, driver, and Rust toolchains. Build and run both tracks on the 4090. Demonstrate operations over the same context, stream, and allocation without a host round trip, double ownership, or premature release. Inspect compatibility between the shared runtime versions rather than assuming matching crate names imply compatibility.

Deliverable: a repeatable hardware-gated smoke test, explicit toolchain setup, and a decision on the common resource owner. Non-CUDA builds must remain usable without the NVIDIA toolchain. Do not upgrade the host or disrupt another GPU workload as an incidental setup action.

Status (2026-09-10): the pinned revisions are cloned on the host; cuda-oxide's SIMT track builds and runs on the existing toolkit 13.1, and the tile track runs after installing `cuda-toolkit-13-3` beside it (the install repoints `/usr/local/cuda` to 13.3, so set `CUDA_TOOLKIT_PATH` explicitly when a pinned revision matters). The shared-substrate and resource-owner questions are answered by the probe above. `cuda-rust-probe/interop` now demonstrates the mixed-ownership case on the RTX 4090 in both directions — cuda-core owning while cudarc's raw bindings write over the borrowed stream, and cudarc owning while cuda-core adopts all three handles and releases them before the owner — with zero leaks and zero errors under `compute-sanitizer --tool memcheck --leak-check full`. What remains for this gate is an Engine-owned kernel rather than an upstream example: a probe that authors a SIMT or tile kernel and executes it over the shared resources, plus the same smoke test wired to the project's own fixtures.

### 2. Representative quantized and stateful execution

Prove the chain Q8_1 packing → Q4_K integer-dot projection using actual encoded layout and batch-1/batched shapes. Check rounding, signed packed dots, alignment, scale/min corrections, and tail/rejection cases against the existing tests and host references. Q8_1 packing is a warp-reduction/packing operation, not merely elementwise addition.

Separately prove a batched GDN state update, including distinct per-request persistent matrices, output strides, and repeated updates. Resolve how tile partitioning expresses those independently owned allocations; do not assume a contiguous output tensor represents the complete state boundary.

Deliverable: numerical tests, generated-code inspection for packed integer dots, and matched kernel timings. A passing easy kernel alone is not the adoption gate.

### 3. Asynchronous serving integration

Integrate the proven kernels into a real scheduler submission with existing kernels filling unmigrated operations. Prove completion polling, output commitment, in-flight cancellation with peer progress, fallback batch sizes, scratch reuse, partial enqueue failure, and retained resources after uncertain completion. Dropping a future must not silently release in-flight state.

Deliverable: hardware-gated lifecycle tests and same-artifact full-model greedy replay. Keep compilation and specialization outside the per-step scheduler path. Required kernels must be prepared before readiness or use an explicitly qualified fallback; measure cold and warm preparation independently from steady-state execution.

### 4. Complete kernel coverage and converge resources

Migrate remaining quantization families, embeddings, model operations, and both single-row and batched execution paths in verifiable families. Reuse host references and preserve their independence from the new device implementation. Converge on the selected CUDA resource owner and remove transitional allocation/stream adapters when their callers are gone.

Deliverable: the complete text path uses Rust-authored Engine kernels; no permanent legacy/new-backend selection matrix is introduced. Graphs or new algorithms are separate changes unless required for the adopted execution contract.

### 5. Qualify and retire the legacy pipeline

Repeat existing full-model parity and diverse long greedy replay gates, asynchronous failure/cancellation tests, and matched serving sweeps at concurrency 1–8. Record artifact, compiler, driver/toolkit, GPU, execution mode, numerical tolerances, preparation time, memory use, latency, and throughput. Investigate material regressions rather than treating a language change as a performance win by definition.

Remove superseded `.cu` files, embedded CUDA strings, Engine's NVRTC compilation path, temporary migration flags, and unused dependencies only after replacement coverage passes. Keep reference mathematics and qualification fixtures. Historical commits retain the old implementation without requiring it in the shipped backend.

Completion means a qualified Rust-authored NVIDIA path with one coherent resource lifecycle and non-CUDA builds still green—not merely that CUDA Rust examples run.
