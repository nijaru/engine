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

- cuTile documents foreign CUDA context/stream borrowing, external PTX/CUBIN integration, and foreign allocation ownership. Its `cudarc_interop.rs` example uses a stand-in allocation owner, not an actual cudarc integration test. Engine's bridge remains unverified.
- cuTile has attention, RMSNorm, softmax, GEMM, asynchronous execution, graph, and JIT disk-cache examples. Availability is not proof of compatibility with Engine's shapes, state layouts, failure policy, or graph mode.
- cuda-oxide exposes `dp4a` intrinsics, warp operations, shared memory, and barriers. These match mechanisms used by Engine's existing quantized kernels; generated code and arithmetic still require validation.
- cuTile source says sm8x support starts with CUDA 13.2, with CUDA 13.3 recommended. This is more specific than the introductory blog's blanket 13.3 requirement. Check the actual qualification host rather than inferring its toolkit from cudarc's `cuda-13020` feature.
- cuda-oxide source currently pins nightly-2026-08-28 and lists CUDA 13.x host requirements, differing from the blog's older setup. Isolate kernel compilation/toolchain requirements from non-CUDA workspace development where practical; prove the build arrangement rather than assuming a nightly backend invalidates stable core code.

No CUDA Rust kernel has been compiled or executed for Engine yet. Published results on other models and GPUs do not establish performance on Qwen3.8-27B Q4 GGUF or the RTX 4090.

## Ordered proof and migration gates

### 1. Reproducible build and resource interoperability

Pin compatible upstream revisions and verify the Linux GPU host, toolkit, driver, and Rust toolchains. Build and run both tracks on the 4090. Demonstrate operations over the same context, stream, and allocation without a host round trip, double ownership, or premature release. Inspect compatibility between the shared runtime versions rather than assuming matching crate names imply compatibility.

Deliverable: a repeatable hardware-gated smoke test, explicit toolchain setup, and a decision on the common resource owner. Non-CUDA builds must remain usable without the NVIDIA toolchain. Do not upgrade the host or disrupt another GPU workload as an incidental setup action.

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
