# CUDA Rust NVIDIA backend migration

## Decision and scope

CUDA Rust is the intended foundation for Engine's NVIDIA backend: Rust-authored device kernels with cuTile for tile-oriented computations and cuda-oxide for explicit SIMT control. This is a migration direction, not a claim of hardware qualification or improved performance. The existing CUDA C++ implementation is the migration oracle, not a permanent second implementation to maintain after equivalent coverage is qualified.

Focus on NVIDIA now. A second hardware backend should test the semantic boundary when it has a real implementation and qualification target; a speculative Metal/AMD framework or backend survey is not a prerequisite.

Vendor libraries remain valid implementation choices. “Rust-first” does not require rewriting cuBLAS or every third-party kernel. Full migration means replacing Engine-owned CUDA C++ kernels and their NVRTC authoring pipeline, and converging on one coherent NVIDIA resource/execution owner. It does not mean replacing the CUDA driver or NVIDIA's compilers with Engine code.

## Current priority (2026-09-22)

The current gate-2 candidate is **deferred, not accepted** after the bounded
[representative timing comparison](#gate-2-candidate-deferred-2026-09-22). Correctness
progress does not offset its single-row projection and GDN regressions. Do not start
gate 3 or expand kernel coverage on this candidate. Continue the roadmap's 4b/4c path
on the qualified backend; this is not a rejection of the CUDA Rust direction.

Reopen with evidence from a focused source specialization or upstream change that
addresses the measured regressions, then finish the outstanding rejection and loaded-
code checks before promotion. Preserve pinned versions, artifacts and serialized runs.

If gate 2 passes, take one qualified slice through gate 3's existing execution owner,
including completion, cancellation and uncertain-failure retention. Keep the qualified
CUDA C++ oracle for operations not yet replaced. Do not make gates 4–5, complete Rust
kernel coverage, or replacement of every host resource wrapper prerequisites for the
minimal serving result in [the roadmap](roadmap.md#current-execution-order).

A failed numerical/performance/integration gate should produce a bounded diagnosis and
an explicit stop/defer decision, not an open-ended compiler or framework project. Ribn
can continue on its qualified backend. Migration success and competitive inference are
separate claims; test both. This ordering changes priority, not numerical tolerances,
resource-lifetime contracts or the eventual one-owner migration target.

## Runtime integration update (2026-09-11)

The model-neutral `ribn::GenerationExecutor` boundary now has a host-tested runtime
and an experimental Qwen adapter. Qualify that adapter using the existing kernels
before gate 3 integrates CUDA Rust kernels through it. Do not integrate a second
new serving path around the legacy request API. Kernel gate 2 remains independent;
none of the runtime tests claims it has passed. The older boundary descriptions
below explain the retained comparison path during this transition. See
[runtime redesign](runtime-redesign.md) and [roadmap](roadmap.md).

## Boundary review

The current `ComputeBackend` in `crates/core/src/backend.rs` validates semantic plans and typed state, submits batches, polls completion, and releases physical state through an explicit lifecycle operation. No cudarc buffer, stream, event, or CUDA Rust type crosses this interface. Stream and graph support are capability flags, not requirements every backend must implement.

The NVIDIA-specific `NvidiaDispatcher` and adapter now live in `crates/nvidia/src/adapter.rs`, not core. The retained legacy seam still exchanges execution plans, state, submission identities, and outcomes. Its relocation changes ownership of the code, not GPU qualification or CUDA Rust kernel status.

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

## Historical evidence checked before the gate-1 update

Source review on 2026-09-09 used [cuTile Rust](https://github.com/NVlabs/cutile-rs/tree/2eed75e) and [cuda-oxide](https://github.com/NVlabs/cuda-oxide/tree/26754ae). Pin full revisions for the executable proof; these moving upstream sources are not yet Engine dependencies.

- cuTile documents foreign CUDA context/stream borrowing, external PTX/CUBIN integration, and foreign allocation ownership. Its `cudarc_interop.rs` example uses a stand-in allocation owner, not an actual cudarc integration test. The 2026-09-10 probe verifies the ownership half of this on Engine's actual libraries (see below); the kernel half remains unverified.
- cuTile has attention, RMSNorm, softmax, GEMM, asynchronous execution, graph, and JIT disk-cache examples. Availability is not proof of compatibility with Engine's shapes, state layouts, failure policy, or graph mode.
- cuda-oxide exposes `dp4a` intrinsics, warp operations, shared memory, and barriers. These match mechanisms used by Engine's existing quantized kernels; generated code and arithmetic still require validation.
- cuTile source says sm8x support starts with CUDA 13.2, with CUDA 13.3 recommended. This is more specific than the introductory blog's blanket 13.3 requirement. Check the actual qualification host rather than inferring its toolkit from cudarc's `cuda-13020` feature.
- cuda-oxide source currently pins nightly-2026-08-28 and lists CUDA 13.x host requirements, differing from the blog's older setup. Isolate kernel compilation/toolchain requirements from non-CUDA workspace development where practical; prove the build arrangement rather than assuming a nightly backend invalidates stable core code.

That review is now backed by a hardware probe on the qualification host (below). At that initial review no Engine kernel had been authored in Rust; gate 1 below subsequently added smoke kernels. Full-model CUDA Rust execution has not been measured on Qwen3.8-27B Q4 GGUF; published results on other models and GPUs do not transfer to this artifact or device.

## Hardware probe on the qualification host, 2026-09-10

The RTX 4090 host runs Fedora 44 with driver 615.71.09 (CUDA UMD 13.4), an NVIDIA-runfile toolkit 13.1 at `/usr/local/cuda-13.1` (with `/usr/local/cuda` pointing at it), rustup stable plus nightly, and the pinned Qwen3.8-27B Q4 artifact present. Both upstream trees were cloned at the reviewed revisions and checked out there: cuTile Rust `2eed75e`, cuda-oxide `26754ae` — the same commits the table above was reviewed against, so those citations are current rather than stale.

**The SIMT track runs on the existing toolkit.** `cargo oxide doctor` reports every requirement satisfied on CUDA 13.1: `cuda.h`, nvcc 13.1.80, libNVVM 2.0, nvJitLink 13.1, `libdevice.10.bc`, `llc` from the pinned nightly's rustlib, the clang 22 resource directory, the RTX 4090 at compute capability 8.9, and both optional debuggers. `cargo oxide setup` then built and published the `librustc_codegen_cuda.so` backend for revision `26754ae52c`, and `cargo oxide run vecadd` compiled and executed a Rust-authored kernel on the 4090 with all 1024 elements correct. cuda-oxide needs CUDA 13.0+ only; a toolkit upgrade is not on its critical path.

**The tile track needs a newer toolkit before it can run on this GPU.** With `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.1`, the cuTile `hello_world` example generates Tile IR and then fails at `tileiras` discovery with the compiler's own 13.2 floor:

```text
ERROR cuTile requires CUDA 13.2 or newer: the resolved toolkit at /usr/local/cuda-13.1 is CUDA 13.1. Set CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA 13.2+ install (the shared CUDA host-side crates themselves support 13.0+).
```

Three independent checks agree, so this is a real floor rather than a message artifact: `cutile-compiler` sets `MIN_TILE_CUDA_VERSION = 13020` and asserts that exact 13.1 diagnostic in its own tests; the 13.1 `tileiras --help` lists only `sm_100`, `sm_103`, `sm_110`, `sm_120`, and `sm_121` for `--gpu-name`, so it cannot target the 4090's `sm_89`; and the cutile-rs README places `sm_8x` support at CUDA 13.2. `cuda-toolkit-13-3` was therefore installed from the host's already-configured NVIDIA `cuda-fedora44.repo`; it pulls toolkit components only — compiler, libraries, tools, NVML, documentation — and no driver package, so the driver and the 13.1 tree were untouched. CUDA 13.3 is the version cutile-rs recommends and the one `cuda-core`'s default toolkit candidate list names. Its `tileiras` (13.3.36) lists `sm_80` through `sm_121`, including `sm_89`, and the cuTile `hello_world` example then ran on the 4090 and printed `Hello, I am program <0, 0, 0> in a kernel with <1, 1, 1> programs.` One side effect to know about: the package repoints `/usr/local/cuda` from 13.1 to 13.3, so anything that must resolve a specific toolkit should set `CUDA_TOOLKIT_PATH` rather than inherit the symlink.

**One shared substrate, confirmed by dependency direction rather than by convention.** `cuda-bindings`, `cuda-core`, and `cuda-async` are published from NVlabs/cutile-rs at 0.3.1 (crates.io, 2026-09-04); cuda-oxide 0.3.1 depends on those crates.io versions and keeps its SIMT surface in their `simt` modules. cuda-oxide's kernel-authoring crates (`cuda-device`, `cuda-macros`, `cuda-host`, `cuda-artifact-finalizer`) are not published, so any Engine kernel-authoring proof pins the cuda-oxide revision. The shared layer also reuses cuda-oxide's published `oxide-artifacts` 0.2.1 for artifact loading. There is no second host runtime to reconcile and no allocation bridge to invent between the two tracks.

**The resource-owner decision is constrained by what cuda-core can borrow.** Its `runtime` layer exposes deliberately non-owning foreign adoption: `Device::borrow_raw` / `borrow_with_owner`, `Stream::borrow_raw` / `borrow_with_owner`, and module/function `borrow_raw`, documented for external frameworks including cudarc. The `simt` layer that both kernel tracks launch through is different: the only raw-construction path for `DeviceBuffer` is `from_raw_parts`, whose contract transfers ownership and frees the pointer on drop, and `CudaStream` has no foreign-handle constructor at all. So Engine cannot adopt a cudarc allocation into cuda-core without handing over ownership, and should not keep two live wrappers over one allocation. The coherent target is the one the migration already names: make cuda-core the single context/stream/buffer owner, and let any remaining legacy code receive borrowed raw handles until it is retired.

The tile track is the exception worth knowing about, because it is where the borrowed direction is actually supported: cuTile's `Tensor::from_foreign` is documented as its preferred interop entry point, takes a foreign owner token, and holds it alive "so the memory provably outlives every use — no copy, no ownership transfer", naming cudarc and torch as the intended owners. So a tile kernel can run over Engine-owned memory without taking it over, while a SIMT kernel launch through cuda-core's `simt` layer cannot. That asymmetry is what the next probe increment should exercise: `cuda-rust-probe/interop` proves the ownership directions with raw driver work, and an Engine-authored kernel probe should prove the same with a real kernel — `Tensor::from_foreign` for the tile track, and cuda-core-owned memory for the SIMT track.

## Ordered proof and migration gates

### 1. Reproducible build and resource interoperability

Pin compatible upstream revisions and verify the Linux GPU host, toolkit, driver, and Rust toolchains. Build and run both tracks on the 4090. Demonstrate operations over the same context, stream, and allocation without a host round trip, double ownership, or premature release. Inspect compatibility between the shared runtime versions rather than assuming matching crate names imply compatibility.

Deliverable: a repeatable hardware-gated smoke test, explicit toolchain setup, and a decision on the common resource owner. Non-CUDA builds must remain usable without the NVIDIA toolchain. Do not upgrade the host or disrupt another GPU workload as an incidental setup action.

Status (2026-09-10): gate 1 is met.

- The pinned revisions are cloned on the host: cuTile Rust `2eed75e`, cuda-oxide `26754ae`. cuda-oxide's kernel-authoring crates are pinned by revision in `cuda-rust-probe/simt`; `cuda-core`, `cuda-bindings`, and `cuda-async` come from their published 0.3.1 release.
- Toolchain: cuda-oxide's SIMT track builds and runs on the pre-existing CUDA 13.1, and the tile track needed `cuda-toolkit-13-3` installed beside it. 13.3 is pinned explicitly through `CUDA_TOOLKIT_PATH`, because the install repoints `/usr/local/cuda`.
- Both tracks run on the 4090. `cuda-rust-probe/simt` authors a kernel through cuda-oxide and runs it over cuda-core-owned buffers; `cuda-rust-probe/tile` authors a cutile kernel that writes into a cuda-core allocation it borrowed through `Tensor::from_foreign`.
- One deallocation owner, in both directions. `cuda-rust-probe/interop` shows cuda-core owning while cudarc's raw bindings write over the borrowed stream, and cudarc owning while cuda-core adopts all three handles and releases them before the owner. Every probe reports `0 bytes leaked in 0 allocations` and `0 errors` under `compute-sanitizer --tool memcheck --leak-check full`, and the tile probe asserts the borrow is released with `Arc::try_unwrap` before the allocation is freed.
- The smoke test is repeatable: `cuda-rust-probe/run.sh` builds, runs, and sanitizes all three probes, refuses to start while another compute process holds the device, and prints per-probe timings. Cold preparation (`COLD=1`) costs 0.43s against 0.38s warm for one small cutile kernel — a real but small effect that says nothing about a model's worth of kernels.
- Non-CUDA builds stay independent: `cuda-rust-probe` is excluded from the repository workspace, and `cargo fmt --check` plus `cargo test --workspace` pass on a machine with no NVIDIA toolchain at all.

### 2. Representative quantized and stateful execution

Prove the chain Q8_1 packing → Q4_K integer-dot projection using actual encoded layout and batch-1/batched shapes. Check rounding, signed packed dots, alignment, scale/min corrections, and tail/rejection cases against the existing tests and host references. Q8_1 packing is a warp-reduction/packing operation, not merely elementwise addition.

Separately prove a batched GDN state update, including distinct per-request persistent matrices, output strides, and repeated updates. Resolve how tile partitioning expresses those independently owned allocations; do not assume a contiguous output tensor represents the complete state boundary.

Deliverable: numerical tests, generated-code inspection for packed integer dots, and matched kernel timings. A passing easy kernel alone is not the adoption gate.

Record upstream/toolchain revisions, device, actual encoded layouts, tested shapes and
rejection/tail cases, independent tolerances, cold/warm preparation, repeated timings,
failures and unsupported scope. Use unchanged algorithms first to distinguish compiler
or integration effects from algorithm changes. Qualify both quantized projection and
persistent state before promoting the gate; if one is blocked, retain that distinction
rather than declaring the whole migration ready.

#### Partial gate-2 evidence (2026-09-22, `39b44cb`)

[`cuda-rust-probe/gate2`](../cuda-rust-probe/gate2/README.md) now runs real
Rust-authored packing, projection and persistent-state kernels. **Gate 2 remains
open; no production operation has been replaced.** The probe README enumerates
covered shapes and remaining tests, rather than implying the smoke gate has grown
into model qualification.

Host: RTX 4090, driver 615.71.09, toolkit 13.3.73, sanitizer 2026.2.1.0;
nightly-2026-08-28 (`rustc 1.100.0-nightly e457a7b0d`), cuda-oxide
`26754ae52c26c097dc1c465a1e42c4c5d05a3d40`, cuda-core 0.3.1. The probe lock file
pins dependencies; the normal workspace retains its own toolchain. The desktop
checkout was `4d0a627` plus the exact probe sources committed as `39b44cb`.
No other compute process occupied the GPU; existing SIMT artifacts were preserved.

Commands from `cuda-rust-probe/gate2`, all exit 0:

```sh
./run.sh
cargo clippy --locked --all-targets -- -D warnings
BENCH=1 target/release/gate2
```

Results:

- Two host assertions pass (packing geometry rejection and literal tie rounding).
- Q8_1 matches every host/C++ packed word at 1/3/4/5/8/129 blocks.
- Device-to-device Q8_1 → Q4_K matches C++ output bits and independently accumulated
  f64 packed arithmetic at `(K,N)` = `(256,1)`, `(768,5)`, `(4096,128)`,
  `(5120,5120)`, each with M=1/3/8. Signed packed dots, six-bit metadata,
  minimum corrections, batch strides and whole-warp tails are exercised. A follow-up
  on the same date adds exact packed-word checking within this chain and the existing
  input-derived bound against unquantized float-input arithmetic; all shapes pass
  that additional check and memcheck remains clean.
- GDN state/output bits match C++ over eight updates of independently allocated
  request matrices, including inactive-pad preservation, at the three geometries
  in the probe README. This is differential evidence, not yet independent GDN
  numerical acceptance or qualification at the full model's head geometry.
- Memcheck: **0 errors, 0 bytes leaked in 0 allocations**. Emitted PTX contains
  `dp4a.s32.s32`. Local storage is also present; final SASS/register/spill inspection
  remains open. Do not infer occupancy or memory-traffic causes from these timings.
- Root boundaries, formatting, workspace tests and default/CUDA-feature clippy pass.
  The pinned device nightly warns about a pre-existing core `fetch_update`
  deprecation; the normal workspace toolchain is unchanged.

Matched projection timing: CUDA events around 100 launches per sample, seven paired
samples with alternating variant order, preallocated buffers, resident encoded
weights and packed input, no packing/transfers/preparation inside the interval.
Both sides use the **batched** entrypoint, including M=1. Event intervals may include
host launch gaps. Times are microseconds per launch, median [min, max]:

| K=N=5120 | Rust | C++ |
| --- | --- | --- |
| M=1 | 48.693 [48.668, 48.705] | 46.600 [46.582, 46.610] |
| M=3 | 63.406 [62.484, 67.063] | 77.639 [72.140, 77.670] |
| M=8 | 158.474 [158.415, 158.987] | 149.217 [149.128, 149.431] |

An initial translation without explicit unroll requests measured medians
62.564/78.812/168.223 µs at M=1/3/8; requesting the same fixed-loop unrolling as
C++ reduced those measured times. This is a bounded implementation experiment,
not a compiler rewrite or proof of its low-level cause. Results remain mixed:
Rust loses at M=1 and M=8. No performance promotion is justified.

A subsequent same-date qualification increment adds an independent row-major f64
GDN recurrence, with f32 rounding envelopes propagated through the whole history.
The envelope accounts for addition/multiplication, ordinary FMA contraction, and
[PTX's rsqrt relative-error bound](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt);
it is input-derived, not fitted to measured differences. Its domain is finite
normal/zero fixture inputs without overflow or arbitrary reassociation, not general
subnormal operand flushing. Five host tests now pass, including hand-calculated
state recurrence, multi-head/member/value-offset indexing and fused cancellation.
Device acceptance passes for the previous shapes plus `(M,VH,KH,D)` =
`(3,48,16,128)` and `(8,48,16,128)`, both with V_OFFSET=4096. Each case uses eight
varied updates; `(3,4,2,32)` uses 64. Inputs include normalized q/k, gate endpoints,
zero/nonzero initial states and swapped live allocation slots. Every state/output
also remains bit-exact against C++; inactive pads remain unchanged. These are
synthetic arithmetic histories, not full-model qualification.

The updated `run.sh` passes memcheck (zero errors/leaks), initcheck and synccheck
(zero errors each) on these fixtures. This does not prove global-memory race freedom
or the gate-3 cancellation/uncertain-enqueue ownership contract. Root checks and probe
clippy pass. Offline `ptxas -arch=sm_89 -v gate2.ptx` plus `cuobjdump --dump-sass`
reports packing/projection/state at 23/37/36 registers and 0/72/64 stack bytes,
respectively, with zero reported register spill bytes. Projection SASS contains
integer dots and local loads/stores; state also contains a local load. Stack/local
storage is not the same as register spilling. This offline artifact is not the
actual driver-JIT image, and does not establish measured occupancy or a timing cause.

The broader timing/preparation follow-up below supersedes this increment's open
measurement items. Full rejection/sentinel coverage, actual driver-loaded-code
inspection and cuTile state partitioning remain unqualified. Gate 3 remains unstarted.

#### Gate-2 candidate deferred (2026-09-22)

The same pinned host/toolchain now measures packing, projection against both C++
entrypoints, and GDN through the existing C++ operations owner. Seven alternating
paired samples each contain 100 launches on preallocated resident buffers, with CUDA
events excluding preparation/transfers. Short intervals can include host launch gaps.
GDN timing repeats the final input 700 times; both states still match C++ bits afterward,
but the independent f64 qualification covers the varied histories, not those extra
700 timing updates. These are isolated kernels, not serving throughput results.

Microseconds per launch, median [min, max]:

| Operation/shape | Rust | C++ |
| --- | --- | --- |
| Packing, 160 blocks | 1.820 [1.812, 1.838] | 1.770 [1.765, 1.772] |
| Packing, 480 blocks | 1.831 [1.825, 1.839] | 1.812 [1.809, 1.820] |
| Packing, 1280 blocks | 1.976 [1.967, 1.982] | 1.904 [1.867, 1.905] |
| Projection 5120×5120, M=1, C++ batch | 48.842 [48.831, 48.871] | 46.715 [46.705, 46.756] |
| Projection 5120×5120, M=1, C++ single | 48.722 [48.691, 48.852] | 25.006 [24.996, 25.078] |
| Projection 5120×5120, M=3 | 62.788 [62.700, 62.884] | 72.469 [72.448, 72.488] |
| Projection 5120×5120, M=8 | 158.812 [158.761, 158.874] | 149.363 [149.288, 149.432] |
| GDN M=1, VH/KH/D=2/1/16 | 5.601 [5.591, 5.608] | 4.045 [4.044, 4.052] |
| GDN M=3, VH/KH/D=4/2/32 | 9.329 [9.325, 9.339] | 6.346 [6.338, 6.352] |
| GDN M=8, VH/KH/D=4/2/128 | 31.764 [31.764, 31.785] | 20.274 [20.256, 20.289] |
| GDN M=3, VH/KH/D=48/16/128 | 44.460 [44.443, 44.507] | 31.622 [31.601, 31.631] |
| GDN M=8, VH/KH/D=48/16/128 | 45.771 [45.763, 45.852] | 33.609 [33.594, 33.649] |

Preparation measures native wrapper construction: Rust embedded module load plus
entrypoint preparation versus C++ constructor compilation/loading/function lookup.
Contexts already exist; Rust AOT compilation and context initialization are excluded.
Each of seven fresh processes performs one first construction plus seven repeats
on the same context, dropping the wrapper after each measurement. Repeats are **not**
per-request preparation on an already-retained wrapper. The C++ state constructor
prepares the full Qwen operations bundle, whereas Rust loads the probe bundle: these
numbers cannot establish a whole-model startup speedup.

Median milliseconds (7 first-call / 49 repeat observations per row and cache mode):

| Wrapper | Driver cache disabled: first / repeat | Populated default cache: first / repeat |
| --- | --- | --- |
| Rust packing | 29.961 / 26.120 | 0.496 / 0.094 |
| C++ packing | 23.933 / 19.464 | 9.576 / 4.802 |
| Rust projection | 29.896 / 26.086 | 0.483 / 0.095 |
| C++ projection | 136.152 / 50.806 | 10.392 / 0.068 |
| Rust state | 29.952 / 26.091 | 0.516 / 0.101 |
| C++ state | 580.873 / 577.572 | 16.340 / 10.873 |

Reproduce with `BENCH=1 cargo oxide run`. For each component `pack`, `projection`,
`state`, run `PREP_BENCH=<component> target/release/gate2` seven times in fresh processes,
first with `CUDA_CACHE_DISABLE=1`, then with that variable unset after normal runs have
populated the default cache. The mode prints every first/repeat sample. Check occupancy
before every process. The normal `run.sh` clears benchmark modes so they cannot bypass
qualification or accidentally time sanitized execution.

**Decision:** defer this candidate rather than expand it into production or an
open-ended compiler effort. The general Rust projection is about 1.95× the specialized
C++ single-row time, and production-geometry GDN is about 1.36–1.41× the C++ time.
The M=3 projection win and favorable cached preparation cases do not justify gate
promotion across this scope. This diagnoses the current implementation, not CUDA Rust
as a language or future upstream potential. Static single-row specialization is a
concrete re-entry experiment; local-storage/occupancy explanations remain hypotheses,
not measured causes. Keep the C++ path and continue inference-engine work.

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
