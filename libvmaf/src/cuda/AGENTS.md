# AGENTS.md — libvmaf/src/cuda

Orientation for agents working on the CUDA backend runtime. Parent:
[../../AGENTS.md](../../AGENTS.md).

## Scope

The CUDA-side runtime (picture lifecycle, buffer pools, launch helpers).
CUDA **feature kernels** live one level deeper in
[../feature/cuda/](../feature/cuda/). CUDA execution-provider wiring for
ONNX Runtime lives in [../dnn/](dnn/AGENTS.md).

```
cuda/
  common.c/.h          # CUDA context + stream management
  cuda_helper.cuh      # launch macros, error-check, types
  picture_cuda.c/.h    # VmafPicture on a CUDA device
  ring_buffer.c/.h     # picture buffer pool
```

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **Every CUDA call has its error checked.** Use the `cuda_helper.cuh`
  macros; don't invent new wrappers silently.
- **`cudaMemcpyAsync` requires pinned host memory** for true async; using
  pageable host buffers silently serialises. Picture pools allocate pinned.
- **Experimental toolchain flags are enabled** (`--expt-relaxed-constexpr`,
  `--extended-lambda`, `--expt-extended-lambda`; Blackwell `sm_120` gencode).
  See [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md).
  "Experimental" means *feature flags on stable CUDA ≥13.2*, not preview
  branches.
- **Numerical snapshots**: kernels that cannot bit-match the CPU reference
  regenerate `testdata/scores_cpu_cuda.json` via
  [`/regen-snapshots`](../../../.claude/skills/regen-snapshots/SKILL.md)
  with a justification in the commit message. See
  [CLAUDE.md §9](../../../CLAUDE.md).

## Rebase-sensitive invariants

- **`picture_cuda.c` synchronous free**: `vmaf_cuda_picture_free`
  deliberately calls `cuMemFree` — *not* `cuMemFreeAsync` — because
  the previous `cuStreamSynchronize` already drains any pending work
  and the stream handle is about to be destroyed, which made the
  async variant assert (`Assertion 0 failed`) with two or more
  concurrent CUDA sessions. Upstream carries this fix in
  [Netflix#1382](https://github.com/Netflix/vmaf/pull/1382), still
  OPEN as of 2026-04-20. If a rebase reintroduces `cuMemFreeAsync`
  here — whether from an upstream merge of #1382 (unlikely to
  conflict, this is the same substance) or a refactor that "restores
  async symmetry" — keep the synchronous form. The async variant is
  a multi-session data hazard, not a perf optimisation. Tracker:
  [Netflix#1381](https://github.com/Netflix/vmaf/issues/1381). See
  [ADR-0131](../../../docs/adr/0131-port-netflix-1382-cumemfree.md)
  and [rebase-notes 0031](../../../docs/rebase-notes.md).

- **`vmaf_cuda_state_free()` ownership contract** (fork-local,
  ADR-0157): the public API at
  [`include/libvmaf/libvmaf_cuda.h`](../../include/libvmaf/libvmaf_cuda.h)
  now includes `vmaf_cuda_state_free(VmafCudaState *cu_state)`.
  Ownership model: `vmaf_cuda_state_init` allocates →
  `vmaf_cuda_import_state` copies-by-value (no ownership transfer)
  → `vmaf_close` destroys the CUDA stream + context + frees the
  `CudaFunctions` driver table (via fork-local
  `cuda_free_functions()` call in `vmaf_cuda_release`) →
  `vmaf_cuda_state_free` frees the heap allocation itself. The
  call order `vmaf_close → vmaf_cuda_state_free → vmaf_model_destroy`
  is load-bearing; reversing the first two is a
  use-after-free. Mirrors the SYCL backend's
  `vmaf_sycl_state_free()` pattern. **On rebase**: keep the
  fork's public symbol; upstream doesn't have this API as of
  2026-04-24. See
  [ADR-0157](../../../docs/adr/0157-cuda-preallocation-leak-netflix-1300.md)
  and [rebase-notes 0050](../../../docs/rebase-notes.md).
- **`vmaf_ring_buffer_close` mutex destroy order** (fork-local,
  ADR-0157): the function now does
  `pthread_mutex_unlock` → `pthread_mutex_destroy` → `free(pic)`
  → `free(ring_buffer)`. Destroying a locked mutex is POSIX UB;
  the old code destroyed it locked. On rebase: keep the
  unlock-before-destroy order.

- **`CHECK_CUDA` graceful error propagation** (fork-local,
  ADR-0156): the `CHECK_CUDA` macro in
  [`cuda_helper.cuh`](cuda_helper.cuh) does NOT call
  `assert(0)` on CUDA errors — it was replaced wholesale by
  `CHECK_CUDA_GOTO(funcs, CALL, label)` and
  `CHECK_CUDA_RETURN(funcs, CALL)` which log + return a
  `-errno` translated from `CUresult` via
  `vmaf_cuda_result_to_errno`. Every one of the 178 call
  sites across `common.c`, `picture_cuda.c`,
  `libvmaf.c`, and the three `feature/cuda/*.c` feature
  extractors uses the new macros. Twelve `static` helpers
  (`calculate_motion_score`, `filter1d_8/16`,
  `adm_dwt2_*_device`, `adm_csf_device`,
  `i4_adm_csf_device`, `adm_csf_den_*_device`,
  `adm_cm_device`, `i4_adm_cm_device`,
  `integer_compute_adm_cuda`) are `int`-returning to
  carry errors upward. **On rebase**: keep the fork's macro
  definitions and every cleanup-label pattern; upstream
  still uses `assert(0)` as of 2026-04-24. If an upstream
  port adds new `CHECK_CUDA(...)` sites, rewrite them to
  the graceful variants inside the port commit. See
  [ADR-0156](../../../docs/adr/0156-cuda-graceful-error-propagation-netflix-1420.md)
  and [rebase-notes 0049](../../../docs/rebase-notes.md).

## Per-kernel nvcc flag invariants

- `cuda_cu_extra_flags` map in `libvmaf/src/meson.build` routes
  per-kernel nvcc flags. Currently inhabited by `float_adm_score`
  (added in PR #157,
  [ADR-0202](../../../docs/adr/0202-float-adm-cuda-sycl.md)) and
  `ssimulacra2_blur` (added in
  [ADR-0206](../../../docs/adr/0206-ssimulacra2-cuda-sycl.md)).
  Both pass `-Xcompiler=-ffp-contract=off --fmad=false` so the
  recursive / cross-band float reductions keep their CPU-port
  FMUL/FSUB ordering. **On rebase**: never drop these per-kernel
  entries — without them, `float_adm` drifts past `places=4` at
  scale 3, and `ssimulacra2`'s pooled score drifts past `places=2`
  through the IIR + 6-scale pyramid.
- The matching `ssimulacra2_mul` fatbin is a single FMUL with no
  fused-add candidate and intentionally does **not** carry the
  flag — keeping FMA on the kernels where it isn't a precision risk
  preserves whatever optimisation NVCC can apply.

## Governing ADRs

- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — CUDA execution provider mapping.
- [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md) — CUDA 13.2 + experimental flags.
- [ADR-0131](../../../docs/adr/0131-port-netflix-1382-cumemfree.md) —
  `vmaf_cuda_picture_free` synchronous free.
- [ADR-0202](../../../docs/adr/0202-float-adm-cuda-sycl.md) —
  `float_adm_cuda` requires `--fmad=false` on its fatbin to
  match the GLSL `precise` qualifier in `float_adm.comp`. See
  the per-kernel `cuda_cu_extra_flags` dict in
  `libvmaf/src/meson.build`. **On rebase**: do not consolidate
  `float_adm_score` into the global `cuda_flags` block; the
  FMA-off scope is intentionally one fatbin only.

## Build

```bash
meson setup build -Denable_cuda=true -Denable_sycl=false
ninja -C build
```

Requires `/opt/cuda` + `nvcc` on PATH.
