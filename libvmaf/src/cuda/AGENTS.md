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

## Governing ADRs

- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — CUDA execution provider mapping.
- [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md) — CUDA 13.2 + experimental flags.
- [ADR-0131](../../../docs/adr/0131-port-netflix-1382-cumemfree.md) —
  `vmaf_cuda_picture_free` synchronous free.

## Build

```bash
meson setup build -Denable_cuda=true -Denable_sycl=false
ninja -C build
```

Requires `/opt/cuda` + `nvcc` on PATH.
