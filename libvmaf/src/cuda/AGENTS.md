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

## Governing ADRs

- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — CUDA execution provider mapping.
- [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md) — CUDA 13.2 + experimental flags.

## Build

```bash
meson setup build -Denable_cuda=true -Denable_sycl=false
ninja -C build
```

Requires `/opt/cuda` + `nvcc` on PATH.
