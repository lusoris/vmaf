# AGENTS.md — libvmaf/src/hip

Orientation for agents working on the HIP (AMD ROCm) backend. Parent:
[../../AGENTS.md](../../AGENTS.md). Mirrors
[`libvmaf/src/cuda/AGENTS.md`](../cuda/AGENTS.md) — HIP and CUDA share
near-identical async-stream + event APIs, so the rebase / refactor
contracts on this side track CUDA's closely.

## Scope

The HIP-side runtime (picture lifecycle, dispatch strategy, kernel
scaffolding template). HIP **feature kernels** live one level deeper
in [../feature/hip/](../feature/hip/).

```text
hip/
  common.{c,h}          # HIP context + (future) stream management
  picture_hip.{c,h}     # VmafPicture on a HIP device — stub
  dispatch_strategy.{c,h} # Feature-name → kernel routing — stub
  kernel_template.{h,c} # per-feature HIP kernel scaffolding (T7-10 / ADR-0241)
  meson.build           # subdir() include from libvmaf/src/meson.build
```

## Backend status

**Scaffold + first consumer** — landed across two PRs:

1. **T7-10 audit-first scaffold** (ADR-0212) — common, picture, dispatch,
   feature stubs, public header `libvmaf_hip.h`, CI lane
   `Build — Ubuntu HIP (T7-10 scaffold)`, smoke-only `enable_hip` build.
   Every public C-API entry point returns `-ENOSYS`.
2. **T7-10 first consumer** (ADR-0241) — `kernel_template.{h,c}` (mirror
   of `cuda/kernel_template.h`) + `feature/hip/integer_psnr_hip.{c,h}`
   (first kernel-template consumer) + `vmaf_fex_psnr_hip` registration
   under `#if HAVE_HIP`. Template helpers and the consumer's
   submit/collect return `-ENOSYS` until T7-10b.

**Pending** — T7-10b (runtime PR) replaces the `kernel_template.c`
bodies and the consumer's submit/collect kernel-launch chain with
real HIP calls (`hipStreamCreate`, `hipMemcpyAsync`, ...). Remaining
kernel ports (ADM, VIF, motion, ...) follow as their own PRs gated
by the `places=4` cross-backend-diff lane (per [ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md)).

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **No ROCm SDK is currently linked**. The `meson.build` includes an
  optional `dependency('hip-lang', required: false)` probe; the
  scaffold compiles cleanly on stock Ubuntu without `hipcc` /
  `amdhip64`. The runtime PR (T7-10b) flips that to required.
- **HIP runtime types cross headers as `uintptr_t`**. The public
  header `libvmaf_hip.h` and the kernel template's `VmafHipKernelLifecycle`
  carry `uintptr_t` for `hipStream_t` / `hipEvent_t` so consumer TUs
  stay free of `<hip/hip_runtime.h>`. Cast on the implementation side
  only. Mirrors the Vulkan ADR-0184 pattern.
- **Don't drift the kernel-template field shape from CUDA**. The
  templates mirror each other field-for-field; the runtime PR is
  predicated on that mirror. See "Rebase-sensitive invariants" below.

## Rebase-sensitive invariants

- **`kernel_template.h` mirrors `cuda/kernel_template.h`** (fork-local,
  ADR-0241). The struct shapes (`VmafHipKernelLifecycle` ↔
  `VmafCudaKernelLifecycle`, `VmafHipKernelReadback` ↔
  `VmafCudaKernelReadback`) and the helper signatures
  (`vmaf_hip_kernel_lifecycle_init/_close`,
  `vmaf_hip_kernel_readback_alloc/_free`,
  `vmaf_hip_kernel_submit_pre_launch`,
  `vmaf_hip_kernel_collect_wait`) are deliberately one-to-one with
  the CUDA template. Any change to the CUDA template (helper
  signatures, struct fields, semantics) needs a paired HIP change
  in the same PR — otherwise the mirror drifts and consumer call
  sites diverge between the two backends. **On rebase / refactor**:
  if an upstream port or a fork PR touches `cuda/kernel_template.h`,
  walk the diff onto `hip/kernel_template.h` + `kernel_template.c`
  before merging. The HIP variant is out-of-line (`.c` paired with
  `.h`) instead of `static inline` for the reason documented in
  `kernel_template.h`'s preamble — keep the split until the runtime
  PR ships, then re-evaluate. See
  [ADR-0241](../../../docs/adr/0241-hip-first-consumer-psnr.md).

- **`integer_psnr_hip.c` mirrors `integer_psnr_cuda.c` call-graph-for-call-graph**
  (fork-local, ADR-0241). Same `PsnrStateHip`/`PsnrStateCuda` fields
  in the same order, same template-helper invocations in the same
  init/submit/collect/close sequence, same `provided_features`
  contract (`psnr_y` luma-only in v1). The runtime PR (T7-10b) flips
  the `kernel_template.c` bodies; the consumer's call sites stay
  verbatim. **On rebase**: keep the call graph aligned with the
  CUDA twin. If a future PR drifts the CUDA twin's lifecycle (e.g.
  adds a third event), update the HIP twin in the same PR.

- **`vmaf_fex_psnr_hip` is registered without the
  `VMAF_FEATURE_EXTRACTOR_HIP` flag bit set** (fork-local, ADR-0241).
  The flag bit (`1 << 6`) is reserved in the enum but the consumer
  does not set it, because the picture buffer-type check in
  `vmaf_feature_extractor_context_extract` would route a HIP-flagged
  extractor through a (not-yet-existing) HIP buffer-type branch. The
  runtime PR (T7-10b) adds the `VMAF_PICTURE_BUFFER_TYPE_HIP_DEVICE`
  tag and *then* sets the flag on the extractor. **On rebase**: if a
  refactor touches the picture buffer-type dispatch, leave the HIP
  extractor's flags at `0` until T7-10b.

## Governing ADRs

- [ADR-0212](../../../docs/adr/0212-hip-backend-scaffold.md) — HIP
  scaffold-only audit-first PR (T7-10).
- [ADR-0241](../../../docs/adr/0241-hip-first-consumer-psnr.md) —
  this PR: kernel-template mirror + `integer_psnr_hip` first consumer.
- [ADR-0221](../../../docs/adr/0221-gpu-kernel-template.md) — CUDA
  kernel-template decision; the source the HIP mirror tracks.
- [ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md) — `places=4`
  cross-backend gate; the runtime PR's incoming numerics gate.

## Build

```bash
meson setup build -Denable_hip=true -Denable_cuda=false -Denable_sycl=false libvmaf
ninja -C build
meson test -C build
```

The scaffold has zero hard runtime dependencies — no ROCm SDK
required. `Build — Ubuntu HIP (T7-10 scaffold)` in
`.github/workflows/libvmaf-build-matrix.yml` runs this exact
configuration on every PR.
