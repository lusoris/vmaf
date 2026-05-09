# AGENTS.md — libvmaf/src/feature/cuda

Orientation for agents working on per-feature CUDA kernels (host
glue + `.cu` device code). Parent: [../AGENTS.md](../AGENTS.md). The
backend runtime (context, stream, picture-pool) lives one level up
in [`../../cuda/AGENTS.md`](../../cuda/AGENTS.md).

## Scope

```text
feature/cuda/
  <feature>_cuda.{c,h}        # host glue: registration, submit/collect, kernel-template wiring
  <feature>/                  # subdirectory of `.cu` device code (where the host glue is non-trivial)
    *.cu                      # CUDA kernel TUs (compiled with nvcc)
    *.cuh                     # device-side helpers (included from .cu only)
```

Examples: `integer_psnr_cuda.c` is a single-file consumer using the
kernel-template flat shape; `integer_adm/` is a multi-`.cu` consumer
because ADM splits across DWT2 + decouple + CSF + CM passes.

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md) +
  [../../AGENTS.md](../../AGENTS.md) +
  [`../../cuda/AGENTS.md`](../../cuda/AGENTS.md)).
- **Wholly-new fork files use the dual Netflix + Lusoris/Claude
  copyright header** per [ADR-0025](../../../../docs/adr/0025-copyright-handling-dual-notice.md).
  Many TUs here predate the dual-notice rule and carry only the
  Netflix header (with NVIDIA contributor lines on the `integer_adm/`
  CUDA kernels) — that is correct for upstream-mirrored files; do
  not retro-fit.
- **`#include` order** mirrors the SYCL / Vulkan twins:
  `feature_collector.h` / `feature_extractor.h` first, then
  `cuda/integer_<feature>_cuda.h`, then `cuda_helper.cuh` /
  `kernel_template.h`. Don't shuffle.
- **fmaf contraction is OFF for precision-critical kernels.** The
  parent build line passes `--fmad=false` to `nvcc` for feature
  TUs that participate in cross-backend gates with `places=4`.
  Removing it drifts `float_adm_cuda` /  `ssimulacra2_cuda` past
  the gate (mirror of the SYCL `-fp-model=precise` and Vulkan
  GLSL `precise` / `NoContraction` rules). On rebase: keep the
  flag.

## Twin-update rules

Every TU in this directory has at least one cross-backend twin.
A change to one twin **must** ship with the matching change(s) in
the same PR:

| Feature | Twins |
| --- | --- |
| **psnr** | `integer_psnr_cuda.c` ↔ `../sycl/integer_psnr_sycl.cpp` ↔ `../vulkan/psnr_vulkan.c` (+ `psnr.comp`) ↔ `../hip/integer_psnr_hip.c` |
| **ciede** | `integer_ciede_cuda.c` ↔ `../sycl/integer_ciede_sycl.cpp` ↔ `../vulkan/ciede_vulkan.c` (+ `ciede.comp`) ↔ `../hip/ciede_hip.c` |
| **moment** | `integer_moment_cuda.c` ↔ `../sycl/integer_moment_sycl.cpp` ↔ `../vulkan/moment_vulkan.c` (+ `moment.comp`) ↔ `../hip/float_moment_hip.c` |
| **motion** | `integer_motion_cuda.c` ↔ `../sycl/integer_motion_sycl.cpp` ↔ `../vulkan/motion_vulkan.c` (+ `motion.comp`) |
| **motion_v2** | `integer_motion_v2_cuda.c` ↔ `../sycl/integer_motion_v2_sycl.cpp` ↔ `../vulkan/motion_v2_vulkan.c` (+ `motion_v2.comp`) ↔ `../hip/integer_motion_v2_hip.c` |
| **vif (integer)** | `integer_vif_cuda.c` (+ `integer_vif/filter1d.cu`) ↔ `../sycl/integer_vif_sycl.cpp` ↔ `../vulkan/vif_vulkan.c` (+ `vif.comp`) |
| **adm (integer)** | `integer_adm_cuda.c` (+ `integer_adm/*.cu`) ↔ `../sycl/integer_adm_sycl.cpp` ↔ `../vulkan/adm_vulkan.c` (+ `adm.comp`) |
| **ssim** | `integer_ssim_cuda.c` ↔ `../sycl/integer_ssim_sycl.cpp` ↔ `../vulkan/ssim_vulkan.c` (+ `ssim.comp`) |
| **ms_ssim** | `integer_ms_ssim_cuda.c` ↔ `../sycl/integer_ms_ssim_sycl.cpp` ↔ `../vulkan/ms_ssim_vulkan.c` (+ `ms_ssim.comp`) |
| **psnr_hvs** | `integer_psnr_hvs_cuda.c` ↔ `../sycl/integer_psnr_hvs_sycl.cpp` ↔ `../vulkan/psnr_hvs_vulkan.c` (+ `psnr_hvs.comp`) |
| **ssimulacra2** | `ssimulacra2_cuda.c` (+ `ssimulacra2/*.cu`) ↔ `../sycl/ssimulacra2_sycl.cpp` ↔ `../vulkan/ssimulacra2_vulkan.c` (+ `ssimulacra2_*.comp`) |
| **float_*** | `float_adm_cuda.c` / `float_ansnr_cuda.c` / `float_motion_cuda.c` / `float_psnr_cuda.c` / `float_vif_cuda.c` ↔ matching `../sycl/float_*_sycl.cpp` ↔ `../vulkan/float_*_vulkan.c` ↔ partial `../hip/float_*_hip.c` |
| **cambi** | `integer_cambi_cuda.c` (+ `integer_cambi/cambi_score.cu`) ↔ `../vulkan/cambi_vulkan.c` (+ `cambi_*.comp`) — Strategy II hybrid twin. SYCL twin pending (T3-15b). |

The full GPU twin matrix is governed by the GPU long-tail batches:
[ADR-0182](../../../../docs/adr/0182-gpu-long-tail-batch-1.md) (psnr /
ciede / moment), [ADR-0188](../../../../docs/adr/0188-gpu-long-tail-batch-2.md)
(ssim / ms_ssim / psnr_hvs), [ADR-0192](../../../../docs/adr/0192-gpu-long-tail-batch-3.md)
(motion_v2 / float_ansnr / float-twins / ssimulacra2 / cambi).

## Rebase-sensitive invariants

- **`integer_ms_ssim_cuda.c::extract_metrics_*` honours the
  `enable_lcs` GPU contract** (ADR-0243). Emits 15 extra metrics
  (`float_ms_ssim_{l,c,s}_scale{0..4}`) when `enable_lcs=true`,
  matching the CPU `float_ms_ssim` extractor metric-wise (all
  `l_scale*` first, then `c_*`, then `s_*`). Renaming or
  reordering breaks the public API surface and the cross-backend
  parity gate. See [../../AGENTS.md §"MS-SSIM `enable_lcs` GPU
  contract"](../../AGENTS.md).

- **`integer_motion_cuda.c::motion3_postprocess_*` honours the
  motion3 GPU contract** (ADR-0219). Applies CPU's host-side
  post-process to motion2 with no device-side state. Two
  invariants flow: (1) `motion_five_frame_window=true` returns
  `-ENOTSUP` at `init()`; (2) any change to `motion_blend()` /
  `motion_max_val` / moving-average must mirror across the three
  GPU motion twins in the same PR. See [../../AGENTS.md §"motion3_score
  GPU contract"](../../AGENTS.md).

- **`integer_ms_ssim_cuda.c` and `integer_ssim_cuda.c` pass
  `channel=0` to `picture_copy()`** per the upstream
  d3647c73 prerequisite port. If a future upstream commit
  evolves the signature further, update these call sites in
  lockstep with the upstream-mirror callers (`float_*` series).
  See [../../AGENTS.md §"`picture_copy()` carries a `channel`
  parameter"](../../AGENTS.md).

- **`integer_cambi_cuda.c` + `integer_cambi/cambi_score.cu` are
  Strategy II hybrid** (ADR-0360 / T3-15a). The GPU kernels
  (`cambi_spatial_mask_kernel`, `cambi_decimate_kernel`,
  `cambi_filter_mode_kernel`) are bit-exact w.r.t. the CPU
  implementation. The host residual calls `vmaf_cambi_calculate_c_values`
  + `vmaf_cambi_spatial_pooling` via `cambi_internal.h`. If upstream
  Netflix refactors `cambi.c` and renames those entry points,
  `cambi_internal.h` **and** `cambi_vulkan.c` must be updated in the
  same PR. Never remove the `cuStreamSynchronize` calls inside
  `submit_fex_cuda` — they guard the DtoH coherency for the host
  residual. `places=4` gate is load-bearing; do not loosen it.

- **`integer_adm/adm_cm.cu` (and the rest of the `integer_adm/`
  subdirectory) carries an NVIDIA copyright line** alongside the
  Netflix one. This is upstream-mirror — keep both headers
  verbatim on rebase.

- **`kernel_template.h` mirror with HIP** (ADR-0241). The CUDA
  `cuda/kernel_template.h` (one level up) and HIP
  `../hip/kernel_template.h` move in lockstep. Any change to
  the CUDA template's struct fields, helper signatures, or
  semantics requires a paired HIP change in the same PR.
  Consumers of the template (`integer_psnr_cuda.c` and
  follow-on `integer_ciede_cuda.c` / `integer_moment_cuda.c` /
  ...) lock the HIP twins call-graph-for-call-graph; see
  [`../../hip/AGENTS.md`](../../hip/AGENTS.md) for the full
  consumer list.

## Build

CUDA feature TUs compile only when `meson setup -Denable_cuda=true`.
The `enable_cuda` umbrella flag gates inclusion via
`#if HAVE_CUDA` blocks in `feature/feature_extractor.c`.

## Governing ADRs

- [ADR-0182](../../../../docs/adr/0182-gpu-long-tail-batch-1.md) +
  [ADR-0188](../../../../docs/adr/0188-gpu-long-tail-batch-2.md) +
  [ADR-0192](../../../../docs/adr/0192-gpu-long-tail-batch-3.md) —
  GPU long-tail batches. Every CUDA feature kernel here corresponds
  to a row in one of these.
- [ADR-0214](../../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate.
- [ADR-0219](../../../../docs/adr/0219-motion3-gpu-contract.md) —
  motion3 GPU contract.
- [ADR-0241](../../../../docs/adr/0241-hip-first-consumer-psnr.md) —
  kernel-template mirror between CUDA and HIP.
- [ADR-0243](../../../../docs/adr/0243-enable-lcs-gpu.md) — MS-SSIM
  `enable_lcs` GPU contract.
- [ADR-0246](../../../../docs/adr/0246-cuda-kernel-template-feature.md) —
  per-feature CUDA kernel-template scaffolding.
- [ADR-0360](../../../../docs/adr/0360-cambi-cuda.md) —
  CAMBI CUDA port (Strategy II hybrid, T3-15a).
