# AGENTS.md — libvmaf/src/feature/sycl

Orientation for agents working on per-feature SYCL kernels (DPC++).
Parent: [../AGENTS.md](../AGENTS.md). The backend runtime (queue, USM,
dmabuf import) lives one level up in
[`../../sycl/AGENTS.md`](../../sycl/AGENTS.md).

## Scope

```text
feature/sycl/
  <feature>_sycl.cpp           # one TU per kernel: registration + submit/collect + sycl::queue::submit lambda
```

All TUs are compiled with `icpx` (Intel oneAPI) — the build line
under [`../../meson.build`](../../meson.build) adds
`-fsycl -fp-model=precise` for every per-kernel TU.

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md) +
  [../../AGENTS.md](../../AGENTS.md) +
  [`../../sycl/AGENTS.md`](../../sycl/AGENTS.md)).
- **`-fp-model=precise` on the SYCL feature line is load-bearing.**
  Removing it allows `icpx` to FMA-contract inside the kernel
  lambdas, which drifts `float_adm_sycl` past `places=4` at scale 2
  (ADR-0202) and `ssimulacra2_sycl` past `places=2` through the IIR
  (ADR-0206). Matches GLSL `precise` / `NoContraction` and CUDA
  `--fmad=false`.
- **fp64-free kernels are non-negotiable** ([ADR-0220](../../../../docs/adr/0220-sycl-fp64-fallback.md)).
  Every SYCL feature-kernel lambda captures and operates on `float`
  / integer types only. **No `double` operand inside a `parallel_for`
  body**, no `sycl::reduction<double>`, no `sycl::plus<double>`.
  This is hard, not soft: a single fp64 instruction anywhere in the
  TU's SPIR-V module causes the Level Zero runtime to reject the
  entire module on Intel Arc A-series and other fp64-less devices,
  even when the offending kernel is never submitted.
  - `double` is allowed **outside** the kernel lambda — host-side
    post-processing in `extract` / `flush` callbacks, score
    aggregation, log10 normalisation.
  - ADM gain limiting uses int64 Q31 (`gain_limit_to_q31` +
    `launch_decouple_csf<false>` in `integer_adm_sycl.cpp`).
  - VIF gain limiting uses fp32 `sycl::fmin`.
- **Wholly-new fork files use the dual Netflix + Lusoris/Claude
  copyright header** per [ADR-0025](../../../../docs/adr/0025-copyright-handling-dual-notice.md).
  Most TUs in this directory are fork-original SYCL ports of
  Netflix CUDA kernels.

## Twin-update rules

Every SYCL TU in this directory has CUDA + Vulkan twins. The complete
table lives in [`../cuda/AGENTS.md`](../cuda/AGENTS.md); changes to a
SYCL TU **must** ship with matching changes to the CUDA + Vulkan
twins in the **same PR**. The cross-backend parity gate at `places=4`
([`scripts/ci/cross_backend_parity_gate.py`](../../../../scripts/ci/cross_backend_parity_gate.py),
ADR-0214) catches drift but only after a full GPU run.

## Rebase-sensitive invariants

- **`integer_motion_sycl.cpp::motion3_postprocess_*` honours the
  motion3 GPU contract** (ADR-0219). Applies CPU's host-side
  post-process to motion2 with no device-side state.
  `motion_five_frame_window=true` returns `-ENOTSUP` at `init()`.
  See [../../AGENTS.md §"motion3_score GPU contract"](../../AGENTS.md).

- **`integer_ms_ssim_sycl.cpp` honours the `enable_lcs` GPU
  contract** (ADR-0243). Emits 15 extra metrics
  (`float_ms_ssim_{l,c,s}_scale{0..4}`) when `enable_lcs=true`.
  Metric ordering and `places=4` cross-backend contract are part of
  the public API surface. See
  [../../AGENTS.md §"MS-SSIM `enable_lcs` GPU contract"](../../AGENTS.md).

- **`integer_ssim_sycl.cpp` and `integer_ms_ssim_sycl.cpp` are
  self-contained submit/collect** — they do **not** register with
  `vmaf_sycl_graph_register` because the shared `shared_frame` is
  luma-only packed at uint width and SSIM needs float [0, 255]
  intermediates with `picture_copy()` normalisation. The `ciede_sycl`
  TU follows the same pattern. **On rebase**: do not "consolidate"
  these into the graph register — the precision posture is
  load-bearing.

- **`picture_copy()` channel parameter** — `integer_ms_ssim_sycl.cpp`
  and `integer_ssim_sycl.cpp` pass `channel=0` per the d3647c73
  prerequisite port. See
  [../../AGENTS.md §"`picture_copy()` carries a `channel`
  parameter"](../../AGENTS.md).

- **`integer_cambi_sycl.cpp` — Strategy II hybrid: no graph register,
  synchronous per-scale loop** (T3-15 / ADR-0371). The `submit()` runs
  a synchronous per-scale loop: H2D upload → `launch_spatial_mask` →
  per-scale (`launch_decimate` + `launch_filter_mode` H + V → D2H →
  `vmaf_cambi_calculate_c_values` + `vmaf_cambi_spatial_pooling`). The
  CPU-residual phases must stay inside `submit()`, not `collect()`.
  `collect()` only emits `s->score`. Do **not** move the CPU residual
  into `collect()` and do **not** register with `vmaf_sycl_graph_register`
  — the per-scale D2H readback and host histogram pass are incompatible
  with the graph-replay model. The CUDA twin (ADR-0360) follows the same
  pattern; keep both in sync.

- **`integer_adm_sycl.cpp` / `float_adm_sycl.cpp` expose three ADM
  tuning parameters** (`adm_csf_scale`, `adm_csf_diag_scale`,
  `noise_weight`) with the same defaults as the CPU path (PR #731).
  If upstream Netflix adds or renames these parameters in
  `integer_adm.c` / `float_adm.c`, the SYCL twins must be updated
  in the same PR.

- **`motion_fps_weight` cross-backend parity** — see the canonical
  invariant note in [`../cuda/AGENTS.md`](../cuda/AGENTS.md).
  `integer_motion_v2_sycl.cpp` and `float_motion_sycl.cpp` both carry
  the `motion_fps_weight` option and apply it in `flush()` /
  `collect()` exactly as documented there. Any future change to the
  weight application math must span all motion-family GPU twins in
  the same PR.

- **VAAPI / dmabuf zero-copy import** — the FFmpeg `libvmaf_sycl`
  filter (`ffmpeg-patches/0005-*.patch`) consumes
  `vmaf_sycl_import_va_surface`. Public-surface change touches the
  patch file too — see CLAUDE.md §12 r14 +
  [ADR-0183](../../../../docs/adr/0183-ffmpeg-libvmaf-sycl-filter.md).

## icpx-aware clang-tidy

Stock LLVM `clang-tidy` cannot resolve `<sycl/sycl.hpp>`. Use
[`scripts/ci/clang-tidy-sycl.sh`](../../../../scripts/ci/clang-tidy-sycl.sh)
which injects the oneAPI SYCL include path +
`-D__SYCL_DEVICE_ONLY__=0` and locates `icpx` via `$ICPX_ROOT` (or
`/opt/intel/oneapi/compiler/latest`). The CI lane
`Clang-Tidy SYCL (Changed Files, Advisory)` runs the wrapper.
When adding a new SYCL TU, no AGENTS.md update is needed — the
wrapper finds it via the changed-file diff. See
[ADR-0217](../../../../docs/adr/0217-sycl-toolchain-cleanup.md).

## Build

SYCL feature TUs compile only when `meson setup -Denable_sycl=true`.
Requires oneAPI (`source /opt/intel/oneapi/setvars.sh`) or equivalent
DPC++ toolchain with `icpx` on PATH.

## Governing ADRs

- [ADR-0182](../../../../docs/adr/0182-gpu-long-tail-batch-1.md) +
  [ADR-0188](../../../../docs/adr/0188-gpu-long-tail-batch-2.md) +
  [ADR-0192](../../../../docs/adr/0192-gpu-long-tail-batch-3.md) —
  GPU long-tail batches. Every SYCL feature kernel here corresponds
  to a row in one of these.
- [ADR-0202](../../../../docs/adr/0202-float-adm-cuda-sycl.md) +
  [ADR-0206](../../../../docs/adr/0206-ssimulacra2-cuda-sycl.md) —
  CUDA + SYCL ports that pinned `-fp-model=precise` as load-bearing.
- [ADR-0214](../../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate.
- [ADR-0217](../../../../docs/adr/0217-sycl-toolchain-cleanup.md) —
  icpx-aware clang-tidy wrapper.
- [ADR-0219](../../../../docs/adr/0219-motion3-gpu-contract.md) —
  motion3 GPU contract.
- [ADR-0220](../../../../docs/adr/0220-sycl-fp64-fallback.md) — SYCL
  feature kernels are unconditionally fp64-free (T7-17).
- [ADR-0243](../../../../docs/adr/0243-enable-lcs-gpu.md) — MS-SSIM
  `enable_lcs` GPU contract.
