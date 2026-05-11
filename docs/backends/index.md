# Backends

libvmaf supports multiple compute backends for hardware-accelerated quality
assessment. Backends are **opt-in at build time** via meson options and
**selected per-invocation at runtime** through `vmaf` CLI flags or the C API.

| Backend | Meson option | Default-on? | Runtime opt-out | Status |
| --- | --- | --- | --- | --- |
| CPU scalar | always on | yes | n/a | stable |
| x86 AVX2 | auto-detected | yes, when host supports | `--cpumask` | stable |
| x86 AVX-512 | `-Denable_avx512=true` | build-time opt-in | `--cpumask` | stable |
| ARM NEON | auto-detected on aarch64 | yes | `--cpumask` | stable — see [arm/overview.md](arm/overview.md) |
| CUDA | `-Denable_cuda=true` | no | `--no_cuda` | stable — see [cuda/overview.md](cuda/overview.md) |
| SYCL / oneAPI | `-Denable_sycl=true` | no | `--no_sycl` / `--sycl_device N` | stable — see [sycl/overview.md](sycl/overview.md) |
| HIP (AMD) | `-Denable_hip=true` | no | n/a | 8/11 real kernels (T7-10b, batch 1–4) — see [hip/overview.md](hip/overview.md); 3 stubs (adm, vif, integer_motion) remain `-ENOSYS` |
| Metal (Apple Silicon) | `-Denable_metal=auto/enabled` | auto on macOS | n/a | 8/17 consumers registered (scaffold, T8-1) — see [metal/index.md](metal/index.md); all entry points return `-ENOSYS` until the runtime PR (ADR-0361 / T8-1b) |

## Runtime selection

Backend selection is **not** controlled by environment variables in this fork.
Backends are selected via CLI flags on `vmaf` (see
[../usage/cli.md](../usage/cli.md) — "Backend selection") or programmatically
through `VmafConfiguration` fields in the C API (`gpu_enable`, `cuda_state`,
`sycl_state`).

> `VMAF_FORCE_BACKEND` is **not** read by `libvmaf` — it appeared in earlier
> drafts of this page as a planned selection mechanism, but the implemented
> surface is CLI-flag-based. If you are scripting alternate backends, set
> `--no_cuda` / `--no_sycl` / `--sycl_device <N>` on the `vmaf` command line.

Dispatch precedence inside `libvmaf` (highest first):

1. User-disabled backends are removed from the candidate list
   (`--no_cuda` / `--no_sycl` / `--cpumask` ISA bits).
2. If a feature has a GPU kernel and a GPU backend survives the filter, the GPU
   path runs.
3. Otherwise the best available CPU SIMD twin runs; scalar C is the universal
   fallback.

Not every feature has every twin — the coverage matrix is in
[../metrics/features.md](../metrics/features.md) per feature and in each
per-backend page below.

## Guides

- [x86 SIMD (AVX2 / AVX-512)](x86/avx512.md) — SIMD optimisation notes
- [ARM NEON](arm/overview.md) — aarch64 backend + build / runtime /
  per-feature coverage
- [CUDA](cuda/overview.md) — NVIDIA GPU backend + build / invocation
- [NVTX profiling](nvtx/profiling.md) — profiling CUDA kernels with NVIDIA Nsight
- [SYCL / oneAPI](sycl/overview.md) — Intel GPU backend + build / invocation
- [SYCL bundling](sycl/bundling.md) — self-contained deployment without oneAPI
  runtime
- [Vulkan](vulkan/overview.md) — opt-in backend; vif + motion + adm live
  (T5-1c), full default-model coverage
- [HIP / AMD ROCm](hip/overview.md) — opt-in backend; 8/11 real
  kernels (psnr, integer_psnr, float_ansnr, float_motion, float_moment,
  float_ssim, ciede, integer_motion_v2); 3 stubs pending
- [Metal / Apple Silicon (scaffold)](metal/index.md) — auto-on-macOS;
  4 consumers registered, all `-ENOSYS` until T8-1b runtime PR

## Cross-backend parity

Every backend pair is gated on every PR by the **GPU-parity matrix
gate** (T6-8 / [ADR-0214](../adr/0214-gpu-parity-ci-gate.md)). The
gate diffs per-frame metrics with a feature-specific absolute
tolerance and emits one JSON / Markdown report per CI run. See
[../development/cross-backend-gate.md](../development/cross-backend-gate.md)
for the tolerance table, how to read failure output, and how to add
a new feature to the matrix.

## Related

- [../usage/cli.md](../usage/cli.md) — `--no_cuda` / `--no_sycl` /
  `--sycl_device` / `--cpumask` / `--gpumask` flags.
- [ADR-0022](../adr/0022-inference-runtime-onnx.md) — tiny-AI runtime (separate
  from classic VMAF backend dispatch; tiny-AI uses ONNX Runtime execution
  providers).
- [ADR-0027](../adr/0027-non-conservative-image-pins.md) — base-image / toolchain
  pins for GPU CI.
