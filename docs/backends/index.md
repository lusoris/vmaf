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
| HIP (AMD) | _(not yet scaffolded)_ | no | n/a | planned — meson option does not exist yet; `/add-gpu-backend hip` is the scaffolding path |

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
- [Vulkan (scaffold)](vulkan/overview.md) — opt-in scaffold returning
  `-ENOSYS` until the runtime PR (T5-1b)

## Related

- [../usage/cli.md](../usage/cli.md) — `--no_cuda` / `--no_sycl` /
  `--sycl_device` / `--cpumask` / `--gpumask` flags.
- [ADR-0022](../adr/0022-inference-runtime-onnx.md) — tiny-AI runtime (separate
  from classic VMAF backend dispatch; tiny-AI uses ONNX Runtime execution
  providers).
- [ADR-0027](../adr/0027-non-conservative-image-pins.md) — base-image / toolchain
  pins for GPU CI.
