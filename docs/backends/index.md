# Backends

VMAF supports multiple compute backends for hardware-accelerated quality assessment.
The backend is selected at build time via meson options and at runtime via environment
variables or the C API.

| Backend | Meson option | Runtime flag | Status |
|---------|-------------|--------------|--------|
| CPU (scalar) | always on | default | stable |
| x86 AVX2 | auto-detected | auto | stable |
| x86 AVX-512 | `-Denable_avx512=true` | auto | stable |
| CUDA | `-Denable_cuda=true` | `VMAF_FORCE_BACKEND=cuda` | stable |
| SYCL / oneAPI | `-Denable_sycl=true` | `VMAF_FORCE_BACKEND=sycl` | stable |

## Guides

- [x86 SIMD (AVX2 / AVX-512)](x86/avx512.md) — SIMD optimization notes
- [CUDA](cuda/overview.md) — NVIDIA GPU backend
- [NVTX profiling](nvtx/profiling.md) — profiling CUDA kernels with NVIDIA Nsight
- [SYCL / oneAPI](sycl/overview.md) — Intel GPU backend
- [SYCL bundling](sycl/bundling.md) — self-contained deployment without oneAPI runtime
