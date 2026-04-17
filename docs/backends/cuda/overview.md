# CUDA Backend

The CUDA backend runs VMAF's core feature extractors (VIF, ADM, Motion)
directly on an NVIDIA GPU, keeping frames on the device across the full
pipeline to avoid PCIe round-trips.

## Build

```bash
meson setup build -Denable_cuda=true
ninja -C build
```

Requires the CUDA toolkit (`nvcc`, driver API headers). The build uses the
driver API only — through `ffnvcodec` dynlink wrappers — so applications
that already load CUDA through FFmpeg share the same primary context.

Meson options:

- `-Denable_cuda=true` — compile the CUDA backend + kernels.
- `-Denable_nvtx=true` — instrument kernels with NVTX ranges (see [nvtx/profiling.md](../nvtx/profiling.md)).
- `-Denable_nvcc=true` — build NVCC-compiled kernel objects (default when `enable_cuda` is on).

## Runtime

When the binary is built with CUDA, the backend is auto-selected on GPU-capable
hosts. CLI controls:

```bash
./build/tools/vmaf ...            # CUDA used automatically
./build/tools/vmaf --no_cuda ...  # force CPU path
```

The FFmpeg filter name is `libvmaf_cuda` — see [usage/ffmpeg.md](../../usage/ffmpeg.md)
for a hwaccel pipeline that keeps decoded frames on the GPU.

## Source layout

```text
libvmaf/src/cuda/                # queue, picture, ring-buffer runtime
libvmaf/src/feature/cuda/        # per-feature kernels
  integer_vif_cuda.{c,h}         # VIF extractor dispatch
  integer_vif/                   # VIF .cu kernels
  integer_adm_cuda.{c,h}         # ADM extractor dispatch
  integer_adm/                   # ADM .cu kernels
  integer_motion_cuda.{c,h}      # Motion extractor dispatch
  integer_motion/                # Motion .cu kernels
```

Adding a new CUDA extractor: see [`/add-feature-extractor`](../../../.claude/skills/add-feature-extractor/SKILL.md).

## Design notes

- **Driver API only.** We link against `cuda.h` via `ffnvcodec` and do not
  depend on the CUDA Runtime API. This keeps libvmaf linkable against FFmpeg
  builds that already load CUDA dynamically.
- **Pinned host staging.** Input pictures are uploaded from
  `cuMemHostAlloc`-pinned buffers. See [picture_cuda.c](../../../libvmaf/src/cuda/picture_cuda.c).
- **Non-default streams per extractor.** Each feature extractor owns its own
  stream so submit/collect for different features can overlap.
- **Ring-buffered double-buffer submit.** Frame N+1 starts uploading while
  frame N is still on the device — see [ring_buffer.c](../../../libvmaf/src/cuda/ring_buffer.c).
- **Shared primary context.** We retain the device's primary context with
  `cuDevicePrimaryCtxRetain` so FFmpeg and VMAF share one GPU context rather
  than fighting over time-sliced contexts.

## Profiling

See [nvtx/profiling.md](../nvtx/profiling.md) for Nsight Systems recipes that
rely on the backend's NVTX annotations.

## Numerical tolerance vs the CPU scalar path

CUDA integer kernels are bit-identical to the CPU fixed-point path by
design — the shipped Netflix golden suite
(`make test-netflix-golden`) compares CUDA scores against the CPU
scores at the reported `--precision` and fails the build on any
divergence beyond the last printed digit.

There is **no floating-point slack** on VIF / ADM / Motion2: the integer
kernels produce the same 64-bit accumulator values as the CPU scalar
twins. If you see a per-frame delta greater than 0 at `--precision 17`,
treat it as a backend bug and run
[`/cross-backend-diff`](../../../.claude/skills/cross-backend-diff/SKILL.md)
before shipping.

## Known gaps

- **CAMBI** — no CUDA kernel. Runs on CPU even when the rest of the
  pipeline is on the GPU; the frame is downloaded to host memory for
  CAMBI and the CUDA twin is used for everything else.
- **CIEDE2000** — no CUDA kernel (same CPU-fallback behaviour).
- **SSIM / MS-SSIM / PSNR / PSNR-HVS / ANSNR** — no CUDA kernels; these
  are rare enough in production that the CPU twin is sufficient.
- **Float-twin extractors (`float_*`)** — the CUDA backend only
  implements the fixed-point integer extractors. Requesting
  `--feature float_adm` with `--no_cuda=false` will still dispatch to
  CPU for that feature.
- **HIP / AMD** — not yet scaffolded; see
  [backends/index.md](../index.md) for the status row.

See [metrics/features.md](../../metrics/features.md) for the
per-extractor coverage matrix.

## References

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/) (informational — libvmaf itself uses the Driver API)
