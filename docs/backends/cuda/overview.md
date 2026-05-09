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

### GPU architecture coverage

The fork ships cubins for every currently-shipping consumer Nvidia
generation from Turing through Blackwell whenever the host `nvcc`
supports them, plus a `compute_80` PTX as an unconditional JIT
fallback:

| Generation | Arch     | Emitted as      | Host `nvcc` gate |
| ---------- | -------- | --------------- | ---------------- |
| Turing     | `sm_75`  | cubin           | always           |
| Ampere     | `sm_80`  | cubin + PTX     | always           |
| Ampere     | `sm_86`  | cubin           | always           |
| Ada        | `sm_89`  | cubin           | always           |
| Hopper     | `sm_90`  | cubin           | `nvcc` > 11.8    |
| Blackwell  | `sm_100` | cubin           | `nvcc` > 12.8    |
| Blackwell  | `sm_120` | cubin + PTX     | `nvcc` > 12.8    |

The `compute_80` PTX is emitted unconditionally so any `sm_80`+ GPU
that lacks a matching cubin (future minor revisions, headless Tegra
variants) can still JIT a compatible kernel at driver-load time. This
diverges from upstream Netflix's meson.build, which ships cubins only
at Txx major boundaries; see [ADR-0122](../../adr/0122-cuda-gencode-coverage-and-init-hardening.md).

## Runtime requirements

The CUDA backend is compiled against `nv-codec-headers` but **does not
link** against `libcuda` — instead it `dlopen`s the driver library at
runtime through the `cuda_load_functions()` helper from
`ffnvcodec/dynlink_loader.h`. This keeps libvmaf linkable in
environments where the GPU driver may not be present at build time
(CI images, cross-compilation), but it means two things must be true
at run time on any host that actually dispatches the backend:

1. **`libcuda.so.1` exists and is reachable by the dynamic loader.**
   On Linux the driver stub is typically installed by the Nvidia
   driver package at `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (Debian/
   Ubuntu), `/usr/lib64/libcuda.so.1` (RHEL/Fedora), or under the
   distribution-specific Nvidia path. Check:

   ```bash
   ldconfig -p | grep -iE 'libcuda|libnvcuvid'
   ```

   If the line is missing, the backend will fail to initialise with a
   multi-line error message pointing at this section.

2. **The driver userspace matches the kernel module.** A fresh
   driver install that hasn't been followed by a reboot (or a
   `modprobe -r nvidia && modprobe nvidia`) commonly reports
   `cuInit(0)` returning a non-zero code even though `libcuda.so.1`
   loaded successfully. The log message for that case names
   `cuInit(0)` and the return code so the failure mode is
   distinguishable from the dlopen case above.

Statically-linked consumers (for example, ffmpeg binaries built with
`--enable-libvmaf` in static mode) are **not** exempt: the driver
library is loaded through `dlopen`, which bypasses `DT_NEEDED` and
therefore does not show up in `ldd <binary>`. An otherwise
self-contained static ffmpeg will still fail on the first frame if
`libcuda.so.1` is not on the loader path.

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
  float_adm_cuda.{c,h}           # float ADM extractor dispatch (ADR-0202)
  float_adm/                     # float ADM .cu kernels (single fatbin compiled with --fmad=false)
  integer_motion_cuda.{c,h}      # Motion extractor dispatch
  integer_motion/                # Motion .cu kernels
  integer_cambi_cuda.{c,h}      # CAMBI extractor dispatch (T3-15a / ADR-0360)
  integer_cambi/                 # CAMBI .cu kernels (cambi_score.cu)
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
- **Engine-scope fence batching (T-GPU-OPT).** Each feature extractor owns
  a private non-blocking stream + a `finished` event for its DtoH readback;
  the engine collects every frame's pending events in a single thread-local
  drain batch
  ([`src/cuda/drain_batch.c`](../../../libvmaf/src/cuda/drain_batch.c))
  and waits on them in one `cuStreamSynchronize(drain_str)` between submit
  and collect phases. A frame's per-extractor `collect()` calls then become
  host-side buffer reads only — the per-stream sync is short-circuited via
  `vmaf_cuda_kernel_collect_wait`'s `lc->drained` fast path. Participating
  extractors at time of writing: `psnr_cuda`, `motion_cuda`, `adm_cuda`,
  `vif_cuda`, `ssimulacra2_cuda`, and `integer_ms_ssim_cuda`
  ([ADR-0271](../../adr/0271-cuda-drain-batch-ms-ssim.md), the most recent
  joiner — its 5-scale pyramid required allocating per-scale partials
  buffers so all DtoH copies could enqueue back-to-back on the same stream).
  Bit-exactness is preserved (same kernels, same stream order — only the
  host wait point moves).

## Profiling

See [nvtx/profiling.md](../nvtx/profiling.md) for Nsight Systems recipes that
rely on the backend's NVTX annotations.

## Numerical tolerance vs the CPU scalar path

CUDA kernels target **close agreement** with the CPU fixed-point path,
not bit-exact equality. Different reduction orders, FMA contractions,
and parallel-prefix sums can perturb the final integer accumulator by a
fraction of a ULP — this has always been the case for VMAF on GPU,
not a fork regression. In practice the per-frame pooled VMAF agrees
to ~6 decimal places (the default `%.6f` truncation hides the delta
entirely; `--precision=max` exposes it). See
[ADR-0119](../../adr/0119-cli-precision-default-revert.md) for the
precision-default rationale.

The **Netflix golden-data gate is CPU-only** — the three reference
pairs in `python/test/` (1 normal + 2 checkerboard) are hardcoded
`assertAlmostEqual` values that only the CPU scalar + fixed-point
path is required to match exactly. See
[docs/principles.md §3.1](../../principles.md#31-netflix-golden-data-gate).

GPU regression is caught by fork-added per-backend snapshot tests
(`testdata/scores_cpu_*.json` + `testdata/netflix_benchmark_results.json`),
which record what each backend produces today at a small ULP
tolerance. Regenerate intentionally via
[`/regen-snapshots`](../../../.claude/skills/regen-snapshots/SKILL.md)
with a commit-message justification; use
[`/cross-backend-diff`](../../../.claude/skills/cross-backend-diff/SKILL.md)
to surface an unexpected delta.

## Known gaps

- **CIEDE2000** — no CUDA kernel (same CPU-fallback behaviour).
- **SSIM / MS-SSIM / PSNR / PSNR-HVS / ANSNR** — no CUDA kernels; these
  are rare enough in production that the CPU twin is sufficient.
- **Float-twin extractors (`float_*`)** — the CUDA backend
  implements the float twins for ANSNR / PSNR / Motion / VIF / ADM
  ([ADR-0202](../../adr/0202-float-adm-cuda-sycl.md)). Requesting
  `--feature float_<x>` with `--no_cuda=false` dispatches to GPU
  for those metrics.
- **`float_motion` extra options (`motion_add_scale1`,
  `motion_add_uv`, `motion_filter_size`, `motion_max_val`,
  `motion3_score`)** — these were added to the CPU `float_motion`
  extractor by the upstream port from Netflix/vmaf
  [`b949cebf`](https://github.com/Netflix/vmaf/commit/b949cebf)
  (2026-04-29). As of T3-15(c) /
  [ADR-0219](../../adr/0219-motion3-gpu-coverage.md), the
  `integer_motion_cuda` kernel emits `motion3_score` in 3-frame
  window mode via host-side `motion_blend()` post-processing of
  `motion2_score`; the full options surface
  (`motion_blend_factor`, `motion_blend_offset`, `motion_fps_weight`,
  `motion_max_val`, `motion_moving_average`) is exposed.
  `motion_five_frame_window=true` is rejected with `-ENOTSUP` at
  `init()` (the 5-deep blur ring is still deferred). The
  `motion_add_uv=true` path is independent from motion3 and
  remains **not yet wired through to the CUDA backend**. The CUDA
  `picture_copy()` callsite at
  [`src/feature/cuda/integer_ms_ssim_cuda.c`](../../../libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c)
  passes `0` for the new trailing `channel` argument (Y-plane only,
  preserving CUDA pre-port behaviour). UV-plane motion on GPU is a
  follow-up tracked in [docs/state.md](../../state.md).
- **SSIMULACRA 2** — `ssimulacra2_cuda` shipped per
  [ADR-0206](../../adr/0206-ssimulacra2-cuda-sycl.md) (hybrid
  host/GPU pipeline, IIR fatbin pinned with `--fmad=false`).
- **HIP / AMD** — not yet scaffolded; see
  [backends/index.md](../index.md) for the status row.

See [metrics/features.md](../../metrics/features.md) for the
per-extractor coverage matrix.

## References

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/) (informational — libvmaf itself uses the Driver API)
