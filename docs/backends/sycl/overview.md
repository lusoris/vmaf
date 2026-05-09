# SYCL Backend

The SYCL / oneAPI backend runs VMAF's core feature extractors (VIF, ADM,
Motion) on any SYCL-capable accelerator. It is the fork's primary path
for Intel GPUs (Arc, integrated UHD / Iris Xe, Data Center GPU Flex/Max)
and also targets AMD via the HIP plugin and NVIDIA via the CUDA plugin
when the DPC++ compiler is built with those backends.

## Build

```bash
meson setup build -Denable_sycl=true
ninja -C build
```

Requires Intel oneAPI DPC++ (`icpx`). A bundled self-contained deployment —
useful for shipping the binary to hosts that don't have oneAPI installed —
is described in [bundling.md](bundling.md).

Meson options:

- `-Denable_sycl=true` — compile the SYCL backend + kernels.
- `-Denable_cuda=true` can be set in parallel; both backends can coexist
  in a single binary.

## Runtime

When built with SYCL, the backend is auto-selected on hosts that expose a
Level Zero device. CLI controls:

```bash
./build/tools/vmaf ...                   # SYCL used automatically
./build/tools/vmaf --no_sycl ...         # force CPU path
./build/tools/vmaf --sycl_device 1 ...   # pick device index 1 explicitly
```

Device index `0` (the default) is whichever device SYCL's default selector
picks — usually the first discrete GPU. Use `--sycl_device` to pin an
iGPU or a specific Arc card.

## Source layout

```text
libvmaf/src/sycl/                        # queue, USM, surface import
  common.{cpp,h}                         # SYCL queue + device selection
  picture_sycl.{cpp,h}                   # USM picture upload / CPU path
  dmabuf_import.{cpp,h}                  # Linux: zero-copy VA-API dmabuf import
  d3d11_import.cpp                       # Windows: D3D11 staging-texture import
libvmaf/src/feature/sycl/                # per-feature kernels
  integer_vif_sycl.cpp
  integer_adm_sycl.cpp
  float_adm_sycl.cpp                     # float ADM extractor (ADR-0202)
  integer_motion_sycl.cpp
```

## Design notes

- **Single-source DPC++.** Kernels are ordinary C++ lambdas submitted via
  `queue::parallel_for`. No GLSL shaders, no separate SPIR-V assets — device
  code is linked into the binary at build time by `clang-offload-wrapper`.
- **Unified Shared Memory (USM).** The backend uses `malloc_device` for
  per-feature scratch buffers and a shared allocation for pictures when
  zero-copy isn't available.
- **Zero-copy dmabuf import.** When the input is a VA-API surface (e.g. from
  a QSV-decoded FFmpeg frame), the backend imports the dmabuf directly via
  `ext::oneapi::experimental::external_memory` — no CPU upload. See
  [dmabuf_import.cpp](../../../libvmaf/src/sycl/dmabuf_import.cpp).
- **D3D11 staging-texture import (Windows).** The `vmaf_sycl_import_d3d11_surface`
  API accepts an `ID3D11Texture2D*` from a Windows decoder (MediaFoundation, DXVA2,
  Direct3D11 VideoProcessor). The implementation creates a staging texture with
  `D3D11_USAGE_STAGING + D3D11_CPU_ACCESS_READ`, calls `CopyResource` to pull the
  GPU surface into staging, `Map`s the staging tex for CPU read, and forwards the
  mapped pointer + row pitch into `vmaf_sycl_upload_plane`. This is **not zero-copy**
  — throughput is bounded by PCIe upstream (staging Map) + PCIe downstream (SYCL H2D).
  A zero-copy equivalent would need DXGI NT-handle sharing + DPC++ D3D11 interop,
  which isn't documented in oneAPI as of 2025.1. See [d3d11_import.cpp](../../../libvmaf/src/sycl/d3d11_import.cpp)
  and ADR-0103.
- **In-order queues per extractor.** Each feature extractor owns a SYCL
  in-order queue. The host-side dispatcher submits work without explicit
  event dependencies; dependencies within an extractor are handled by the
  in-order semantics.

## fp64-less device contract (T7-17)

All SYCL feature kernels in this fork are designed to run on devices that
**lack `sycl::aspect::fp64`** (Intel Arc A-series, most Intel iGPUs, many
mobile / embedded GPUs). No kernel emits double-precision floating-point
SPIR-V instructions, so the JIT does not need to fall back to int64
emulation, and there is no per-kernel performance penalty on fp64-less
devices.

Concretely:

- **ADM gain limiting** uses an int64 Q31 fixed-point split-multiply
  (`gain_limit_to_q31` in `integer_adm_sycl.cpp`). The CPU reference
  multiplies a 32-bit DWT coefficient by a `double` gain in `[1.0, 100.0]`;
  the device path replaces this with `gain_q31 = round(gain * 2^31)` and a
  16-bit-split int64 multiply, exact for the production gain values
  (`1.0`, `100.0`) and within ±1 LSB for fractional gains.
- **VIF gain limiting** runs entirely in fp32 (`sycl::fmin(g,
  vif_enhn_gain_limit)` over float operands). The host stores the gain as
  a `double` for parity with the CPU API; the launcher casts to `float`
  before kernel submission.
- **CIEDE / SSIM accumulators** avoid `sycl::reduction<double>`; partials
  are accumulated in 32-bit fixed point and reduced via
  `sycl::plus<int64_t>` over subgroups.

`VmafSyclState` records `has_fp64` at queue construction so future
fp64-only optimisations can branch on it; current kernels do not. The
init log line at `VMAF_LOG_LEVEL_INFO` confirms which path was taken — on
an fp64-less device it reads "device lacks native fp64 — kernels already
use fp32 + int64 paths, no emulation overhead". A previous WARNING-level
line ("using int64 emulation for gain limiting") was misleading: it
suggested an emulation-overhead fallback that never existed. See
[ADR-0220](../../adr/0220-sycl-fp64-fallback.md).

If you add a new SYCL kernel and it captures a `double` operand or calls
`sycl::reduction<double>`, the entire SPIR-V module is rejected by the
Level Zero runtime on Arc A-series — even if the offending kernel is
never submitted. Audit the lambda capture list and any `sycl::reduce*`
calls before merging.

## Picture pre-allocation

`vmaf_sycl_preallocate_pictures()` + `vmaf_sycl_picture_fetch()` back a
2-deep ring of USM-backed `VmafPicture` instances that callers hand to
`vmaf_read_pictures()`. Three modes:

| `pic_prealloc_method` | Backing | Use case |
| --- | --- | --- |
| `VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_NONE` | No pool; `vmaf_sycl_picture_fetch` falls back to host `vmaf_picture_alloc` | CPU-fed pipelines, test harnesses |
| `VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_DEVICE` | `sycl::malloc_device` (GPU-resident) | Zero-copy decoder interop (decoder writes directly into device USM) |
| `VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_HOST` | `sycl::malloc_host` (coherent, CPU-visible) | Decoders that must write from the CPU but want pool reuse |

The pool depth (2) matches the double-buffered shared-frame upload in
`VmafSyclState`, so frame N+1 can start filling slot 1 while frame N's
compute still consumes slot 0. The caller owns the ref returned by
`vmaf_sycl_picture_fetch` and must release it via `vmaf_picture_unref` when
done with it; the pool retains its own ref until `vmaf_close()`.

Minimal example:

```c
VmafSyclPictureConfiguration cfg = {
    .pic_params = { .w = 1920, .h = 1080, .bpc = 8, .pix_fmt = VMAF_PIX_FMT_YUV420P },
    .pic_prealloc_method = VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_DEVICE,
};
vmaf_sycl_preallocate_pictures(vmaf, cfg);

for (unsigned i = 0; i < n_frames; i++) {
    VmafPicture ref, dis;
    vmaf_sycl_picture_fetch(vmaf, &ref);   /* device USM, caller writes */
    vmaf_sycl_picture_fetch(vmaf, &dis);
    /* ... fill ref.data[0] and dis.data[0] via decoder/upload ... */
    vmaf_read_pictures(vmaf, &ref, &dis, i);
}
vmaf_read_pictures(vmaf, NULL, NULL, 0);
```

See [ADR-0101](../../adr/0101-sycl-usm-picture-pool.md) for the design
rationale (Y-plane only, pool depth 2, refcount semantics).

## Profiling

- Intel VTune (`vtune-gui`) with the GPU Compute analysis type for kernel
  occupancy and EU utilization.
- `onetrace` from the [pti-gpu](https://github.com/intel/pti-gpu) project
  for Level Zero API-level tracing.
- For end-to-end wall-time comparisons against the CUDA / CPU paths, use
  `make test-netflix-golden` which records per-backend scores and timings.
- Programmatic profiling via `VmafSyclState.enable_profiling` — see
  [api/gpu.md](../../api/gpu.md#profiling-helpers) for the queue-event
  query API.

## Numerical tolerance vs the CPU scalar path

SYCL kernels target **close agreement** with the CPU fixed-point
path, not bit-exact equality. Like every GPU path for VMAF, different
reduction orders, parallel-prefix scans, and FMA contractions can
perturb the final accumulator by a fraction of a ULP. Agreement is
typically at ~6 decimal places of the pooled VMAF score.

The **Netflix golden-data gate is CPU-only** — see
[docs/principles.md §3.1](../../principles.md#31-netflix-golden-data-gate).
The SYCL backend's per-build numerics are pinned by fork-added snapshot
tests, not by the Netflix goldens.

Accelerator-dependent controls that reduce (but do not eliminate)
the deviation:

- **fp16 path is disabled for scoring.** Some Intel GPUs expose fp16
  arithmetic; libvmaf forces fp32 on the kernel so scores are portable
  across hosts with different fp16 rounding modes.
- **Work-group reductions use fixed iteration order** so the most
  common source of cross-run drift (non-deterministic reduction tree)
  is eliminated; the remaining deltas come from unavoidable arithmetic
  restructuring between scalar and parallel-prefix code.

## Known gaps

- **CAMBI** — no SYCL kernel; runs on CPU. Frame download is unavoidable
  for CAMBI when the rest of the pipeline is on the GPU.
- **CIEDE2000** — no SYCL kernel; CPU fallback.
- **SSIM / MS-SSIM / PSNR / PSNR-HVS / ANSNR** — no SYCL kernels.
- **Float-twin extractors (`float_*`)** — the SYCL backend
  implements ANSNR / PSNR / Motion / VIF / ADM
  ([ADR-0202](../../adr/0202-float-adm-cuda-sycl.md)).
- **`float_motion` extra options (`motion_add_scale1`,
  `motion_add_uv`, `motion_filter_size`, `motion_max_val`,
  `motion3_score`)** — these CPU options came in via the upstream
  port from Netflix/vmaf
  [`b949cebf`](https://github.com/Netflix/vmaf/commit/b949cebf)
  (2026-04-29). As of T3-15(c) /
  [ADR-0219](../../adr/0219-motion3-gpu-coverage.md), the SYCL
  `integer_motion` extractor emits `motion3_score` in 3-frame
  window mode via host-side `motion_blend()` post-processing of
  `motion2_score`; the full options surface
  (`motion_blend_factor`, `motion_blend_offset`, `motion_fps_weight`,
  `motion_max_val`, `motion_moving_average`) is exposed.
  `motion_five_frame_window=true` is rejected with `-ENOTSUP` at
  `init()` (the 5-deep blur ring is still deferred). The
  `motion_add_uv=true` path is independent from motion3 and remains
  **not yet wired through to the SYCL backend** — UV-plane motion
  stays CPU-only. The SYCL `picture_copy()` callsites at
  [`src/feature/sycl/integer_ms_ssim_sycl.cpp`](../../../libvmaf/src/feature/sycl/integer_ms_ssim_sycl.cpp)
  and
  [`src/feature/sycl/integer_ssim_sycl.cpp`](../../../libvmaf/src/feature/sycl/integer_ssim_sycl.cpp)
  pass `0` for the new trailing `channel` argument (Y-plane only,
  preserving SYCL pre-port behaviour).
- **SSIMULACRA 2** — `ssimulacra2_sycl` shipped per
  [ADR-0206](../../adr/0206-ssimulacra2-cuda-sycl.md) (hybrid
  host/GPU pipeline, kernel lambdas held in IEEE-754 strict mode by
  the existing `-fp-model=precise`).
- **dmabuf import is Linux-only.** The VA-API → dmabuf fast path is
  gated on `#ifndef _WIN32` in `sycl/dmabuf_import.cpp`; on Windows,
  `vmaf_sycl_dmabuf_import` and `vmaf_sycl_import_va_surface` return
  `-ENOSYS` so the caller falls back to the D3D11 staging path
  (`d3d11_import.cpp`). DMA-BUF is a Linux kernel interface
  (`ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF`); Level Zero on Windows uses
  NT handles instead.
- **HIP / ROCm via SYCL** — requires building DPC++ with the HIP plugin;
  the shipped Intel oneAPI binaries only include the Level Zero +
  OpenCL CPU + CUDA plugins.

See [metrics/features.md](../../metrics/features.md) for the
per-extractor coverage matrix and [api/gpu.md](../../api/gpu.md#sycl)
for the programmatic surface.

## References

- [SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
- [Intel oneAPI DPC++ Compiler](https://github.com/intel/llvm)
- [Level Zero Specification](https://spec.oneapi.io/level-zero/latest/)
- [Intel oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)
