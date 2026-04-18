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

## Profiling

- Intel VTune (`vtune-gui`) with the GPU Compute analysis type for kernel
  occupancy and EU utilization.
- `onetrace` from the [pti-gpu](https://github.com/intel/pti-gpu) project
  for Level Zero API-level tracing.
- For end-to-end wall-time comparisons against the CUDA / CPU paths, use
  `make test-netflix-golden` which records per-backend scores and timings.

## References

- [SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
- [Intel oneAPI DPC++ Compiler](https://github.com/intel/llvm)
- [Level Zero Specification](https://spec.oneapi.io/level-zero/latest/)
- [Intel oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)
