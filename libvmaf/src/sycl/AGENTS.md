# AGENTS.md — libvmaf/src/sycl

Orientation for agents working on the SYCL / DPC++ backend runtime. Parent:
[../../AGENTS.md](../../AGENTS.md).

## Scope

The SYCL-side runtime (queue management, USM, dmabuf import). SYCL
**feature kernels** live in [../feature/sycl/](../feature/sycl/).

```
sycl/
  common.cpp/.h             # queue creation, device selection, error-check
  picture_sycl.cpp/.h       # VmafPicture on a SYCL device (USM-backed)
  dmabuf_import.cpp/.h      # Linux DMA-BUF import path for zero-copy Level Zero
```

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **Compiler is `icpx` (Intel oneAPI) or `clang++` with `-fsycl`.** Do not
  assume MSVC-style extensions.
- **Experimental flags enabled**: `-fsycl-unnamed-lambda`,
  `-fsycl-allow-func-ptr`, `-fsycl-device-code-split=per_kernel`. See
  [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md).
- **USM allocations**: prefer device USM for kernel-resident data, shared
  USM for cross-side communication. Host USM only when an adjacent API
  requires a pointer the host can dereference directly.
- **dmabuf import** is Linux-only and gated at build time; no callers
  should assume the FD path exists on other OSes.
- **Numerical snapshots**: same rule as CUDA — see CLAUDE.md §9.

## Governing ADRs

- [ADR-0002](../../../docs/adr/0002-merge-path-master-default.md) — sycl branch → master merge history.
- [ADR-0016](../../../docs/adr/0016-sycl-to-master-merge-conflict-policy.md) — merge-conflict policy.
- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — OpenVINO EP mapping for SYCL.
- [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md) — experimental SYCL flags.

## Build

```bash
meson setup build -Denable_cuda=false -Denable_sycl=true
ninja -C build
```

Requires oneAPI (`source /opt/intel/oneapi/setvars.sh`) or equivalent DPC++
toolchain with `icpx` on PATH.
