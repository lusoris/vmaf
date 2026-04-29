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
- **`-fp-model=precise` is load-bearing.** The SYCL feature build
  line in `libvmaf/src/meson.build` adds `-fp-model=precise` to
  every per-kernel TU. This blocks `icpx` from FMA contraction in
  the kernel lambdas and matches the GLSL `precise` /
  `NoContraction` decorations on the Vulkan twins. Removing it
  drifts `float_adm_sycl` ([ADR-0202](../../../docs/adr/0202-float-adm-cuda-sycl.md))
  past `places=4` at scale 2 and `ssimulacra2_sycl`
  ([ADR-0206](../../../docs/adr/0206-ssimulacra2-cuda-sycl.md))
  past `places=2` through the IIR. **On rebase**: keep this flag
  on the SYCL feature line.
- **fp64-free kernels are load-bearing
  ([ADR-0220](../../../docs/adr/0220-sycl-fp64-fallback.md), T7-17).**
  Every SYCL feature-kernel lambda must capture and operate on
  `float` / integer types only. No `double` operand inside a
  `parallel_for` body, no `sycl::reduction<double>`, no
  `sycl::plus<double>`. This is hard, not soft: a single fp64
  instruction anywhere in the TU's SPIR-V module causes the
  Level Zero runtime to reject the entire module on Intel Arc
  A-series and other fp64-less devices, even when the offending
  kernel is never submitted. `double` is allowed *outside* the
  kernel lambda (host-side post-processing in `extract` /
  `flush` callbacks, score aggregation, log10 normalisation).
  ADM gain limiting uses int64 Q31 (`gain_limit_to_q31` +
  `launch_decouple_csf<false>` in `integer_adm_sycl.cpp`); VIF
  gain limiting uses fp32 `sycl::fmin`. **On rebase**: if an
  upstream cherry-pick brings a `double` into a kernel lambda,
  refactor it to int64 / fp32 before merging.

## Governing ADRs

- [ADR-0002](../../../docs/adr/0002-merge-path-master-default.md) — sycl branch → master merge history.
- [ADR-0016](../../../docs/adr/0016-sycl-to-master-merge-conflict-policy.md) — merge-conflict policy.
- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — OpenVINO EP mapping for SYCL.
- [ADR-0027](../../../docs/adr/0027-non-conservative-image-pins.md) — experimental SYCL flags.
- [ADR-0220](../../../docs/adr/0220-sycl-fp64-fallback.md) — SYCL feature kernels are unconditionally fp64-free (T7-17).

## Build

```bash
meson setup build -Denable_cuda=false -Denable_sycl=true
ninja -C build
```

Requires oneAPI (`source /opt/intel/oneapi/setvars.sh`) or equivalent DPC++
toolchain with `icpx` on PATH.
