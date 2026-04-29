# Meson build flags

Complete reference for every build-time option libvmaf exposes via
[`libvmaf/meson_options.txt`](../../libvmaf/meson_options.txt), plus the
standard Meson options that materially change what ships in the binary.

Per [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md), build
flags are a user-discoverable surface and ship documentation in the
same PR as the code.

Pass any option via `-D<name>=<value>` on `meson setup`:

```bash
meson setup build -Denable_cuda=true -Denable_avx512=false -Dbuildtype=release
```

Reconfigure an existing build tree without wiping it:

```bash
meson configure build -Denable_sycl=true
ninja -C build
```

## Project options (`meson_options.txt`)

| Option            | Type      | Default   | Effect                                                          |
|-------------------|-----------|-----------|-----------------------------------------------------------------|
| `enable_tests`    | bool      | `true`    | Build `libvmaf/test/` unit tests; `meson test -C build` needs this |
| `enable_docs`     | bool      | `true`    | Build the Doxygen C-API HTML under `build/libvmaf/doc/`         |
| `enable_tools`    | bool      | `true`    | Build the `vmaf` and `vmaf_bench` CLI binaries                  |
| `enable_asm`      | bool      | `true`    | Compile any `*.asm` source files (nasm); disables all SIMD paths when `false` |
| `enable_avx512`   | bool      | `true`    | Build the AVX-512 kernels (requires nasm ≥ 2.14); auto-disabled on hosts without AVX-512 headers |
| `built_in_models` | bool      | `true`    | Compile the default `.json` VMAF models into the library (`version=vmaf_v0.6.1` etc. resolve without disk I/O) |
| `enable_float`    | bool      | `false`   | Compile the `float_*` feature extractors (`float_psnr`, `float_ssim`, `float_ms_ssim`, `float_vif`, `float_adm`, `float_motion`, `float_ansnr`); off by default because the fixed-point core is the ship path |
| `enable_cuda`     | bool      | `false`   | Compile the CUDA backend + `.cu` kernels; requires CUDA toolkit (`nvcc`) |
| `enable_nvtx`     | bool      | `false`   | Instrument CUDA kernels with NVTX ranges for Nsight Systems — see [backends/nvtx/profiling.md](../backends/nvtx/profiling.md) |
| `enable_nvcc`     | bool      | `true`    | Use `nvcc` to compile the CUDA kernel objects (the alternative is the clang CUDA driver); only takes effect when `enable_cuda=true` |
| `enable_sycl`     | bool      | `false`   | Compile the SYCL / oneAPI backend + DPC++ kernels; requires `icpx` on PATH |
| `sycl_compiler`   | string    | `icpx`    | Path or name of the SYCL compiler — only consulted when `enable_sycl=true` |
| `enable_dnn`      | feature   | `auto`    | Build the tiny-AI ONNX Runtime surface. `auto` tries to link ORT and silently disables if it's missing; `enabled` fails the configure step when ORT is unavailable; `disabled` omits the `dnn.h` symbols entirely |
| `enable_vulkan`   | feature   | `disabled`| Compile the Vulkan compute backend. Scaffold landed via [ADR-0175](../adr/0175-vulkan-backend-scaffold.md), runtime via ADR-0178 (T5-1b), default-model kernel matrix complete per ADR-0193 (VIF + ADM + motion + motion_v2 + ssimulacra2 plus the GPU long-tail batches). Default `disabled` — `auto` would silently flip on in builds with a Vulkan SDK installed and we keep it opt-in. When enabled, requires `volk` + Vulkan SDK ≥ 1.3 + `glslc` + VMA. |
| `enable_mcp`      | bool      | `false`   | Compile the embedded MCP (Model Context Protocol) server scaffold ([ADR-0209](../adr/0209-mcp-embedded-scaffold.md)). Audit-first stub: every entry point in `libvmaf_mcp.h` returns `-ENOSYS`. Runtime bodies (cJSON + mongoose, dedicated MCP pthread, SPSC ring buffer, transports) land in T5-2b. See [`docs/mcp/embedded.md`](../mcp/embedded.md). |
| `enable_mcp_sse`  | bool      | `false`   | Compile in the SSE (Server-Sent Events / loopback HTTP) transport for the embedded MCP server. Requires `enable_mcp=true`. Stub-only until T5-2b ships transport bodies. |
| `enable_mcp_uds`  | bool      | `false`   | Compile in the Unix-domain-socket transport. Requires `enable_mcp=true`. POSIX-only; non-POSIX hosts return `-ENODEV` at runtime. Stub-only until T5-2b. |
| `enable_mcp_stdio`| bool      | `false`   | Compile in the stdio (LSP-framed JSON-RPC on caller-supplied fd pair) transport. Requires `enable_mcp=true`. Stub-only until T5-2b. |
| `enable_hip`      | bool      | `false`   | Compile the HIP (AMD ROCm) compute backend scaffold. Default off; every public C-API entry point returns `-ENOSYS` until the runtime PR (T7-10b) lands — see [ADR-0209](../adr/0209-hip-backend-scaffold.md) and [backends/hip/overview.md](../backends/hip/overview.md). The scaffold has no hard runtime dependencies; ROCm 6+ is required only when the kernels arrive. |

### Flag interactions

- **`enable_asm=false` disables every SIMD path.** This is an escape hatch for
  toolchains that can't compile the `*.asm` kernels — it gives a pure-scalar
  binary much slower than even the AVX2 path.
- **`enable_avx512` is auto-downgraded.** Even with `-Denable_avx512=true`,
  the build will disable the AVX-512 source files if `nasm --version` is
  < 2.14 or if the host headers don't expose the required intrinsics.
  Scripted builds should not assume this flag survives configure-time.
- **`enable_cuda` + `enable_sycl` can coexist.** Both backends compile into
  the same binary; the runtime picks one per extractor based on which
  backend has a kernel available. Use `--no_cuda` / `--no_sycl` at run
  time to pin a backend for A/B comparisons.
- **`enable_nvcc=false` is experimental.** Clang CUDA support lags `nvcc`
  on newer CUDA toolchains; this flag is intended for maintainers
  investigating codegen regressions, not for day-to-day builds.
- **`enable_float=true` does not turn off the integer path.** The
  fixed-point extractors always compile; this flag only adds the float
  twins on top.
- **`enable_dnn=auto` vs `enabled`.** Always set `enabled` in CI if you
  want tiny-AI coverage — `auto` will silently skip the DNN tests if ORT
  fails to link and will not flag the gap as a failure.

### Options referenced in docs but not present

(All build flags referenced elsewhere in the docs are now defined in
`libvmaf/meson_options.txt`. `enable_hip` was added by ADR-0209
(T7-10) — the option exists but the backend it enables is a scaffold
returning `-ENOSYS`; see the table above and
[backends/hip/overview.md](../backends/hip/overview.md).)

## Standard Meson options that matter

These come from Meson itself, not `meson_options.txt` — but they change
the emitted artifact, so they belong in the build-time surface.

| Option             | Default        | Effect                                                               |
|--------------------|----------------|----------------------------------------------------------------------|
| `buildtype`        | `debug`        | `debug` / `debugoptimized` / `release` / `minsize` / `plain`         |
| `default_library`  | `shared`       | `shared` / `static` / `both` — `both` is required for the test suite layout |
| `b_ndebug`         | `false`        | Disable C `assert()` when `true`; set with `-Db_ndebug=true`         |
| `b_sanitize`       | `none`         | `address`, `undefined`, `address,undefined`, `thread`, `memory`      |
| `b_lto`            | `false`        | Enable LTO; measurable speedup on the scalar / AVX2 paths            |
| `c_args`           | (empty)        | Extra C flags. Passing `-DVMAF_PICTURE_POOL` enables the picture-pool allocator (gated test target) |
| `prefix`           | `/usr/local`   | Install prefix for `ninja install`                                   |
| `pkg_config_path`  | (system)       | Useful when linking against a non-system ONNX Runtime for `enable_dnn` |

## Recommended configurations

```bash
# Fast iteration (CPU only, debug assertions on, AVX2/AVX-512 kept)
meson setup build \
  -Dbuildtype=debugoptimized \
  -Denable_cuda=false -Denable_sycl=false

# Release with CUDA + NVTX for profiling
meson setup build \
  -Dbuildtype=release \
  -Denable_cuda=true -Denable_nvtx=true \
  -Denable_sycl=false

# CI golden-gate config (matches the Netflix CPU golden job)
meson setup build \
  -Dbuildtype=release \
  -Denable_cuda=false -Denable_sycl=false \
  -Denable_float=true -Dbuilt_in_models=true

# Tiny-AI tests (CI)
meson setup build \
  -Dbuildtype=release \
  -Denable_dnn=enabled

# Sanitizer run (for `make test`)
meson setup build \
  -Dbuildtype=debug \
  -Db_sanitize=address,undefined
```

## How feature flags land in the binary

`libvmaf/src/meson.build` reads each option once and stores it in a
`configuration_data()` block that drives:

- the `#define HAVE_AVX512 …` macro baked into the library headers,
- whether `.cu` / `.cpp` / `.asm` source files are added to the target,
- whether the `libvmaf_cuda.h` / `libvmaf_sycl.h` headers are installed
  (these headers are omitted from the installed tree when their backend
  is disabled at build — see
  [libvmaf/include/libvmaf/meson.build](../../libvmaf/include/libvmaf/meson.build)).

To inspect the resolved configuration of an existing build tree:

```bash
meson configure build | head -40
```

## Related

- [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) — project-wide
  doc-substance rule (this page satisfies the Build-flag bar).
- [backends/index.md](../backends/index.md) — how build flags turn into
  runtime backend availability.
- [ADR-0022](../adr/0022-inference-runtime-onnx.md) — `enable_dnn` + ORT
  choice.
- [getting-started/building-on-windows.md](../getting-started/building-on-windows.md)
  — platform-specific toolchain setup.
- [development/release.md](release.md) — release build + signing flow.
