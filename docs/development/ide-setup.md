# IDE setup (VS Code + clangd)

`.vscode/settings.json` ships with clangd as the C/C++ language
server (Microsoft IntelliSense is explicitly disabled). clangd
reads compile flags from `${workspaceFolder}/build/compile_commands.json`
which **meson generates automatically** during `meson setup`.

## Make sure `build/` covers every backend you touch

`compile_commands.json` only contains entries for files that
were actually compiled. If `build/` was set up CPU-only, clangd
has no include paths for `volk.h` / `vk_mem_alloc.h` / CUDA /
SYCL headers and lights up every `VkInstance` / `VmafCudaBuffer`
/ `sycl::queue` symbol as "undeclared identifier".

Fix: configure the IDE build with every backend you have a
toolchain for. The minimum useful setup for working on
`libvmaf/src/feature/vulkan/`:

```bash
meson setup build -Denable_vulkan=enabled -Denable_float=true
```

For CUDA + SYCL contributors:

```bash
source /opt/intel/oneapi/setvars.sh
CC=icx CXX=icpx meson setup build \
    -Denable_cuda=true -Denable_sycl=true \
    -Denable_vulkan=enabled -Denable_float=true
```

Then **restart clangd** in VS Code (Ctrl+Shift+P → "Restart
Language Server") so it re-reads `compile_commands.json`.

## If you need separate build dirs for backend-specific testing

Don't lose the IDE build to an enable-only-one-backend
reconfigure. Use named per-backend dirs alongside the IDE one:

```bash
# IDE (everything enabled):
meson setup build  -D…all-backends-on…

# CUDA-only test build:
meson setup build-cuda-test -Denable_cuda=true -Denable_float=true

# Vulkan-only test build:
meson setup build-vulkan-test -Denable_vulkan=enabled -Denable_float=true

# SYCL-only test build (icx/icpx required):
meson setup build-sycl-test -Denable_sycl=true -Denable_float=true
```

The cross-backend gate scripts under `scripts/ci/` accept any
of these via `--vmaf-binary`.

## Symptoms of a misconfigured `build/`

- `unknown type name 'VkInstance'` (or `VkDevice`, `VkPipeline`,
  `VkCommandBuffer`, …) on every file under
  `libvmaf/src/feature/vulkan/` or `libvmaf/src/vulkan/`.
- `unknown type name 'VmafCudaBuffer'` / `VmafCudaState'` on
  files under `libvmaf/src/feature/cuda/`.
- `'sycl/sycl.hpp' file not found` on files under
  `libvmaf/src/feature/sycl/`.
- `Included header errno.h is not used directly` warnings (a
  consequence of the indexer giving up after the first error).

If you see these and clangd is otherwise working, the cause is
99% of the time that `build/` was set up without the relevant
backend.
