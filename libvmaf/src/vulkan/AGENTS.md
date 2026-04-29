# AGENTS.md — libvmaf/src/vulkan

Orientation for agents working on the Vulkan compute backend
runtime. Parent: [../../AGENTS.md](../../AGENTS.md).

## Scope

The Vulkan-side runtime (instance / device / queue creation,
memory allocator via VMA, picture lifecycle, dispatch helpers,
zero-copy `VkImage` import). Vulkan **feature kernels** live in
[../feature/vulkan/](../feature/vulkan/) (`.c` host glue +
`shaders/*.comp` GLSL).

```text
vulkan/
  common.c              # VkInstance / VkPhysicalDevice / VkQueue
  vulkan_common.h       # public-internal types
  vulkan_internal.h     # private helpers
  picture_vulkan.c/.h   # VmafPicture on a Vulkan device
  import.c              # zero-copy VkImage / external-memory import
  import_picture.h      # per-state staging buffers + ImageToBuffer copy
  dispatch_strategy.c/.h  # PRIMARY_CMDBUF vs SECONDARY_CMDBUF_REUSE
  vma_impl.cpp          # AMD Vulkan Memory Allocator translation unit
  spv_embed.py          # build-time helper: SPV → C array embed
  meson.build
```

Public header: [`include/libvmaf/libvmaf_vulkan.h`](../../include/libvmaf/libvmaf_vulkan.h).

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **Volk dynamic loader** is bundled. All `vk*` symbols are
  remapped to `vmaf_priv_vk*` at the C preprocessor level via a
  force-included header — see
  [ADR-0185](../../../docs/adr/0185-vulkan-hide-volk-symbols.md) +
  [ADR-0198](../../../docs/adr/0198-volk-priv-remap-static-archive.md) +
  [ADR-0200](../../../docs/adr/0200-volk-priv-remap-pkgconfig-leak-fix.md).
  Both the shared `.so` and static `.a` builds drop zero `vk*`
  GLOBAL symbols; this is load-bearing for BtbN-style fully-static
  FFmpeg link environments.
- **`enable_vulkan` meson option** defaults to `disabled`. The
  scaffold from [ADR-0175](../../../docs/adr/0175-vulkan-backend-scaffold.md)
  has zero runtime dependencies; the runtime PR
  ([ADR-0127](../../../docs/adr/0127-vulkan-compute-backend.md)) wires
  `dependency('vulkan')` + volk + glslc + VMA.
- **GLSL FMA contraction is OFF for precision-critical kernels.**
  Compile shaders with `-O0` or use `precise` / `NoContraction`
  decorations on the load-bearing accumulators (matches CUDA
  `--fmad=false` and SYCL `-fp-model=precise`). Whenever you add
  a new GPU twin, run the cross-backend gate at the contracted
  `places` precision target before declaring success.

## Rebase-sensitive invariants

- **VkImage zero-copy import surface
  ([ADR-0184](../../../docs/adr/0184-vulkan-image-import-scaffold.md) +
  [ADR-0186](../../../docs/adr/0186-vulkan-image-import-impl.md))**:
  `vmaf_vulkan_import_image` / `vmaf_vulkan_wait_compute` /
  `vmaf_vulkan_read_imported_pictures` /
  `vmaf_vulkan_state_init_external` /
  `vmaf_vulkan_state_build_pictures` are public symbols consumed
  by `ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch`.
  Touching the surface requires updating that patch in the same
  PR (CLAUDE.md §12 r14). Synchronous v1 design: per-state ref/dis
  staging `VkBuffer` pair (HOST_VISIBLE, DATA_ALIGN-strided),
  `vkCmdCopyImageToBuffer` + timeline-semaphore wait per frame.
  Async pending-fence v2 + true zero-copy GPU compute deferred.
- **Header purity**: VkImage / VkSemaphore / VkDevice cross the
  ABI as `uintptr_t` (mirrors `libvmaf_cuda.h`). Do not include
  `<vulkan/vulkan.h>` from the public header.
- **Per-state dispatch strategy
  ([ADR-0181](../../../docs/adr/0181-feature-characteristics-registry.md))**:
  `dispatch_strategy.c` consumes the per-feature
  `VmafFeatureCharacteristics` descriptor + frame dims + env
  override (`VMAF_VULKAN_DISPATCH=feature:strategy,...`) and
  returns `PRIMARY_CMDBUF` vs `SECONDARY_CMDBUF_REUSE` (stub —
  reuse path lands with T7-18 follow-up).
- **GPU long-tail terminus** — every registered feature
  extractor now has at least one Vulkan twin. Cambi was the
  terminus, integrated by PR #196 (T7-36, ADR-0210 placeholder)
  in the hybrid host/GPU pattern from
  [ADR-0205](../../../docs/adr/0205-cambi-gpu-feasibility.md).
  Future feature additions need a same-PR Vulkan twin OR an
  explicit ADR-recorded deferral.
- **Cross-backend gate
  ([ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md))**:
  the lavapipe lane (`vulkan-parity-matrix-gate`) runs on every
  PR. Adding a new GLSL kernel requires a `FEATURE_TOLERANCE`
  entry if it relaxes places=4, plus a row in
  [`docs/development/cross-backend-gate.md`](../../../docs/development/cross-backend-gate.md).

## Governing ADRs

- [ADR-0127](../../../docs/adr/0127-vulkan-compute-backend.md) —
  Vulkan compute backend strategic decision.
- [ADR-0175](../../../docs/adr/0175-vulkan-backend-scaffold.md) —
  scaffold-only audit-first PR (T5-1).
- [ADR-0176](../../../docs/adr/0176-vulkan-vif-cross-backend-gate.md) —
  Vulkan VIF cross-backend gate (lavapipe + Arc A380 lanes).
- [ADR-0177](../../../docs/adr/0177-vulkan-motion-kernel.md),
  [ADR-0178](../../../docs/adr/0178-vulkan-adm-kernel.md),
  [ADR-0185](../../../docs/adr/0185-vulkan-hide-volk-symbols.md),
  [ADR-0186](../../../docs/adr/0186-vulkan-image-import-impl.md),
  [ADR-0187](../../../docs/adr/0187-ciede-vulkan.md)..
  [ADR-0201](../../../docs/adr/0201-ssimulacra2-vulkan-kernel.md) —
  per-kernel records.
- [ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate.

## Build

```bash
meson setup build -Denable_vulkan=enabled
ninja -C build
```

Requires `dependency('vulkan')`, `glslc` (Vulkan SDK or shaderc
package), and the bundled volk + VMA subprojects. The lavapipe
software rasteriser (Mesa) is sufficient for the cross-backend
gate; physical GPUs (Arc A380, RTX 4090) cover the advisory lanes.
