# Included from libvmaf/src/meson.build via `subdir('metal')`. Paths
# below are relative to libvmaf/src/metal/. The first feature stub
# lives under libvmaf/src/feature/metal/, so we use a `..` walk to
# reach it — same shape ADR-0212 uses for the HIP scaffold and
# ADR-0175 uses for the Vulkan scaffold.
#
# Scaffold-only (ADR-0338 / T8-1): every entry point returns -ENOSYS
# until the runtime PR (T8-1b) lands. The build has no hard Metal
# SDK requirement on non-macOS hosts: the dependency() probes below
# are `required: false` so the scaffold compiles cleanly on Linux /
# Windows runners (where the option resolves to disabled via the
# auto-probe in libvmaf/src/meson.build) and on macOS where the
# system frameworks are guaranteed to be present.
#
# Kernel-template helpers (T8-1 / ADR-0338): `kernel_template.c`
# carries the per-frame async lifecycle helpers (command queue +
# event pair + buffer alloc/free) that mirror
# `hip/kernel_template.h` modulo the unified-memory buffer collapse
# (one MTLBuffer with `MTLResourceStorageModeShared` instead of the
# (device, pinned-host) readback pair). Bodies return -ENOSYS while
# the runtime is absent; the runtime PR (T8-1b) replaces them with
# real Metal calls without touching consumer TUs.
#
# `feature/metal/integer_motion_v2_metal.c` is the first kernel-
# template consumer — registers `vmaf_fex_integer_motion_v2_metal`
# (extractor name `motion_v2_metal`, TEMPORAL flag) and exercises
# every template helper in init/submit/collect/close. -ENOSYS at
# the template layer surfaces verbatim through the consumer.
metal_sources += files(
    'common.c',
    'picture_metal.c',
    'dispatch_strategy.c',
    'kernel_template.c',
    '../feature/metal/integer_motion_v2_metal.c',
    # T8-1 batch-1 scaffolds (ADR-0361): 3 additional consumers.
    '../feature/metal/integer_psnr_metal.c',
    '../feature/metal/float_ssim_metal.c',
    '../feature/metal/integer_motion_metal.c',
    # T8-1 batch-2 scaffolds (ADR-0361): 4 additional float consumers.
    '../feature/metal/float_psnr_metal.c',
    '../feature/metal/float_ansnr_metal.c',
    '../feature/metal/float_motion_metal.c',
    '../feature/metal/float_moment_metal.c',
)

# Optional probe — does not gate the build on non-macOS hosts. Real
# linkage against `Metal.framework` / `MetalKit.framework` arrives
# with the runtime PR (T8-1b); the scaffold's TUs are pure C99 +
# errno.h and link cleanly without the frameworks. The probe runs
# anyway so a host that already has the SDK (every macOS dev box)
# sees the dependency resolve and the runtime PR's diff is a
# `required: true` flip rather than a new dependency() block.
metal_dep = dependency('Metal', required: false)
metalkit_dep = dependency('MetalKit', required: false)
if metal_dep.found()
    metal_deps += metal_dep
endif
if metalkit_dep.found()
    metal_deps += metalkit_dep
endif

# TODO (post-scaffold, T8-1b runtime):
#   - xcrun-metal-driven .metal -> .air -> .metallib compilation
#   - MetalCpp single-header inclusion (`<Metal/Metal.hpp>`)
#   - .mm / Objective-C++ TUs for the device-init bridge
#   - link `-framework Metal -framework MetalKit -framework Foundation`
# Until then the TUs above use only C99 + errno.h and the build is
# host-toolchain-agnostic.
