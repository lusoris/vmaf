# Included from libvmaf/src/meson.build via `subdir('hip')`. Paths
# below are relative to libvmaf/src/hip/. The feature stubs live
# under libvmaf/src/feature/hip/, so we use a `..` walk to reach
# them — same shape ADR-0175 uses for the Vulkan scaffold.
#
# Scaffold-only (ADR-0209 / T7-10): every entry point returns -ENOSYS
# until the kernels land, so the build has no HIP / ROCm SDK
# requirement yet. The dependency() probe below stays optional and
# is only consulted if the toolchain happens to expose pkg-config
# metadata for `hip-lang`; the scaffold compiles cleanly on a stock
# Ubuntu runner without any AMD packages installed.
hip_sources += files(
    'common.c',
    'picture_hip.c',
    'dispatch_strategy.c',
    '../feature/hip/adm_hip.c',
    '../feature/hip/vif_hip.c',
    '../feature/hip/motion_hip.c',
)

# Optional probe — does not gate the build. Real linkage against
# `hip-lang` / `amdhip64` arrives with the runtime PR; this line
# documents the intended dependency name for that follow-up.
hip_runtime_dep = dependency('hip-lang', required: false)
if hip_runtime_dep.found()
    hip_deps += hip_runtime_dep
endif

# TODO (post-scaffold): hipcc-driven kernel compilation, hipMallocAsync
# arena, ROCm 6.x runtime linkage. Until then the TUs above use only
# C99 + errno.h.
