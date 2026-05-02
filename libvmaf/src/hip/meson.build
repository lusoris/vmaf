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
# Kernel-template helpers (T7-10 first-consumer / ADR-0241):
# `kernel_template.c` carries the per-frame async lifecycle helpers
# (stream + event pair + readback alloc/free) that mirror
# `cuda/kernel_template.h`. Bodies return -ENOSYS while the runtime
# is absent; the runtime PR (T7-10b) replaces them with real HIP
# calls without touching consumer TUs.
#
# `feature/hip/integer_psnr_hip.c` is the first kernel-template
# consumer — registers `vmaf_fex_psnr_hip` and exercises every
# template helper in init/submit/collect/close. -ENOSYS at the
# template layer surfaces verbatim through the consumer.
hip_sources += files(
    'common.c',
    'picture_hip.c',
    'dispatch_strategy.c',
    'kernel_template.c',
    '../feature/hip/adm_hip.c',
    '../feature/hip/vif_hip.c',
    '../feature/hip/motion_hip.c',
    '../feature/hip/integer_psnr_hip.c',
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
