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
#
# `feature/hip/ciede_hip.c` is the third kernel-template consumer
# (ADR-0257) — registers `vmaf_fex_ciede_hip` and mirrors the CUDA
# twin (`feature/cuda/integer_ciede_cuda.c`). Same scaffold posture:
# the consumer compiles host-only and `init()` surfaces -ENOSYS
# through the kernel-template helpers until T7-10b lands real
# `hipStreamCreate` / `hipMemcpyAsync` bodies.
#
# `feature/hip/float_moment_hip.c` is the fourth kernel-template
# consumer (ADR-0258) — registers `vmaf_fex_float_moment_hip` and
# mirrors the CUDA twin (`feature/cuda/integer_moment_cuda.c`).
# Same scaffold posture as the third consumer; emits four features
# (`float_moment_ref{1st,2nd}`, `float_moment_dis{1st,2nd}`) once
# the runtime kernel arrives.
# `feature/hip/float_ansnr_hip.c` is the fifth kernel-template
# consumer (ADR-0266) — registers `vmaf_fex_float_ansnr_hip` and
# mirrors the CUDA twin (`feature/cuda/float_ansnr_cuda.c`). Same
# scaffold posture as the prior consumers; the submit path
# intentionally bypasses `vmaf_hip_kernel_submit_pre_launch`
# because the kernel writes per-block (sig, noise) float partials
# directly (no atomic, no memset).
#
# `feature/hip/integer_motion_v2_hip.c` is the sixth kernel-template
# consumer (ADR-0267) — registers `vmaf_fex_integer_motion_v2_hip`
# and mirrors the CUDA twin
# (`feature/cuda/integer_motion_v2_cuda.c`). Pins the **temporal**
# extractor shape (`VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag,
# `flush()` callback) and the ping-pong buffer carry-over needed
# for motion-class metrics. Same scaffold posture: `init()`
# returns -ENOSYS until T7-10b.
hip_sources += files(
    'common.c',
    'picture_hip.c',
    'dispatch_strategy.c',
    'kernel_template.c',
    '../feature/hip/adm_hip.c',
    '../feature/hip/vif_hip.c',
    '../feature/hip/motion_hip.c',
    '../feature/hip/integer_psnr_hip.c',
    '../feature/hip/ciede_hip.c',
    '../feature/hip/float_moment_hip.c',
    '../feature/hip/float_ansnr_hip.c',
    '../feature/hip/integer_motion_v2_hip.c',
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
