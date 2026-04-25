# Included from libvmaf/src/meson.build via `subdir('vulkan')`. Paths
# below are relative to libvmaf/src/vulkan/. The feature stubs live
# under libvmaf/src/feature/vulkan/, so we use a `..` walk to reach
# them — same shape ADR-0175 uses for the dnn subdirectory.
#
# Scaffold-only (ADR-0175 / T5-1): every entry point returns -ENOSYS
# until the kernels land, so the build has no Vulkan SDK requirement
# yet. The dependency() lines below stay commented until the first
# real kernel PR brings volk + glslc + VMA in deliberately.
vulkan_sources += files(
    'common.c',
    'picture_vulkan.c',
    '../feature/vulkan/adm_vulkan.c',
    '../feature/vulkan/vif_vulkan.c',
    '../feature/vulkan/motion_vulkan.c',
)

# TODO (post-scaffold): vulkan_deps += dependency('vulkan', required: true)
# + volk wrapping + glslc shader compilation + VMA bundled subproject.
