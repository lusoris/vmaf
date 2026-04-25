# Included from libvmaf/src/meson.build via `subdir('vulkan')`. Paths
# below are relative to libvmaf/src/vulkan/. The feature kernels live
# under libvmaf/src/feature/vulkan/.
#
# T5-1b runtime wiring (post-T5-1 scaffold, ADR-0175 follow-up):
#  - dependency('vulkan'): system Vulkan loader (libvulkan.so / vulkan-1.dll).
#  - volk: thin Vulkan function loader so we can dlopen at runtime
#    instead of statically linking every entry point. Bundled via
#    libvmaf/subprojects/volk.wrap (vulkan-sdk-1.4.341.0 sha256-pinned).
#  - vk-mem-alloc (VMA): Vulkan Memory Allocator, single-header.
#    Bundled via libvmaf/subprojects/vk-mem-alloc.wrap (v3.3.0 pinned).
#  - glslc: GLSL → SPIR-V compiler from the Vulkan SDK. Used in a
#    custom_target per .comp shader; the .spv blobs are then embedded
#    via xxd-style header generation so the .so carries them inline.

vulkan_sdk_dep = dependency('vulkan', required : true)
volk_dep       = dependency('volk',        fallback : ['volk', 'volk_dep'])
vma_dep        = dependency('vk-mem-alloc', fallback : ['vk-mem-alloc', 'vk_mem_alloc_dep'])

glslc = find_program('glslc', required : true)

# Compute shaders → SPIR-V → embedded C header. Each `.comp` becomes a
# pair: a `<name>.spv` binary + a `<name>_spv.h` header that declares
# `static const uint32_t <name>_spv[]` for direct embedding into the
# host-side dispatch wrapper.
vulkan_shader_sources = [
    'shaders/vif.comp',
]

vulkan_shader_headers = []
foreach shader : vulkan_shader_sources
    shader_name = shader.split('/')[-1].replace('.comp', '')
    spv = custom_target(shader_name + '.spv',
        input : '../feature/vulkan/' + shader,
        output : shader_name + '.spv',
        command : [glslc, '--target-env=vulkan1.3', '-O', '@INPUT@', '-o', '@OUTPUT@'],
    )
    embed = custom_target(shader_name + '_spv.h',
        input : spv,
        output : shader_name + '_spv.h',
        command : ['python3', '-c',
            'import sys; data=open(sys.argv[1],"rb").read(); ' +
            'words=[int.from_bytes(data[i:i+4],"little") for i in range(0,len(data),4)]; ' +
            'name=sys.argv[3]; ' +
            'out=open(sys.argv[2],"w"); ' +
            'out.write("/* Auto-generated from " + name + ".comp by libvmaf/src/vulkan/meson.build */\\n"); ' +
            'out.write("static const unsigned int " + name + "_spv[] = {\\n  "); ' +
            'out.write(", ".join(f"0x{w:08x}" for w in words)); ' +
            'out.write("\\n};\\n"); ' +
            'out.write(f"static const unsigned int " + name + "_spv_size = " + str(len(words)*4) + ";\\n"); ' +
            'out.close()',
            '@INPUT@', '@OUTPUT@', shader_name],
    )
    vulkan_shader_headers += embed
endforeach

vulkan_sources += files(
    'common.c',
    'picture_vulkan.c',
    'vma_impl.cpp',
    '../feature/vulkan/adm_vulkan.c',
    '../feature/vulkan/vif_vulkan.c',
    '../feature/vulkan/motion_vulkan.c',
)

vulkan_sources += vulkan_shader_headers

vulkan_deps += vulkan_sdk_dep
vulkan_deps += volk_dep
vulkan_deps += vma_dep
