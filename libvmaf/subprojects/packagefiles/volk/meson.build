project('volk', 'c',
    version : '1.4.341',
    license : 'MIT',
    meson_version : '>= 0.56.1',
    default_options : ['c_std=c11', 'warning_level=0'])

py3 = import('python').find_installation('python3')

# ADR-0198: prefix volk's `vk*` symbols with `vmaf_priv_` via a
# force-included header so libvmaf.a (fully-static builds) can be
# linked alongside a consumer's libvulkan.a without colliding on
# every Vulkan API name. Generates from the upstream volk.h so
# coverage tracks the wrap version (1.4.341 → 784 entry points).
volk_priv_remap_h = custom_target('volk_priv_remap_h',
    output : 'volk_priv_remap.h',
    input  : 'volk.h',
    command : [py3, files('gen_priv_remap.py'), '@INPUT@', '@OUTPUT@'])

# Path to the generated header, used for `-include`. Lives in the
# build dir; consumers that pull in `volk_dep` get it via the
# generated_sources path below + the `-include` compile arg.
volk_priv_remap_h_path = meson.current_build_dir() / 'volk_priv_remap.h'

# volk is a thin Vulkan loader; it dlopens libvulkan.so at runtime so we
# do NOT link against the loader here. The host binary that uses volk
# must call `volk_initialize()` before any Vulkan function.
volk_inc = include_directories('.')
volk_lib = static_library('volk',
    ['volk.c', volk_priv_remap_h],
    include_directories : volk_inc,
    c_args : ['-include', volk_priv_remap_h_path],
    pic : true,
    install : false)

volk_dep = declare_dependency(
    include_directories : volk_inc,
    link_with : volk_lib,
    sources : volk_priv_remap_h,
    compile_args : ['-DVK_NO_PROTOTYPES',
                    '-include', volk_priv_remap_h_path])
