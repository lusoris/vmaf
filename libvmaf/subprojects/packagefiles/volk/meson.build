project('volk', 'c',
    version : '1.4.341',
    license : 'MIT',
    meson_version : '>= 0.56.1',
    default_options : ['c_std=c11', 'warning_level=0'])

# volk is a thin Vulkan loader; it dlopens libvulkan.so at runtime so we
# do NOT link against the loader here. The host binary that uses volk
# must call `volk_initialize()` before any Vulkan function.
volk_inc = include_directories('.')
volk_lib = static_library('volk',
    ['volk.c'],
    include_directories : volk_inc,
    pic : true,
    install : false)

volk_dep = declare_dependency(
    include_directories : volk_inc,
    link_with : volk_lib,
    compile_args : ['-DVK_NO_PROTOTYPES'])
