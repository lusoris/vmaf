# Included from libvmaf/src/meson.build via `subdir('metal')`. Paths
# below are relative to libvmaf/src/metal/. The first feature stub
# lives under libvmaf/src/feature/metal/, so we use a `..` walk to
# reach it — same shape ADR-0212 uses for the HIP scaffold and
# ADR-0175 uses for the Vulkan scaffold.
#
# T8-1b runtime (ADR-0420): the backend's three runtime TUs are
# Objective-C++ (`.mm`) and use Apple's Foundation / Metal frameworks
# directly. The per-feature scaffolds under `feature/metal/` stay
# pure-C (they pass through to the kernel_template helpers and the
# extractor table) — the runtime PR's contract is that consumer TUs
# do NOT have to be Obj-C++ to use the Metal backend.
#
# Obj-C++ TUs:
#   - common.mm           — MTLDevice / MTLCommandQueue lifecycle
#   - picture_metal.mm    — MTLBuffer allocator (Shared storage)
#   - kernel_template.mm  — command-queue / event-pair / blit fill
#
# Pure-C TUs (kept for compatibility with consumers and Linux dev hosts):
#   - dispatch_strategy.c                — no Metal API surface yet
#   - feature/metal/*.c                  — extractor registration only
#
# `metal_sources` is consumed by libvmaf_feature_static_lib in the
# parent meson.build; both file kinds (`.c` and `.mm`) end up in the
# same compile_commands.json. Meson auto-detects Obj-C / Obj-C++
# compilers when the file extension is `.mm` (Apple Clang ships them
# as part of clang).
# T8-1b runtime needs Obj-C++ added to the project's language list
# before any `.mm` source can be added to a target — meson errors with
# "No host machine compiler for 'src/metal/common.mm'" otherwise (CI
# run 25685041032 / job 75406405007). `add_languages` is the canonical
# meson API for late-binding a language; gated behind the parent's
# `is_metal_enabled` (Darwin-only). `required: true` because by the
# time we reach this branch the build IS Metal-enabled.
add_languages('objcpp', native: false, required: true)

metal_sources += files(
    # Runtime TUs (T8-1b, Obj-C++ via Apple Clang).
    'common.mm',
    'picture_metal.mm',
    'kernel_template.mm',
    # Dispatch-strategy stub — pure C, no Metal SDK needed.
    'dispatch_strategy.c',
    # T8-1c first real kernel (ADR-0421): integer_motion_v2 dispatch
    # converted to .mm so it can build MTLComputePipelineState and
    # dispatch `motion_v2_kernel_{8,16}bpc` from the embedded metallib.
    '../feature/metal/integer_motion_v2_metal.mm',
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

# Apple framework linkage. macOS ships these in
# /System/Library/Frameworks; `dependency('Foundation')` /
# `dependency('Metal')` resolves to a meson framework-dependency that
# threads `-framework Foundation -framework Metal` into the link line.
#
# On non-macOS hosts the scaffold C files compile cleanly with no
# Metal SDK; the `.mm` TUs do not compile there (Obj-C++ is mac-only
# in practice), so we only register the framework dependencies — and
# the `.mm` sources — when host_machine.system() is 'darwin'. The
# parent meson.build already gates `subdir('metal')` behind
# is_metal_enabled which is `host == darwin` AND option != disabled,
# so we don't have to re-check here, but the framework probes are
# `required: true` because by the time we reach this branch we DO
# need Metal.
foundation_dep = dependency('Foundation', required: true)
metal_dep      = dependency('Metal', required: true)
metalkit_dep   = dependency('MetalKit', required: false)
metal_deps += foundation_dep
metal_deps += metal_dep
if metalkit_dep.found()
    metal_deps += metalkit_dep
endif

# Obj-C++ compile arguments for the .mm TUs. ARC is on so the bridge-
# retain/transfer casts in the implementation are real reference
# counts (the alternative is manual retain/release, which the .mm
# files do not use). `-fno-objc-arc-exceptions` keeps the codegen
# small — these TUs don't catch Obj-C exceptions.
metal_objcpp_args = [
    '-fobjc-arc',
    '-fno-objc-arc-exceptions',
    # Avoid pulling all of Foundation into every TU; this matches
    # MetalCpp's recommended posture.
    '-fobjc-weak',
]

# Meson's `objcpp_args` is the canonical way to pass per-TU flags for
# Obj-C++ sources mixed into the same target. Adding the args to a
# top-level `add_project_arguments(..., language: 'objcpp')` call
# would leak across other backends; keeping them scoped here.
add_project_arguments(metal_objcpp_args, language: 'objcpp')

# ---------------------------------------------------------------------
# Metallib compilation (T8-1c / ADR-0421).
#
# Pipeline: `.metal` → `.air` → `.metallib` via `xcrun -sdk macosx
# metal` and `xcrun -sdk macosx metallib`. The compiled metallib is
# then embedded into the libvmaf shared/static library via the
# linker's `-sectcreate __TEXT __metallib <path>` flag — the
# consumer's .mm code (`integer_motion_v2_metal.mm`) reads the
# embedded byte range via `section$start$__TEXT$__metallib` /
# `section$end$__TEXT$__metallib` symbols and hands the bytes to
# `[MTLDevice newLibraryWithData:]` at init time. No filesystem path
# dependency at runtime.
#
# Same embedded-blob pattern the CUDA backend uses for its cubin
# (`vmaf_cuda_compiled_kernels_start` / `_end`).
xcrun = find_program('xcrun', required: true)

# Compile each .metal kernel to .air (intermediate IR) via xcrun
# metal. Single target per kernel, hard-coded names — meson's
# `foreach` over file objects doesn't expose a stable basename to
# stamp into custom_target names. Adding kernels (T8-1d through
# T8-1k) is a one-line edit per kernel here.
mv2_air = custom_target('integer_motion_v2_air',
    input: '../feature/metal/integer_motion_v2.metal',
    output: 'integer_motion_v2.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

metal_air_files = [mv2_air]

# Link all .air files into a single default.metallib.
metallib_target = custom_target('libvmaf-metallib',
    input: metal_air_files,
    output: 'default.metallib',
    command: [xcrun, '-sdk', 'macosx', 'metallib',
              '@INPUT@', '-o', '@OUTPUT@'],
)

# Threading the metallib into the libvmaf binary's __TEXT,__metallib
# section is done via the link line. The parent meson.build extends
# the library's link_args with the -sectcreate flag below; surface
# the metallib path so it can pick it up.
metal_deps += declare_dependency(
    sources: metallib_target,
    link_args: ['-Wl,-sectcreate,__TEXT,__metallib,' +
                metallib_target.full_path()],
)
