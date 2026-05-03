if not get_option('enable_tests')
    subdir_done()
endif

test_inc = include_directories('.')

test_context = executable('test_context',
    ['test.c', 'test_context.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies:[stdatomic_dependency, pthread_dependency, cuda_dependency, sycl_dependency, vulkan_deps],
)

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c', '../src/ref.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies:[stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps],
)

test_propagate_metadata = executable('test_propagate_metadata',
    ['test.c', 'test_propagate_metadata.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c', '../src/log.c', '../src/predict.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/feature/'), include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency, sycl_dependency, vulkan_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_log = executable('test_log',
    ['test.c', 'test_log.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_thread_pool = executable('test_thread_pool',
    ['test.c', 'test_thread_pool.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : [pthread_dependency, thread_lib, sycl_dependency],
)

test_model = executable('test_model',
    ['test.c', 'test_model.c', '../src/dict.c', '../src/pdjson.c', '../src/read_json_model.c', '../src/log.c', json_model_c_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [vmaf_cflags_common, '-DJSON_MODEL_PATH="'+join_paths(meson.project_source_root(), '../model/')+'"'],
    dependencies : [pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_predict = executable('test_predict',
    ['test.c', 'test_predict.c', '../src/dict.c', '../src/metadata_handler.c',
     '../src/feature/feature_collector.c', '../src/feature/alias.c', '../src/model.c', '../src/log.c',
     '../src/read_json_model.c', '../src/pdjson.c', json_model_c_sources, '../src/feature/feature_name.c', '../src/feature/feature_extractor.c',],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    dependencies : [pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_feature_extractor = executable('test_feature_extractor',
    # dnn_sources pulled in directly so feature_lpips.c can resolve its
    # libvmaf/dnn.h references (session_open/run/close,
    # vmaf_tensor_from_rgb_imagenet) without linking against libvmaf.so.
    ['test.c', 'test_feature_extractor.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c',
     dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

test_dict = executable('test_dict',
    ['test.c', 'test_dict.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_opt = executable('test_opt',
    ['test.c', 'test_opt.c', '../src/opt.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [math_lib, pthread_dependency, sycl_dependency],
)

test_cpu = executable('test_cpu',
    ['test.c', 'test_cpu.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    objects : libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
    dependencies: [pthread_dependency, sycl_dependency],
)

test_ref = executable('test_ref',
    ['test.c', 'test_ref.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_feature = executable('test_feature',
    ['test.c', 'test_feature.c', '../src/feature/alias.c', '../src/dict.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_ciede = executable('test_ciede',
    ['test.c', 'test_ciede.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency, sycl_dependency, vulkan_deps],
)

# LPIPS registration tests — verify feature wiring without depending on a
# real ORT session. Links against libvmaf_feature_static_lib for the
# extractor list and pulls in opt.c for the VmafOption typedef.
test_lpips = executable('test_lpips',
    ['test.c', 'test_lpips.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# FastDVDnet temporal pre-filter (T6-7) — same registration-only test
# pattern as test_lpips. Verifies the 5-frame extractor wiring without
# requiring a real ORT session; CLI smoke against
# model/tiny/fastdvdnet_pre.onnx covers the live inference path.
test_fastdvdnet_pre = executable('test_fastdvdnet_pre',
    ['test.c', 'test_fastdvdnet_pre.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# MobileSal saliency registration tests (T6-2a) — same pattern as
# test_lpips above. Verifies the extractor wires up without depending
# on a real ORT session; full inference is exercised by the CLI smoke
# gate against model/tiny/mobilesal.onnx.
test_mobilesal = executable('test_mobilesal',
    ['test.c', 'test_mobilesal.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# TransNet V2 shot-boundary detector — structural test, no real ORT
# session required (mirrors test_lpips). CLI smoke against
# model/tiny/transnet_v2.onnx covers the live inference path.
test_transnet_v2 = executable('test_transnet_v2',
    ['test.c', 'test_transnet_v2.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

test_cambi = executable('test_cambi',
    ['test.c', 'test_cambi.c', '../src/picture.c', '../src/mem.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency, sycl_dependency, vulkan_deps],
)

# speed_chroma + speed_temporal registration smoke tests — link against
# libvmaf_feature_static_lib for the extractor list (mirrors test_lpips).
# Both extractors live in libvmaf/src/feature/speed.c, which is only
# compiled when -Denable_float=true (see libvmaf/src/meson.build); the
# corresponding entries in feature_extractor.c's extractor list are
# wrapped in `#if VMAF_FLOAT_FEATURES`. Gate the test on the same option
# so a default `meson setup` (enable_float=false) doesn't fail on a
# config that legitimately omits these extractors.
if get_option('enable_float')
test_speed = executable('test_speed',
    ['test.c', 'test_speed.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency, vulkan_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)
endif

test_luminance_tools = executable('test_luminance_tools',
    ['test.c', 'test_luminance_tools.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
)

test_cli_parse = executable('test_cli_parse',
    ['test.c', 'test_cli_parse.c', '../tools/cli_parse.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), include_directories('../tools/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [compat_cflags],
    dependencies: [pthread_dependency, getopt_dependency, sycl_dependency],
)

test_psnr = executable('test_psnr',
    ['test.c', 'test_psnr.c', '../src/picture.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
)

test_framesync = executable('test_framesync',
    ['test.c', 'test_framesync.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [pthread_dependency, thread_lib, sycl_dependency],
)

# Ported from upstream Netflix/vmaf c70debb1 ("libvmaf/test: port new
# adm/vif/speed tests"). The vif_tools and speed_chroma test files from that
# commit depend on upstream symbols (NUM_KERNELSCALES, vif_validate_kernelscale,
# speed_get_antialias_filter, the SpEED feature) that have not been ported to
# the fork; only the adm_csf and barten_csf cases are wired here.
test_adm_csf = executable('test_adm_csf',
    ['test.c', 'test_adm_csf.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, pthread_dependency, sycl_dependency],
)

test_barten_csf = executable('test_barten_csf',
    ['test.c', 'test_barten_csf.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, pthread_dependency, sycl_dependency],
)

# Bit-exactness between the scalar-separable reference and the arch-
# specific SIMD paths (x86 AVX2/AVX-512 and arm64 NEON). Links the
# platform SIMD object libraries directly so the kernels can be called
# without CPU-flag gating from inside the test. See
# docs/adr/0125-ms-ssim-decimate-simd.md.
iqa_convolve_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if iqa_convolve_test_archs.contains(host_machine.cpu_family())
test_iqa_convolve = executable('test_iqa_convolve',
    ['test.c', 'test_iqa_convolve.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

psnr_hvs_avx2_test_archs = ['x86_64', 'x86']
if psnr_hvs_avx2_test_archs.contains(host_machine.cpu_family())
test_psnr_hvs_avx2 = executable('test_psnr_hvs_avx2',
    ['test.c', 'test_psnr_hvs_avx2.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

motion_v2_simd_test_archs = ['x86_64', 'x86']
if motion_v2_simd_test_archs.contains(host_machine.cpu_family())
test_motion_v2_simd = executable('test_motion_v2_simd',
    ['test.c', 'test_motion_v2_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

ssimulacra2_simd_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if ssimulacra2_simd_test_archs.contains(host_machine.cpu_family())
# `-ffp-contract=off` is required for bit-exactness on aarch64:
# GCC 10+ and clang default to `-ffp-contract=fast` on aarch64, which
# auto-fuses plain C `a*b+c` into `fmla`. The scalar reference in the
# test TU would fuse; NEON intrinsics do not fuse (compiler treats
# them as opaque). x86 avoids this because `-mfma` is scoped to the
# AVX2/AVX-512 static libs only, so scalar code can't fuse on x86
# regardless of contract setting. See ADR-0161 follow-up notes.
test_ssimulacra2_simd = executable('test_ssimulacra2_simd',
    ['test.c', 'test_ssimulacra2_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines + ['-ffp-contract=off'],
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

psnr_hvs_neon_test_archs = ['aarch64', 'arm64']
if psnr_hvs_neon_test_archs.contains(host_machine.cpu_family())
test_psnr_hvs_neon = executable('test_psnr_hvs_neon',
    ['test.c', 'test_psnr_hvs_neon.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

moment_simd_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if float_enabled and moment_simd_test_archs.contains(host_machine.cpu_family())
test_moment_simd = executable('test_moment_simd',
    ['test.c', 'test_moment_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

ms_ssim_decimate_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if ms_ssim_decimate_test_archs.contains(host_machine.cpu_family())
test_ms_ssim_decimate = executable('test_ms_ssim_decimate',
    ['test.c', 'test_ms_ssim_decimate.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

test_locale_handling = executable('test_locale_handling',
    ['test.c', 'test_locale_handling.c'],
    include_directories : [
        libvmaf_inc,
        test_inc,
        include_directories('../src/'),
        include_directories('..')
    ],
    c_args : ['-I' + meson.project_build_root() / 'src'] + vmaf_cflags_common,
    dependencies : [pthread_dependency],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

if get_option('enable_cuda')
test_gpu_picture_pool = executable('test_gpu_picture_pool',
    ['test.c', 'test_gpu_picture_pool.c', '../src/gpu_picture_pool.c', '../src/cuda/picture_cuda.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)

test_cuda_pic_preallocation = executable('test_cuda_pic_preallocation',
    ['test.c', 'test_cuda_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)

test('test_gpu_picture_pool', test_gpu_picture_pool)
test('test_cuda_pic_preallocation', test_cuda_pic_preallocation)

# Netflix#1420 — graceful-OOM reducer. Exercises the wholesale
# CHECK_CUDA_GOTO/_RETURN rewrite by requesting an impossible
# 1 TiB cuMemAlloc and asserting the returned errno is -ENOMEM
# rather than an aborted process. GPU-gated: skips cleanly if
# no driver/device is visible to the process.
test_cuda_buffer_alloc_oom = executable('test_cuda_buffer_alloc_oom',
    ['test.c', 'test_cuda_buffer_alloc_oom.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)
test('test_cuda_buffer_alloc_oom', test_cuda_buffer_alloc_oom)

# Netflix#1300 / ADR-0157 — preallocation-leak reducer. Runs N
# init/preallocate/fetch/close cycles with FULL cleanup on each cycle
# (vmaf_close + vmaf_cuda_state_free + vmaf_model_destroy). ASan is the
# gate: pre-fix, the framework leaked the VmafCudaState struct, the
# CudaFunctions dlopen table, and the ring-buffer mutex state on every
# cycle. GPU-gated: skips cleanly if no driver/device is visible.
test_cuda_preallocation_leak = executable('test_cuda_preallocation_leak',
    ['test.c', 'test_cuda_preallocation_leak.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)
test('test_cuda_preallocation_leak', test_cuda_preallocation_leak)
endif

# T6-2b / ADR-0247 — vmaf-roi sidecar core helpers (per-CTU mean reduce
# + saliency-to-QP-offset). Header-only test that compiles
# libvmaf/tools/vmaf_roi_core.h and exercises clamp / sign / monotonicity
# without needing the full binary.
test_vmaf_roi_libm = cc.find_library('m', required : false)
test_vmaf_roi = executable('test_vmaf_roi',
    ['test.c', 'test_vmaf_roi.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../tools/')],
    dependencies: [pthread_dependency, test_vmaf_roi_libm],
)
test('test_vmaf_roi', test_vmaf_roi)

# picture pool test - exercises vmaf_preallocate_pictures / vmaf_fetch_preallocated_picture.
# These symbols are always linked into libvmaf (no build gate), so this test
# always runs.
test_pic_preallocation = executable('test_pic_preallocation',
    ['test.c', 'test_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, thread_lib],
)
test('test_pic_preallocation', test_pic_preallocation)

# ADR-0152 — monotonic-index guard on vmaf_read_pictures (Netflix#910).
test_read_pictures_monotonic = executable('test_read_pictures_monotonic',
    ['test.c', 'test_read_pictures_monotonic.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, thread_lib],
)
test('test_read_pictures_monotonic', test_read_pictures_monotonic)

# ADR-0154 — score_pooled returns -EAGAIN instead of -EINVAL for
# not-yet-written features (Netflix#755).
test_score_pooled_eagain = executable('test_score_pooled_eagain',
    ['test.c', 'test_score_pooled_eagain.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, pthread_dependency, thread_lib],
)
test('test_score_pooled_eagain', test_score_pooled_eagain)

# ADR-0153 — float_ms_ssim init rejects <176x176 (Netflix#1414).
test_float_ms_ssim_min_dim = executable('test_float_ms_ssim_min_dim',
    ['test.c', 'test_float_ms_ssim_min_dim.c', '../src/mem.c', '../src/picture.c',
     '../src/ref.c', '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    cuda_dependency, sycl_dependency, vulkan_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)
test('test_float_ms_ssim_min_dim', test_float_ms_ssim_min_dim)

if get_option('enable_sycl')
test_sycl = executable('test_sycl',
    ['test.c', 'test_sycl.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
    c_args: ['-DHAVE_SYCL=1']
)

test_sycl_pic_preallocation = executable('test_sycl_pic_preallocation',
    ['test.c', 'test_sycl_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
    c_args: ['-DHAVE_SYCL=1']
)

test('test_sycl', test_sycl)
test('test_sycl_pic_preallocation', test_sycl_pic_preallocation)
endif

# ADR-0175 / T5-1: Vulkan backend scaffold smoke. Pins the -ENOSYS
# contract so a future runtime PR can't accidentally enable the
# backend without flipping the smoke expectations.
if get_option('enable_vulkan').enabled()
test_vulkan_smoke = executable('test_vulkan_smoke',
    ['test.c', 'test_vulkan_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_VULKAN=1']
)
test('test_vulkan_smoke', test_vulkan_smoke)

# T7-29 part 4 (ADR-0251): contract smoke for the v2 async
# pending-fence ring. Public-API only — the GPU-end-to-end
# ring-wrap correctness path lives in the FFmpeg filter
# downstream (T7-29 part 3) and the cross-backend gate.
test_vulkan_async_pending_fence = executable('test_vulkan_async_pending_fence',
    ['test.c', 'test_vulkan_async_pending_fence.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_VULKAN=1']
)
test('test_vulkan_async_pending_fence', test_vulkan_async_pending_fence)

# ADR-0238: contract smoke for the public Vulkan picture-preallocation
# surface (vmaf_vulkan_preallocate_pictures / vmaf_vulkan_picture_fetch).
test_vulkan_pic_preallocation = executable('test_vulkan_pic_preallocation',
    ['test.c', 'test_vulkan_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_VULKAN=1']
)
test('test_vulkan_pic_preallocation', test_vulkan_pic_preallocation)
endif

# ADR-0209 / T5-2: embedded MCP scaffold smoke. Pins the -ENOSYS
# contract on every public entry point so a future runtime PR
# (T5-2b) cannot wire a transport without flipping the expectations.
if get_option('enable_mcp')
test_mcp_smoke = executable('test_mcp_smoke',
    ['test.c', 'test_mcp_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_MCP=1']
)
test('test_mcp_smoke', test_mcp_smoke)
endif

# ADR-0209 / T7-10: HIP (AMD ROCm) backend scaffold smoke. Same pattern
# as the Vulkan scaffold above — pins the -ENOSYS contract for every
# public C-API entry point until the runtime PR lands.
if get_option('enable_hip') == true
test_hip_smoke = executable('test_hip_smoke',
    ['test.c', 'test_hip_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_HIP=1']
)
test('test_hip_smoke', test_hip_smoke)
endif

test('test_picture', test_picture)
test('test_feature_collector', test_feature_collector)
test('test_thread_pool', test_thread_pool)
test('test_model', test_model)
test('test_predict', test_predict)
test('test_feature_extractor', test_feature_extractor)
test('test_dict', test_dict)
test('test_opt', test_opt)
test('test_cpu', test_cpu)
test('test_ref', test_ref)
test('test_feature', test_feature)
test('test_ciede', test_ciede)
test('test_lpips', test_lpips)
test('test_fastdvdnet_pre', test_fastdvdnet_pre)
test('test_mobilesal', test_mobilesal)
test('test_transnet_v2', test_transnet_v2)
test('test_cambi', test_cambi)
if get_option('enable_float')
test('test_speed', test_speed)
endif
test('test_luminance_tools', test_luminance_tools)
test('test_cli_parse', test_cli_parse)
test('test_psnr', test_psnr)
test('test_framesync', test_framesync)
test('test_propagate_metadata', test_propagate_metadata)
test('test_adm_csf', test_adm_csf)
test('test_barten_csf', test_barten_csf)
if ms_ssim_decimate_test_archs.contains(host_machine.cpu_family())
test('test_ms_ssim_decimate', test_ms_ssim_decimate)
endif

if float_enabled and moment_simd_test_archs.contains(host_machine.cpu_family())
test('test_moment_simd', test_moment_simd)
endif

if iqa_convolve_test_archs.contains(host_machine.cpu_family())
test('test_iqa_convolve', test_iqa_convolve)
endif

if psnr_hvs_avx2_test_archs.contains(host_machine.cpu_family())
test('test_psnr_hvs_avx2', test_psnr_hvs_avx2)
endif

if motion_v2_simd_test_archs.contains(host_machine.cpu_family())
test('test_motion_v2_simd', test_motion_v2_simd)
endif

if psnr_hvs_neon_test_archs.contains(host_machine.cpu_family())
test('test_psnr_hvs_neon', test_psnr_hvs_neon)
endif

if ssimulacra2_simd_test_archs.contains(host_machine.cpu_family())
test('test_ssimulacra2_simd', test_ssimulacra2_simd)
endif

test('test_locale_handling', test_locale_handling)

subdir('dnn')
