if not get_option('enable_tests')
    subdir_done()
endif

test_inc = include_directories('.')

test_context = executable('test_context',
    ['test.c', 'test_context.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies:[stdatomic_dependency, pthread_dependency, cuda_dependency, sycl_dependency],
)

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c', '../src/ref.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies:[stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency],
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
    dependencies: [pthread_dependency, cuda_dependency, sycl_dependency],
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
    dependencies : [pthread_dependency, thread_lib, cuda_dependency, sycl_dependency],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_predict = executable('test_predict',
    ['test.c', 'test_predict.c', '../src/dict.c', '../src/metadata_handler.c',
     '../src/feature/feature_collector.c', '../src/feature/alias.c', '../src/model.c', '../src/log.c',
     '../src/read_json_model.c', '../src/pdjson.c', json_model_c_sources, '../src/feature/feature_name.c', '../src/feature/feature_extractor.c',],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    dependencies : [pthread_dependency, thread_lib, cuda_dependency, sycl_dependency],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_feature_extractor = executable('test_feature_extractor',
    # dnn_sources pulled in directly so feature_lpips.c can resolve its
    # libvmaf/dnn.h references (session_open/run/close,
    # vmaf_tensor_from_rgb_imagenet) without linking against libvmaf.so.
    ['test.c', 'test_feature_extractor.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency,
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
    dependencies: [pthread_dependency, cuda_dependency, sycl_dependency],
)

# LPIPS registration tests — verify feature wiring without depending on a
# real ORT session. Links against libvmaf_feature_static_lib for the
# extractor list and pulls in opt.c for the VmafOption typedef.
test_lpips = executable('test_lpips',
    ['test.c', 'test_lpips.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, cuda_dependency, sycl_dependency,
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
    dependencies: [pthread_dependency, cuda_dependency, sycl_dependency],
)

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

# Bit-exactness between the scalar-separable reference and the x86 SIMD
# paths. Links the x86 AVX2/AVX-512 object libraries directly so the
# kernels can be called without CPU-flag gating from inside the test.
# See docs/adr/0125-ms-ssim-decimate-simd.md.
if host_machine.cpu_family() == 'x86_64' or host_machine.cpu_family() == 'x86'
test_ms_ssim_decimate = executable('test_ms_ssim_decimate',
    ['test.c', 'test_ms_ssim_decimate.c',
     '../src/mem.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/')],
    c_args : vmaf_cflags_common,
    dependencies : [math_lib, pthread_dependency, sycl_dependency],
    objects : [
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

if get_option('enable_cuda')
test_ring_buffer = executable('test_ring_buffer',
    ['test.c', 'test_ring_buffer.c', '../src/cuda/ring_buffer.c', '../src/cuda/picture_cuda.c'],
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

test('test_ring_buffer', test_ring_buffer)
test('test_cuda_pic_preallocation', test_cuda_pic_preallocation)
endif

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
test('test_cambi', test_cambi)
test('test_luminance_tools', test_luminance_tools)
test('test_cli_parse', test_cli_parse)
test('test_psnr', test_psnr)
test('test_framesync', test_framesync)
test('test_propagate_metadata', test_propagate_metadata)
if host_machine.cpu_family() == 'x86_64' or host_machine.cpu_family() == 'x86'
test('test_ms_ssim_decimate', test_ms_ssim_decimate)
endif

subdir('dnn')
