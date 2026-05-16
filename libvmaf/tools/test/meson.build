if get_option('enable_cuda')
    test_vmaf_cuda_gpumask = find_program('test_vmaf_cuda_gpumask.sh')
    test('test_vmaf_cuda_gpumask', test_vmaf_cuda_gpumask)
endif

# T6-3b / ADR-0222 — per-shot CRF predictor sidecar smoke test.
test_vmaf_per_shot = find_program('test_vmaf_per_shot.sh')
test('test_vmaf_per_shot', test_vmaf_per_shot,
     env: [
         'MESON_BUILD_ROOT=' + meson.project_build_root(),
         'MESON_SOURCE_ROOT=' + meson.project_source_root(),
     ])
