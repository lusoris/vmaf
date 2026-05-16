Add CUDA extractor parity smoke tests for the top-5 CHUG extractors
(adm_cuda, float_vif_cuda, motion_v2_cuda, psnr_cuda, ssimulacra2_cuda).
Each test skips cleanly when no CUDA device is present and is tagged
`suite: ['fast', 'gpu']` so `meson test -C build --suite=gpu` runs all
five plus the existing `test_integer_cambi_sycl`. Closes the gap
identified in audit-test-coverage-2026-05-16 §1 (60+ GPU symbols with
zero C unit-test coverage; `meson test` gave no signal when CUDA CI jobs
were skipped).
