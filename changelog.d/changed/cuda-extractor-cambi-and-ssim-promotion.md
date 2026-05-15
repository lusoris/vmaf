- The K150K/CHUG feature extractor now routes `float_ssim` through
  the CUDA primary pass (with explicit `scale=1` per the
  `float_ssim_cuda` v1 contract) instead of the CPU residual pass.
  `cambi` was originally also planned for promotion but stays on the
  CPU residual: `cambi_cuda` segfaults on every input on the
  rebuilt 2026-05-15 binary (Issue #857). Per-clip wall time on CUDA
  workers improves by roughly the CPU SSIM cost; CAMBI remains on
  the CPU residual until the CUDA-side bug is fixed.
