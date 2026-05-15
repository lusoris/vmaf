- The K150K/CHUG feature extractor now routes `cambi` and
  `float_ssim` through the CUDA pass instead of the CPU residual
  pass. Both have shipped CUDA implementations
  (`libvmaf/src/feature/cuda/integer_cambi_cuda.c` +
  `float_ssim_cuda.c`) for some time; the residual-pass routing was
  historical from before they landed. Per-clip wall time on CUDA
  workers should improve by roughly the CPU CAMBI cost (the long
  pole on the existing CUDA pass + CPU-residual sequence). The CPU
  residual path stays structurally present in the script for future
  feature additions that may not have CUDA implementations.
