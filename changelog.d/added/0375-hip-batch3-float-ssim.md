## Added

- **`float_ssim_hip` real HIP kernel** (`libvmaf/src/feature/hip/float_ssim/ssim_score.hip`):
  promotes the `float_ssim_hip` feature extractor from the `-ENOSYS` scaffold posture
  (ADR-0274) to a real AMD GCN/RDNA kernel. Two-pass separable 11-tap Gaussian SSIM
  (mirrors `cuda/integer_ssim/ssim_score.cu`): Pass 1 writes five intermediate float
  buffers `(W-10)×H`; Pass 2 applies vertical tap + per-pixel SSIM combine + per-block
  float partial sum `(W-10)×(H-10)`. Warp-64 GCN/RDNA warp shuffle reduction
  (`SSIM_WARPS_PER_BLOCK=2`). Active when built with `enable_hip=true enable_hipcc=true`;
  falls back to `-ENOSYS` without `hipcc`. HIP real-kernel count: 5/11. (ADR-0375)
