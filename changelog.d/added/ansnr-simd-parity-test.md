## Added

- **`libvmaf/test/test_ansnr_simd.c`**: numerical-parity regression test for
  the three `ansnr_mse_line_*` SIMD kernels (AVX2, AVX-512, NEON).
  Closes the coverage gap identified in the 2026-05-15 SIMD audit.
  Tests 64x64 tail-handling, 1920x1080 production size, two random seeds,
  and a ref==dis identity fixture.  Tolerance: 1e-5 relative (vs the
  snapshot gate's `places=4` ~= 5e-5 absolute).
