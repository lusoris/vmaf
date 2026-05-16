Add `cuModuleUnload` calls to all CUDA feature extractor `close()` functions
(`integer_motion_cuda`, `integer_motion_v2_cuda`, `integer_psnr_cuda`,
`integer_psnr_hvs_cuda`, `integer_ciede_cuda`, `integer_adm_cuda`,
`integer_ms_ssim_cuda`, `integer_ssim_cuda`, `integer_moment_cuda`,
`integer_vif_cuda`, `integer_cambi_cuda`, `float_adm_cuda`, `float_vif_cuda`,
`float_motion_cuda`, `float_psnr_cuda`, `float_ansnr_cuda`). Each
`cuModuleLoadData` allocates GPU-resident module backing store that was
previously leaked until context destruction. The module handle is now stored in
the extractor state struct and unloaded in `close()`.
