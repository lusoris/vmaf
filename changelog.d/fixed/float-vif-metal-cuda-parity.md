**fixed(metal):** `float_vif_metal` now exposes `vif_kernelscale` option (rejected with
`-EINVAL` for non-1.0, matching CUDA) and emits the eleven debug features (`vif`,
`vif_num`, `vif_den`, `vif_num_scaleN`, `vif_den_scaleN`) when `debug=true`, reaching
full option-table and debug-feature parity with `float_vif_cuda` (ADR-0463).
