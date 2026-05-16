## CAMBI Vulkan v2 — parity gaps closed (ADR-0465)

The Vulkan CAMBI extractor (`vmaf_fex_cambi_vulkan`) now produces scores
bit-identical to the CPU reference (`vmaf_fex_cambi`) on the cross-backend
parity gate.

Six host-orchestration bugs present since v1 (ADR-0210) are fixed:

1. **Default constants corrected**: `cambi_max_val` default changed from 5.0
   to 1000.0; `window_size` from 63 to 65; `cambi_vis_lum_threshold` from
   1000.0 to 0.0; minimum input dimension from 64 to 216 px. All now match
   the CPU extractor defaults.

2. **`adjust_window_size` formula corrected**: now uses CPU's integer formula
   `((ws*(w+h))/375) >> 4` instead of a float geometric-mean approach. This
   affects all sub-4K inputs (e.g., 576×324 now gets `window=9` matching CPU,
   not `window=11` from v1).

3. **`high_res_speedup` window halving enabled**: the `cambi_high_res_speedup`
   option now halves the adjusted window as the CPU does (was silently ignored).

4. **`tvi_for_diff` bisection corrected**: now finds the last sample where the
   visibility threshold condition holds (was inverted — found last sample where
   it does NOT hold — producing inflated threshold values and inflated scores).
   `vlt_luma` calculation also corrected to match CPU `get_vlt_luma` exactly.

5. **`topk` / `cambi_topk` selection corrected**: mirrors CPU logic (prefer
   `topk` if non-default, else `cambi_topk`).

6. **Histogram buffer sizing corrected**: `c_values_histograms` now allocated
   with the correct `v_band_size` after TVI init, matching CPU allocation.

GPU shaders (`cambi_derivative.comp`, `cambi_filter_mode.comp`,
`cambi_mask_dp.comp`, `cambi_decimate.comp`) are unchanged.

**Expected cross-backend result**: ULP=0 (`places=4` gate).
