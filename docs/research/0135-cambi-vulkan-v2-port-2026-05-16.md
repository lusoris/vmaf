# Research Digest 0135 — CAMBI Vulkan v2: v1↔CPU Parity Gap Analysis

**Date**: 2026-05-16
**Related ADR**: [ADR-0456](../adr/0456-cambi-vulkan-v2-parity.md)
**Files audited**:
- `libvmaf/src/feature/vulkan/cambi_vulkan.c` (v1)
- `libvmaf/src/feature/cambi.c` (CPU reference)
- `libvmaf/src/feature/cambi_internal.h`
- `libvmaf/src/feature/vulkan/shaders/cambi_*.comp`

---

## Background

The v1 Vulkan CAMBI kernel (`ADR-0205` + `ADR-0210`) ships a Strategy II
hybrid: GPU handles derivative, SAT-based spatial mask, 2x decimate, and 3-tap
mode filter; the CPU handles `calculate_c_values` (sliding histogram) and
top-K spatial pooling. The design guarantees `places=4` parity because the
GPU phases are integer-bit-exact and the host residual runs the exact CPU code
via `cambi_internal.h`.

This digest documents the parity gaps found between v1 and the CPU reference,
and the v2 fixes that close them.

---

## Gap Inventory

### Gap 1 — Wrong default constants (severity: critical)

Three macro defaults in v1 diverge from CPU:

| Constant | v1 value | CPU value | Effect |
|---|---|---|---|
| `CAMBI_VK_DEFAULT_MAX_VAL` | 5.0 | 1000.0 | Clips scores above 5, masking real banding |
| `CAMBI_VK_DEFAULT_WINDOW_SIZE` | 63 | 65 | Wrong pooling window |
| `CAMBI_VK_DEFAULT_VLT` | 1000.0 | 0.0 | All low-luminance pixels masked as invisible |
| `CAMBI_VK_MIN_WIDTH_HEIGHT` | 64 | 216 | Accepts frames the CPU would reject |

**Fix**: All four constants corrected to match `cambi.c`.

### Gap 2 — `adjust_window_size` formula wrong (severity: critical)

CPU formula (`cambi.c` line 472):
```c
*window_size = (((*window_size) * (input_width + input_height)) / 375) >> 4;
```
This is an arithmetic-mean scaling: `ws * (w+h) / 6000` where 6000 ≈ 3840+2160.

v1 used:
```c
double scale = sqrt((double)w * (double)h) / sqrt(3840.0 * 2160.0);
adjusted = (int)((double)*window_size * scale + 0.5);
```
This is a geometric-mean scaling. For 1080p input (1920×1080):
- CPU: `ws * (1920+1080) / 6000 = ws * 0.500`, then `>> 4` = `ws * 0.03125`… wait, that
  can't be right. Let me re-check: `ws=65, (1920+1080)=3000, /375=8.0, >>4=0.5`, then
  `|= 1` → 1. Actually this gives 0 for most sub-4K. Wait: `65*3000/375 = 520 >> 4 = 32 |= 1 = 33`.
- v1: `65 * sqrt(1920*1080)/sqrt(3840*2160) = 65 * 1440/2880 = 65*0.5 = 32.5 → 33`.

At 1080p the results coincidentally agree. At 576×324:
- CPU: `65 * (576+324) / 375 = 65 * 900/375 = 156 >> 4 = 9 | 1 = 9`.
- v1: `65 * sqrt(576*324)/sqrt(8294400) = 65 * 431.5/2880 = 65 * 0.1498 = 9.74 → 10+1=11`.

At 576×324, CPU gives `window=9`, v1 gives `window=11`. This causes different pooling
window areas, producing score divergence even with otherwise identical GPU stages.

**Fix**: Replace with the verbatim CPU formula (integer arithmetic, `>> 4`).

### Gap 3 — `high_res_speedup` window halving ignored (severity: high)

v1's `cambi_vk_adjust_window` took a `high_res` parameter but immediately discarded
it (`(void)high_res`). The CPU halves the adjusted window when `cambi_high_res_speedup`
is active.

**Fix**: Apply the halving `(*window_size + 1) >> 1` when `high_res` is non-zero.

### Gap 4 — `tvi_for_diff` bisection inverted (severity: critical)

CPU `get_tvi_for_diff` uses `tvi_hard_threshold_condition` to find the last sample S
where `tvi_condition(S)=true AND tvi_condition(S+1)=false`, where:
```c
tvi_condition(S) = (L(S+diff) - L(S)) > tvi_threshold * L(S)
```

This is the last sample where the perceptual contrast is above threshold — a small
luma value (e.g., sample 20–100 at 10-bit for bt1886 default TVI).

v1 implemented the complementary search:
```c
if (diff_lum < s->tvi_threshold * sample_lum) { found = mid; lo = mid + 1; }
```
This finds the **largest** sample where the condition does NOT hold — a value near
max (900+). The resulting `tvi_for_diff[d]` is set near 900+num_diffs instead of
~100+num_diffs.

Consequence: `c_value_pixel` checks `value <= tvi_thresholds[d]`. With the inflated
threshold, every pixel passes the TVI gate and `c_value_pixel` runs for all pixels,
producing a systematically elevated CAMBI score on the Vulkan backend.

**Fix**: Rewrite bisect to find the last sample where `tvi_condition(S)=true` (moving
`foot` up when condition holds, `head` down when it fails), matching the CPU exactly.

#### Gap 4b — `vlt_luma` off-by-one

CPU `get_vlt_luma` finds the **smallest** luma sample at or above `cambi_vis_lum_threshold`
(returns 0 if that's `luma_range.foot`). v1 found the **largest** sample below the
threshold (off by 1 in the other direction). For the default VLT=0.0 (now corrected
from gap 1), both compute `vlt_luma=0`. For non-default VLT the off-by-one matters.

**Fix**: Replicate `get_vlt_luma` exactly.

### Gap 5 — `topk` / `cambi_topk` selection (severity: medium)

CPU selects `topk` for pooling as follows:
```c
if (s->topk != DEFAULT_CAMBI_TOPK_POOLING) topk = s->topk; else topk = s->cambi_topk;
```
v1 always passed `s->topk`, ignoring the `cambi_topk` alias when `topk` was at default.

**Fix**: Mirror the CPU selection logic.

### Gap 6 — `c_values_histograms` sizing inconsistent (severity: low)

v1 allocated the histogram with `1024 + 2*num_diffs` bins (the old Netflix formula
before the `v_band_size` optimisation). The CPU now allocates `v_band_size` bins, where:
```c
v_band_size = tvi_for_diff[num_diffs-1] + 1 - max(0, vlt_luma - 3*num_diffs + 1)
```
With the corrected `tvi_for_diff` (gap 4 fix), `v_band_size` is significantly smaller
than `1024 + 2*num_diffs`. The old allocation was not a buffer overflow (it was larger),
but its size did not match what `vmaf_cambi_calculate_c_values` would actually read.

**Fix**: Move histogram allocation to after `cambi_vk_init_tvi` so `tvi_for_diff` is
available, then use `v_band_size` directly.

---

## Shader audit — no changes needed

The five GLSL compute shaders (`cambi_derivative.comp`, `cambi_filter_mode.comp`,
`cambi_mask_dp.comp`, `cambi_decimate.comp`, `cambi_preprocess.comp`) are correct
implementations of their respective algorithms and match the CPU reference. The SAT
implementation in `cambi_mask_dp.comp` uses the full 2D SAT materialization (not the
CPU's cyclic DP), but is numerically equivalent. No shader changes are required in v2.

---

## Cross-backend gate projection

With all six gaps fixed, the Vulkan backend operates:
1. GPU phases (integer, bit-exact): same GPU shaders as v1.
2. Host phases: `cambi_vk_init_tvi` now produces `tvi_for_diff[]` and `vlt_luma`
   identical to the CPU extractor.
3. `adjusted_window`: now identical to CPU for all input resolutions.
4. `effective_topk`: now selects the same value as CPU.
5. `c_values_histograms`: sized identically to CPU's v_band_size-based alloc.

Expected cross-backend ULP result: **ULP=0** (bit-identical to CPU scalar path).
Gate: `places=4` (5e-5 threshold) in `Build — Linux GPU (Vulkan) parity`.

---

## References

- `libvmaf/src/feature/cambi.c` — CPU reference (lines 411–478: TVI/VLT/window)
- `libvmaf/src/feature/cambi_internal.h` — exported API
- `docs/adr/0205-cambi-gpu-feasibility.md` — original Strategy II architecture
- `docs/adr/0210-cambi-vulkan-integration.md` — v1 integration
- `docs/adr/0214-gpu-parity-ci-gate.md` — places=4 gate
