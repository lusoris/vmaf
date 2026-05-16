# MS-SSIM

MS-SSIM (Multi-Scale Structural Similarity Index Measure) extends SSIM to a
multi-resolution pyramid, providing a perceptual similarity metric that is
robust to viewing distance and display resolution variation. Each scale
captures structural information at a different spatial frequency.

## Variant

The fork ships one CPU MS-SSIM extractor:

| Extractor name | Algorithm | Speed | Options |
|---|---|---|---|
| `float_ms_ssim` | Floating-point IQA library, 5-scale Gaussian pyramid | Moderate | `enable_lcs`, `enable_db`, `clip_db`, `enable_chroma` |

GPU twins (`float_ms_ssim_cuda`, `float_ms_ssim_sycl`, `float_ms_ssim_vulkan`)
currently expose only `enable_lcs`; `enable_chroma` support for GPU backends
is a planned follow-up.

## `float_ms_ssim` extractor

The extractor uses the IQA library's Gaussian-window floating-point
implementation with a 5-scale Laplacian pyramid (Wang et al. 2004). It is the
extractor invoked when VMAF model JSON files reference `"float_ms_ssim"`.

The minimum supported input resolution is 176×176. Smaller inputs cause the
5-level pyramid to fall below the 11-tap Gaussian kernel footprint and are
rejected with an error at init time (Netflix#1414 / ADR-0153).

### Output features

| Feature name | Description | Condition |
|---|---|---|
| `float_ms_ssim` | MS-SSIM on the luma (Y) plane | Always |
| `float_ms_ssim_cb` | MS-SSIM on the Cb (U) chroma plane | `enable_chroma=true` only |
| `float_ms_ssim_cr` | MS-SSIM on the Cr (V) chroma plane | `enable_chroma=true` only |
| `float_ms_ssim_l_scale{0–4}` | Per-scale luminance component (L) | `enable_lcs=true`, luma only |
| `float_ms_ssim_c_scale{0–4}` | Per-scale contrast component (C) | `enable_lcs=true`, luma only |
| `float_ms_ssim_s_scale{0–4}` | Per-scale structure component (S) | `enable_lcs=true`, luma only |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `enable_chroma` | bool | `false` | Compute and emit per-plane MS-SSIM for Cb and Cr in addition to luma. Automatically disabled for YUV 4:0:0 (monochrome) input. |
| `enable_lcs` | bool | `false` | Emit per-scale luminance, contrast, and structure intermediate components for the luma plane. |
| `enable_db` | bool | `false` | Report the luma MS-SSIM score as dB (`-10 * log10(1 - score)`). |
| `clip_db` | bool | `false` | Clip the dB score at the theoretical peak for the input bit depth and frame size. Only meaningful when `enable_db=true`. |

### Notes

- The default (`enable_chroma=false`) preserves backward-compatible behaviour:
  a single `float_ms_ssim` score per frame is emitted, matching the upstream
  Netflix extractor.
- When `enable_chroma=true`, three scores are emitted per frame:
  `float_ms_ssim`, `float_ms_ssim_cb`, and `float_ms_ssim_cr`. These are raw
  per-plane MS-SSIM values in [0, 1]; they are not averaged or weighted.
- The `enable_lcs` option only applies to the luma plane. Per-scale LCS
  components for chroma are not currently emitted.
- Chroma MS-SSIM is useful for evaluating chrominance structural fidelity
  independently of luma, particularly for HDR or wide-gamut content where
  Cb/Cr distortions are perceptually significant.

### How to run

```bash
# Luma-only MS-SSIM (default)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature float_ms_ssim --output /dev/stdout

# Per-channel MS-SSIM (luma + Cb + Cr)
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'float_ms_ssim:enable_chroma=true' --output /dev/stdout

# MS-SSIM with per-scale LCS components
libvmaf/build/tools/vmaf \
    --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature 'float_ms_ssim:enable_lcs=true' --output /dev/stdout
```

## See also

- [ADR-0461](../adr/0461-float-ms-ssim-enable-chroma.md) — decision record for
  `enable_chroma` addition to the `float_ms_ssim` extractor
- [ADR-0153](../adr/0153-ms-ssim-min-dim-guard.md) — minimum resolution guard
  (Netflix#1414)
- [SSIM](ssim.md) — single-scale structural similarity
- [SSIMULACRA2](ssimulacra2.md) — perceptually tuned alternative
