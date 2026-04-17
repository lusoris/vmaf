# Features

A **feature extractor** is a per-frame computation that libvmaf runs as part of
scoring. Each extractor publishes one or more named metrics into the result
report; VMAF models then fuse these metrics into the final VMAF score. An
extractor can also be requested individually (no model, no fusion) — useful
when all you want is PSNR / SSIM / CIEDE2000 / etc.

This page is the per-extractor reference. For each extractor it lists:

- **Invocation** — the string identifier you pass to `--feature <name>` on the
  CLI, to `av_opt_set` in the ffmpeg `libvmaf` filter, or to
  `vmaf_use_features_from_model()` / `vmaf_use_feature()` in the C API.
- **Output metrics** — the keys that appear in the JSON / XML / CSV report.
- **Output range** — numeric bounds and saturation behaviour.
- **Input formats** — which pixel formats and bit depths the extractor accepts.
- **Options** — per-extractor tuning keys (name, type, default, range).
- **Backends** — which SIMD and GPU backends have a specialised path.
- **Limitations** — known gaps, temporal state, etc.

Per [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) every
user-discoverable extractor ships what / range / invocation / input formats /
limitations in the same PR as the code.

## Extractor overview

| Feature name       | Invocation name | Core feature? | Output metrics                                                                                | SIMD                | GPU    |
|--------------------|-----------------|---------------|-----------------------------------------------------------------------------------------------|---------------------|--------|
| VIF (fixed-point)  | `vif`           | Yes           | `vif_scale0`, `vif_scale1`, `vif_scale2`, `vif_scale3`                                        | AVX2, AVX-512, NEON | CUDA   |
| VIF (float)        | `float_vif`     | Yes           | `float_vif_scale0..3`                                                                         | —                   | —      |
| Motion2 (fixed)    | `motion`        | Yes           | `motion2` (+ `motion` if `debug=true`)                                                        | AVX2, AVX-512, NEON | CUDA   |
| Motion2 (float)    | `float_motion`  | Yes           | `float_motion2` (+ `float_motion` if `debug=true`)                                            | AVX2, AVX-512, NEON | —      |
| ADM (fixed-point)  | `adm`           | Yes           | `adm2`, `adm_scale0`, `adm_scale1`, `adm_scale2`, `adm_scale3`                                | AVX2, AVX-512, NEON | CUDA   |
| ADM (float)        | `float_adm`     | Yes           | `float_adm2`, `float_adm_scale0..3`                                                           | AVX2, AVX-512, NEON | —      |
| [CAMBI](cambi.md)  | `cambi`         | No            | `cambi`                                                                                       | —                   | —      |
| CIEDE2000          | `ciede`         | No            | `ciede2000`                                                                                   | AVX2, AVX-512, NEON | —      |
| PSNR (fixed)       | `psnr`          | No            | `psnr_y`, `psnr_cb`, `psnr_cr` (+ MSE / APSNR optional)                                       | AVX2, AVX-512, NEON | —      |
| PSNR (float)       | `float_psnr`    | No            | `float_psnr_y`, `float_psnr_cb`, `float_psnr_cr`                                              | AVX2, AVX-512, NEON | —      |
| PSNR-HVS           | `psnr_hvs`      | No            | `psnr_hvs`, `psnr_hvs_y`, `psnr_hvs_cb`, `psnr_hvs_cr`                                        | —                   | —      |
| SSIM (fixed)       | `ssim`          | No            | `ssim`                                                                                        | —                   | —      |
| SSIM (float)       | `float_ssim`    | No            | `float_ssim` (+ L/C/S if enabled)                                                             | AVX2, AVX-512, NEON | —      |
| MS-SSIM            | `float_ms_ssim` | No            | `float_ms_ssim` (+ per-scale L/C/S if enabled)                                                | AVX2, AVX-512, NEON | —      |
| ANSNR              | `float_ansnr`   | No            | `float_ansnr`, `float_anpsnr`                                                                 | —                   | —      |

**Core** extractors are required inputs for the shipped VMAF models (see
[models/overview.md](../models/overview.md)); non-core extractors are
standalone.

Depending on your build configuration not every backend is available — see
[`backends/`](../backends/index.md) for the runtime dispatch rules.

## Core features

### VIF — Visual Information Fidelity

VIF measures information-fidelity loss between reference and distorted at four
Gaussian-pyramid scales. In the original Sheikh/Bovik formulation the scales
are combined into a single score; in VMAF each scale is kept as a separate
feature so the model can learn per-scale weights.

**Invocation**

- CLI: `--feature vif` (fixed-point, default) or `--feature float_vif`.
- ffmpeg: `libvmaf=feature=name=vif` / `feature=name=float_vif`.
- C API: `vmaf_use_feature(ctx, "vif", opts)`.

**Output metrics**

- `vif_scale0`, `vif_scale1`, `vif_scale2`, `vif_scale3` — per-scale fidelity
  ratios in `[0, 1]`. Higher is better (1 = reference-identical).
- With `debug=true`: also `vif`, `vif_num`, `vif_den`, and per-scale
  `*_num` / `*_den`.

**Output range** — each scale `[0, 1]`.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0, 8 / 10 / 12 / 16 bpc.
Operates on the Y plane only.

**Options**

| Option                 | Alias | Type   | Default | Range      | Effect                                                                 |
|------------------------|-------|--------|---------|------------|------------------------------------------------------------------------|
| `debug`                | —     | bool   | `false` | —          | Emit `vif`, `vif_num`, `vif_den`, plus per-scale numerator/denominator |
| `vif_enhn_gain_limit`  | `egl` | double | `1.4`   | `1.0–1.4`  | Cap enhancement-gain ratio so over-sharpened output cannot saturate    |
| `vif_kernelscale`      | —     | double | `1.0`   | `0.1–4.0`  | Scale the Gaussian kernel std-dev — only `float_vif`                   |

`egl=1.0` disables the enhancement-gain path entirely (matches pre-v1.3
behaviour).

**Backends** — `vif`: AVX2, AVX-512, NEON, CUDA. `float_vif`: scalar only.

**Reference** — Sheikh H. R., Bovik A. C., "Image information and visual
quality," IEEE TIP 15(2):430–444, 2006.

### Motion2

A simple temporal-difference feature: blur both frames with a fixed
low-pass filter and take the mean absolute pixel difference between the
current reference and the previous reference luma. Published as `motion2`
(the improved version with proper padding / boundary handling); the
unfixed `motion` is kept behind `debug=true` for back-compat.

**Invocation**

- CLI: `--feature motion` (fixed) or `--feature float_motion`.
- ffmpeg: `libvmaf=feature=name=motion`.
- C API: `vmaf_use_feature(ctx, "motion", opts)`.

**Output metrics**

- `motion2` — the shipped feature.
- `motion` — the legacy unfixed variant, only when `debug=true`.

**Output range** — `[0, ∞)`. Zero for a frozen reference, grows with motion
content. No upper bound.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc. Y plane
only.

**Options**

| Option              | Alias     | Type | Default | Effect                                                        |
|---------------------|-----------|------|---------|---------------------------------------------------------------|
| `debug`             | —         | bool | `true`  | Emit legacy `motion` alongside `motion2`                      |
| `motion_force_zero` | `force_0` | bool | `false` | Override all scores to `0.0` — for deterministic test fixtures |

**Backends** — AVX2, AVX-512, NEON; CUDA for `motion` (fixed-point).

**Limitations** — Temporal. The extractor carries state across frames (two
previous blurred references) and has a flush callback that emits the final
frame's score after the input stream ends. Single-frame scoring is not
supported; Motion2 on frame 0 is defined as `0.0`.

### ADM — Additive Detail Metric (née DLM)

ADM separately measures **detail loss** (the component that affects
content visibility) and **additive impairment** (which distracts attention)
at four wavelet sub-band scales. VMAF uses only the detail-loss branch.
Numerical edge cases (black frames, flat areas) are handled specifically
to avoid divide-by-zero.

**Invocation**

- CLI: `--feature adm` (fixed-point) or `--feature float_adm`.
- ffmpeg: `libvmaf=feature=name=adm`.
- C API: `vmaf_use_feature(ctx, "adm", opts)`.

**Output metrics**

- `adm2` — the fused final value (range `[0, 1]`), published as the
  VMAF-model input.
- `adm_scale0..3` — per-wavelet-scale fidelity.
- With `debug=true`: `adm`, `adm_num`, `adm_den`, and per-scale
  numerator / denominator.

**Output range** — `[0, 1]`. Higher is better.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc. Y plane
only.

**Options**

| Option                  | Alias | Type   | Default | Range       | Effect                                                           |
|-------------------------|-------|--------|---------|-------------|------------------------------------------------------------------|
| `debug`                 | —     | bool   | `false` | —           | Emit debug metrics                                               |
| `adm_enhn_gain_limit`   | `egl` | double | `1.2`   | `1.0–1.2`   | Cap enhancement-gain ratio                                       |
| `adm_norm_view_dist`    | `nvd` | double | `3.0`   | `0.75–24.0` | Normalised viewing distance (distance ÷ display height)          |
| `adm_ref_display_height`| `rdf` | int    | `1080`  | `1–4320`    | Reference display height in pixels (for viewing-distance scaling)|
| `adm_csf_mode`          | `csf` | int    | `0`     | `0–9`       | Contrast-sensitivity-function model index                        |

**Backends** — `adm`: AVX2, AVX-512, NEON, CUDA. `float_adm`: AVX2, AVX-512,
NEON.

**Reference** — Li S., Zhang F., Ma L., Ngan K., "Image Quality Assessment by
Separately Evaluating Detail Losses and Additive Impairments," IEEE
Transactions on Multimedia 13(5):935–949, 2011.

## Additional features

### CAMBI — Contrast-Aware Multiscale Banding Index

See the [dedicated CAMBI page](cambi.md) — it is parameter-heavy enough to
warrant its own reference.

Quick facts:

- **Invocation** — `--feature cambi`.
- **Output** — `cambi` in `[0, ∞)`; 0 = no banding, larger = more visible
  banding. Typical "bad" content sits in `1–10`.
- **Input formats** — YUV 4:2:0, 8 / 10 bpc.
- **Backends** — scalar only.

### CIEDE2000 — colour-difference metric

Converts both YCbCr frames to CIELAB and computes the CIEDE2000 ΔE per pixel,
averaged. Captures chroma distortion that luma-only metrics miss (chroma
subsampling, colour-space conversion errors, 4:2:0 vs 4:4:4 differences).

**Invocation**

- CLI: `--feature ciede`.
- C API: `vmaf_use_feature(ctx, "ciede", NULL)`.

**Output metrics** — `ciede2000`.

**Output range** — `[0, ~100]`. Smaller is better.

- `< 1` — imperceptible difference.
- `1–5` — perceptible on close inspection.
- `> 15` — obviously different colour.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc. Requires
chroma — **does not accept 4:0:0**.

**Options** — none.

**Backends** — AVX2, AVX-512, NEON.

**Limitations** — Assumes BT.709 YCbCr → RGB → CIELAB. No override for
BT.2020 or BT.601 input yet. Ported from the `av-metrics` Rust crate.

### PSNR

Peak Signal-to-Noise Ratio on each colour plane. The fixed-point `psnr`
path is the default; the `float_psnr` path is kept for parity with upstream
consumers of the float pipeline.

**Invocation**

- CLI: `--feature psnr` or `--feature float_psnr`.
- ffmpeg: `libvmaf=feature=name=psnr`.

**Output metrics** (fixed) — `psnr_y`, `psnr_cb`, `psnr_cr`. With
`enable_mse=true` also `mse_y/cb/cr`. With `enable_apsnr=true` also
`apsnr_y/cb/cr` (aggregate across the whole clip, emitted at flush).

**Output range** — dB, saturated at `6 × bpc + 12` when the two planes are
identical (MSE=0): 60 dB for 8 bpc, 72 dB for 10 bpc, 84 dB for 12 bpc,
108 dB for 16 bpc. Override the cap via `min_sse`.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0, 8 / 10 / 12 / 16 bpc.

**Options**

| Option             | Type   | Default | Effect                                                                        |
|--------------------|--------|---------|-------------------------------------------------------------------------------|
| `enable_chroma`    | bool   | `true`  | Include `psnr_cb` / `psnr_cr`; set `false` for luma-only                      |
| `enable_mse`       | bool   | `false` | Emit `mse_y/cb/cr` alongside PSNR                                             |
| `enable_apsnr`     | bool   | `false` | Emit clip-aggregate `apsnr_y/cb/cr` at flush                                  |
| `reduced_hbd_peak` | bool   | `false` | Scale HBD peak to match 8-bit content                                         |
| `min_sse`          | double | `0.0`   | Clamp the minimum MSE (and so the PSNR ceiling) — useful for identical-frame tests |

**Backends** — AVX2, AVX-512, NEON.

**Limitations** — Temporal flag set only because of `apsnr` accumulation;
per-frame PSNR itself is stateless.

### PSNR-HVS

PSNR weighted by a human-visual-system contrast-sensitivity function
applied in the DCT domain. Empirically correlates better with subjective
quality than plain PSNR on blocking-style distortions.

**Invocation** — `--feature psnr_hvs`.

**Output metrics** — `psnr_hvs`, `psnr_hvs_y`, `psnr_hvs_cb`, `psnr_hvs_cr`.

**Output range** — dB, typically `20–60`.

**Input formats** — 8 bpc only, YUV 4:2:0 / 4:2:2 / 4:4:4. **10-bit content
is rejected at init.**

**Options** — none.

**Backends** — scalar only (Xiph reference implementation).

### SSIM / MS-SSIM

Structural Similarity Index on luma. MS-SSIM extends SSIM to five Gaussian-
pyramid scales and fuses them with the Wang 2003 weights.

**Invocation**

- CLI: `--feature ssim` (fixed), `--feature float_ssim`, or
  `--feature float_ms_ssim`.
- ffmpeg: `libvmaf=feature=name=ssim` etc.

**Output metrics**

- `ssim` (`ssim` invocation) — one scalar in `[0, 1]`.
- `float_ssim` — scalar in `[0, 1]`. With `enable_lcs=true` also
  `float_ssim_l`, `float_ssim_c`, `float_ssim_s` (luminance / contrast /
  structure components).
- `float_ms_ssim` — scalar in `[0, 1]`. With `enable_lcs=true` also the
  per-scale L/C/S triples `float_ms_ssim_{l,c,s}_scale{0..4}`.

**Output range** — `[0, 1]`, higher is better. With `enable_db=true` the
score is reported on a dB scale via `-10 × log10(1 − score)`; use
`clip_db=true` to cap infinite values when reference ≡ distorted.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.

**Options** (apply to `float_ssim` / `float_ms_ssim` only)

| Option       | Type | Default | Range  | Effect                                                               |
|--------------|------|---------|--------|----------------------------------------------------------------------|
| `enable_lcs` | bool | `false` | —      | Emit the L / C / S components (per-scale for MS-SSIM)                |
| `enable_db`  | bool | `false` | —      | Report `-10·log10(1-score)` instead of the raw ratio                 |
| `clip_db`    | bool | `false` | —      | Cap dB values based on the minimum representable MSE                 |
| `scale`      | int  | `0`     | `0–10` | Downsampling factor for `float_ssim`; `0` = auto per Wang 2003        |

**Backends** — `ssim` (fixed): scalar only. `float_ssim` / `float_ms_ssim`:
AVX2, AVX-512, NEON.

### ANSNR — Adjusted Noise SNR

SNR after a noise-shaping Wiener filter. Historical VMAF input that no
shipped model still consumes; kept for back-compat with external callers.

**Invocation** — `--feature float_ansnr`.

**Output metrics** — `float_ansnr`, `float_anpsnr`.

**Output range** — dB, saturated at `6 × bpc + 12` (same as PSNR).

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.

**Options** — none.

**Backends** — scalar only.

## Invoking features from the CLI

```bash
# Single extractor, no model
vmaf --reference ref.yuv --distorted dis.yuv \
     --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
     --no_prediction --feature psnr \
     --output score.json --json
```

Multiple `--feature` flags can stack:

```bash
vmaf ... --feature psnr --feature ssim --feature ciede --feature cambi ...
```

Per-feature options go into the name string:

```bash
vmaf ... --feature "psnr=enable_mse=true|enable_apsnr=true" ...
vmaf ... --feature "adm=adm_enhn_gain_limit=1.0" ...
```

See [usage/cli.md](../usage/cli.md) for the full CLI grammar.

## Invoking features from the C API

```c
VmafFeatureDictionary *opts = NULL;
vmaf_feature_dictionary_set(&opts, "enable_mse", "true");
vmaf_use_feature(ctx, "psnr", opts);
```

See [api/index.md](../api/index.md#vmaffeaturedictionary--tuning-extractors)
for the dictionary ownership rules.

## Related

- [CAMBI](cambi.md) — banding-specific extractor.
- [Confidence Interval](confidence-interval.md) — bootstrapped uncertainty
  on the final VMAF score.
- [Bad cases](bad-cases.md) — how to report content where extractors
  disagree with subjective ratings.
- [Backends](../backends/index.md) — which SIMD / GPU paths get picked at
  runtime.
- [Models](../models/overview.md) — how the fixed-point core extractors
  feed into the shipped VMAF models.
- [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) — the
  per-surface doc bar this page satisfies.
