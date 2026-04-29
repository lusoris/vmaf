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

| Feature name       | Invocation name | Core feature? | Output metrics                                                                                | SIMD                | GPU                |
|--------------------|-----------------|---------------|-----------------------------------------------------------------------------------------------|---------------------|--------------------|
| VIF (fixed-point)  | `vif`           | Yes           | `vif_scale0`, `vif_scale1`, `vif_scale2`, `vif_scale3`                                        | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| VIF (float)        | `float_vif`     | Yes           | `float_vif_scale0..3`                                                                         | —                   | CUDA, SYCL, Vulkan |
| Motion2 (fixed)    | `motion`        | Yes           | `motion2` (+ `motion` if `debug=true`)                                                        | AVX2, AVX-512, NEON | CUDA, Vulkan       |
| Motion v2 (fixed)  | `motion_v2`     | No            | `VMAF_integer_feature_motion_v2_sad_score`, `VMAF_integer_feature_motion2_v2_score`           | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| Motion2 (float)    | `float_motion`  | Yes           | `float_motion2` (+ `float_motion` if `debug=true`)                                            | AVX2, AVX-512, NEON | —                  |
| Motion v2 (fixed)  | `motion_v2`     | No            | `VMAF_integer_feature_motion_v2_sad_score`, `VMAF_integer_feature_motion2_v2_score`           | AVX2, AVX-512, NEON | —                  |
| Motion2 (float)    | `float_motion`  | Yes           | `float_motion2` (+ `float_motion` if `debug=true`)                                            | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| ADM (fixed-point)  | `adm`           | Yes           | `adm2`, `adm_scale0`, `adm_scale1`, `adm_scale2`, `adm_scale3`                                | AVX2, AVX-512, NEON | CUDA, Vulkan       |
| ADM (float)        | `float_adm`     | Yes           | `float_adm2`, `float_adm_scale0..3`                                                           | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| [CAMBI](cambi.md)  | `cambi`         | No            | `cambi`                                                                                       | —                   | —                  |
| CIEDE2000          | `ciede`         | No            | `ciede2000`                                                                                   | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| PSNR (fixed)       | `psnr`          | No            | `psnr_y`, `psnr_cb`, `psnr_cr` (+ MSE / APSNR optional)                                       | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan¹|
| PSNR (float)       | `float_psnr`    | No            | `float_psnr` (luma only — the CPU extractor emits a single luma score)                        | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| PSNR-HVS           | `psnr_hvs`      | No            | `psnr_hvs`, `psnr_hvs_y`, `psnr_hvs_cb`, `psnr_hvs_cr`                                        | AVX2, NEON          | CUDA, SYCL, Vulkan |
| SSIM (fixed)       | `ssim`          | No            | `ssim`                                                                                        | —                   | —                  |
| SSIM (float)       | `float_ssim`    | No            | `float_ssim` (+ L/C/S if enabled)                                                             | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| MS-SSIM            | `float_ms_ssim` | No            | `float_ms_ssim` (+ per-scale L/C/S if enabled)                                                | AVX2, AVX-512, NEON | CUDA, SYCL, Vulkan |
| ANSNR              | `float_ansnr`   | No            | `float_ansnr`, `float_anpsnr`                                                                 | —                   | CUDA, SYCL, Vulkan |
| SSIMULACRA 2       | `ssimulacra2`   | No            | `ssimulacra2`                                                                                 | AVX2, AVX-512, NEON | —                  |
| ANSNR              | `float_ansnr`   | No            | `float_ansnr`, `float_anpsnr`                                                                 | —                   | —                  |
| SSIMULACRA 2       | `ssimulacra2`   | No            | `ssimulacra2`                                                                                 | AVX2, AVX-512, NEON | Vulkan             |
| Float moment       | `float_moment`  | No            | `float_moment_ref1st`, `float_moment_dis1st`, `float_moment_ref2nd`, `float_moment_dis2nd`    | AVX2, NEON          | CUDA, SYCL, Vulkan |
| LPIPS (tiny-AI)    | `lpips`         | No            | `lpips`                                                                                       | —                   | —                  |
| FastDVDnet pre     | `fastdvdnet_pre`| No            | `fastdvdnet_pre_l1_residual`                                                                  | —                   | —                  |
| TransNet V2        | `transnet_v2`   | No            | `shot_boundary_probability`, `shot_boundary`                                                  | —                   | —                  |

**Core** extractors are required inputs for the shipped VMAF models (see
[models/overview.md](../models/overview.md)); non-core extractors are
standalone.

¹ The `psnr_cuda`, `psnr_sycl`, and `psnr_vulkan` GPU extractors
emit luma-only (`psnr_y`). The CPU `psnr` extractor emits the full
`psnr_y` / `psnr_cb` / `psnr_cr` set when `enable_chroma=true`
(default). Chroma support on GPU is a focused follow-up — the
existing GPU upload paths are luma-only today.

Depending on your build configuration not every backend is available — see
[`backends/`](../backends/index.md) for the runtime dispatch rules.

## Per-feature GPU dispatch hints (T7-26 / ADR-0181)

Each feature carries a small `VmafFeatureCharacteristics` descriptor
that drives the per-backend dispatch decision (graph-replay vs
direct submit on SYCL; graph-capture vs streams on CUDA;
secondary-cmdbuf reuse vs primary on Vulkan). The descriptor lives
on the extractor and is consumed by the per-backend
`dispatch_strategy` modules under
[`libvmaf/src/{cuda,sycl,vulkan}/dispatch_strategy.{c,h}`](../../libvmaf/src/sycl/dispatch_strategy.cpp).

Defaults are calibrated to match pre-T7-26 SYCL behaviour
byte-for-byte (graph replay above 720p area, direct submit below).
For tuning, three env-var override surfaces are available — each
takes a comma-separated list of `feature:strategy` pairs and wins
over the registry default for the named features:

| Env var | Strategy values | Effect |
| --- | --- | --- |
| `VMAF_SYCL_DISPATCH` | `graph` / `direct` | Per-feature SYCL graph-replay override. |
| `VMAF_CUDA_DISPATCH` | `graph` / `direct` | Per-feature CUDA graph-capture override (today CUDA stub returns DIRECT for every input; the override surface ships now so future graph-capture work doesn't change the user contract). |
| `VMAF_VULKAN_DISPATCH` | `reuse` / `primary` | Per-feature Vulkan secondary-cmdbuf-reuse override (today the Vulkan stub returns PRIMARY_CMDBUF for every input; same forward-compat reasoning). |

Examples:

```bash
# Force ADM to direct submit on SYCL (default below 720p, override above):
VMAF_SYCL_DISPATCH=adm:direct vmaf [...] --feature adm_sycl --backend sycl

# Mix per-feature strategies:
VMAF_SYCL_DISPATCH=vif:graph,motion:direct,adm:graph vmaf [...]
```

Legacy global knobs `VMAF_SYCL_USE_GRAPH=1` / `VMAF_SYCL_NO_GRAPH=1`
are kept as aliases (force every feature to graph / direct
respectively); per-feature `VMAF_SYCL_DISPATCH` takes precedence.

## Core features

### VIF — Visual Information Fidelity

VIF measures information-fidelity loss between reference and distorted at four
Gaussian-pyramid scales. In the original Sheikh/Bovik formulation the scales
are combined into a single score; in VMAF each scale is kept as a separate
feature so the model can learn per-scale weights.

#### Invocation

- CLI: `--feature vif` (fixed-point, default) or `--feature float_vif`.
- ffmpeg: `libvmaf=feature=name=vif` / `feature=name=float_vif`.
- C API: `vmaf_use_feature(ctx, "vif", opts)`.

#### Output metrics

- `vif_scale0`, `vif_scale1`, `vif_scale2`, `vif_scale3` — per-scale fidelity
  ratios in `[0, 1]`. Higher is better (1 = reference-identical).
- With `debug=true`: also `vif`, `vif_num`, `vif_den`, and per-scale
  `*_num` / `*_den`.

**Output range** — each scale `[0, 1]`.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0, 8 / 10 / 12 / 16 bpc.
Operates on the Y plane only.

#### Options

| Option                 | Alias | Type   | Default | Range      | Effect                                                                 |
|------------------------|-------|--------|---------|------------|------------------------------------------------------------------------|
| `debug`                | —     | bool   | `false` | —          | Emit `vif`, `vif_num`, `vif_den`, plus per-scale numerator/denominator |
| `vif_enhn_gain_limit`  | `egl` | double | `1.4`   | `1.0–1.4`  | Cap enhancement-gain ratio so over-sharpened output cannot saturate    |
| `vif_kernelscale`      | —     | double | `1.0`   | `0.1–4.0`  | Scale the Gaussian kernel std-dev — only `float_vif`                   |

`egl=1.0` disables the enhancement-gain path entirely (matches pre-v1.3
behaviour).

**Backends** — `vif`: AVX2, AVX-512, NEON, CUDA, SYCL, Vulkan
(`vif_vulkan`, T5-1b — see
[backends/vulkan/overview.md](../backends/vulkan/overview.md)).
`float_vif`: scalar only.

**Reference** — Sheikh H. R., Bovik A. C., "Image information and visual
quality," IEEE TIP 15(2):430–444, 2006.

### Motion2

A simple temporal-difference feature: blur both frames with a fixed
low-pass filter and take the mean absolute pixel difference between the
current reference and the previous reference luma. Published as `motion2`
(the improved version with proper padding / boundary handling); the
unfixed `motion` is kept behind `debug=true` for back-compat.

#### Invocation

- CLI: `--feature motion` (fixed) or `--feature float_motion`.
- ffmpeg: `libvmaf=feature=name=motion`.
- C API: `vmaf_use_feature(ctx, "motion", opts)`.

#### Output metrics

- `motion2` — the shipped feature.
- `motion` — the legacy unfixed variant, only when `debug=true`.

**Output range** — `[0, ∞)`. Zero for a frozen reference, grows with motion
content. No upper bound.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc. Y plane
only.

#### Options

| Option                | Alias     | Type  | Default     | Effect                                                                                                            |
|-----------------------|-----------|-------|-------------|-------------------------------------------------------------------------------------------------------------------|
| `debug`               | —         | bool  | `true`      | Emit legacy `motion` alongside `motion2`                                                                          |
| `motion_force_zero`   | `force_0` | bool  | `false`     | Override all scores to `0.0` — for deterministic test fixtures                                                    |
| `motion_add_scale1`   | —         | bool  | `false`     | (`float_motion` only) Add a half-resolution bilinear-downsampled SAD term on top of the full-resolution SAD       |
| `motion_add_uv`       | —         | bool  | `false`     | (`float_motion` only) Sum the U and V plane SADs into the score in addition to the Y plane                        |
| `motion_filter_size`  | —         | int   | `5`         | (`float_motion` only) Blur kernel size, `3` or `5`. `5` is the original Motion2 filter; `3` is a cheaper variant  |
| `motion_max_val`      | —         | float | `+∞`        | (`float_motion` only) Upper clamp applied to the emitted `motion2_score` and `motion3_score`                      |

The `motion_add_scale1`, `motion_add_uv`, `motion_filter_size`, and
`motion_max_val` options were ported from upstream Netflix/vmaf
[`b949cebf`](https://github.com/Netflix/vmaf/commit/b949cebf) (2026-04-29).
With the defaults left untouched, output is bit-identical to the pre-port
baseline on the Y-plane SIMD fast path; non-default options route through
the scalar `compute_motion()` path. The same port also enables
`float_motion` to emit a `motion3_score` (a perceptual blend of `motion2`
described by `motion_blend_factor` / `motion_blend_offset`) on the second
frame; the trained VMAF models do not consume `motion3_score` and remain
unchanged.

**Backends** — AVX2, AVX-512, NEON; CUDA for `motion` (fixed-point). The
`motion_add_uv=true` path is currently CPU-only — see
[backends/cuda/overview.md §Known gaps](../backends/cuda/overview.md#known-gaps)
and [backends/sycl/overview.md §Known gaps](../backends/sycl/overview.md#known-gaps).

**Limitations** — Temporal. The extractor carries state across frames (two
previous blurred references) and has a flush callback that emits the final
frame's score after the input stream ends. Single-frame scoring is not
supported; Motion2 on frame 0 is defined as `0.0`.

### Motion v2 — pipelined Motion2

A pipelined re-implementation of Motion2 that exploits the linearity of
the blur kernel: instead of storing the blurred previous reference across
frames, it folds the frame difference, blur, and absolute-sum into a
single row-at-a-time pipeline that needs only one scratch row. The
score is identical to Motion2 (modulo the SAD vs sum semantics described
below); the variant is offered as a separate extractor so callers can
opt into the pipelined arithmetic without touching the legacy
`motion` registry entry.

#### Invocation

- CLI: `--feature motion_v2`.
- ffmpeg: `libvmaf=feature=name=motion_v2`.
- C API: `vmaf_use_feature(ctx, "motion_v2", NULL)`.

#### Output metrics

- `VMAF_integer_feature_motion_v2_sad_score` — per-frame sum of
  absolute blurred differences. Frame 0 always emits `0.0`.
- `VMAF_integer_feature_motion2_v2_score` — Motion2-equivalent
  score (current frame plus the next frame's score, divided by 2,
  matching the legacy temporal smoothing).

**Output range** — `[0, ∞)`. Same units as Motion2.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.
Y plane only.

**Options** — none.

**Backends** — AVX2, AVX-512, NEON, CUDA, SYCL, Vulkan
([`motion_v2_vulkan`](../../libvmaf/src/feature/vulkan/motion_v2_vulkan.c),
[`integer_motion_v2_cuda.c`](../../libvmaf/src/feature/cuda/integer_motion_v2_cuda.c),
[`integer_motion_v2_sycl.cpp`](../../libvmaf/src/feature/sycl/integer_motion_v2_sycl.cpp)).

All three GPU kernels (per [ADR-0193](../adr/0193-motion-v2-vulkan.md))
are **bit-exact** vs the CPU scalar reference on 8-bit and 10-bit
inputs (max_abs_diff = 0.0 across the cross-backend gate fixture).
They share the design: single dispatch / launch over
`(prev_ref - cur_ref)` exploiting convolution linearity to skip
the per-frame blurred-state buffer the CPU pipeline uses; a raw-
pixel ping-pong of two private device buffers caches the previous
frame's Y plane; per-WG `int64` SAD partials reduce on the host;
`motion2_v2_score = min(score[i], score[i+1])` is emitted in
`flush()`. Mirror padding **diverges** from the corresponding
`motion_*` kernels by one pixel at the boundary (CPU
`integer_motion_v2.c` uses edge-replicating reflective mirror
`2*size - idx - 1`).

**Limitations** — Temporal: the extractor caches its own previous
ref in a GPU-side ping-pong (the framework's `prev_ref` slot is
not used for the GPU paths), but Motion v2 on frame 0 is still
defined as `0.0` and the final frame's smoothed score is emitted
via the flush callback (same behaviour as `motion`).

### ADM — Additive Detail Metric (née DLM)

ADM separately measures **detail loss** (the component that affects
content visibility) and **additive impairment** (which distracts attention)
at four wavelet sub-band scales. VMAF uses only the detail-loss branch.
Numerical edge cases (black frames, flat areas) are handled specifically
to avoid divide-by-zero.

#### Invocation

- CLI: `--feature adm` (fixed-point) or `--feature float_adm`.
- ffmpeg: `libvmaf=feature=name=adm`.
- C API: `vmaf_use_feature(ctx, "adm", opts)`.

#### Output metrics

- `adm2` — the fused final value (range `[0, 1]`), published as the
  VMAF-model input.
- `adm_scale0..3` — per-wavelet-scale fidelity.
- With `debug=true`: `adm`, `adm_num`, `adm_den`, and per-scale
  numerator / denominator.

**Output range** — `[0, 1]`. Higher is better.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc. Y plane
only.

#### Options

| Option                  | Alias | Type   | Default | Range       | Effect                                                           |
|-------------------------|-------|--------|---------|-------------|------------------------------------------------------------------|
| `debug`                 | —     | bool   | `false` | —           | Emit debug metrics                                               |
| `adm_enhn_gain_limit`   | `egl` | double | `1.2`   | `1.0–1.2`   | Cap enhancement-gain ratio                                       |
| `adm_norm_view_dist`    | `nvd` | double | `3.0`   | `0.75–24.0` | Normalised viewing distance (distance ÷ display height)          |
| `adm_ref_display_height`| `rdf` | int    | `1080`  | `1–4320`    | Reference display height in pixels (for viewing-distance scaling)|
| `adm_csf_mode`          | `csf` | int    | `0`     | `0–9`       | Contrast-sensitivity-function model index                        |

**Backends** — `adm`: AVX2, AVX-512, NEON, CUDA. `float_adm`: AVX2, AVX-512,
NEON, CUDA, SYCL, Vulkan.

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

#### Invocation

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

#### Invocation

- CLI: `--feature psnr` or `--feature float_psnr`.
- ffmpeg: `libvmaf=feature=name=psnr`.

**Output metrics** (fixed) — `psnr_y`, `psnr_cb`, `psnr_cr`. With
`enable_mse=true` also `mse_y/cb/cr`. With `enable_apsnr=true` also
`apsnr_y/cb/cr` (aggregate across the whole clip, emitted at flush).

**Output range** — dB, saturated at `6 × bpc + 12` when the two planes are
identical (MSE=0): 60 dB for 8 bpc, 72 dB for 10 bpc, 84 dB for 12 bpc,
108 dB for 16 bpc. Override the cap via `min_sse`.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0, 8 / 10 / 12 / 16 bpc.

#### Options

| Option             | Type   | Default | Effect                                                                             |
|--------------------|--------|---------|------------------------------------------------------------------------------------|
| `enable_chroma`    | bool   | `true`  | Include `psnr_cb` / `psnr_cr`; set `false` for luma-only                           |
| `enable_mse`       | bool   | `false` | Emit `mse_y/cb/cr` alongside PSNR                                                  |
| `enable_apsnr`     | bool   | `false` | Emit clip-aggregate `apsnr_y/cb/cr` at flush                                       |
| `reduced_hbd_peak` | bool   | `false` | Scale HBD peak to match 8-bit content                                              |
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

**Backends** — scalar (Xiph reference), AVX2
([ADR-0159](../adr/0159-psnr-hvs-avx2-bitexact.md)), NEON aarch64
([ADR-0160](../adr/0160-psnr-hvs-neon-bitexact.md)). The 8×8 integer
DCT block is vectorized 8-rows-in-parallel via butterfly→transpose→
butterfly→transpose; float accumulators stay scalar by construction
to preserve byte-identity with the reference. Verified bit-identical
to scalar on all three Netflix golden pairs; ~3.58× DCT microbench
speedup on AVX2.

### SSIM / MS-SSIM

Structural Similarity Index on luma. MS-SSIM extends SSIM to five Gaussian-
pyramid scales and fuses them with the Wang 2003 weights.

#### Invocation

- CLI: `--feature ssim` (fixed), `--feature float_ssim`, or
  `--feature float_ms_ssim`.
- ffmpeg: `libvmaf=feature=name=ssim` etc.

#### Output metrics

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

**Minimum dimensions** — `float_ms_ssim` requires at least **176×176**
luma. The five Gaussian-pyramid scales force a `2⁴ = 16`× downsample
on the smallest level; sub-176×176 inputs (e.g. QCIF) cause the
decimate kernel to produce undefined output. The init path rejects
smaller inputs with `-EINVAL` and a clear log message — see
[ADR-0153](../adr/0153-float-ms-ssim-min-size.md). `ssim` /
`float_ssim` have no such constraint.

**Options** (apply to `float_ssim` / `float_ms_ssim` only)

| Option       | Type | Default | Range  | Effect                                                                |
|--------------|------|---------|--------|-----------------------------------------------------------------------|
| `enable_lcs` | bool | `false` | —      | Emit the L / C / S components (per-scale for MS-SSIM)                 |
| `enable_db`  | bool | `false` | —      | Report `-10·log10(1-score)` instead of the raw ratio                  |
| `clip_db`    | bool | `false` | —      | Cap dB values based on the minimum representable MSE                  |
| `scale`      | int  | `0`     | `0–10` | Downsampling factor for `float_ssim`; `0` = auto per Wang 2003        |

**Backends** — `ssim` (fixed): scalar only. `float_ssim` / `float_ms_ssim`:
AVX2, AVX-512, NEON, plus the GPU twins `float_ms_ssim_cuda`,
`float_ms_ssim_sycl` and `float_ms_ssim_vulkan`. The `enable_lcs`
option ships across **all** backends — CPU + CUDA + Vulkan emit the
same 15 `float_ms_ssim_{l,c,s}_scale{0..4}` metrics on top of the
combined score (T7-35 / [ADR-0215](../adr/0215-enable-lcs-gpu.md)).
The SYCL twin does not expose `enable_lcs` at the option level;
follow-up work tracked under T7-35.

**MS-SSIM decimate (fork-local)** — the 9-tap 9/7 biorthogonal wavelet
LPF that produces scales 1–4 runs through `ms_ssim_decimate` in
[`libvmaf/src/feature/ms_ssim_decimate.c`](../../libvmaf/src/feature/ms_ssim_decimate.c).
SIMD variants live in
[`libvmaf/src/feature/x86/ms_ssim_decimate_avx2.c`](../../libvmaf/src/feature/x86/ms_ssim_decimate_avx2.c)
(8-wide),
[`libvmaf/src/feature/x86/ms_ssim_decimate_avx512.c`](../../libvmaf/src/feature/x86/ms_ssim_decimate_avx512.c)
(16-wide), and
[`libvmaf/src/feature/arm64/ms_ssim_decimate_neon.c`](../../libvmaf/src/feature/arm64/ms_ssim_decimate_neon.c)
(4-wide). Dispatch prefers AVX-512 > AVX2 > scalar on x86 and
NEON > scalar on aarch64 at runtime via `vmaf_get_cpu_flags()`; all
four paths are strictly **byte-identical** (per-lane `fmaf` /
`_mm{256,512}_fmadd_ps` / `vfmaq_n_f32` with broadcast coefficients
and scalar-fallback borders). The contract is verified by
`libvmaf/test/test_ms_ssim_decimate.c` across
1x1 / 8x8 / 9x9 / border-edge / 1920x1080 cases. See
[ADR-0125](../adr/0125-ms-ssim-decimate-simd.md).

### ANSNR — Adjusted Noise SNR

SNR after a noise-shaping Wiener filter. Historical VMAF input that no
shipped model still consumes; kept for back-compat with external callers.

**Invocation** — `--feature float_ansnr`.

**Output metrics** — `float_ansnr`, `float_anpsnr`.

**Output range** — dB, saturated at `6 × bpc + 12` (same as PSNR).

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.

**Options** — none.

**Backends** — scalar (CPU) plus CUDA, SYCL, Vulkan
([ADR-0194](../adr/0194-float-ansnr-gpu.md)). All three GPU
kernels are single-dispatch — they apply the CPU's 3x3 ref filter
and 5x5 dis filter inline from a 20×20 shared/SLM tile, accumulate
per-pixel `sig = ref_filtr²` and `noise = (ref_filtr − filtd)²`
into per-WG float partials, and let the host reduce in `double`
before applying the `10·log10` transforms. Empirical floor on the
cross-backend gate fixture: `max_abs_diff = 6e-6` on 8-bit and
`2e-6` on 10-bit, identical across Vulkan / CUDA / SYCL — well
below the `places=4` threshold.

### SSIMULACRA 2 — perceptual similarity in XYB space

Fork-added scalar port of the libjxl reference metric, including a
bit-close C port of libjxl's `FastGaussian` 3-pole recursive IIR as
the pyramid blur. See
[ADR-0130](../adr/0130-ssimulacra2-scalar-implementation.md) for the
scope and algorithm choice, and
[Research-0007](../research/0007-ssimulacra2-scalar-port.md) for the
engineering rationale.

**Invocation** — `--feature ssimulacra2`.

**Output metrics** — `ssimulacra2` (one scalar per frame).

**Output range** — `[0, 100]`, higher is better. Identical reference and
distorted frames return exactly `100`. A reference table from the
upstream algorithm author:

| Score band | Perceptual meaning                          |
|------------|---------------------------------------------|
| 90–100     | Visually lossless                           |
| 70–90      | High quality, only noticeable on close look |
| 50–70      | Medium quality, clearly lossy               |
| 30–50      | Low quality, obvious artifacts              |
| 0–30       | Very low quality                            |

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 bpc. Chroma is
nearest-neighbor upsampled to luma resolution; BT.709 limited-range is
the default YUV→RGB matrix.

**Options** (one, controlling the YUV→RGB matrix)

| Option       | Type | Default | Range | Effect                                                               |
|--------------|------|---------|-------|----------------------------------------------------------------------|
| `yuv_matrix` | int  | `0`     | `0–3` | 0: BT.709 limited, 1: BT.601 limited, 2: BT.709 full, 3: BT.601 full |

**Backends** — AVX2, AVX-512, NEON, SVE2. Three SIMD ports landed
2026-04-25: pointwise + reduction kernels
([ADR-0161](../adr/0161-ssimulacra2-simd-bitexact.md)),
IIR blur ([ADR-0162](../adr/0162-ssimulacra2-iir-blur-simd.md)),
and `picture_to_linear_rgb` ([ADR-0163](../adr/0163-ssimulacra2-ptlr-simd.md)).
ARM64 SVE2 ports for IIR-blur and PTLR followed 2026-04-29
([ADR-0213](../adr/0213-ssimulacra2-sve2.md)), flipping the SVE2
deferral notes in [Research-0016](../research/0016-ssimulacra2-iir-blur-simd.md)
and [Research-0017](../research/0017-ssimulacra2-ptlr-simd.md) from
"deferred" to "shipped"; the SVE2 path runs alongside NEON on hosts
that advertise the `sve2` HWCAP and falls back to NEON otherwise.
All SIMD paths build with `-ffp-contract=off` in dedicated split
static libraries to pin cross-host bit-exactness. CUDA / SYCL
backends remain optional follow-up work (BACKLOG T3-8).

**Bit-exactness** — scalar and SIMD outputs are byte-identical on the
fork's host matrix (verified by `libvmaf/test/test_ssimulacra2_simd.c`,
11 unit tests). Cross-host determinism is pinned by replacing libm
`cbrtf` and `powf(x, 2.4)` with deterministic polynomials —
`vmaf_ss2_cbrtf` (bit-trick init + 2 Newton-Raphson iterations,
~7e-7 accuracy) and a 1024-entry sRGB-EOTF LUT (~5e-7 accuracy). See
[ADR-0164](../adr/0164-ssimulacra2-snapshot-gate.md). The CI snapshot
gate (`python/test/ssimulacra2_test.py`) pins 48-frame
mean/min/max/hmean/frame-0/frame-47 values at `places=4` tolerance.

**Limitations** —

- Coefficient derivation in `create_recursive_gaussian` uses Cramer's
  rule in doubles, which produces identical `n2`/`d1` floats to
  libjxl's `Inv3x3Matrix` for σ=1.5 at 10-decimal precision but is
  not guaranteed bit-exact at every σ. The fork pins σ=1.5, matching
  libjxl's `kSigma`.
- CUDA + SYCL twins shipped per
  [ADR-0206](../adr/0206-ssimulacra2-cuda-sycl.md) (T3-8 closed).

**GPU twins** — `ssimulacra2_vulkan` (T7-23 / batch 3 part 7,
[ADR-0201](../adr/0201-ssimulacra2-vulkan-kernel.md)),
`ssimulacra2_cuda`, and `ssimulacra2_sycl`
([ADR-0206](../adr/0206-ssimulacra2-cuda-sycl.md)). All three
backends share a hybrid host/GPU pipeline: host runs YUV →
linear-RGB, 2×2 pyramid downsample, linear-RGB → XYB (bit-exact
port of CPU `linear_rgb_to_xyb`), and the per-pixel SSIM +
EdgeDiff combine in double precision; GPU runs the 3-plane
elementwise multiplies (`ssimulacra2_mul3` / `ssimulacra2_mul.comp`)
and the 5 separable IIR blurs across 6 scales
(`ssimulacra2_blur_h` + `ssimulacra2_blur_v` /
`ssimulacra2_blur.comp`). The host-side XYB + SSIM combine is
required for `places=4` parity — GPU `cbrtf` differs from libm by
up to 42 ULP and that drift cascaded to a 1.59e-2 pooled-score
drift on the Vulkan first iteration; running XYB on the host
collapses the drift to ~1e-7. Min input dimension: 8×8 (host loop
early-exits at each scale that drops below). The CUDA fatbin for
the IIR kernel is built with `--fmad=false` so the recursive
expression `n2*sum - d1*prev1 - prev2` keeps its CPU
FMUL/FSUB ordering; SYCL relies on `-fp-model=precise` for the
same effect. Cross-backend gates: Vulkan/lavapipe Netflix normal
pair holds at `max_abs_diff = 1.81e-7`; CUDA on RTX 4070 lands at
`1.0e-6` on the normal pair and bit-exact (0.0) on both
checkerboard pairs.

Invocation: `--feature ssimulacra2_vulkan` /
`--feature ssimulacra2_cuda` / `--feature ssimulacra2_sycl`
(pair with the matching `--backend` flag for exclusive GPU
dispatch).

### Float moment — first / second statistical moments

Computes the per-plane mean (first moment) and mean-of-squares (second
moment) of the reference and distorted luma planes. Used as a building
block for higher-level statistical metrics and as a sanity-check
extractor when validating decoder output.

#### Invocation

- CLI: `--feature float_moment`.
- ffmpeg: `libvmaf=feature=name=float_moment`.
- C API: `vmaf_use_feature(ctx, "float_moment", NULL)`.

#### Output metrics

- `float_moment_ref1st`, `float_moment_dis1st` — first moment (mean).
- `float_moment_ref2nd`, `float_moment_dis2nd` — second moment
  (mean of squares).

**Output range** — for 8-bit luma, `[0, 255]` for first moment and
`[0, 65 025]` for second moment; scales with `2^bpc - 1`.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.
Y plane only.

**Options** — none.

**Backends** — scalar only.

**Limitations** — Stateless per-frame. Float pipeline (the picture
plane is copied to float32 before the moments are computed); the
fixed-point twin is not currently shipped.

### LPIPS — learned perceptual image patch similarity (tiny-AI)

A perceptual-distance metric backed by an ONNX model with two image
inputs (`ref`, `dist`). Distinct from the classic VMAF feature
extractors in that the heavy lifting is delegated to ONNX Runtime via
the tiny-AI surface. The model is loaded once at extractor init and
runs per frame.

#### Invocation

- CLI: `--feature lpips=model_path=/path/to/lpips.onnx`.
- ffmpeg: `libvmaf=feature=name=lpips:model_path=...`.
- C API: `vmaf_use_feature(ctx, "lpips", opts)` with
  `model_path` set on the dictionary.

**Output metrics** — `lpips` (one scalar per frame). Lower is more
similar.

**Output range** — model-defined; the reference LPIPS network produces
values in roughly `[0, 1]` for natural content but is not bounded by
construction.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 bpc only. 4:0:0 is
rejected (chroma is required for the RGB conversion). 10 / 12 / 16
bpc inputs return `-ENOTSUP` at init.

#### Options

| Option       | Type   | Default | Effect                                                                                                                         |
|--------------|--------|---------|--------------------------------------------------------------------------------------------------------------------------------|
| `model_path` | string | unset   | Filesystem path to the LPIPS ONNX model (two-input). If unset, falls back to the `VMAF_LPIPS_MODEL_PATH` environment variable. |

**Backends** — scalar only on the libvmaf side (the ONNX model itself
is dispatched to whichever ORT execution provider is selected via
`--tiny-device`; see [`docs/ai/inference.md`](../ai/inference.md)).

**Limitations** — 8-bit only; depends on the
[tiny-AI runtime](../ai/overview.md). The extractor errors out at
init if no model path is provided (neither the option nor the
environment variable); the registry under
[`model/tiny/registry.json`](../../model/tiny/registry.json) tracks
the canonical LPIPS ONNX checkpoint.

### FastDVDnet pre — temporal denoising pre-filter (tiny-AI)

A *temporal* denoising pre-filter backed by an ONNX model with a
single input tensor stacking five luma planes
``[t-2, t-1, t, t+1, t+2]`` along the channel axis; the network emits
a denoised version of frame ``t``. Structurally a feature extractor
(registered in `feature_extractor_list[]` and discoverable by name)
but logically a *pre-filter*, not a quality metric — denoise-before-
encode is a bitrate lever, not a score. Runs through ORT once at
init; per-frame inference uses a 5-slot ring buffer of float32 luma
planes with reflection-pad-light end behaviour.

See also [`docs/ai/models/fastdvdnet_pre.md`](../ai/models/fastdvdnet_pre.md)
and [ADR-0215](../adr/0215-fastdvdnet-pre-filter.md) for the full
surface contract and the placeholder-checkpoint rationale.

#### Invocation

- CLI: `--feature fastdvdnet_pre=model_path=/path/to/fastdvdnet_pre.onnx`.
- ffmpeg: `libvmaf=feature=name=fastdvdnet_pre:model_path=...`.
- C API: `vmaf_use_feature(ctx, "fastdvdnet_pre", opts)` with
  `model_path` set on the dictionary.

**Output metrics** — `fastdvdnet_pre_l1_residual` (one scalar per
frame): mean-absolute difference between the centre frame ``t``
(normalised to `[0, 1]`) and the denoised output. Exists so libvmaf's
per-frame plumbing has a scalar to record; **not** a quality metric.
Downstream pipelines that want the actual denoised pixel data should
consume the FFmpeg `vmaf_pre_temporal` filter once T6-7b lands.

**Output range** — `[0.0, 1.0]` by construction (mean-absolute on
normalised luma). Typical values: `~0.0` for the placeholder
near-identity model or quiet/flat content; `~0.05` for a working
FastDVDnet on lightly noisy content; `~0.20+` on heavy denoising or
saturated placeholder passes.

**Input formats** — YUV 4:2:0 / 4:2:2 / 4:4:4, 8 / 10 / 12 / 16 bpc.
Y plane only (chroma is ignored).

#### Options

| Option       | Type   | Default | Effect                                                                                                                                                                      |
|--------------|--------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path` | string | unset   | Filesystem path to the FastDVDnet ONNX model (5-frame `frames` input, single-frame `denoised` output). Overrides the `VMAF_FASTDVDNET_PRE_MODEL_PATH` environment variable. |

**Backends** — scalar only on the libvmaf side (the ONNX model
itself is dispatched to whichever ORT execution provider is selected
via `--tiny-device`; see [`docs/ai/inference.md`](../ai/inference.md)).

**Limitations** — Stateful (5-frame sliding window); not a metric
(`fastdvdnet_pre_l1_residual` is a diagnostic residual, not a
perceptual score). Depends on the [tiny-AI runtime](../ai/overview.md);
extractor init fails with `-EINVAL` if no model path is provided
(neither the `model_path` option nor the
`VMAF_FASTDVDNET_PRE_MODEL_PATH` env var). Returns `-ENOSYS` from
init if libvmaf was built without ORT. The shipped checkpoint at
`model/tiny/fastdvdnet_pre.onnx` is a smoke-only placeholder with
randomly-initialised weights that respects the I/O shape contract;
it is not a working denoiser. Real upstream-derived FastDVDnet
weights are tracked as backlog item T6-7b. Per ADR-0215 the
placeholder is intentional — the surface, plumbing, and FFmpeg
patch land first; weights follow.

### `mobilesal` — MobileSal saliency map (tiny-AI, NR / single-input)

Runs the MobileSal RGB saliency network on each distorted frame and
emits a per-frame saliency mean. Companion ADR
[`docs/adr/0218-mobilesal-saliency-extractor.md`](../adr/0218-mobilesal-saliency-extractor.md)
records the extractor design + the synthetic-placeholder ONNX shipped
in this PR (real upstream Yun-Liu MobileSal weights are tracked as a
T6-2a-followup row). T6-2b will add the encoder-side `tools/vmaf-roi`
that consumes the saliency map for per-CTU QP offsets.

#### Invocation

- CLI: `--feature mobilesal=model_path=/path/to/mobilesal.onnx`.
- ffmpeg: `libvmaf=feature=name=mobilesal:model_path=...`.
- C API: `vmaf_use_feature(ctx, "mobilesal", opts)` with
  `model_path` set on the dictionary.

**Output metrics** — `mobilesal` (one scalar per frame: mean saliency
across the H×W output map).

**Backends** — scalar only on the libvmaf side; ORT-dispatched to the
selected execution provider.

**Limitations** — placeholder ONNX is smoke-only; real-weight follow-up
tracked in T6-2a-followup. Depends on the
[tiny-AI runtime](../ai/overview.md).

### `transnet_v2` — TransNet V2 shot-boundary detector (tiny-AI, NR / single-input)

Runs the TransNet V2 shot-boundary detector on a sliding 100-frame
window of 27x48 RGB thumbnails (downsampled from the distorted
stream's luma + reconstructed chroma) and emits a per-frame shot-
boundary probability plus a thresholded binary flag. Companion ADR
[`docs/adr/0223-transnet-v2-shot-detector.md`](../adr/0223-transnet-v2-shot-detector.md)
records the extractor design and the synthetic-placeholder ONNX
shipped in this PR. Real upstream Soucek & Lokoc 2020 weights
(MIT-licensed) are tracked as a T6-3a-followup row; the per-shot CRF
predictor that consumes these features is T6-3b.

#### Invocation

- CLI: `--feature transnet_v2=model_path=/path/to/transnet_v2.onnx`.
- ffmpeg: `libvmaf=feature=name=transnet_v2:model_path=...`.
- C API: `vmaf_use_feature(ctx, "transnet_v2", opts)` with
  `model_path` set on the dictionary.

**Output metrics** — `shot_boundary_probability` (sigmoid of the most
recent frame's boundary logit, range `[0.0, 1.0]`) and `shot_boundary`
(binary flag `0.0` / `1.0`, thresholded at `0.5` against
`shot_boundary_probability`). Downstream consumers (per-shot CRF
predictor T6-3b, FFmpeg shot-cut filter) bind to these two names.

**Backends** — scalar only on the libvmaf side; ORT-dispatched to
the selected execution provider.

**Limitations** — Stateful (100-frame sliding window; the first
99 frames emit boundary probabilities computed against a partially-
filled window). Depends on the [tiny-AI runtime](../ai/overview.md);
extractor init fails with `-EINVAL` if no model path is provided
(neither the `model_path` option nor the
`VMAF_TRANSNET_V2_MODEL_PATH` env var). Returns `-ENOSYS` from init
if libvmaf was built without ORT. The shipped checkpoint at
`model/tiny/transnet_v2.onnx` is a smoke-only placeholder with
randomly-initialised weights that respects the I/O shape contract;
it is not a working shot detector. Per
[ADR-0223](../adr/0223-transnet-v2-shot-detector.md) the placeholder
is intentional — surface, plumbing, and FFmpeg patch land first;
weights follow.

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

See [api/index.md](../api/index.md#vmaffeaturedictionary)
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
