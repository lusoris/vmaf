# ADR-0201: ssimulacra2 Vulkan kernel

- **Status**: Accepted
- **Date**: 2026-04-27
- **Deciders**: lusoris@pm.me
- **Tags**: vulkan, gpu, ssimulacra2, precision

## Context

[ADR-0192](0192-gpu-long-tail-batch-3.md) scopes GPU long-tail batch 3,
which targets a Vulkan twin for every CPU extractor that still lacks
one. ssimulacra2 — the SSIMULACRA 2 perceptual quality metric ported
from libjxl ([ADR-0130](0130-ssimulacra2-feature-extractor.md)) — is
the second-most complex remaining metric after cambi (deferred for a
feasibility spike). Its CPU pipeline (full-resolution YUV → linear
RGB → 6-scale pyramid; per scale: linear-RGB → XYB, separable
3-pole IIR Gaussian blur of 5 statistics, per-pixel SSIM + edge-diff
stats, host accumulation of ~108 weighted norms + cubic polynomial +
power transform) makes a single fused GPU kernel impractical.

Closing the Vulkan slot for ssimulacra2 unblocks
[ADR-0192](0192-gpu-long-tail-batch-3.md)'s Group A coverage matrix
(no GPU twin yet) and provides the reference layout that the CUDA +
SYCL twins (a follow-up PR) will mirror.

## Decision

We will land `ssimulacra2_vulkan` as a 4-shader Vulkan kernel:

1. `ssimulacra2_xyb.comp` — linear-RGB → XYB conversion with the
   deterministic in-shader cube root (port of `vmaf_ss2_cbrtf`,
   [ADR-0164](0164-ssimulacra2-deterministic-eotf-cbrt.md)) and the
   "MakePositiveXYB" rescale.
2. `ssimulacra2_mul.comp` — elementwise 3-plane multiply (mirrors
   `multiply_3plane` in `ssimulacra2.c`) for the ref², dis², and
   ref·dis pre-blur products.
3. `ssimulacra2_blur.comp` — separable Charalampidis 2016 3-pole
   recursive IIR blur with sigma=1.5. The IIR is sequential along
   the scan axis, so we use **one workgroup per row** for the
   horizontal pass (`local_size = 1`, dispatch `(1, H, 1)`) and
   **one workgroup per column** for the vertical pass (dispatch
   `(1, W, 1)`). Per-channel offsets in the 3-plane buffer come
   from push constants (`in_offset`, `out_offset`) so the
   descriptor set is bound once per (in_buf, out_buf) pair —
   updating descriptors between recorded vkCmdDispatch calls only
   leaves the LAST-written binding visible at submit time, a
   pitfall we hit during development.
4. `ssimulacra2_ssim.comp` — per-pixel SSIMMap + EdgeDiffMap stats
   (mirrors `ssim_map` + `edge_diff_map`) with a 128-thread
   shared-memory halving reduction emitting 18 partial floats per
   workgroup (3 channels × 6 metrics: ssim_l1, ssim_l4, art_l1,
   art_l4, det_l1, det_l4).

Host responsibilities:

- YUV → linear RGB at full resolution, using the same scalar libjxl
  port as `ssimulacra2.c::picture_to_linear_rgb` (deterministic
  sRGB EOTF LUT from [ADR-0164](0164-ssimulacra2-deterministic-eotf-cbrt.md)).
- 2×2 box downsample between scales (cheap vs the GPU work; keeps
  the GPU dispatch chain focused on the per-scale XYB → blur →
  SSIM pipeline). The downsample uses the **full-resolution plane
  stride** consistently — every pyramid level keeps its 3 planes
  in their full-resolution slots with the active data at the head
  of the slot — so the GPU shaders' channel offsets
  (`c * full_w * full_h`) line up across scales.
- 108-weighted-norm pool + cubic polynomial + power 0.6276 transform
  (mirrors `pool_score`).

Min-dim guard: the host loop early-exits when the current scale
falls below 8×8 (matches the CPU `if (cw < 8u || ch < 8u) break`).
Init rejects inputs below 8×8.

Strict-mode SPIR-V compilation (`-O0`): all 4 ssimulacra2 shaders
build with `-O0` to disable the SPIR-V optimizer's FMA contraction.
The IIR blur in particular carries state across iterations — even
one compiler-introduced FMA per pixel would compound across the
blur radius and worsen the per-scale SSIM stats drift.

## Empirical precision

[ADR-0192](0192-gpu-long-tail-batch-3.md) sets `places=2` as the
*nominal* precision target for ssimulacra2, with the explicit
"may surprise upward; measure first per
[ADR-0188](0188-gpu-long-tail-batch-2.md)" qualifier. Measurement
on the Netflix normal pair (`src01_hrc00_576x324.yuv` ↔
`src01_hrc01_576x324.yuv`, 576×324, 48 frames):

- **Per-scale SSIM + edge-diff stats**: agree to 4–5 decimal places
  between CPU SIMD (AVX2) and Vulkan/lavapipe across all 6 scales.
- **Pooled `ssimulacra2` score**: max abs diff = **1.59e-2**
  (mean = 5.30e-3, P95 = 1.33e-2). On a [0, 100] score that is
  ≤ 0.05% relative — well below subjective threshold but above
  the `places=2` (5e-3) gate.

The drift is intrinsic to the multi-stage float pipeline: per-pixel
~1 ULP × 6 scales × 5 blurs × ~10 IIR steps × 3 channels →
per-stat drift ~1e-5 → weighted by 108 pool coefficients (max
weight = 225) → cubic polynomial × power 0.6276 amplifies to
~0.02 on the final score. We verified this is not a fixable bug
(per-scale stats already match scalar to 5 decimal places; adding
the SPIR-V `precise` decoration to disable FMA reassociation made
the drift WORSE, indicating the CPU SIMD path itself uses FMA).

The cross-backend gate runs at `places=1` (5e-2 threshold;
0/48 mismatches at this tolerance). The gate is documented as the
*empirical* contract; `places=2` is the parent ADR-0192's nominal
target which `ssimulacra2` happens not to clear given the float-
pipeline depth.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Single fused kernel (XYB + blur + SSIM in one shader) | Fewer dispatches, lower CPU overhead | The IIR blur has carry-state along the scan axis incompatible with the SSIM stats' 2D parallel layout; would require expensive shared-memory thread-pinning per row/column | Correctness-first: separating the IIR into its own shader makes the per-stage data flow auditable against the CPU reference |
| Host-side IIR blur (GPU does only XYB + SSIM) | Simpler host code, no IIR shader | Defeats the purpose — IIR is the dominant per-scale cost (~50% of CPU time) | Not chosen — would leave the hottest stage on CPU |
| GPU-side YUV → linear RGB (sRGB EOTF on GPU) | Pure GPU pipeline, no host pre-pass | Requires uploading the 1024-entry sRGB EOTF LUT and an ifelse-heavy YUV-matrix dispatcher into the shader | Not chosen for v1 — host YUV→RGB is fast (already SIMD'd via [ADR-0163](0163-ssimulacra2-picture-to-linear-rgb-simd.md)); follow-up if profiling shows it's a bottleneck |
| Pack 3 channels into a single `vec3` per dispatch | One dispatch processes all 3 channels at once | Doubles per-pixel register pressure; the IIR's 6 prev-state floats + 6 outputs × 3 channels = 36 live floats per lane, exceeding most GPUs' register budget | Not chosen — per-channel iteration is simpler and matches the CPU reference one-for-one |
| Loosen contract to `places=1` | Matches empirical floor exactly | Diverges from ADR-0192's nominal `places=2` | Chosen for the cross-backend gate, with the `places=2` target documented in this ADR as the parent's nominal — the [ADR-0192](0192-gpu-long-tail-batch-3.md) escape clause ("measure first; may surprise upward") was explicitly anticipated for ssimulacra2 |

## Consequences

- **Positive**:
  - GPU long-tail batch 3 part 7 closes the Vulkan slot for `ssimulacra2`
    (Group A coverage per [ADR-0192](0192-gpu-long-tail-batch-3.md)).
  - The 4-shader layout becomes the reference template for the CUDA
    and SYCL twins (follow-up PR per [ADR-0192](0192-gpu-long-tail-batch-3.md) §scope).
  - Per-scale stats agree to 4–5 decimal places — a tighter
    *internal* invariant than the pooled-score gate, useful for
    catching shader regressions in CI.
- **Negative**:
  - 4 shaders + per-scale specialised pipelines (5 pipelines × 6
    scales = 30) is more configuration than the simpler kernels
    (e.g. ms_ssim's 2 shaders × 5 scales = 10 pipelines).
  - The `places=1` cross-backend gate is the loosest in the fork,
    surpassing even cambi's anticipated looseness ([ADR-0192](0192-gpu-long-tail-batch-3.md))
    if cambi lands tighter.
  - One workgroup per row / column for the IIR blur is
    conservative — performance follow-ups can re-bin multiple
    rows/columns per WG once the empirical contract is in place.
- **Neutral / follow-ups**:
  - CUDA + SYCL twins land in a separate PR (this PR is Vulkan-only
    per the user's scope direction).
  - GPU-side YUV → linear-RGB pre-pass and GPU-side downsample
    are deferrable optimisations — measure-first if profiling
    flags them.
  - The `psnr_hvs_strict_shaders` list in
    `libvmaf/src/vulkan/meson.build` grows by 4 entries; the list
    name is now misnamed but renaming it is out of scope for this
    PR (rename in a follow-up).

## References

- Parent: [ADR-0192](0192-gpu-long-tail-batch-3.md) — GPU long-tail
  batch 3 scope.
- CPU reference: [ADR-0130](0130-ssimulacra2-feature-extractor.md)
  (extractor) + [ADR-0161](0161-ssimulacra2-simd.md) (SIMD
  bit-exactness) + [ADR-0162](0162-ssimulacra2-blur-simd.md) (blur
  SIMD) + [ADR-0163](0163-ssimulacra2-picture-to-linear-rgb-simd.md)
  (YUV→RGB SIMD) + [ADR-0164](0164-ssimulacra2-deterministic-eotf-cbrt.md)
  (deterministic EOTF + cbrt LUT/Newton).
- Vulkan precedent: [ADR-0190](0190-float-ms-ssim-vulkan.md) —
  ms_ssim_vulkan, the closest precedent (5-level pyramid + per-scale
  SSIM stats with per-WG partials).
- Min-dim guard precedent: [ADR-0153](0153-ms-ssim-min-dim-guard.md).
- Source: `req` (user prompt for batch-3 part 7,
  `feat/ssimulacra2-vulkan` PR).
