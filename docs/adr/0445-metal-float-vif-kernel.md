# ADR-0445: Metal float_vif kernel (T8-1k)

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `gpu`, `metal`, `apple-silicon`, `kernel`, `fork-local`

## Context

The Metal backend shipped eight feature kernels in T8-1c through T8-1j
(ADR-0421). VIF is the next-highest-priority metric in the VMAF model
pipeline that lacks a Metal kernel. Shipping float_vif_metal unblocks
full VMAF scoring on Apple Silicon without falling back to the CPU path
for the VIF sub-score.

VIF is algorithmically heavier than SSIM: it uses four dyadic scales,
each with a separable Gaussian filter (widths 17, 9, 5, 3) followed by
an information-theoretic statistic (`vif_statistic_s`, `matching_matlab`
mode). The CUDA twin (`float_vif_score.cu`, T7-23 / ADR-0192) provides
an exact mathematical reference; this ADR documents the decision to port
that reference to MSL following the established Metal pattern from T8-1j
(float_ssim_metal).

## Decision

We add one `.metal` kernel file and one `.mm` host-binder to
`libvmaf/src/feature/metal/` implementing `float_vif_metal` (T8-1k).

The MSL implementation (`float_vif.metal`) contains two kernel functions:

- `float_vif_compute` — separable V→H Gaussian filter + VIF statistic +
  per-threadgroup (num, den) reduction. One dispatch per scale.
- `float_vif_decimate` — apply scale filter at full input dimensions,
  sample at (2x, 2y), produce half-resolution output for the next scale.
  Three dispatches (scales 0–2; scale 3 has no next scale).

The host binder (`float_vif_metal.mm`) drives 7 MSL dispatches per frame
(4 compute + 3 decimate) and collects four per-scale scores
(`VMAF_feature_vif_scale{0,1,2,3}_score`), matching the CPU `float_vif`
extractor's output contract.

Pixel convention matches `float_vif.c` / `picture_copy` with offset=-128:
`float = pixel - 128` (8 bpc), `/4 - 128` (10 bpc), etc.

Border handling uses mirror padding (`VIF_OPT_HANDLE_BORDERS`), identical
to the CUDA and Vulkan ports. kernelscale=1.0 only (v1 scope, matching
the CUDA and Vulkan v1 constraint).

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|--------|------|------|----------------|
| Port integer_vif_metal instead | integer path matches the VMAF default model | Requires integer arithmetic with int64 accumulators; MSL has no native atomic_ulong (confirmed by CI run 25685703780); would need uvec2-carry workaround | float_vif_metal has simpler arithmetic and a direct float CUDA twin |
| Use shared-tile approach (like CUDA's s_ref/s_dis) | Reduces global-memory bandwidth | Increases MSL threadgroup memory pressure; the VIF halo at scale 0 is 8 px, requiring a 32×32 tile — 32×32×2×4 = 8 KB per threadgroup, exceeding the 4 KB recommended limit for Apple Family 7 GPUs | Chose direct global-memory access per-tap with mirror indexing; matches the Vulkan shaders/vif.comp reference |
| Skip VIF Metal and port PSNR-HVS next | Smaller dependency surface | PSNR-HVS is DCT-heavy and not needed by the primary VMAF v2 model; VIF is the largest gap in the Metal pipeline | VIF closes the biggest missing feature in VMAF model scoring |

## Consequences

- **Positive**: Full VMAF float scoring on Apple Silicon without CPU
  fallback for the VIF sub-score. The `float_vif_metal` extractor
  provides all four per-scale VIF scores consumed by the VMAF v2 model.
- **Negative**: 7 Metal dispatches per frame (versus 1–2 for simpler
  kernels). The `n_dispatches_per_frame = 7` field in the extractor
  descriptor signals the scheduler.
- **Neutral / follow-ups**: The `places=4` cross-backend ULP gate
  (ADR-0214) must pass on macOS CI before the PR can merge. Local
  validation is not possible from a Linux host; the macOS Metal CI lane
  (`Build — macOS Metal (T8-1 scaffold)`) is the gate.

## References

- ADR-0421 — Metal kernel batch T8-1c through T8-1j.
- ADR-0420 — Metal backend runtime (T8-1b).
- ADR-0214 — GPU parity CI gate (`places=4`).
- ADR-0192, ADR-0197 — CUDA float_vif reference implementation.
- `libvmaf/src/feature/cuda/float_vif/float_vif_score.cu` — math reference.
- `libvmaf/src/feature/metal/float_ssim_metal.mm` — host-binder pattern.
