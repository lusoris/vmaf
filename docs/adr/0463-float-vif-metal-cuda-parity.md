# ADR-0463: float_vif_metal CUDA-parity gaps — vif_kernelscale + debug features

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: gpu, metal, apple-silicon, kernel, parity, fork-local

## Context

PR #869 (ADR-0462) shipped `float_vif_metal` — a 4-scale VIF kernel on the Metal
backend.  A post-merge audit against `float_vif_cuda.c` identified two gaps:

1. `vif_kernelscale` option absent from Metal.  CUDA exposes the option and
   immediately rejects values other than 1.0 (`-EINVAL`), keeping the API surface
   consistent while signalling that non-unity kernelscale is not yet implemented.
   Metal had no such option, so callers who set `vif_kernelscale` on a feature-set
   containing both CUDA and Metal extractors would see asymmetric option-validation
   behaviour.

2. Debug provided-features absent from Metal.  CUDA emits `vif`, `vif_num`,
   `vif_den`, and per-scale `vif_num_scaleN` / `vif_den_scaleN` when `debug=true`.
   Metal only emitted the four `VMAF_feature_vif_scaleN_score` features.  Tooling
   that probes debug features would silently fail on Metal while succeeding on CUDA.

## Decision

Add `vif_kernelscale` to Metal's option table (identical range 0.1–4.0, default 1.0,
`VMAF_OPT_FLAG_FEATURE_PARAM`) with a guard in `init_fex_metal` that returns
`-EINVAL` for any value other than 1.0, matching the CUDA guard exactly.

Add the eleven debug provided-features to Metal's `provided_features[]` array and
implement the corresponding collection logic in `collect_fex_metal` behind the
existing `debug` flag, summing the already-accumulated per-scale (num, den) pairs.

## Alternatives considered

No alternatives: this is a straightforward parity fix with exactly one correct
implementation per gap. The only decision is whether to carry the option-vs-guard
pattern from CUDA or to silently ignore unsupported kernelscales; the guard is
chosen because silent ignoring is a misuse trap.

## Consequences

- Metal and CUDA `float_vif` option tables are now identical.
- Debug feature collection is now available on Metal for tooling and regression
  analysis parity with CUDA.
- No kernel changes; no numerical impact on normal (non-debug) runs.

## References

- ADR-0462 (float_vif_metal kernel, PR #869)
- `libvmaf/src/feature/cuda/float_vif_cuda.c` — reference implementation
- req: "PR #869 added Metal float_vif_metal. Audit it vs CUDA float_vif_cuda.c —
  implement any gaps (enable_chroma, vif_skip_scale0, etc)."
