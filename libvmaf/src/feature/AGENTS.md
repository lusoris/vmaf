# AGENTS.md — libvmaf/src/feature

Orientation for agents working on feature extractors (the VMAF metric
components: VIF, ADM, motion, integer-valued VIF/ADM/motion, CIEDE, CAMBI,
PSNR, SSIM, MS-SSIM, LPIPS, …). Parent: [../../AGENTS.md](../../AGENTS.md).

## Scope

Every VMAF "feature" is a small C module with a `VmafFeatureExtractor`
registration:

```
feature/
  feature_extractor.c/.h     # the registry + lifecycle contract (init/extract/flush/close)
  feature_collector.c/.h     # per-frame score aggregator
  vif.c / adm.c / …          # scalar CPU reference implementations
  integer_*.c                # integer-math reference implementations
  feature_lpips.c            # DNN-backed extractor (opens vmaf_dnn_session_*)
  x86/                       # AVX2 / AVX-512 SIMD paths — must match scalar bit-for-bit
  arm64/                     # NEON SIMD paths — must match scalar bit-for-bit
  cuda/                      # CUDA kernels + launchers
  sycl/                      # SYCL kernels (DPC++)
  common/                    # cross-arch helpers
```

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **Bit-exactness with the scalar reference** is non-negotiable for SIMD
  paths. Reductions, FMA-ordering, and rounding must match the scalar path
  exactly — no "close enough". See
  [add-simd-path](../../../.claude/skills/add-simd-path/SKILL.md) for the
  dispatch pattern (`cpu.c` + feature_name_avx2.c + feature_name_avx512.c).
- **CUDA / SYCL kernels** should match the CPU reference within the
  documented tolerance. If a kernel cannot match exactly, file a snapshot
  justification in the commit message and regenerate
  `testdata/scores_cpu_*.json` via
  [`/regen-snapshots`](../../../.claude/skills/regen-snapshots/SKILL.md).
- **Registration is discoverable by both name and provided-feature-name**:
  `vmaf_get_feature_extractor_by_name()` and
  `vmaf_get_feature_extractor_by_feature_name()`. Both must resolve.
- **Options tables** must have non-NULL `help` for every entry; see
  [../../test/test_lpips.c](../../test/test_lpips.c) for the unit-test
  pattern that enforces this.
- **DNN-backed extractors** open sessions through
  [src/dnn/](../dnn/AGENTS.md) — never call ONNX Runtime directly from a
  `feature_*.c` file.

## Workflows

| Task | Skill |
| --- | --- |
| Add a feature extractor | [add-feature-extractor](../../../.claude/skills/add-feature-extractor/SKILL.md) |
| Add a SIMD path | [add-simd-path](../../../.claude/skills/add-simd-path/SKILL.md) |
| Cross-backend diff | [cross-backend-diff](../../../.claude/skills/cross-backend-diff/SKILL.md) |
| Profile a hot path | [profile-hotpath](../../../.claude/skills/profile-hotpath/SKILL.md) |

## Rebase-sensitive invariants

- `ssimulacra2.c` is fork-local (not upstream). It embeds several
  constant tables that must stay in lock-step with libjxl even across
  a rebase:
  - **Opsin absorbance matrix** (`kM00`…`kM22`) and bias `kB` — see
    libjxl `lib/jxl/opsin_params.h`.
  - **`MakePositiveXYB` offsets** — `B=(B-Y)+0.55`, `X*=14`, `X+=0.42`,
    `Y+=0.01`.
  - **108 pooling weights (`kWeights[]`)** and the final polynomial
    transform (`0.9562382…`, `2.326765…`, `-0.0208845…`,
    `6.2484966e-05`, `0.6276336…`) — from `tools/ssimulacra2.cc`.
  - **FastGaussian coefficient derivation** — `3.2795·σ + 0.2546`
    radius, k∈{1,3,5}, Cramer's-rule 3×3 solve for β. Any drift from
    libjxl's `lib/jxl/gauss_blur.cc` formulas breaks bit-exactness of
    the scalar blur.
  If libjxl changes any of these upstream, update the scalar extractor
  in the same PR (same for the SIMD follow-ups, which will mirror the
  same coefficient path).

## Governing ADRs

- [ADR-0024](../../../docs/adr/0024-netflix-golden-preserved.md) — the three CPU golden pairs never change.
- [ADR-0041](../../../docs/adr/0041-lpips-sq-extractor.md) — LPIPS extractor registration pattern.
- [ADR-0042](../../../docs/adr/0042-tinyai-docs-required-per-pr.md) — DNN-backed extractors ship docs under `docs/ai/`.
- [ADR-0126](../../../docs/adr/0126-ssimulacra2-feature-extractor.md) +
  [ADR-0130](../../../docs/adr/0130-ssimulacra2-scalar-implementation.md)
  — SSIMULACRA 2 extractor scope + scalar implementation.
