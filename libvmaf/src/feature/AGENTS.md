# AGENTS.md — libvmaf/src/feature

Orientation for agents working on feature extractors (the VMAF metric
components: VIF, ADM, motion, integer-valued VIF/ADM/motion, CIEDE, CAMBI,
PSNR, SSIM, MS-SSIM, LPIPS, …). Parent: [../../AGENTS.md](../../AGENTS.md).

## Scope

Every VMAF "feature" is a small C module with a `VmafFeatureExtractor`
registration:

```text
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
- **MS-SSIM decimate LPF coefficients**: the 9-tap 9/7 biorthogonal
  filter table (`ms_ssim_lpf_h` / `ms_ssim_lpf_v`) appears verbatim in
  four TUs that must stay byte-identical for the bit-exactness
  contract — `ms_ssim_decimate.c`, `x86/ms_ssim_decimate_avx2.c`,
  `x86/ms_ssim_decimate_avx512.c`, and
  `arm64/ms_ssim_decimate_neon.c`. The source of truth upstream is
  `g_lpf_h` / `g_lpf_v` in `ms_ssim.c`. If a rebase touches any of
  those five files, diff all five against each other before pushing.
  See [ADR-0125](../../../docs/adr/0125-ms-ssim-decimate-simd.md).
- **KBND_SYMMETRIC mirror**: `ms_ssim_decimate_mirror` is duplicated
  across the same four TUs and must match the upstream
  `KBND_SYMMETRIC` branch in `iqa/convolve.c`. Changing the boundary
  semantics in any one of them breaks bit-identity.
- **SSIM / MS-SSIM SIMD bit-exactness invariants** (fork-local,
  ADR-0138 + ADR-0139 + ADR-0140): the AVX2 / AVX-512 / NEON paths
  in `x86/ssim_avx2.c` / `x86/ssim_avx512.c` /
  `arm64/ssim_neon.c` / `x86/convolve_avx2.c` /
  `x86/convolve_avx512.c` / `arm64/convolve_neon.c` are
  bit-identical to the scalar reference under FLT_EVAL_METHOD == 0.
  Two rules are load-bearing and must be preserved on rebase:
  1. **Convolve taps**: each tap is *single-rounded `float * float`
     → widen to `double` → `double` add*. No FMA. Mirrors scalar
     `sum += img[i] * k[j]` in
     [`iqa/convolve.c`](iqa/convolve.c). Changing scalar to `fmaf`
     or to a double-mul pattern requires matching all three SIMD
     variants.
  2. **SSIM accumulate**: the `2.0 *` literal in
     [`ssim_accumulate_default_scalar`](iqa/ssim_tools.c)
     (`2.0 * ref_mu[i] * cmp_mu[i] + C1` and
     `2.0 * srsc + C2`) is a C `double` literal, which promotes
     the float operands to double before the multiply. All three
     SIMD accumulators do the `2.0 *` numerator + division + final
     `l*c*s` product per-lane in scalar double to match. If
     upstream ever changes the `2.0` literal to `2.0f` (or
     restructures the l/c numerators), all three SIMD variants
     need a matching rewrite.
- **`simd_dx.h` DX macros** (fork-local, ADR-0140): the header
  [`simd_dx.h`](simd_dx.h) is fork-internal and has no upstream
  equivalent. On rebase, keep the fork's version. The macros
  (`SIMD_WIDEN_ADD_F32_F64_*`, `SIMD_ALIGNED_F32_BUF_*`,
  `SIMD_LANES_*`) encode ADR-0138 / ADR-0139 bit-exactness patterns
  by construction — changing their expansion without auditing the
  three SSIM / convolve consumers (`ssim_accumulate_*`,
  `iqa_convolve_*`) is a bit-exactness break waiting to happen.
  Macro names are ISA-suffixed on purpose; do not collapse them
  into cross-ISA aliases (the fork's SIMD policy rules out
  Highway / simde / xsimd — see user memory
  `feedback_simd_dx_scope.md`).
- **`feature_collector.c` mount/unmount traversal**: the fork rewrites
  `vmaf_feature_collector_mount_model` and `unmount_model` to walk a
  local cursor instead of advancing the pointer-to-head — upstream
  [Netflix#1406](https://github.com/Netflix/vmaf/pull/1406) is still
  OPEN as of 2026-04-20 and its body corrupts the list on ≥3 mounted
  models. `unmount_model` additionally returns `-ENOENT` (not
  `-EINVAL`) for "model not mounted". If upstream ever merges #1406,
  **keep the fork's version on conflict** — the traversal is correct
  and the errno split lets callers distinguish misuse from not-found.
  Test coverage in [`../../test/test_feature_collector.c`](../../test/test_feature_collector.c)
  uses the shared `load_three_test_models` / `destroy_three_test_models`
  helpers; upstream's PR inlines 60 LoC of per-model scaffolding that
  would trip clang-tidy `readability-function-size` (JPL-P10 rule 4).
  See [ADR-0132](../../../docs/adr/0132-port-netflix-1406-feature-collector-model-list.md)
  and [rebase-notes 0031](../../../docs/rebase-notes.md).
- **VIF `vif_sigma_nsq` AVX2 parity** (fork-local, ADR-0142): the
  fork-local [`x86/vif_statistic_avx2.c`](x86/vif_statistic_avx2.c)
  `vif_statistic_s_avx2` signature carries a 14th `double
  vif_sigma_nsq` parameter that upstream does not know about
  (upstream ships no AVX2 variant). Two rules preserve the
  default-path bit-identity invariant on rebase:
  1. **Float discipline**: the compute sites use a local
     `const float sigma_nsq = (float)vif_sigma_nsq;` and
     `sigma_max_inv = powf((float)vif_sigma_nsq, 2.0f) / (255² f)`
     — NOT the upstream scalar's double-promotion pattern.
     Keep the float-cast on conflict.
  2. **Signature lockstep**: if upstream extends the scalar
     `vif_statistic_s` signature further, the AVX2 variant's
     declaration in [`vif_tools.c`](vif_tools.c) and definition in
     `x86/vif_statistic_avx2.c` must move in lockstep with it.
  See [ADR-0142](../../../docs/adr/0142-port-netflix-18e8f1c5-vif-sigma-nsq.md)
  and [rebase-notes 0035](../../../docs/rebase-notes.md).

## Governing ADRs

- [ADR-0024](../../../docs/adr/0024-netflix-golden-preserved.md) —
  the three CPU golden pairs never change.
- [ADR-0041](../../../docs/adr/0041-lpips-sq-extractor.md) — LPIPS
  extractor registration pattern.
- [ADR-0042](../../../docs/adr/0042-tinyai-docs-required-per-pr.md) —
  DNN-backed extractors ship docs under `docs/ai/`.
- [ADR-0125](../../../docs/adr/0125-ms-ssim-decimate-simd.md) —
  MS-SSIM decimate separable SIMD + bit-exactness contract.
- [ADR-0126](../../../docs/adr/0126-ssimulacra2-feature-extractor.md) +
  [ADR-0130](../../../docs/adr/0130-ssimulacra2-scalar-implementation.md)
  — SSIMULACRA 2 extractor scope + scalar implementation.
- [ADR-0138](../../../docs/adr/0138-iqa-convolve-avx2-bitexact-double.md) —
  `_iqa_convolve` widen-then-add bit-exactness pattern.
- [ADR-0139](../../../docs/adr/0139-ssim-simd-bitexact-double.md) —
  SSIM accumulate per-lane scalar-double reduction pattern.
- [ADR-0140](../../../docs/adr/0140-simd-dx-framework.md) — SIMD DX
  framework (`simd_dx.h` + `/add-simd-path` skill upgrade).
