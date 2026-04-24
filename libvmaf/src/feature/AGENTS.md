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
- **Generalised AVX convolve scanline helpers** (fork-local,
  ADR-0143): the four `convolution_f32_avx_s_1d_*_scanline`
  helpers in [`common/convolution_avx.c`](common/convolution_avx.c)
  are `static` in the fork (upstream leaves them extern out of
  habit). Strides are `ptrdiff_t` inside helpers, `int` at the
  public `convolution_f32_avx_*_s` wrappers, with `(ptrdiff_t)`
  casts at pointer-offset multiplication sites. On rebase: keep
  the fork's `static` and `ptrdiff_t` unless upstream adopts them.
  See [ADR-0143](../../../docs/adr/0143-port-netflix-f3a628b4-generalized-avx-convolve.md)
  and [rebase-notes 0036](../../../docs/rebase-notes.md).
- **`motion_v2` NEON shift semantics** (fork-local, ADR-0145):
  [`arm64/motion_v2_neon.c`](arm64/motion_v2_neon.c) uses
  **arithmetic** right-shift throughout (`vshrq_n_s64(v, 16)` for
  the Phase-2 known shift, `vshlq_s64(v, -(int64_t)bpc)` for the
  Phase-1 runtime shift). The fork's AVX2 variant
  [`x86/motion_v2_avx2.c`](x86/motion_v2_avx2.c) uses
  `_mm256_srlv_epi64` (*logical*) which can diverge from scalar on
  negative-diff pixels. NEON matches scalar, AVX2 does not — this
  is intentional until the AVX2 audit lands. On rebase: keep the
  arithmetic-shift form in NEON; do NOT port AVX2's logical pattern
  even if it looks simpler. 4-lane stride + scalar tails on both
  sides of the row are load-bearing for the x_conv edge-mirror
  contract. See
  [ADR-0145](../../../docs/adr/0145-motion-v2-neon-bitexact.md)
  and [rebase-notes 0038](../../../docs/rebase-notes.md).
- **IQA / VIF SIMD helper decomposition** (fork-local, ADR-0146):
  `iqa_convolve` in
  [`iqa/convolve.c`](iqa/convolve.c) is split into
  `iqa_convolve_horizontal_pass` + `iqa_convolve_vertical_pass`
  composed by `iqa_convolve_1d_separable` (for `IQA_CONVOLVE_1D`)
  and `iqa_convolve_2d`; `iqa_ssim` in
  [`iqa/ssim_tools.c`](iqa/ssim_tools.c) is split into
  `ssim_workspace_alloc` / `_free` + `ssim_compute_stats` +
  `ssim_init_args` around an explicit `struct ssim_workspace`;
  `vif_statistic_s_avx2` in
  [`x86/vif_statistic_avx2.c`](x86/vif_statistic_avx2.c) is split
  into `vif_stat_simd8_compute` + `vif_stat_simd8_reduce` around
  an explicit `struct vif_simd8_lane` that carries `__m256` lane
  state between the two halves. **Load-bearing**: the per-lane
  scalar-float reduction via 32-byte aligned `tmp_n[8]` / `tmp_d[8]`
  in `vif_stat_simd8_reduce` preserves ADR-0139 exactly; the
  convolve pass ordering in `iqa_convolve_1d_separable` preserves
  ADR-0138 exactly. On rebase: if upstream rewrites any of these
  three functions, prefer upstream's shape **only** if it maintains
  both invariants; otherwise keep the fork's split and re-document
  divergence in
  [rebase-notes 0039](../../../docs/rebase-notes.md). Also: the
  TU-static rename `_calc_scale` → `iqa_calc_scale` in
  `iqa/convolve.c` is fork-local — keep on rebase. See
  [ADR-0146](../../../docs/adr/0146-nolint-sweep-function-size.md).
- **IQA reserved-identifier rename** (fork-local, ADR-0148):
  every `_iqa_*` / `struct _kernel` / `_ssim_int` /
  `_map_reduce` / `_map` / `_reduce` / `_context` /
  `_ms_ssim_*` / `_ssim_*` / `_alloc_buffers` /
  `_free_buffers` symbol and the four underscore-prefixed
  header guards (`_CONVOLVE_H_`, `_DECIMATE_H_`,
  `_SSIM_TOOLS_H_`, `__VMAF_MS_SSIM_DECIMATE_H__`) was renamed
  to its non-reserved spelling. The IQA tree is now baseline
  lint-clean. **Load-bearing NOLINTs** (do not collapse on
  rebase): scoped
  `NOLINTBEGIN/END(clang-analyzer-security.ArrayBound)` around
  the inner kernel loops in `ssim_accumulate_row` and
  `ssim_reduce_row_range` of
  [`integer_ssim.c`](integer_ssim.c) — the
  `k_min`/`k_max` clamping is provably correct but the
  analyzer can't follow it across the helper boundary; scoped
  `NOLINTBEGIN/END(clang-analyzer-unix.Malloc)` around
  `check_simd_variant` and `check_case` in
  [`../../test/test_iqa_convolve.c`](../../test/test_iqa_convolve.c)
  — test exits process on failure path; small allocations
  leak by design at test end; cross-TU
  `NOLINTNEXTLINE(misc-use-internal-linkage)` on `compute_ssim`
  in [`ssim.c`](ssim.c) and `compute_ms_ssim` in
  [`ms_ssim.c`](ms_ssim.c) — declared in `ssim.h` /
  `ms_ssim.h`, called from `float_ssim.c` /
  `float_ms_ssim.c`; clang-tidy runs per-TU and can't see the
  bridge. On rebase, keep all these brackets verbatim. See
  [ADR-0148](../../../docs/adr/0148-iqa-rename-and-cleanup.md)
  and [rebase-notes 0041](../../../docs/rebase-notes.md).

- **`integer_adm.c` i4_adm_cm int32 rounding overflow**
  (fork-inherited, ADR-0155): both `add_bef_shift_flt[]`
  initialiser loops in
  [`integer_adm.c`](integer_adm.c) (scales 1–3) assign
  `1u << 31 = 0x80000000` into `int32_t`, which wraps to
  `-2147483648`. The rounding term is sign-negated; every
  downstream `(prod + add_bef_shift) >> 32` subtracts 2^31
  instead of adding it. **Deliberately preserved** — the buggy
  arithmetic is encoded in the Netflix golden
  `assertAlmostEqual` values (project hard rule #1 /
  [ADR-0024](../../../docs/adr/0024-netflix-golden-preserved.md)).
  Do NOT widen `add_bef_shift_flt[]` to `uint32_t` or `int64_t`
  without a coordinated Netflix-authored golden-number update
  (the [ADR-0142](../../../docs/adr/0142-port-netflix-18e8f1c5-vif-sigma-nsq.md)
  carve-out). Netflix upstream #955 is OPEN since 2020 with no
  maintainer response — until it closes with a fix, the
  overflow stays. See
  [ADR-0155](../../../docs/adr/0155-adm-i4-rounding-deferred-netflix-955.md)
  and [rebase-notes 0048](../../../docs/rebase-notes.md).

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
  `iqa_convolve` widen-then-add bit-exactness pattern.
- [ADR-0139](../../../docs/adr/0139-ssim-simd-bitexact-double.md) —
  SSIM accumulate per-lane scalar-double reduction pattern.
- [ADR-0140](../../../docs/adr/0140-simd-dx-framework.md) — SIMD DX
  framework (`simd_dx.h` + `/add-simd-path` skill upgrade).
