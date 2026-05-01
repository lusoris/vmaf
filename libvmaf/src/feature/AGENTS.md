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

- **`psnr_hvs` AVX2 DCT bit-exactness** (fork-local, ADR-0159):
  [`x86/psnr_hvs_avx2.c`](x86/psnr_hvs_avx2.c) vectorizes the
  Xiph/Daala 8×8 integer DCT across 8 rows in parallel
  (`__m256i`, 8× int32) via **butterfly → transpose → butterfly
  → transpose**. Byte-identical `od_coeff` output to scalar
  under `FLT_EVAL_METHOD == 0`; float accumulators (means /
  variances / mask / error) kept scalar by construction per
  ADR-0139 precedent. **On rebase**: never introduce a
  horizontal-reduce vectorization of the float accumulators
  without replicating the per-lane scalar-float reduction
  pattern. Keep `#pragma STDC FP_CONTRACT OFF` at the TU
  header — removing it allows `fmaf` and breaks the 1-ulp
  guarantee. The scalar TU
  [`third_party/xiph/psnr_hvs.c`](third_party/xiph/psnr_hvs.c)
  is the bit-exact reference; don't touch its butterfly block
  without matching changes in the AVX2 TU. See
  [ADR-0159](../../../docs/adr/0159-psnr-hvs-avx2-bitexact.md)
  and [rebase-notes 0052](../../../docs/rebase-notes.md).

- **SSIMULACRA 2 end-to-end regression gate** (fork-local, ADR-0164):
  [`python/test/ssimulacra2_test.py`](../../../python/test/ssimulacra2_test.py)
  pins pooled + per-frame `--feature ssimulacra2` output on two
  checked-in YUV fixtures. **On rebase**: if the scalar or any SIMD
  path changes semantically (should never happen per ADR-0161's
  bit-exact contract), the test will fail with values that differ
  by more than 1e-4. Don't update the pinned floats unilaterally —
  figure out which kernel drifted and fix it. The Netflix golden
  assertions in `quality_runner_test.py` et al. remain untouched.

- **SSIMULACRA 2 `picture_to_linear_rgb` SIMD** (fork-local, ADR-0163):
  `ssimulacra2_picture_to_linear_rgb_{avx2,avx512,neon}` vectorises
  the last scalar hot path (2×/frame). Strategy: per-lane scalar
  reads (all chroma ratios + 8/16-bit), SIMD matmul + normalise +
  clamp, per-lane scalar `powf` for sRGB EOTF. New decoupling
  header `ssimulacra2_simd_common.h` defines `simd_plane_t`; the
  dispatch wrapper in `ssimulacra2.c` unpacks `VmafPicture` into it.
  **On rebase**: (1) keep scalar-order matmul chain
  `G = Yn + cb_g*Un; G += cr_g*Vn;` — regrouping drifts ~1 ulp;
  (2) per-lane scalar `powf` is load-bearing — no vector
  polynomial; (3) `simd_plane_t` layout `{data, stride, w, h}`
  is assumed by all three SIMD TUs; (4) arbitrary chroma ratios
  (non-420/422/444) must still work — don't delete the `int64_t`
  fallback branch. SSIMULACRA 2 now has **zero scalar hot paths**.
  See
  [ADR-0163](../../../docs/adr/0163-ssimulacra2-ptlr-simd.md) and
  [rebase-notes 0055](../../../docs/rebase-notes.md).

- **SSIMULACRA 2 FastGaussian IIR blur SIMD** (fork-local, ADR-0162):
  `ssimulacra2_blur_plane_{avx2,avx512,neon}` vectorises the 30×/frame
  2-pass separable IIR blur. Horizontal pass batches rows (AVX2: 8,
  AVX-512: 16, NEON: 4) and uses gather/lane-set loads to pull
  column-n values from N rows into a SIMD vector; vertical pass
  SIMD-iterates columns over the per-column `prev1_*`/`prev2_*`
  state arrays. **On rebase**: (1) preserve left-to-right summation
  `(o0 + o1) + o2` and `n2*sum - d1*prev1 - prev2` chaining — any
  re-grouping drifts by ~1 ulp; (2) `col_state` layout is
  `[prev1_0|prev1_1|prev1_2|prev2_0|prev2_1|prev2_2]` in 6×w
  contiguous floats; SIMD loads assume this; (3) NEON lane-set
  pattern (4 `vsetq_lane_f32` per input) replaces the
  non-existent aarch64 gather intrinsic; (4) row-batching lane
  layout: lane i holds row (y_base + i). Regression test
  `test_blur` in `test_ssimulacra2_simd.c` catches all four. See
  [ADR-0162](../../../docs/adr/0162-ssimulacra2-iir-blur-simd.md)
  and [rebase-notes 0054](../../../docs/rebase-notes.md).

- **SSIMULACRA 2 SIMD bit-exactness** (fork-local, ADR-0161):
  [`x86/ssimulacra2_avx2.c`](x86/ssimulacra2_avx2.c),
  [`x86/ssimulacra2_avx512.c`](x86/ssimulacra2_avx512.c),
  [`arm64/ssimulacra2_neon.c`](arm64/ssimulacra2_neon.c) and
  [`arm64/ssimulacra2_sve2.c`](arm64/ssimulacra2_sve2.c) (T7-38,
  ADR-0213) all produce byte-identical output to scalar on the 5
  vectorised kernels (`multiply_3plane`, `linear_rgb_to_xyb`,
  `downsample_2x2`, `ssim_map`, `edge_diff_map`) under
  `FLT_EVAL_METHOD == 0`, plus the IIR blur and PTLR ports
  (ADR-0162 / ADR-0163). **On rebase**: (1) preserve left-to-right
  scalar summation order in every matmul + downsample chain —
  a `(a+b)+(c+d)` pairing drifts by 1 ULP and the regression test
  `test_ssimulacra2_simd` catches it; (2) `cbrtf` stays per-lane
  scalar libm — no vector polynomial; (3) reductions in
  `ssim_map`/`edge_diff_map` use the ADR-0139 per-lane `double`
  scalar tail; (4) the SVE2 sister TU is locked to a fixed 4-lane
  predicate (`svwhilelt_b32(0, 4)`) so its arithmetic order
  matches the NEON sibling regardless of runtime vector length —
  do **not** widen to `svptrue_b32()` without a separate ADR
  + snapshot regen, even if it looks like a free perf win. See
  [ADR-0161](../../../docs/adr/0161-ssimulacra2-simd-bitexact.md),
  [ADR-0213](../../../docs/adr/0213-ssimulacra2-sve2.md), and
  [rebase-notes 0053](../../../docs/rebase-notes.md) /
  [rebase-notes 0074](../../../docs/rebase-notes.md).

- **`psnr_hvs` NEON DCT bit-exactness** (fork-local, ADR-0160):
  [`arm64/psnr_hvs_neon.c`](arm64/psnr_hvs_neon.c) is the aarch64
  sister port to the AVX2 TU. NEON's 4-wide `int32x4_t` splits
  each 8-column row into `r_k_lo` (cols 0-3) + `r_k_hi` (cols
  4-7); the 30-butterfly runs twice per DCT pass, and 8×8
  transpose = four `transpose4x4_s32` (via `vtrn1q_s32` /
  `vtrn2q_s32` / `vtrn1q_s64` / `vtrn2q_s64`) + a top-right
  ↔ bottom-left block swap. **On rebase**: the two SIMD TUs
  (AVX2 + NEON) must move in lockstep with the scalar Xiph
  reference — any change to the butterfly in `psnr_hvs.c`
  requires matched edits to both SIMD TUs and a re-run of
  `test_psnr_hvs_{avx2,neon}`. `accumulate_error()` must keep
  threading the outer `ret` by pointer (ADR-0159 summation-order
  lesson; a local float accumulator would drift the Netflix
  golden by ~5.5e-5). `#pragma STDC FP_CONTRACT OFF` is ignored
  by aarch64 GCC (non-fatal `-Wunknown-pragmas`) but kept for
  portability; aarch64 GCC does not contract `a + b * c` across
  statements at default optimization anyway. See
  [ADR-0160](../../../docs/adr/0160-psnr-hvs-neon-bitexact.md)
  and [rebase-notes 0052](../../../docs/rebase-notes.md).
- **`fastdvdnet_pre.c` 5-frame-window contract** (fork-local,
  ADR-0215): the FastDVDnet temporal pre-filter extractor is wired
  to the I/O contract `frames: float32 NCHW [1, 5, H, W]` (channel
  axis stacks `[t-2, t-1, t, t+1, t+2]`) → `denoised: float32 NCHW
  [1, 1, H, W]`. Three pieces are load-bearing on rebase: (1) the
  centre index is 2 (`FASTDVDNET_PRE_CENTRE`) — `gather_window`
  computes channel-k offsets relative to it; (2) the ring buffer
  holds 5 slots and replicates the closest available end frame for
  channel positions outside the available window (clip start +
  end); (3) the registered feature name is
  `fastdvdnet_pre_l1_residual` — downstream consumers (FFmpeg
  `vmaf_pre_temporal` filter shipping with T6-7b, training
  harnesses) bind to that exact string. The placeholder ONNX
  shipped under `model/tiny/fastdvdnet_pre.onnx` is smoke-only
  (`smoke: true` in the registry); when T6-7b swaps in real
  upstream weights, keep the I/O names (`frames` / `denoised`)
  byte-identical. See
  [ADR-0215](../../../docs/adr/0215-fastdvdnet-pre-filter.md).
- **`transnet_v2.c` 100-frame-window contract** (fork-local,
  ADR-0223 TransNet V2 shot-boundary detector is wired to
  the I/O contract `frames: float32 [1, 100, 3, 27, 48]`
  (100-frame window of 27x48 RGB thumbnails) → `boundary_logits:
  float32 [1, 100]`. Three pieces are load-bearing on rebase:
  (1) the ring buffer holds 100 slots and replicates the *oldest*
  available frame across pre-clip slots (head-clamp at clip
  start) — the corresponding output logit is read from
  `output_logits[WINDOW-1]` because `gather_window` lays the
  most-recent push at the LAST channel; (2) the dual feature-name
  surface — the extractor emits both
  `shot_boundary_probability` (sigmoid of the centre-slot
  logit) **and** `shot_boundary` (binary 0/1 thresholded at 0.5);
  downstream consumers (the per-shot CRF predictor T6-3b, the
  FFmpeg shot-cut filter shipping with T6-3b) bind to *both*
  exact strings; (3) the placeholder ONNX shipped under
  `model/tiny/transnet_v2.onnx` is smoke-only (`smoke: true` in
  the registry); when T6-3a-followup swaps in real upstream
  weights, keep the I/O names (`frames` / `boundary_logits`) and
  the shape (`[1, 100, 3, 27, 48]` → `[1, 100]`)
  byte-identical, and switch the `luma_to_thumbnail` path from
  nearest-neighbour + luma-broadcast to bilinear + true RGB
  decode (the published TransNet V2 uses bilinear). See
  [ADR-0220](../../../docs/adr/0223-transnet-v2-shot-detector.md).

- **`cambi.c` GPU port is hybrid host/GPU per
  [ADR-0205](../../../docs/adr/0205-cambi-gpu-feasibility.md) +
  [ADR-0210](../../../docs/adr/0210-cambi-vulkan-integration.md)
  (T7-36 integration).** The Vulkan kernel offloads only the
  embarrassingly-parallel phases (preprocessing scaffold +
  derivative + 7×7 SAT spatial mask + 2× decimate + 3-tap mode
  filter) to the GPU; the precision-sensitive
  `calculate_c_values` sliding-histogram pass + top-K spatial
  pooling stay on the host. Any CPU-side change to the c-value
  formula or the histogram update protocol must keep the host
  residual call site
  (`cambi_vulkan.c::cambi_vk_extract` → `vmaf_cambi_calculate_c_values`)
  lock-step with the CPU `calculate_c_values` — they are
  intentionally the same code, called against the GPU-produced
  image + mask buffers.
  - **`cambi_internal.h` invariant**: this internal-only header
    exposes cambi.c's file-static helpers (`get_spatial_mask`,
    `decimate`, `filter_mode`, `calculate_c_values`,
    `spatial_pooling`, `weight_scores_per_scale`,
    `get_pixels_in_window`, `cambi_preprocessing`,
    `increment_range` / `decrement_range` /
    `get_derivative_data_for_row` callbacks) to the GPU twin via
    a thin trampoline block at the bottom of `cambi.c`. **Do not
    rename or change the signatures of those helpers without
    updating the trampoline block + the header in the same PR
    or the GPU build breaks.** The trampoline body is the *only*
    fork-added code inside `cambi.c`; the upstream-mirror body
    above stays byte-identical to keep Netflix sync clean.
  - Strategy III (fully-on-GPU c-values via direct per-pixel
    histogram) is documented in
    [research digest 0020](../../../docs/research/0020-cambi-gpu-strategies.md)
    but deferred to a future batch — *do not* attempt to
    optimise it inside the v1 hybrid integration.

- **VIF kernelscale stays on the precomputed
  `vif_filter1d_table_s` flow — Strategy E in Research-0024.**
  The fork carries an 11-entry `enum vif_kernelscale_enum`
  plus `vif_filter1d_table_s[11][4][65]` of frozen `const float`
  Gaussian taps in [`vif_tools.h`](vif_tools.h). The Netflix
  upstream chain (`4ad6e0ea` runtime helpers, `8c645ce3`
  prescale options, `41d42c9e` edge-mirror bugfix) computes
  Gaussians at runtime — that loses the SIMD bit-exact
  contract that ADR-0138 / 0139 / 0142 / 0143 froze. **Do not
  port `4ad6e0ea` / `8c645ce3` verbatim.** A future port that
  adds runtime helpers as an *opt-in second path* (Strategy C)
  is allowed; it must not touch the default
  `vif_kernelscale=1.0` + `vif_prescale=1.0` code path.
  Mirror bugfix `41d42c9e` is a separate decision — must come
  with paired `places=4 → places=3` golden loosening per the
  ADR-0142 Netflix-authority precedent. See
  [Research-0024](../../../docs/research/0024-vif-upstream-divergence.md)
  for the full divergence analysis + decision matrix.

- **`compute_adm` signature stays on the fork's parameter
  list — Strategy E in Research-0024.** Netflix upstream
  `4dcc2f7c` adds 12 new parameters (`luminance_level`,
  `adm_csf_scale`, `adm_csf_diag_scale`, `adm_noise_weight`,
  `adm_bypass_cm`, `adm_p_norm`, `adm_f1s0..3`, `adm_f2s0..3`,
  `adm_skip_aim_scale`, `adm_skip_scale0`) plus a new
  `score_aim` output. Threading those through the SIMD paths
  (`adm_avx2.c` / `adm_avx512.c` / `adm_neon.c`) **and** the
  GPU twins (`adm_vulkan.c` / `adm_cuda.c` / `adm_sycl.cpp`)
  is multi-day work, and the new `aim` feature has no fork-
  side golden values yet. **Do not port `4dcc2f7c` until
  there is concrete user demand for `aim` and a coordinated
  cross-backend port plan.** See
  [Research-0024 §"Same divergence test for motion + float_adm"](../../../docs/research/0024-vif-upstream-divergence.md).

### `picture_copy()` carries a `channel` parameter

Upstream commit `d3647c73` (T-NEW-1, ported via this fork's
`upstream/port-d3647c73-feature-speed`) widened the
`picture_copy()` / `picture_copy_hbd()` signatures with a new
`int channel` argument so the new `speed_chroma` and
`speed_temporal` extractors can lift U / V planes from
`VmafPicture`. Every fork-local extractor that calls
`picture_copy()` (`cuda/integer_ms_ssim_cuda.c`,
`vulkan/ssim_vulkan.c`, `vulkan/ms_ssim_vulkan.c`) passes
`channel=0`; the upstream-mirror `float_*` callers already do.
**If a future upstream commit evolves the signature further
(extra parameter, type change), update those four fork-local
call sites in lockstep with the upstream-mirror ones — silently
trailing the upstream signature change will fail compilation
on any GPU backend.** See
[`docs/rebase-notes.md` §0075](../../../docs/rebase-notes.md).

### `speed_chroma` / `speed_temporal` are float-build-only

The two upstream Speed extractors register inside the
`#if VMAF_FLOAT_FEATURES` block in `feature_extractor.c`. They
are absent from a default `meson setup` build; users who want
them must pass `-Denable_float=true`. Do **not** lift them out
of the `#if` block — they call into the Speed-specific helpers
in `vif_tools.c` that are themselves only compiled in the
float-features path.

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
- [ADR-0182](../../../docs/adr/0182-gpu-long-tail-batch-1.md) +
  [ADR-0188](../../../docs/adr/0188-gpu-long-tail-batch-2.md) +
  [ADR-0192](../../../docs/adr/0192-gpu-long-tail-batch-3.md) —
  GPU long-tail batches 1–3. Every registered feature extractor
  now has at least one GPU twin (lpips remains ORT-delegated).
- [ADR-0193](../../../docs/adr/0193-motion-v2-vulkan.md) —
  `motion_v2` Vulkan kernel; edge-replicating mirror diverges
  from `motion.comp` non-replicating mirror — load-bearing per
  the underlying CPU code path.
- [ADR-0205](../../../docs/adr/0205-cambi-gpu-feasibility.md) +
  [ADR-0210](../../../docs/adr/0210-cambi-vulkan-integration.md) —
  cambi Vulkan integration (Strategy II, hybrid host/GPU).
  Precision-sensitive `calculate_c_values` + top-K stay on host;
  GPU phases are integer + bit-exact.
- [ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  GPU-parity CI gate: per-feature `FEATURE_TOLERANCE` map in
  `scripts/ci/cross_backend_parity_gate.py` is single source of
  truth. Every new GPU twin needs an entry.

## Newly-arrived shipped surfaces (rebase awareness)

- **MS-SSIM `enable_lcs` GPU implementation (T7-35, PR #207 open)**
  — wires the existing CPU `enable_lcs` 15-extra-metrics through
  CUDA + Vulkan + SYCL MS-SSIM kernels. On rebase: ensure the
  option metadata stays declared on the GPU paths even if the
  body is still TODO.
- **psnr chroma Vulkan (T3-15(b), PR #204 open, ADR-0216
  placeholder)** — `psnr_cb` + `psnr_cr` Vulkan twins next to
  `psnr_y`
  ([ADR-0182](../../../docs/adr/0182-gpu-long-tail-batch-1.md)).
- **MobileSal saliency extractor (T6-2a, PR #208 open, ADR-0218
  placeholder)** — first half of T6-2 (encoder-side ROI bundle).
  DNN-backed; opens sessions through
  [`../dnn/`](../dnn/AGENTS.md).
- **TransNet V2 shot-boundary extractor (T6-3a, PR #210 open)** —
  second half of T6-2 bundle, ~1M params. DNN-backed.
- **FastDVDnet temporal pre-filter (T6-7, PR #203 open, ADR-0215
  placeholder)** — 5-frame window pre-filter feeding
  ssim/ms_ssim. DNN-backed.
- **SVE2 SIMD ports (T7-38, PR #201 open, ADR-0213 placeholder)**
  — SSIMULACRA 2 PTLR + IIR-blur SVE2; same bit-exact contract
  as the existing NEON ports per
  [ADR-0161](../../../docs/adr/0161-ssimulacra2-simd-bitexact.md)
  / [ADR-0162](../../../docs/adr/0162-ssimulacra2-iir-blur-simd.md)
  / [ADR-0163](../../../docs/adr/0163-ssimulacra2-ptlr-simd.md).
- **Upstream ports**: `feature/motion` options from `b949cebf`
  (T-NEW-1) MERGED via PR #197 (2026-04-29). `feature/speed`
  port from `d3647c73` (`speed_chroma` + `speed_temporal`) is
  PR #213 (open). 32-bit ADM/cpu fallbacks (`8a289703` +
  `1b6c3886`) are PR #212 (open).
