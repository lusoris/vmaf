# ADR-0125: MS-SSIM decimate SIMD fast paths (AVX2 + AVX-512)

- **Status**: Proposed
- **Date**: 2026-04-20
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: simd, testing, agents

## Context

MS-SSIM runs a 5-scale pyramid. Each scale reduction in
[`libvmaf/src/feature/ms_ssim.c`](../../libvmaf/src/feature/ms_ssim.c)
calls [`_iqa_decimate`](../../libvmaf/src/feature/iqa/decimate.c) to
low-pass-filter the previous-scale image with a fixed 9×9 9/7
biorthogonal wavelet kernel (`g_lpf`) and subsample by factor 2. The
vendored decimate implementation walks every destination pixel and
invokes `_iqa_filter_pixel`, which evaluates the full 9×9 kernel in
scalar C (81 multiply-adds per destination pixel) and delegates
out-of-bounds access through a function pointer
([`KBND_SYMMETRIC`](../../libvmaf/src/feature/iqa/convolve.c)). On
1920×1080 inputs the decimate step alone accounts for roughly 40 % of
MS-SSIM wall time in profiling; the per-pixel function-pointer
indirection defeats the compiler's ability to keep kernel coefficients
in registers.

The SSIM primitives MS-SSIM reuses already have AVX2 and AVX-512
specialisations wired through `_iqa_ssim_set_dispatch` in
[`libvmaf/src/feature/iqa/ssim_tools.c`](../../libvmaf/src/feature/iqa/ssim_tools.c),
so x86 SIMD paths already exist elsewhere on this hot path; the decimate
step is the remaining scalar gap. The 9×9 LPF is also known to be
separable — `ms_ssim.c` defines the 1-D horizontal and vertical
coefficients (`g_lpf_h`, `g_lpf_v`) alongside the 2-D form, but
`_iqa_decimate` never uses them.

Two fork-wide rules shape the implementation:

- The `libvmaf/src/feature/iqa/` subtree is a verbatim BSD-2011 Tom
  Distler import. The fork treats it the same as upstream Netflix code
  for rebase hygiene — we do not modify it unless we have to.
- The Netflix CPU golden gate ([CLAUDE.md §8](../../CLAUDE.md)) does
  not exercise MS-SSIM (the three reference pairs assert VMAF and
  feature scores, not MS-SSIM), but any new SIMD path must still match
  scalar output exactly across cross-backend snapshot tests before
  landing.

## Decision

We add MS-SSIM-specialised decimate kernels under
`libvmaf/src/feature/x86/ms_ssim_decimate_avx2.{c,h}` and
`libvmaf/src/feature/x86/ms_ssim_decimate_avx512.{c,h}`. Each kernel
fuses the separable 9-tap LPF (using `g_lpf_h` / `g_lpf_v`) with the
factor-2 subsampling so only every other row of the horizontal pass is
written and only every other column of the vertical pass is computed —
reducing FLOP count from 81/dest-pixel (2-D) to ~9/dest-pixel
(separable + stride-2). The kernels assume the MS-SSIM use case:
`factor == 2`, `LPF_LEN == 9`, `normalized == 1`,
`bnd_opt == KBND_SYMMETRIC`, `float` input; any caller outside these
invariants falls back to the scalar `_iqa_decimate`. Dispatch lives in
`ms_ssim.c` (or a companion `ms_ssim_dispatch.c` if it grows), chosen
by the same `vmaf_get_cpu()` flags the SSIM dispatch uses. The vendored
`libvmaf/src/feature/iqa/decimate.c` is **not modified** — the
dispatching happens in the MS-SSIM caller, not in the iqa/ helper.

Bit-exactness is the correctness contract. A new
`libvmaf/test/test_ms_ssim_decimate.c` runs scalar, AVX2, and AVX-512
against the same synthetic + real-YUV inputs and asserts byte-identical
float output. Because the SIMD path reorders FMA summation relative to
the scalar 2-D form (separable pass order + 8-lane horizontal sum
instead of row-major scan), the scalar reference used for comparison is
a **new scalar-separable path** (also added in this PR) that evaluates
the separable 1-D form in the same summation order as the SIMD
kernels. MS-SSIM end-to-end scores may shift by ≤ 1 ULP relative to the
pre-change 2-D scalar output; we compare the new output against
**snapshot-refreshed** MS-SSIM scores from
`testdata/scores_cpu_ms_ssim.json` rather than the pre-change values,
and document the snapshot refresh in the PR description per
`/regen-snapshots`.

NEON is deferred to a smaller follow-up PR on `arm64`; the dispatch
scaffold leaves a clear slot for `ms_ssim_decimate_neon` to plug into.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Specialise decimate for MS-SSIM's 9×9 separable LPF factor-2 case, x86 AVX2 + AVX-512 (chosen) | Largest speed-up (9×9 → separable ≈ 9× FLOP reduction; SIMD another 8–16×); leaves vendored iqa/ file untouched; matches the established per-feature `x86/*_avx2.c` layout; realistic day-scale delivery | Needs a new scalar-separable reference for bit-exactness (SIMD reorders sums vs 2-D form); snapshot refresh required; no immediate benefit for non-MS-SSIM callers of `_iqa_decimate` | Decision |
| Generalise `_iqa_decimate` itself with an AVX2/AVX-512 inner loop (edit the vendored file) | One code path for every decimate caller; smaller dispatch surface | Modifies BSD-2011 Tom Distler code — increases rebase noise if Netflix ever re-syncs upstream iqa/; the only other caller of `_iqa_decimate` in-tree today is also MS-SSIM, so the "generalisation" has zero extra consumers; mixing SIMD intrinsics into the vendored file pollutes a clean subtree | Rejected — no second caller exists, and modifying vendored code without cause is bad rebase hygiene (CLAUDE.md §10 / AGENTS) |
| Port the decimate step to the CUDA / SYCL backends as well in the same PR | Cross-backend parity in one shot | GPU backends are not bit-exact to CPU (per memory rule `feedback_golden_gate_cpu_only.md`); a bit-exactness PR that also touches GPU must either accept non-exactness (contradicting the contract) or duplicate the scalar-separable reference on device (huge scope creep) | Deferred to a follow-up; keep this PR CPU-only to keep the bit-exactness claim crisp |
| Keep scalar decimate; optimise `_iqa_filter_pixel` instead (inline it, drop the function pointer) | Smaller diff; no new files | Solves the indirection cost but not the 81-vs-9 FLOP gap; leaves the largest win on the floor | Rejected — the FLOP count is the dominant cost, not the indirection |
| Ship AVX2 + AVX-512 + NEON all in one PR | Single review for all ISAs | NEON needs a separate arm64 CI leg and the fork's arm64 NEON SIMD coverage is already ahead of x86 (VIF, ADM); bundling triples the review surface | Per user direction (popup 2026-04-20): "AVX2 + AVX-512 only, NEON follow-up" |

## Consequences

- **Positive**: MS-SSIM wall time drops by the decimate-share fraction
  (expect ~25–30 % end-to-end on 1080p with AVX-512); the vendored
  `iqa/` subtree stays identical to its 2011 upstream; the new kernels
  follow the established `libvmaf/src/feature/x86/*_avx2.c` pattern and
  are trivially unit-tested against scalar; the dispatch scaffold makes
  the NEON follow-up a pure add rather than a refactor.
- **Negative**: MS-SSIM scalar output shifts by up to 1 ULP versus the
  pre-change 2-D scalar path (because the separable-sum reduction
  order differs), forcing a snapshot refresh of
  `testdata/scores_cpu_ms_ssim.json`; reviewers have to check that the
  refresh is limited to MS-SSIM columns; the SIMD kernel files add
  ~600 LOC of intrinsics that need Power-of-10 §5 assertion density
  coverage (≥1 `VMAF_ASSERT_DEBUG` per function ≥20 lines).
- **Neutral / follow-ups**:
  - Research digest `docs/research/0002-ms-ssim-decimate-simd.md`
    (assigned ID 0002 — next available) captures the benchmark
    methodology, FLOP accounting, and prior art for separable wavelet
    decimation.
  - `docs/metrics/ms_ssim.md` (or a dedicated section in the nearest
    existing MS-SSIM doc) gets a "SIMD paths" subsection per ADR-0100
    per-surface bar for SIMD paths.
  - `.github/AGENTS.md` (or `libvmaf/src/feature/AGENTS.md` if we add
    a per-subtree one) gets a rebase-sensitive invariant note: the
    9×9 LPF coefficients in `ms_ssim.c` must match the coefficients
    hard-coded in `ms_ssim_decimate_{avx2,avx512}.c`; any upstream
    change to `g_lpf_h` / `g_lpf_v` requires re-deriving the SIMD
    coefficients.
  - `CHANGELOG.md` "lusoris fork" entry.
  - `docs/rebase-notes.md` entry: MS-SSIM SIMD paths are fork-only;
    re-test with `meson test -C build -t ms_ssim_decimate` after any
    upstream MS-SSIM touch.
  - NEON follow-up PR (arm64): `ms_ssim_decimate_neon.c`, same
    bit-exactness harness, smaller scope.
  - Reproducer command (for PR description): `meson test -C build
    test_ms_ssim_decimate` + `./build/libvmaf/tools/vmaf --feature
    ms_ssim -r python/test/resource/yuv/src01_hrc00_576x324.yuv -d
    python/test/resource/yuv/src01_hrc01_576x324.yuv -w 576 -h 324 -p
    yuv420p`.

## References

- Source: user popup (2026-04-20) — workstream selection:
  "SIMD gap-fill: ms_ssim AVX2/AVX-512".
- Scope source: user popup (2026-04-20) — "Vectorize `_iqa_decimate` +
  bit-exactness tests".
- ISA scope source: user popup (2026-04-20) — "AVX2 + AVX-512 only,
  NEON follow-up".
- Predecessor ADRs:
  [ADR-0012](0012-coding-standards-jpl-cert-misra.md) — coding
  standards (Power-of-10 §5 assertion density applies to new fork
  files).
  [ADR-0106](0106-adr-maintenance-rule.md) — this ADR is written
  before any implementation commit on `feat/ms-ssim-decimate-simd`.
  [ADR-0108](0108-deep-dive-deliverables-rule.md) — six deep-dive
  deliverables apply to this PR.
- Upstream MS-SSIM kernel: Rouse & Hemami, *Analyzing the Role of
  Visual Structure in the Recognition of Natural Image Content with
  Multi-Scale SSIM* (the `ms_ssim.c` comment block cites "MS-SSIM*
  (Rouse/Hemami)").
- Vendored decimate origin:
  [Tom Distler IQA library, 2011](http://tdistler.com) — header in
  [`libvmaf/src/feature/iqa/decimate.c`](../../libvmaf/src/feature/iqa/decimate.c).
