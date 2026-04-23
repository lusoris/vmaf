# ADR-0144: Port Netflix upstream VIF edge-mirror bugfix + golden update

- **Status**: Accepted
- **Date**: 2026-04-22
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: upstream-port, correctness, vif, testing, golden-gate

## Context

Netflix upstream commits
[`41d42c9e`](https://github.com/Netflix/vmaf/commit/41d42c9e)
"feature/vif: port helper functions, bugfix for edge mirroring"
(Kyle Swanson, 2026-04-20) and
[`bc744aa3`](https://github.com/Netflix/vmaf/commit/bc744aa3)
"loosen assertion precision for vif mirror bugfix" (Kyle Swanson,
2026-04-21) are a paired correctness fix for the VIF edge-mirror
indexing in the convolve path.

The bug: `convolution_internal.h` (`convolution_edge_s`,
`convolution_edge_sq_s`, `convolution_edge_xy_s`) and
`vif_tools.c` (`vif_filter1d_s`, `vif_filter1d_sq_s`) used
`j_tap = width - (j_tap - width + 1)` for the mirror reflection
across the right/bottom edge. The correct formula is
`j_tap = width - (j_tap - width + 2)` — the `+ 1` form indexes the
sample right at the edge twice instead of reflecting past it.
Netflix's upstream fixes this in five sites. The fix is
intentionally non-bit-exact: final VIF scores shift by ~1e-4 (and
final VMAF scores downstream by ~1e-3 through model feature
weights) because every frame's edge region computes slightly
different values.

Netflix's companion commit `bc744aa3` loosens ~40 `assertAlmostEqual`
calls across six python tests (result_test, feature_assembler_test,
local_explainer_test, routine_test, vmafexec_feature_extractor_test,
feature_extractor_test) from `places=4` to `places=3` on
intermediate VIF feature values, so the tests absorb the drift
from the fixed mirror. Netflix did not retrain any VMAF model —
they explicitly accept that downstream final-VMAF drift is below
the model's training noise floor (sub-model-retraining threshold,
confirmed in session discussion with user re: fork policy).

This shift creates a decision for the fork: project rule #1
(CLAUDE.md §8) says "never modify Netflix golden assertions".
But here Netflix *themselves* moved the golden assertions upstream
with a correctness fix. The fork follows Netflix's authority
(same precedent as ADR-0142 / ADR-0143). Rule #1 addresses
silent fork drift, not upstream-authored test updates the fork
must track to stay synchronised.

## Decision

We will port both `41d42c9e` and `bc744aa3` paired — the bugfix
and its companion golden loosening are treated as one atomic
upstream authority. Specific rules:

1. **Bugfix in 5 sites** — three inline `convolution_edge_*_s`
   helpers in `convolution_internal.h` and two scalar fallback
   paths in `vif_tools.c` (`vif_filter1d_s:432`,
   `vif_filter1d_sq_s:500`). `+ 1` → `+ 2` change, single-byte
   per site. Fork keeps its clang-format-ed layout (4-space
   indent, braces-same-line) — upstream uses tabs; no behavioural
   change from the reformat.
2. **Python test deltas** — upstream's full file content for the
   affected tests is adopted (the deltas span ~250 individual
   `assertAlmostEqual` value updates plus the `places=` looseners).
   Fork's black/isort pass restores fork-standard formatting
   without changing any numeric value.
3. **No model retrain** — per user direction 2026-04-22 (popup
   answer: "Port bugfix + adopt Netflix's updated goldens"), the
   final-VMAF drift relative to the existing model is accepted as
   sub-model-floor noise. When Netflix publishes a retrained
   model, the drift self-resolves.
4. **Project rule #1 interpretation** — the rule freezes fork
   drift against Netflix's baseline. Upstream-authored test
   updates that the fork ports ≠ fork modifying Netflix golden
   data. Documented here as the standing interpretation for this
   class of port (ADR-0142 established the precedent).
5. **ADR-0141 touched-file cleanup** — `convolution_internal.h`
   and `vif_tools.c` are now zero clang-tidy warnings. Drop
   redundant `#pragma once`, add braces to inline `if/else` mirror
   branches, split multi-declarations. Four upstream
   `readability-function-size` hits (`vif_statistic_s`,
   `vif_filter1d_{s,sq_s,xy_s}`) get NOLINT with T7-5 sweep
   citation — their shape is load-bearing for upstream rebase.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| **Port bugfix + adopt Netflix's updated goldens paired (chosen)** | Correctness fix lands; fork stays synchronised with upstream; rule #1 stays coherent via the ADR-0142 precedent; T7-5-aligned NOLINTs keep rebase story clean | Fork diverges ~1e-3 at final-VMAF scale from pre-port master; requires documented carve-out for rule #1 | **Decision** — per user popup 2026-04-22 |
| Defer until Netflix tags a retrained model | Strict rule #1 compliance | Fork carries the known-wrong mirror indexing indefinitely; drift from upstream VIF feature values grows every subsequent commit; correctness vs rule-bureaucracy trade-off favours correctness | Rejected — user explicitly rejected this option |
| Port bugfix, keep fork's `places=4` assertions | Fork's own stricter tolerance | Netflix CPU golden CI leg goes permanently red on ~40 assertions; tests would need to be suppressed individually; fork bits-rots | Rejected — creates a hidden-CI state |
| Port bugfix, regenerate ALL snapshots via `/regen-snapshots` | Fork-internal test invariants refreshed | Netflix golden assertions are not fork-added snapshots (they live in `python/test/`, which CLAUDE.md §8 explicitly flags as Netflix-owned); regenerating them is exactly what rule #1 forbids | Rejected — category error |

## Consequences

- **Positive**:
  - VIF edge math is correct; future model retrains can safely
    use fork-produced features.
  - Fork stays synchronised with upstream VIF path; subsequent
    VIF upstream commits (`4ad6e0ea` helpers, `18e8f1c5`
    sigma_nsq, etc.) rebase on the correct math.
  - Two touched C files are zero clang-tidy warnings after the
    ADR-0141 cleanup.
- **Negative**:
  - Final VMAF scores shift by ~1e-3 vs pre-port master. Below
    perceptual discriminability (VMAF scale 0-100, subjective
    discriminability ~1-2 points, PLCC-vs-subjective ~0.9+).
  - Per-frame VIF feature values shift by ~1e-4 — user-visible
    in `vmaf --feature float_vif` output.
  - Downstream consumers with pinned sub-1e-3 VMAF test
    tolerances (e.g. cross-backend ULP comparisons) need to
    re-baseline.
- **Neutral / follow-ups**:
  - `docs/rebase-notes.md` entry 0037 (both commits + the rule-#1
    interpretation carve-out).
  - `CHANGELOG.md` entry under Unreleased → Fixed (the edge-mirror
    bug itself) + note on the score shift.
  - `libvmaf/src/feature/AGENTS.md` invariant: mirror indexing is
    `+ 2`, never `+ 1` — on upstream sync, reject any reversion.
  - When Netflix publishes a retrained VMAF model that
    incorporates the fixed mirror, port it and note the drift
    resolution in the model's sidecar.

## References

- Upstream commits:
  [Netflix/vmaf@41d42c9e](https://github.com/Netflix/vmaf/commit/41d42c9e),
  [Netflix/vmaf@bc744aa3](https://github.com/Netflix/vmaf/commit/bc744aa3).
- Related ADRs:
  [ADR-0024](0024-netflix-golden-preserved.md) — rule #1 source;
  [ADR-0138](0138-iqa-convolve-avx2-bitexact-double.md) /
  [ADR-0139](0139-ssim-simd-bitexact-double.md) — the IQA-stack
  bit-exactness work (related file `iqa/convolve.c`, separate from
  `vif_tools.c`);
  [ADR-0141](0141-touched-file-cleanup-rule.md) — touched-file
  cleanup applied here;
  [ADR-0142](0142-port-netflix-18e8f1c5-vif-sigma-nsq.md) /
  [ADR-0143](0143-port-netflix-f3a628b4-generalized-avx-convolve.md)
  — upstream-authority precedent for golden-assertion updates.
- Source: user direction 2026-04-22 ("manually integrate latest
  commits on upstream" + popup answer "Port bugfix + adopt
  Netflix's updated goldens" + subsequent "go on").
