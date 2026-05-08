# ADR-0325: `vmaf-tune` Phase F — `auto` adaptive recipe-aware tuning

- **Status**: Proposed
- **Date**: 2026-05-08
- **Deciders**: Lusoris
- **Tags**: tooling, automation, vmaf-tune, ffmpeg, codec, fork-local

## Context

Phases A through E of `vmaf-tune` ship as eight standalone CLI
subcommands (`corpus`, `recommend`, `fast`, `predict`,
`tune-per-shot`, `recommend-saliency`, `ladder`, `compare`) plus
three orthogonal modes (HDR auto-detect via [ADR-0300](0300-vmaf-tune-hdr-aware.md),
sample-clip via [ADR-0301](0301-vmaf-tune-sample-clip.md), and
resolution-aware model selection via [ADR-0289](0289-vmaf-tune-resolution-aware.md)).
The operator-facing question — "give me the cheapest encode that
meets target VMAF for this content" — currently requires the
operator to compose roughly eight phases manually, ≈ 5–6 hours of
wall-clock for a 2-hour 1080p title (see
[Research-0067](../research/0067-vmaf-tune-phase-f-feasibility-2026-05-08.md)
cost table). The user's 2026-05-08 vision text (paraphrased: "the
real long-term potential is building an adaptive encoding ecosystem
around community-generated training data, perceptual analysis and
continual model improvement") frames Phase F as the first
composition layer that exposes this ecosystem behind a single CLI
verb.

[Research-0067](../research/0067-vmaf-tune-phase-f-feasibility-2026-05-08.md)
walks the cost model, the seven short-circuit cases, and the four
failure modes; it concludes that a deterministic decision tree
hits the explainability + reproducibility floor the fork requires
without sacrificing the wall-time savings a learned policy would
deliver.

## Decision

We will ship `vmaf-tune auto` as the Phase F entry point, implemented
as a **deterministic decision tree** in
`tools/vmaf-tune/src/vmaftune/auto.py` that composes the existing
phase subcommands sequentially. The tree is hand-coded (no learned
policy at runtime), every branch maps to an existing per-phase ADR
contract, and the full tree fits within a 30-line pseudocode
specification (see Research-0067 §"Phase F decision tree"). The
phased rollout below splits the work into four follow-up PRs so each
slice ships with its own validation:

- **F.0 — this ADR (design only).** No code. Establishes the
  decision tree, short-circuit list, escalation policy, and the
  rule that Phase F never invents new sub-phases.
- **F.1 — scaffolded `vmaf-tune auto` (`tools/vmaf-tune/src/vmaftune/auto.py`).**
  Sequential composition of the existing subcommands; no
  short-circuits, no escalation. The CLI flags `--src`,
  `--target-vmaf`, `--max-budget-bitrate`, `--allow-codecs`,
  `--codec`, `--smoke` are stable from this PR forward. `--smoke`
  exercises the composition end-to-end with mocked sub-phases (no
  ffmpeg, no ONNX); production wiring lands in F.2.
- **F.2 — short-circuit logic.** The seven cases from
  Research-0067 §"When Phase F should short-circuit" become
  conditional branches: single-rung ladder when source < 2160p;
  codec known; predictor verdict GOSPEL; short / low-variance
  source skips Phase D; non-animation / non-screen-content skips
  saliency; SDR source skips HDR pipeline; sample-clip propagates
  to internal sweeps.
- **F.3 — confidence-aware fallbacks.** When Phase C's predictor
  returns FALL_BACK on a (rung, codec) cell, escalate **only that
  cell** to `recommend.coarse_to_fine`. The escalation is per-cell,
  bounded, and logged. GOSPEL and LIKELY verdicts skip the
  escalation. Encoder-ROI / saliency-binary missing degrades to a
  warning, never aborts.
- **F.4 — per-content-type recipe overrides.** Auto-detect
  animation / live-action / screen-content via a fork-local
  classifier (TransNet V2 is already present and exposes a
  shot-cut histogram that correlates well with the three classes;
  fallback heuristics in `auto.py` cover the no-classifier case).
  The class drives saliency / preset / per-shot defaults; users
  override via explicit flags.

The tree is the v1 surface; learned policy stays a research
follow-up after F.1–F.4 have produced enough labelled compositions
to seed a future supervised baseline. No closed-source ML services,
no Internet calls during encode, no runtime learned-policy
inference — Phase F is integration, not invention.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Deterministic decision tree (chosen)** | Explainable; reproducible across runs; every branch maps to an existing ADR contract; testable with mocked sub-phases; no runtime ML dependency on the auto path. | Hard-codes priorities; new sub-phases require tree edits. | Picked: matches the fork's "no learned policy at runtime" constraint, the user's explainability requirement, and the per-phase contract carve-outs in ADR-0237 / ADR-0276 / ADR-0295. |
| Pure-grid composition (today's manual workflow) | Zero new code; fully reproducible. | 8-step manual composition; ≈ 5–6 h wall-clock for a typical 2-hour movie; high operator-error rate; the colleague's "day per movie" pain point. | Rejected: the cost is exactly why Phase F is a backlog item. |
| Optuna over the full composition space | Strong optimum; reuses Phase A.5 search infrastructure. | Per-source TPE warm-up cost; no closed-form way to express "skip Phase D when source is short"; opaque to operators ("why did it pick x265?"); too few independent samples per source for Bayesian search to beat a hand-tuned tree. | Rejected: search-over-recipes is the wrong model for a discrete composition problem with operator-explainability requirements. |
| Learned policy (RL or supervised over a Phase A corpus) | Adapts to corpus drift; aligns with the long-term "continual model improvement" arm of the user's vision text. | Requires a labelled "this composition was right" dataset that doesn't yet exist; runtime inference adds an ONNX dependency to the auto path; reproducibility suffers (model drift between runs); violates the fork's "no learned-policy at runtime" constraint. | Rejected for v1; revisit as a research experiment once the deterministic tree has emitted a labelled corpus of recipe choices. |
| One mega-subcommand replacing all phases | Single operator surface. | Breaks every existing per-phase contract; downstream consumers (CI, MCP server, the FFmpeg patch series) lose stable per-phase hooks; ADR-0237 / ADR-0276 / ADR-0295 explicitly carve those contracts. | Rejected: the carve-outs are load-bearing. |
| Ship `auto` as a thin shell-script wrapper | No new Python module. | No way to express the short-circuit conditions or the FALL_BACK escalation cleanly in `bash`; smoke testing harder; operator-error surface (quoting, env vars) larger. | Rejected: the harness is Python; `auto.py` keeps the contract testable. |

## Consequences

- **Positive**:
  - Operators get a single CLI verb (`vmaf-tune auto`) for the
    common "encode at target VMAF" workflow; the eight-step manual
    composition collapses to one invocation.
  - Wall-clock floor on short-circuit-eligible content (1080p
    SDR photographic, single-codec, GOSPEL predictor) is the
    final encode itself — no redundant sweeps.
  - Every branch is testable in isolation; F.1 ships with mocked
    sub-phases for end-to-end smoke coverage.
  - Phase F's deterministic-tree footprint provides a labelled
    audit trail (which short-circuits fired, which fallbacks
    escalated) that a future learned-policy study can train on.

- **Negative**:
  - The decision tree is hand-tuned; recipe drift (new codec,
    new preset) requires tree edits, not just a re-trained model.
  - Adding a new sub-phase (F.5 onwards) means tree-and-test
    edits, not configuration-only changes.
  - Operator must still understand what Phase F **chose** — the
    explainability surface (per-cell verdicts, escalation log) is
    a new doc burden.

- **Neutral / follow-ups**:
  - F.1 PR ships `tools/vmaf-tune/src/vmaftune/auto.py`,
    `tests/test_auto.py`, and the `--smoke` mode. Adds the
    `auto` subcommand to `cli.py`.
  - F.1 PR adds a docs page under `docs/usage/vmaf-tune.md`
    documenting the `auto` flags and the decision tree (the doc
    must reproduce the pseudocode block from Research-0067 so
    operators can predict what `auto` will do without reading
    the code).
  - F.2 / F.3 / F.4 are sibling PRs gated on F.1 landing; each
    extends `auto.py` with smoke and integration tests.
  - Future: when a labelled-composition corpus exists, evaluate a
    supervised classifier as a recipe selector; the deterministic
    tree stays as the canonical fallback per the no-runtime-ML
    constraint.
  - Adds rebase-notes entry: Phase F ties together the per-phase
    contracts; future upstream syncs that touch any one phase
    must re-validate the tree.

## References

- [ADR-0237](0237-quality-aware-encode-automation.md) — `vmaf-tune`
  umbrella decision (Phase A scaffold).
- [ADR-0276 fast](0276-vmaf-tune-fast-path.md) — Phase A.5 proxy +
  Bayesian.
- [ADR-0276 phase-d](0276-vmaf-tune-phase-d-per-shot.md) — Phase D
  per-shot scaffold.
- [ADR-0289](0289-vmaf-tune-resolution-aware.md) — resolution-aware
  model selection.
- [ADR-0293](0293-vmaf-tune-saliency-aware.md) — saliency-aware ROI
  tuning.
- [ADR-0295](0295-vmaf-tune-phase-e-bitrate-ladder.md) — Phase E
  per-title ABR ladder.
- [ADR-0300](0300-vmaf-tune-hdr-aware.md) — HDR-aware encoding +
  scoring.
- [ADR-0301](0301-vmaf-tune-sample-clip.md) — sample-clip mode.
- [ADR-0306](0306-vmaf-tune-coarse-to-fine.md) — coarse-to-fine CRF
  search.
- [Research-0060](../research/0060-vmaf-tune-fast-path.md) — Phase
  A.5 cost model (parent).
- [Research-0067](../research/0067-vmaf-tune-phase-f-feasibility-2026-05-08.md)
  — Phase F feasibility (companion digest, this ADR).
- Source: `req` — paraphrased user vision text from the 2026-05-08
  ChatGPT exchange ("the real long-term potential is building an
  adaptive encoding ecosystem around community-generated training
  data, perceptual analysis and continual model improvement"); the
  English-translated paraphrase lives in this ADR's Context to
  satisfy the user-quote-handling rule for non-References sections.
