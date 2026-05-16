# ADR-0453: Tiny-AI training scaffold — second iteration (Netflix corpus prep, 2026-05-16)

- **Status**: Proposed
- **Date**: 2026-05-16
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: `ai`, `training`, `mcp`, `fork-local`, `onnx`, `docs`

## Context

[ADR-0242](0242-tiny-ai-netflix-training-corpus.md) (2026-04-27) established the
architecture decisions for tiny-AI training on the original Netflix VMAF corpus:
distillation teacher, loader API, data-path safety invariants, and evaluation harness.
[ADR-0417](0417-tiny-ai-netflix-training-scaffold-pr.md) (2026-05-11) registered the
first scaffold draft PR; that PR merged as PR #759, landing `ai/data/netflix_loader.py`,
`test_netflix_loader.py`, the MCP smoke-test suite, and the corpus-path documentation.

This ADR registers the **second iteration** of the scaffold branch
(`ai/tiny-netflix-training-scaffold`), triggered by the daily prep-scaffolding
routine on 2026-05-16 after PR #152 (`fix/volk-static-archive-priv-remap`, ADR-0198)
merged to master. The branch carries incremental additions the routine identified as
missing or stale since the first scaffold PR merged:

- Research digest update (digest 0136) — extends the 2026-04-27 survey (digest 0019)
  and the 2025-era update (digest 0099) with current distillation and ONNX Runtime
  literature.
- CHANGELOG fragment under `changelog.d/added/` per ADR-0221 (the first scaffold PR
  pre-dated the fragment convention's enforcement).
- `docs/rebase-notes.md` entry for the `ai/` and `mcp-server/` surfaces (previously
  missing).
- `docs/ai/training-data.md` cross-reference to this ADR.

The corpus itself is on the user's local machine at `.workingdir2/netflix/{ref,dis}/`
(37 GB, gitignored). The 9 reference and 70 distorted YUVs follow the Netflix
encoding-ladder naming convention documented in `docs/ai/training-data.md`. The branch
does **not** run training, download data, or modify Netflix golden assertions.

The MCP smoke-test (`mcp-server/vmaf-mcp/tests/test_smoke_e2e.py`) and the loader
implementation (`ai/data/netflix_loader.py`) shipped in the first scaffold PR and are
not changed here. The branch touches only documentation and tooling surfaces.

## Decision

We will open a second scaffold draft PR to deliver the four missing items listed above.
The PR is marked DRAFT. No architecture decision is revisited — ADR-0242 remains the
governing decision for the training architecture. This ADR records the PR's scope and
the rationale for a second iteration rather than amending the first.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| (a) Amend ADR-0417 / re-open PR #759 | Single PR history | PR #759 merged; reopening would require a revert + re-merge | Not possible post-merge |
| (b) Land changes directly on master via a hotfix commit | Fewer PRs | Bypasses review gate; no draft-PR discoverability | Violates CLAUDE.md §12 rule 3 (branch + PR required) |
| (c) Defer until the actual training run PR | Consolidates changes | Research digest and CHANGELOG fragment block the training PR's CI | Blocks training; deferred items compound |
| **(d) Second scaffold draft PR (chosen)** | Clean history; unblocks training PR; idempotent daily routine | One extra PR in the log | Preferred: minimal blast radius, follows the established scaffold pattern |

## Consequences

- **Positive**: The CHANGELOG and rebase-notes gaps are filled before the training
  run PR opens; the CI deliverables checker (ADR-0108) passes on the training PR.
- **Negative**: An extra PR in the history for what is largely a housekeeping pass.
- **Neutral / follow-ups**: After this PR merges, the next PR in the workstream is
  the actual training run (`vmaf-train fit --config ai/configs/fr_tiny_v1.yaml`).
  That PR will need its own ADR documenting the chosen architecture (distill from
  `vmaf_v0.6.1` vs train from scratch) and the GPU access plan.

## References

- [ADR-0242](0242-tiny-ai-netflix-training-corpus.md) — architecture and distillation
  policy decisions.
- [ADR-0417](0417-tiny-ai-netflix-training-scaffold-pr.md) — first scaffold PR
  registration (PR #759, merged 2026-05-15).
- [ADR-0198](0198-volk-priv-remap-static-archive.md) — volk static-archive fix (PR
  #152) whose merge triggered this routine run.
- [ADR-0108](0108-deep-dive-deliverables-rule.md) — deep-dive deliverables rule.
- [ADR-0221](0221-changelog-adr-fragment-pattern.md) — changelog fragment pattern.
- Research digest 0019 — `docs/research/0019-tiny-ai-netflix-training.md`.
- Research digest 0099 — `docs/research/0099-tiny-ai-netflix-training-update.md`.
- Research digest 0136 — `docs/research/0136-tiny-ai-netflix-training-refresh-2026-05-16.md`.
- User memory: `project_netflix_training_corpus_local.md` (corpus location and naming
  convention; paraphrased here per the global user-quote-handling rule).
- Source: `req` — daily prep-scaffolding routine, 2026-05-16.
