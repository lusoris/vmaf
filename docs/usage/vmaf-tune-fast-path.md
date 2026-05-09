# `vmaf-tune` fast-path prod wiring (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADRs for the authoritative shape; full prose follows in a
> later PR.

`vmaf-tune` ships a *fast-path* — a learnt regressor (the FR-regressor
model family) that estimates the per-cell VMAF score without
running the full `vmaf` CLI for every grid cell. When enabled, the
encode runs as normal and the score is predicted from encoder /
preset / CRF / source features. Used as a coarse pre-filter before
running the exact-VMAF score on the most-promising cells.

Status: Accepted (Phase 0); production wiring per
[ADR-0304](../adr/0304-vmaf-tune-fast-path-prod-wiring.md). The
underlying model is trained per
[ADR-0276](../adr/0276-vmaf-tune-fast-path.md) and shipped via the
`fr_regressor_v3` model card
([`docs/ai/models/fr_regressor_v3.md`](../ai/models/fr_regressor_v3.md)).

## See also

- [`vmaf-tune.md`](vmaf-tune.md) — the base tool.
- [`docs/ai/models/fr_regressor_v3.md`](../ai/models/fr_regressor_v3.md)
  — the model card for the regressor that powers the fast-path.
- [ADR-0276](../adr/0276-vmaf-tune-fast-path.md) /
  [ADR-0304](../adr/0304-vmaf-tune-fast-path-prod-wiring.md) —
  design decisions.
