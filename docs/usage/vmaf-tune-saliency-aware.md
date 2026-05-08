# `vmaf-tune --saliency-aware` (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

The `--saliency-aware` mode (per
[ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md)) routes the
score axis through the saliency-weighted VMAF path
([`vmaf-roi-score.md`](vmaf-roi-score.md)) so foreground pixels
dominate the per-cell verdict instead of background regions. The
companion flags are `--saliency-offset` and `--saliency-model` (the
latter selects the trained ONNX saliency model — see
[`saliency_student_v1.md`](../ai/models/saliency_student_v1.md)).

Status: Accepted. Wired in `tools/vmaf-tune/src/vmaftune/cli.py`.

## See also

- [`vmaf-tune.md`](vmaf-tune.md) — the base tool.
- [`vmaf-roi-score.md`](vmaf-roi-score.md) — the saliency-weighted
  score path that this flag delegates to.
- [`docs/ai/models/saliency_student_v1.md`](../ai/models/saliency_student_v1.md)
  — the shipped saliency model.
- [ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md) — design
  decision.
