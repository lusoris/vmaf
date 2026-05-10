# Ensemble training kit (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows once
> the kit reaches Accepted status.

The *ensemble training kit* (per
[ADR-0324](../adr/0324-ensemble-training-kit.md)) is a portable
bundle of the Phase-A corpus producer + LOSO retrain runbook used
to land a second-hardware verdict on the
`fr_regressor_v2_ensemble_v1` retrain. The kit collects the
already-in-tree pieces:

- `scripts/dev/hw_encoder_corpus.py` — Phase-A producer.
- `ai/scripts/run_ensemble_v2_real_corpus_loso.sh` — LOSO wrapper.
- `ai/scripts/train_fr_regressor_v2_ensemble_loso.py` — trainer.
- `ai/scripts/validate_ensemble_seeds.py` — gate verdict emitter.
- `ai/scripts/export_ensemble_v2_seeds.py` — per-seed ONNX exporter.
- `tools/ensemble-training-kit/` — the portable bundle entry-point
  (binaries + tests).
- [`docs/ai/ensemble-v2-real-corpus-retrain-runbook.md`](ensemble-v2-real-corpus-retrain-runbook.md)
  — the operator runbook.

Status: **Proposed** in ADR-0324; the on-disk bundle exists under
`tools/ensemble-training-kit/` but the kit is not yet on the
"ship-as-package" critical path.

## See also

- [`ensemble-v2-real-corpus-retrain-runbook.md`](ensemble-v2-real-corpus-retrain-runbook.md)
  — the operator runbook the kit packages.
- [`docs/ai/models/fr_regressor_v2.md`](models/fr_regressor_v2.md) —
  model card for the regressor the kit retrains.
- [ADR-0324](../adr/0324-ensemble-training-kit.md) — design.
