# Ensemble Training Kit

> **Status**: Proposed packaging surface. The component scripts exist in
> tree; the portable `tools/ensemble-training-kit/` bundle is not yet a
> required release artefact.

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

## What The Kit Packages

The kit is a convenience wrapper around the real training flow. It does
not define a new model format and does not replace the LOSO trainer.

| Step | Existing entry point | Output |
| --- | --- | --- |
| Capture hardware corpus | `scripts/dev/hw_encoder_corpus.py` | Phase-A JSONL rows |
| Train LOSO folds | `ai/scripts/run_ensemble_v2_real_corpus_loso.sh` | fold metrics + checkpoints |
| Validate seeds | `ai/scripts/validate_ensemble_seeds.py` | `PROMOTE.json` verdict |
| Export seed ONNXs | `ai/scripts/export_ensemble_v2_seeds.py` | `fr_regressor_v2_ensemble_v1_seed*.onnx` |

## Current Use

Use the runbook directly when retraining today:

```bash
bash ai/scripts/run_ensemble_v2_real_corpus_loso.sh
python ai/scripts/validate_ensemble_seeds.py runs/ensemble_v2_real/
python ai/scripts/export_ensemble_v2_seeds.py --help
```

The kit becomes release-critical only when ADR-0324 is accepted and the
portable bundle is promoted from convenience packaging to a supported
operator surface.

## See also

- [`ensemble-v2-real-corpus-retrain-runbook.md`](ensemble-v2-real-corpus-retrain-runbook.md)
  — the operator runbook the kit packages.
- [`docs/ai/models/fr_regressor_v2.md`](models/fr_regressor_v2.md) —
  model card for the regressor the kit retrains.
- [ADR-0324](../adr/0324-ensemble-training-kit.md) — design.
