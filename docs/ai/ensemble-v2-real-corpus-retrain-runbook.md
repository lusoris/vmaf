# `fr_regressor_v2` ensemble — real-corpus retrain runbook

This runbook walks through the **real-corpus LOSO retrain** of the
five `fr_regressor_v2_ensemble_v1_seed{0..4}` members of the deep
ensemble, including how to interpret the verdict and how to roll
back if the registry was flipped prematurely.

- **Parent design**: [ADR-0272](../adr/0272-fr-regressor-v2-codec-aware-scaffold.md)
  (codec-aware scaffold).
- **Gate definition**: [ADR-0303](../adr/0303-fr-regressor-v2-ensemble-prod-flip.md)
  (ensemble production-flip trainer + CI gate — `mean(PLCC) ≥ 0.95`
  AND `max(PLCC) - min(PLCC) ≤ 0.005`).
- **Harness scope**: [ADR-0309](../adr/0309-fr-regressor-v2-ensemble-real-corpus-retrain.md)
  (this runbook's PR — wrapper script + validator + verdict files;
  registry flip is a separate follow-up PR).

---

## Prerequisites

| Requirement      | Expected value                                                                 |
|------------------|--------------------------------------------------------------------------------|
| Corpus location  | `.workingdir2/netflix/` (overridable via `$CORPUS_ROOT`)                       |
| Corpus contents  | ≥ 1 reference YUV + ≥ 1 distorted YUV (full Netflix Public Dataset = 9 + 70)   |
| GPU memory       | ≥ 8 GB VRAM (NVIDIA / SYCL — the trainer is single-GPU per seed)               |
| Free disk        | ≥ 5 GB under `runs/ensemble_v2_real/` (logs + per-seed JSON, no model weights) |
| Wall time        | 6–12 h on a single 8 GB GPU (5 seeds × 9 LOSO folds, sequential)               |
| Python deps      | `ai/pyproject.toml` editable install (`pip install -e ai/`)                    |

The corpus is gitignored — it was provided locally by lawrence on
2026-04-27 (memory-pinned in
`feedback_netflix_training_corpus_local`). Do **not** check it in.

---

## Step-by-step run

### 1. Verify the corpus

```bash
ls -d .workingdir2/netflix/  # should exist
find .workingdir2/netflix/ -name '*.yuv' | wc -l  # should be > 0
```

### 2. Kick off the wrapper

```bash
bash ai/scripts/run_ensemble_v2_real_corpus_loso.sh
```

The wrapper:

1. Validates `$CORPUS_ROOT` exists and contains ≥ 1 ref + ≥ 1 dis YUV.
2. Loops `seed ∈ {0,1,2,3,4}`, calling
   `train_fr_regressor_v2_ensemble_loso.py --seed N
   --corpus-root $CORPUS_ROOT
   --output runs/ensemble_v2_real/loso_seed{N}.json` per seed.
3. Tees a timestamped log per seed under
   `runs/ensemble_v2_real/logs/seed{N}_<UTC>.log`.
4. Prints a one-line summary on completion (elapsed seconds + next
   command to run).

The wrapper is **not** a daemon — it blocks until all 5 seeds
finish. Use `tmux` / `nohup` if you want to detach.

### 3. Apply the production-flip gate

```bash
python ai/scripts/validate_ensemble_seeds.py runs/ensemble_v2_real/
```

The validator:

- Loads `loso_seed{0..4}.json`.
- Calls the gate from `scripts/ci/ensemble_prod_gate.py`
  (single source of truth for the threshold constants).
- Snapshots the corpus YUV file list (sha256 over sorted `relpath\tsize`,
  not YUV bytes — that would dominate validate runtime on a 37 GB corpus).
- Writes `runs/ensemble_v2_real/PROMOTE.json` *or* `HOLD.json`.
- Exits 0 on PROMOTE, 1 on HOLD, 2 on input error.

---

## Interpreting the verdict

### `PROMOTE.json` — gate cleared

```jsonc
{
  "verdict": "PROMOTE",
  "gate": {
    "passed": true,
    "mean_plcc": 0.96,
    "plcc_spread": 0.003,
    "per_seed_plccs": {"0": 0.961, "1": 0.962, ...}
  },
  "corpus": {"sha256": "...", "yuv_count": 79, ...},
  "recommendation": "flip seeds smoke->false in model/tiny/registry.json ..."
}
```

Action: open a **separate follow-up PR** that flips the five
`fr_regressor_v2_ensemble_v1_seed{0..4}` rows in
`model/tiny/registry.json` from `smoke: true` to `smoke: false`.
Cite the `PROMOTE.json` corpus sha256 in the PR description so the
flip is traceable to the exact retrain that earned it.

### `HOLD.json` — gate failed

```jsonc
{
  "verdict": "HOLD",
  "gate": {
    "passed": false,
    "mean_plcc": 0.951,
    "plcc_spread": 0.012,
    "plcc_spread_pass": false,
    "failing_seeds": [3]
  },
  "recommendation": "keep smoke=true; investigate diversity / hyperparameters. Failing aspects: plcc_spread 0.0120 > 0.0050"
}
```

Action: do **not** flip the registry. Common remediations:

- **Spread too wide (`max - min > 0.005`)**: the seeds disagree on
  which Netflix sources they generalise to. Try
  re-seeding (cheap), increase epochs, or widen ensemble diversity
  via different weight-init scales — see ADR-0303 §Decision and
  Research-0081 §Seed-diversity hyperparameters.
- **Mean PLCC too low**: corpus / loader regression. Re-derive the
  Phase A canonical-6 features and confirm against ADR-0291's
  deterministic v2 baseline first.
- **One seed below 0.95 with healthy spread**: drop / replace that
  seed before re-running. The flip is per-seed (a seed flips only
  after it individually clears the gate).

---

## Rollback procedure

If the registry was flipped prematurely (i.e. before a passing
`PROMOTE.json` existed, or against an outdated corpus snapshot):

1. **Revert the flip commit** — `git revert <flip-sha>` from `master`
   via a fresh PR. Direct push to `master` is host-blocked
   (CLAUDE §12 r2 + ADR-0037).
2. **Verify the registry** — `python ai/scripts/validate_model_registry.py`
   should pass; the five rows must read `"smoke": true` again.
3. **Verify the C-side ORT loader** — re-run
   `python ai/tests/test_registry.py` to confirm the smoke graphs
   still load (they always have, since the ONNX bytes never change
   on a smoke→smoke revert).
4. **Audit the consumers** — anything that read the now-reverted
   non-smoke ensemble (e.g. `vmaf-tune --quality-confidence`)
   silently degrades to ADR-0291's deterministic point estimate
   when `smoke: true` is back in force; surface that in the revert
   PR description.
5. **Open a HOLD-tracking issue** — link the original `PROMOTE.json`
   that was used for the bad flip (if any) and the new `HOLD.json`
   that justifies the revert. The issue stays open until a clean
   real-corpus retrain produces a fresh `PROMOTE.json`.

---

## See also

- [ADR-0303](../adr/0303-fr-regressor-v2-ensemble-prod-flip.md) —
  gate definition (mean ≥ 0.95 AND spread ≤ 0.005).
- [ADR-0272](../adr/0272-fr-regressor-v2-codec-aware-scaffold.md) —
  parent design for the codec-aware FR regressor v2.
- [Research-0081](../research/0081-fr-regressor-v2-ensemble-real-corpus-methodology.md)
  — corpus-size sufficiency, LOSO fold sizing, seed-diversity
  hyperparameters.
- [`scripts/ci/ensemble_prod_gate.py`](../../scripts/ci/ensemble_prod_gate.py) —
  single source of truth for the threshold constants.
- [`ai/scripts/run_ensemble_v2_real_corpus_loso.sh`](../../ai/scripts/run_ensemble_v2_real_corpus_loso.sh) —
  the wrapper this runbook drives.
- [`ai/scripts/validate_ensemble_seeds.py`](../../ai/scripts/validate_ensemble_seeds.py) —
  the validator that emits `PROMOTE.json` / `HOLD.json`.
