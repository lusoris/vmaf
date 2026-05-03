# ADR-0270: vmaf_tiny_v5 — corpus expansion (4-corpus + YouTube UGC vp9 subset) — DEFERRED

- **Status**: Deferred — corpus expansion did not clear the 1-σ ship gate
- **Date**: 2026-05-03
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: ai, dnn, tiny-ai, model, registry, training-data, fork-local

## Context

`vmaf_tiny_v2` (ADR-0216) shipped a Phase-3-validated mlp_small (6 → 16 → 8 → 1, ~257 params) trained on a 4-corpus parquet — Netflix Public + KoNViD-1k + BVI-DVC A+B+C+D, 330 499 frame-rows, with `vmaf_v0.6.1` as the per-frame teacher. Subsequent ADRs ladder the architecture upward (v3 = mlp_medium per ADR-0241; v4 = mlp_large per ADR-0242) but the PLCC ladder saturates at v4. The user requested a fifth attempt that holds the v2 architecture constant and instead expands the *training corpus* — testing whether more frames from a less-curated distribution buy headroom on the published Netflix LOSO PLCC=0.9978 ± 0.0021 baseline.

The session audited four candidate datasets: LIVE-VQC (UT-Austin landing page returns 404 — corpus URL is dead), MCL-V (Google-Drive-only download with no programmatic confirm-token path), TID2013 (image-level only, not useful for video VMAF), and YouTube UGC ([gs://ugc-dataset/](https://storage.googleapis.com/ugc-dataset/) — direct GCS bucket, CC-BY license per the bucket-root ATTRIBUTION file). Only YouTube UGC was reachable without a clickwrap; we ingested the 30 smallest 4-tuples from `vp9_compressed_videos/` (90 (orig, dis) pairs across cbr/vod/vodlb encodes) decoded to 640×360 yuv420p with 300-frame caps, yielding 27 000 additional rows for a 5-corpus parquet of 357 499 rows.

## Decision

**Do not ship `vmaf_tiny_v5.onnx`.** The 9-fold Netflix LOSO sweep on the 5-corpus parquet (357 499 rows) yielded mean PLCC = 0.999928 ± 0.000055, vs the same-axes v2-baseline (4-corpus, 330 499 rows) of mean PLCC = 0.999874 ± 0.000130. The +0.000054 delta is *below* the v2-baseline 1-σ envelope (0.000130) — the corpus-expansion hypothesis fails the parent-task ship gate. Ship the ingest + extraction + evaluation pipeline (`fetch_youtube_ugc_subset.py`, `extract_ugc_features.py`, `train_vmaf_tiny_v5.py`, `eval_loso_vmaf_tiny_v5.py`, `export_vmaf_tiny_v5.py`) as fork-local research infrastructure so the experiment is reproducible, but **do not register a v5 model** in `model/tiny/registry.json` and do not flip the production default. `vmaf_tiny_v2` remains the canonical small variant; v3 / v4 stay as the higher-PLCC ladder rungs. The 5-corpus parquet (`runs/full_features_ugc.parquet`) is gitignored.

Subsidiary finding (worth recording but not ship-relevant): v5 has 58 % lower LOSO PLCC variance (σ 0.000055 vs 0.000130) and 23 % lower mean RMSE (0.322 vs 0.418) — i.e. the corpus expansion *did* tighten estimator consistency, just not enough on the headline mean-PLCC axis to justify another model in the registry. If a future PR re-runs with multi-seed averaging or stratified-VMAF UGC sampling, the variance-shrink finding gives a plausible direction.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Ship v5 only when ≥1σ over v2** (chosen if PLCC gate met) | Same statistical bar as v3/v4 ship gates; ensures the corpus-expansion delta is not a seed-noise artefact | Requires a docs-only path when the corpus turns out to be neutral / harmful | Right trade-off — the user's spec encoded this rule directly |
| Ship v5 unconditionally | Captures any small positive delta | Risks shipping a regressed estimator if UGC's narrow VMAF distribution skews the regressor | Rejected — would also leak corpus-distribution drift into a production-default-adjacent model |
| Add LIVE-VQC + MCL-V before deciding | Larger, more diverse expansion | LIVE-VQC URL is dead (404 on `live.ece.utexas.edu/research/LIVE_VQC/index.html` 2026-05-03); MCL-V is Google Drive only with no programmatic confirm-token flow | Rejected — explicit user instruction: "If any require email-form clickwrap with no programmatic access, document the blocker and skip that one. Don't bake in fake credentials." |
| Re-train v5 with the v3 architecture (mlp_medium) | Stacks the corpus-expansion hypothesis on top of the arch-ladder hypothesis | Confounds two changes; can't attribute any delta cleanly | Rejected — v5 deliberately holds arch constant to test corpus expansion in isolation |
| Use the full UGC bucket (~1500 stems) | Maximises corpus diversity | Compressed source ~150-300 MB per stem × 1500 = 200+ GB downloads; YUV intermediate ~1-2 TB; days of VMAF compute for a hypothesis that may not pan out | Rejected — start with a 30-stem probe; expand only if the probe shows positive signal |
| Train on UGC alone (drop 4-corpus) | Tests whether UGC by itself is a viable training corpus | UGC pairs cluster at high VMAF (median 94.5 vs 4-corpus median 73.6); the regressor would lose all coverage of the low-quality regime | Rejected — UGC is too narrow on its own; only useful as a complement |

## Consequences

- **Positive (if PLCC gate met)**: ships a corpus-expanded mlp_small variant with the same ONNX size, opset, and trust-root as v2 — opt-in users get the corpus-diversity benefit for zero deployment cost. The ingestion + extraction pipeline (`fetch_youtube_ugc_subset.py`, `extract_ugc_features.py`) is reusable for future BVI / Bristol / VQEG corpus additions.
- **Negative (if shipped)**: registry surface grows to four tiny FR fusion models (v1, v2, v3, v4, v5) — each one cheap individually but the cumulative cognitive cost is non-trivial; future Phase-4 work should prune.
- **Negative (if deferred)**: the corpus-ingest scripts ship anyway as fork-local research infrastructure; `runs/full_features_ugc.parquet` lives under gitignored `runs/`; no model artefact lands in `model/tiny/`.
- **Neutral / follow-ups**: if PLCC delta is borderline (positive but <1σ), backlog a multi-seed sweep (5 seeds × 9 folds = 45 trainings) before declaring final ship-or-defer; if PLCC clearly regressed, file a Research follow-up to investigate the UGC distribution skew (median 94.5 vs 4-corpus median 73.6) as a corpus-balancing problem rather than abandoning UGC outright.

## References

- Source: `req` (parent-task spec — paraphrased: "Expand the tiny-AI training corpus beyond today's 4-corpus set and retrain `vmaf_tiny_v2` on the expanded corpus. Compare PLCC vs the shipped checkpoint. If improved, ship as `vmaf_tiny_v5` (don't overwrite v2).")
- v2 baseline: [ADR-0216](0216-vmaf-tiny-v2.md)
- v3 arch ladder: [ADR-0241](0241-vmaf-tiny-v3-mlp-medium.md)
- v4 arch ladder: ADR-0242 (mlp_large)
- Research digest: [Research-0057](../research/0057-vmaf-tiny-v5-corpus-expansion.md)
- Trainer: [`ai/scripts/train_vmaf_tiny_v5.py`](../../ai/scripts/train_vmaf_tiny_v5.py)
- UGC fetcher: [`ai/scripts/fetch_youtube_ugc_subset.py`](../../ai/scripts/fetch_youtube_ugc_subset.py)
- UGC feature extractor: [`ai/scripts/extract_ugc_features.py`](../../ai/scripts/extract_ugc_features.py)
- LOSO eval: [`ai/scripts/eval_loso_vmaf_tiny_v5.py`](../../ai/scripts/eval_loso_vmaf_tiny_v5.py)
- LOSO results: `runs/vmaf_tiny_v5_loso_metrics.json`
- YouTube UGC dataset: <https://media.withyoutube.com/> + GCS bucket `gs://ugc-dataset/` (CC-BY per bucket-root ATTRIBUTION file)
- UGC paper: Wang et al., "YouTube UGC Dataset for Video Compression Research" (CoINVQ.pdf at the bucket root)
