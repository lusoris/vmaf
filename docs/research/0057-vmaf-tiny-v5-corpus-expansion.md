# Research-0057 — vmaf_tiny_v5 corpus expansion (5-corpus = 4-corpus + UGC)

- **Status**: Active
- **Companion ADR**: [ADR-0270](../adr/0270-vmaf-tiny-v5-corpus-expansion.md)
- **Date**: 2026-05-03

## Question

Does adding YouTube-UGC vp9 (orig, dis) pairs to the existing
4-corpus parquet (Netflix Public + KoNViD-1k + BVI-DVC A+B+C+D)
buy measurable headroom over `vmaf_tiny_v2`'s shipped Netflix-LOSO
PLCC = 0.9978 ± 0.0021 baseline, holding architecture and
hyperparameters constant?

The arch ladder (v2 → v3 → v4) saturated at v4 (mlp_large, ADR-0242)
without unlocking further headroom. This digest tests the
orthogonal hypothesis — *more data*, same architecture — for whether
a less-curated, distribution-shifted corpus (UGC web video, encoded
to three VP9 ladder rungs) helps the regressor generalise.

## Corpus access audit

| Candidate | URL | Access | Outcome |
|---|---|---|---|
| **YouTube UGC** | `gs://ugc-dataset/` (also `media.withyoutube.com`) | Direct GCS bucket, CC-BY per ATTRIBUTION | **Used** — 30 smallest 4-tuples ingested |
| LIVE-VQC | `live.ece.utexas.edu/research/LIVE_VQC/index.html` | UT-Austin landing page | **Skipped** — URL returns 404 (corpus unreachable 2026-05-03) |
| MCL-V | `mcl.usc.edu/mcl-v-database/` | Google Drive ZIP behind a copyright-acceptance UI | **Skipped** — programmatic confirm-token flow not implementable without baking in fake credentials, per parent task instruction |
| TID2013 | tampere.fi | Email-form download | **Skipped** — image-level only, not useful for video VMAF |

## Methodology

* **New rows source**: `gs://ugc-dataset/vp9_compressed_videos/` —
  the 30 smallest stems (each is an `(orig.mp4, cbr.webm, vod.webm,
  vodlb.webm)` 4-tuple). Picked smallest by total compressed bytes
  to keep ingest wall-time and disk reasonable for a probe run.
  Total compressed download = 1.25 GB.
* **Decoding**: each stem decoded to a common geometry (640×360,
  yuv420p, 8-bit) via ffmpeg with `scale=W:H:flags=bicubic`,
  capped at 300 frames per clip (~10 s @ 30 fps). The 360p height
  cap keeps the YUV intermediate at ~330 KB/frame and the VMAF
  compute fast; the trade-off is documented in the ADR's
  Alternatives table.
* **Pair generation**: each 4-tuple yields three (orig, dis) pairs
  (`orig→cbr`, `orig→vod`, `orig→vodlb`), so 30 stems × 3 dis
  variants = 90 ref/dis pairs.
* **Feature extraction**: `build-cpu/tools/vmaf` with the canonical-6
  feature flags (`adm`, `vif`, `motion`) plus the bundled
  `vmaf_v0.6.1` predictor as the per-frame teacher. Output one row
  per (pair, frame) with the same column schema as the existing
  4-corpus parquet (canonical-6 columns + `vmaf` populated; the
  remaining 16 columns are NaN — v5 only consumes the canonical-6
  inputs anyway).
* **Combined corpus**: 4-corpus (330 499 rows) + UGC (27 000 rows)
  = 357 499 rows in the v5 training parquet.
* **Training**: identical recipe to v2 — `mlp_small` (6 → 16 → 8 →
  1, 257 params), Adam @ lr=1e-3, MSE, 90 epochs, batch_size 256,
  seed=0. Corpus-wide StandardScaler fit on training rows; baked
  into the exported ONNX as Constant nodes (ADR-0216 trust-root).
* **LOSO eval**: 9-fold leave-one-source-out on the Netflix subset
  only, for both v2-baseline (4-corpus, hold out 1 NF source) and
  v5-candidate (5-corpus, hold out 1 NF source) — same axes so the
  delta is attributable to the corpus expansion.
* **Decision rule**: ship v5 iff `mean_v5_PLCC - mean_v2_PLCC ≥ σ_v2`
  (i.e. ≥ 1 v2 LOSO standard deviation). Otherwise file as a
  research finding and do not ship.

## Findings

### Corpus-distribution skew

UGC clip-level VMAF (90 pairs, 27 000 frames) clusters at the high
end of the scale:

| Stat | UGC | 4-corpus base |
| --- | ---:| ---:|
| min | 8.43 | 0.77 |
| 25 % | 89.04 | 67.97 |
| **median** | **94.45** | **73.60** |
| 75 % | 96.52 | 78.58 |
| max | 100.00 | 100.00 |
| mean | 91.45 | 72.76 |

UGC's vp9 cbr/vod/vodlb encodes are typically high-VMAF — they're
production-quality YouTube ladder rungs, not the broad
codec-degradation sweep that the BVI-DVC and Netflix encodes
provide. Adding 27 000 high-VMAF rows shifts the training-set
class distribution toward the high end, which is a known risk for
regressor calibration in the 60–80 VMAF region.

### LOSO PLCC delta (v5 vs v2 baseline)

Same-axes comparison: 9-fold leave-one-Netflix-source-out, single
seed=0, mlp_small architecture for both, the only delta is the
training parquet (4-corpus → 5-corpus). Both runs share the
identical Netflix held-out clips for evaluation.

| Metric | v2 (4-corpus) | v5 (5-corpus) | Δ |
| --- | ---:| ---:| ---:|
| mean PLCC | 0.999874 | **0.999928** | +0.000054 |
| std PLCC  | 0.000130 | **0.000055** | -58 % |
| mean SROCC | 0.998961 | 0.998844 | -0.000117 |
| std SROCC  | 0.001387 | 0.001669 | +20 % |
| mean RMSE | 0.4185 | **0.3217** | -23 % |
| std RMSE  | 0.1948 | 0.1361 | -30 % |

### Decision: defer (do not ship v5 ONNX)

The mean PLCC delta is +0.000054 — *below* the v2 1-σ envelope
(0.000130). The parent task spec defined the ship gate as
"PLCC ≥ shipped v2 by ≥ 1σ". v5 narrowly misses by ~0.4σ.

The variance-shrink (PLCC σ -58 %, RMSE -23 % mean / -30 % std) is
a real signal — v5 is a *more consistent* estimator across the 9
LOSO folds — but the parent-task ship rule keys on the mean PLCC
axis, not on variance, so the decision is to ship the experiment
as a docs-only research finding and **not** register a `vmaf_tiny_v5`
ONNX in `model/tiny/`.

The ingest + extraction + training + LOSO scripts are still useful
for follow-up work and ship as fork-local research infrastructure;
the 5-corpus parquet (`runs/full_features_ugc.parquet`) is
gitignored under the existing `runs/` rule.

### Per-fold detail

v2 baseline:

| fold | n | PLCC | SROCC | RMSE |
| --- | ---:| ---:| ---:| ---:|
| BigBuckBunny | 1500 | 0.99979 | 0.99926 | 0.532 |
| BirdsInCage  | 1440 | 0.99998 | 0.99996 | 0.269 |
| CrowdRun     | 1050 | 0.99998 | 0.99995 | 0.219 |
| ElFuente1    | 1260 | 0.99997 | 0.99884 | 0.221 |
| ElFuente2    | 1620 | 0.99988 | 0.99992 | 0.391 |
| FoxBird      | 900  | 0.99959 | 0.99676 | 0.758 |
| OldTownCross | 1050 | 0.99996 | 0.99997 | 0.502 |
| Seeking      | 1500 | 0.99982 | 0.99647 | 0.618 |
| Tennis       | 720  | 0.99990 | 0.99951 | 0.255 |

v5 candidate (5-corpus):

| fold | n | PLCC | SROCC | RMSE |
| --- | ---:| ---:| ---:| ---:|
| BigBuckBunny | 1500 | 0.99995 | 0.99913 | 0.441 |
| BirdsInCage  | 1440 | 0.99999 | 0.99999 | 0.150 |
| CrowdRun     | 1050 | 1.00000 | 1.00000 | 0.243 |
| ElFuente1    | 1260 | 0.99996 | 0.99891 | 0.189 |
| ElFuente2    | 1620 | 0.99979 | 0.99986 | 0.464 |
| FoxBird      | 900  | 0.99989 | 0.99563 | 0.402 |
| OldTownCross | 1050 | 0.99988 | 0.99996 | 0.388 |
| Seeking      | 1500 | 0.99988 | 0.99644 | 0.466 |
| Tennis       | 720  | 0.99996 | 0.99974 | 0.152 |

Total LOSO wall: 1534 s on a 32-core CPU (no GPU). 9 v2-baseline
folds + 9 v5-candidate folds = 18 trainings of mlp_small @ 90 ep.

## Reproducer

```bash
# Fetch the 30 smallest 4-tuples from the UGC bucket (~1.25 GB)
python3 ai/scripts/fetch_youtube_ugc_subset.py \
    --out-dir .workingdir2/ugc/download \
    --n-stems 30 \
    --manifest .workingdir2/ugc/manifest.json

# Decode + extract features (90 pairs, 27 000 rows, ~40 s wall on
# an 8-core CPU)
python3 ai/scripts/extract_ugc_features.py \
    --manifest .workingdir2/ugc/manifest.json \
    --yuv-dir .workingdir2/ugc/yuv \
    --vmaf-bin build-cpu/tools/vmaf \
    --out-parquet runs/full_features_ugc.parquet \
    --max-height 360 \
    --max-frames 300 \
    --threads 8

# 9-fold Netflix LOSO (trains 18 models — 9 v2-baseline + 9 v5-
# candidate); writes the JSON report.
python3 ai/scripts/eval_loso_vmaf_tiny_v5.py \
    --parquet-base  runs/full_features_4corpus.parquet \
    --parquet-extra runs/full_features_ugc.parquet \
    --out-json      runs/vmaf_tiny_v5_loso_metrics.json
```

## Threats to validity

1. **Corpus skew** — UGC's high-VMAF cluster may dilute the
   regressor's discrimination in the perceptually-interesting
   60–80 region. Multi-seed re-runs with stratified-VMAF sampling
   (e.g. balance UGC contribution by VMAF decile) is the obvious
   follow-up if v5 underperforms.
2. **Frame-cap artefact** — capping every UGC pair at 300 frames
   (~10 s) may bias toward early-clip statistics; the 4-corpus
   sources span 700–1620 frames per clip. Consider equalising
   per-pair row counts in a follow-up.
3. **Resolution downsample** — 720p / 1080p UGC sources decoded
   to 640×360 yuv420p; this is below the 4-corpus 1920×1080 base
   geometry. The geometry shift is captured in the canonical-6
   features (which are scale-aware via VIF / ADM scales) but is a
   known confounder. Re-running at 720p decode would double the
   YUV intermediate to ~24 GB and the VMAF compute proportionally;
   acceptable as a follow-up if the 360p probe shows positive
   signal.
4. **Single-seed training** — v5 used seed=0, like v2's ship
   recipe; a multi-seed sweep (5 seeds × 9 folds = 45 trainings)
   would tighten the PLCC error bar to compare against the
   published 0.9978 ± 0.0021 v2 figure on the same multi-seed
   axis. Deferred.

## References

- v2 baseline ADR: [ADR-0216](../adr/0216-vmaf-tiny-v2.md)
- v3 arch ladder: [ADR-0241](../adr/0241-vmaf-tiny-v3-mlp-medium.md)
- v4 arch ladder: ADR-0242 (mlp_large)
- This digest's ADR: [ADR-0270](../adr/0270-vmaf-tiny-v5-corpus-expansion.md)
- YouTube UGC dataset homepage: <https://media.withyoutube.com/>
- YouTube UGC GCS bucket: `gs://ugc-dataset/`
- UGC paper: Wang et al., "YouTube UGC Dataset for Video
  Compression Research" (CoINVQ.pdf at the bucket root)
