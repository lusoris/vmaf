# FR regressor v2 — codec-aware (vmaf-tune corpus consumer)

`fr_regressor_v2` — codec-conditioned successor to
[`fr_regressor_v1`](fr_regressor_v1.md). Maps a 6-D canonical libvmaf
feature vector plus an 8-D codec block to a VMAF teacher score.
Trained on the JSONL corpus emitted by `vmaf-tune corpus` (Phase A,
[ADR-0237](../../adr/0237-quality-aware-encode-automation.md)).

> **Status: SCAFFOLD.** This PR ships the training script + ONNX export
> plumbing + registry plumbing. The shipped ONNX is generated via
> `--smoke` mode (synthetic corpus, 1 epoch) and is registered with
> `smoke: true` so the quality-metric harness skips it. The production
> training run is a follow-up PR gated on (1) a multi-codec Phase A
> corpus with ≥50 refs / ≥5 encoders, (2) per-frame feature emission
> in the Phase A schema, and (3) clearing v1's 0.95 LOSO PLCC ship
> threshold with a ≥0.005 multi-codec lift per
> [ADR-0235](../../adr/0235-codec-aware-fr-regressor.md). See
> [Research-0054](../../research/0058-fr-regressor-v2-feasibility.md).

## Inputs

Two named tensors, dynamic batch axis:

- **`features`**, shape `(N, 6)` — canonical-6 libvmaf features
  (StandardScaler-normalised at training time using the mean/std
  baked into the sidecar JSON):

  | Index | Feature        |
  |-------|----------------|
  | 0     | `adm2`         |
  | 1     | `vif_scale0`   |
  | 2     | `vif_scale1`   |
  | 3     | `vif_scale2`   |
  | 4     | `vif_scale3`   |
  | 5     | `motion2`      |

- **`codec`**, shape `(N, 8)` — codec block, **not** normalised
  (already in `[0, 1]`):

  | Index | Slot                               |
  |-------|------------------------------------|
  | 0     | `encoder_onehot[libx264]`          |
  | 1     | `encoder_onehot[libx265]`          |
  | 2     | `encoder_onehot[libsvtav1]`        |
  | 3     | `encoder_onehot[libvvenc]`         |
  | 4     | `encoder_onehot[libvpx-vp9]`       |
  | 5     | `encoder_onehot[unknown]`          |
  | 6     | `preset_norm`  (preset ordinal / 9)|
  | 7     | `crf_norm`     (CRF / 63)          |

  Encoder vocabulary is closed and ordered — index 0..5 is
  load-bearing; bumping the vocabulary requires a re-train. The
  `unknown` bucket lets corpora without codec metadata pass an
  all-zeros + `unknown=1` vector and degrade gracefully.

  CRF normalised by **63** — the union upper bound across all
  supported encoders (libsvtav1 / libvpx-vp9 max). x264 / x265 use
  CRF up to 51; values above their per-encoder max are clipped at
  read time.

  Preset ordinal table per encoder lives in
  [`ai/scripts/train_fr_regressor_v2.py`](../../../ai/scripts/train_fr_regressor_v2.py)
  (`PRESET_ORDINAL`); the canonical 0..9 scale carries the
  speed-quality direction consistently across encoders. libsvtav1's
  numeric 0..13 presets are squashed to 0..9.

## Output

`score`, shape `(N,)` — a scalar VMAF-aligned quality score per
sample, same MOS range as v1 (typically `[0, 100]`).

## Codec-blind fallback

For inference paths that don't carry codec metadata, pass an
all-zeros codec vector with `encoder_onehot[unknown]=1` and
`preset_norm=0.5`, `crf_norm=0.5`. The model degrades to a v1-like
estimate; no graph surgery required.

## Training corpus

vmaf-tune Phase A JSONL (`tools/vmaf-tune/src/vmaftune/corpus.py`).
One row per `(source, encoder, preset, crf)` cell with
`schema_version=1`. The trainer reads the JSONL row-by-row; the
canonical-6 features come from the optional
`per_frame_features` payload (a Phase A follow-up; the current
schema does not emit them — the smoke path synthesises them
deterministically).

## CLI

```bash
# Smoke (synthetic corpus, validates the pipeline)
python ai/scripts/train_fr_regressor_v2.py --smoke

# Production (real Phase A corpus)
python ai/scripts/train_fr_regressor_v2.py \
    --corpus runs/vmaf_tune_corpus.jsonl \
    --epochs 30
```

The script bakes the StandardScaler over the canonical-6 dims into
the sidecar JSON (`feature_mean` / `feature_std`); the codec block
is unscaled. Output ONNX is opset 17, dynamic batch axis, op-allowlist
checked.

## See also

- [ADR-0272](../../adr/0272-fr-regressor-v2-codec-aware-scaffold.md)
  — this scaffold's decision record.
- [ADR-0235](../../adr/0235-codec-aware-fr-regressor.md) — the
  parent codec-aware decision.
- [ADR-0237](../../adr/0237-quality-aware-encode-automation.md) —
  vmaf-tune Phase A (the corpus producer).
- [ADR-0249](../../adr/0249-fr-regressor-v1.md) —
  `fr_regressor_v1` baseline.
- [Research-0054](../../research/0058-fr-regressor-v2-feasibility.md)
  — feasibility digest, including the open question on production
  corpus diversity.
- [`fr_regressor_v2_codec_aware.md`](fr_regressor_v2_codec_aware.md)
  — the older ADR-0235 model card (canonical-9 features, training
  blocked on cache reachability). The two cards describe different
  takes on the codec-aware idea — `fr_regressor_v2` (this card) is
  the vmaf-tune Phase B prerequisite that ships the corpus consumer.
