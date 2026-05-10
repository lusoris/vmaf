# FR regressor v2 — codec-aware (planned)

`vmaf_tiny_fr_regressor_v2_codec_aware` — a codec-conditioned successor
to the v1 FR MOS regressor (`fr_regressor_v1.onnx`). Maps a libvmaf
`FULL_FEATURES` vector **plus a one-hot codec id** to a single MOS
scalar, lifting cross-codec PLCC/SROCC by 1–3 points per the
literature cited in
[ADR-0235](../../adr/0235-codec-aware-fr-regressor.md) and
[Research-0040](../../research/0040-codec-aware-fr-conditioning.md).

> **Status: training BLOCKED.** This PR ships the model surface +
> training-side plumbing only. The cached Netflix Public training
> corpus at `~/.cache/vmaf-tiny-ai/` is not reachable from the
> sandbox that authored the change, so the empirical PLCC/SROCC
> delta is not yet measured. A follow-up PR (backlog item
> **T7-CODEC-AWARE**) re-runs the trainer + ships the ONNX +
> registers it. Until then, no `model/tiny/fr_regressor_v2_codec_aware.onnx`
> is committed and no `registry.json` row exists.

## What the output means

One scalar `score` per (frame, codec) pair, calibrated to the same
MOS range as `fr_regressor_v1` (typically `[0, 100]`, with the
training corpus's per-clip MOS pooling determining the floor / ceiling
empirically).

## Codec vocabulary

The model concatenates a one-hot codec id to its `FULL_FEATURES`
input. The vocabulary is closed and order-stable — `ai/src/vmaf_train/codec.py`:

| Index | Codec    | Distortion signature                                       |
|-------|----------|------------------------------------------------------------|
| 0     | `x264`   | Block edges, deblocking-filter aliasing                    |
| 1     | `x265`   | CTU-boundary blur, SAO ringing                             |
| 2     | `libsvtav1` | DCT ringing, CDEF + loop-restoration smoothing          |
| 3     | `libvvenc`  | Large-CTU deblocking, ALF banding                       |
| 4     | `libvpx-vp9`| Block-transform + loop-filter signature                 |
| 5     | `unknown`   | Fallback for corpora without codec metadata             |

Aliases at parquet-ingest time: `h264` → `x264`, `hevc` → `x265`,
`av1` → `libsvtav1`, `vp9` → `libvpx-vp9`, `vvc` / `h266` →
`libvvenc`. Anything else collapses to `unknown`.

The vocabulary is versioned via `CODEC_VOCAB_VERSION = 1`. Adding a
codec is a schema bump that requires retraining; the model-card
sidecar pins the vocabulary version against which the ONNX was
trained.

## ONNX I/O contract (planned)

Two-input session, matching the LPIPS-Sq precedent
([ADR-0040](../../adr/0040-dnn-session-multi-input-api.md),
[ADR-0041](../../adr/0041-lpips-sq-extractor.md)):

```
features:    float32 [N, 22]   # FULL_FEATURES vector
codec_onehot: float32 [N, 6]    # one-hot codec id

score:       float32 [N]        # MOS scalar
```

Variance-mode export emits `score_logvar: float32 [N, 2]`
(matching the v1 `emit_variance` precedent —
[ADR-0207](../../adr/0207-tinyai-qat-design.md) + the `confidence.py`
gaussian NLL path).

## Backwards compatibility with v1

`FRRegressor(num_codecs=0)` is the v1 contract — single input,
single output, identical to `fr_regressor_v1.onnx`. Existing
checkpoints load unchanged, the v1 ONNX in `model/tiny/` continues
to ship, and inference callers without codec metadata can keep
calling the v1 graph. v2 ships only when the empirical lift
clears the 0.005 PLCC bar (per ADR-0235's "no negative releases"
rule).

## Training recipe (planned)

Same as v1 (`ai/src/vmaf_train/train.py` driver, `FRRegressor`
hyperparameters per the C1 baseline) with three deltas:

1. Constructor: `FRRegressor(in_features=22, num_codecs=6, ...)`.
2. Datamodule: emit 3-tuple `(features, codec_onehot, mos)` batches
   instead of 2-tuple `(features, mos)`. The codec one-hot comes
   from the parquet's `codec` column via `vmaf_train.codec.codec_one_hot_batch`.
3. Export: `export_to_onnx` call uses two input names
   (`features`, `codec_onehot`) with dynamic batch axis on both.

## Re-extracting features with codec metadata

```bash
# Netflix Public corpus — distortions are pre-encoded, no codec metadata
python ai/scripts/extract_full_features.py --codec unknown

# BVI-DVC — script encodes internally with libx264 at CRF 35
python ai/scripts/bvi_dvc_to_full_features.py --tier D --codec x264

# KoNViD-1k — codec inferred per-clip via ffprobe (see ai/scripts/konvid_to_full_features.py)
# After the konvid pipeline lands, the per-clip codec is the ffprobe stream=codec_name
# value aliased through ai/src/vmaf_train/codec.py.
```

## Known limitations (carried into v2)

- **22-feature `FULL_FEATURES` set only.** Re-running with a
  different feature pool requires a fresh export.
- **Codec one-hot, not embedding.** New codecs can't be added at
  inference time without retraining (see ADR-0235's alternatives
  table for why we chose one-hot over `nn.Embedding`).
- **No CUDA/SYCL/Vulkan EP wiring.** The v1 ONNX runs on the ORT
  CPU EP today; v2 inherits that limitation.
- **Single-frame regression.** No temporal pooling inside the
  graph; pool with `mean` / `harmonic_mean` at the `VmafContext`
  level.

## See also

- [ADR-0235](../../adr/0235-codec-aware-fr-regressor.md) — design decision.
- [Research-0040](../../research/0040-codec-aware-fr-conditioning.md) — empirical context.
- `ai/src/vmaf_train/codec.py` — vocabulary + one-hot helpers.
- `ai/src/vmaf_train/models/fr_regressor.py` — codec-aware model surface.
- [overview.md](../overview.md) — C1 capability map.
