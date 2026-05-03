# ADR-0284: `fr_regressor_v2` codec one-hot expansion (6 → 16 slots)

- **Status**: Accepted
- **Date**: 2026-05-03
- **Deciders**: lusoris (with Claude)
- **Tags**: ai, fr-regressor, codec, schema, hardware-encoder, fork-local

## Context

The codec-aware FR regressor introduced in
[ADR-0235](0235-codec-aware-fr-regressor.md) shipped its v1
vocabulary (six slots: `x264`, `x265`, `libsvtav1`, `libvvenc`,
`libvpx-vp9`, `unknown`) under the assumption that distortion-signature
lift would come from software codecs alone.

Parallel `vmaf-tune` work is now landing 13+ codec adapters across
software (x264, x265, svtav1, libaom) **and** hardware (NVENC ×3,
QSV ×3, AMF ×3, VideoToolbox ×2 — see
[ADR-0283](0283-vmaf-tune-videotoolbox-adapters.md)). The v1 6-slot
vocabulary cannot fit the new corpus: every hardware-encoded clip
would silently bucket into `unknown`, defeating the codec-conditioning
the regressor is supposed to learn.

## Decision

Expand the codec one-hot to 16 slots and bump
`CODEC_VOCAB_VERSION` from 1 to 2. The v2 ordering is closed and
load-bearing — every `fr_regressor_v2_*.onnx` file's first Linear
layer bakes this column index into its weight tensor:

```
0  x264              5  hevc_nvenc        10 h264_amf
1  x265              6  av1_nvenc         11 hevc_amf
2  libsvtav1         7  h264_qsv          12 av1_amf
3  libaom            8  hevc_qsv          13 h264_videotoolbox
4  h264_nvenc        9  av1_qsv           14 hevc_videotoolbox
                                          15 reserved
```

Index 15 is reserved for the next codec added without forcing a
third schema bump. Unrecognised labels (including `vp9`, `vvc`,
`h266`, and the v1 `libvvenc` / `libvpx-vp9` slots) bucket to
`reserved` via the unknown-fallback path.

The shipped `fr_regressor_v2_hw` model collapses the formerly
two-input session (`features` + `codec_onehot`) into a single
24-D wide-input vector (`6 features + 16 codec one-hot + 1
preset_norm + 1 crf_norm`). The export stays single-input; the
sidecar's `feature_layout` field documents the slot offsets.

`fr_regressor_v1.onnx` and the v1 `CODEC_VOCAB_V1` tuple stay
shipped and unaffected — v1 callers that need the legacy 6-slot
vocabulary continue to import `CODEC_VOCAB_V1`.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| 16 slots, closed-set with trailing `reserved` (chosen) | Fits today's 13 adapters + headroom for one more codec without a v3 schema; deterministic mapping; shipped models stay valid as long as ordering is preserved. | One slot is wasted today. | Minimal-risk path; one slot of headroom is cheap. |
| Variable-length embedding (`nn.Embedding`) | New codecs can be added at inference time without retraining. | Loses the closed-set guarantee; lookup-table size baked into the ONNX graph anyway; ADR-0235 already rejected this. | Same reasoning as ADR-0235. |
| 32 slots for "more headroom" | More future-proof. | 22 wasted columns today; doubles the first-Linear-layer width for no measurable gain; the fork doesn't track 16+ live encoders. | YAGNI — bumping again later is a 5-minute change once we have 16 live codecs. |
| Two-input ONNX session (features + codec_onehot) | Matches the LPIPS-Sq precedent (ADR-0040 / ADR-0041); cleaner separation between feature and codec axes. | Existing v1 `fr_regressor_v1.onnx` is single-input; making the v2 graph two-input forces every consumer to update. | Wide-input concat is an internal-only refactor; the harness already concatenates at corpus-emit time, so this matches the data shape. |
| Drop `libvvenc` / `libvpx-vp9` without a `reserved` slot | Smallest possible vocabulary. | No headroom; the next codec triggers another schema bump and another retrain. | The reserved slot is one wasted column for one PR-amount of saved future work. |

## Consequences

- **Positive**: `fr_regressor_v2_hw` can distinguish software vs.
  every major hardware-encoder family. Adding the 13+ codec
  adapters parallel agents are landing now uses pre-existing
  one-hot columns — no further schema bump needed until the 17th
  codec arrives.
- **Negative**: This is a closed-set rebuild — every shipped
  `fr_regressor_v2_*.onnx` must be retrained against the new
  vocabulary. v1 users upgrading to v2 must remap any
  `vp9` / `libvvenc` clips (they now bucket to `reserved`); a
  follow-up PR can re-add VVC / VP9 if real corpora demand it.
- **Neutral / follow-ups**: the SMOKE `fr_regressor_v2_hw.onnx`
  shipped in this PR is trained on synthetic deterministic data —
  T7-CODEC-AWARE-V2 retrains against a real multi-codec corpus and
  reports empirical PLCC/SROCC. Until then, the registry entry
  carries `smoke: true`.

## References

- [ADR-0235](0235-codec-aware-fr-regressor.md) — v1 codec-aware regressor.
- [ADR-0283](0283-vmaf-tune-videotoolbox-adapters.md) — companion VideoToolbox adapters.
- [ADR-0237](0237-quality-aware-encode-automation.md) — `vmaf-tune` umbrella.
- [Research-0040](../research/0040-codec-aware-fr-conditioning.md) — empirical lift evidence.
- [`docs/ai/models/fr_regressor_v2_codec_aware.md`](../ai/models/fr_regressor_v2_codec_aware.md) — model card.
- Source: `req` (paraphrased) — user requested expanding the 6-slot codec one-hot to 16 slots so all software + hardware codecs fit, alongside Apple VideoToolbox adapters.
