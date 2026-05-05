# ADR-0285: `vmaf-tune` libvvenc adapter — VVC / H.266 with optional NN-VC tools

- **Status**: Accepted
- **Date**: 2026-05-03
- **Deciders**: Lusoris
- **Tags**: tooling, codec, vvc, h266, ai, nnvc, fork-local

## Context

[ADR-0237](0237-quality-aware-encode-automation.md) Phase A wired the
first codec adapter (`libx264`) into `tools/vmaf-tune/`. The harness's
codec-adapter contract is multi-codec from day one — every additional
encoder is a one-file drop under `tools/vmaf-tune/src/vmaftune/codec_adapters/`
plus a registry entry, with no branching in the search loop.

VVenC (Fraunhofer HHI's open-source VVC encoder, BSD-3-Clause-Clear)
is the first codec on the adapter set whose rate-distortion curve
materially differs from x264 / x265 / SVT-AV1 — VVC delivers ~30-50%
better compression than HEVC at the cost of ~3-5× the encode time.
Wiring it in early makes Phase B/C predictors (target-VMAF bisect,
per-title CRF predictor) see a non-trivial codec spread before the
schema is locked.

VVenC is also the first standardised codec on the adapter set with
first-class **neural-network video coding (NNVC)** tools exposed via
the encoder CLI:

- NN-based intra prediction (learned 5×5 / 7×7 / 9×9 conv replaces
  handcrafted intra modes; ~1-3% bitrate gain at iso-VMAF, ~5-10×
  slower intra encode).
- NN-based loop filter (learned post-processing CNN augments / replaces
  the deblocking + SAO + ALF cascade).
- NN-based super-resolution (encode low-res, decode + learned upsample).

This is the closest thing the open-source video stack has to a
"neural-augmented codec" today, and it is the natural counterpart to
the fork's existing tiny-AI *measurement* surface (`vmaf_tiny_v2`,
`fr_regressor_v1`, `nr_metric_v1`). Putting the NNVC tools behind the
same `vmaf-tune` harness lets future predictors learn when the NNVC
tools are worth their compute cost.

The adapter surfaces only the deterministic single-pass QP path in
this PR. Two-pass / per-shot dynamic QP / NN-loop-filter and
NN-super-resolution toggles are deferred to follow-up ADRs.

## Decision

We will ship `tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py` as
the eighth-codec — index TBD against the parallel adapter PRs landing
2026-05-03 — codec adapter on the harness's registry. Surface:

- `quality_knob = "qp"`, `quality_range = (17, 50)`,
  `quality_default = 32`, `invert_quality = True`.
- The harness's canonical 7-name preset vocabulary
  (`placebo / slowest / slower / slow / medium / fast /
  faster / veryfast / superfast / ultrafast`) compresses
  onto VVenC's native 5-level vocabulary (`faster / fast /
  medium / slow / slower`) via a static map. The compression rule:
  anything strictly slower than `slow` pins to `slower`; anything
  strictly faster than `fast` pins to `faster`; the central three
  names map identically. This matches the rule used by the parallel
  HEVC / AV1 adapter PRs landing today.
- One Phase A NNVC toggle: `nnvc_intra: bool = False`. Off by default
  so grids stay deterministic; flipping the toggle emits
  `-vvenc-params IntraNN=1` into the FFmpeg argv and is recorded in
  the corpus row's `extra_params` for downstream predictor
  conditioning.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Direct `vvencapp` CLI driver (no FFmpeg) | Avoids the FFmpeg patch dependency; matches Fraunhofer's reference invocation. | Splits the harness's subprocess seam — every other adapter routes through FFmpeg; corpus rows would need a separate `vvenc_version` parser. | Rejected: harness uniformity beats one-codec convenience. FFmpeg ≥ 6.1 ships `libvvenc` enable-flag and the wrapper forwards `-vvenc-params` opaquely. |
| Surface all three NNVC tools (intra + loop filter + super-res) in this PR | Front-loads the AI-augmented surface in one merge. | NN loop filter and NN super-res have non-trivial encoder-internal state, decoder-side cost, and quality-curve interaction with QP that the corpus needs to learn separately; bundling four toggles in one adapter PR muddies the schema delta. | Rejected: ship the simplest tool (`IntraNN`) first, defer loop-filter and super-res to follow-up ADRs once Phase B has data. |
| Use the wider `quality_range = (0, 63)` (full VVenC scale) | Matches the encoder's documented range exactly. | The 0..16 and 51..63 ends are degenerate for natural content (visually-lossless and severely-banded respectively); Phase B bisect would waste budget. | Rejected: pin the perceptually informative window, mirroring x264's `(15, 40)` posture. |
| Emit `-qp` instead of `-crf` in `build_ffmpeg_command` | Matches VVenC's documented label. | Would require either a per-codec branch in `encode.build_ffmpeg_command` (violates ADR-0237's "no codec branching" rule) or a wider refactor of the shared encode path. | Rejected: VVenC's FFmpeg wrapper accepts the integer value forwarded via `-crf` and the corpus row's `crf` column is already encoder-agnostic semantics. Schema rename can land later as a coordinated bump. |

## Consequences

- **Positive**: VMAF-tune can drive the highest-compression standard
  codec available, including the NN-augmented intra path. The corpus
  row's `extra_params` column carries the NNVC toggle, so any Phase B
  / C predictor can condition on it without a schema bump. The
  adapter set crosses the H.264 → H.265 → AV1 → H.266 boundary with
  uniform interface.
- **Negative**: VVenC encodes are ~3-5× slower than HEVC slower at
  equal preset; running a full Phase A grid against VVenC takes
  noticeably longer than against x264. The integration smoke gate
  needs FFmpeg compiled with `--enable-libvvenc`, which not every
  CI runner has — the unit tests mock subprocess so the gate is
  available locally regardless.
- **Neutral / follow-ups**:
  - Update [`docs/usage/vmaf-tune.md`](../usage/vmaf-tune.md) with a
    "VVenC (H.266 / VVC + NNVC)" section explaining the NN-tool
    semantics. (Done in this PR.)
  - Add a follow-up ADR for the NN loop filter and NN super-res
    toggles once a corpus exists to estimate their quality / cost
    curves.
  - Coordinate with the FR-regressor v2 schema-expansion PR
    ([ADR-0235](0235-codec-aware-fr-regressor.md)) on the codec
    one-hot — VVenC takes the next free slot.

## References

- [ADR-0237](0237-quality-aware-encode-automation.md) — `vmaf-tune`
  parent ADR.
- [ADR-0235](0235-codec-aware-fr-regressor.md) — codec-aware FR
  regressor (one-hot shape coordination).
- VVenC upstream: <https://github.com/fraunhoferhhi/vvenc>.
- VVC NNVC tool description: ITU-T H.266 (V3) — neural-network-based
  video coding tools, 2024-09.
- Source: `req` (direct user instruction 2026-05-03 to add the
  vvenc + NN-VC adapter as a companion to the parallel x264 / x265 /
  svt-av1 / libaom + NVENC / QSV / AMF / VideoToolbox adapter PRs).
