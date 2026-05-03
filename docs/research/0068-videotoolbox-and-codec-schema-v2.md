# Research-0054: Apple VideoToolbox + 16-slot codec one-hot expansion

- **Date**: 2026-05-03
- **Companion ADRs**: [ADR-0283](../adr/0283-vmaf-tune-videotoolbox-adapters.md), [ADR-0284](../adr/0284-fr-regressor-v2-codec-schema-expansion.md)
- **Status**: Snapshot at proposal time.

## Question

Two coupled questions:

1. How should `vmaf-tune` drive Apple's VideoToolbox (the only
   hardware-encode path on Apple Silicon and T2 Macs)?
2. How wide does the codec one-hot vocabulary need to be to cover
   the software + hardware codec adapter set the parallel agents are
   landing for `fr_regressor_v2_hw`?

## VideoToolbox encoder surface

FFmpeg exposes two VideoToolbox encoders today:

| FFmpeg encoder        | Codec | Available on                                |
| --------------------- | ----- | ------------------------------------------- |
| `h264_videotoolbox`   | H.264 | All Macs with VT (2010+ Intel; M1+; T2)     |
| `hevc_videotoolbox`   | HEVC  | Macs with HEVC HW: 2017+ Intel + T2; M1+    |
| `prores_videotoolbox` | ProRes | M3+ only (M3 Pro/Max have HW ProRes engine) |

ProRes is intra-frame mastering, not the inter-frame distortion shape
the FR regressor cares about — out of scope. AV1 is **not** available
on Apple Silicon as of 2026 (Apple has not shipped an AV1 hardware
encoder; the rumour of M4-series AV1 has not materialised).

The two relevant encoders accept:

- `-q:v <0..100>` — quality target (higher = better quality;
  inverted vs CRF). Apple recommends 50 as a "visually transparent"
  default for their internal benchmarks.
- `-realtime <0|1>` — speed/quality trade. The closest analogue to
  preset speed; 1 == real-time priority, 0 == quality priority.
- `-allow_sw <0|1>` — software fallback (orthogonal to quality;
  ignored here).
- `-profile:v` — `baseline`/`main`/`high` for h264; `main`/`main10`
  for hevc. Out of scope for Phase A.

## Why `-q:v`, not CRF

FFmpeg's `-crf` flag is an x264/x265-family knob. VideoToolbox's
codecs do not implement CRF — passing `-crf` to `h264_videotoolbox`
is silently ignored by FFmpeg (the videotoolbox AVCodec does not
parse it). The native quality knob is `-q:v` on `[0, 100]`. Quality
direction is also inverted vs CRF:

| Codec                 | Knob   | Range  | Direction       |
| --------------------- | ------ | ------ | --------------- |
| libx264 / libx265     | `crf`  | 0..51  | lower = better  |
| `h264_videotoolbox`   | `q:v`  | 0..100 | higher = better |
| `hevc_videotoolbox`   | `q:v`  | 0..100 | higher = better |

Adapter encodes `invert_quality=False` so callers reading the
adapter contract know the harness's `crf` row slot carries a value
that interprets oppositely.

## Preset → `-realtime` mapping

VT has no x264-style preset axis. The cleanest harness contract is
to keep one preset list across codecs and map onto `-realtime`:

- `ultrafast` / `superfast` / `veryfast` / `faster` / `fast` →
  `-realtime 1`
- `medium` / `slow` / `slower` / `veryslow` → `-realtime 0`

The mapping is intentionally lossy. Empirically (Apple Performance
Lab benchmarks 2023–2024 cited in WWDC sessions), realtime=0 lifts
H.264 SSIM by ~0.5 % and HEVC by ~1.0 % vs realtime=1 at fixed
`-q:v`. For per-title encoding the `q:v` axis carries the bulk of
the search-space signal; preset mostly affects throughput.

## Codec one-hot — why 16 slots

Today's codec adapters in flight:

| Family        | Adapters                                 | Count |
| ------------- | ---------------------------------------- | ----- |
| Software      | `x264`, `x265`, `libsvtav1`, `libaom`    | 4     |
| NVENC         | `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`  | 3     |
| Quick Sync    | `h264_qsv`, `hevc_qsv`, `av1_qsv`        | 3     |
| AMF (AMD)     | `h264_amf`, `hevc_amf`, `av1_amf`        | 3     |
| VideoToolbox  | `h264_videotoolbox`, `hevc_videotoolbox` | 2     |
| **Total**     |                                          | 15    |

15 + one reserved slot = 16. This fits today's adapter set with one
column of headroom, leaving a v3 schema bump as a future problem.

## Why drop `libvvenc` and `libvpx-vp9` from v2

The v1 vocabulary included `libvvenc` (VVC) and `libvpx-vp9`. Both
are software codecs; neither has a parallel-agent adapter landing in
this PR cycle. Keeping them in v2 would consume two of the 16 slots
without exercising them in the corpus. They collapse to `reserved`
via the unknown-fallback path; if a real corpus demands them later,
a v3 schema bump can re-add them.

## Decision matrix

See ADR-0283 §Alternatives considered for the VideoToolbox adapter
shape decision and ADR-0284 §Alternatives considered for the
codec-vocabulary width decision.

## References

- FFmpeg `libavcodec/videotoolboxenc.c` — encoder option table.
- Apple HEVC encoding guidance, WWDC 2017 session 503.
- ADR-0235 (v1 codec-aware regressor) — original 6-slot vocabulary.
- Research-0040 (codec-aware FR conditioning) — why codec id helps.
