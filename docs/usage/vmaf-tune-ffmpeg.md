# Using vmaf-tune with FFmpeg's encoder-side hooks

The `tools/vmaf-tune/` orchestrator drives encodes through FFmpeg.
Three FFmpeg-side hooks (added via the `ffmpeg-patches/` 0007–0009
series, ADR-0312) make that integration first-class instead of
out-of-band.

This page documents how each hook plugs into vmaf-tune's three
operating modes: **saliency-aware encoding**, **CRF recommendation**,
and **2-pass autotuning**.

## Prerequisites

Apply the fork's FFmpeg patches against a clean `n8.1` checkout:

```bash
cd /path/to/ffmpeg && git checkout n8.1
for p in /path/to/vmaf/ffmpeg-patches/000*-*.patch; do
    git am --3way "$p" || break
done
./configure --enable-libvmaf --enable-libx264 --enable-libsvtav1 --enable-libaom --enable-gpl
make -j$(nproc)
```

Confirm the new options are recognised:

```bash
./ffmpeg -h encoder=libx264   2>&1 | grep -i qpfile
./ffmpeg -h encoder=libsvtav1 2>&1 | grep -i qpfile
./ffmpeg -h encoder=libaom-av1 2>&1 | grep -i qpfile
./ffmpeg -h filter=libvmaf_tune
./ffmpeg -h | grep -i pass-autotune
```

## Hook 1: `-qpfile <path>` (patch 0007)

The new `-qpfile` AVOption on `libx264`, `libsvtav1`, and `libaom-av1`
consumes the qpfile format emitted by vmaf-tune's saliency module
(`tools/vmaf-tune/src/vmaftune/saliency.py`):

```text
<frame_idx> <I|P|B> <baseline_qp>
<delta_0_0> <delta_0_1> ... <delta_0_(bw-1)>
<delta_1_0> <delta_1_1> ... <delta_1_(bw-1)>
...
```

**Where each `delta` is a per-block QP offset clamped to `[-12, +12]`.**

The shared parser at `libavcodec/qpfile_parser.{c,h}` reads this
format once, then each encoder adapter dispatches it to its native
ROI/QP-offset API.

### libx264 — fully wired

```bash
# 1. vmaf-tune emits a saliency-driven qpfile
python -m vmaftune.saliency --source clip.yuv --width 1920 --height 1080 \
    --output clip.qpfile.txt

# 2. ffmpeg consumes it directly
ffmpeg -f rawvideo -s 1920x1080 -pix_fmt yuv420p -i clip.yuv \
    -c:v libx264 -crf 23 -qpfile clip.qpfile.txt clip.mp4
```

x264 honours per-MB QP deltas natively (since r2390); the patch
forwards `-qpfile <path>` to `x264_param_parse(... "qpfile", path)`.

### libsvtav1 — scaffold (parses, doesn't apply)

```bash
ffmpeg -f rawvideo ... -c:v libsvtav1 -crf 32 -qpfile clip.qpfile.txt clip.av1
```

Today this **logs a warning** and continues without applying the
ROI map:

```text
[libsvtav1 @ 0x…] libsvtav1: qpfile=clip.qpfile.txt parsed
(frames=240, 120x68 blocks); SVT-AV1 ROI translation deferred
(ADR-0312). See docs/usage/vmaf-tune-ffmpeg.md.
```

For end-to-end behaviour today, drive the encode through
`vmaf-tune corpus` (which uses SVT-AV1's existing
`-svtav1-params roi-map=…` plumbing).

### libaom-av1 — scaffold

Same posture: parses, validates, logs. Use vmaf-tune's CLI for
end-to-end ROI behaviour until the `AV1E_SET_ROI_MAP` bridge lands.

## Hook 2: `-vf libvmaf_tune` (patch 0008)

A new 2-input video filter that runs alongside a 1-pass encode and
emits a recommended CRF for the next pass:

```bash
ffmpeg -i input.mp4 -i reference.mp4 \
    -lavfi "[0:v][1:v]libvmaf_tune=recommend_target_vmaf=92:recommend_crf_min=18:recommend_crf_max=40" \
    -f null -
```

At the end of the run the filter logs:

```text
[Parsed_libvmaf_tune_0 @ 0x…] recommended_crf=24.3 (target_vmaf=92.0, n_frames=240)
```

### Options

| Option                    | Default  | Notes                                                       |
|---------------------------|----------|-------------------------------------------------------------|
| `model`                   | `version=vmaf_v0.6.1` | libvmaf model spec.                                       |
| `feature`                 | (none)   | Optional `:`-separated feature spec.                         |
| `n_threads`               | `0`      | Worker threads (0 = libvmaf default).                        |
| `recommend_target_vmaf`   | `95.0`   | Target score the recommendation aims for.                    |
| `recommend_crf_min`       | `18.0`   | Lower CRF bound considered.                                  |
| `recommend_crf_max`       | `51.0`   | Upper CRF bound considered.                                  |
| `recommend_passes`        | `1`      | Probe-pass count (scaffold: ignored).                        |

### Scaffold caveat

The recommended CRF is a **linear interpolation** between the
configured min/max around the target. The full Optuna TPE search
that vmaf-tune uses internally
(`tools/vmaf-tune/src/vmaftune/recommend.py`) is **not** invoked by
this filter — that orchestration stays in the Python tool. The
filter is the FFmpeg-side ABI surface that future iterations can
grow into.

## Hook 3: `-pass-autotune` (patch 0009)

A new advisory CLI flag for vmaf-tune-driven 2-pass encodes:

```bash
ffmpeg -i input.mp4 -c:v libx264 -pass-autotune -f null -
```

```text
[ffmpeg] -pass autotune: drive vmaf-tune externally; pass 1 frames
will be available for analysis. See docs/usage/vmaf-tune-ffmpeg.md.
```

The flag is **glue only** — when set, FFmpeg behaves like a normal
1-pass encode and prints the advisory line. Real 2-pass orchestration
(probe pass → recommend → final pass) lives in
`tools/vmaf-tune/src/vmaftune/recommend.py`. The flag exists so
shell scripts that call ffmpeg directly can signal the user-visible
intent without inventing their own log conventions.

## End-to-end recipe: saliency-aware libx264 encode

```bash
# 1. emit qpfile
python -m vmaftune.saliency \
    --source ref.yuv --width 1920 --height 1080 \
    --output ref.qpfile.txt --foreground-offset -4

# 2. encode with the qpfile
ffmpeg -f rawvideo -s 1920x1080 -pix_fmt yuv420p -i ref.yuv \
    -c:v libx264 -preset medium -crf 23 -qpfile ref.qpfile.txt out.mp4

# 3. score the result
ffmpeg -i ref.yuv -i out.mp4 \
    -lavfi "[0:v][1:v]libvmaf_tune=recommend_target_vmaf=95" \
    -f null - 2>&1 | grep recommended_crf
```

If step 3 reports `recommended_crf` significantly different from 23,
re-encode with the suggested value.

## Troubleshooting

### `unknown option qpfile`

You are running unpatched FFmpeg. Re-apply the patch series and
rebuild — see Prerequisites above.

### `libx264: failed to load qpfile=… (x264 ret=-1)`

Either the file does not exist, or its format does not match
x264's qpfile reader. Run `python -m vmaftune.saliency --validate
<path>` to round-trip the file through the parser.

### `libsvtav1: qpfile=… parsed (…); SVT-AV1 ROI translation deferred`

Expected today — the parser ran but the bridge to SVT-AV1's ROI
input is not yet wired. Use `vmaf-tune corpus --codec svtav1` for
end-to-end behaviour, or wait for the ADR-0312 follow-up patch.

## See also

- [ADR-0312](../adr/0312-ffmpeg-patches-vmaf-tune-integration.md) —
  decision context.
- [Research-0084](../research/0084-ffmpeg-patch-vmaf-tune-integration-survey.md)
  — survey of VQA-tool / FFmpeg integration patterns.
- [`tools/vmaf-tune/`](../../tools/vmaf-tune/) — the harness that emits
  qpfiles and runs the recommend loop.
- [`ffmpeg-patches/`](../../ffmpeg-patches/) — the patch series.
