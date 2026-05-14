# `vmaf-tune --saliency-aware`

`vmaf-tune recommend-saliency` runs a single saliency-aware encode. It
materialises a saliency sidecar from the shipped saliency model,
translates that sidecar into codec-specific ROI/QP controls, and then
dispatches the encode through the normal codec-adapter path.

The implementation lives in `tools/vmaf-tune/src/vmaftune/saliency.py`
and is wired through `tools/vmaf-tune/src/vmaftune/cli.py`.

## Quick Start

```shell
vmaf-tune recommend-saliency \
    --src ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --encoder libx264 \
    --preset medium --crf 23 \
    --saliency-offset -3 \
    --output roi.mp4
```

## What Happens

1. `load_saliency_sidecar()` loads an existing sidecar or runs the
   saliency model to create one.
2. `build_roi_plan()` converts frame-level saliency into encoder ROI
   controls.
3. `run_saliency_encode()` dispatches the codec-specific encode.
4. Unsupported ROI encoders fall back to a plain encode with a warning
   rather than failing the whole run.

The shipped default model is documented in
[`saliency_student_v1.md`](../ai/models/saliency_student_v1.md).

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--src PATH` | — | Source clip. |
| `--width / --height` | — | Source geometry. |
| `--pix-fmt` | `yuv420p` | Source pixel format. |
| `--framerate` | `24.0` | Source framerate. |
| `--duration` | `0.0` | Source duration. |
| `--encoder` | `libx264` | Codec adapter. |
| `--preset` | `medium` | Codec preset. |
| `--crf` | `23` | Base quality before ROI offsets. |
| `--saliency-offset` | `-3` | QP/quality offset applied to salient regions. |
| `--saliency-model PATH` | shipped model | Override saliency ONNX path. |
| `--ffmpeg-bin` | `ffmpeg` | FFmpeg binary. |
| `--output PATH` | — | Encoded output. |

## See Also

- [`vmaf-tune.md`](vmaf-tune.md) — base tool.
- [`vmaf-tune-ffmpeg.md`](vmaf-tune-ffmpeg.md) — FFmpeg integration
  recipe.
- [`vmaf-roi-score.md`](vmaf-roi-score.md) — saliency-weighted scoring.
- [ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md) — design
  decision.
