# FastDVDnet temporal pre-filter (5-frame window)

`fastdvdnet_pre` — a *temporal* denoising pre-filter for noisy / grainy
sources. Unlike the per-frame `learned_filter_v1` (capability C3), this
extractor consumes a 5-frame sliding window
``[t-2, t-1, t, t+1, t+2]`` and emits a denoised version of frame
``t``. Denoise-before-encode is a well-validated bitrate lever: removing
sensor noise and grain before the encoder sees them produces smaller
files at the same subjective quality.

> **Status — placeholder weights only.** The ONNX shipped under
> `model/tiny/fastdvdnet_pre.onnx` in this PR is a smoke-only stand-in
> with a tiny randomly-initialised CNN that respects the I/O shape
> contract. It is **not** a working denoiser. The real upstream-derived
> FastDVDnet weights drop is tracked as backlog item **T6-7b** alongside
> the FFmpeg `vmaf_pre_temporal` filter that actually consumes the
> denoised frame buffer. See [ADR-0215](../../adr/0215-fastdvdnet-pre-filter.md)
> for the placeholder rationale.

## What the output means

The extractor appends a single per-frame feature named
`fastdvdnet_pre_l1_residual`, defined as the mean-absolute difference
between the input centre frame `t` (normalised to `[0, 1]`) and the
denoised output. It exists so libvmaf's per-frame plumbing has a scalar
to record; it is **not** a quality metric. Downstream pipelines that
want the actual denoised pixel data should consume the FFmpeg
`vmaf_pre_temporal` filter once T6-7b lands — that filter writes the
denoised frame buffer to its output stream.

| Value | Interpretation |
| --- | --- |
| **~0.0** | Output ≈ input centre (placeholder near-identity, or quiet/flat content). |
| **~0.05** | Typical residual on a working FastDVDnet model and lightly noisy content. |
| **~0.20+** | Heavy denoising (working model on grainy / film content) — or a saturated placeholder pass. |

## Shipped checkpoint

| Field | Value |
| --- | --- |
| Model name | `vmaf_tiny_fastdvdnet_pre_placeholder_v0` |
| Location | `model/tiny/fastdvdnet_pre.onnx` |
| Size | ~6 KB (placeholder; real weights drop in T6-7b will be ~10 MB) |
| ONNX opset | 17 |
| Input | `frames` — float32 NCHW `[1, 5, H, W]` (5-frame stack along channel axis) |
| Output | `denoised` — float32 NCHW `[1, 1, H, W]` (denoised frame `t`) |
| Smoke flag | `smoke: true` in registry — load-path probe, not a quality model |
| License path | T6-7b will inherit upstream MIT (m-tassano/fastdvdnet) when real weights ship |

The sidecar JSON at `model/tiny/fastdvdnet_pre.json` carries the input /
output names plus `frame_window: 5`, `centre_index: 2` so downstream
consumers can validate the contract without parsing the ONNX graph.

## Frame window contract

The C extractor (`libvmaf/src/feature/fastdvdnet_pre.c`) maintains a
five-slot ring buffer of the most recent normalised luma planes. Each
`extract()` call:

1. Pushes the new luma plane (any bpc; rescaled to `[0, 1]`) into the
   ring at `next_slot`.
2. Gathers the five window slots into a `[1, 5, H, W]` input tensor.
   At clip start (when fewer than 5 frames have been seen) and clip end
   (no future frames available) the missing slots replicate the closest
   available end frame — a replicate-edge convention that matches
   FastDVDnet's published reflection-pad-light behaviour.
3. Calls `vmaf_dnn_session_run` with the named bindings
   `frames` (input) and `denoised` (output).
4. Computes the L1 residual against the centre input frame and appends
   it via `vmaf_feature_collector_append`.

## Integration recipe

```bash
# 1. Build libvmaf with DNN support enabled.
meson setup build -Denable_dnn=true
ninja -C build

# 2. Run the extractor against a clip, supplying the model path.
build/libvmaf/tools/vmaf \
    --reference ref.yuv --distorted dis.yuv \
    --width 1920 --height 1080 --pixel_format yuv420p --bitdepth 8 \
    --feature fastdvdnet_pre=model_path=model/tiny/fastdvdnet_pre.onnx

# Or via env var (matches lpips_sq's pattern):
VMAF_FASTDVDNET_PRE_MODEL_PATH=model/tiny/fastdvdnet_pre.onnx \
    build/libvmaf/tools/vmaf --feature fastdvdnet_pre …
```

The extractor declines cleanly (non-fatal `-EINVAL`) if neither
`model_path` nor `VMAF_FASTDVDNET_PRE_MODEL_PATH` is set, the same
contract as the LPIPS extractor.

## Smoke test

The C-side registration + options-table contract is exercised by
`libvmaf/test/test_fastdvdnet_pre.c`, which runs in every build (no ORT
session required). To exercise the live ORT path against the placeholder
ONNX, enable DNN and run the suite:

```bash
meson test -C build --suite=fast --print-errorlogs test_fastdvdnet_pre
```

## Path to real weights (T6-7b)

When upstream FastDVDnet weights are vendored:

1. Replace `model/tiny/fastdvdnet_pre.onnx` with the trained checkpoint
   (re-export from PyTorch with the same input/output names).
2. Update the `sha256` digest, `notes`, `name`, and remove `smoke: true`
   from `model/tiny/registry.json`.
3. Update the sidecar JSON `model/tiny/fastdvdnet_pre.json` (drop the
   `smoke: true` flag, refresh `name`).
4. Add the FFmpeg `vmaf_pre_temporal` filter under
   `ffmpeg-patches/0005-add-vmaf_pre_temporal-filter.patch`.
5. Verify the op allowlist in
   [`libvmaf/src/dnn/op_allowlist.c`](../../../libvmaf/src/dnn/op_allowlist.c)
   covers every op in the trained graph — FastDVDnet's published
   architecture uses `Conv`, `Relu`, `BatchNormalization`,
   `PixelShuffle`, and `Concat`, all of which are already allowed.

## References

- Tassano, Delon, Veit. *FastDVDnet: Towards Real-Time Deep Video
  Denoising Without Flow Estimation*, CVPR 2020. arXiv:1907.01361.
- Reference implementation:
  [github.com/m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet)
- [ADR-0215](../../adr/0215-fastdvdnet-pre-filter.md) — design decision.
- [Roadmap §3.3](../roadmap.md) — Wave 1 schedule.
