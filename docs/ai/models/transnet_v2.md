# TransNet V2 shot-boundary detector (100-frame window)

`transnet_v2` — a *shot-change* detector that consumes a 100-frame
sliding window of small RGB thumbnails and emits one shot-boundary
probability per frame. The first half of the Wave 1 §2.4 content-
adaptive encoding pipeline (the second half — per-shot CRF prediction —
is **T6-3b**, a follow-up that consumes these per-frame probabilities
through the existing feature collector).

> **Status — placeholder weights only.** The ONNX shipped under
> `model/tiny/transnet_v2.onnx` in this PR is a smoke-only stand-in
> with a tiny randomly-initialised MLP that respects the I/O shape
> contract. It is **not** a working shot detector. The real
> upstream-derived TransNet V2 weights drop is tracked as backlog item
> **T6-3a-followup**. See [ADR-0220](../../adr/0220-transnet-v2-shot-detector.md)
> for the placeholder rationale.

## What the outputs mean

The extractor appends two per-frame features:

| Feature name | Type | Meaning |
| --- | --- | --- |
| `shot_boundary_probability` | float32 in `[0, 1]` | Sigmoid of the network's logit for the current frame. ~0.0 = no cut, ~1.0 = high-confidence cut. |
| `shot_boundary` | float32 ∈ `{0.0, 1.0}` | Binary flag thresholded at 0.5 against the probability. Drop-in for naive consumers. |

Downstream consumers (the per-shot CRF predictor T6-3b, the FFmpeg
shot-cut filter shipping with T6-3b) bind to those exact strings.

| Probability | Interpretation |
| --- | --- |
| **~0.05** | No shot change — typical mid-shot frame on a working detector. |
| **~0.50** | Detector uncertain — common on the placeholder graph (sigmoid of near-zero logits). |
| **~0.95** | High-confidence shot cut on a working detector. |

## Shipped checkpoint

| Field | Value |
| --- | --- |
| Model name | `vmaf_tiny_transnet_v2_placeholder_v0` |
| Location | `model/tiny/transnet_v2.onnx` |
| Size | ~125 KB (placeholder; real weights drop in T6-3a-followup will be ~4 MB / ~1M params) |
| ONNX opset | 17 |
| Input | `frames` — float32 `[1, 100, 3, 27, 48]` (100-frame stack of RGB thumbnails) |
| Output | `boundary_logits` — float32 `[1, 100]` (per-frame logits before sigmoid) |
| Smoke flag | `smoke: true` in registry — load-path probe, not a real detector |
| License | BSD-3-Clause-Plus-Patent (placeholder); T6-3a-followup will inherit upstream MIT (soCzech/TransNetV2) when real weights ship |

The sidecar JSON at `model/tiny/transnet_v2.json` carries the input /
output names plus `frame_window: 100`, `thumbnail_h: 27`,
`thumbnail_w: 48`, `boundary_threshold: 0.5` so downstream consumers
can validate the contract without parsing the ONNX graph.

## Frame window contract

The C extractor (`libvmaf/src/feature/transnet_v2.c`) maintains a
100-slot ring buffer of pre-resized RGB thumbnail tensors. Each
`extract()` call:

1. Resizes the input luma plane (any bpc; rescaled to `[0, 1]`) down
   to a 27x48 grid via nearest-neighbour, then broadcasts that single
   plane across all three RGB channels (placeholder — T6-3a-followup
   will replace this with a true RGB decode + bilinear resize).
2. Pushes the resized frame into the ring at `next_slot`.
3. Gathers the 100 ring slots into a `[1, 100, 3, 27, 48]` input
   tensor. At clip start (when fewer than 100 frames have been seen)
   the missing slots replicate the oldest available frame
   (head-clamp).
4. Calls `vmaf_dnn_session_run` with the named bindings `frames`
   (input) and `boundary_logits` (output).
5. Reads the most recent slot's logit (index `WINDOW-1`), sigmoids
   it, and appends both `shot_boundary_probability` and
   `shot_boundary` (thresholded at 0.5) via
   `vmaf_feature_collector_append`.

The first ~50 frames of any clip should be treated as warm-up: the
detector hasn't seen enough context to make a confident decision.

## Integration recipe

```bash
# 1. Build libvmaf with DNN support enabled.
meson setup build -Denable_dnn=true
ninja -C build

# 2. Run the extractor against a clip, supplying the model path.
build/libvmaf/tools/vmaf \
    --reference ref.yuv --distorted dis.yuv \
    --width 1920 --height 1080 --pixel_format yuv420p --bitdepth 8 \
    --feature transnet_v2=model_path=model/tiny/transnet_v2.onnx

# Or via env var (matches lpips_sq / fastdvdnet_pre):
VMAF_TRANSNET_V2_MODEL_PATH=model/tiny/transnet_v2.onnx \
    build/libvmaf/tools/vmaf --feature transnet_v2 ...
```

The extractor declines cleanly (non-fatal `-EINVAL`) if neither
`model_path` nor `VMAF_TRANSNET_V2_MODEL_PATH` is set, the same
contract as the LPIPS and FastDVDnet extractors.

## Smoke test

The C-side registration + options-table contract + dual-feature
surface is exercised by `libvmaf/test/test_transnet_v2.c`, which
runs in every build (no ORT session required). To exercise the live
ORT path against the placeholder ONNX:

```bash
meson test -C build --suite=fast --print-errorlogs test_transnet_v2
```

To smoke the full 100-frame round-trip via Python ORT:

```bash
python3 -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('model/tiny/transnet_v2.onnx',
                            providers=['CPUExecutionProvider'])
x = np.random.RandomState(7).rand(1, 100, 3, 27, 48).astype(np.float32)
y = sess.run(['boundary_logits'], {'frames': x})[0]
print('shape', y.shape, 'mean prob',
      float((1.0/(1.0+np.exp(-y))).mean()))
"
```

## Path to real weights (T6-3a-followup)

When upstream TransNet V2 weights are vendored:

1. Convert the upstream TensorFlow checkpoint at
   [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)
   to ONNX via `tf2onnx`. Manual graph cleanup may be required to
   match the fork's strict op allowlist.
2. Replace `model/tiny/transnet_v2.onnx` with the converted
   checkpoint (preserve input/output names: `frames` /
   `boundary_logits`, shapes: `[1, 100, 3, 27, 48]` / `[1, 100]`).
3. Update the `sha256` digest, `notes`, `name`, `license`
   (→ `MIT`), and remove `smoke: true` from
   `model/tiny/registry.json`.
4. Update the sidecar JSON `model/tiny/transnet_v2.json` (drop the
   `smoke: true` flag, refresh `name`, update `license` to upstream
   MIT).
5. Switch the C extractor's `luma_to_thumbnail` from
   nearest-neighbour resize + luma-broadcast to bilinear resize +
   true RGB decode (the published TransNet V2 uses bilinear).
6. Verify the op allowlist in
   [`libvmaf/src/dnn/op_allowlist.c`](../../../libvmaf/src/dnn/op_allowlist.c)
   covers every op in the upstream graph — TransNet V2 uses
   3D dilated convolutions (DDCNN), `LayerNormalization`, `Concat`,
   `Reshape`, and `MatMul`.

## References

- Soucek, Lokoc. *TransNet V2: An effective deep network architecture
  for fast shot transition detection*, 2020.
  [arXiv:2008.04838](https://arxiv.org/abs/2008.04838).
- Reference implementation:
  [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2).
- [ADR-0220](../../adr/0220-transnet-v2-shot-detector.md) — design decision.
- [Roadmap §2.4](../roadmap.md) — Wave 1 schedule.
- [ADR-0215](../../adr/0215-fastdvdnet-pre-filter.md) — sister
  placeholder-ONNX pattern (5-frame window FastDVDnet).
