# MobileSal saliency (no-reference scoring-side extractor)

`vmaf_tiny_mobilesal_placeholder_v0` — a no-reference saliency feature
extractor that runs a tiny ONNX saliency model over the distorted frame
and emits the **mean of its per-pixel saliency map** as a scalar feature
named `saliency_mean`. It is the scoring-side surface for
[Wave 1 §2.3 of the tiny-AI roadmap](../roadmap.md), the first half of
backlog item T6-2 (T6-2a). Encoder-side ROI tooling
(`tools/vmaf-roi`, per-CTU QP-offset sidecars) is the deliberate T6-2b
follow-up.

> The shipped checkpoint is a **smoke-only synthetic placeholder** that
> matches the upstream MobileSal I/O contract bit-for-bit, but emits
> ~constant saliency. The original plan to swap in real upstream
> weights from
> [yuhuan-wu/MobileSal](https://github.com/yuhuan-wu/MobileSal) is
> **deferred indefinitely** (see
> [ADR-0257](../../adr/0257-mobilesal-real-weights-deferred.md) +
> [Research-0053](../../research/0053-mobilesal-real-weights-blocker.md)):
> upstream is licensed CC BY-NC-SA 4.0 (incompatible with the fork's
> BSD-3-Clause-Plus-Patent), distributes weights through Google Drive
> viewer URLs (no GitHub release, no raw-download URL the export
> script can pin), and is an RGB-D model (the C contract is RGB-only).
> The recommended replacement path is to swap the underlying model
> family to U-2-Net's `u2netp` variant under Apache-2.0; that work
> is tracked as backlog row T6-2a-replace-with-u2netp. Until then,
> use the placeholder to wire pipelines up end-to-end and treat
> `saliency_mean` as a content-independent constant.

Upstream paper: Wu, Liu, Cheng, Lu, Cheng, *"MobileSal: Extremely
Efficient RGB-D Salient Object Detection"*, IEEE TPAMI 2021.

## What the output means

The extractor emits a single feature named `saliency_mean`, one value
per frame (the distorted frame is the input — saliency is no-reference).

| Value | Interpretation |
| --- | --- |
| **~0.0** | Flat / featureless content; no salient subject |
| **~0.2 – 0.4** | Typical natural-content frame |
| **~0.5** | Foreground subject occupies a sizeable fraction of the frame |
| **~0.8+** | Subject dominates; mostly-salient content |
| **1.0** | Saturated (every pixel maxed) — usually a sign of model misuse |

Saliency-mean is **not** a quality score on its own — it is a content
descriptor. Downstream consumers correlate `saliency_mean` against
existing metric features (e.g. `vmaf`, `lpips`, `psnr`) to study how
foreground-vs-background distortion affects subjective quality.

The full saliency map is computed internally to derive the mean; it is
intentionally not exposed as a per-pixel feature in T6-2a. The encoder
side (`tools/vmaf-roi` per-CTU QP-offset sidecar) consumes the same
model in T6-2b and exports the map in encoder-native format.

## Shipped checkpoint

| Field | Value |
| --- | --- |
| Model id | `mobilesal_placeholder_v0` |
| Display name | `vmaf_tiny_mobilesal_placeholder_v0` |
| Location | `model/tiny/mobilesal.onnx` |
| Size | 330 bytes (synthetic placeholder) |
| SHA-256 | `f122631089977c4be7d60b9bf3d4daf186d275bd0587db2c9878578e006b91d4` |
| ONNX opset | 17 |
| Upstream source (paper) | [yuhuan-wu/MobileSal](https://github.com/yuhuan-wu/MobileSal) (HEAD `8f42ded5`; not currently shippable — see ADR-0257) |
| License (placeholder) | BSD-3-Clause-Plus-Patent (this fork) |
| License (upstream MobileSal weights) | CC BY-NC-SA 4.0 — **incompatible with the fork**; per `yuhuan-wu/MobileSal/README.md` §License. ADR-0218's MIT claim was inaccurate; corrected here and in ADR-0257. |
| Exporter (placeholder) | `scripts/gen_mobilesal_placeholder_onnx.py` |
| Registry entry | `mobilesal_placeholder_v0` in `model/tiny/registry.json` (smoke=true) |
| Status | Placeholder — real weights tracked as T6-2a-followup |

The placeholder ONNX is deterministic (no `doc_string`, fixed
`producer_version`, deterministic protobuf serialisation) so the
sha256 stays stable across re-runs of the export script.

## Input / output contract

The C extractor binds tensors by name, so any future drop-in (real
upstream MobileSal export, distilled student, etc.) must declare the
exact same names:

```text
inputs:
  input         float32[1, 3, H, W]   ImageNet-normalised RGB, NCHW
outputs:
  saliency_map  float32[1, 1, H, W]   per-pixel saliency in [0, 1]
```

`H` and `W` are dynamic — both the placeholder and the upstream graph
match whatever resolution the C side feeds. ImageNet normalisation
(mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) is applied
in the C side via the shared `vmaf_tensor_from_rgb_imagenet()` helper,
identical to LPIPS's wiring (see [`lpips_sq.md`](lpips_sq.md)).

## Usage — CLI

```bash
vmaf \
    --reference ref.yuv \
    --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --feature mobilesal \
    --feature_params mobilesal:model_path=model/tiny/mobilesal.onnx \
    --output score.json
```

The output JSON gains a per-frame `saliency_mean` column alongside any
other features requested in the same run. Combine with `lpips` and
`vmaf` for the full saliency-quality picture:

```bash
vmaf --reference ref.yuv --distorted dist.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --feature vmaf \
    --feature lpips \
    --feature_params lpips:model_path=model/tiny/lpips_sq.onnx \
    --feature mobilesal \
    --feature_params mobilesal:model_path=model/tiny/mobilesal.onnx \
    --output combined.json
```

Equivalently, set the model path via env var:

```bash
VMAF_MOBILESAL_MODEL_PATH=model/tiny/mobilesal.onnx \
    vmaf --reference ref.yuv --distorted dist.yuv \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --feature mobilesal --output score.json
```

## Usage — C API

```c
#include <libvmaf/libvmaf.h>

VmafFeatureDictionary *opts = NULL;
vmaf_feature_dictionary_set(&opts, "model_path", "model/tiny/mobilesal.onnx");
int err = vmaf_use_feature(ctx, "mobilesal", opts);
/* ... vmaf_score_pooled(ctx, ..., "saliency_mean", ...) for the per-frame mean */
```

Equivalent to setting `VMAF_MOBILESAL_MODEL_PATH` before
`vmaf_use_feature(ctx, "mobilesal", NULL)`.

## Known limitations

- **Bit depth**: 8-bit YUV only in this build (`bpc != 8` returns
  `-ENOTSUP`). 10-bit support is gated on the same loader path landing
  for LPIPS, see [`lpips_sq.md`](lpips_sq.md).
- **Pixel format**: `YUV420P`, `YUV422P`, `YUV444P` accepted;
  `YUV400P` (luma-only) is rejected at `init()` because the model
  requires three RGB channels.
- **Colour space**: BT.709 limited-range Y'CbCr → RGB at the C side,
  matching `feature_lpips.c`. BT.2020 / full-range is approximate
  (deliberate trade-off — see `feature_mobilesal.c` comment).
- **Resolution**: bounded by the synthetic placeholder's dynamic shape
  (no minimum); the real upstream MobileSal recommends ≥ 224×224.
- **CPU vs GPU path**: served via `vmaf_dnn_session_run()` which picks
  CPU EP by default; CUDA EP is used automatically when libvmaf is
  built with `-Denable_cuda=true` and the graph is supported. The
  placeholder is single-Conv-plus-Sigmoid so any EP works.
- **Score interpretation**: with the placeholder, `saliency_mean` is
  ~0.5 regardless of input — the placeholder exists to lock down the
  pipeline, not to score quality. Once the real weights land, the
  score becomes content-dependent.

## How the placeholder is regenerated

```bash
python scripts/gen_mobilesal_placeholder_onnx.py
# wrote model/tiny/mobilesal.onnx (330 bytes, sha256=f1226...)
# wrote model/tiny/mobilesal.json
# updated model/tiny/registry.json
```

Re-running on the same `numpy` / `onnx` versions produces byte-identical
output. CI verifies the sha256 against `registry.json` before
`CreateSession`.

## Related

- [`lpips_sq.md`](lpips_sq.md) — sister full-reference DNN extractor;
  shares the YUV → ImageNet-RGB plumbing.
- [`../roadmap.md`](../roadmap.md) §2.3 — Wave 1 MobileSal scope.
- [ADR-0218](../../adr/0218-mobilesal-saliency-extractor.md) — design
  notes (smoke-only placeholder, scoring-vs-encoder split, scalar-vs-map
  output).
- [ADR-0257](../../adr/0257-mobilesal-real-weights-deferred.md) —
  blocker decision deferring the T6-2a-followup real-weights swap.
- [Research-0053](../../research/0053-mobilesal-real-weights-blocker.md)
  — upstream survey, licence analysis, and alternatives walk.
- [ADR-0042](../../adr/0042-tinyai-docs-required-per-pr.md) — tiny-AI
  doc-substance rule this page satisfies.
