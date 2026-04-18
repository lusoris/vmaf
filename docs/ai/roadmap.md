# Tiny-AI — roadmap

Where the tiny-AI surface is going. Four capabilities already in-tree
(see [overview.md](overview.md)); this roadmap captures the *expansion* —
what we're adding that the current scope ([ADR-0020](../adr/0020-tinyai-four-capabilities.md)
– [ADR-0023](../adr/0023-tinyai-user-surfaces.md)) doesn't cover.

> **Status.** Wave 1 locked by
> [ADR-0107](../adr/0107-tinyai-wave1-scope-expansion.md) (supersedes
> [ADR-0036](../adr/0036-tinyai-wave1-scope-expansion.md), original
> 2026-04-17 popup approval). Subsequent waves are non-binding; they
> document direction.

## 1. Where we are

Shipped and wired:

- **Training** — `ai/` (PyTorch + Lightning), `vmaf-train` CLI.
- **Inference** — `libvmaf/src/dnn/` (ONNX Runtime C API behind a 67-op
  allowlist, ≤ 50 MB model cap, path-hardened loader).
- **C API** — `vmaf_use_tiny_model()`, `VmafDnnSession` open/run/close.
- **CLI** — `vmaf --tiny-model PATH --tiny-device {auto|cpu|cuda|openvino|rocm}`.
- **FFmpeg** — `ffmpeg-patches/0001` adds tiny-model options to
  `vf_libvmaf`; `ffmpeg-patches/0002` adds the `vmaf_pre` learned-filter
  filter.

Not shipped yet:

- **No checkpoints.** `model/tiny/` is empty. The whole surface is a
  loaded gun with no bullets.
- **No GPU-parity CI.** Cross-execution-provider variance is verified
  manually.
- **No model signing verification** at load time. The `--tiny-model-verify`
  flag is stubbed.

## 2. Wave 1 — what lands next

All four sub-lists below were approved in the popup that produced
[ADR-0107](../adr/0107-tinyai-wave1-scope-expansion.md) (paraphrased
re-statement of the original [ADR-0036](../adr/0036-tinyai-wave1-scope-expansion.md)).
Order is rough; "ship baselines" is the blocker on everything else.

### 2.1 Ship baselines

Without actual ONNX files the surfaces are cosmetic.

| Model | Role | Target |
| --- | --- | --- |
| `fr_regressor_v1.onnx` | C1 FR | Match or beat `vmaf_v0.6.1` PLCC on NFLX public |
| `nr_metric_v1.onnx` | C2 NR | Useful on live-encode + UGC without a reference |
| `learned_filter_v1.onnx` | C3 filter | Residual luma denoise, ≤ +2% encode time |

First training run also exercises the `vmaf-train` CLI end-to-end and
proves the sidecar-JSON round-trip.

### 2.2 LPIPS-SqueezeNet as an FR baseline

**Why.** Industry-standard perceptual FR. Complements our homegrown C1
with an externally-validated reference point. SqueezeNet variant fits
comfortably under the size cap (~2.5M params + ~1.25M frozen features).

**Integration.** New feature extractor under `libvmaf/src/feature/` that
calls `vmaf_dnn_session_*`. Emits `lpips_sq` per frame alongside VMAF's
own composite features.

**ONNX notes.** Stock convs + global pooling, static input shape.
Exports cleanly at opset 17. No custom ops. Upstream reference:
[`richzhang/PerceptualSimilarity`](https://github.com/richzhang/PerceptualSimilarity).

### 2.3 MobileSal → saliency-weighted VMAF *and* encoder ROI

**Why.** The same ~2.5M saliency model feeds two surfaces:

1. **Scoring side** — multiply the saliency map into the per-pixel
   residual before spatial pooling in existing feature extractors. This
   is the SVMAF variant published in academic work but never shipped.
2. **Encoder side** — emit a per-CTU QP-offset map consumed by
   `x265 --qpfile` or the SVT-AV1 ROI API. Big bitrate win at fixed
   subjective quality.

**Integration.** Two outputs from one model:

- A new `vmaf_saliency_weighted` feature extractor inside libvmaf.
- A new CLI `tools/vmaf-roi` that writes an encoder-native sidecar
  (format matches whatever encoder we're feeding).

**ONNX notes.** MobileSal is MobileNet-V3-based, simple to export.

### 2.4 Per-shot CRF predictor + TransNet V2 shot boundaries

**Why.** Content-adaptive encoding without an ML framework in the
encoder. Smallest models in this entire roadmap (< 1M each) with
disproportionate bitrate-at-quality savings.

**Two-step pipeline**:

1. **Shot boundaries** — `TransNet V2` (~1M) produces per-frame
   shot-change scores → list of shot timestamps.
2. **Per-shot CRF** — a small CNN/MLP takes a downsampled per-shot
   thumbnail + classical features (motion energy, spatial complexity)
   and predicts the CRF that hits target VMAF on that shot.

**Integration.** Standalone CLI (`tools/vmaf-perShot`) that writes an
encoder-ingestible sidecar. Does **not** run inside libvmaf — its output
is a parameter hint, not a quality score.

## 3. FFmpeg / encoder expansion

Approved slots that the current `ffmpeg-patches/` don't fill:

### 3.1 `vmaf_pre` extension — 10-bit + chroma

**Current.** Luma-8bit only, chroma passes through untouched.

**Expansion.** Accept `yuv420p10le` / `yuv422p10le` / `yuv444p10le`; run
the learned filter on chroma planes too (either a single 3-channel
model or three single-channel sessions). This is where the real bitrate
wins live — HDR content and chroma-heavy sources are exactly where
classical pre-filters leave budget on the table.

**ONNX notes.** Input tensor becomes `[1, C, H, W]` with `C ∈ {1, 2, 3}`.
Requires touching `tensor_io.c` to normalize across bit depths (the
`luma8` helper assumes 8-bit).

### 3.2 New `vmaf_post` filter (post-reconstruction NR scoring)

**Why.** Today we score the source pair (reference + distorted). A post
filter lets us score the *actually-decoded* stream in an ffmpeg pipeline,
using the C2 NR model. Shares the backbone with the in-tree NR metric.

**Integration.** New `ffmpeg-patches/0004-add-vmaf_post-filter.patch`
with a filter mirroring `vmaf_pre`'s shape — frame-in → score-out (no
frame-out — it's measurement-only).

### 3.3 FastDVDnet temporal pre-filter

**Why.** Published temporal denoise CNN (~2.5M, 5-frame window).
Denoise-before-encode is a well-validated bitrate lever for noisy /
grainy sources.

**Cost.** Needs a 5-frame buffer inside the filter; bigger lift than
per-frame filters. Deferred if Wave 1 is already too wide.

**Integration.** New `vmaf_pre_temporal` filter, or a mode flag on
`vmaf_pre`.

## 4. Op-allowlist expansion — bounded `Loop` / `If`

**Decision.** Whitelist `Loop` and `If` **with a bounded-iteration
guard**: reject models whose `Loop` `trip_count` attribute is missing
or whose inferred upper bound exceeds a configurable cap (default
1024). Published transformer / optical-flow architectures that target
ONNX export always have bounded loops; unbounded loops are a sandbox
risk (infinite compute, adversarial model).

**Unlocks.**

- **MUSIQ** (~27M) NR transformer — multi-scale attention.
- **RAFT-Small** (~1M) optical flow — iterative GRU update.
- **Small VLMs** (SmolVLM 256M family) — transformer decoder.

**Implementation.** Extend `libvmaf/src/dnn/op_allowlist.c`:

1. Add `Loop` and `If` to the allowed set.
2. During `model_loader` graph walk, if a `Loop` node is present, read
   its `trip_count` input. If that input is a graph constant, verify
   ≤ cap. If it's computed from inputs, reject.
3. Log the bound at load time for operators.

**Non-goal.** We are not adding `Scan`, which has more expressive
iteration semantics and would need a much larger analysis pass.

## 5. MCP + LLM surfaces

### 5.1 `describe_worst_frames` MCP tool

**Why.** When VMAF says "frame 847 is bad," the user still has to open
the frame to see *why*. A local VLM closes that loop with plain English:
*"frame is underexposed in the foreground; mild banding on the sky
gradient."* Debugging affordance, not a scoring component.

**Implementation.** New method in `mcp-server/vmaf-mcp/`. Inputs: VMAF
JSON output path, N. Steps:

1. Pick the N frames with the largest VMAF delta from the per-frame
   scores.
2. Extract those frames as PNGs (reusing ffmpeg).
3. Run SmolVLM (~256M) locally with a prompt template that asks for
   artifact types + plausible causes.
4. Return a JSON list of `{frame_index, vmaf, caption}`.

**Model choice.** SmolVLM family. If the 256M variant misses, fall back
to Moondream2 (1.8B quantized Q4 fits in 4 GB VRAM).

**Sandbox.** VLM runs via ONNX Runtime under the extended allowlist
(§4). Absolute path resolution and ≤ 50 MB cap still apply; larger VLMs
will need the env-override `VMAF_MAX_MODEL_BYTES` raised at runtime.

## 6. Training-side items

Not in Wave 1 but called out here so they don't get forgotten:

- **`vmaf-train tune`** (Optuna) — already stubbed in [`training.md`](training.md).
- **CLIP-IQA pseudo-labeler** — offline bootstrap for NR datasets.
- **KADID-10k synthetic distortion pipeline** — classical augmentation.
- **Hyperparameter-tuning Ray backend** — once `tune` stabilizes.

## 7. Infrastructure items

- **GPU-parity CI** — CPU ↔ CUDA, CPU ↔ OpenVINO cross-device variance,
  as a required status check (≤ 1e-4 FP32, ≤ 1e-2 FP16 per
  [`inference.md`](inference.md)).
- **Sigstore verification** — wire the stubbed `--tiny-model-verify`
  flag through to `cosign verify-blob` on the sidecar bundle.
- **Model registry** — current sidecar JSON is informal. A tracked
  registry file under `model/tiny/registry.json` with SHA + Sigstore
  bundle path + license gives us an auditable manifest.

## 8. Out of scope (non-goals)

Not on the roadmap, for clarity:

- Training **inside libvmaf**. ML framework deps stay in `ai/` / Python.
- Adding a second inference runtime (TFLite, ggml). ONNX Runtime is the
  one runtime.
- Cloud-only / API-dependent models. Everything runs local.
- Models > 50 MB by default. Raise the cap per-invocation with
  `VMAF_MAX_MODEL_BYTES` when it genuinely helps.
- `Scan` and arbitrary control flow. See §4 non-goal.

## 9. Related documents

- [overview.md](overview.md) — the four existing capabilities.
- [training.md](training.md) — `vmaf-train` CLI and dataset flow.
- [inference.md](inference.md) — CLI / C API / ffmpeg surfaces.
- [benchmarks.md](benchmarks.md) — PLCC/SROCC/RMSE methodology.
- [security.md](security.md) — op allowlist and size cap (expanded by §4).
- [ADR-0107](../adr/0107-tinyai-wave1-scope-expansion.md) — this roadmap's
  authority (supersedes
  [ADR-0036](../adr/0036-tinyai-wave1-scope-expansion.md)).
