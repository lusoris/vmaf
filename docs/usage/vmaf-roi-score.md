# vmaf-roi-score — saliency-weighted VMAF scoring

`vmaf-roi-score` is a fork-local Python tool that produces a
**region-of-interest VMAF score** by combining a standard full-frame
VMAF run with a saliency-masked variant. Useful when a downstream
caller wants to weight quality measurements toward salient regions
(faces, focal subjects) rather than treat every pixel as equally
important.

> **Don't confuse this with `vmaf-roi`.** That binary
> (`libvmaf/tools/vmaf_roi.c`, ADR-0247) consumes a saliency map and
> emits **encoder QP-offset sidecars** for x265 / SVT-AV1. `vmaf-roi-score`
> is the *scoring*-side counterpart — same model lineage, different
> output.
>
> | Tool | Output | Surface |
> |---|---|---|
> | `vmaf-roi` (ADR-0247) | per-CTU QP offsets (encoder steering) | C binary |
> | `vmaf-roi-score` (ADR-0288) | saliency-weighted VMAF score | Python tool |

## What it computes

```text
roi_vmaf = (1 - w) * vmaf_full + w * vmaf_masked
```

- `vmaf_full` — pooled VMAF on the full frame (standard libvmaf run).
- `vmaf_masked` — pooled VMAF on a saliency-masked variant of the
  distorted YUV: low-saliency pixels are replaced with reference-
  matching pixels, so the masked run's score is dominated by the
  salient region.
- `w ∈ [0, 1]` — caller-controlled weight. `w=0` returns the standard
  full-frame VMAF unchanged; `w=1` returns the masked VMAF only.

## What it explicitly does **not** do

- **It is not "true" per-pixel saliency-weighted VMAF.** The masked
  run is a pixel substitution, not a per-feature weight. A salient
  region surrounded by a "perfect-match" zone scores differently than
  the same region under a *zero-weight* zone — VMAF's edge-sensitive
  features (motion, ADM) react to the boundary between the salient
  region and the substituted background. Option A (per-pixel feature
  pooling weighted by saliency in libvmaf C code) is tracked as
  deferred work in ADR-0288.
- **It makes no claim about MOS correlation.** This tool is a
  prototype that exposes a saliency-weighted *signal*; whether that
  signal tracks subjective quality better than uniform VMAF is an
  open research question. Validation against a labelled MOS dataset
  is a separate exercise — see Research-0063 §"What we deliberately
  don't measure".

## Install

```bash
pip install -e tools/vmaf-roi-score
# Optional, for ONNX inference once T6-2c lands:
pip install -e 'tools/vmaf-roi-score[runtime]'
```

The `vmaf` binary must be on `$PATH` (or pass `--vmaf-bin` explicitly).

## Quick start (synthetic-mask smoke)

The synthetic-mask path skips ONNX inference and scores the same
distorted YUV twice. Use it to verify your install + the combine math
without depending on the deferred saliency materialiser.

```bash
vmaf-roi-score \
    --reference path/to/ref.yuv --distorted path/to/dis.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --synthetic-mask 0.5 --weight 0.5
```

Output (JSON to stdout):

```json
{
  "schema_version": 1,
  "vmaf_full": 87.5,
  "vmaf_masked": 87.5,
  "weight": 0.5,
  "vmaf_roi": 87.5,
  "model": "vmaf_v0.6.1",
  "saliency_model": "synthetic",
  "reference": "path/to/ref.yuv",
  "distorted": "path/to/dis.yuv"
}
```

In synthetic mode `vmaf_full == vmaf_masked` by construction, so
`vmaf_roi` equals both regardless of `--weight`. This is the contract
the smoke tests verify — once the real saliency-mask materialiser
lands, the two scalars will diverge.

## Saliency-model mode (deferred — currently exits 64)

The `--saliency-model` path is wired and validated but the YUV
reader/writer + ONNX inference loop is **scaffolded only** in this
PR (see ADR-0288 §Implementation phasing, T6-2c). The CLI surfaces a
clear error today rather than silently degrading:

```bash
vmaf-roi-score \
    --reference ref.yuv --distorted dis.yuv \
    --width 1920 --height 1080 \
    --saliency-model model/tiny/saliency_student_v1.onnx
# stderr: vmaf-roi-score: --saliency-model is scaffolded but mask
#         materialisation is not wired yet (see ADR-0288). Use
#         --synthetic-mask for the combine-math smoke today.
# exit code: 64
```

When PR #359 (`saliency_student_v1`) merges and T6-2c follow-up lands,
this path will materialise the masked YUV and score it for real.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--reference PATH` | (required) | raw reference YUV |
| `--distorted PATH` | (required) | raw distorted YUV |
| `--width N`, `--height N` | (required) | YUV dimensions |
| `--pix-fmt FMT` | `yuv420p` | ffmpeg pix_fmt; mapped to libvmaf's `--pixel_format` |
| `--saliency-model PATH` | none | path to saliency ONNX (mutually exclusive with `--synthetic-mask`) |
| `--synthetic-mask FILL` | none | constant-value mask in `[0, 1]` (testing only; mutually exclusive with `--saliency-model`) |
| `--weight W` | `0.5` | saliency-masked component weight in `[0, 1]` |
| `--model NAME` | `vmaf_v0.6.1` | VMAF model version passed through to `vmaf` |
| `--vmaf-bin PATH` | `vmaf` | location of the libvmaf CLI binary |
| `--output PATH` | stdout | write JSON result to this path |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 64 | feature scaffolded but not yet wired (`--saliency-model` path until T6-2c) |
| 65 | `vmaf` ran but produced JSON missing the pooled scalar |
| other | passed through from the underlying `vmaf` invocation |

## See also

- [ADR-0288](../adr/0288-vmaf-roi-saliency-weighted.md) — the design ADR
  for this tool.
- [Research-0063](../research/0063-vmaf-roi-saliency-weighted.md) —
  Option A vs B vs C decision matrix.
- [`docs/usage/vmaf-roi.md`](vmaf-roi.md) — the encoder-steering
  sibling tool (different surface, related model).
- [`docs/ai/models/saliency_student_v1.md`](../ai/models/saliency_student_v1.md)
  — the saliency model this tool will consume once T6-2c lands.
