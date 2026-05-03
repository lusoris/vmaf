# `vmaf-tune` — quality-aware encode automation harness

`vmaf-tune` is a fork-added Python tool
([ADR-0237](../adr/0237-quality-aware-encode-automation.md),
[Research-0044](../research/0044-quality-aware-encode-automation.md))
that drives FFmpeg over an encoder parameter grid, scores each encode
with [`vmaf`](cli.md), and emits a JSONL corpus of
`(source, encoder, params, bitrate, vmaf)` rows.

This doc covers **Phase A** of the six-phase roadmap: a `libx264` grid
sweep that produces the corpus the later phases consume. Phases B (target-VMAF
bisect), C (per-title CRF predictor), D (per-shot dynamic CRF), E (Pareto ABR
ladder) and F (MCP tools) are not implemented yet — see ADR-0237.

## Pipeline

```text
ref.yuv ──► vmaf-tune corpus ──► encode (libx264) ──► vmaf score ──► corpus.jsonl
              │
              └─► encodes are written to --encode-dir, deleted post-score
                  unless --keep-encodes
```

## Install

The tool ships under `tools/vmaf-tune/` as a standalone Python package.
Phase A has zero runtime dependencies beyond the standard library.

```shell
pip install -e tools/vmaf-tune
# or run directly from the checkout:
python tools/vmaf-tune/vmaf-tune --help
```

External binaries required at runtime:

- `ffmpeg` with `--enable-libx264` on `PATH` (or `--ffmpeg-bin`).
- `vmaf` (this fork's CLI, built via meson) on `PATH` (or `--vmaf-bin`).

## Quick start

Generate a 6-cell corpus row over `(medium, slow) × (22, 28, 34)` for one
1080p source clip:

```shell
vmaf-tune corpus \
    --source ref.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p \
    --framerate 24 --duration 10 \
    --preset medium --preset slow \
    --crf 22 --crf 28 --crf 34 \
    --output corpus.jsonl
```

`--source` is repeatable — pass one flag per source clip. The grid is the
Cartesian product of `--preset × --crf`.

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--source PATH` | — | Required. Repeatable for multi-source sweeps. |
| `--width / --height` | — | Required. Source resolution. |
| `--pix-fmt PFMT` | `yuv420p` | Forwarded to ffmpeg `-pix_fmt`. |
| `--framerate F` | `24.0` | Source framerate. |
| `--duration S` | `0` | Source duration in seconds (used for bitrate calc). |
| `--encoder NAME` | `libx264` | Phase A wires `libx264` only. |
| `--preset P` | — | Required. Repeatable. x264 preset name. |
| `--crf N` | — | Required. Repeatable. x264 CRF integer. |
| `--output PATH` | `corpus.jsonl` | JSONL destination. |
| `--encode-dir PATH` | `.workingdir2/encodes` | Scratch dir; gitignored by convention. |
| `--keep-encodes` | off | Retain encoded files after scoring. |
| `--vmaf-model NAME` | `vmaf_v0.6.1` | Forwarded to `vmaf --model`. |
| `--ffmpeg-bin PATH` | `ffmpeg` | Override the ffmpeg binary. |
| `--vmaf-bin PATH` | `vmaf` | Override the vmaf binary. |
| `--no-source-hash` | off | Skip `src_sha256` (faster on large YUVs; loses provenance). |

## Corpus JSONL schema

Each row is one JSON object on its own line. The full key list is
exported as `vmaftune.CORPUS_ROW_KEYS` for programmatic consumers and
versioned via `vmaftune.SCHEMA_VERSION` (currently `1`). Bumping the
schema is a coordinated change with Phase B/C; do not edit row shape
without bumping the version.

| Key | Type | Description |
|---|---|---|
| `schema_version` | int | Currently `1`. |
| `run_id` | str | Per-row UUID4 hex. |
| `timestamp` | str | UTC ISO-8601 (seconds precision). |
| `src` | str | Path to the reference YUV. |
| `src_sha256` | str | SHA-256 of the reference (empty if `--no-source-hash`). |
| `width` / `height` | int | Source dimensions. |
| `pix_fmt` | str | Source pixel format. |
| `framerate` | float | Source framerate. |
| `duration_s` | float | Source duration in seconds. |
| `encoder` | str | Codec adapter name (e.g. `libx264`). |
| `encoder_version` | str | Detected encoder version (e.g. `libx264-164`). |
| `preset` | str | Encoder preset. |
| `crf` | int | Quality knob value. |
| `extra_params` | list[str] | Additional encoder argv (Phase A: `[]`). |
| `encode_path` | str | Path to encoded file (empty if not retained). |
| `encode_size_bytes` | int | Encoded file size. |
| `bitrate_kbps` | float | `(encode_size_bytes × 8 / 1000) / duration_s`. |
| `encode_time_ms` | float | Wall-clock encode time. |
| `vmaf_score` | float | Pooled-mean VMAF (`NaN` if scoring skipped/failed). |
| `vmaf_model` | str | Model version string (e.g. `vmaf_v0.6.1`). |
| `score_time_ms` | float | Wall-clock scoring time. |
| `ffmpeg_version` | str | Detected ffmpeg version. |
| `vmaf_binary_version` | str | Detected vmaf binary version. |
| `exit_status` | int | First non-zero of (encode, score) exit codes. |

### Example row

```json
{
  "schema_version": 1,
  "run_id": "0a3b1c8b...",
  "timestamp": "2026-05-03T16:00:00+00:00",
  "src": "ref.yuv",
  "src_sha256": "",
  "width": 1920, "height": 1080, "pix_fmt": "yuv420p",
  "framerate": 24.0, "duration_s": 10.0,
  "encoder": "libx264", "encoder_version": "libx264-164",
  "preset": "medium", "crf": 28,
  "extra_params": [],
  "encode_path": "",
  "encode_size_bytes": 845210,
  "bitrate_kbps": 676.168,
  "encode_time_ms": 4321.0,
  "vmaf_score": 92.41,
  "vmaf_model": "vmaf_v0.6.1",
  "score_time_ms": 1820.5,
  "ffmpeg_version": "6.1.1",
  "vmaf_binary_version": "3.0.0-lusoris.0",
  "exit_status": 0
}
```

## What Phase A does **not** do

- No target-VMAF bisect (Phase B).
- No per-title or per-shot CRF prediction (Phase C / D).
- No Pareto ABR ladder generation (Phase E).
- No MCP tools wiring (Phase F).
- Only `libx264` — `libx265` / `libsvtav1` / `libvpx-vp9` / `libvvenc`
  are next via the codec adapter interface in
  `tools/vmaf-tune/src/vmaftune/codec_adapters/`.

## Phase A.5 — opt-in `fast` recommend

The `fast` subcommand
([ADR-0276](../adr/0276-vmaf-tune-fast-path.md),
[Research-0060](../research/0060-vmaf-tune-fast-path.md)) is an
opt-in recommendation surface that combines three acceleration
levers — VMAF proxy via `fr_regressor_v2`, Bayesian search via
Optuna's TPE sampler, and GPU-accelerated VMAF for the verify step —
to replace the exhaustive grid for the *recommendation* use case.
The slow `corpus` path stays the canonical ground truth.

> **Status: scaffold only.** This PR ships the Optuna search loop,
> the smoke-mode synthetic predictor, the CLI subcommand, and the
> production-shape entry point. The real encode-extract-predict loop
> (real ffmpeg sample encode + canonical-6 extraction + ONNX
> inference + GPU verify) is a follow-up PR. Run with `--smoke` to
> exercise the pipeline end-to-end.

### Install

```shell
pip install -e 'tools/vmaf-tune[fast]'   # adds Optuna
```

The core install path stays zero-dependency; the `[fast]` extra is
strictly opt-in.

### Smoke-mode quick start

```shell
vmaf-tune fast --smoke --target-vmaf 92
```

Smoke mode runs Optuna over a synthetic x264-shaped CRF→VMAF curve.
No ffmpeg, no ONNX Runtime, no GPU is touched. Output is a single
JSON object:

```json
{
  "encoder": "libx264",
  "target_vmaf": 92.0,
  "recommended_crf": 18,
  "predicted_vmaf": 92.39,
  "predicted_kbps": 4121.09,
  "n_trials": 50,
  "smoke": true,
  "notes": "smoke mode — synthetic predictor; ..."
}
```

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--src PATH` | — | Source video. Required outside `--smoke`. |
| `--target-vmaf F` | `92.0` | Quality target on VMAF [0, 100] scale. |
| `--encoder NAME` | `libx264` | Codec adapter (Phase A.5: x264 only). |
| `--crf-lo N` | `10` | Lower bound of the CRF search. |
| `--crf-hi N` | `51` | Upper bound of the CRF search. |
| `--n-trials N` | `50` | Optuna TPE trial count. |
| `--time-budget-s N` | `300` | Soft wall-clock budget (advisory in scaffold). |
| `--smoke` | off | Synthetic predictor — exercises the pipeline without ffmpeg / ONNX. |

### Speedup model

Per Research-0060 §Speedup model:

| Combination | Speedup vs Phase A grid |
| --- | --- |
| Phase A grid (baseline) | 1× |
| `fast` (proxy + Bayesian + GPU verify) | ≈20–50× |
| `fast` + NVENC (lever C, follow-up) | ≈100–500× |

These are upper bounds. The production claim is gated on a
recommendation-quality benchmark against the slow grid.

### What's needed for production

The scaffold deliberately leaves the following as follow-up PR
work; flipping ADR-0276 from `Proposed` to `Accepted` requires all
of these:

1. Real `fr_regressor_v2.onnx` weights trained on the Phase A
   corpus (gated on PR #347 + corpus generation).
2. ONNX Runtime wiring for the inference call (the current scaffold
   exposes a `predictor=` injection seam for follow-up PRs).
3. Sample-chunk encode loop — encode a 5-second representative
   chunk per CRF, extract canonical-6 features, feed the proxy.
4. GPU verify pass — invoke `vmaf` with `--cuda` / `--vulkan` /
   `--sycl` (auto-detected) at the recommended CRF and report the
   proxy / verify gap.
5. NVENC / QSV / AMF auto-detection (lever C, ≈10× more speedup).
6. Per-shot parallelisation (lever D, integrates with TransNet V2
   and `vmaf-perShot`).
7. Recommendation-quality benchmark — for ≥3 sources, compare
   `fast` vs the slow grid at the recommended CRF; gate Acceptance
   on a small VMAF tolerance (≤ 1.0 VMAF gap median).

## Tests

```shell
pytest tools/vmaf-tune/tests/
```

The shipped suite mocks `subprocess.run` so it neither requires
`ffmpeg` nor a built `vmaf`. Real-binary integration coverage will land
when the codec adapter set widens.
