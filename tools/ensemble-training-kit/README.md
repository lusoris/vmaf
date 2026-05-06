# Ensemble training kit

Portable, one-command bundle for running the `fr_regressor_v2_ensemble_v1`
Phase-A corpus generation + 5-seed × 9-fold LOSO retrain + verdict
emission against an operator-supplied set of Netflix reference YUVs.

ADR: [ADR-0324](../../docs/adr/0324-ensemble-training-kit.md).
Parent design: [ADR-0303](../../docs/adr/0303-fr-regressor-v2-ensemble-prod-flip.md)
(production-flip gate definition), [ADR-0319](../../docs/adr/0319-ensemble-loso-trainer-real-impl.md)
(LOSO trainer real impl), [ADR-0321](../../docs/adr/0321-fr-regressor-v2-ensemble-full-prod-flip.md)
(full production flip + ONNX export script).

## What this kit does

- Generates a Phase-A canonical-6 JSONL corpus from operator-supplied
  reference YUVs by encoding with NVENC at the standard CQ grid,
  decoding back to raw YUV, and scoring with libvmaf (CUDA backend).
- Runs the 5-seed × 9-fold LOSO retrain via the in-tree LOSO trainer,
  emitting one `loso_seed{0..4}.json` per seed.
- Applies the ADR-0303 two-part production-flip gate (`mean(PLCC) ≥ 0.95`
  AND `max(PLCC) − min(PLCC) ≤ 0.005`) and writes either `PROMOTE.json`
  or `HOLD.json`.
- On PROMOTE, exports per-seed ONNX members fitted on the FULL corpus
  (no held-out fold) and tars verdict + ONNX exports + manifest for
  return transport to the lead user.

## Prerequisites

| Requirement     | Expected                                                                              |
|-----------------|---------------------------------------------------------------------------------------|
| OS              | Linux (Ubuntu 22.04+ / Fedora 40+ / Arch / similar)                                   |
| GPU             | NVIDIA, ≥ 6 GB VRAM free, driver supporting CUDA toolkit ≥ 12.x                       |
| ffmpeg          | Built with `--enable-libnvenc` (`ffmpeg -encoders \| grep nvenc` shows `h264_nvenc`)  |
| libvmaf (CUDA)  | `libvmaf/build-cuda/tools/vmaf` — `meson setup build-cuda -Denable_cuda=true && ninja -C build-cuda` |
| Python          | ≥ 3.12 with `torch` (CUDA-enabled), `numpy`, `pandas`, `onnx`, `onnxruntime`, `scipy`, `scikit-learn` |
| Reference YUVs  | One per source, as raw `.yuv` files in a single directory                             |
| Free disk       | ≥ 5 GB for Phase-A intermediate mp4/yuv + ≥ 1 GB under the run output dir             |

The script `01-prereqs.sh` checks each of these and fails loudly with a
remediation hint if anything is missing. See "Troubleshooting" below.

QSV (Intel Arc / iHD) is **optional** — if `vainfo` does not surface
the iHD driver, the corpus generator silently skips QSV encoders and
runs NVENC-only. The model trains successfully on NVENC-only data;
the `encoder` one-hot collapses onto a single column.

## How to run

```bash
# One-command full pipeline (assumes prereqs are met; checks them first).
bash tools/ensemble-training-kit/run-full-pipeline.sh \
    --ref-dir /path/to/netflix/ref
```

Optional flags:

```bash
bash tools/ensemble-training-kit/run-full-pipeline.sh \
    --ref-dir   /path/to/netflix/ref \
    --encoders  h264_nvenc,hevc_nvenc \
    --cqs       19,25,31,37 \
    --out-dir   /path/to/run_output \
    --seeds     0,1,2,3,4
```

Defaults: `--encoders=h264_nvenc`, `--cqs=19,25,31,37`,
`--out-dir=$REPO_ROOT/runs/ensemble_v2_real`, `--seeds=0,1,2,3,4`.

The script exits with rc=0 on PROMOTE, rc=1 on HOLD, rc=2 on
prerequisite or input error.

### Step-by-step (manual)

The orchestrator is just five shell scripts chained together. Run them
individually if you need to retry a single stage:

```bash
bash tools/ensemble-training-kit/01-prereqs.sh
REF_DIR=/path/to/netflix/ref bash tools/ensemble-training-kit/02-generate-corpus.sh
bash tools/ensemble-training-kit/03-train-loso.sh
bash tools/ensemble-training-kit/04-validate.sh
bash tools/ensemble-training-kit/05-bundle-results.sh
```

Each step honours the same env vars (`REF_DIR`, `OUT_DIR`,
`CORPUS_JSONL`, `LIBVMAF_BIN`, `ENCODERS`, `CQS`).

## What you get back

After a successful run:

```
$OUT_DIR/
├── PROMOTE.json (or HOLD.json)        # ADR-0303 verdict file
├── loso_seed0.json … loso_seed4.json  # per-seed LOSO artefacts
├── exports/                           # PROMOTE only — per-seed ONNX members
│   ├── fr_regressor_v2_ensemble_v1_seed0.onnx
│   ├── fr_regressor_v2_ensemble_v1_seed0.json
│   └── … (× 5 seeds)
├── logs/                              # per-seed stdout/stderr logs
└── manifest.json                      # sha256 + sizes of every artefact
```

The bundle script tars `$OUT_DIR/` whole into
`lawrence-ensemble-results-<ts>.tar.gz`. The verdict file's structure
is documented in [`docs/ai/ensemble-v2-real-corpus-retrain-runbook.md`](../../docs/ai/ensemble-v2-real-corpus-retrain-runbook.md).

### `PROMOTE.json` shape

```json
{
  "schema_version": 1,
  "verdict": "PROMOTE",
  "generated_at_utc": "2026-05-06T...",
  "seeds": [0, 1, 2, 3, 4],
  "gate": {
    "passed": true,
    "mean_plcc": 0.997,
    "plcc_spread": 0.001,
    ...
  },
  "corpus": { "sha256": "...", "yuv_count": 9, ... },
  "recommendation": "flip seeds smoke->false in model/tiny/registry.json (...)",
  "adr": "ADR-0309",
  "parent_adr": "ADR-0303"
}
```

## Troubleshooting

### `ffmpeg present but lacks h264_nvenc encoder`

Distro packages occasionally strip NVENC. Install ffmpeg from a build
that links the NVIDIA Video Codec SDK headers (`nv-codec-headers`),
or download a static build from `johnvansickle.com/ffmpeg/`.

Verify with `ffmpeg -encoders | grep h264_nvenc`.

### `libvmaf-CUDA binary not found`

Build it from the repo root:

```bash
meson setup build-cuda -Denable_cuda=true -Denable_sycl=false
ninja -C build-cuda
ls libvmaf/build-cuda/tools/vmaf
```

If meson cannot find nvcc, set `CUDA_HOME=/opt/cuda` (or wherever the
toolkit lives) and re-run setup. CUDA toolkit ≥ 12.0 is required.

### `torch.cuda.is_available() is False`

Re-install torch with the CUDA build matching your driver:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

(Pick the channel matching your CUDA runtime: `cu121`, `cu124`, etc.)

### `GPU has only N MiB free`

Close other CUDA workloads, then re-run. The training pipeline needs
~6 GiB peak (libvmaf scoring shares the GPU with torch).

### Encode loop produces 0 rows

The `hw_encoder_corpus.py` source assumes 1920×1080@25fps `yuv420p` by
default; override with `WIDTH=`, `HEIGHT=`, `PIX_FMT=`, `FRAMERATE=`
before invoking step 02 if your YUVs differ.

### LOSO step reports `corpus has only N rows; need >=100`

Check `runs/phase_a/full_grid/per_frame_canonical6.jsonl` exists and
has the expected ~5640 rows for a 9-source × 4-CQ NVENC-only run. If
empty, step 02 silently skipped every source; re-run with the
`--encoders` / `--cqs` flags spelled correctly.

## How to send back the results

The lead user needs **only** the bundled tarball:

```
lawrence-ensemble-results-<ts>.tar.gz
```

Untar to inspect contents; the manifest at `manifest.json` lists every
file with sha256 + size. On PROMOTE, the lead user's follow-up PR
re-checksums the per-seed ONNX exports and updates
`model/tiny/registry.json` to point at them.

Do **not** include the original `.yuv` reference files — they're not
part of the kit's output and are typically several GiB.

## Generating a portable bundle (lead user only)

For shipping the kit to a collaborator who doesn't have the vmaf fork
checked out:

```bash
bash tools/ensemble-training-kit/make-distribution-tarball.sh \
    /tmp/vmaf-ensemble-training-kit-$(date -u +%Y%m%d).tar.gz
```

Output is a self-contained tarball (~ < 50 MiB) with the kit + every
in-repo script it invokes + the runbook, no model weights, no test
data, no build artefacts. The collaborator extracts and treats the
extracted directory as `$REPO_ROOT`.
