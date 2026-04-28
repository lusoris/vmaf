# Tiny AI — training

Everything happens through `vmaf-train`, the typer CLI in
[`ai/`](../../ai/). Five subcommands: `extract-features`, `fit`, `export`,
`eval`, `register`.

## Install

```bash
pip install -e ai
# optional extras
pip install -e 'ai[tune,viz]'
```

This pulls `torch>=2.4,<2.6` + `lightning>=2.4,<2.5`. If you have a
GPU-capable PyTorch wheel installed separately, the extras will not
reinstall it.

## Dataset acquisition

Datasets are NOT committed. `ai/src/vmaf_train/data/datasets.py` knows
five canonical sources and caches them under
`${VMAF_DATA_ROOT:-~/.cache/vmaf-train}/datasets/<name>/`. Each dataset
ships a `manifests/<name>.yaml` SHA-256 manifest so downloads are
verifiable.

| Dataset | Use | License | Purpose |
| --- | --- | --- | --- |
| Netflix Public (NFLX) | C1, C2 | Netflix research | Same source as upstream `vmaf_v0.6.1` |
| KoNViD-1k | C2 | CC BY 4.0 | NR-friendly UGC clips with MOS |
| LIVE-VQC | C2 | Academic | NR validation |
| YouTube-UGC | C2 | CC BY 3.0 | Large-scale NR |
| BVI-DVC | C3 | Academic | Encoder distortion pairs for learned filters |

You are responsible for complying with each dataset's license. The
manifests only record hashes, not bytes.

## C1 — FR regressor walkthrough

```bash
# 1. Extract feature vectors from the NFLX pairs using the existing
#    libvmaf CPU backend.
vmaf-train extract-features \
    --dataset nflx \
    --output ai/data/nflx_features.parquet

# 2. Train a 2-layer MLP on the extracted features.
vmaf-train fit \
    --config ai/configs/fr_tiny_v1.yaml \
    --features ai/data/nflx_features.parquet \
    --output runs/fr_tiny_v1/

# 3. Export the trained weights to ONNX and validate roundtrip
#    (torch eval vs onnxruntime within atol=1e-5).
vmaf-train export \
    --checkpoint runs/fr_tiny_v1/last.ckpt \
    --output model/tiny/vmaf_tiny_fr_v1.onnx \
    --opset 17

# 4. Hold-out evaluation.
vmaf-train eval \
    --model model/tiny/vmaf_tiny_fr_v1.onnx \
    --features ai/data/nflx_features.parquet \
    --split test
# → PLCC, SROCC, RMSE vs MOS.

# 5. Write a sidecar and register into model/tiny/.
vmaf-train register \
    --model model/tiny/vmaf_tiny_fr_v1.onnx \
    --kind fr \
    --dataset nflx \
    --license CDLA-Permissive-2.0 \
    --train-commit "$(git rev-parse HEAD)"
```

The sidecar `model/tiny/vmaf_tiny_fr_v1.json` pins:

```json
{
  "schema_version": 1,
  "name": "vmaf_tiny_fr_v1",
  "kind": "fr",
  "onnx_opset": 17,
  "input_name": "features",
  "output_name": "score",
  "input_normalization": { "mean": [...], "std": [...] },
  "expected_output_range": [0.0, 100.0],
  "dataset": "nflx",
  "train_commit": "…",
  "train_config_hash": "sha256:…",
  "license": "CDLA-Permissive-2.0"
}
```

## C1 (Netflix corpus) — runnable training prep

Once the local Netflix corpus exists at `.workingdir2/netflix/` (see
[training-data.md](training-data.md) for the layout and ADR-0199 for
scope), the prep stack under [`ai/data/`](../../ai/data/) and
[`ai/train/`](../../ai/train/) replaces the parquet-driven flow above
with a runnable end-to-end pipeline. ADR-0203 records the
implementation decisions (distillation source, val-split policy,
architecture roster, cache layout).

### One-command training

```bash
# Builds libvmaf if you haven't yet:
meson setup build -Denable_cuda=false -Denable_sycl=false
ninja -C build

# Defaults: arch=mlp_small, val-source=Tennis, epochs=10.
# The first invocation pre-warms the per-clip cache at
# $VMAF_TINY_AI_CACHE (default ~/.cache/vmaf-tiny-ai); subsequent runs
# only re-train.
python ai/train/train.py \
    --data-root .workingdir2/netflix \
    --model-arch mlp_small \
    --epochs 30 \
    --batch-size 256 \
    --lr 1e-3 \
    --out-dir runs/tiny_nflx
```

Or, equivalently, via the wrapper script:

```bash
bash ai/scripts/run_training.sh
```

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--data-root` | `.workingdir2/netflix` | Directory with `ref/` and `dis/`. |
| `--model-arch` | `mlp_small` | One of `linear`, `mlp_small`, `mlp_medium`. |
| `--epochs` | 10 | `0` runs the smoke-export path and exits. |
| `--batch-size` | 256 | SGD batch size. |
| `--lr` | 1e-3 | Adam learning rate. |
| `--out-dir` | `runs/tiny_nflx` | ONNX checkpoints land at `<out-dir>/<arch>_epoch<n>.onnx` and `<arch>_final.onnx`. |
| `--val-source` | `Tennis` | Source name held out for validation. |
| `--max-pairs` | unset | Cap on (ref, dis) pairs (smoke / debugging). |
| `--no-export-onnx` | unset | Skip per-epoch ONNX dump (final still written). |
| `--assume-dims WxH` | unset | For tests / mock corpora with non-1080p YUVs. |

### Architectures

| Arch | Layers | Params (feature_dim=6) |
|---|---|---|
| `linear` | `Linear(6, 1)` | 7 |
| `mlp_small` | `Linear(6,16) -> ReLU -> Linear(16,8) -> ReLU -> Linear(8,1)` | 257 |
| `mlp_medium` | `Linear(6,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)` | 2 561 |

### Expected runtime + GPU requirements

| Phase | CPU-only (8-core) | CUDA (RTX 3060) |
|---|---|---|
| Cache warm (full corpus, 70 pairs) | 30–60 min (libvmaf-bound) | 5–8 min (libvmaf CUDA backend) |
| Train 30 epochs `mlp_small` | 1–2 min | <30 s |
| Train 30 epochs `mlp_medium` | 2–4 min | <60 s |
| ONNX export | <1 s | <1 s |

The cache is the bottleneck on first run; subsequent training runs
re-use the JSON cache and skip libvmaf entirely. To force a re-extract,
delete `$VMAF_TINY_AI_CACHE`.

### Evaluation

```bash
python -c "
from pathlib import Path
import numpy as np
from ai.train.dataset import NetflixFrameDataset
from ai.train.eval import evaluate

val = NetflixFrameDataset(Path('.workingdir2/netflix'), split='val')
X, y = val.numpy_arrays()
report = evaluate(
    features=X,
    targets=y,
    onnx_path=Path('runs/tiny_nflx/mlp_small_final.onnx'),
    out_path=Path('runs/tiny_nflx/eval_report.json'),
)
print(report)
"
```

The JSON report contains `n_samples`, `plcc`, `srocc`, `krocc`, `rmse`,
`latency_ms_p50_per_clip`, `latency_ms_p95_per_clip`, `model`,
`feature_dim`. Latency is measured against a synthetic 240-frame clip
on the CPU EP — the whole point of the tiny model is being meaningfully
faster than the SVR.

### Smoke command

CI runs only the `--epochs 0` smoke test (the real corpus and a
real training run are not GitHub-runner-friendly). The smoke command
lives in `ai/tests/test_train_smoke.py` and the equivalent shell
invocation is:

```bash
python ai/train/train.py \
    --epochs 0 \
    --data-root /tmp/mock_corpus \
    --assume-dims 16x16 \
    --val-source BetaSrc \
    --out-dir /tmp/tiny_smoke
```

This exports an initial-weights ONNX file without touching the real
corpus or invoking libvmaf, and is the documented reproducer in the PR
template.

## C2 — NR metric

Same flow, different config: [`ai/configs/nr_mobilenet_v1.yaml`](../../ai/configs/nr_mobilenet_v1.yaml).
`extract-features` is replaced by a direct frame loader
([`frame_loader.py`](../../ai/src/vmaf_train/data/frame_loader.py)) that
feeds ffmpeg-decoded tensors into training.

## C3 — Learned filter

[`ai/configs/filter_residual_v1.yaml`](../../ai/configs/filter_residual_v1.yaml)
trains a residual CNN where the model is clamped to `x + residual` in
normalized space. Target is BVI-DVC encoder-distortion pairs.

## Determinism

`vmaf-train fit` seeds Python, NumPy, and PyTorch with the config's `seed`
field and sets Lightning's `deterministic=True`. Combined with
`train_commit + train_config_hash + dataset_manifest_sha + seed` the output
weights are reproducible to within float-rounding nondeterminism (which CI
will flag as a regression when it exceeds a tight allclose).

## Hyperparameter sweeps

The `ai[tune]` extra pulls in Optuna + Ray Tune. A first-class `vmaf-train
tune` subcommand is planned but not yet wired — for now, drive the sweep
from a standalone script that imports the training entry point and
dispatches configs through Optuna.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `extract-features` is slow | libvmaf CPU-only | rebuild with `-Denable_cuda=true` and rerun |
| `fit` OOM | batch size too big for GPU | edit `ai/configs/*.yaml` `batch_size`, or drop `precision` to `16-mixed` |
| Export roundtrip fails atol=1e-5 | op using `float16` with a value near `inf` | retrain in `float32` end-to-end, or tighten clamping |
