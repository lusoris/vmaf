# AGENTS.md — ai/

Orientation for agents working on the tiny-AI **training** side. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

Python package for training, exporting, and registering tiny-AI
checkpoints that are then consumed by [libvmaf/src/dnn/](../libvmaf/src/dnn/AGENTS.md)
at runtime. Stack: PyTorch + Lightning → ONNX.

```
ai/
  pyproject.toml          # package metadata (training-only deps)
  src/                    # vmaf-train CLI + model definitions + dataset loaders
  tests/                  # pytest unit tests
  configs/                # dataset manifests + training recipes
  lpips_export.py         # re-exports richzhang/PerceptualSimilarity → ONNX (example)
```

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **Boundary is `.onnx` + sidecar JSON on disk.** Training lives here,
  runtime lives in `libvmaf/src/dnn/`, and the two communicate only through
  files in `model/tiny/`. No imports cross this boundary.
- **Every shipped `.onnx` has a registry entry** in
  [`../model/tiny/registry.json`](../model/tiny/) with sha256, upstream
  source, license, and opset. See
  [ADR-0039](../docs/adr/0039-onnx-runtime-op-walk-registry.md).
- **ONNX opset**: export requests opset 17 but torch dynamo may emit 18
  (downconvert sometimes fails in `onnx.version_converter`). Record the
  emitted opset in the registry sidecar rather than failing the export.
- **ImageNet normalisation lives in the graph**, not in the C helper — for
  any ImageNet-family model, absorb the inverse ImageNet transform into
  the exported graph so the C side uses the shared
  `vmaf_tensor_from_rgb_imagenet()` helper unchanged. See
  [ADR-0041](../docs/adr/0041-lpips-sq-extractor.md).
- **Roundtrip-validate** every export against `onnxruntime` to atol=1e-5
  before committing. See [ADR-0021](../docs/adr/0021-training-stack-pytorch-lightning.md).
- **Docs**: every new model or training recipe ships a page under
  `docs/ai/` in the same PR. See
  [ADR-0042](../docs/adr/0042-tinyai-docs-required-per-pr.md).
- **Bisect-cache fixture is byte-stable** — `ai/testdata/bisect/` is
  the deterministic placeholder for the nightly `bisect-model-quality`
  workflow. Regenerate via `python ai/scripts/build_bisect_cache.py`
  with seeds `FEATURE_SEED=20260418` / `MODEL_SEED=20260419`. CI runs
  the same script with `--check` and asserts byte-equality before the
  bisect; toolchain bumps that change `pandas` / `pyarrow` / `onnx`
  serialisation will fail the workflow until the cache is regenerated
  and committed. See
  [ADR-0109](../docs/adr/0109-nightly-bisect-model-quality.md) +
  [Research-0001](../docs/research/0001-bisect-model-quality-cache.md).

## Governing ADRs

- [ADR-0020](../docs/adr/0020-tinyai-four-capabilities.md) — four capabilities (C1–C4).
- [ADR-0021](../docs/adr/0021-training-stack-pytorch-lightning.md) — PyTorch + Lightning training stack.
- [ADR-0023](../docs/adr/0023-tinyai-user-surfaces.md) — `vmaf-train` CLI as one of four surfaces.
- [ADR-0036](../docs/adr/0036-tinyai-wave1-scope-expansion.md) — Wave 1 scope (LPIPS, MobileSal, TransNet V2, …).
- [ADR-0039](../docs/adr/0039-onnx-runtime-op-walk-registry.md) — runtime op-allowlist + registry schema.
- [ADR-0041](../docs/adr/0041-lpips-sq-extractor.md) — LPIPS export pattern (ImageNet-in-graph).
- [ADR-0042](../docs/adr/0042-tinyai-docs-required-per-pr.md) — doc-substance rule.
- [ADR-0218](../docs/adr/0218-mobilesal-saliency-extractor.md) — MobileSal saliency extractor (T6-2a) ships a smoke-only synthetic ONNX placeholder under `model/tiny/mobilesal.onnx`; the C extractor binds tensors by name (`input` → `saliency_map`) so a real upstream MobileSal export drops in without C changes. Saliency-weighted FR features and the `tools/vmaf-roi` CTU sidecar are the T6-2b follow-up — do not bundle them into the T6-2a surface.
- [ADR-0109](../docs/adr/0109-nightly-bisect-model-quality.md) — nightly bisect workflow + synthetic placeholder cache.
- [ADR-0235](../docs/adr/0235-codec-aware-fr-regressor.md) — codec-aware FR regressor (`fr_regressor_v2`). `CODEC_VOCAB` in [`src/vmaf_train/codec.py`](src/vmaf_train/codec.py) is **closed and order-stable** — the index of each codec is the one-hot column index baked into trained ONNX. Adding a codec appends to the tuple and bumps `CODEC_VOCAB_VERSION`; reordering silently invalidates every shipped `fr_regressor_v2_*.onnx`. `FRRegressor(num_codecs=0)` must remain the v1 single-input contract — flipping the default would break every existing `model/tiny/fr_regressor_v1.onnx` consumer. Feature-dump scripts emit a `codec` column tagged at the call site (BVI-DVC: `"x264"`, Netflix Public: `"unknown"`); never silently default to a codec that doesn't match what the script actually encoded.

## Netflix-corpus training prep (ADR-0199 / ADR-0203)

The top-level [`ai/data/`](data/) and [`ai/train/`](train/) packages
(distinct from the `vmaf_train` package under `src/`) host the
runnable Netflix-corpus prep stack:

- [`ai/data/netflix_loader.py`](data/netflix_loader.py) — pair distorted
  YUVs with their ref by parsing the Netflix ladder filename
  convention. `iter_pairs(data_root, *, sources=, max_pairs=,
  assume_dims=)` is the only public surface.
- [`ai/data/feature_extractor.py`](data/feature_extractor.py) — wraps
  the libvmaf CLI in JSON mode. Defaults to `build/tools/vmaf`; honours
  `$VMAF_BIN`. Raises `RuntimeError` with explicit build instructions
  on missing binary.
- [`ai/data/scores.py`](data/scores.py) — `vmaf_v0.6.1` distillation
  scores (per-frame + pooled). Honours `$VMAF_MODEL_PATH`.
- [`ai/train/dataset.py`](train/dataset.py) — `NetflixFrameDataset`
  with explicit `payload_provider=` + `assume_dims=` injection points
  for unit tests.
- [`ai/train/eval.py`](train/eval.py) — PLCC / SROCC / KROCC / RMSE +
  latency. Either `onnx_path=` or `predictions=` (exactly one).
- [`ai/train/train.py`](train/train.py) — CLI entry point. Runs
  standalone (`python ai/train/train.py …`) or as a module
  (`python -m ai.train.train`); both forms work because the script
  fixes `sys.path` when `__package__` is empty.

**Rebase-sensitive invariants** (track when upstream Netflix/vmaf adds
its own training surface):

- The `iter_pairs` filename regex is fork-specific. If upstream adds a
  loader with a different ladder convention, do NOT merge them — keep
  ours under `ai/data/` and theirs under whatever path they pick.
- The per-clip JSON cache schema (`{features:{feature_names,
  per_frame, n_frames}, scores:{per_frame, pooled}}`) is consumed by
  both the dataset and any downstream consumer. Bumping the schema
  must invalidate `$VMAF_TINY_AI_CACHE` (or version-tag the path).
- The smoke command `python ai/train/train.py --epochs 0
  --assume-dims 16x16` MUST stay runnable without a built `vmaf`
  binary — the `_make_zero_payload` helper in `ai.train.dataset`
  injects a fake payload so CI gates don't drag a libvmaf build into
  the Python test surface.

## Quantization-Aware Training (ADR-0207 / ADR-0208)

The QAT trainer hook lives in [`ai/train/qat.py`](train/qat.py) and
the CLI driver in [`ai/scripts/qat_train.py`](scripts/qat_train.py).
The default config example is
[`ai/configs/learned_filter_v1_qat.yaml`](configs/learned_filter_v1_qat.yaml).

**Pipeline (per ADR-0207 + ADR-0208 implementation bridge):**

1. fp32 warm-start training.
2. FX fake-quant insertion via
   `torch.ao.quantization.quantize_fx.prepare_qat_fx` with the
   default symmetric per-tensor activation + per-channel weight
   qconfig.
3. QAT fine-tune at 10× reduced LR.
4. Copy QAT-conditioned weights into a fresh fp32 module, export
   to ONNX (`dynamo=False`), then ORT static-quantize with a
   calibration set drawn from the QAT distribution. Output is a
   QDQ `.int8.onnx`.

**Rebase-sensitive invariants:**

- The two-step pipeline (PyTorch QAT → fp32 ONNX → ORT
  static-quantize) is load-bearing. Do NOT collapse to
  `convert_fx → torch.onnx.export` — both PyTorch 2.11 ONNX
  exporters refuse the `convert_fx` output (legacy emits
  `quantized::conv2d`; TorchDynamo trips on
  `Conv2dPackedParamsBase.__obj_flatten__`). Re-check on each
  PyTorch upgrade.
- State-dict transfer in `_copy_qat_weights_into_fp32` matches
  by submodule name + tensor shape. Models using top-level
  `nn.Sequential` will break this (FX renames Sequential
  children to numeric indices); the `RuntimeError("0 tensors
  copied")` guard catches it.
- FX preparation runs on CPU (PyTorch 2.11's symbolic tracer is
  flaky on CUDA buffers); the trainer migrates to CPU before
  `prepare_qat_fx` and back to the accelerator afterwards.
- `torch.ao.quantization` is deprecated and will be removed in
  PyTorch 2.10. Migration target is `torchao.quantization.pt2e`
  (`prepare_pt2e` / `convert_pt2e`); only the FX-prep call
  changes — the rest of the pipeline (ORT static-quantize) is
  unaffected.

## Local workflow

```bash
pip install -e ai/
vmaf-train --help
vmaf-train register model/tiny/lpips_sq.onnx   # adds to registry.json
python ai/lpips_export.py                      # re-export LPIPS from the reference repo

# Netflix-corpus training (ADR-0203):
bash ai/scripts/run_training.sh
```
