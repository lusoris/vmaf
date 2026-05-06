# AGENTS.md — ai/

Orientation for agents working on the tiny-AI **training** side. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

Python package for training, exporting, and registering tiny-AI
checkpoints that are then consumed by [libvmaf/src/dnn/](../libvmaf/src/dnn/AGENTS.md)
at runtime. Stack: PyTorch + Lightning → ONNX.

```text
ai/
  pyproject.toml   # package metadata (training-only deps)
  src/             # vmaf-train CLI + model defs + dataset loaders
  tests/           # pytest unit tests
  configs/         # dataset manifests + training recipes
  lpips_export.py  # re-export richzhang/PerceptualSimilarity → ONNX
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
- **Bisect-cache fixture is content-stable** — `ai/testdata/bisect/`
  is the deterministic placeholder for the nightly
  `bisect-model-quality` workflow. Regenerate via
  `python ai/scripts/build_bisect_cache.py` with seeds
  `FEATURE_SEED=20260418` / `MODEL_SEED=20260419`. CI runs the same
  script with `--check`. As of ADR-0262 the parquet leg of the check
  uses logical `pyarrow.Table.equals` content comparison (schema + row
  count + values), tolerating writer-version-string drift in the
  `created_by` parquet header — but ONNX still compares byte-for-byte
  via `filecmp.cmp(shallow=False)`, which means ONNX-side determinism
  must stay intact. **Do not** remove the
  `model.producer_name = "vmaf-train.bisect-cache"`,
  `model.producer_version = "1"`, or `model.ir_version = 9` pins in
  `_save_linear_fr`: those three lines are what stabilises ONNX bytes
  across `onnx` minor versions. See
  [ADR-0262](../docs/adr/0262-bisect-cache-logical-comparison.md) +
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
- [ADR-0286](../docs/adr/0286-saliency-student-fork-trained-on-duts.md) — `saliency_student_v1` is the fork-trained tiny U-Net that replaces `mobilesal_placeholder_v0` as the production weights for the `mobilesal` extractor. Tensor-name contract (`input`, `saliency_map`) and NCHW shapes are unchanged from ADR-0218, so any future weights swap (multi-dataset student, distilled u2netp, …) drops in without C changes. The decoder uses `ConvTranspose` for stride-2 upsampling — keep it that way unless `Resize` lands on `libvmaf/src/dnn/op_allowlist.c` first. DUTS-TR images are *not* committed in-tree; only the trained `.onnx` + sidecar are. Trainer at [`scripts/train_saliency_student.py`](scripts/train_saliency_student.py); reproducer in [`docs/ai/models/saliency_student_v1.md`](../docs/ai/models/saliency_student_v1.md).
- [ADR-0109](../docs/adr/0109-nightly-bisect-model-quality.md) — nightly bisect workflow + synthetic placeholder cache.
- [ADR-0235](../docs/adr/0235-codec-aware-fr-regressor.md) — codec-aware FR regressor (`fr_regressor_v2`). `CODEC_VOCAB` in [`src/vmaf_train/codec.py`](src/vmaf_train/codec.py) is **closed and order-stable** — the index of each codec is the one-hot column index baked into trained ONNX. Adding a codec appends to the tuple and bumps `CODEC_VOCAB_VERSION`; reordering silently invalidates every shipped `fr_regressor_v2_*.onnx`. `FRRegressor(num_codecs=0)` must remain the v1 single-input contract — flipping the default would break every existing `model/tiny/fr_regressor_v1.onnx` consumer. Feature-dump scripts emit a `codec` column tagged at the call site (BVI-DVC: `"x264"`, Netflix Public: `"unknown"`); never silently default to a codec that doesn't match what the script actually encoded.
- [ADR-0305](../docs/adr/0305-encoder-knob-space-pareto-analysis.md) — **knob-sweep corpus invariant.** The 12,636-cell sweep at `runs/phase_a/full_grid/comprehensive.jsonl` (gitignored, locally generated) is the source of truth for `tools/vmaf-tune/codec_adapters/*` recipe defaults. Pareto frontiers are stratified per `(source, codec, rc_mode)` slice — never collapsed to a global hull (companion [Research-0063](../docs/research/0063-encoder-knob-space-cq-vs-vbr-stratification.md) shows the global-hull failure mode regresses NVENC h264/hevc by ~4 VMAF at cq=30). **Recipes that regress vs the bare encoder at matched bitrate within the same slice MUST NOT ship as adapter defaults.** The regression-detection check lives in `ai/scripts/analyze_knob_sweep.py` (`detect_recipe_regressions(...)`) and is exercised by `ai/tests/test_knob_sweep_analysis.py::test_recipe_regression_detection`; new codec adapter PRs cite the per-(codec, rc_mode) hull row from `reports/summary.md` (or "no hull entry yet — bare default") in their PR description. Methodology + scaffolded findings: [Research-0077](../docs/research/0077-encoder-knob-space-pareto-frontiers.md).

## Netflix-corpus training prep (ADR-0242 / ADR-0203)

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
- **`vmaf_tiny_v2` ONNX contract (ADR-0244).** The shipped ONNX
  embeds the StandardScaler `(mean, std)` as Constant `Sub` + `Div`
  nodes that run before the MLP. The runtime feeds raw canonical-6
  feature values; do NOT add an external scaler step. Re-exporting
  via [`ai/scripts/export_vmaf_tiny_v2.py`](scripts/export_vmaf_tiny_v2.py)
  is the only supported path — it pulls `mean` / `std` from the
  trainer checkpoint and bakes them as graph initialisers, so the
  `model/tiny/registry.json` sha256 covers the calibration values
  too. Input name is `features` ([N, 6] float32), output `vmaf`
  ([N] float32); feature column order is fixed at
  `(adm2, vif_scale0, vif_scale1, vif_scale2, vif_scale3, motion2)`
  and must not be reordered without a full Phase-3 re-validation.
- **`vmaf_tiny_v3` ships alongside v2 (ADR-0241).** Same ONNX
  contract as v2 (input `features [N, 6]` float32, output
  `vmaf [N]` float32, opset 17, scaler-baked-into-graph) — only the
  architecture differs (`mlp_medium` 6 → 32 → 16 → 1, 769 params vs
  v2's `mlp_small` 257). **Production default stays v2**;
  [`docs/ai/inference.md`](../docs/ai/inference.md) and the model-card
  cross-references both keep v2 as the recommended `--tiny-model`.
  v3 is the higher-PLCC / lower-variance option (Netflix LOSO mean
  PLCC 0.9986 ± 0.0015 vs v2's 0.9978 ± 0.0021). Do NOT replace v2
  with v3 wholesale — both file paths are referenced by name in
  user-facing docs and the registry, and the small mean delta does
  not justify a default flip without multi-seed + KoNViD 5-fold
  parity (documented as Phase-3e follow-up). Same scripts pattern:
  `train_vmaf_tiny_v3.py` / `export_vmaf_tiny_v3.py` /
  `validate_vmaf_tiny_v3.py` / `eval_loso_vmaf_tiny_v3.py` —
  do **not** modify the v2 scripts when iterating on v3.
- **`vmaf_tiny_v3` and `vmaf_tiny_v4` opt-in tiers
  (ADR-0241 / ADR-0242).** v3 (`mlp_medium`, 769 params, ADR-0241)
  and v4 (`mlp_large`, 3 073 params, ADR-0242) ship *alongside* v2,
  not as replacements. Production default stays `vmaf_tiny_v2`. The
  three rungs share the canonical-6 input contract, the bundled
  StandardScaler, and the 90 ep / Adam@1e-3 / MSE / bs=256 recipe;
  only the architecture differs. **Do NOT modify v2 or v3 scripts
  when iterating on later rungs** — each version owns its own
  `train_vmaf_tiny_vN.py` / `export_vmaf_tiny_vN.py` /
  `validate_vmaf_tiny_vN.py` / `eval_loso_vmaf_tiny_vN.py` quartet.
  The arch ladder **stops at v4**: the v3 → v4 LOSO PLCC delta is
  +0.0001 (well below 1 std), demonstrating saturation on the
  canonical-6 + 4-corpus regime. Future quality gains require
  regime change (richer features, larger corpus, ensembles), not
  deeper MLPs. See ADR-0242 § Alternatives considered for the
  mlp_huge rejection rationale.

## `fr_regressor_v1` (C1 baseline — ADR-0249)

The Wave-1 C1 baseline trainer is
[`ai/scripts/train_fr_regressor.py`](scripts/train_fr_regressor.py). It
consumes `runs/full_features_netflix.parquet` (produced by
`ai/scripts/extract_full_features.py` over the local Netflix Public
drop at `.workingdir2/netflix/`), runs 9-fold leave-one-source-out
(LOSO), and exports `model/tiny/fr_regressor_v1.onnx` only when mean
LOSO PLCC ≥ 0.95 against the `vmaf_v0.6.1` per-frame teacher.

**Contract row** (do not regress without an ADR amendment):

- **Input** — `[N, 6]` float32, feature order
  `(adm2, vif_scale0, vif_scale1, vif_scale2, vif_scale3, motion2)`,
  standardised with the per-feature `feature_mean` / `feature_std`
  vectors pinned in the sidecar JSON. Standardisation is **not**
  baked into the ONNX so callers can swap feature pools without
  re-export.
- **Output** — `[N]` float32, VMAF-scale (0–100 typical).
- **Architecture** — stock `vmaf_train.models.FRRegressor` with the
  Wave-1 spec hparams (hidden=64, depth=2, dropout=0.1, GELU). Larger
  / smaller variants must register a new model id, not overwrite this
  one.
- **Ship gate** — mean LOSO PLCC ≥ 0.95 vs `vmaf_v0.6.1`. The trainer
  exits 3 and refuses to overwrite the registry on failure; lowering
  the threshold is a soft-fail of policy, not a code change.

**Rebase-sensitive invariants:**

- The canonical-6 feature order is load-bearing — `vmaf_v0.6.1`
  consumes the same six features in the same order, and the ONNX
  graph weight matrix is column-aligned to it. Reordering the
  sidecar `feature_order` field invalidates the checkpoint.
- Netflix Public Dataset is non-redistributable. CI cannot retrain
  end-to-end; only the smoke path
  (`python ai/scripts/train_fr_regressor.py --epochs 3 --no-export`)
  runs in CI when the parquet is locally available.

## Dynamic-PTQ tiny-MLP family (ADR-0275)

`vmaf_tiny_v3` and `vmaf_tiny_v4` carry dynamic-PTQ int8 sidecars
produced by `ai/scripts/ptq_dynamic.py`. The recipe is identical to
`learned_filter_v1` (ADR-0174) and `nr_metric_v1` (ADR-0248): a
single CLI invocation, no calibration data. The on-disk size win is
proportional to weight mass — `mlp_large` (v4) shrinks 45 %,
`mlp_medium` (v3) shrinks 5 % because Constant scaler nodes and op
metadata dominate that graph. v2 (`mlp_small`) stays fp32: too
little weight mass for an int8 sidecar to be worth the audit cost.

**Invariants:**

- The fp32 `<basename>.onnx` stays on disk as the regression
  baseline; the runtime redirect from ADR-0174 picks the
  `.int8.onnx` sibling only when the registry overlay declares
  `quant_mode != "fp32"`.
- `python ai/scripts/measure_quant_drop.py --all` is the gate. Both
  v3 and v4 sit two orders of magnitude under the 0.01 PLCC
  budget; treat any future drop > 1e-3 as a regression worth
  investigating before merging an int8 refresh.
- Re-running `ptq_dynamic.py` is deterministic on a fixed fp32
  input — but the sha256 of the int8 output can shift across ORT
  versions. When ORT is bumped, regenerate both sidecars and
  refresh `int8_sha256` in `model/tiny/registry.json` +
  `vmaf_tiny_v{3,4}.json` in the same PR.
## `fr_regressor_v2_ensemble_v1` — probabilistic head (ADR-0279)

The probabilistic successor to the codec-aware
`fr_regressor_v2` is a deep ensemble of N=5 v2 members trained under
distinct seeds, packaged as 5 ONNX files plus a manifest sidecar
`model/tiny/fr_regressor_v2_ensemble_v1.json`. Trainer:
[`ai/scripts/train_fr_regressor_v2_ensemble.py`](scripts/train_fr_regressor_v2_ensemble.py);
evaluator: [`ai/scripts/eval_probabilistic_proxy.py`](scripts/eval_probabilistic_proxy.py);
model card:
[`docs/ai/models/fr_regressor_v2_probabilistic.md`](../docs/ai/models/fr_regressor_v2_probabilistic.md).

**Rebase-sensitive invariants:**

- **Per-member ONNX I/O contract is the v2 two-input shape**: inputs
  `features [N, 6]` (canonical-6, StandardScaler-normalised by the
  manifest's `feature_mean` / `feature_std`) + `codec_onehot
  [N, NUM_CODECS]`; output `score [N]` float32. Each member is a
  stock `FRRegressor(num_codecs=NUM_CODECS)` — flipping that to a
  single-input v1-shaped graph silently invalidates every shipped
  ensemble.
- **Manifest layout is the runtime entry point**, not the registry
  rows. Each member is also added to `model/tiny/registry.json` as
  `kind: "fr"` (with id `<ensemble_id>_seed<N>`) so the existing
  tiny-model verifier can SHA-256-check each member without a
  schema bump, but the manifest sidecar's `members[]` list is the
  canonical ordered set the C-side adapter iterates over. Adding a
  schema-version field to `registry.schema.json` for a `fr_ensemble`
  kind is a future option; until then, ensemble lookups go through
  the manifest, not the registry.
- **Ensemble size is part of the contract.** The manifest's
  `ensemble_size` field pins N; the C-side adapter must open
  `ensemble_size` sessions. Changing N requires a new ensemble id +
  manifest, not an in-place mutation.
- **Confidence rule is a one-of**: `confidence.method` is either
  `"ensemble"` (use `gaussian_z` as the multiplier on `sigma`) or
  `"ensemble+conformal"` (use `conformal_q_residual` instead). The
  trainer emits the conformal scalar only when
  `--conformal-calibration-frac > 0` and the calibration split is
  large enough; otherwise the field stays `null` and the Gaussian
  rule applies.
- **`CODEC_VOCAB` parity with v2 is required.** The manifest pins
  `codec_vocab` + `codec_vocab_version`; the runtime must refuse to
  load when these disagree with the live `ai/src/vmaf_train/codec.py`
  vocabulary. Bumping the vocabulary requires retraining the
  ensemble; the existing closed-vocabulary invariant from ADR-0235
  carries over verbatim.
- **Smoke artefact is a load-path probe**, not a quality model. The
  shipped `model/tiny/fr_regressor_v2_ensemble_v1*.onnx` files come
  from a synthetic 100-row corpus + 1 epoch / member; do NOT
  benchmark them against real VMAF. Production training is gated on
  the multi-codec Phase A corpus and is tracked as backlog item
  T7-FR-REGRESSOR-V2-PROBABILISTIC.
- **Ensemble registry invariant (ADR-0303)**: each ensemble member's
  `smoke: true` registry row flips to `false` **only after** that
  individual seed clears the `PLCC_i ≥ 0.95` LOSO ship gate
  (ADR-0235 / ADR-0291). The ensemble-mean entry — if/when one is
  added to `model/tiny/registry.json` as a `fr_ensemble`-kind row —
  flips **only after all five seeds clear** the per-seed gate *and*
  the variance bound `max_i(PLCC_i) - min_i(PLCC_i) ≤ 0.005` holds.
  The decision lives in [`scripts/ci/ensemble_prod_gate.py`](../scripts/ci/ensemble_prod_gate.py);
  the trainer that emits the per-seed `loso_seed{N}.json` artefacts
  the gate consumes is
  [`ai/scripts/train_fr_regressor_v2_ensemble_loso.py`](scripts/train_fr_regressor_v2_ensemble_loso.py).
  Do NOT flip individual seed rows by hand without running the gate
  against a real-corpus LOSO output — the variance bound is what
  protects the predictive-distribution semantics; flipping seeds
  ad-hoc would silently bake in an unbounded across-seed spread.
- **Registry-flip is a separate PR (ADR-0309)**: the
  `fr_regressor_v2_ensemble_v1_seed{0..4}` rows in
  `model/tiny/registry.json` flip `smoke: true → false` **only** in a
  dedicated follow-up PR that cites a passing
  `runs/ensemble_v2_real/PROMOTE.json` produced by
  [`ai/scripts/validate_ensemble_seeds.py`](scripts/validate_ensemble_seeds.py).
  **Never** flip these rows during a `/sync-upstream` rebase or as a
  side-effect of any other PR — the harness in
  [`ai/scripts/run_ensemble_v2_real_corpus_loso.sh`](scripts/run_ensemble_v2_real_corpus_loso.sh)
  and the validator emit the verdict file but **do not** mutate the
  registry. Auto-flipping on PROMOTE was rejected in ADR-0309's
  alternatives matrix specifically because rebase-time mutation of
  shipped registry rows is the foot-gun this invariant exists to
  prevent.

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

## fr_regressor_v2 — codec block layout (ADR-0272)

`ai/scripts/train_fr_regressor_v2.py` consumes the vmaf-tune Phase A
JSONL corpus and emits `model/tiny/fr_regressor_v2.onnx`. The codec
block layout is **load-bearing** — bumping it requires a re-train.
Pinned invariants:

- `ENCODER_VOCAB = ("libx264", "libx265", "libsvtav1", "libvvenc",
  "libvpx-vp9", "unknown")`. Order matches the encoder-onehot index
  baked into the trained ONNX. Append-only; bump
  `ENCODER_VOCAB_VERSION` and re-train when adding a new entry.
- 8-D codec block layout: `[encoder_onehot[0..5], preset_norm,
  crf_norm]`. Both `preset_norm` and `crf_norm` live in `[0, 1]`.
- `crf_norm = crf / 63.0` — `63` is the union upper bound across
  supported encoders (libsvtav1 / libvpx-vp9 max).
- `preset_norm = preset_ordinal / 9.0` — per-encoder ordinal table
  in `train_fr_regressor_v2.py::PRESET_ORDINAL`. libsvtav1's numeric
  0..13 presets are squashed to 0..9.
- Two-input ONNX: `features` (N, 6) + `codec` (N, 8) -> `score` (N,).
  Mirrors the LPIPS-Sq two-input precedent (ADR-0040 / ADR-0041).
- StandardScaler is applied to `features` only; the codec block
  passes through unscaled. `feature_mean` / `feature_std` ship in
  the sidecar JSON.

The current shipped ONNX is from `--smoke` mode and is registered
`smoke: true` in `model/tiny/registry.json`. Production training run
is gated on a multi-codec Phase A corpus + per-frame feature emission
in the Phase A schema. See ADR-0272 + Research-0054.
## v5 corpus-expansion probe — research-only (ADR-0287)

The `*_vmaf_tiny_v5.py` scripts
(`fetch_youtube_ugc_subset.py`, `extract_ugc_features.py`,
`train_vmaf_tiny_v5.py`, `eval_loso_vmaf_tiny_v5.py`) are
research infrastructure for the deferred v5 corpus-expansion
probe. **No `vmaf_tiny_v5.onnx` ships** — the 1-σ ship gate did
not clear (Δ PLCC = +0.00005 at seed=0, far below the 1-σ_v2
threshold). When extending these scripts:

- Do not add a `vmaf_tiny_v5` row to
  `model/tiny/registry.json` unless a follow-up run actually
  clears the ship gate documented in
  [ADR-0287](../docs/adr/0287-vmaf-tiny-v5-corpus-expansion.md).
- The fetcher hits a public GCS bucket (`gs://ugc-dataset/`,
  CC-BY); raw videos and the resulting
  `runs/full_features_ugc.parquet` must NEVER be committed
  (the `runs/` and `.workingdir2/` trees are gitignored).
- The dual-arm LOSO trains 18 mlp_small models
  (9 v2-baseline plus 9 v5-candidate); single invocation
  wall-time is ~10–25 min depending on CPU. Do NOT launch it
  concurrently with another training process — the two share
  BLAS threads and serialise badly.

## v3 retrain invariant — `ENCODER_VOCAB` 13 → 16 (ADR-0302)

The `ENCODER_VOCAB_V3` parallel constant in
[`scripts/train_fr_regressor_v2.py`](scripts/train_fr_regressor_v2.py)
documents the target 16-slot vocab (adds `libsvtav1`,
`h264_videotoolbox`, `hevc_videotoolbox` to v2's 13 slots). The
**live `ENCODER_VOCAB` and `ENCODER_VOCAB_VERSION = 2` are the source
of truth** until the follow-up retrain PR clears the LOSO PLCC
ship gate.

**Invariants that the v3 retrain PR must honour** (per ADR-0235 +
ADR-0291 + ADR-0302):

- A schema bump (v2 → v3) requires a fresh LOSO run that clears
  **mean LOSO PLCC ≥ 0.95** across all 9 Netflix sources (matches
  the gate ADR-0291 cleared on v2). The trainer must exit non-zero
  and refuse to overwrite the registry entry on failure — same
  pattern `fr_regressor_v1` already enforces.
- Multi-codec lift over the v1 single-input regressor must remain
  **≥ +0.005 PLCC**. ADR-0235 set this as the codec-block invariant;
  the v2 production checkpoint cleared it comfortably and v3 must
  not regress.
- The in-tree v2 ONNX (`model/tiny/fr_regressor_v2.onnx`) **must
  not be replaced** until the new v3 ONNX clears the gate. The
  load-fallback shim collapses unknown v3 strings into the v2
  `unknown` column and lets v2 keep serving every consumer in the
  meantime.
- Append-only ordering is load-bearing — the 13 v2 slot indices
  (0..12) keep their column positions verbatim under v3; the three
  new slots append at indices 13/14/15. Reordering silently
  invalidates every shipped `fr_regressor_v2_*.onnx`. ADR-0235
  documents this rule for `CODEC_VOCAB`; ADR-0302 §Decision
  re-asserts it for `ENCODER_VOCAB`.
- Slot strings must match the vmaf-tune codec-adapter registry keys
  exactly (`libsvtav1`, `h264_videotoolbox`, `hevc_videotoolbox`).
  ADR-0235 §References pins this rule globally for all corpus
  emitters.
