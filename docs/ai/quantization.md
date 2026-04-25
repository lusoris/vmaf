# Tiny-AI int8 quantisation

The fork supports three post-training quantisation (PTQ) modes for
shipped tiny-AI ONNX models, plus quantisation-aware training (QAT).
Each model carries its quant decision in `model/tiny/registry.json`
and an accuracy budget that the CI harness enforces against the fp32
baseline.

Audited and scaffolded in
[ADR-0173](../adr/0173-ptq-int8-audit-impl.md); policy origin
[ADR-0129](../adr/0129-tinyai-ptq-int8-modes.md).

## Per-model registry fields

| Field                          | Type                                                | Default | Required when                          |
|--------------------------------|-----------------------------------------------------|---------|----------------------------------------|
| `quant_mode`                   | `fp32` / `dynamic` / `static` / `qat`               | `fp32`  | always present (default fp32)          |
| `quant_calibration_set`        | path (relative to repo root)                        | absent  | `quant_mode == "static"`               |
| `quant_accuracy_budget_plcc`   | number in `[0, 1]`                                  | `0.01`  | always (the CI gate honours per-entry) |

`fp32` keeps the loader on the `<basename>.onnx` file. The other three
modes redirect the loader to a sibling `<basename>.int8.onnx`
produced by the scripts below; the fp32 file stays on disk as the
regression baseline.

## Mode selection

| Mode | Accuracy | Cost to produce | Best for |
|---|---|---|---|
| `fp32`     | reference                   | none                       | new models, debug builds |
| `dynamic`  | small accuracy hit (~0.5%)  | one CLI call               | models without a calibration set; deployment box differs from training box |
| `static`   | small accuracy hit (~0.2%)  | one calibration pass       | models we own + control + can pin a calibration set |
| `qat`      | reference (within ~0.05%)   | extra training phase, ~1.5× fp32 train time | models where static drops accuracy past the per-model budget |

Pick the cheapest mode that stays inside the
`quant_accuracy_budget_plcc` budget.

## Producing int8 artefacts

### Dynamic PTQ

```bash
python ai/scripts/ptq_dynamic.py model/tiny/nr_metric_v1.onnx
# -> model/tiny/nr_metric_v1.int8.onnx
```

No calibration data needed. Wraps `onnxruntime.quantization.quantize_dynamic`.

### Static PTQ

Build a calibration `.npz` first — one entry per ONNX input name, each
a stack of `[N, ...]` representative samples. Then:

```bash
python ai/scripts/ptq_static.py model/tiny/nr_metric_v1.onnx \
    --calibration ai/calibration/nr_metric_v1.npz
```

The output goes to `<input>.int8.onnx`. Add the calibration path to
the registry's `quant_calibration_set` field.

### Quantisation-aware training (QAT)

```bash
python ai/scripts/qat_train.py --config ai/configs/c2_qat.yaml \
    --output model/tiny/nr_metric_v1.int8.onnx
```

**Status:** scaffolded; the trainer hook is filled in by the per-model
QAT PR that uses it (per ADR-0129's audit-first sequence). Calling it
today raises `NotImplementedError` with instructions on the missing
trainer integration.

## CI accuracy gate (planned, T5-3b)

The `ai-quant-accuracy` CI job will run each quantised model against
its per-model VMAF soak fixture and assert the PLCC drop is below
`quant_accuracy_budget_plcc`. Not yet wired — lands with the first
model that flips its `quant_mode` away from `fp32`. ADR-0173
intentionally ships the registry + scripts only (audit-first).

## Per-model PR template

When proposing a model for quantisation:

1. Run `ai/scripts/ptq_<mode>.py` to produce the int8 file.
2. Compute fp32 vs int8 PLCC on the soak fixture.
3. In the PR description: paste the PLCC numbers + the ratio of
   inference time fp32 / int8 on at least one CPU.
4. Update `model/tiny/registry.json`:
   - flip `quant_mode` to the chosen mode,
   - set `quant_accuracy_budget_plcc` (default 0.01 = 1 PLCC point),
   - add `quant_calibration_set` if `static`.
5. Land the int8 ONNX next to the fp32 file.

The reviewer compares the measured drop against the budget. If a
static run misses budget, escalate to QAT in a follow-up PR — don't
relax the budget.

## Caveats

- **No model is currently quantised.** This page documents the
  harness; per-model decisions follow as separate PRs.
- **Calibration sets are not redistributable** by default. Operators
  build their own from a parquet feature cache (the
  `ai/scripts/build_calibration_set.py` helper is queued — until it
  lands, hand-craft the `.npz`).
- **VNNI / DLBoost** speedup applies only on Intel CPUs Cascade Lake
  and newer; ARMv8.2+ has int8 dot-product. On CPUs without either,
  the int8 path runs slower than fp32 due to QDQ overhead. The
  loader is bit-depth-agnostic — it still picks the int8 model when
  the registry says so; runtime perf is the operator's problem to
  measure.
