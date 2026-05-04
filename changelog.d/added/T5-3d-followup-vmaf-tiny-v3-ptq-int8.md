- **`vmaf_tiny_v2` int8 PTQ sidecar (ADR-0270, Research-0060,
  T5-3d-followup).** Third registered tiny-AI model on the
  audit-first PTQ harness from ADR-0173 / ADR-0174 (after
  `learned_filter_v1` and `nr_metric_v1`). New artefact
  `model/tiny/vmaf_tiny_v2.int8.onnx` (3 680 bytes; sha256
  `db2272c0…`); `model/tiny/registry.json` and
  `model/tiny/vmaf_tiny_v2.json` flip to `quant_mode: "dynamic"`
  with `int8_sha256` + `quant_accuracy_budget_plcc: 0.01`. Default
  per-tensor `quantize_dynamic` (no `--per-channel`); fp32 stays as
  the audit baseline. Measured PLCC drop = 0.000245 on the
  canonical 16-sample seed-0 harness via
  `ai/scripts/measure_quant_drop.py`, 40× under budget.
  Scope note: the brief asked for `vmaf_tiny_v3`, but v3 is not yet
  in the registry; per the brief's fallback clause the work landed
  on v2, with the same PTQ recipe ready to apply once v3 promotes.
  No new C-side or build-side code; the runtime redirect from
  ADR-0174 already covers this path.
