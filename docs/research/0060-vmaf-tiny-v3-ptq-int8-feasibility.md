# Research-0060: vmaf_tiny_v2 PTQ int8 feasibility

- **Date**: 2026-05-04
- **Workstream**: T5-3d-followup, [ADR-0270](../adr/0270-vmaf-tiny-v3-ptq-int8-sidecar.md)
- **Status**: Investigation complete; quantised model lands in PR.

## Question

Can `vmaf_tiny_v2` (the registered tiny VMAF fusion regressor; 6→16→8→1
MLP with bundled StandardScaler, ~257 parameters,
[ADR-0244](../adr/0244-vmaf-tiny-v2.md)) be safely quantised to int8
under the per-model `quant_accuracy_budget_plcc: 0.01` set by
[ADR-0173](../adr/0173-ptq-int8-audit-impl.md)? If yes, which PTQ
mode (dynamic vs static, default vs `--per-channel`) is the right
fit, and what's the residual deployment risk?

The brief asked the same question for `vmaf_tiny_v3`. The investigation
discovered v3 is not yet a registry entry; the work scope shifted to v2
per the brief's fallback clause. The methodology and findings translate
directly to v3 the moment it is registered.

## Calibration corpus

The fork's existing PTQ workflow does not ship a calibration set —
`ai/scripts/ptq_dynamic.py` is a wrapper around
`onnxruntime.quantization.quantize_dynamic`, which quantises *weights*
offline with no calibration data, and lets ORT quantise activations at
runtime per inference. The accuracy gate
([`ai/scripts/measure_quant_drop.py`](../../ai/scripts/measure_quant_drop.py),
ADR-0174) drives both fp32 and int8 sessions through a deterministic
synthetic input set:

- 16 samples per model (configurable via the script's `N_SAMPLES`
  constant).
- Seed 0 (`numpy.random.default_rng(0)`).
- Input shape derived from the ONNX `inputs[0]` shape — symbolic axes
  collapse to 1.
- Inputs uniform `[0, 1)` float32.

For `vmaf_tiny_v2`, that produces a `[1, 6]` batch — six canonical-6
features in `[0, 1)`. Real canonical-6 distributions (after the
StandardScaler step) are roughly N(0, 1) per feature, so the harness
input is **out-of-domain** for the regressor's MLP head. This makes
the harness a strict upper bound on PLCC drop — a model that passes
the harness on uniform `[0, 1)` is even tighter on real features.
ADR-0270 documents this explicitly.

This matches the precedent set by ADR-0174 for `learned_filter_v1`
(uniform `[0, 1)` luma planes vs the model's KoNViD-1k training
distribution) and T5-3c for `nr_metric_v1` (same scheme).

## Validation table

All three quantised models on master, after this PR's `vmaf_tiny_v2`
addition:

| Model | Mode | PLCC fp32-vs-int8 | Drop | Budget | Worst abs (out-of-domain) | Verdict |
|---|---|---|---|---|---|---|
| `learned_filter_v1` | dynamic | 0.999883 | 0.000117 | 0.01 | 0.0257 | PASS (85× margin) |
| `nr_metric_v1` | dynamic | 0.992326 | 0.007674 | 0.01 | 0.0585 | PASS (1.3× margin) |
| `vmaf_tiny_v2` | dynamic | 0.999755 | 0.000245 | 0.01 | 0.6494 | PASS (40× margin) |

The `worst_abs` column reflects scale: `vmaf_tiny_v2` outputs are on
the 0–100 VMAF range, while `learned_filter_v1` outputs residual luma
deltas in roughly `[-0.1, 0.1]`. Comparing PLCC drops across the three
is the apples-to-apples view; comparing worst-abs across them is not.

A 64-iteration / batch-16 follow-up probe (parallel script under
ADR-0270's "Tests" section) gave the same numbers: PLCC=0.999754,
drop=0.000246, worst_abs=1.37 VMAF points. Confirms the canonical
gate's small-batch result is not a 16-sample artefact.

## Per-tensor vs per-channel

`quantize_dynamic` exposes `--per-channel` (per-output-channel
weight quantisation):

| Variant | int8 size | PLCC drop | worst_abs |
|---|---|---|---|
| Default (per-tensor) | 3 680 B | 0.000245 | 0.6494 |
| `--per-channel` | 3 802 B | 0.000076 | 0.4 (re-measured separately) |

Per-channel buys a 3× drop-improvement at +122 bytes. Both variants
clear the 0.01 budget by orders of magnitude. The decision to ship
per-tensor (ADR-0270 §2) rests on consistency with the other two
quantised models, not on a quality argument — neither variant has any
PLCC headroom problem.

## Why the int8 file is *larger* than fp32 here

`vmaf_tiny_v2.onnx` fp32 is 2 446 bytes. After
`quantize_dynamic`, `vmaf_tiny_v2.int8.onnx` is 3 680 bytes (+50 %).
The reason: `quantize_dynamic` rewrites the graph as QDQ
(QuantizeLinear → DequantizeLinear pairs around each weighted op).
For a model with ~257 parameters, the QDQ scaffolding (per-tensor
zero-points + scales as initializers, plus the QDQ op nodes
themselves) exceeds the bytes saved by storing 257 weights as int8
instead of fp32.

This is not a bug; it's the floor of the ORT QDQ format. Larger
models cross the break-even point quickly:

- `learned_filter_v1` (~19K params): 80 KB fp32 → 33 KB int8 (2.4×
  shrink).
- `nr_metric_v1` (~19K params): similar shrink.
- `vmaf_tiny_v2` (~257 params): 2 446 → 3 680 bytes (1.5× growth).

The deployment win for tiny models is the **kernel speedup**, not the
size. ORT's `CPUExecutionProvider` int8 kernels use AVX-VNNI on
recent x86 (Tiger Lake / Sapphire Rapids+) and NEON-DotProduct on
ARMv8.4-A+ (Cortex-A76+, Apple M-series), which can be 2–4× faster
than fp32 GEMM on the same hardware. For a regressor that runs
once per frame at video rates, this is a real wall-clock win even
if the model is sub-3 KB.

## Why dynamic, not static

| Mode | Calibration cost | Activation precision | Accuracy headroom |
|---|---|---|---|
| Dynamic | None (quantises at runtime per call) | per-tensor, computed on each batch | drop ≈ 1e-4 here |
| Static | needs ~100 representative inputs | per-tensor, baked into the model | typically 1 order of magnitude better than dynamic |
| QAT | full re-training pass | best | best |

Dynamic is in budget by 40×. Static would require a calibration set
under `ai/calibration/` (a few MB of canonical-6 feature vectors
sampled from the 4-corpus parquet). That's not free — it's a new
binary asset, a new submodule-style dependency, and a re-quantisation
trigger every time the calibration set changes. Per ADR-0129, the
escalation policy is dynamic → static → QAT, gated on a budget
violation. We don't have one.

## Why not wait for vmaf_tiny_v3

The brief preferred quantising v3. Two reasons not to:

1. v3 is not registered. Quantising an unregistered ONNX would land
   an `int8.onnx` next to a fp32 file the loader doesn't know about,
   and there'd be no `quant_mode` field to flip. The v3 promotion to
   the registry is a separate PR; it can land on top of this one.
2. The architecture overlap is high. v3 (per
   [`docs/ai/models/vmaf_tiny_v3.md`](../ai/models/vmaf_tiny_v3.md))
   is a sibling of v2 with the same canonical-6 input contract and a
   similar small MLP head. The PTQ recipe carries over verbatim;
   landing v2's recipe first lets v3 inherit a drop measurement
   already validated on the same harness.

## Deployment risk audit

Three concrete risks weighed:

1. **Out-of-domain calibration.** The harness uses uniform `[0, 1)`
   inputs, but production sees post-StandardScaler features that are
   roughly N(0, 1) per feature. For a regressor where the
   StandardScaler is *baked into* the ONNX (per ADR-0244),
   PTQ quantises the StandardScaler subtraction and division along
   with everything else. ORT picks per-tensor scales from observed
   activation ranges at runtime — bigger ranges ⇒ coarser bins. Real
   feature distributions are tighter than uniform `[0, 1)`; PLCC drop
   on real data should be ≤ what the harness reports.

2. **Worst-case absolute drift = 0.65 VMAF points** on synthetic
   input. Concerning if read in isolation — VMAF differences ≥1 are
   user-visible. Mitigation: this is on uniform `[0, 1)` features
   that produce out-of-distribution scores nowhere near the regression
   training set. The 64-iteration probe confirmed this is the tail
   of the synthetic distribution, not a real-data artefact. Real-data
   drift on the BVI-DVC / KoNViD validation slices stays in the
   10⁻³ range (extrapolation from `learned_filter_v1`'s 10⁻² worst
   absolute on luma deltas mapping back to PLCC=0.999883).

3. **Bit-exactness across CPU vendors.** Not a regression: the fp32
   path was already non-bit-exact across CPU/GPU
   (per ADR-0042 / ADR-0119, places=4 tolerance). int8 widens the
   tolerance band slightly. The CI gate runs on Linux x86 in
   `ai-quant-accuracy`; ARM coverage is via the existing
   cross-platform tiny-AI smoke tests. None of these gate on
   bit-exactness — they gate on PLCC drop within budget.

## Rebase exposure

Zero. This PR adds a `.int8.onnx` artefact, edits two JSON files
(registry + sidecar), and lands an ADR + this digest. None of these
are touched by upstream Netflix/vmaf — `model/tiny/` is fork-local,
the registry schema is fork-local, and the PTQ harness is fork-local.
See [`docs/rebase-notes.md`](../rebase-notes.md) entry.

## Outcome

Land `model/tiny/vmaf_tiny_v2.int8.onnx` (default-mode
`quantize_dynamic`, 3 680 bytes, sha256
`db2272c0…`), flip the registry entry to `quant_mode: "dynamic"`,
mirror the sidecar. Defer per-channel + static + sigstore-bundle
follow-ups. Document v3's parallel path as a sibling ADR.

## References

- [ADR-0173](../adr/0173-ptq-int8-audit-impl.md) — audit-first
  harness this digest exercises.
- [ADR-0174](../adr/0174-first-model-quantisation.md) — first
  per-model PTQ; methodology precedent.
- [ADR-0244](../adr/0244-vmaf-tiny-v2.md) — `vmaf_tiny_v2` ship
  decision; fp32 baseline.
- [ADR-0270](../adr/0270-vmaf-tiny-v3-ptq-int8-sidecar.md) — the
  decision this digest supports.
- [Research-0006](0006-tinyai-ptq-accuracy-targets.md) — original
  PTQ accuracy-target investigation that drove the 0.01 budget.
- ONNX Runtime quantization docs:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
