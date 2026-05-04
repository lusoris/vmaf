# ADR-0270: PTQ int8 sidecar for `vmaf_tiny_v2` (T5-3d-followup)

- **Status**: Accepted
- **Date**: 2026-05-04
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: tiny-ai, onnx, quantization, registry, fork-local

## Context

The brief that landed this PR asked for an int8 PTQ sidecar for
`vmaf_tiny_v3`. On inspection,
[`model/tiny/registry.json`](../../model/tiny/registry.json) currently
ships `vmaf_tiny_v2` as the only registered tiny VMAF fusion regressor;
`vmaf_tiny_v3.onnx` exists in the tree as an experimental candidate
(see [`docs/ai/models/vmaf_tiny_v3.md`](../ai/models/vmaf_tiny_v3.md))
but has not been promoted to a registry entry yet. Per the brief's
fallback clause, this ADR scopes the work down to **`vmaf_tiny_v2`**
and treats it as a follow-up to ADR-0174's "first per-model PTQ"
sequence. The slug retains the original `vmaf-tiny-v3-…` form so the
PR title and ADR slug match the planning record; once `vmaf_tiny_v3`
is promoted to the registry, the same script can be applied verbatim
and a sibling ADR landed.

ADR-0173 introduced the audit-first PTQ harness (registry schema,
three scripts, sidecar parser); ADR-0174 quantised
`learned_filter_v1` and wired the runtime redirect; ADR-0166-style
follow-up PRs (`nr_metric_v1`, T5-3c) extended coverage to the C2
NR model. `vmaf_tiny_v2` is the third registered model to flip from
`quant_mode: "fp32"` to `"dynamic"`. The motivation matches the
earlier per-model PRs:

1. The runtime int8 redirect in `vmaf_dnn_session_open` is already
   load-bearing (ADR-0174); shipping a quantised `vmaf_tiny_v2`
   exercises the same path on a different op mix.
2. The fp32 file stays as the audit baseline; the int8 sidecar is
   a deployment optimisation.
3. `vmaf_tiny_v2` is the smallest registered tiny-AI model
   (~257 params, 2.4 KB fp32). Quantisation is **not** a
   size-reduction story for this model — int8 + QDQ wrapping
   actually grows the file from 2 446 → 3 680 bytes (1.5× larger).
   The reason to ship int8 anyway is **harness coverage** plus the
   int8-kernel speedup on VNNI / DLBoost / NEON-DotProduct CPUs that
   the runtime gets for free once the redirect fires. Same trade-off
   reasoning ADR-0174 documented for `learned_filter_v1`'s 33 KB →
   33 KB story.

## Decision

### 1. `vmaf_tiny_v2` flips to `quant_mode: "dynamic"`

- Quantised file: `model/tiny/vmaf_tiny_v2.int8.onnx`
  (3 680 bytes, sha256 `db2272c0dd942371fdf39987c85c3ba8de2b277621fa1ea8e937442792156c96`).
- Registry entry gains `quant_mode: "dynamic"` +
  `int8_sha256: "db2272…"` + `quant_accuracy_budget_plcc: 0.01`.
  Notes line records the provenance: `Dynamic-PTQ int8 sidecar via
  ai/scripts/ptq_dynamic.py (T5-3d-followup / ADR-0270);
  drop=0.000245.`
- Sidecar JSON
  ([`model/tiny/vmaf_tiny_v2.json`](../../model/tiny/vmaf_tiny_v2.json))
  gains the same three fields. The C-side
  `vmaf_dnn_sidecar_load` parser (ADR-0173) reads `quant_mode`;
  `vmaf_dnn_session_open` (ADR-0174) does the `.int8.onnx` redirect.

### 2. Default-mode `quantize_dynamic` (no `--per-channel`)

The PTQ tool is the unmodified `ai/scripts/ptq_dynamic.py` from
ADR-0173. We use the default (per-tensor) weight quantisation, not
`--per-channel`. Rationale:

| Variant | int8 size | PLCC drop (canonical 16-sample, seed 0) |
|---|---|---|
| Default per-tensor | 3 680 B | 0.000245 |
| `--per-channel` | 3 802 B | 0.000076 |

Both pass the 0.01 budget by two-to-three orders of magnitude. The
3× drop-improvement from per-channel is academic for a 257-parameter
model, and the per-tensor path matches the precedent already set by
`learned_filter_v1` (ADR-0174) and `nr_metric_v1` (T5-3c). Future
reviewers don't need to figure out why one tiny model got a different
flag.

### 3. PLCC validation against the 16-sample CI gate

`python ai/scripts/measure_quant_drop.py --all` (the
`ai-quant-accuracy` CI step from ADR-0174) reports
`vmaf_tiny_v2 mode=dynamic PLCC=0.999755 drop=0.000245
budget=0.0100 worst_abs=0.6494`. The worst-abs of 0.65 VMAF points
is on uniform-random `[0, 1]` input — this is **out-of-domain** for
a regressor whose StandardScaler is calibrated against real
canonical-6 distributions. PLCC stays the gate; absolute drift on
synthetic random inputs is not a quality signal for this model.

The `worst_abs` column is reported for parity with the other
quantised models, not as a gate.

### 4. No new C-side or build-side changes

This PR is **data-only**: a new `.int8.onnx` artefact, a registry
update, and a sidecar update. No new code, no new CLI flag, no new
meson option, no public-header touch. The runtime redirect (ADR-0174)
and the sidecar parser (ADR-0173) already cover this path.

## Alternatives considered

1. **Land per-channel `quantize_dynamic` instead of per-tensor.**
   Better PLCC headroom (0.000076 vs 0.000245 drop), 122 extra bytes.
   Rejected because the per-tensor path is already the precedent for
   the other two registered quantised models, and a 257-parameter
   model with an MLP head doesn't have a meaningful "channel" axis
   to amortise quantisation error against. Keeping all three models
   on the same flag profile cuts the cognitive surface for future
   maintainers.
2. **Ship static PTQ instead of dynamic.** Static would cut activation
   error too (vs dynamic, which only quantises weights offline),
   typically gaining another order of magnitude on PLCC. Rejected
   because static PTQ requires shipping a calibration `.npz` under
   `ai/calibration/` (≥ a few MB of canonical-6 feature vectors), and
   we're already at drop=0.000245 — 40× under budget. Per ADR-0129's
   policy, escalation order is: dynamic → static → QAT, gated by a
   concrete budget violation. We don't have one.
3. **Wait for `vmaf_tiny_v3` to be registered, then quantise it
   instead.** The brief's stated preference. Rejected because v3 is
   still in the experimental tree (`docs/ai/models/vmaf_tiny_v3.md`,
   `ai/scripts/eval_loso_vmaf_tiny_v3.py`) and the registry-promotion
   PR for v3 is a separate workstream. The same PTQ recipe applies
   the moment v3 lands; this ADR documents that as a sibling
   follow-up.
4. **Quantise both v2 and v3 in one PR.** Rejected per ADR-0129's
   audit-first directive — each quantisation gets its own per-model
   PR with its own PLCC measurement. v3's promotion to the registry
   itself is a separable decision (it changes the default tiny VMAF
   regressor); bundling these two reviews would conflate them.
5. **Skip the int8 sidecar entirely because the model is already
   <3 KB.** Rejected: same reasoning ADR-0174 used for
   `learned_filter_v1`. The size story is irrelevant; the speedup on
   int8-capable CPUs (VNNI / NEON-DotProduct) is the real win. The
   harness existing on master and not exercising `vmaf_tiny_v2`
   would also leave a coverage hole — when v3 (or v4) lands, the
   first PR to flip would be debugging the redirect against an
   untested model on top of an untested artefact pipeline.
6. **Sign the new `.int8.onnx` with a Sigstore bundle now.**
   Deferred. Sigstore bundles for the existing tiny models are
   referenced in registry but not yet committed (no
   `*.sigstore.json` files exist on disk for any model); the bundles
   are produced at release-please tag time. Adding a sigstore
   reference for `vmaf_tiny_v2.int8.onnx` would either lie about a
   non-existent file or pre-empt the release pipeline. Tracked as a
   release-pipeline follow-up.

## Consequences

**Positive:**

- Third registered model on the int8 path. The
  `ai-quant-accuracy` CI gate now exercises three different op mixes
  (FR fusion regressor, residual filter, NR metric) on each PR.
- `vmaf_tiny_v2` deployments on int8-capable CPUs get the int8
  kernel speedup transparently.
- Closes the "v2 still on fp32" coverage hole that
  `measure_quant_drop.py --all` flagged on master before this PR.

**Negative:**

- Net file growth (+1 234 bytes) for this specific model. Acceptable
  given the speedup story; called out explicitly so future readers
  don't expect the 2.4× shrink that ADR-0174 reported on
  `learned_filter_v1`.
- The PR's slug is `vmaf-tiny-v3-…` even though the artefact is
  `vmaf_tiny_v2.int8.onnx`. Documented in the Context section above
  to avoid future-grep confusion.

**Neutral:**

- No effect on the Netflix CPU golden gate (it doesn't exercise
  tiny-AI).
- No public C-API change.
- No MCP-side change.

## Tests

- `python ai/scripts/measure_quant_drop.py --all` →
  ```
  [PASS] learned_filter_v1   mode=dynamic PLCC=0.999883 drop=0.000117 …
  [PASS] nr_metric_v1        mode=dynamic PLCC=0.992326 drop=0.007674 …
  [PASS] vmaf_tiny_v2        mode=dynamic PLCC=0.999755 drop=0.000245 …
  ```
- Manual 64-iteration probe (seed 0, batch 16, uniform `[0, 1]`):
  PLCC=0.999754, drop=0.000246, worst_abs=1.37 VMAF points (out-of-
  domain input).
- Schema validation:
  `python -c "import json,jsonschema; jsonschema.validate(
  json.load(open('model/tiny/registry.json')),
  json.load(open('model/tiny/registry.schema.json'))); print('OK')"`
  → `OK`.
- `ai-quant-accuracy` CI step from ADR-0174 covers the same call
  on every PR.

## Reproducer

```bash
# Quantise from the fp32 file:
python ai/scripts/ptq_dynamic.py model/tiny/vmaf_tiny_v2.onnx
# -> model/tiny/vmaf_tiny_v2.int8.onnx (3 680 bytes; 1.50× larger,
#    int8 kernel-speedup is the win, not size).

# Measure PLCC drop:
python ai/scripts/measure_quant_drop.py model/tiny/vmaf_tiny_v2.onnx
# Expected: PASS, drop ~ 0.000245.

# Validate registry + sidecar against the schema:
python -c "
import json, jsonschema
schema = json.load(open('model/tiny/registry.schema.json'))
reg    = json.load(open('model/tiny/registry.json'))
jsonschema.validate(reg, schema); print('OK')
"
```

## References

- [ADR-0129](0129-tinyai-ptq-quantization.md) — Proposed PTQ policy.
- [ADR-0173](0173-ptq-int8-audit-impl.md) — Audit-first harness.
- [ADR-0174](0174-first-model-quantisation.md) — First per-model PTQ
  (`learned_filter_v1`); precedent + runtime redirect this PR rides.
- [ADR-0244](0244-vmaf-tiny-v2.md) — `vmaf_tiny_v2` ship decision
  (the fp32 baseline this ADR quantises).
- [Research-0060](../research/0060-vmaf-tiny-v3-ptq-int8-feasibility.md) —
  PTQ feasibility analysis + held-out PLCC validation table.
- `req` — task brief 2026-05-04: "Produce an int8 PTQ sidecar for
  the existing `vmaf_tiny_v3` model (if v3 is in the registry;
  otherwise fall back to `vmaf_tiny_v2`)."
