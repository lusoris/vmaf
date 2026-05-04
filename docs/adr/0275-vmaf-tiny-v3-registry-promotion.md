# ADR-0275: vmaf_tiny_v3 — registry promotion

- **Status**: Accepted
- **Date**: 2026-05-04
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: ai, dnn, tiny-ai, model, registry, fork-local

## Context

[ADR-0241](0241-vmaf-tiny-v3-mlp-medium.md) accepted the decision to
ship `vmaf_tiny_v3.onnx` alongside `vmaf_tiny_v2.onnx`, and the trained
artifact + sidecar + model card landed in-tree. The registry promotion
itself was deferred — the ONNX file, the sidecar
`model/tiny/vmaf_tiny_v3.json`, and the model card
[`docs/ai/models/vmaf_tiny_v3.md`](../ai/models/vmaf_tiny_v3.md) all
exist, but `model/tiny/registry.json` carries no `vmaf_tiny_v3` entry.
The runtime tiny-model loader keys on the registry (sha256 trust root +
sidecar discovery), so until v3 is registered it cannot be selected by
`--tiny-model=vmaf_tiny_v3` and `--tiny-model-verify` cannot gate it.

PR #383 (the int8 PTQ sidecar workstream) tripped over this: that PR
was originally scoped to v3 but had to scope-shift back to v2 because
v3 was unregistered. Promoting v3 now unblocks #383 (and any future v3
follow-ups — multi-seed LOSO, KoNViD 5-fold eval, PTQ) without
re-litigating the ship decision in ADR-0241.

## Decision

Add the `vmaf_tiny_v3` row to `model/tiny/registry.json` mirroring the
v2 entry shape (kind `fr`, opset 17, BSD-3-Clause-Plus-Patent, sha256
of the in-tree ONNX, `smoke: false`). No `sigstore_bundle` field is
set — the bundle is generated at release time by the supply-chain
workflow, matching the v2 release-time convention. No `quant_mode` is
set — v3 ships fp32 (the model is already 4 496 bytes; PTQ is PR
#383's scope, not this one). Sidecar JSON, model card, and the trained
ONNX file are already in-tree from the ADR-0241 prep work and are not
modified.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Single-row registry edit (chosen)** | Minimal diff; reuses validated v2 row shape; doesn't touch v3's ADR-0241 ship decision | Requires a separate ADR for traceability | Correct — ADR-0241 ships the *what* (architecture, training recipe, alongside-v2 placement); ADR-0275 ships the *registry-row* mechanical promotion |
| Promote v3 inside PR #383 (the PTQ PR) | Single PR ships both v3 registration and v3 PTQ | Conflates two decisions; PR #383 already had to scope-shift to v2 because v3 was unregistered | Rejected — ordering is registration first, PTQ second; #383 rebases onto this once it lands |
| Defer until multi-seed v3 sweep | Higher-confidence single-seed-vs-multi-seed delta before promotion | Blocks #383 and any v3 consumer indefinitely on a follow-up that ADR-0241 already deferred | Rejected — ADR-0241 explicitly ships v3 on seed=0 with multi-seed deferred |
| Promote v3 + bump production default to v3 | One step instead of two | Contradicts ADR-0241's "production default stays v2" directive (the +0.0008 PLCC mean delta is below the Phase-3 multi-seed envelope) | Rejected — ADR-0241 is explicit about keeping v2 default; v3 is opt-in |

## Consequences

- **Positive**: `--tiny-model=vmaf_tiny_v3` resolves through the
  registry; `--tiny-model-verify` covers v3 once a Sigstore bundle is
  generated at release; PR #383 unblocks for a v3-targeted PTQ
  follow-up.
- **Negative**: registry trust-root row count grows by one; release-time
  Sigstore signing now has 11 entries instead of 10.
- **Neutral / follow-ups**: PTQ for v3 (PR #383's scope, rebases onto
  this); multi-seed v3 LOSO sweep (ADR-0241 follow-up backlog); KoNViD
  5-fold eval for v3.

## References

- Parent decision: [ADR-0241](0241-vmaf-tiny-v3-mlp-medium.md) — v3 ship
- Sibling: [ADR-0244](0244-vmaf-tiny-v2.md) — v2 promotion precedent
- Research digest reused (no new digest):
  [Research-0046 — vmaf_tiny_v3 mlp_medium evaluation](../research/0046-vmaf-tiny-v3-mlp-medium-evaluation.md)
- Model card: [`docs/ai/models/vmaf_tiny_v3.md`](../ai/models/vmaf_tiny_v3.md)
- Sidecar: [`model/tiny/vmaf_tiny_v3.json`](../../model/tiny/vmaf_tiny_v3.json)
- Validator: [`ai/scripts/validate_model_registry.py`](../../ai/scripts/validate_model_registry.py)
- Source: `req` (paraphrased per-user request: promote `vmaf_tiny_v3.onnx` from the experimental tree into `model/tiny/registry.json` so the fork's runtime tiny-model loader picks it up; PR #383's PTQ workstream had to scope-shift to v2 because v3 was unregistered, this PR fixes that).
