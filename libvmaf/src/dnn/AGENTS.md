# AGENTS.md — libvmaf/src/dnn

Orientation for agents working on the ONNX Runtime integration (tiny-AI
inference layer). Parent: [../../AGENTS.md](../../AGENTS.md).

## Scope

The C-side runtime for tiny-AI checkpoints. Sits between the feature
extractors and ONNX Runtime.

```
dnn/
  dnn_api.c / dnn_ctx.h    # public vmaf_dnn_* surface (opened from feature extractors)
  model_loader.c/.h        # loads model/tiny/registry.json, pins paths, checks sha256
  onnx_scan.c/.h           # wire-format scanner — walks ModelProto for banned ops
  op_allowlist.c/.h        # allowlist of ONNX ops we permit (no Scan, bounded Loop/If)
  ort_backend.c/.h         # thin wrapper over ONNX Runtime C API (session + tensors)
  tensor_io.c/.h           # tensor helpers (luma8, RGB + ImageNet normalisation)
  meson.build
```

Public API: [../../include/libvmaf/dnn.h](../../include/libvmaf/dnn.h). The
feature-extractor side consumes this API; no feature code talks to ONNX
Runtime directly.

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **Trust boundary**: any `.onnx` loaded via `--tiny-model` or the registry
  is untrusted input. `onnx_scan.c` is the gate; `op_allowlist.c` is the
  policy; `model_loader.c` does `realpath` + symlink-escape hardening.
  See [ADR-0039](../../../docs/adr/0039-onnx-runtime-op-walk-registry.md).
- **No skipping the scan**: `CreateSession` must not be called before
  `vmaf_dnn_validate_onnx` returns success.
- **Tensor bindings are named**: multi-input graphs bind by ONNX input name
  when `VmafDnnInput::name != NULL`; positional fallback is for single-input
  legacy paths only. See
  [ADR-0040](../../../docs/adr/0040-dnn-session-multi-input-api.md).
- **ImageNet normalisation lives in the graph**, not in the C helper —
  exporters absorb the inverse transform so the C side feeds tensors from
  the shared `vmaf_tensor_from_rgb_imagenet()` helper unchanged. See
  [ADR-0041](../../../docs/adr/0041-lpips-sq-extractor.md).
- **Every tiny-AI change ships docs** under `docs/ai/` in the same PR. See
  [ADR-0042](../../../docs/adr/0042-tinyai-docs-required-per-pr.md).
- **Tiny-AI extractor template is the dedup contract**
  ([ADR-0250](../../../docs/adr/0250-tiny-ai-extractor-template.md)).
  New tiny-AI feature extractors use the helpers in
  [`tiny_extractor_template.h`](tiny_extractor_template.h)
  (`vmaf_tiny_ai_resolve_model_path` / `vmaf_tiny_ai_open_session` /
  `vmaf_tiny_ai_yuv8_to_rgb8_planes` / `VMAF_TINY_AI_MODEL_PATH_OPTION`).
  The user-facing log lines (`<name>: no model path …`, `<name>:
  vmaf_dnn_session_open(<path>) failed: <rc>`) are wire-format-stable
  across extractors — downstream tooling greps them. Don't introduce
  per-extractor variants of the path / session-open shape; if the
  contract needs to change, update the helpers in one place. The recipe
  lives in
  [`docs/ai/extractor-template.md`](../../../docs/ai/extractor-template.md).
- **Registry schema is the trust contract** (T6-9 / [ADR-0211](../../../docs/adr/0211-model-registry-sigstore.md)).
  Every entry in [`model/tiny/registry.json`](../../../model/tiny/registry.json)
  must satisfy [`registry.schema.json`](../../../model/tiny/registry.schema.json):
  required `id` / `kind` / `onnx` / `sha256`, plus `license` and
  `sigstore_bundle` for `schema_version: 1` entries. New fields are
  added by extending the schema first, then the registry, then any
  consumers — never the other way around. The `--tiny-model-verify`
  path in `model_loader.c` parses the registry inline (no JSON dep) and
  spawns `cosign` via `posix_spawnp(3p)`; `system(3)` is and stays
  banned.

## Governing ADRs

- [ADR-0020](../../../docs/adr/0020-tinyai-four-capabilities.md) — the four capabilities.
- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — ORT runtime + execution-provider mapping.
- [ADR-0023](../../../docs/adr/0023-tinyai-user-surfaces.md) — CLI / C API / ffmpeg / training surfaces.
- [ADR-0036](../../../docs/adr/0036-tinyai-wave1-scope-expansion.md) — Wave 1 scope (LPIPS, MobileSal, TransNet V2, …).
- [ADR-0039](../../../docs/adr/0039-onnx-runtime-op-walk-registry.md) — op-allowlist walk + registry schema.
- [ADR-0040](../../../docs/adr/0040-dnn-session-multi-input-api.md) — multi-input/output API with named bindings.
- [ADR-0041](../../../docs/adr/0041-lpips-sq-extractor.md) — LPIPS-SqueezeNet extractor + ImageNet-in-graph.
- [ADR-0042](../../../docs/adr/0042-tinyai-docs-required-per-pr.md) — doc-substance rule.
- [ADR-0169](../../../docs/adr/0169-onnx-allowlist-loop-if.md) +
  [ADR-0171](../../../docs/adr/0171-bounded-loop-trip-count.md) —
  `Loop` + `If` admitted with bounded trip-count guard
  (`VMAF_DNN_MAX_LOOP_NODES = 16`); `Scan` stays rejected.
- [ADR-0258](../../../docs/adr/0258-onnx-allowlist-resize.md) —
  `Resize` admitted (op-type-only gate) for U-2-Net / mobilesal /
  saliency / segmentation models. Wire scanner stays op-type-only
  per ADR D39; consumers shipping their own ONNX should keep
  `mode in ("nearest", "linear")` (`cubic` not exercised in-tree).
- [ADR-0207](../../../docs/adr/0207-tinyai-qat-design.md) +
  [ADR-0208](../../../docs/adr/0208-learned-filter-v1-qat-impl.md)
  — QAT pipeline (PyTorch QAT → fp32 ONNX → ORT static-quantize
  bridge for PyTorch 2.11 ONNX-exporter limitations).

## Rebase-sensitive invariants (DNN-side surfaces in flight)

- **Op-allowlist additions for TransNet V2 (ADR-0257)**:
  `BitShift`, `GatherND`, `Pad`, `Reciprocal`, `ReduceProd`,
  and `ScatterND` are now load-bearing for
  `model/tiny/transnet_v2.onnx` (the upstream ColorHistograms +
  FrameSimilarity branches require all six). On rebase: removing
  any of them from `op_allowlist.c` is a model-breakage event;
  keep the trailing block above the `Loop` / `If` control-flow
  block intact. Future tiny-AI models that want to leverage
  these ops inherit them transparently.
- **Model registry + Sigstore (T6-9, PR #199 open, ADR-0211
  placeholder)**: `--tiny-model-verify` flag wires through to
  `cosign verify-blob` against the Sigstore bundle declared in
  the registry. Pairs with the `quant_mode` / `int8_sha256`
  fields from
  [ADR-0173](../../../docs/adr/0173-ptq-int8-audit-impl.md) /
  [ADR-0174](../../../docs/adr/0174-first-model-quantisation.md).
  On merge: every shipped tiny-AI model needs a Sigstore bundle
  path in `model/tiny/registry.json`.
- **MobileSal (T6-2a, PR #208 open, ADR-0218 placeholder)** —
  saliency feature extractor; opens session via `vmaf_dnn_*`.
- **TransNet V2 (T6-3a, PR #210 open)** — shot-boundary detector
  ~1M params; uses bounded-Loop guard from ADR-0171.
- **FastDVDnet (T6-7, PR #203 open, ADR-0215 placeholder)** —
  5-frame window pre-filter; same DNN session contract.

## Testing

```bash
meson test -C build --suite=dnn
```

Unit tests live under [../../test/dnn/](../../test/dnn/). CI also runs a
`--tiny-model` smoke gate loading a generated 1KB `smoke_v0.onnx` through
the full loader → scanner → session-open path.
