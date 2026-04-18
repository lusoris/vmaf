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

## Governing ADRs

- [ADR-0020](../../../docs/adr/0020-tinyai-four-capabilities.md) — the four capabilities.
- [ADR-0022](../../../docs/adr/0022-inference-runtime-onnx.md) — ORT runtime + execution-provider mapping.
- [ADR-0023](../../../docs/adr/0023-tinyai-user-surfaces.md) — CLI / C API / ffmpeg / training surfaces.
- [ADR-0036](../../../docs/adr/0036-tinyai-wave1-scope-expansion.md) — Wave 1 scope (LPIPS, MobileSal, TransNet V2, …).
- [ADR-0039](../../../docs/adr/0039-onnx-runtime-op-walk-registry.md) — op-allowlist walk + registry schema.
- [ADR-0040](../../../docs/adr/0040-dnn-session-multi-input-api.md) — multi-input/output API with named bindings.
- [ADR-0041](../../../docs/adr/0041-lpips-sq-extractor.md) — LPIPS-SqueezeNet extractor + ImageNet-in-graph.
- [ADR-0042](../../../docs/adr/0042-tinyai-docs-required-per-pr.md) — doc-substance rule.

## Testing

```bash
meson test -C build --suite=dnn
```

Unit tests live under [../../test/dnn/](../../test/dnn/). CI also runs a
`--tiny-model` smoke gate loading a generated 1KB `smoke_v0.onnx` through
the full loader → scanner → session-open path.
