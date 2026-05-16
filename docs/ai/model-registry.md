# Tiny-model registry — schema and verification

The registry at [`model/tiny/registry.json`](../../model/tiny/registry.json)
is the **trust root** for libvmaf's tiny-AI surface. Every ONNX model
shipped under `model/tiny/` is indexed here with a SHA-256 pin, license
metadata, and a Sigstore bundle path. T6-9 / [ADR-0211](../adr/0211-model-registry-sigstore.md)
formalised the schema and wired `--tiny-model-verify` to `cosign verify-blob`.

## Registry shape

```jsonc
{
  "$schema": "./registry.schema.json",
  "schema_version": 1,
  "models": [
    {
      "id": "learned_filter_v1",          // kebab-case; --tiny-model=<id>
      "kind": "filter",                    // fr | nr | filter
      "onnx": "learned_filter_v1.onnx",
      "opset": 17,
      "sha256": "412d537…e27",
      "quant_mode": "dynamic",             // fp32 | dynamic | static | qat
      "int8_sha256": "1cff6fe…2d3",
      "quant_accuracy_budget_plcc": 0.01,
      "license": "BSD-3-Clause-Plus-Patent",
      "license_url": "https://github.com/lusoris/vmaf/blob/master/LICENSE",
      "sigstore_bundle": "learned_filter_v1.onnx.sigstore.json",
      "description": "Tiny residual filter for vmaf_pre — degraded → clean luma.",
      "notes": "Self-supervised on KoNViD-1k …"
    }
  ]
}
```

The full JSON Schema is at
[`model/tiny/registry.schema.json`](../../model/tiny/registry.schema.json).
Fields documented inline; key invariants:

- `sha256` is **lowercase hex, 64 chars**. Mismatch against the on-disk
  ONNX bytes is a hard error.
- `int8_sha256` is required iff `quant_mode != "fp32"`.
- `sigstore_bundle` is a path relative to `model/tiny/` and must end in
  `.sigstore.json`. The bundle file itself is generated at release time
  by [`.github/workflows/supply-chain.yml`](../../.github/workflows/supply-chain.yml);
  pre-release the path is declared but the file may be absent. The
  runtime verifier (`--tiny-model-verify`) treats absence as a fail-closed
  signal.
- `license` is an SPDX identifier when possible; for fork-trained models
  this is `BSD-3-Clause-Plus-Patent` (matches libvmaf). Upstream-derived
  models carry the upstream license verbatim (e.g. LPIPS-Sq is `BSD-2-Clause`).

## Validating the registry

The Python validator at
[`ai/scripts/validate_model_registry.py`](../../ai/scripts/validate_model_registry.py)
runs both the JSON Schema check and the cross-file consistency check
(every ONNX exists, every sha256 matches, every non-smoke entry has a
sidecar). It is a CI gate; run it locally before pushing:

```bash
python3 ai/scripts/validate_model_registry.py
# → OK: 5 registry entries valid against registry.schema.json
```

Pass a different registry / schema explicitly when working off-tree:

```bash
python3 ai/scripts/validate_model_registry.py /path/to/other-registry.json \
    --schema /path/to/registry.schema.json
```

The validator falls back to a structural check when `jsonschema` is not
installed, so distros without `python-jsonschema` still get the
required-field invariants. Install `jsonschema` for full Draft 2020-12
coverage.

## Runtime verification — `--tiny-model-verify`

```bash
vmaf -r ref.y4m -d dis.y4m \
     --tiny-model model/tiny/learned_filter_v1.onnx \
     --tiny-model-verify
```

When the flag is set the loader:

1. Looks up the model's basename in `model/tiny/registry.json`
   (alongside the `.onnx`, by default).
2. Reads the entry's `sigstore_bundle` path.
3. Spawns
   `cosign verify-blob --bundle=<path> --certificate-identity-regexp …
   --certificate-oidc-issuer https://token.actions.githubusercontent.com
   <onnx>` via `posix_spawnp(3p)` (no shell — explicit argv array).
4. Refuses to load on any non-zero exit, missing `cosign`, missing
   bundle, or missing registry entry.

The flag is **off by default** for dev-friendliness. Production
deployments should set it on. `cosign` must be on `$PATH`; install
prebuilt binaries from the [Sigstore release page](https://github.com/sigstore/cosign/releases).

The C entry point is `vmaf_dnn_verify_signature(onnx_path, registry_path)`
in [`libvmaf/include/libvmaf/dnn.h`](../../libvmaf/include/libvmaf/dnn.h);
both arguments are NULL-tolerant in the documented way.

## What the registry is *not*

- **Not** the inference contract — per-model input/output names,
  normalisation, and expected ranges live in the sidecar JSON next to
  the ONNX (`<basename>.json`). The registry stays small and easy to
  audit; the sidecar carries the runtime knobs.
- **Not** the operator-allowlist source of truth — that's
  `libvmaf/src/dnn/op_allowlist.c`. The registry pins identity; the
  allowlist constrains content.

## Adding a new model

1. Drop the `.onnx` and matching `<basename>.json` sidecar under `model/tiny/`.
2. Compute the sha256: `sha256sum model/tiny/<name>.onnx`.
3. Add an entry to `registry.json` — see existing entries as templates;
   the schema enforces required fields.
4. Bundle generation happens at release time; pre-release the
   `sigstore_bundle` path may point at a not-yet-existing file.
5. Run `python3 ai/scripts/validate_model_registry.py` and fix any reported issues.

The `/add-model <path>` skill scaffolds steps 1–4 for you.
