# `u2netp_mirror` — model card (scaffold; binary upload pending)

> **Status — scaffold-only.** Per
> [ADR-0325](../../adr/0325-u2netp-fork-mirror-scaffold.md) this
> page documents a planned fork-local mirror of the upstream
> U-2-Net `u2netp` checkpoint. The binary itself
> (`model/u2netp_mirror.onnx`, optionally
> `model/u2netp_mirror.pth`) lands in a separate PR after the
> three open compliance questions in ADR-0325 §References are
> answered. Until then, **the recommended weights for the
> `mobilesal` extractor remain
> [`saliency_student_v1`](saliency_student_v1.md)** — see
> "When to use this" below.

This card follows the 5-point bar of
[ADR-0042](../../adr/0042-tinyai-docs-required-per-pr.md):
provenance, training recipe, op-allowlist coverage, deployment
contract, licence-compliance receipt.

## 1. Provenance

| Field                | Value                                                                |
| -------------------- | -------------------------------------------------------------------- |
| Upstream repository  | <https://github.com/xuebinqin/U-2-Net>                               |
| Upstream commit pin  | `ac7e1c81` (re-pinned at binary upload time to the audited HEAD)     |
| Upstream paper       | Qin et al., *U^2-Net*, Pattern Recognition 2020                      |
| Upstream model file  | `u2netp.pth` (~4.7 MB)                                               |
| Upstream license     | Apache-2.0 (no NOTICE file in upstream tree as of HEAD `ac7e1c81`)   |
| Fork release tag     | `u2netp-mirror-v1` (scheme pending compliance answer 2 in ADR-0325)  |
| Fork artefact path   | `model/u2netp_mirror.onnx` (gitignored; release attachment only)     |
| Fork artefact sha256 | _to be filled at binary upload_ — written into
                        `model/tiny/registry.json` if/when the registry registration follow-up lands |
| Sigstore bundle URL  | _to be filled at binary upload_ — emitted by
                        `.github/workflows/release-please.yml`'s `u2netp-mirror-attach` step |

## 2. Training recipe

The fork does **not** re-train this model. The mirrored
checkpoint is the upstream `u2netp` weights byte-identical to
the upstream Google Drive download. The upstream training recipe
(per the U-2-Net paper) is:

- Backbone: `U^2-Net` (nested U-structure) — small variant
  (`U2NETP`) with ~4.7 M parameters.
- Dataset: DUTS-TR (10 553 images).
- Loss: BCE + IoU multi-supervision over 7 saliency heads.
- Optimiser: Adam, lr `1e-3`, batch size `12`, 600 K iterations.
- Image size: 320×320 with random crop / horizontal flip.

For a fork-trained alternative on the same DUTS-TR corpus see
[`saliency_student_v1`](saliency_student_v1.md) — that one is
~113 K parameters (40× smaller) and ships under
BSD-3-Clause-Plus-Patent rather than Apache-2.0.

## 3. ONNX op-allowlist coverage

Upstream `u2netp` uses the following ONNX ops (after
`torch.onnx.export` at opset 17):

| Op             | On `op_allowlist.c`?              | Notes |
| -------------- | --------------------------------- | --- |
| `Conv`         | yes                               |
| `BatchNormalization` | yes                         |
| `Relu`         | yes                               |
| `MaxPool`      | yes                               |
| `Concat`       | yes                               |
| `Sigmoid`      | yes                               |
| `Resize`       | yes (added by [ADR-0258](../../adr/0258-onnx-allowlist-resize.md)) | Bilinear `F.upsample` lowers here. Was the axis-2 blocker in [ADR-0265](../../adr/0265-u2netp-saliency-replacement-blocked.md); resolved. |

Coverage at opset 17 is full — the converted ONNX graph loads
unchanged against the fork's wire-format scanner.

## 4. Deployment contract

- **Input tensor**: `input` `[1, 3, H, W]` `float32` (per
  upstream's `U2NETP.forward` signature).
- **Output tensor**: upstream emits 7 saliency heads
  (`d0..d6`); the rewrap selects `d0` (the highest-resolution
  multi-scale fusion) and renames it to `saliency_map`
  `[1, 1, H, W]`. This matches the C-side `feature_mobilesal.c`
  contract, so the mirror is a drop-in for the existing
  `mobilesal` extractor with no C changes.
- **Pre-processing**: tile luma into RGB (`Y → [Y, Y, Y]`),
  same as the existing `LumaAdapter` pattern from PR #326.
- **Post-processing**: take the per-frame mean of
  `saliency_map` for `saliency_mean` (no change from existing
  C extractor logic).

The deployment contract is intentionally identical to
`saliency_student_v1`'s — both models can be swapped in by
flipping the registry entry, no C-side rebuild needed.

## 5. Licence-compliance receipt

Apache-2.0 §4 (a)–(d) are addressed as follows (full walk in
[Research-0086](../../research/0086-u2netp-fork-mirror-license-compliance.md)):

- **§4 (a)** — full Apache-2.0 text ships at
  [`LICENSES/Apache-2.0-u2netp.txt`](../../../LICENSES/Apache-2.0-u2netp.txt)
  and is uploaded alongside the binary in every release that
  carries the mirror.
- **§4 (b)** — applies only to the ONNX rewrap (derivative
  work). The export script writes a `metadata_props` block on
  the ONNX graph stating "derived from xuebinqin/U-2-Net @
  <sha>; converted via torch.onnx.export opset 17; weights
  byte-identical to upstream `u2netp.pth`". Verbatim `.pth`
  redistribution is **not** a derivative-work modification, so
  §4 (b) is moot in that case.
- **§4 (c)** — attribution block in
  [`LICENSES/Apache-2.0-u2netp.txt`](../../../LICENSES/Apache-2.0-u2netp.txt)
  cites upstream copyright, paper, repository, and commit pin.
- **§4 (d)** — moot. Upstream tree carries no `NOTICE` file
  (verified against HEAD `ac7e1c81`).

The fork's own redistribution metadata (model card, registry
sidecar) does **not** modify the Apache-2.0 license; it is
"additional attribution notices" per §4 (d) ("…provided that
such additional attribution notices cannot be construed as
modifying the License").

## 6. When to use this

Prefer [`saliency_student_v1`](saliency_student_v1.md) by
default. It is fork-owned, ~40× smaller, ships under the same
license as the rest of `model/tiny/`, and was trained on the
same DUTS-TR corpus the upstream u2netp paper used.

Reach for `u2netp_mirror` when one of these applies:

- You are reproducing a published baseline that explicitly
  cites the upstream `u2netp` checkpoint and need
  byte-identical weights for citation.
- You are running a comparative evaluation where the
  upstream-trained signal is the ground truth and the
  fork-trained student is the candidate under test.
- Your downstream pipeline already pins to upstream u2netp
  behaviour (e.g. saliency masks were generated against
  upstream u2netp and you need the matching encoder hook).

In every case, run both models against your own validation set
before committing — the absolute scores differ (the upstream
model has ~40× more parameters; expect mIoU and saliency
distribution differences).

## 7. Operator workflow

The operator-facing fetch + verification recipe lives at
[`docs/ai/u2netp-mirror.md`](../u2netp-mirror.md). Short
version:

```bash
gh release download <tag> --repo lusoris/vmaf \
  --pattern 'u2netp_mirror_v*.onnx' \
  --pattern 'u2netp_mirror_v*.onnx.bundle' \
  --pattern 'Apache-2.0-u2netp.txt'

cosign verify-blob \
  --bundle u2netp_mirror_v1.onnx.bundle \
  --certificate-identity-regexp '^https://github\.com/lusoris/vmaf' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  u2netp_mirror_v1.onnx
```

The verify-blob step is mandatory before the binary is loaded by
any production pipeline — it gates on Sigstore's keyless OIDC
identity (`lusoris/vmaf` workflow + GitHub OIDC issuer), so a
tampered or wrong-origin binary fails verification.

## 8. Open compliance questions

These three are open until the user confirms before the binary
upload PR ships (paraphrased from ADR-0325 §References):

1. Apache-2.0 §4 redistribution read — confirm the licence text
   + attribution block at
   [`LICENSES/Apache-2.0-u2netp.txt`](../../../LICENSES/Apache-2.0-u2netp.txt)
   meets the fork's bar, or flag a stricter read.
2. Tag scheme — `u2netp-mirror-v1` (this PR's recommendation),
   or fold the asset into the main `vX.Y.Z-lusoris.N` release
   tag, or a different scheme.
3. Artefact format — verbatim `.pth`, ONNX rewrap only, or
   both.

## References

- [ADR-0325](../../adr/0325-u2netp-fork-mirror-scaffold.md) —
  the scaffold decision this card documents.
- [ADR-0265](../../adr/0265-u2netp-saliency-replacement-blocked.md)
  — the blocker decision this scaffold partially unblocks.
- [ADR-0286](../../adr/0286-saliency-student-fork-trained-on-duts.md)
  — the recommended primary path (`saliency_student_v1`).
- [ADR-0258](../../adr/0258-onnx-allowlist-resize.md) —
  `Resize` allowlist addition that resolves ADR-0265's axis-2
  blocker.
- [Research-0086](../../research/0086-u2netp-fork-mirror-license-compliance.md)
  — full compliance walk + alternatives table.
- Upstream paper: Qin et al., *U^2-Net*, Pattern Recognition
  2020.
- Upstream code: <https://github.com/xuebinqin/U-2-Net>.
