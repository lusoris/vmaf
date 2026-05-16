# DISTS extractor (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows once
> the `dists_sq` ONNX extractor lands in tree.

DISTS (*Deep Image Structure and Texture Similarity*; Ding 2020) is
the proposed FR companion to `lpips_sq`
([`docs/ai/models/lpips_sq.md`](../ai/models/lpips_sq.md)) — a
VGG-feature deep FR metric that combines a texture-similarity
term (channel-wise mean) with a structural-similarity term
(channel-wise variance), documented to correlate better than LPIPS
on synthetic-distortion benchmarks. DISTS will ship as a tiny-AI
feature extractor `dists_sq` mirroring the existing `lpips_sq`
shape (same five VGG layers, same NCHW float32 input convention,
same ImageNet normalisation).

Status: **Proposed** — the design is locked in
[ADR-0236](../adr/0236-dists-extractor.md); the implementation is
tracked as backlog item `T7-DISTS`. Once the extractor lands the
invocation will be `vmaf --feature dists_sq …`, parallel to
`--feature lpips_sq`.

## See also

- [`docs/metrics/features.md`](features.md) — the master feature-
  extractor matrix; `dists_sq` lands here once shipped.
- [`docs/ai/models/lpips_sq.md`](../ai/models/lpips_sq.md) — the
  shipped FR sister metric whose ABI shape DISTS mirrors.
- [ADR-0236](../adr/0236-dists-extractor.md) — design decision +
  Bristol VI-Lab actionable-item context.
