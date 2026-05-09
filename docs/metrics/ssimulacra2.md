# SSIMULACRA 2 extractor (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> The bulk of the SSIMULACRA 2 documentation already lives in
> [`features.md`](features.md) (§SSIMULACRA 2) and the
> per-backend overview pages; this stub is the discoverable
> entry point.

SSIMULACRA 2 (*Structural SIMilarity Unveiling Local And Compression
Related Artifacts*, version 2; Jyrki Alakuijala / Cloudinary /
libjxl) is a fork-added feature extractor that closes the modern-
codec quality-metric gap left by traditional SSIM and PSNR. It
combines a multi-scale SSIM-like structural comparison with an
**asymmetric** penalty that punishes lost texture energy more
heavily than added noise, and is the de-facto modern-codec quality
metric in the JPEG XL and AV1 image-compression communities.

## Invocation

```shell
vmaf --feature ssimulacra2 --reference ref.yuv --distorted dist.yuv ...
```

Output: one scalar per frame (`ssimulacra2`).

## Status and coverage

- **Design**: [ADR-0126](../adr/0126-ssimulacra2-extractor.md)
  (Status: Proposed — the C scalar implementation has shipped per
  [ADR-0130](../adr/0130-ssimulacra2-scalar-implementation.md); the
  ADR remains in Proposed state pending the per-feature parity
  matrix lock).
- **SIMD bit-exactness**: AVX2 / AVX-512 / NEON paths (see
  ADR-0161, ADR-0162, ADR-0163) and SVE2 ([ADR-0213](../adr/0213-ssimulacra2-sve2.md)).
- **GPU coverage**: CUDA, SYCL ([ADR-0206](../adr/0206-ssimulacra2-cuda-sycl.md)),
  Vulkan ([ADR-0201](../adr/0201-ssimulacra2-vulkan-kernel.md)).
- **Snapshot regression gate**:
  [ADR-0164](../adr/0164-ssimulacra2-snapshot-gate.md).

## See also

- [`features.md`](features.md) — full feature-extractor matrix
  (already carries the SSIMULACRA 2 row).
- [`docs/backends/vulkan/overview.md`](../backends/vulkan/overview.md),
  [`cuda/overview.md`](../backends/cuda/overview.md),
  [`sycl/overview.md`](../backends/sycl/overview.md) — per-backend
  kernel-coverage lists that include SSIMULACRA 2.
- [ADR-0126](../adr/0126-ssimulacra2-extractor.md) — design
  decision.
