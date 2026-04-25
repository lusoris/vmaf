# ARM NEON backend

libvmaf's aarch64 path uses ARMv8-A NEON intrinsics. Unlike the GPU
backends, NEON is **always built** when the host (or cross) compiler
targets aarch64 — there is no `-Denable_neon` toggle. Kernels live
under [`libvmaf/src/feature/arm64/`](../../../libvmaf/src/feature/arm64/)
and are dispatched at runtime via `vmaf_get_cpu_flags()`.

## Build

```bash
meson setup build           # NEON sources compile automatically on aarch64
ninja -C build
```

The only switch that affects NEON code generation is the global
`enable_asm` flag — `-Denable_asm=false` disables every SIMD path
(NEON included) and falls back to scalar C. See
[../../development/build-flags.md](../../development/build-flags.md).

## Runtime control

NEON dispatch is per-extractor. To force scalar fallback (debugging,
A/B against the reference) mask out the NEON ISA bit at the CLI:

```bash
vmaf --cpumask 0 ...        # disable every CPU SIMD ISA, scalar only
```

`--cpumask` accepts a 64-bit hex value matching the bits returned by
`vmaf_get_cpu_flags()`; passing `0` is the simplest "scalar only"
override.

There is no per-feature NEON disable flag — extractors that have a
NEON kernel will pick it whenever `--cpumask` allows it.

## Per-feature coverage

The table below tracks which extractors have a NEON kernel. Coverage
matches the `Backends` column in
[../../metrics/features.md](../../metrics/features.md).

| Feature        | NEON kernel | Notes                                                         |
|----------------|-------------|---------------------------------------------------------------|
| `vif`          | yes         | matches AVX2 path bit-for-bit                                 |
| `adm`          | yes         | matches AVX2 path bit-for-bit                                 |
| `motion`       | yes         | fixed-point legacy `motion`                                   |
| `motion_v2`    | yes         | pipelined fused-blur variant                                  |
| `float_motion` | yes         | float-pipeline twin                                           |
| `float_adm`    | yes         | float-pipeline twin                                           |
| `float_psnr`   | yes         | per-plane float PSNR                                          |
| `ciede`        | yes         | YUV → CIELAB ΔE                                               |
| `psnr`         | yes         | fixed-point per-plane                                         |
| `psnr_hvs`     | yes         | bit-identical to scalar — see [ADR-0160](../../adr/0160-psnr-hvs-neon-bitexact.md) |
| `ssim` / `float_ssim` | yes  | shared decimate kernel                                        |
| `float_ms_ssim`| yes         | 9-tap 9/7 wavelet decimate via `ms_ssim_decimate_neon`        |
| `ssimulacra2`  | yes         | bit-identical to scalar; see [ADR-0161](../../adr/0161-ssimulacra2-simd-bitexact.md) and follow-up ADR-0162 / ADR-0163 |
| `cambi`        | yes         | scalar fallback also retained                                 |
| `ansnr` / `float_ansnr` | yes | NEON via shared `ansnr_neon`                                  |

## Bit-exactness

NEON outputs are byte-identical to the scalar C reference for the
features that ship a determinism contract:

- `psnr_hvs` — pinned by ADR-0160; verified across all three Netflix
  golden pairs.
- `ssimulacra2` — pinned by ADR-0161 / ADR-0162 / ADR-0163;
  cross-host determinism via `vmaf_ss2_cbrtf` and the sRGB-EOTF LUT.
- `ms_ssim_decimate` — pinned by ADR-0125; per-lane `vfmaq_n_f32`
  with broadcast coefficients matches the scalar
  `fmaf` chain exactly.

Other extractors are numerically equivalent to their scalar twins
within `places=4` of the snapshot tolerance but do not carry an
explicit byte-identity contract. The Netflix golden CPU gate
(`make test-netflix-golden`) is the cross-arch correctness check.

## Build / CI matrix

The `Build — Ubuntu ARM clang (CPU)` job in the libvmaf build matrix
([`libvmaf-build-matrix.yml`][libvmaf-build-matrix]) runs on
`ubuntu-24.04-arm` against clang and exercises the full unit-test + tox
suite on real aarch64 hardware (not qemu).

[libvmaf-build-matrix]: ../../../.github/workflows/libvmaf-build-matrix.yml

`make test-netflix-golden` runs on aarch64 in the same matrix and
must remain green — see [`docs/principles.md`](../../principles.md)
§ 8 (Netflix golden gate).

## Limitations

- No per-feature override: every NEON kernel runs whenever
  `--cpumask` permits. To bisect a suspected NEON regression use
  `--cpumask 0` to drop to scalar across all extractors at once,
  then re-enable per-extractor by running individual `--feature`
  invocations.
- No GPU path on aarch64 yet. CUDA / SYCL / Vulkan backends compile
  for x86_64 only in the current matrix; the Vulkan scaffold
  (ADR-0175) is the planned cross-arch GPU surface.

## Related

- [../index.md](../index.md) — backend dispatch rules.
- [../x86/avx512.md](../x86/avx512.md) — x86 SIMD twin reference.
- [../../metrics/features.md](../../metrics/features.md) — full per-feature
  Backends column.
- [ADR-0125](../../adr/0125-ms-ssim-decimate-simd.md) — MS-SSIM decimate
  bit-exactness contract.
- [ADR-0160](../../adr/0160-psnr-hvs-neon-bitexact.md) — `psnr_hvs`
  NEON bit-exactness.
- [ADR-0161](../../adr/0161-ssimulacra2-simd-bitexact.md),
  [ADR-0162](../../adr/0162-ssimulacra2-iir-blur-simd.md),
  [ADR-0163](../../adr/0163-ssimulacra2-ptlr-simd.md) — SSIMULACRA2
  SIMD ports including NEON.
