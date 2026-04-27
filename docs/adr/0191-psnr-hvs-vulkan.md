# ADR-0191: float_psnr_hvs Vulkan kernel — overlapping 8×8 DCT blocks + per-plane log transform

- **Status**: Accepted
- **Date**: 2026-04-27
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: vulkan, gpu, feature-extractor, fork-local, places-2

## Context

[ADR-0188](0188-gpu-long-tail-batch-2.md) scopes batch 2 as
`ssim` → `ms_ssim` → `psnr_hvs`. ssim shipped via PRs #139 +
#140; ms_ssim via PRs #141 + #142. psnr_hvs is the last metric
in batch 2 and the first DCT-based GPU kernel in the fork.

The active CPU `psnr_hvs` extractor
([`third_party/xiph/psnr_hvs.c`](../../libvmaf/src/feature/third_party/xiph/psnr_hvs.c))
is the Xiph integer-DCT reference, ~470 LOC. Per-block flow
(matches `calc_psnrhvs`):

1. Slide an 8×8 window over `(W-7) × (H-7)` at **step=7**
   (one-pixel overlap row/column). Each plane (Y, Cb, Cr) is
   processed independently.
2. For each block, read the 8×8 source + distorted samples,
   compute per-block global mean, per-quadrant means (4 × 4×4
   sub-blocks via `((i & 12) >> 2) + ((j & 12) >> 1)`),
   per-block variances.
3. Forward `od_bin_fdct8x8` — Xiph's lifting-based integer
   DCT. 8 row-passes + 8 col-passes via `od_bin_fdct8` with
   per-stage `OD_DCT_RSHIFT` (unbiased right-shift). Output
   is `int32` DCT coefficients.
4. Compute masking thresholds: `s_mask = sqrt(s_mask · s_gvar) / 32`
   where `s_mask` accumulates `dct[i][j]² · mask[i][j]` over
   AC coefficients (skipping `[0][0]` DC) and `s_gvar` is a
   variance ratio (`(sum sub-vars) / global-var`). Same for
   distorted. Take `max(s_mask, d_mask)`.
5. Per-coefficient masked error: subtract
   `s_mask / mask[i][j]` from `|dct_s - dct_d|` per AC
   coefficient (DC unmasked); accumulate
   `(err · csf[i][j])²` into the running plane total.
6. After all blocks: `score = total / pixel_count / samplemax²`.
   Plane score in dB: `10·log10(1 / score)` via
   `convert_score_db(score, 1.0)`.
7. Combined: `psnr_hvs = 0.8·score_y + 0.1·(score_cb + score_cr)`,
   then `10·log10(1 / combined)`.

The CSF tables are per-plane (`csf_y`, `csf_cb420`, `csf_cr420`).
The mask table is `(csf · 0.3885746225901003)²` — pre-squared.

Provided features: `psnr_hvs_y`, `psnr_hvs_cb`, `psnr_hvs_cr`,
`psnr_hvs`. CPU rejects `bpc > 12` and `YUV400P` (no chroma).

## Decision

Ship `float_psnr_hvs_vulkan` as **one GLSL compute shader +
~700-LOC host orchestrator**. Per-plane invocation; results
combined on host.

### Shader: `psnr_hvs.comp`

One workgroup per 8×8 output block, one thread per coefficient
(8×8 = 64 threads/WG). Per-WG flow:

1. Cooperatively load the 8×8 source + distorted samples
   from device memory (each plane uploaded as `float`-domain
   ref/cmp at full precision via picture_copy).
2. Cooperatively compute global mean / per-quadrant means /
   global variance / per-quadrant variances using
   `subgroupAdd`-style reductions.
3. Run a **scalar `od_bin_fdct8x8`** in the WG's thread 0 —
   the Xiph DCT is data-dependent within a row (lifting
   stages with intra-stage RSHIFT) and parallelising it
   across 8 threads is not trivially worth it for an 8-tap
   1D pass. Two passes (row → col) with `int32` arithmetic;
   matches CPU bit-for-bit modulo language differences.
4. Per-thread compute its `(err · csf[i][j])²` masked
   contribution.
5. Per-WG `subgroupAdd` of all 64 contributions →
   `block_partial[wg_idx]` written to a per-block float
   buffer.
6. Host accumulates partials in `double` per plane,
   normalises by pixels + `samplemax²`, applies
   `10·log10(1/score)`.

DC step-by-step computation matches `calc_psnrhvs` line-for-line.

### Host orchestrator: `psnr_hvs_vulkan.c`

```
init:
    reject bpc > 12 + YUV400P (mirror CPU)
    allocate per-plane:
        ref + cmp float buffers (sized W×H per plane)
        block_partials buffer ((W-7)/7 × (H-7)/7)
        host pinned partials buffer for D2H
    bake CSF tables into shader specialisation constants
        (one specialisation per plane × bpc — 3 × 2 = 6 pipelines)

extract per frame:
    picture_copy each plane (uint → float in [0, 255] for 8-bit,
        scaled accordingly for 10/12)
    upload all 3 planes to device
    for plane in [Y, Cb, Cr]:
        dispatch psnr_hvs.comp with the plane's pipeline
            (correct CSF + mask via specialisation constants)
        readback partials
        sum on host in `double`
        score[plane] = total / pixels / samplemax²
        emit psnr_hvs_{y,cb,cr} = 10·log10(1/score[plane])
    combined = 0.8·score[0] + 0.1·(score[1] + score[2])
    emit psnr_hvs = 10·log10(1/combined)
```

### Precision contract

Target `places=2` per ADR-0188. The CPU integer-DCT is
deterministic across hosts (modular int32 arithmetic), but
GPU-side `int32` matches CPU exactly; the precision floor
comes from the **per-block float reductions** (`s_mask`,
`s_gvar`) and the **per-plane sum + log10**. Empirical floor
will be measured on the cross-backend gate fixture.

If empirical exceeds `places=2`, accept and document. If it
exceeds `places=1` (1e-2), investigate before accepting.

### Why one shader (not two)

`ssim_vulkan` (ADR-0189) used **two passes** (horiz + vert).
Why doesn't `psnr_hvs.comp` need that? The DCT is **per-block,
not per-row**. Each WG processes one independent 8×8 block
end-to-end. There's no cross-block dependency, so the whole
compute fits in one dispatch. The complexity is per-block
(64 threads × ~2 KB shared memory for the `int32` DCT
coefficients + `float` masking partials), well within compute
limits.

### Step=7 overlap handling

CPU iterates `for y in 0..H-7 step 7; for x in 0..W-7 step 7`.
Block grid is `((W-7)/7 + 1) × ((H-7)/7 + 1)`. Last partial
block past `W-7` is skipped — same on GPU. Dispatch grid =
`ceil((W-7)/7) × ceil((H-7)/7)`; per-WG check `if (block_x*7 + 7 > W
|| block_y*7 + 7 > H) return` masks out-of-bounds blocks.
Pixel count for normalisation = `block_count · 64` matches
CPU's `pixels` accumulator (incremented per coefficient).

### Chroma sampling

`extract` uses `pic->w[i]` / `pic->h[i]` per plane, so
chroma 4:2:0 is half-resolution. The shader doesn't care —
each plane gets its own dispatch sized to that plane's
`(w[i] - 7) × (h[i] - 7)` block grid. CSF picks the right
table per plane via specialisation constants.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| **Float DCT (AAN-style) instead of integer Xiph DCT** | Cleaner GLSL (no `int32` lifting / RSHIFT) | Diverges from CPU bit-exactness target; the `od_bin_fdct8` integer arithmetic is what AVX2 + NEON SIMD paths mirror | Xiph integer DCT is the contract — the masking math depends on its specific quantisation |
| **Per-thread block (one thread = one block, 64 blocks/WG)** | Higher WG occupancy | Each thread serially computes a full 8×8 DCT — no parallelism within a block; register pressure for 128-byte int32 storage per thread | One-WG-per-block keeps the DCT scalar (which is fine — 64 ops × 8 stages) and parallelises only the per-coefficient masking + accumulation |
| **Two-pass design like `ssim.comp`** | Reuse SSIM scaffolding | DCT is per-block, not separable across blocks; two passes would need a global write-then-read buffer for the DCT coefficients (~64× memory pressure vs in-shared) | One-pass is the natural fit — DCT, masking, accumulation all live in WG-shared memory |
| **Bit-exact `places=4` via emulating CPU's exact float reduction order** | Tighter precision | Per-block float ops in `s_gvar` / mask use float order-of-operations the GPU can't replicate without forcing serial reductions | Set by measurement, not by guessing — same approach as ciede / ms_ssim |

## Consequences

- **Positive**: matches CPU `psnr_hvs` at `places=2` on the
  gate fixture (subject to empirical measurement). One
  shader, one dispatch per plane (3 dispatches per frame).
  All Y/Cb/Cr scores + combined emitted from a single Vulkan
  pass per plane.
- **Negative**: 6 pipelines (3 planes × 2 bpc paths) — more
  pipeline objects than ssim/ms_ssim, but each is small
  (~100 SPIR-V ops). 3× picture_copy on the host per frame
  for the chroma planes. Negligible at 1080p.
- **Neutral / follow-ups**:
  1. **Batch 2 part 3b** — `float_psnr_hvs_cuda`. CUDA can
     reuse the `0x7` upload bitmask landed in PR #137 to
     get all three planes uploaded for free; the picture_cuda
     pipeline already produces the device-side ref/cmp.
  2. **Batch 2 part 3c** — `float_psnr_hvs_sycl`. Same shape
     as `ms_ssim_sycl`: self-contained submit/collect,
     host-pinned USM for picture_copy normalisation,
     `nd_range<2>` per-block kernel with `sycl::reduce_over_group`.
  3. **`bpc > 8` precision** — 10/12-bit inputs use the same
     integer DCT but with `(1 << depth) - 1` samplemax. Bake
     `samplemax²` into a specialisation constant so the
     bpc=8 vs bpc=10/12 split is just a different pipeline.

## Verification

- **Empirical**: 48 frames at 576×324 on Intel Arc A380 +
  Mesa anv vs CPU scalar — `max_abs = 8.2e-5` across all four
  metrics (`psnr_hvs_y` 8.2e-5, `psnr_hvs_cb` 3.2e-5,
  `psnr_hvs_cr` 3.8e-5, `psnr_hvs` 7.9e-5), `0/48
  places=3 mismatches`. Better than the ADR-0188 `places=2`
  floor — gate is set to `places=3` (threshold 5e-4).
- New CI step `psnr_hvs cross-backend diff (CPU vs
  Vulkan/lavapipe)` in
  `.github/workflows/tests-and-quality-gates.yml` runs at
  `places=3`.
- New entry in `cross_backend_vif_diff.py FEATURE_METRICS`:
  `psnr_hvs` with the four sub-features (`psnr_hvs_y`,
  `psnr_hvs_cb`, `psnr_hvs_cr`, `psnr_hvs`).
- The extractor name is `psnr_hvs_vulkan` (not
  `float_psnr_hvs_vulkan`) — matches the CPU extractor's
  `psnr_hvs` name without a `float_` prefix, since there is
  no fixed-point variant.

## References

- Parent: [ADR-0188](0188-gpu-long-tail-batch-2.md) — batch 2
  scope.
- Sibling: [ADR-0189](0189-ssim-vulkan.md) — ssim Vulkan.
- Sibling: [ADR-0190](0190-ms-ssim-vulkan.md) — ms_ssim
  Vulkan.
- CPU reference:
  [`third_party/xiph/psnr_hvs.c`](../../libvmaf/src/feature/third_party/xiph/psnr_hvs.c).
- AVX2 / NEON parity:
  [`x86/psnr_hvs_avx2.c`](../../libvmaf/src/feature/x86/psnr_hvs_avx2.c),
  [`arm64/psnr_hvs_neon.c`](../../libvmaf/src/feature/arm64/psnr_hvs_neon.c).
- Xiph DCT background: Ponomarenko et al., "On between-coefficient
  contrast masking of DCT basis functions", VPQM-07.
