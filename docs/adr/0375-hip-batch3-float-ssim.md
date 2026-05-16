# ADR-0375: HIP batch-3 — `float_ssim_hip` real kernel

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `hip`, `gpu`, `build`, `feature-extractor`, `fork-local`

## Context

After ADR-0373 landed batch-2 (`float_motion_hip`), seven of eleven HIP feature
extractors remained at the `-ENOSYS` scaffold posture. The batch-3 goal is to
promote `float_ssim_hip` to a real HIP module-API consumer, following the pattern
established by PR #612 (`float_psnr_hip`, ADR-0254), batch-1 (ADR-0372), and
batch-2 (ADR-0373).

`float_ssim_hip` is the most architecturally complex of the tractable batch-3
targets: a two-pass separable 11-tap Gaussian (horizontal pass into five
intermediate float buffers, vertical pass + SSIM combine + per-block float
partial sum), directly mirroring `cuda/integer_ssim/ssim_score.cu`. It is
scale=1-only in v1, matching the CUDA and Vulkan twins.

There is no `float_ssim_cuda.c` in the repo — the CUDA side uses
`integer_ssim_cuda.c`. The HIP port targets the same algorithm exposed through
the `float_ssim` feature name on AMD GCN/RDNA hardware.

## Decision

Promote `float_ssim_hip` from the `-ENOSYS` scaffold to a real `#ifdef HAVE_HIPCC`
dual-path consumer.

**`float_ssim_hip`** (`hip/float_ssim/ssim_score.hip` + updated
`float_ssim_hip.c` / `float_ssim_hip.h`):

- Three kernels: `calculate_ssim_hip_horiz_8bpc`, `calculate_ssim_hip_horiz_16bpc`,
  `calculate_ssim_hip_vert_combine`.
- Warp size 64 (GCN/RDNA): `SSIM_WARPS_PER_BLOCK = 128 / 64 = 2` (vs CUDA's
  `128 / 32 = 4`). The shared-memory warp-sum array is sized accordingly.
- Five intermediate float device buffers (`d_ref_mu`, `d_cmp_mu`, `d_ref_sq`,
  `d_cmp_sq`, `d_refcmp`) allocated via `hipMalloc` (no wrapper struct, unlike
  the CUDA `VmafCudaBuffer` layer).
- Two luma staging buffers (`ref_in`, `cmp_in`) for HtoD copies
  (`hipMemcpy2DAsync`, `hipMemcpyHostToDevice`), consistent with T7-10b's
  "pictures arrive as CPU VmafPictures" posture.
- Both passes run on the same private stream; implicit stream ordering provides
  the happens-before between Pass 1 writes and Pass 2 reads.
- v1 constraint: `scale=1` only, enforced at `init()` with `-EINVAL`.
- Host: per-block float partials accumulated in `double`, divided by
  `(W-10)×(H-10)` → single `float_ssim` feature.

The host TU follows the `float_psnr_hip.c` `#ifdef HAVE_HIPCC` dual-path pattern:
helper functions that call HIP APIs are inside `#ifdef HAVE_HIPCC`; free-helpers
called from error labels are defined outside the guard with internal guards (mirrors
`float_psnr_hip_module_free`). Without `enable_hipcc`, `init()` returns `-ENOSYS`
(scaffold contract preserved, per ADR-0374).

`ssim_score` is added to the `hip_kernel_sources` dict in `libvmaf/src/meson.build`,
extending the `hipcc --genco` → HSACO → `xxd -i` pipeline established by ADR-0372.

HIP real-kernel count after this PR: **5 of 11 extractors**.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Promote `ciede_hip` instead | Simpler single-pass reduction | Less user-visible than SSIM; SSIM closes the more prominent quality metric gap | `float_ssim_hip` chosen for higher impact |
| Include `float_moment_hip` in same PR | Two more extractors done together | Mixed PR harder to review; float_ssim is already architecturally complex | Deferred to separate batch PR |
| Use `vmaf_hip_kernel_submit_pre_launch` to zero partials | Uniform pre-launch contract | SSIM uses per-block writes, not atomics — no zero needed; the helper is for atomic accumulator consumers | Not used; per-block partials are naturally populated by the kernel |

## Consequences

- **Positive**: `float_ssim_hip` produces real SSIM scores on AMD GCN/RDNA
  hardware when built with `enable_hip=true enable_hipcc=true`.
- **Positive**: CPU-only and HIP-without-hipcc builds preserve the `-ENOSYS`
  scaffold posture unchanged.
- **Positive**: `ssim_score.hip` joins the HSACO build pipeline; future
  GPU-specific SSIM work starts from working device code.
- **Negative**: One new HSACO fat binary in the build output; adds approximately
  15-30 s to `hipcc --genco` build time per target architecture.
- **Neutral**: v1 rejects `scale != 1` with `-EINVAL`, matching the CUDA and
  Vulkan twins. Lifting this constraint is a future batch item.

## References

- ADR-0254: `float_psnr_hip` — first real HIP kernel, establishes the
  `#ifdef HAVE_HIPCC` dual-path pattern.
- ADR-0372: HIP batch-1 (`integer_psnr_hip`, `float_ansnr_hip`).
- ADR-0373: HIP batch-2 (`float_motion_hip`).
- ADR-0374: Disabled-build `-ENOSYS` contract (scaffold posture invariant).
- ADR-0274: `float_ssim_hip` scaffold (eighth consumer).
- CUDA twin: `cuda/integer_ssim/ssim_score.cu`, `cuda/integer_ssim_cuda.c`.
