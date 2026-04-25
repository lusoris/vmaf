# ADR-0178: Vulkan ADM kernel (T5-1c-adm)

- **Status**: Accepted
- **Date**: 2026-04-26
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: `vulkan`, `gpu`, `feature-extractor`, `numerical-correctness`

## Context

T5-1c extends the Vulkan kernel matrix beyond VIF to ADM, motion, and
motion_v2. PR #119 (commit `32e31e45`) landed the motion kernel; this
ADR covers the **ADM** half of T5-1c (the wavelet-based kernel that's
the largest and most complex of the three).

The CPU integer-ADM extractor at `libvmaf/src/feature/integer_adm.c`
(3527 LOC) implements a 4-scale DWT (CDF 9/7 / DB2 wavelet) followed
by per-band decoupling, CSF weighting, and contrast masking. The SYCL
implementation at `integer_adm_sycl.cpp` (1626 LOC) was the closest
existing GPU port; the Vulkan kernel mirrors its 5-stage pipeline:

1. DWT vertical pass.
2. DWT horizontal pass (yields LL, LH, HL, HH per scale).
3. Decouple + CSF fused.
4. CSF denominator reduction.
5. Contrast measure reduction.

Run for each of 4 scales = 20 stage executions per frame. The shader
uses one pipeline per `(scale, stage)` pair via `VkSpecializationInfo`
(16 pipelines after merging stages 4+5 into a single fused reduction
pass, matching the SYCL pattern).

## Decision

We will land a Vulkan compute kernel for `integer_adm` that produces
the standard `integer_adm2`, `integer_adm_scale0..3` outputs (plus the
underlying num/den intermediates and debug features when `debug=true`),
gated against the CPU scalar reference by a third lavapipe step inside
the existing `Vulkan VIF Cross-Backend (lavapipe, places=4)` lane.
The Arc nightly lane gets the matching step.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| One pipeline per (scale, stage), 16 total | Clear flow; matches SYCL structure | More pipelines to manage | Adopted; matches the structure proven on VIF (4 scales × 1 stage = 4 pipelines) |
| One mega-shader with per-stage `STAGE` push constant | Fewer pipelines to compile | Larger shader; harder to verify per-stage correctness | The 16-pipeline split is what the SYCL reference uses; symmetry simplifies the diff |
| Defer ADM to a follow-up PR | Smaller motion PR review | Two more weeks of partial Vulkan coverage; T5-1c stays open | User answered "Two PRs: motion + motion_v2 first, ADM second" — this is PR 2 of 2 |
| Pure-`int64` reductions across all scales | Maximum bit-exactness | Some host-side double-precision finalisation matches CPU exactly already (reduction-order doesn't affect places=4) | Hybrid `int64` partials + host double sum is what SYCL does and what motion/vif do; matches CPU at places=4 with ULP=0 baseline at default JSON precision |

The empirical baseline as of this commit is **ULP=0** vs the CPU
scalar reference for all 5 ADM metrics on the Netflix normal pair (48
frames at 576×324) AND the 1920×1080 checkerboard pair (3 frames). At
full IEEE-754 precision the residual on scales 1–3 is ~7e-7 from
host-side double-summation order; well under the `places=4` contract.

## Consequences

- **Positive**: Vulkan kernel matrix matches the SYCL/CUDA kernel sets
  for the production VMAF model (VIF + motion + ADM = the three
  features the default `vmaf_v0.6.1` model consumes). T5-1c closes.
  The cross-backend gate now covers all three features on every PR.
- **Negative**: 16 pipelines per ADM extractor = longer init time
  than VIF's 4. Acceptable since init runs once per `vmaf_init` call.
  The shader is the largest in the fork (~660 LOC GLSL); future Mesa
  Vulkan compiler bumps will recompile longer.
- **Neutral / follow-ups**:
  - **AIM and adm3 features**: not emitted (matches the SYCL ADM
    scope; AIM would need a separate kernel). No shipped model
    requires them.
  - The shared-memory `s_csf` / `s_cm` arrays in stage 3 are sized
    `[16]` (max 16 subgroups × 16-thread WG). For hardware with
    smaller subgroup sizes (e.g. wave64 on RDNA), the WG configuration
    may need re-sizing — fine for the current 256-thread WG with
    subgroup_size=32 default.
  - Motion3 (5-frame-window mode) remains deferred per ADR-0177.

## References

- ADR-0177 — Vulkan motion kernel + cross-backend gate generalization.
- ADR-0176 — Vulkan VIF cross-backend gate (the gate this kernel
  joins).
- ADR-0175 — Vulkan backend scaffold.
- ADR-0127 — Vulkan backend governance decision.
- PR #119 — T5-1c motion (commit `32e31e45`).
- PR #118 — T5-1b-v cross-backend gate + CLI (commit `50758ea8`).
- `req` — user direction 2026-04-25 (paraphrased): "Two PRs: motion +
  motion_v2 first, ADM second". Selected via popup. PR #119 closed
  the first; this ADR's PR closes the second.
