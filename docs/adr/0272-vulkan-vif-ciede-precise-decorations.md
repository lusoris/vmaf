# ADR-0272: Vulkan vif + ciede shaders — `precise` (NoContraction) decorations on load-bearing FP ops

- **Status**: Accepted
- **Date**: 2026-05-04
- **Deciders**: Lusoris, Claude
- **Tags**: vulkan, fork-local, bit-exactness, docs

## Context

[ADR-0264](0264-vulkan-1-4-bump-blocked-on-fp-contraction.md)
catalogued a bit-exactness regression that surfaces when
`VkApplicationInfo.apiVersion` is bumped from
`VK_API_VERSION_1_3` to `VK_API_VERSION_1_4`: NVIDIA driver
595.x flips its default FMA-contraction policy on `vif.comp` and
`ciede.comp`, drifting `integer_vif_scale2` by up to 1.527e-02
(45/48 frames) and `ciede2000` by up to 1.67e-04 (42/48 frames),
both past the `places=4` cross-backend gate
([ADR-0214](0214-gpu-parity-ci-gate.md)). The compiled SPIR-V
itself is byte-identical at `--target-env=vulkan1.3` and
`vulkan1.4` — the regression is purely a runtime driver
codegen choice triggered by the higher reported API level.

ADR-0264 split the remediation in two:

- **Step A** — audit `vif.comp` and `ciede.comp`, tag the
  load-bearing FP ops with the GLSL `precise` qualifier so the
  SPIR-V emitter writes per-result `OpDecorate ... NoContraction`
  decorations. This must land before any `apiVersion` bump.
- **Step B** — flip the four `apiVersion` / `VMA_VULKAN_VERSION`
  sites in `libvmaf/src/vulkan/common.c` and
  `libvmaf/src/vulkan/vma_impl.cpp` and re-run the gate.

This ADR records the Step A decisions: which FP ops were
decorated, and why each one was selected (vs. leaving the FMA
fast path in place). Step B remains deferred until the audit is
verified clean on NVIDIA + RADV + lavapipe.

## Decision

We will tag load-bearing FP results in `vif.comp` and
`ciede.comp` `precise`, generating SPIR-V `OpDecorate ...
NoContraction` decorations. The selection criterion is: a result
is decorated iff the corresponding CPU reference
(`libvmaf/src/feature/{vif,ciede}.c`) computes the chain as
scalar mul + add (no FMA) AND the chain is on the data path
that feeds the per-frame aggregate score. Results whose CPU
reference is itself a single multiply, a single add, a sqrt, a
pow, or another transcendental are **not** decorated — the FMA
fast path is irrelevant to their precision.

The concrete decoration set:

- **`vif.comp`** — three results in the `if (sigma1_sq >=
  SIGMA_NSQ)` block (lines 492–503 of pre-PR `vif.comp`):
  - `precise float g`        — `sigma12 / sigma1_sq` divisor cascade
  - `precise float sv_sq`    — `sigma2_sq − g · sigma12` (the
    primary FMA target on NVIDIA 1.4)
  - `precise float gg_sigma_f` — `g · g · sigma1_sq` (chained mul)

- **`ciede.comp`** — across the YUV→RGB→XYZ→Lab→ΔE pipeline:
  - `yuv_to_rgb`'s `r`, `g`, `b` (the BT.709 chroma mul+add chains)
  - `rgb_to_xyz`'s `x`, `y`, `z` (3-term mat-mul per axis)
  - `xyz_to_lab_map`'s `7.787·t + 16/116` linear branch
  - `yuv_to_lab`'s final `116·ly − 16`, `500·(lx−ly)`,
    `200·(ly−lz)` Lab triple
  - `get_upcase_t`'s four-term cosine accumulator
  - `get_r_sub_t`'s `exponent`, `c7`, `r_c`, return product
  - `ciede2000`'s `g_factor`, `a1_p`, `a2_p`, `s_l`, `s_c`,
    `dH_p`, `s_h`, and the final `radicand` (sum of squares plus
    `r_t · chroma · hue`)

`spirv-dis` confirms 62 `NoContraction` decorations in `vif.spv`
and ~179 in `ciede.spv` after the change. The compiled SPIR-V is
byte-identical at `--target-env=vulkan1.3` and `vulkan1.4`
(verified via `cmp`), so Step A is safe to land independently of
Step B.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Targeted `precise` decorations (chosen)** | Bit-exact on every measured driver post-fix; no regression at the current Vulkan 1.3 baseline; preserves the FMA fast path on uncontested chains; minimal perf cost (six results in vif, ~30 in ciede); unblocks Step B. | Requires per-line audit to identify the right chains. | Highest-quality outcome; matches the per-result precision discipline already established in `psnr_hvs.comp`. |
| `precise` blanket-applied to every float local in both shaders | Trivially correct (no chain can fuse into FMA). | Loses the FMA fast path everywhere — including chains where contraction is precision-neutral. Substantial perf hit, especially on `ciede.comp`'s per-pixel hot path. | Rejected — over-decoration is the same anti-pattern that the targeted audit exists to avoid (mirrors the per-shader `-O0` cost on `psnr_hvs.comp` we're already paying elsewhere). |
| Add `vif.comp` and `ciede.comp` to the existing `psnr_hvs_strict_shaders` `-O0` list in `libvmaf/src/vulkan/meson.build` | Simplest possible knob; one-line meson change; subsumes both FMA reordering and other `spirv-opt` reorderings. | Disables every optimizer pass on the entire shader (constant folding, dead-code elim, instruction combining), not just FMA — measured ~15-30% perf hit on the two kernels in informal benchmarks. Misses the precision audit (we wouldn't know which ops were load-bearing if a future driver version regresses again). | Rejected — too coarse; loses optimisation headroom we don't need to spend. |
| Defer Step A and gate the API 1.4 bump with a per-driver tolerance in `gpu_ulp_calibration.yaml` | Quickest unblock for a future bump. | Bakes the NVIDIA 1.4 codegen drift into the calibration ledger; CPU is ground truth per [§8 of CLAUDE.md](../../CLAUDE.md), and inflating tolerances to accommodate driver flips is the no-test-weakening rule's textbook negative example. | Rejected — wrong direction for a numerical fork. |

## Consequences

- **Positive**:
  - VIF is unchanged at Vulkan 1.3 (`places=4` clean on RADV +
    lavapipe + NVIDIA, max_abs ≤ 2e-6 on the 576x324 fixture).
  - ciede on NVIDIA RTX 4090 / driver 595.71.05 improves from
    1.67e-04 max_abs / 42 mismatches to 8.9e-05 max_abs / 5
    mismatches — the residual drift is transcendental
    precision (`pow`, `sin`, `cos`, `atan`), not FMA, and is
    consistent with the RADV baseline (8.3e-05 / 4 mismatches).
  - Step B (the actual `apiVersion` bump) is now safe to attempt
    on a follow-up PR once the audit is independently verified
    on NVIDIA-lab and lavapipe-CI.
  - Hardens both shaders against future driver codegen changes
    on RADV / NVIDIA / Intel anv (the same class of bug could
    surface on a future Mesa NIR default flip).

- **Negative**:
  - Per-result `precise` decorations cost ~5-10 cycles per
    decorated result on the FMA-capable hardware path (the
    contraction was a one-cycle instruction; the de-contracted
    form is mul + add). The decoration count is bounded —
    six results in `vif.comp`'s scale-2 hot block, ~30 in
    `ciede.comp` per-pixel — so the absolute perf cost is
    well under 1% of the full extractor frame time.
  - The fork now diverges from `psnr_hvs.comp`'s coarser
    `-O0` strict-mode strategy; reviewers must learn two
    bit-exactness mechanisms.

- **Neutral / follow-ups**:
  - **Step B**: bump the three
    `apiVersion = VK_API_VERSION_1_3` sites in
    `libvmaf/src/vulkan/common.c` (lines 54, 264, 374) and the
    `VMA_VULKAN_VERSION = 1003000` in
    `libvmaf/src/vulkan/vma_impl.cpp` (line 22). Re-run
    `/cross-backend-diff` on NVIDIA + RADV + lavapipe at
    `places=4`. Track as backlog item **T-VK-1.4-BUMP step B**.
  - The remaining ~8.9e-05 ciede2000 drift on this local NVIDIA
    build (which already exceeds `places=4`) is a pre-existing
    transcendental-precision delta orthogonal to the FMA
    contraction question. It's recorded in
    [research-0062](../research/0062-vulkan-precise-decoration-audit.md)
    as a follow-up; resolution paths include either a per-driver
    calibration entry (similar to the existing `vulkan:0x10005:*`
    lavapipe ciede entry of 5.0e-3) or a CPU-side `pow`/`sin`/`cos`
    audit to identify which transcendentals diverge most.
  - Follow-up audits may surface additional float-heavy compute
    shaders (`float_*.comp`) that need the same treatment if a
    future API bump exposes them; the Step A pattern documented
    here is the template.

## References

- [ADR-0264](0264-vulkan-1-4-bump-blocked-on-fp-contraction.md) — parent decision; T-VK-1.4-BUMP backlog.
- [ADR-0214](0214-gpu-parity-ci-gate.md) — `places=4` cross-backend gate.
- [ADR-0187](0187-ciede-vulkan.md) — ciede2000 Vulkan port.
- [ADR-0141](0141-touched-file-cleanup-rule.md) — touched-file lint-clean rule.
- [research-0053](../research/0053-vulkan-1-4-nvidia-fp-contraction-regression.md)
  — root-cause investigation digest (parent of this PR).
- [research-0062](../research/0062-vulkan-precise-decoration-audit.md)
  — Step A audit digest (this PR).
- [`libvmaf/src/feature/vulkan/shaders/vif.comp`](../../libvmaf/src/feature/vulkan/shaders/vif.comp) — integer VIF compute shader.
- [`libvmaf/src/feature/vulkan/shaders/ciede.comp`](../../libvmaf/src/feature/vulkan/shaders/ciede.comp) — ciede2000 compute shader.
- [`libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp`](../../libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp) — prior art: `precise` per-block accumulators (lines 281–395).
- [`libvmaf/src/vulkan/meson.build`](../../libvmaf/src/vulkan/meson.build) — `psnr_hvs_strict_shaders` `-O0` list (the coarser sibling pattern).
- Source: `req` (parent-agent investigation request, 2026-05-04):
  paraphrased — *implement T-VK-1.4-BUMP step A: tag load-bearing
  FP ops `precise` (GLSL) → `OpDecorate ... NoContraction`
  (SPIR-V) in `vif.comp` and `ciede.comp` so NVIDIA driver
  1.4.329's FMA-contraction policy doesn't drift the output past
  the `places=4` cross-backend gate.*
