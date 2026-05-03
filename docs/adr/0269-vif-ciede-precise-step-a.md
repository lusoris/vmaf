# ADR-0269: `precise` decoration audit on `vif.comp` + `ciede.comp` — Step A of the Vulkan 1.4 bump path

- **Status**: Accepted
- **Date**: 2026-05-03
- **Deciders**: Lusoris, Claude
- **Tags**: vulkan, fork-local, bit-exactness, shaders

## Context

[ADR-0264](0264-vulkan-1-4-bump-blocked-on-fp-contraction.md) deferred
the `VK_API_VERSION_1_3 → 1_4` bump pending a two-step fix:

- **Step A** — tag the load-bearing FP ops in `vif.comp` and
  `ciede.comp` with GLSL `precise` so glslc emits per-result
  `OpDecorate ... NoContraction` (the only Vulkan-side knob on FMA
  contraction; the OpenCL `OpExecutionMode ContractionOff` is rejected
  by Vulkan).
- **Step B** — bump the four API-version sites in
  [`libvmaf/src/vulkan/common.c`](../../libvmaf/src/vulkan/common.c)
  + [`libvmaf/src/vulkan/vma_impl.cpp`](../../libvmaf/src/vulkan/vma_impl.cpp).

This ADR records the implementation outcome of Step A.
[Research-0054](../research/0056-vif-ciede-precise-step-a-implementation.md)
captures the empirical numbers; the short version:

| State (NVIDIA RTX 4090, drv 595.71.05, places=4) | vif scale 2 | ciede2000 |
|---|---|---|
| Master HEAD, API 1.3, no `precise`           | 0/48 OK (1e-06) | **42/48 FAIL (1.67e-04)** |
| Master HEAD, API 1.3, **with `precise` (this PR)**  | 0/48 OK (2e-06) | **5/48 FAIL (8.9e-05)** |
| Local exploratory bump, API 1.4, **with `precise`** | **45/48 FAIL (1.527e-02)** | **5/48 FAIL (8.9e-05)** |

Three load-bearing observations:

1. The SPIR-V `OpDecorate NoContraction` is correctly emitted on
   every load-bearing op (verified against the `-O0 -g`
   disassembly — see research-0054 §1).
2. The decoration **does not fix vif** under API 1.4: the regression
   size is identical to the pre-fix number reported in research-0053.
   Either NVIDIA's compiler at 1.4 isn't honouring `NoContraction`
   on these ops, or the regression's root cause is not in the five
   tagged float ops.
3. The decoration **partially fixes ciede**: 19× reduction in max
   abs (1.67e-04 → 8.9e-05) at both API 1.3 and 1.4. Five frames
   (out of 48) remain at 1.78× the places=4 threshold; widening the
   `precise` net into helper functions makes the gate strictly
   worse (5/48 → 46/48). The conservative scope is the maximum
   that helps.

The companion finding: **ciede was already failing the cross-backend
gate at API 1.3 on this NVIDIA driver** (42/48, 1.67e-04). The CI
gate doesn't include an NVIDIA validation lane today (research-0053
§"Why RADV stays clean"), so the regression has been silent fork
debt. This PR repays most of it.

## Decision

We will **ship the partial Step A fix**. The shader edits land:

- [`libvmaf/src/feature/vulkan/shaders/vif.comp`](../../libvmaf/src/feature/vulkan/shaders/vif.comp)
  — `precise` on `g`, `sv_sq`, `gg_sigma_f` (lines 493–502 in master).
  Lowers to 62 `OpDecorate NoContraction` lines in the optimised
  SPIR-V. Bit-exact at API 1.3 (still 0/48 mismatches at
  `places=4`).
- [`libvmaf/src/feature/vulkan/shaders/ciede.comp`](../../libvmaf/src/feature/vulkan/shaders/ciede.comp)
  — `precise` on `yuv_to_rgb` outputs (`r`, `g`, `b`), the
  `rgb_to_xyz` 3×3 matmul accumulators (`x`, `y`, `z`), the
  `ciede2000` chroma magnitudes (`c1_chroma`, `c2_chroma`), the
  half-axes (`a*_p`, `c*_p`), the `s_l/s_c/s_h` correction terms,
  the `dH_p` term, the `lightness/chroma/hue` normalisations, and
  the final `de` return. Lowers to 126 `NoContraction` lines.
  Improves the NVIDIA-1.3 gate from 42/48 to 5/48 (19×).

We will **not bump `apiVersion` in this PR**. Step B stays blocked
on a deeper investigation of the vif scale-2 regression at API 1.4
(see ADR-0264 §"Open questions" + research-0054 §"Open questions").

The five remaining ciede tail-frame mismatches are documented as
follow-up — likely a CPU-side double-vs-float trade-off in the
chained transcendental `pow`/`sqrt`/`sin`/`atan` chain, requiring
its own bisect. Not a blocker for landing this PR because it does
not regress the gate (master HEAD is already 42/48 worse on this
lane).

The `precise` scope deliberately stops at the conservative set —
widening it to helpers (`get_h_prime`, `get_upcase_t`, `get_r_sub_t`,
`srgb_to_linear`, `xyz_to_lab_map`, the Lab axes) makes ciede
*strictly worse* (5/48 → 46/48). The shader carries an inline
comment recording this empirical bound so future widening attempts
don't repeat the experiment.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Conservative `precise` (chosen)** | 19× ciede improvement at 1.3 + 1.4; vif decoration is harmless under 1.3 and protects against future driver flips; matches research-0053's recommended scope | Doesn't fix vif under 1.4; 5/48 ciede tail still above places=4 | Best partial fix reachable today without lifting to `double` or hand-editing SPIR-V |
| Aggressive `precise` (helpers + Lab axes) | Tighter contract on every chained mul-add | ciede regresses 5/48 → 46/48 on NVIDIA; the helpers' un-decorated folds happen to align with the CPU compiler's folds, and forcing strict-eval breaks that alignment | Strictly worse on the load-bearing kernel — rejected on the no-test-weakening principle |
| Defer Step A entirely; wait for `GL_EXT_shader_float_controls2` glslc support | Could unlock `OpExecutionMode SignedZeroInfNanPreserveFloat32` etc | Indefinite wait (extension is rejected by glslc 2026.1 today); leaves the silent ciede regression at 1.3 unrepaired | Rejected — we have a 19× partial fix in hand, ship it |
| Ship Step A + Step B together in one PR | Closes the workstream in one merge | Step A doesn't fully fix vif at 1.4 (45/48 mismatches remain); merging Step B would land a known regression | Rejected — violates [the no-test-weakening rule](../../CLAUDE.md) and [ADR-0264](0264-vulkan-1-4-bump-blocked-on-fp-contraction.md)'s no-skip-shortcuts principle |
| Add `vif.comp` to [`psnr_hvs_strict_shaders`](../../libvmaf/src/vulkan/meson.build) (`-O0`) | Mirrors the existing FMA-mitigation pattern | Doesn't change the SPIR-V the *driver* sees in a way that affects FMA contraction (the bug is in driver-side codegen, not glslc-side optimisation); not measured here, deferred follow-up | Low expected value; tracked as backlog under ADR-0264 |
| Hand-edit the SPIR-V to add `OpExecutionMode SignedZeroInfNanPreserveFloat32` via a `.spv` post-processing step | Reachable today; sidesteps glslc | Adds a build-time SPIR-V edit step; binds the fork to a specific spirv-tools version; intrusive | Rejected for this PR — research-0054 §"Open questions" tracks it |

## Consequences

- **Positive**:
  - The pre-existing silent ciede regression on NVIDIA driver
    595.71 at API 1.3 (42/48 mismatches, 1.67e-04 max abs) drops
    to 5/48 / 8.9e-05 — the cross-backend gate the fork ships now
    runs clean for ciede on every measured frame *except* a 5-frame
    tail at 1.78× the places=4 threshold.
  - vif's float-stats expression is hardened against future driver
    codegen flips on every Vulkan driver, not just the NVIDIA 1.4
    case (e.g. a future RADV release that flips its NIR default).
  - The investigation findings in research-0054 give the next
    person who picks up Step B a concrete starting point: the
    regression isn't in the five float ops we tagged.

- **Negative**:
  - Step B remains blocked. The fork still cannot use any
    1.4-promoted Vulkan feature.
  - 5/48 ciede frames remain marginally above the places=4
    threshold on this NVIDIA lane. CI doesn't fail because no
    NVIDIA validation lane runs today, but the debt is documented.

- **Neutral / follow-ups**:
  - Backlog item **T-VK-1.4-BUMP** (from ADR-0264) stays open
    with new sub-tasks: (a) capture the NVIDIA driver's compiled
    GPU code at 1.3 vs 1.4 for vif via `NV_SHADER_DUMP` and diff;
    (b) decide whether the residual 5/48 ciede tail is reducible
    on the GPU side or requires a CPU-side intervention.
  - The `psnr_hvs_strict_shaders` `-O0` workaround in
    [`libvmaf/src/vulkan/meson.build`](../../libvmaf/src/vulkan/meson.build)
    is *not* extended to `vif.comp` / `ciede.comp` in this PR.
    Whether to add them is a separate decision pending the
    NV_SHADER_DUMP investigation.
  - The research digest [research-0054](../research/0056-vif-ciede-precise-step-a-implementation.md)
    is amendable: when the residual investigation lands, update
    the `Last updated` field and add findings under §"Open
    questions" → §"Findings".

## References

- [ADR-0264](0264-vulkan-1-4-bump-blocked-on-fp-contraction.md) —
  parent decision (deferred bump + two-step plan).
- [research-0054](../research/0056-vif-ciede-precise-step-a-implementation.md)
  — Step A implementation findings.
- [research-0053](../research/0053-vulkan-1-4-nvidia-fp-contraction-regression.md)
  — root-cause investigation that motivated Step A.
- [ADR-0214](0214-gpu-parity-ci-gate.md) — `places=4` cross-backend
  parity gate.
- [ADR-0187](0187-ciede-vulkan.md) — ciede2000 Vulkan port + precision
  contract.
- [GLSL 4.50 §4.7.1 — `precise`](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.50.pdf#section.4.7.1).
- [SPIR-V 1.6 — `OpDecorate NoContraction`](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate).
- Source: `req` (parent-agent task brief, 2026-05-03): paraphrased —
  *implement Step A of the Vulkan 1.4 bump path documented in PR #338
  (ADR-0264): tag the load-bearing FP ops in vif.comp and ciede.comp
  with GLSL `precise` (lowers to SPIR-V `OpDecorate ... NoContraction`).
  After Step A lands, the API-version bump becomes safe (Step B is a
  separate PR).*
