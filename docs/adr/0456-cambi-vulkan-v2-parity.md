# ADR-0456: CAMBI Vulkan v2 — parity gap closure

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: `vulkan`, `gpu`, `cambi`, `feature-extractor`, `fork-local`, `places-4`, `parity`

## Context

The v1 Vulkan CAMBI kernel (`ADR-0205` + `ADR-0210`) ships the Strategy II
hybrid: GPU handles the embarrassingly-parallel phases (derivative, 7×7 SAT
spatial mask, 2× decimate, 3-tap mode filter); the host CPU runs
`calculate_c_values` (sliding histogram) and top-K spatial pooling via
`cambi_internal.h`. The architecture guarantees `places=4` parity with the CPU
scalar path because the GPU phases are integer-bit-exact and the host residual
is the exact CPU code.

A systematic audit of `cambi_vulkan.c` against `cambi.c` revealed six parity
gaps — none in the GPU shaders, all in the host-side orchestration code — that
caused the Vulkan extractor to produce scores that diverge from the CPU
reference even in simple no-op cases. The gaps are documented in detail in
[research digest 0135](../research/0135-cambi-vulkan-v2-port-2026-05-16.md).

## Decision

Close all six v1↔CPU parity gaps in a single PR:

1. **Gap 1** (default constants): Correct `CAMBI_VK_DEFAULT_MAX_VAL` from 5.0
   to 1000.0, `CAMBI_VK_DEFAULT_WINDOW_SIZE` from 63 to 65,
   `CAMBI_VK_DEFAULT_VLT` from 1000.0 to 0.0, and `CAMBI_VK_MIN_WIDTH_HEIGHT`
   from 64 to 216 — all to match `cambi.c`.

2. **Gap 2** (`adjust_window_size` formula): Replace the geometric-mean float
   formula `sqrt(w*h)/sqrt(4K)` with the CPU's integer arithmetic
   `((ws*(w+h))/375) >> 4`. At 576×324 this produces `window=9` (CPU) vs `11`
   (v1 bug).

3. **Gap 3** (`high_res_speedup` window halving): Apply `(*ws+1)>>1` when the
   `cambi_high_res_speedup` option is active; v1 silently ignored it.

4. **Gap 4** (`tvi_for_diff` bisection): Rewrite the bisect to find the last
   sample where `tvi_condition(S)=true` (v1 searched the complementary
   direction, producing values near max instead of near the perceptual
   threshold). Also fix `vlt_luma` to match CPU `get_vlt_luma` exactly (off
   by one in v1).

5. **Gap 5** (`topk` / `cambi_topk` selection): Mirror the CPU's conditional
   `if (topk != DEFAULT) use topk else use cambi_topk`.

6. **Gap 6** (`c_values_histograms` sizing): Move histogram allocation to after
   `cambi_vk_init_tvi` and size it with the CPU's `v_band_size` formula.

GPU shaders (`cambi_derivative.comp`, `cambi_filter_mode.comp`,
`cambi_mask_dp.comp`, `cambi_decimate.comp`) require no changes — they are
correct. Strategy III (fully-on-GPU `calculate_c_values`) remains deferred
per `ADR-0205`.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Fix only gap 2 (window formula) as a minimal patch | Smallest diff, fastest review | Leaves other divergences; scores still wrong at non-default VLT/topk | All gaps are in the same file; fixing together is cleaner |
| Delay until Strategy III lands | No patch needed if the whole host residual is replaced | Strategy III is a separate ADR with its own timeline; gaps cause wrong scores NOW | Gaps cause incorrect scores in production |
| Add a hybrid-mode CPU fallback when defaults diverge | Graceful degradation | Masks the root bugs; no ULP clarity | Fixes preferred over workarounds |

## Consequences

- **Positive**: CAMBI Vulkan v2 is bit-identical to the CPU scalar path on the
  cross-backend gate (`ULP=0`, `places=4`) for the default parameter set.
- **Positive**: Non-default options (`topk`, `cambi_topk`, `high_res_speedup`,
  non-zero VLT) now also produce CPU-parity scores.
- **Negative**: `CAMBI_VK_DEFAULT_MAX_VAL` changes from 5.0 to 1000.0. Any
  pipeline that compared Vulkan CAMBI scores with CPU/CUDA scores and observed
  the expected clip-to-5 behaviour will see higher scores on GPU after this
  fix. This is a correct behaviour change.
- **Neutral**: GPU shaders unchanged; no SPIR-V regeneration required.
- **Follow-up**: Strategy III (fully-on-GPU c-values) remains the next target.

## References

- `ADR-0205`: CAMBI GPU feasibility spike; Strategy II architecture decision.
- `ADR-0210`: v1 CAMBI Vulkan integration.
- `ADR-0214`: GPU parity CI gate (`places=4`).
- `ADR-0345`: CAMBI × {CUDA, SYCL, HIP} port strategy.
- Research digest 0135: full gap analysis with numeric examples.
- Source: per user direction (agent task brief 2026-05-16).
