# ADR-0328: Port upstream motion_v2 four-commit cluster (2026-05-08)

- **Status**: Accepted
- **Date**: 2026-05-08
- **Deciders**: lusoris, claude
- **Tags**: upstream-port, libvmaf, motion_v2, simd, cuda, sycl, vulkan, neon, ffmpeg-patches

## Context

Netflix/vmaf master grew a four-commit cluster on 2026-05-07/08 that
extends `motion_v2`'s option surface and tightens its mirror-padding
behaviour:

1. `856d3835` — `libvmaf/motion_v2: fix mirroring behavior, since
   a44e5e61` (CPU + AVX2 + AVX-512). Switches the mirror form from
   `2 * size - idx - 1` (edge-replicating, `a-b-c|c-b-a`) to
   `2 * size - idx - 2` (skip-boundary, `a-b-c|b-a`). Aligns motion_v2
   with classic `motion`'s `dev_mirror_motion`.
2. `c17dd898` — `libvmaf/motion_v2: add motion_max_val`. Adds the
   `motion_max_val` (alias `mmxv`) clamp option, plumbs
   `feature_name_dict` through `init` / `extract` / `flush` / `close`,
   and adds the `_with_dict` collector calls that allow option-bound
   model-side renaming via `VMAF_OPT_FLAG_FEATURE_PARAM`.
3. `a2b59b77` — `libvmaf/motion_v2: add motion_five_frame_window`.
   Adds the n-2 reference-picture path end-to-end:
   `feature_extractor.h::prev_prev_ref`, `VmafContext::prev_prev_ref`,
   thread-data plumbing in `libvmaf.c`, picture-pool sizing
   (`n_threads * 2` → `n_threads * 2 + 2`), CLI `pic_cnt` budget
   (`+ 1` → `+ 2`), and a new `motion_five_frame_window` (`mffw`)
   bool option that switches `extract` to read from `prev_prev_ref`
   and `flush` to use a windowed motion2 with `stride = window-1`.
   `flush` is also taught to lazy-create `feature_name_dict` so it can
   be safely called before `init` populates it.
4. `4e469601` — `libvmaf/motion_v2: port remaining options`. Adds
   `motion_force_zero`, `motion_blend_factor`, `motion_blend_offset`,
   `motion_fps_weight`, `motion_moving_average`, and the
   `VMAF_integer_feature_motion3_v2_score` provided feature emitted by
   `flush()` via `motion_blend_tools.h::motion_blend`. Reuses the
   `motion_max_val` clamp from commit 2.

[Research-0089](../research/0089-upstream-sync-coverage-2026-05-08.md)
reported `c17dd898` as silently present on fork master, but the
fork's `MotionV2State` had neither the `motion_max_val` field, the
`feature_name_dict`, the options table, nor the `_with_dict` collector
wiring. `c17dd898` is therefore included in this batch alongside the
nominal three. The corrected coverage is recorded in this ADR's
References below.

The fork carries CUDA / SYCL / Vulkan / NEON twins of motion_v2's
mirror-padding helper. To keep all backends consistent with the
upstream-CPU semantics the mirror fix is propagated to all four;
the new options (motion_max_val, motion_five_frame_window, the rest)
remain CPU-only because their consumers (option parsing, score
post-processing in `flush`) live in the host/CPU side and the kernels
only emit the raw SAD.

## Decision

We port the four-commit cluster as a single PR with four commits in
the order above. Each commit is a manual port (rather than a `git
cherry-pick`) because the fork's `integer_motion_v2.c` had been
reformatted to the project's clang-format style, which makes a textual
cherry-pick conflict on every formatting hunk; the semantic change is
applied directly. Commit messages cite the upstream SHA via
`(cherry picked from commit <sha>)`.

The mirror-padding fix is also propagated to four fork-internal twins
(NEON scalar `mirror`, CUDA `mv2_mirror`, SYCL `dev_mirror_mv2`,
Vulkan GLSL `dev_mirror`) plus the SIMD-vs-scalar audit's local copy
in `libvmaf/test/test_motion_v2_simd.c`. The new options added by
commits 2-4 stay CPU-only; the kernels do not consume them.

The fork's `read_pictures_update_prev_ref` helper (a fork-local
extraction not present upstream) is updated in-place to do the same
`prev_prev <- prev; prev <- ref` shift that upstream now does inline
in `vmaf_read_pictures`.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Skip the mirror fix on CUDA/SYCL/Vulkan/NEON twins | Smaller patch; backends keep their existing comment about "DIFFERS by one pixel" | The fork's `/cross-backend-diff` would now report a fresh per-frame divergence between CPU and GPU/SYCL/NEON for motion_v2; ADR-0214 GPU-parity gate flags it. | Fork already invests in keeping mirror semantics aligned; flipping CPU and leaving GPUs stale is the worse spot to be in. |
| Wait for upstream Batch D's new motion_v2 SIMD files before porting commit 1 | Single coordinated batch | Upstream's commit 1 (mirror fix) directly targets fork's existing `motion_v2_avx2.c` / `_avx512.c` (which the fork already has from PR #419's Vulkan-batch carry-over). Commit 1 stands alone and isn't blocked on Batch D. | Decoupling lands the fix sooner with a smaller diff. |
| Port the new options to GPU kernels in this PR | Single source of truth for behaviour | The kernels only compute SAD; the new options (`motion_force_zero`, `motion_blend_factor`, `motion_max_val`, `motion_fps_weight`, `motion_five_frame_window`, `motion_moving_average`) all act in `extract`/`flush` host-side post-processing. There is no kernel work to do. | No-op: the host-side CPU code path runs the post-processing for every backend. |
| Cherry-pick via `git cherry-pick -x` | Preserves upstream metadata | The fork has reformatted `integer_motion_v2.c` to project clang-format style; cherry-pick conflicts on every formatting hunk. | Manual port + `(cherry picked from)` trailer keeps the audit trail without the conflict-resolution churn. |

## Consequences

- **Positive**: motion_v2 is at upstream parity for the new option
  surface (`motion_max_val`, `motion_five_frame_window`,
  `motion_force_zero`, `motion_blend_factor`, `motion_blend_offset`,
  `motion_fps_weight`, `motion_moving_average`) and emits the new
  `motion3_v2` provided feature. The mirror-padding fix removes a
  one-pixel boundary divergence.
- **Negative**: Per-frame motion_v2 scores for clips with content at
  the right/bottom boundaries will shift by O(1 pixel) because of the
  mirror fix. The fork's `motion_v2` snapshot tests under
  `testdata/scores_cpu_*.json` do not exist (no fork snapshots for
  motion_v2), and Netflix golden tests do not assert motion_v2 raw
  values (they assert end-to-end VMAF score, which uses motion2_v2 in
  the regressor). Verified locally: Netflix golden gate passes after
  the port (50/50 unit tests, full Netflix golden pytest suite green
  with `enable_float=true`).
- **Neutral / follow-ups**:
  - Upstream is shipping new motion_v2 AVX2 / AVX-512 SIMD files in a
    later batch (Batch D). The fork already has `motion_v2_avx2.c` /
    `motion_v2_avx512.c` from a prior carry-over; that batch will
    likely be a no-op or a refactor-only landing.
  - HIP twin (`integer_motion_v2_hip.c`) is currently a stub and has
    no kernel mirror to update. When the HIP runtime PR (T7-10b)
    lands, its kernel must use `2 * size - idx - 2` from the start.
  - `ffmpeg-patches/` is unaffected: no public C-API surface, no
    `meson_options.txt` flag, no public header, no `vf_libvmaf` field
    is touched. Verified — see CLAUDE §14 ffmpeg-patches replay log
    in the PR description.

## References

- Upstream commits:
  - `856d3835` — mirror fix
  - `c17dd898` — motion_max_val
  - `a2b59b77` — motion_five_frame_window
  - `4e46960` — remaining options
- Source: `req` (direct user instruction in this session — port the
  motion_v2 three-commit cluster, plus motion_max_val if not present).
- Coverage correction: [Research-0089](../research/0089-upstream-sync-coverage-2026-05-08.md)
  reported `c17dd898` as silently ported. Ground truth (verified by
  inspection of `MotionV2State`, the absence of `feature_name_dict`,
  and the absence of an `options` table on master): not ported.
  Backfilled here.
- Related: ADR-0214 (GPU-parity CI gate), ADR-0123 (PREV_REF
  CUDA-only path null-guard), ADR-0152 (motion sliding-window
  monotonic-index requirement), ADR-0192 / ADR-0193 (CUDA motion_v2
  twin).
