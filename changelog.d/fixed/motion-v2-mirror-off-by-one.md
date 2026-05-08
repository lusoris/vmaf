- **`motion_v2` vertical-edge mirror off-by-one fix** (upstream commit
  `856d3835`, "libvmaf/motion_v2: fix mirroring behavior, since
  a44e5e61"). The `mirror()` helper in `integer_motion_v2.c` and its
  AVX2 / AVX-512 / NEON twins selected `2 * size - idx - 1`
  (edge-replicating, a-b-c|c-b-a) for the bottom-edge tap of the 5-tap
  Gaussian blur. Upstream's matching `integer_motion.c` (motion v1)
  uses `2 * size - idx - 2` (skip-boundary, a-b-c|b-a); the two
  extractors are supposed to agree on the SAD-of-blur value modulo
  bit-shift precision, and the off-by-one introduced a 1-pixel offset
  on the bottom row of every frame. Effect on the Netflix 576x324
  golden pair is below the visible-precision threshold (`mean`
  unchanged at 4.322764 / 3.296707 for `motion_v2_sad` / `motion2_v2`),
  but adversarial inputs with strong bottom-edge content would diverge.
  Applied in lockstep across CPU scalar + AVX2 + AVX-512 + arm64 NEON
  twins (4 files). The GPU twins (cuda + sycl + hip + vulkan) are
  intentionally left at the old formula in this PR — flipping them
  would shift `scores_cpu_motion_v2_*.json` snapshots and is scoped to
  a follow-up PR with a regen-snapshots run; see Known follow-ups in
  `docs/rebase-notes.md` §0320.
