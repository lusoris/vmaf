- **libvmaf** (lusoris fork): port the upstream Netflix/vmaf
  `motion_v2` four-commit cluster (`856d3835` mirror skip-boundary
  fix + `c17dd898` `motion_max_val` clamp + `a2b59b77`
  `motion_five_frame_window` + `4e469601` remaining options).
  Adds the `motion3_v2_score` provided feature plus the new options
  `motion_force_zero` (`force_0`), `motion_blend_factor` (`mbf`),
  `motion_blend_offset` (`mbo`), `motion_fps_weight` (`mfw`),
  `motion_max_val` (`mmxv`), `motion_five_frame_window` (`mffw`),
  `motion_moving_average` (`mma`). Mirror skip-boundary fix
  propagated to the fork's CUDA / SYCL / Vulkan / NEON twins so all
  backends stay aligned with the new CPU semantics. CLI
  `--feature motion_v2=...` interface is unchanged; the new options
  are opt-in via the existing `:key=value` syntax. See
  [ADR-0325](../docs/adr/0325-port-upstream-motion-v2-cluster-2026-05-08.md).
