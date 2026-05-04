- **Vulkan backend:** T-VK-1.4-BUMP step A — added per-result GLSL
  `precise` qualifiers on the FMA-contractable mul+add chains in
  `libvmaf/src/feature/vulkan/shaders/vif.comp` (the `g`, `sv_sq`,
  `gg_sigma_f` triple in the `integer_vif_scale2` output block) and
  `libvmaf/src/feature/vulkan/shaders/ciede.comp` (the YUV→RGB /
  RGB→XYZ matrix multiplies, the `xyz_to_lab_map` linear branch,
  the final Lab triple, the `get_upcase_t` four-term cosine
  accumulator, the `get_r_sub_t` chain, and every scalar local in
  `ciede2000`'s per-pixel composition). The qualifier maps to
  per-result `OpDecorate ... NoContraction` SPIR-V decorations (62
  in `vif.spv`, ~179 in `ciede.spv`). At the current Vulkan 1.3
  baseline the decoration is a no-op — `places=4` cross-backend
  numbers are unchanged on RADV / lavapipe / Intel anv — but
  unblocks the deferred `VK_API_VERSION_1_3 → VK_API_VERSION_1_4`
  bump (Step B), which under NVIDIA driver 595.x otherwise drifts
  `integer_vif_scale2` by 1.527e-02 (45/48 frames) and `ciede2000`
  by 1.67e-04 (42/48 frames). Verified on NVIDIA RTX 4090 / driver
  595.71.05 (ciede 1.67e-04 → 8.9e-05) and RADV (unchanged at
  8.3e-05). See [ADR-0264](../../docs/adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md)
  (parent), [ADR-0272](../../docs/adr/0272-vulkan-vif-ciede-precise-decorations.md)
  (this PR), and [research-0062](../../docs/research/0062-vulkan-precise-decoration-audit.md).
