## Added

- **Vulkan motion3 parity test** (`libvmaf/test/test_vulkan_motion3_parity.c`,
  T3-15(c) / ADR-0219): cross-backend C unit test that drives `motion_vulkan`
  and the CPU `motion` extractor on byte-identical synthetic frames and asserts
  `VMAF_integer_feature_motion3_score` matches at `places=4` (5e-5 absolute).
  Gracefully skips on hosts without a Vulkan-capable compute device.
  Closes the remaining Vulkan sub-item of T3-15(c).
