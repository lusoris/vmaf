## Vulkan backend: two-level GPU reduction for VIF / ADM / motion (T-GPU-PERF-VK-3)

The per-workgroup accumulator readback bottleneck (59.73% CPU self-time at
1080p on discrete GPU, perf-hunt 2026-05-09) is eliminated for the three
highest-impact kernels.

**Before**: the host CPU read ~1.2 MB of per-WG int64 accumulator slots per
frame from PCIe BAR-mapped memory (uncached reads) and summed them in a loop.

**After**: a second compute dispatch (`vif_reduce.comp`, `adm_reduce.comp`,
`motion_reduce.comp`) reduces all per-WG slots on-GPU. The CPU reads only
56 bytes (VIF, per scale), 48 bytes (ADM, per scale), or 8 bytes (motion) —
**~424 bytes total** for all three kernels combined, a ~5,800× reduction.

The GPU reduction is bit-identical to the previous host loop: int64
two's-complement addition is commutative and associative. places=4
cross-backend parity is maintained on all target Vulkan drivers.

Requires: `VK_EXT_shader_atomic_int64` (`shaderBufferInt64Atomics`).

New API: `vmaf_vulkan_buffer_invalidate()` added to
`libvmaf/src/vulkan/picture_vulkan.{h,c}` — no public header change.

See [ADR-0350](docs/adr/0350-vulkan-two-level-gpu-reduction.md) and
[research digest 0091](docs/research/0091-vulkan-gpu-reduction-perf-analysis.md).
