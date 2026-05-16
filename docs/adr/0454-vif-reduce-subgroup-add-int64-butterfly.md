# ADR-0454: Replace `subgroupAdd(int64_t)` with XOR-swap butterfly in `vif_reduce.comp`

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `vulkan`, `gpu`, `vif`, `correctness`, `bug`, `fork-local`, `t-vk-vif-1.4-residual`

## Context

The Vulkan VIF two-level reduction shader (`vif_reduce.comp`, ADR-0350) used bare
`subgroupAdd(int64_t)` calls for its per-thread accumulator reduction.  The
`vif.comp` Phase-4 reduction had already replaced the equivalent call with an
explicit XOR-swap butterfly (`reduce_i64_subgroup`) in response to
non-deterministic results observed on NVIDIA RTX 4090 + driver 595.71.05 + Vulkan
API 1.4 (research-0090 / ADR-0269).  The same NVIDIA driver bug routes
`subgroupAdd(int64_t)` through a broken int64 subgroup-add lowering path, producing
non-deterministic accumulator values at scale 2.

The `vif_reduce.comp` shader was authored after the `vif.comp` fix and was not
updated to apply the same workaround, leaving the residual ULP gap tracked as
`T-VK-VIF-1.4-RESIDUAL` (BACKLOG.md ~168-180).

## Decision

We replace every `subgroupAdd(t[i])` call in `vif_reduce.comp` with
`reduce_i64_subgroup(t[i])` — the same XOR-swap butterfly already used in
`vif.comp`.  The helper is copied verbatim from `vif.comp` with a cross-reference
comment.  `GL_KHR_shader_subgroup_shuffle` is added to the extension list, matching
`vif.comp`.  No C-side changes are required: the pipeline, descriptor sets, and
push-constant layout are unchanged.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep `subgroupAdd(int64_t)` + file a driver bug | No shader change | Non-deterministic on every affected NVIDIA + RADV user; no upstream fix timeline | Unacceptable correctness regression for shipping code |
| Split int64 into two `subgroupAdd(int32_t)` halves | Avoids shuffle extension | More code; carries + sign propagation error-prone | XOR butterfly is already proven in `vif.comp` and is simpler |
| Move reduction entirely to CPU | Eliminates GPU non-determinism | Reverts ADR-0350 bandwidth saving (56 bytes → full per-WG SSBO on PCIe) | Disproportionate performance cost for a one-function fix |

## Consequences

- **Positive**: Closes `T-VK-VIF-1.4-RESIDUAL`.  All Vulkan VIF accumulations
  (both the per-WG kernel and the second-level reducer) now use the same
  deterministic int64 subgroup reduction strategy.  NVIDIA RTX + RADV users
  running Vulkan API 1.4 are no longer affected by the non-determinism.
- **Negative**: Adds a `GL_KHR_shader_subgroup_shuffle` dependency to
  `vif_reduce.comp`.  This extension is supported on every device that supports
  `GL_KHR_shader_subgroup_arithmetic` (same tier), so the practical support
  surface is unchanged.
- **Neutral / follow-ups**: The `vif_reduce_spv.h` SPIR-V header must be
  regenerated from the updated shader source before the next build
  (`glslc vif_reduce.comp -o vif_reduce.spv` + `xxd -i`).  The build system
  auto-regenerates these headers as part of the Meson `generated_headers`
  custom target, so no manual step is needed in CI.

## References

- research-0090 — NVIDIA Vulkan 1.4 int64 subgroup-add non-determinism investigation.
- [ADR-0269](0269-vulkan-1-4-memory-model-barrier-fix.md) — Phase-3b barrier fix + `reduce_i64_subgroup` introduction in `vif.comp`.
- [ADR-0350](0350-vif-vulkan-two-level-gpu-reduction.md) — Two-level GPU reduction that introduced `vif_reduce.comp`.
- `T-VK-VIF-1.4-RESIDUAL` — BACKLOG.md ~168-180 tracking entry.
- `GL_KHR_shader_subgroup_shuffle` Vulkan extension specification.
