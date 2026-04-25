# Vulkan compute backend

> **Status: T5-1b runtime live + first kernel (`vif_vulkan`).**
> `vmaf_vulkan_context_new` / `vmaf_vulkan_buffer_*` initialise +
> allocate against a real Vulkan loader (volk + VMA + glslc). The
> first feature kernel — `vif_vulkan` — runs end-to-end on Intel
> Arc A380 and produces the standard
> `VMAF_integer_feature_vif_scale0..3_score` outputs. ADM /
> motion / motion_v2 kernels follow as T5-1c. Cross-backend
> bit-exactness gate against scalar / SYCL reference is the next
> milestone (T5-1b-v). See
> [ADR-0127](../../adr/0127-vulkan-backend-decision.md),
> [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md).

## What's in this PR

- Public header
  [`libvmaf/include/libvmaf/libvmaf_vulkan.h`](../../../libvmaf/include/libvmaf/libvmaf_vulkan.h)
  declaring `VmafVulkanState`, `VmafVulkanConfiguration`,
  `vmaf_vulkan_state_init` / `_import_state` / `_state_free`,
  `vmaf_vulkan_list_devices`, `vmaf_vulkan_available`.
- Backend tree under
  [`libvmaf/src/vulkan/`](../../../libvmaf/src/vulkan/) — `common.{c,h}`,
  `picture_vulkan.{c,h}`, plus a `meson.build` that's `subdir()`-included
  when `-Denable_vulkan=enabled`.
- Three feature kernel stubs under
  [`libvmaf/src/feature/vulkan/`](../../../libvmaf/src/feature/vulkan/) —
  `adm_vulkan.c`, `vif_vulkan.c`, `motion_vulkan.c`. Each declares
  `_init` / `_run` entry points returning `-ENOSYS` /
  do-nothing.
- Build system hookup: new `enable_vulkan` feature option (default
  **disabled**) in
  [`libvmaf/meson_options.txt`](../../../libvmaf/meson_options.txt);
  conditional `subdir('vulkan')` in
  [`libvmaf/src/meson.build`](../../../libvmaf/src/meson.build);
  `vulkan_sources` + `vulkan_deps` threaded through the `library()`
  call alongside the existing CUDA / SYCL / DNN aggregations.
- Smoke test at
  [`libvmaf/test/test_vulkan_smoke.c`](../../../libvmaf/test/test_vulkan_smoke.c) —
  4 sub-tests pinning the scaffold contract (context_new / NULL-out /
  destroy-NULL noop / device_count returns 0). Wired in
  [`libvmaf/test/meson.build`](../../../libvmaf/test/meson.build).
- New CI matrix row "Build — Ubuntu Vulkan Scaffold (stub kernels)"
  in
  [`.github/workflows/libvmaf-build-matrix.yml`](../../../.github/workflows/libvmaf-build-matrix.yml)
  that compiles with `-Denable_vulkan=enabled` to gate the scaffold
  against bit-rot.

## Building

```bash
meson setup build -Denable_cuda=false -Denable_sycl=false \
                  -Denable_vulkan=enabled
ninja -C build src/libvmaf.a
```

The scaffold has **zero runtime dependencies** — no Vulkan SDK,
no `volk`, no glslc, no VMA. Adding those is the responsibility
of the first real-kernel PR per ADR-0175 § "Alternatives
considered" (the rejected alternative was "pull all build deps in
now"; doing so would gate the scaffold's CI run on a Vulkan SDK
that no kernel uses yet).

## What lands next (rough sequence per ADR-0127)

1. Runtime PR: VkInstance / VkDevice / compute queue selection;
   wire `volk` for symbol loading; `dependency('vulkan')` becomes
   the build-time requirement.
2. **VIF as pathfinder** — first feature on the Vulkan compute
   path. Picks VIF specifically (per ADR-0127 § "Pathfinder
   selection") because its bit-exactness contract is well-defined
   and its arithmetic is easier to GPU-map than ADM's wavelet.
3. ADM, motion, the rest of the matrix.
4. CI lavapipe smoke — the Mesa software Vulkan implementation
   that runs without a real GPU. Lets every PR exercise the
   compute path on a stock GitHub-hosted Ubuntu runner.
5. `/cross-backend-diff` ULP gate — once kernels claim
   bit-exactness, the per-backend ULP diff joins the existing
   CPU / CUDA / SYCL trio.

## What's NOT in this PR

- No real Vulkan calls. Every entry point's body is a `TODO`.
- No `volk` / `vulkan-headers` / `glslc` / VMA dependencies.
- No `lavapipe` CI smoke — that requires the runtime PR's `volk`
  bring-up first.
- No bit-exactness claim — the kernels don't exist.
- No `/cross-backend-diff` integration — same reason.

## Caveats

- The `enable_vulkan` option is `feature` (auto/enabled/disabled)
  defaulting to **disabled**. Auto would silently flip on in
  builds that happen to have a Vulkan SDK installed; we want the
  scaffold to opt-in only until the runtime PR lands.
- The kernel stubs intentionally do not register with the feature
  registry — they would otherwise dispatch to no-op
  implementations on operators who flip `enable_vulkan` for the
  scaffold. Registration arrives with the runtime PR.

## References

- [ADR-0127](../../adr/0127-vulkan-backend-decision.md) — the
  Q2 governance decision to add a Vulkan backend.
- [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md) — the
  scaffold-only audit-first PR that ships this surface.
- [`/add-gpu-backend`](../../../.claude/skills/add-gpu-backend/SKILL.md)
  — the skill that produced the initial scaffold (subsequently
  hand-finished here).
