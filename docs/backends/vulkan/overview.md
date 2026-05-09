# Vulkan compute backend

> **Status: T5-1c closed — full default-model coverage (vif + motion + adm).**
> `vmaf_vulkan_state_init` / `_import_state` / `_state_free` plumb
> the public state-level API; the CLI flags `--vulkan_device <N>`,
> `--no_vulkan`, and `--backend {auto,cpu,cuda,sycl,vulkan}` drive
> end-to-end execution on a real Vulkan ICD. Live extractors:
> `vif_vulkan` (4-scale VIF), `motion_vulkan` (motion + motion2),
> and `adm_vulkan` (4-scale ADM + adm2). All three are gated
> against the CPU scalar reference at `places=4` by the
> `Vulkan VIF Cross-Backend (lavapipe)` CI lane on every PR (one
> step per feature); the Arc-A380 nightly lane (advisory, parked
> until the self-hosted runner is registered) catches
> lavapipe-vs-real-driver drift. Empirical baseline on **Intel
> Arc A380 via Mesa anv** (the path the lavapipe lane mirrors):
> `places=4` clean for all three kernels, with max_abs ≤ 3.1e-5
> (essentially JSON %f precision noise on most metrics). The
> default `vmaf_v0.6.1` model runs end-to-end on Vulkan. The
> earlier "ULP=0" claim was bogus — the gate was running
> CPU-vs-CPU due to three latent build/state/script bugs fixed
> in PR #120; see ADR-0178 § "Bug history" for the corrected
> per-backend matrix and the kernel-side follow-ups for CUDA /
> SYCL / NVIDIA-Vulkan paths. See
> [ADR-0127](../../adr/0127-vulkan-backend-decision.md),
> [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md),
> [ADR-0176](../../adr/0176-vulkan-vif-cross-backend-gate.md),
> [ADR-0177](../../adr/0177-vulkan-motion-kernel.md),
> [ADR-0178](../../adr/0178-vulkan-adm-kernel.md).

## What's wired

- Public state-level API in
  [`libvmaf/include/libvmaf/libvmaf_vulkan.h`][hdr] —
  `VmafVulkanState`, `VmafVulkanConfiguration`,
  `vmaf_vulkan_state_init` / `_import_state` / `_state_free`,
  `vmaf_vulkan_list_devices`, `vmaf_vulkan_available`. The
  zero-copy import surface lands in T7-29: `vmaf_vulkan_state_init_external`
  (adopts caller-supplied VkInstance/VkDevice handles —
  required when those handles come from FFmpeg's
  `AVVulkanDeviceContext`), `vmaf_vulkan_import_image`,
  `vmaf_vulkan_wait_compute`, `vmaf_vulkan_read_imported_pictures`.
  See [ADR-0186](../../adr/0186-vulkan-image-import-impl.md).
  The import path runs the v2 async pending-fence ring
  (default depth 4, frames-in-flight pipelined) per
  [ADR-0251](../../adr/0251-vulkan-async-pending-fence.md);
  `vmaf_vulkan_import_image` is non-blocking and
  `vmaf_vulkan_wait_compute` drains every outstanding
  fence in submission order. Staging-buffer footprint
  scales `2 × ring_size` per state (~16 MiB host-visible
  at 1080p 8-bit Y, default depth). Ring depth is tunable
  via `VmafVulkanConfiguration.max_outstanding_frames` (0
  = default 4; clamped to [1, 8]); read back the clamped
  value with `vmaf_vulkan_state_max_outstanding_frames()`.
- Backend runtime under
  [`libvmaf/src/vulkan/`](../../../libvmaf/src/vulkan/) —
  `common.{c,h}` (volk + VkInstance / VkDevice / compute queue +
  VMA allocator + command pool), `picture_vulkan.{c,h}` (VkBuffer
  alloc / flush / mapped-host pointer accessors), `vma_impl.cpp`
  (VMA C++17 implementation TU).
- Live feature kernels under
  [`libvmaf/src/feature/vulkan/`](../../../libvmaf/src/feature/vulkan/) —
  - `vif_vulkan.c` + GLSL shader
    [`shaders/vif.comp`](../../../libvmaf/src/feature/vulkan/shaders/vif.comp).
    Four pipelines (one per `SCALE` specialization constant) compiled
    to SPIR-V via `glslc`, embedded as a byte array, dispatched in a
    single command buffer with pipeline barriers between scales.
  - `motion_vulkan.c` + GLSL shader
    [`shaders/motion.comp`](../../../libvmaf/src/feature/vulkan/shaders/motion.comp).
    Separable 5-tap Gaussian blur (`{3571, 16004, 26386, 16004, 3571}`,
    sum=65536) + per-workgroup `int64` SAD reduction; ping-pong
    blurred-frame storage between calls; motion2 emitted with a
    1-frame lag. `motion3` is now emitted in 3-frame window mode
    (T3-15(c) / [ADR-0219](../../adr/0219-motion3-gpu-coverage.md))
    via host-side `motion_blend()` post-processing of motion2
    plus optional moving-average; the 5-frame window mode
    (`motion_five_frame_window=true`) returns `-ENOTSUP` at
    `init()` since the GPU still uses a 2-deep blur ring.
  - `adm_vulkan.c` + GLSL shader
    [`shaders/adm.comp`](../../../libvmaf/src/feature/vulkan/shaders/adm.comp).
    4-scale CDF 9/7 DWT + decouple+CSF fused pass + per-band CSF
    denominator and contrast-measure reductions. 16 pipelines per
    extractor (one per `(scale, stage)`). Produces the standard
    `integer_adm2` + `integer_adm_scale0..3` outputs.
  - All three kernels use native `int64` accumulators
    (`GL_EXT_shader_explicit_arithmetic_types_int64`) for
    deterministic reductions matching the CPU integer reference.
  - `cambi_vulkan.c` (T7-36 / [ADR-0210](../../adr/0210-cambi-vulkan-integration.md))
    + GLSL shaders
    [`shaders/cambi_preprocess.comp`](../../../libvmaf/src/feature/vulkan/shaders/cambi_preprocess.comp),
    [`cambi_derivative.comp`](../../../libvmaf/src/feature/vulkan/shaders/cambi_derivative.comp),
    [`cambi_filter_mode.comp`](../../../libvmaf/src/feature/vulkan/shaders/cambi_filter_mode.comp),
    [`cambi_decimate.comp`](../../../libvmaf/src/feature/vulkan/shaders/cambi_decimate.comp),
    [`cambi_mask_dp.comp`](../../../libvmaf/src/feature/vulkan/shaders/cambi_mask_dp.comp).
    Strategy II hybrid: GPU services preprocess (scaffold, see ADR-0210),
    per-pixel derivative, the 7×7 spatial-mask SAT (one shader,
    `PASS=0/1/2` spec const for row-SAT / col-SAT / threshold), 2×
    decimate, and 3-tap separable mode filter; the
    precision-sensitive `calculate_c_values` sliding-histogram pass
    + top-K spatial pooling stay on the host. Bit-exact w.r.t. CPU
    by construction (every GPU phase is integer arithmetic + the
    host residual runs the unmodified CPU code on byte-identical
    buffers); cross-backend gate runs at `places=4`. Closes the
    GPU long-tail matrix terminus declared in
    [ADR-0192](../../adr/0192-gpu-long-tail-batch-3.md) — every
    registered feature extractor in the fork now has at least one
    GPU twin (lpips delegates to ORT EPs per
    [ADR-0022](../../adr/0022-inference-runtime-onnx.md)).
  - `psnr_vulkan.c` + GLSL shader
    [`shaders/psnr.comp`](../../../libvmaf/src/feature/vulkan/shaders/psnr.comp).
    Single plane-agnostic compute shader (per-pixel `(ref - dis)²`
    + per-WG `int64` reduction), dispatched three times per frame
    against per-plane buffers — Y, Cb, Cr. Per-plane width / height
    arrive via push constants; chroma sizing follows `pix_fmt`
    (4:2:0 → w/2 × h/2, 4:2:2 → w/2 × h, 4:4:4 → w × h); YUV400
    clamps to luma-only. Provided features: `psnr_y`, `psnr_cb`,
    `psnr_cr`. Bit-exact vs CPU `integer_psnr` on integer YUV
    (`max_abs_diff = 0.0` on the 576×324 lavapipe gate). See
    [ADR-0182](../../adr/0182-gpu-long-tail-batch-1.md) +
    [ADR-0216](../../adr/0216-vulkan-chroma-psnr.md).
  - `ssimulacra2_vulkan.c` — 6-scale pyramid extractor. Per-frame
    pipeline: host YUV→linear-RGB, host linear-RGB→XYB (replaces
    GPU XYB shader for bit-exactness per ADR-0201), GPU IIR blur
    (separable, 3-pole), GPU elementwise products, GPU SSIM +
    EdgeDiff per-WG reductions, host double-precision pooling.
    Precision: `places=2` (ADR-0192). The three host-side hot
    paths — YUV→linear-RGB, linear-RGB→XYB, and 2×2 box downsample
    — are SIMD-accelerated (AVX2 on x86-64, NEON on aarch64) via
    runtime dispatch in `init()` (ADR-0252). Measured speedup on
    576×324: 2× for the XYB kernel (cbrtf-bound), 3.2× for the
    downsample kernel. Bit-exact to the CPU extractor at the
    `memcmp` level (`test_host_xyb`, `test_host_downsample`). See
    [ADR-0201](../../adr/0201-ssimulacra2-vulkan-kernel.md) and
    [ADR-0252](../../adr/0252-ssimulacra2-host-xyb-simd.md).
- Build system: `enable_vulkan` feature option (default **disabled**)
  in [`libvmaf/meson_options.txt`](../../../libvmaf/meson_options.txt);
  conditional `subdir('vulkan')` in
  [`libvmaf/src/meson.build`](../../../libvmaf/src/meson.build);
  `vulkan_sources` folded into `libvmaf_feature_static_lib` so test
  binaries link them; `vulkan_deps` (volk + VMA + dependency on
  `glslc`) threaded through.
- CLI plumbing in
  [`libvmaf/tools/vmaf.c`](../../../libvmaf/tools/vmaf.c) +
  [`libvmaf/tools/cli_parse.c`](../../../libvmaf/tools/cli_parse.c) —
  `--vulkan_device <N>` (auto-pick = `-1`, default disabled) and
  `--no_vulkan`. Routing happens through
  `VMAF_FEATURE_EXTRACTOR_VULKAN = 1 << 5` and
  `compute_fex_flags()` in
  [`libvmaf/src/libvmaf.c`](../../../libvmaf/src/libvmaf.c) — the
  dispatcher prefers the Vulkan-flagged extractor over the CPU
  default whenever a Vulkan state has been imported.
- Cross-backend gate at [`cross_backend_vif_diff.py`][diff-script] —
  runs `vmaf` twice on the Netflix normal pair (CPU + Vulkan),
  diffs `integer_vif_scale0..3` at `places=4`. Two CI lanes: the
  `lavapipe` lane runs on every PR (Mesa software ICD on
  `ubuntu-24.04`); the Arc-A380 lane runs nightly (parked until a
  self-hosted runner with label `vmaf-arc` is registered).
- Smoke test at [`libvmaf/test/test_vulkan_smoke.c`][smoke-test] —
  pins the runtime contract (`device_count >= 0`, `context_new`
  succeeds when devices ≥ 1, NULL-safety, out-of-range rejection).

## Building

```bash
meson setup build -Denable_cuda=false -Denable_sycl=false \
                  -Denable_vulkan=enabled
ninja -C build
```

Build dependencies: `vulkan-headers`, `glslc` (from the Vulkan SDK
or `glslang` package), and a Vulkan loader at runtime (`libvulkan.so`
on Linux, supplied by Mesa for lavapipe / `vulkan-mesa-drivers`
for Intel anv / NVIDIA's proprietary stack / etc.).

The `volk` and `VulkanMemoryAllocator` (VMA) submodules are pulled
via Meson wrap files; no system install required.

### Static-archive builds (BtbN-style fully-static FFmpeg)

When libvmaf is built with `default_library=static -Denable_vulkan=enabled`,
volk's `vk*` PFN dispatchers are renamed to `vmaf_priv_vk*` at the C
preprocessor level via a force-included header
([`subprojects/packagefiles/volk/gen_priv_remap.py`](../../../libvmaf/subprojects/packagefiles/volk/gen_priv_remap.py)).
The rename lets `libvmaf.a` coexist with the Khronos `libvulkan.a`
in a fully-static link line (`gcc -static main.o libvmaf.a libvulkan.a
-ldl`) without GNU-ld multi-definition errors. See
[ADR-0185](../../adr/0185-vulkan-hide-volk-symbols.md) (shared case) +
[ADR-0198](../../adr/0198-volk-priv-remap-static-archive.md)
(static case).

## Using

```bash
# Auto-pick the first compute-capable device:
build/tools/vmaf --reference REF.yuv --distorted DIS.yuv \
                 --width W --height H --pixel_format 420 \
                 --bitdepth 8 --feature vif \
                 --vulkan_device 0 --json --output out.json

# List devices the runtime can see:
# (implemented as `vmaf_vulkan_list_devices`; CLI surface lands
# with the next runtime PR if needed.)
```

`--vulkan_device <N>` selects the Nth compute-capable device.
Without the flag, libvmaf runs on CPU exactly as before — Vulkan
is fully opt-in.

## Cross-backend gate

Run the lavapipe-equivalent locally with any Vulkan ICD (Arc anv,
NVIDIA proprietary, Mesa radv, etc.):

```bash
python3 scripts/ci/cross_backend_vif_diff.py \
  --vmaf-binary build/tools/vmaf \
  --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
  --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --width 576 --height 324 --places 4
```

Pass `--feature psnr` to gate `psnr_y`, `psnr_cb`, and `psnr_cr`
together against the CPU integer PSNR reference (per-plane
dispatches; ADR-0216).

The script exits 0 when every per-frame `integer_vif_scale0..3`
score agrees with the CPU scalar reference at the configured
decimal-place tolerance, 1 on a mismatch, 2 on a binary or
fixture failure. The default `places=4` matches the fork's GPU vs
CPU snapshot contract; the `--places` flag tightens (e.g.
`--places 6` for ULP-strict gating).

## Submit-pool hot-path optimization (ADR-0256 / ADR-0353)

All Vulkan kernels are progressively migrated from a per-frame allocation
pattern to a pre-allocated submit pool (`VmafVulkanKernelSubmitPool`,
ADR-0256). The pool pre-allocates command buffers and fences at `init()`
and recycles them each frame via `vmaf_vulkan_kernel_submit_acquire` /
`vmaf_vulkan_kernel_submit_end_and_wait`, eliminating per-frame
`vkAllocateCommandBuffers`, `vkCreateFence`, and `vkAllocateDescriptorSets`
from the hot-path frame loop.

**Migration status:**

| Kernel | Pool slots | Descriptor writes | PR |
|---|---|---|---|
| `adm_vulkan.c` | 1 | once at init (4 pre-allocated sets) | PR-A (#563) |
| `motion_vulkan.c` | 1 | per-frame (ping-pong blur cur/prev) | PR-A (#563) |
| `psnr_vulkan.c` | 1 | once at init (3 pre-allocated sets) | PR-A (#563) |
| `ssim_vulkan.c` | 1 | once at init | PR-B (ADR-0353) |
| `ciede_vulkan.c` | 1 | once at init | PR-B (ADR-0353) |
| `ms_ssim_vulkan.c` | 1 (decimate) + 5 (SSIM) | once at init (13 total sets) | PR-B (ADR-0353) |
| `motion_v2_vulkan.c` | 1 | per-frame (ping-pong ref_buf cur/prev) | PR-B (ADR-0353) |
| `float_psnr_vulkan.c` | 1 | once at init | PR-B (ADR-0353) |
| `float_motion_vulkan.c` | 1 | per-frame (ping-pong blur cur/prev) | PR-B (ADR-0353) |
| `ansnr_vulkan.c` | planned | planned | PR-C |
| `vif_vulkan.c` | planned | planned | PR-C |
| `ssimulacra2_vulkan.c` | planned | planned | PR-C |
| `cambi_vulkan.c` | planned | planned | PR-C |

**T-GPU-OPT-VK-4** (descriptor pre-allocation): kernels with fully-stable
SSBO handles call `vkUpdateDescriptorSets` once at `init()` and reuse the
pre-allocated set on every subsequent frame. Ping-pong kernels retain one
`vkUpdateDescriptorSets` per frame because the cur/prev buffer assignment
changes each frame.

The required tear-down ordering — pool destroy before pipeline destroy —
is documented in `libvmaf/src/feature/vulkan/AGENTS.md`.

## What lands next

- **T5-1c** — ADM, motion, motion_v2 Vulkan kernels using the
  same `vif_vulkan` scaffolding (lazy-context fallback +
  imported-state borrow path, `VkSpecializationInfo`-driven
  pipelines, host-side reduction).
- Self-hosted Arc runner registration to flip the `Vulkan VIF
  Cross-Backend (Arc A380, advisory)` lane from `if: false` to
  active.

## Caveats

- The `enable_vulkan` option is `feature` (auto/enabled/disabled)
  defaulting to **disabled**. Auto would silently flip on in
  builds that happen to have Vulkan headers installed; we want
  Vulkan to be explicit opt-in until the kernel matrix matches
  CUDA/SYCL.
- The lavapipe lane uses Mesa's software ICD; per-frame timings
  are not representative of GPU performance. Hardware perf
  numbers come from the Arc nightly lane (when registered).
- Vulkan 1.3 is required for the `int64`-arithmetic shader
  extension; older drivers reject the SPIR-V at pipeline
  creation. The runtime errors with `-ENOSYS` / `-ENODEV` and
  the CLI prints `problem during vmaf_vulkan_state_init`.
- **NVIDIA-hardware ciede2000 places=4 5/48 fork debt
  (T-VK-CIEDE-F32-F64).** On NVIDIA proprietary drivers (verified
  on RTX 4090 + driver 595.71.05 with PR #346 shader changes
  applied), `cross_backend_vif_diff.py --feature ciede --backend
  vulkan` reports 5/48 mismatches at max abs `8.9e-05` (1.78× the
  places=4 threshold). This is a **structural f32 vs f64
  precision gap** on the highest-ΔE frames — the CPU reference's
  `ciede.c::get_lab_color` runs the BT.709 → linear-RGB → XYZ →
  Lab chain in `double` while every Vulkan kernel runs in
  `float`. See
  [ADR-0273](../../adr/0273-ciede-vulkan-nvidia-f32-f64-precision-gap.md)
  +
  [research-0055](../../research/0055-ciede-vulkan-nvidia-f32-f64-root-cause.md).
  The CI lavapipe parity gate (places=4, 0/48) remains
  authoritative; NVIDIA hardware validation is a manual local
  gate. Tracked under
  [`docs/state.md`](../../state.md) Open bugs.

## References

- [ADR-0127](../../adr/0127-vulkan-backend-decision.md) — the
  Q2 governance decision to add a Vulkan backend.
- [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md) — the
  scaffold-only audit-first PR.
- [ADR-0176](../../adr/0176-vulkan-vif-cross-backend-gate.md) —
  this gate (T5-1b-v).
- [`/add-gpu-backend`](../../../.claude/skills/add-gpu-backend/SKILL.md)
  — the skill that produced the initial scaffold (subsequently
  hand-finished here).

[hdr]: ../../../libvmaf/include/libvmaf/libvmaf_vulkan.h
[smoke-test]: ../../../libvmaf/test/test_vulkan_smoke.c
[diff-script]: ../../../scripts/ci/cross_backend_vif_diff.py
