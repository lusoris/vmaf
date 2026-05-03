# HIP (AMD ROCm) compute backend (scaffold + eight kernel-template consumers)

> **Status: scaffold + eight kernel-template consumers (host
> scaffolding only).** Every entry point in
> [`libvmaf_hip.h`](../../../libvmaf/include/libvmaf/libvmaf_hip.h)
> currently returns `-ENOSYS` pending the runtime PR (T7-10b). The
> kernel-template consumers shipped or in flight are:
>
> 1. `integer_psnr_hip` (name `psnr_hip`) under
>    [ADR-0241](../../adr/0241-hip-first-consumer-psnr.md).
> 2. `float_psnr_hip` (name `float_psnr_hip`) under
>    [ADR-0254](../../adr/0254-hip-second-consumer-float-psnr.md).
> 3. `ciede_hip` (name `ciede_hip`) under
>    [ADR-0259](../../adr/0259-hip-third-consumer-ciede.md) — pins
>    the kernel-template's "no-memset bypass" path (single float per
>    block, no atomic).
> 4. `float_moment_hip` (name `float_moment_hip`) under
>    [ADR-0260](../../adr/0260-hip-fourth-consumer-float-moment.md) —
>    pins the kernel-template's "memset multiple uint64 counters"
>    path (four atomic counters in one kernel pass).
> 5. `float_ansnr_hip` (name `float_ansnr_hip`) under
>    [ADR-0266](../../adr/0266-hip-fifth-consumer-float-ansnr.md) —
>    pins the **interleaved (sig, noise) per-block float-partial**
>    readback shape (PR #340 in flight as of this PR).
> 6. `motion_v2_hip` (name `motion_v2_hip`) under
>    [ADR-0267](../../adr/0267-hip-sixth-consumer-motion-v2.md) —
>    pins the **temporal-extractor** shape (`flush()` callback +
>    ping-pong buffer carry, PR #340 in flight as of this PR).
> 7. `float_motion_hip` (name `float_motion_hip`) under
>    [ADR-0273](../../adr/0273-hip-seventh-consumer-float-motion.md) —
>    pins the **three-buffer ping-pong** (raw-pixel cache plus
>    blurred-frame ping-pong) and the `motion_force_zero` short-circuit
>    posture (this PR).
> 8. `float_ssim_hip` (name `float_ssim_hip`) under
>    [ADR-0274](../../adr/0274-hip-eighth-consumer-float-ssim.md) —
>    first **multi-dispatch** HIP consumer
>    (`chars.n_dispatches_per_frame == 2`); pins the five-intermediate
>    float-buffer pyramid and the v1 `scale=1` `-EINVAL` validation
>    surface (this PR).
>
> All eight register and are looked-up-able from the feature engine,
> but their `init()` returns `-ENOSYS` because the
> `kernel_template.c` helper bodies they call still return `-ENOSYS`
> (`float_ssim_hip` may instead return `-EINVAL` if the input
> dimensions trigger auto-decimation, mirroring `ssim_cuda`).
> The runtime PR flips all of them at once. Rollout cadence mirrors
> the Vulkan scaffold-then-runtime split that
# HIP (AMD ROCm) compute backend (scaffold + six host-scaffolded consumers)

> **Status: scaffold + six host-scaffolded kernel-template consumers.**
> Every entry point in
> [`libvmaf_hip.h`](../../../libvmaf/include/libvmaf/libvmaf_hip.h)
> currently returns `-ENOSYS` pending the runtime PR (T7-10b). The
> first six kernel-template consumers ship their host scaffolding
> across multiple PRs:
>
> 1. `integer_psnr_hip` — [ADR-0241](../../adr/0241-hip-first-consumer-psnr.md) (first / SSE accumulator).
> 2. `float_psnr_hip` — [ADR-0254](../../adr/0254-hip-second-consumer-float-psnr.md) (second / float partials, in flight as PR #324).
> 3. `ciede_hip` — [ADR-0259](../../adr/0259-hip-third-consumer-ciede.md) (third / `submit_pre_launch` bypass, in flight as PR #330).
> 4. `float_moment_hip` — [ADR-0260](../../adr/0260-hip-fourth-consumer-float-moment.md) (fourth / four-uint64 atomic counters, in flight as PR #330).
> 5. `float_ansnr_hip` — [ADR-0266](../../adr/0266-hip-fifth-consumer-float-ansnr.md) (fifth / interleaved (sig, noise) float partials, **this PR**).
> 6. `motion_v2_hip` — [ADR-0267](../../adr/0267-hip-sixth-consumer-motion-v2.md) (sixth / temporal extractor + ping-pong buffer carry, **this PR**).
>
> Each consumer registers under the name `<feature>_hip` and is
> looked-up-able from the feature engine, but its `init()` returns
> `-ENOSYS` because the `kernel_template.c` helper bodies it calls
> still return `-ENOSYS`. The runtime PR flips both at once.
> Rollout cadence mirrors the Vulkan scaffold-then-runtime split that
> [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md) /
> [ADR-0176](../../adr/0176-vulkan-vif-cross-backend-gate.md) used
> (T5-1 → T5-1b).

## What's in this PR

- Public header
  [`libvmaf/include/libvmaf/libvmaf_hip.h`](../../../libvmaf/include/libvmaf/libvmaf_hip.h)
  declaring `VmafHipState`, `VmafHipConfiguration`,
  `vmaf_hip_state_init`, `vmaf_hip_import_state`,
  `vmaf_hip_state_free`, `vmaf_hip_list_devices`,
  `vmaf_hip_available`. Mirrors the CUDA + Vulkan + SYCL surface
  shapes.
- Backend tree under
  [`libvmaf/src/hip/`](../../../libvmaf/src/hip/) — `common.{c,h}`,
  `picture_hip.{c,h}`, `dispatch_strategy.{c,h}`, plus a `meson.build`
  that's `subdir()`-included when `-Denable_hip=true`.
- Three feature kernel stubs under
  [`libvmaf/src/feature/hip/`](../../../libvmaf/src/feature/hip/) —
  `adm_hip.c`, `vif_hip.c`, `motion_hip.c`. Each declares `_init` /
  `_run` entry points returning `-ENOSYS` / do-nothing.
- Build wiring: new `enable_hip` boolean option (default **false**) in
  [`libvmaf/meson_options.txt`](../../../libvmaf/meson_options.txt);
  conditional `subdir('hip')` in
  [`libvmaf/src/meson.build`](../../../libvmaf/src/meson.build);
  `hip_sources` + `hip_deps` threaded through the
  `libvmaf_feature_static_lib` aggregation alongside CUDA / SYCL /
  Vulkan / DNN.
- Smoke test at
  [`libvmaf/test/test_hip_smoke.c`](../../../libvmaf/test/test_hip_smoke.c) —
  9 sub-tests pinning the scaffold contract (internal context
  lifecycle + every public C-API entry point's `-ENOSYS` / `-EINVAL`
  return). Wired in
  [`libvmaf/test/meson.build`](../../../libvmaf/test/meson.build).
- New CI matrix row "Build — Ubuntu HIP (T7-10 scaffold)" in
  [`.github/workflows/libvmaf-build-matrix.yml`](../../../.github/workflows/libvmaf-build-matrix.yml)
  that compiles with `-Denable_hip=true` to gate the scaffold against
  bit-rot. No ROCm SDK is installed on the runner — the scaffold has
  no SDK requirement.

## Building

```bash
meson setup build -Denable_cuda=false -Denable_sycl=false \
                  -Denable_hip=true
ninja -C build
meson test -C build
```

The scaffold has **zero hard runtime dependencies** — no ROCm SDK,
no `hipcc`, no `amdhip64`. The Meson build files include an
optional `dependency('hip-lang', required: false)` probe so a host
that already has ROCm installed will see the dependency resolve;
the scaffold compiles cleanly without it.

Adding the real linkage is the responsibility of the first kernel
PR per [ADR-0212](../../adr/0212-hip-backend-scaffold.md) §
"Alternatives considered" (the rejected alternative was "pull all
build deps in now"; doing so would gate the scaffold's CI run on a
ROCm SDK that no kernel uses yet).

## What lands next (rough sequence)

1. **Runtime PR (T7-10b)**: `hipInit` / `hipGetDeviceCount` /
   `hipDeviceGetName` probe; `hipStreamCreate` per state;
   `vmaf_hip_state_init` returns `0` on a real device. The
   smoke contract flips from "`-ENOSYS` everywhere" to
   "device_count >= 0, state_init succeeds when devices >= 1,
   skip when none".
2. **VIF kernel PR**: first feature on the HIP compute path (same
   pathfinder choice as Vulkan T5-1b — VIF's bit-exactness contract
   is well-defined and its arithmetic GPU-maps cleanly).
3. **ADM, motion, the long tail**: parity with the CPU + CUDA + SYCL,
   Vulkan kernel matrix.
3. **ADM, motion, the long tail**: parity with the CPU + CUDA + SYCL
   plus Vulkan kernel matrix.
4. **CI ROCm runner** (post-runtime): if and when GitHub Actions
   exposes AMD GPU runners — until then the runtime PR's smoke test
   skips cleanly on hosts with no devices, matching the
   [Vulkan lavapipe pattern](../vulkan/overview.md).
5. **`/cross-backend-diff` ULP gate** — once kernels claim
   bit-exactness, the per-backend ULP diff joins the existing
   CPU / CUDA / SYCL / Vulkan trio.

## What's NOT in this PR

- No real HIP calls. Every entry point's body is a `TODO` comment.
- No hard `dependency('hip-lang')` requirement (the `required: false`
  probe stays optional).
- No `hipify`-based CUDA-to-HIP translation layer — see ADR-0212 §
  "Alternatives considered" for why the fork keeps a hand-written
  HIP backend instead of auto-translating the existing CUDA path.
- No CI runner with a real AMD GPU — none currently exists in
  GitHub-hosted infrastructure.
- No bit-exactness claim — the kernels don't exist.
- No `/cross-backend-diff` integration — same reason.

## Caveats

- The `enable_hip` option is `boolean` defaulting to **false**.
  Operators who want the scaffold's smoke test to run pass
  `-Denable_hip=true` explicitly. The Vulkan scaffold uses a
  `feature` option instead for parity with `enable_dnn`; HIP follows
  the **boolean** convention used by `enable_cuda` and `enable_sycl`
  to keep the AMD/NVIDIA/Intel triad uniform — see ADR-0212 §
  "Decision".
- The original ADR-0212 scaffold deliberately did not register the
  ADM / VIF / motion stubs with the feature registry. ADR-0241
  flipped that posture for **integer PSNR**, ADR-0254 extended it
  to **float PSNR**, and ADR-0259 / ADR-0260 extended it to
  **ciede** and **float moment**. All four extractors
  (`vmaf_fex_psnr_hip`, `vmaf_fex_float_psnr_hip`,
  `vmaf_fex_ciede_hip`, `vmaf_fex_float_moment_hip`) are now in
  `feature_extractor_list` under `#if HAVE_HIP`, so a caller asking
  for any of those features by name (`vmaf --feature psnr_hip`,
  `--feature ciede_hip`, ...) gets a clean "extractor found, runtime
  not ready" surface (`-ENOSYS` at `init()`) instead of "no such
  extractor". The runtime PR (T7-10b) keeps these rows verbatim
  and adds the remaining siblings (ADM, VIF, motion). The
  remaining scaffold stubs (`adm_hip.c` / `vif_hip.c` /
  `motion_hip.c`) stay unregistered until they grow their own
  kernel-template consumer host scaffolding the same way these
  four have.
  ADM / VIF / motion stubs with the feature registry. ADR-0241 +
  ADR-0254 + ADR-0259 + ADR-0260 + ADR-0266 + ADR-0267 flip that
  posture for the six kernel-template consumers: `vmaf_fex_psnr_hip`,
  `vmaf_fex_float_psnr_hip`, `vmaf_fex_ciede_hip`,
  `vmaf_fex_float_moment_hip`, `vmaf_fex_float_ansnr_hip`, and
  `vmaf_fex_integer_motion_v2_hip` are now in
  `feature_extractor_list` under `#if HAVE_HIP`, so a caller asking
  for any of those features by name (`vmaf --feature psnr_hip`,
  `... --feature float_ansnr_hip`, `... --feature motion_v2_hip`,
  etc., or the C-API equivalent) gets a clean "extractor found,
  runtime not ready" surface (`-ENOSYS` at `init()`) instead of "no
  such extractor". The runtime PR (T7-10b) keeps these rows verbatim
  and adds its siblings (ADM, VIF, full motion). The remaining stubs
  stay unregistered until they grow their own first-consumer host
  scaffolding.
- HIP runtime types (`hipDevice_t`, `hipStream_t`) cross the public
  ABI as `uintptr_t`. This keeps `libvmaf_hip.h` free of
  `<hip/hip_runtime.h>`, mirroring the pattern Vulkan adopted in
  ADR-0184 (handles cross as `uintptr_t` so the public header is
  consumable without the SDK installed).

## References

- [ADR-0212](../../adr/0212-hip-backend-scaffold.md) — the
  scaffold-only audit-first PR that ships this surface.
- [ADR-0241](../../adr/0241-hip-first-consumer-psnr.md) — first
  kernel-template consumer (`integer_psnr_hip`); mirrors
  [ADR-0246](../../adr/0246-gpu-kernel-template.md)'s GPU template
  decision onto HIP.
- [ADR-0254](../../adr/0254-hip-second-consumer-float-psnr.md) —
  second kernel-template consumer (`float_psnr_hip`); float
  partials precision posture.
- [ADR-0259](../../adr/0259-hip-third-consumer-ciede.md) — third
  kernel-template consumer (`ciede_hip`); pins the
  kernel-template's `submit_pre_launch` bypass shape (no atomic,
  no memset).
- [ADR-0260](../../adr/0260-hip-fourth-consumer-float-moment.md) —
  fourth kernel-template consumer (`float_moment_hip`); pins the
  multi-counter uint64 readback shape so `submit_pre_launch` can
  later memset multiple counters in one helper call.
  second consumer (`float_psnr_hip`).
- [ADR-0259](../../adr/0259-hip-third-consumer-ciede.md) — third
  consumer (`ciede_hip`).
- [ADR-0260](../../adr/0260-hip-fourth-consumer-float-moment.md) —
  fourth consumer (`float_moment_hip`).
- [ADR-0266](../../adr/0266-hip-fifth-consumer-float-ansnr.md) —
  fifth consumer (`float_ansnr_hip`, this PR).
- [ADR-0267](../../adr/0267-hip-sixth-consumer-motion-v2.md) — sixth
  consumer (`motion_v2_hip`, this PR).
- [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md) — the
  Vulkan precedent this PR mirrors.
- [Research-0033](../../research/0033-hip-applicability.md) —
  AMD market-share + ROCm Linux maturity check.
- [`/add-gpu-backend`](../../../.claude/skills/add-gpu-backend/SKILL.md)
  — the skill that produced the initial scaffold (subsequently
  hand-finished here).
