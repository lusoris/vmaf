# AGENTS.md — libvmaf/src/metal

Orientation for agents working on the Metal (Apple Silicon) backend.
Parent: [../../AGENTS.md](../../AGENTS.md). Mirrors
[`libvmaf/src/hip/AGENTS.md`](../hip/AGENTS.md) — the HIP and Metal
backends share an audit-first scaffold story, and Metal's `MTLDevice`
/ `MTLCommandQueue` / `MTLBuffer` API parallels HIP's `hipDevice_t`
/ `hipStream_t` / `hipMalloc` surface closely enough that the rebase
/ refactor contracts on this side track the HIP scaffold's.

## Scope

The Metal-side runtime (picture lifecycle, dispatch strategy, kernel
scaffolding template). Metal **feature kernels** live one level
deeper in [../feature/metal/](../feature/metal/).

```text
metal/
  common.{c,h}          # Metal context + (future) command-queue management
  picture_metal.{c,h}   # VmafPicture on a Metal device — stub
  dispatch_strategy.{c,h} # Feature-name → kernel routing — stub
  kernel_template.{c,h} # per-feature kernel scaffolding (T8-1 / ADR-0361)
  meson.build           # subdir() include from libvmaf/src/meson.build
```

## Backend status

**Scaffold + first consumer (motion_v2_metal)** — landed in T8-1
(ADR-0361, this PR):

- `common.c` / `kernel_template.c` / `picture_metal.c` /
  `dispatch_strategy.c` — every entry point returns `-ENOSYS`.
- Public header `libvmaf_metal.h` — every entry point returns
  `-ENOSYS`.
- `feature/metal/integer_motion_v2_metal.c` — registers
  `vmaf_fex_integer_motion_v2_metal` (extractor name
  `motion_v2_metal`, `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag set).
  `init()` returns `-ENOSYS` from the kernel-template helpers.
- CI lane `Build — macOS Metal (T8-1 scaffold)` in
  `.github/workflows/libvmaf-build-matrix.yml` compiles with
  `-Denable_metal=enabled` on `macos-latest` and runs the smoke
  test.
- Smoke-only `enable_metal` build — every public C-API entry point
  returns `-ENOSYS`, every kernel-template helper returns `-ENOSYS`,
  the first-consumer extractor `init()` returns `-ENOSYS`.

**Pending** — T8-1b (runtime PR) replaces the `kernel_template.c`
bodies and the `common.c` device-init path with real Metal calls
(MetalCpp wrapper, `MTLCreateSystemDefaultDevice`,
`[id<MTLDevice> newCommandQueue]`, `[id<MTLDevice>
newBufferWithLength:options:]`, ...). T8-1c lands the first real
kernel (`motion_v2_metal.metal` shader + metallib loader) and the
runtime PR's submit/collect chain. Remaining kernel ports (VIF, ADM,
SSIM, ...) follow as their own PRs gated by the `places=4`
cross-backend-diff lane (per
[ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md)).

## Ground rules

- **Parent rules** apply in full (see [../../AGENTS.md](../../AGENTS.md)).
- **Apple Silicon only** — the runtime PR (T8-1b) gates device
  selection on `MTLGPUFamily.Apple7` (M1 and later) via
  `-[id<MTLDevice> supportsFamily:]`. Intel Macs surface as
  `-ENODEV`. See ADR-0361 §"Apple Silicon-only" for the rationale
  (Apple's discontinuation of Intel-Mac GPU parity, plus
  unified-memory zero-copy is the load-bearing perf story).
- **No Metal SDK is currently linked**. The `meson.build` includes
  optional `dependency('Metal', required: false)` /
  `dependency('MetalKit', required: false)` probes; the scaffold
  compiles cleanly on non-macOS hosts (Linux, Windows) without any
  Apple frameworks installed because the auto-probe resolves to
  disabled there. The runtime PR (T8-1b) flips both probes to
  required when the option is `enabled`.
- **Metal runtime types cross headers as `uintptr_t`**. The public
  header `libvmaf_metal.h` and the kernel template's
  `VmafMetalKernelLifecycle` / `VmafMetalKernelBuffer` carry
  `uintptr_t` for `id<MTLCommandQueue>` / `id<MTLBuffer>` /
  `id<MTLEvent>` so consumer TUs stay free of `<Metal/Metal.h>` /
  `<Metal/Metal.hpp>`. Cast on the implementation side only.
  Mirrors the HIP ADR-0212 and Vulkan ADR-0184 patterns.
- **Don't drift the kernel-template field shape from HIP / CUDA
  beyond the documented unified-memory simplification**. The Metal
  template collapses the (device, pinned-host) pair into a single
  `MTLBuffer` with `MTLResourceStorageModeShared` because Apple
  Silicon has no separate device memory pool. The lifecycle struct
  (`cmd_queue` + `submit` + `finished` events) mirrors the HIP /
  CUDA twins field-for-field. See "Rebase-sensitive invariants"
  below.

## Rebase-sensitive invariants

- **`kernel_template.h` mirrors `hip/kernel_template.h` modulo the
  unified-memory buffer collapse** (fork-local, ADR-0361). The
  `VmafMetalKernelLifecycle` struct mirrors `VmafHipKernelLifecycle`
  field-for-field (one command-queue/stream slot + two event slots).
  The buffer struct `VmafMetalKernelBuffer` deliberately has one
  fewer slot than `VmafHipKernelReadback`: there is no
  `host_pinned` because Apple Silicon's unified memory makes
  `[buffer contents]` host-visible directly. Helper signatures
  (`vmaf_metal_kernel_lifecycle_init/_close`,
  `vmaf_metal_kernel_buffer_alloc/_free`,
  `vmaf_metal_kernel_submit_pre_launch`,
  `vmaf_metal_kernel_collect_wait`) parallel the HIP names with
  `_buffer_` substituted for `_readback_` to reflect the
  zero-copy posture. **On rebase / refactor**: if a fork PR touches
  `hip/kernel_template.h`'s lifecycle (e.g. adds a third event),
  walk the diff onto `metal/kernel_template.h` +
  `kernel_template.c` before merging. The buffer-vs-readback name
  asymmetry is intentional and stays.

- **`integer_motion_v2_metal.c` mirrors `integer_motion_v2_hip.c`
  call-graph-for-call-graph** (fork-local, ADR-0361). Same
  `MotionV2StateMetal` / `MotionV2StateHip` field shape modulo the
  ping-pong buffer slots being one (single `MTLBuffer` with
  `MTLResourceStorageModeShared`) rather than two, since unified
  memory means the previous-frame ref Y plane lives in the same
  address space the kernel reads. The `VMAF_FEATURE_EXTRACTOR_TEMPORAL`
  flag and the `flush()` callback contract stay aligned with the
  HIP / CUDA twins. **On rebase**: if a future PR drifts the HIP
  twin's lifecycle (e.g. adds a third buffer), update the Metal
  twin in the same PR.

- **`vmaf_fex_integer_motion_v2_metal` is registered without the
  `VMAF_FEATURE_EXTRACTOR_METAL` flag bit set** (fork-local,
  ADR-0361). The flag bit is reserved in the enum (the runtime PR
  T8-1b will define it) but the consumer does not set it because
  the picture buffer-type check in
  `vmaf_feature_extractor_context_extract` would route a
  Metal-flagged extractor through a (not-yet-existing) Metal
  buffer-type branch. The runtime PR (T8-1b) adds the
  `VMAF_PICTURE_BUFFER_TYPE_METAL_DEVICE` tag and *then* sets the
  flag on the extractor. **On rebase**: if a refactor touches the
  picture buffer-type dispatch, leave the Metal extractor's flags
  at `VMAF_FEATURE_EXTRACTOR_TEMPORAL` only until T8-1b. Same
  rationale as the HIP twin's `VMAF_FEATURE_EXTRACTOR_HIP`-deferral
  posture.

## Governing ADRs

- [ADR-0361](../../../docs/adr/0361-metal-compute-backend.md) —
  Metal scaffold-only audit-first PR (T8-1, this PR).
- [ADR-0212](../../../docs/adr/0212-hip-backend-scaffold.md) — HIP
  scaffold-only audit-first PR (T7-10); the precedent this PR
  mirrors.
- [ADR-0175](../../../docs/adr/0175-vulkan-backend-scaffold.md) —
  Vulkan scaffold (T5-1); the original audit-first GPU-backend
  precedent.
- [ADR-0246](../../../docs/adr/0246-gpu-kernel-template.md) — GPU
  kernel-template decision; the source the Metal mirror tracks
  (via the HIP twin that mirrors the CUDA twin).
- [ADR-0214](../../../docs/adr/0214-gpu-parity-ci-gate.md) —
  `places=4` cross-backend gate; the runtime PR's incoming
  numerics gate.
- [ADR-0145](../../../docs/adr/0145-motion-v2-neon-bitexact.md) —
  motion_v2 NEON twin on Apple Silicon CPU. Coordinates with this
  ADR: NEON stays the CPU-side path; Metal is the GPU-side path.

## Build

```bash
# On macOS:
meson setup build -Denable_metal=enabled
ninja -C build
meson test -C build test_metal_smoke

# On Linux / Windows: -Denable_metal=auto resolves to disabled
# (no Metal frameworks); -Denable_metal=enabled fails the meson
# setup with a clear missing-framework error from the dependency()
# probe.
```

The scaffold has zero hard runtime dependencies on non-macOS hosts.
On macOS the meson `dependency('Metal') / dependency('MetalKit')`
probes resolve to the system frameworks; `Build — macOS Metal
(T8-1 scaffold)` in `.github/workflows/libvmaf-build-matrix.yml`
runs this exact configuration on every PR.
