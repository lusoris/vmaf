# Metal (Apple Silicon) compute backend (scaffold)

> **Status: scaffold only.** Every entry point in
> [`libvmaf_metal.h`](../../../libvmaf/include/libvmaf/libvmaf_metal.h)
> currently returns `-ENOSYS` pending the runtime PR (T8-1b). Four
> kernel-template consumers are registered:
> `motion_v2_metal`, `psnr_metal`, `float_ssim_metal`, and `motion_metal`.
> Each resolves to a clean "found, runtime not ready" surface so callers
> asking by name get a deterministic `-ENOSYS` rather than "no such
> extractor". The runtime PR (T8-1b) flips all kernel-template helper
> bodies from `-ENOSYS` to real Metal calls at once.
> Rollout cadence mirrors the HIP T7-10 → T7-10b split that landed
> this approach last (see [ADR-0212](../../adr/0212-hip-backend-scaffold.md))
> and the original Vulkan T5-1 → T5-1b split that established it (see
> [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md)).
>
> Governing ADR: [ADR-0361](../../adr/0361-metal-compute-backend.md).
> **4 of 17 planned extractors registered (batch-1).**

## Why Metal

Apple Silicon (M1+) is the perf story for Apple-platform users. The
fork's existing Apple-Silicon coverage is the NEON SIMD CPU path
(per [ADR-0145](../../adr/0145-motion-v2-neon-bitexact.md) and the
wider NEON twin matrix); this backend adds the GPU compute path that
NEON cannot reach.

Three properties make a native Metal backend worth shipping:

1. **Unified memory.** `MTLBuffer` allocations created with
   `MTLResourceStorageModeShared` are zero-copy across CPU↔GPU; the
   submit-side H2D / D2H staging the CUDA / HIP / Vulkan backends
   spend the bulk of their complexity on collapses to host stores
   and direct `[buffer contents]` reads.
2. **First-party Apple compute API.** OpenCL is deprecated since
   macOS 10.14 (2018) and receives no driver updates; Vulkan reaches
   the GPU only through MoltenVK's translation layer (Vulkan command
   buffer → Metal command buffer rewrite) which adds per-dispatch
   overhead. Metal is the supported user-space surface.
3. **No PCIe boundary.** GPU and CPU share the same DRAM with cache
   coherence; the runtime PR can keep the previous-frame ref Y
   plane in one shared buffer rather than ping-ponging two device
   allocations the way the HIP twin does.

See [ADR-0361 §Context](../../adr/0361-metal-compute-backend.md#context)
for the full reasoning and rejected alternatives (MoltenVK, oneAPI,
OpenCL, Swift-based runtime).

## Apple Silicon only

The runtime PR (T8-1b) gates device selection on
`MTLGPUFamily.Apple7` (M1 and later) via
`-[id<MTLDevice> supportsFamily:]`. Intel Macs and non-macOS hosts
surface as `-ENODEV` from `vmaf_metal_state_init`. Reasoning: Apple
discontinued Intel-Mac GPU parity, and the unified-memory zero-copy
story does not apply on Intel-Mac discrete GPUs (Radeon Pro / Vega)
which sit behind PCIe. See
[ADR-0361 §Apple Silicon-only](../../adr/0361-metal-compute-backend.md#apple-silicon-onlyapple-gpu-family-7-reject-intel-mac).

## Build

On macOS:

```bash
meson setup build -Denable_metal=enabled
ninja -C build
meson test -C build test_metal_smoke
```

`-Denable_metal=auto` (the default) auto-resolves to enabled on
`host_machine.system() == 'darwin'` and disabled elsewhere.
`-Denable_metal=disabled` suppresses the auto-probe even on macOS.
`-Denable_metal=enabled` forces the Metal frameworks to be linked;
on non-macOS hosts the meson `dependency('Metal')` probe fails the
setup step with a clear missing-framework error.

The scaffold has zero hard runtime dependencies on non-macOS hosts
(the TUs are pure C99 + errno.h). On macOS the `dependency('Metal')`
/ `dependency('MetalKit')` probes resolve to the system frameworks
at `/System/Library/Frameworks/Metal.framework` and
`/System/Library/Frameworks/MetalKit.framework`, both guaranteed
present on macOS 11+.

## Runtime layer (planned for T8-1b)

The runtime PR (T8-1b) will use Apple's official **MetalCpp** C++
wrapper headers (`<Metal/Metal.hpp>`, `<MetalKit/MetalKit.hpp>`) for
the runtime layer rather than Objective-C `<Metal/Metal.h>` or Swift.
MetalCpp is a single-header, header-only C++ binding that exposes
the Metal API as `NS::*` / `MTL::*` C++ classes with reference-
counted `NS::Object` lifetimes; Apple ships and supports it as the
recommended C++ binding.

Reference: <https://developer.apple.com/metal/cpp/> (accessed
2026-05-09).

This keeps the libvmaf runtime tree in C++ throughout (matches CUDA
`.cu` / SYCL `.cpp` / Vulkan `.cpp` precedent) and avoids dragging
Objective-C runtime dependencies into libvmaf TUs.

The kernel sources themselves will be written in Metal Shading
Language (`.metal`) and compiled to `.air` / `.metallib` archives
via `xcrun metal` at build time.

## Rollout sequence

1. **T8-1 (scaffold PR + batch-1)** — scaffold only. Public header,
   src/metal tree (common, picture_metal, dispatch_strategy,
   kernel_template), four consumer scaffolds (`motion_v2_metal`,
   `psnr_metal`, `float_ssim_metal`, `motion_metal`), `enable_metal`
   meson option, smoke test, CI lane (`Build — macOS Metal (T8-1
   scaffold)`), this doc page. Every entry point returns `-ENOSYS`.
2. **T8-1b (runtime PR)** — `MTLCreateSystemDefaultDevice` /
   `id<MTLCommandQueue>` / `id<MTLBuffer>` lifecycle; MetalCpp
   wrapper introduced. `vmaf_metal_state_init` returns `0` on a
   real Apple Silicon device, `-ENODEV` on Intel Mac or non-Apple-
   Family-7 GPU. The smoke contract flips from "`-ENOSYS`
   everywhere" to "device_count >= 0, state_init succeeds when
   devices >= 1, skip when none".
3. **T8-1c (motion_v2 kernel PR)** — first feature on the Metal
   compute path. The `motion_v2_metal.metal` shader source +
   metallib loader land; the consumer's submit/collect/flush chain
   is wired to real `MTLCommandBuffer` dispatches. Bit-exact-vs-CPU
   validation via `/cross-backend-diff`.
4. **T8-1d…** — remaining kernels (VIF, ADM, SSIM, ...) follow as
   their own PRs gated by the `places=4` cross-backend-diff lane
   (per [ADR-0214](../../adr/0214-gpu-parity-ci-gate.md)).
5. **`enable_metal` default flip** from `auto` to `enabled`: only
   after the kernel matrix proves bit-exactness via the `places=4`
   cross-backend gate (mirrors the `enable_vulkan` and `enable_hip`
   roadmaps).

## Coordination with NEON

The Metal backend targets the GPU on Apple Silicon. The NEON SIMD
twin matrix (per [ADR-0145](../../adr/0145-motion-v2-neon-bitexact.md))
stays the CPU-side path on the same hardware. The two are
complementary:

- Small / latency-sensitive runs land on NEON via the existing CPU
  dispatch (no GPU command-buffer setup overhead).
- Large / throughput-bound runs land on Metal once the runtime PR
  ships; the GPU's parallelism + unified memory eliminate both the
  CPU-bound bottleneck and the H2D / D2H staging cost.

Backend selection follows the standard libvmaf precedence (see
[../index.md](../index.md) §Runtime selection): GPU paths win when
available, CPU SIMD wins otherwise.

## Verification

This scaffold ships compile-only plumbing. The macOS CI lane
`Build — macOS Metal (T8-1 scaffold)` is the ground-truth gate; it
runs on every PR with `-Denable_metal=enabled` and exercises the
smoke test against the `-ENOSYS` contract path. Linux-host dev
sessions (where the bulk of fork development happens) cannot
reproduce the lane locally — `Metal.framework` only exists on macOS
hosts.

Reviewers verifying locally on a Mac:

```bash
meson setup build -Denable_metal=enabled
ninja -C build
meson test -C build test_metal_smoke
```

## References

- [ADR-0361](../../adr/0361-metal-compute-backend.md) — governing
  ADR for this scaffold.
- [ADR-0212](../../adr/0212-hip-backend-scaffold.md) — HIP scaffold
  precedent (T7-10).
- [ADR-0175](../../adr/0175-vulkan-backend-scaffold.md) — Vulkan
  scaffold precedent (T5-1) — the original audit-first GPU-backend
  pattern.
- [ADR-0145](../../adr/0145-motion-v2-neon-bitexact.md) — motion_v2
  NEON twin on Apple Silicon CPU.
- [ADR-0214](../../adr/0214-gpu-parity-ci-gate.md) — `places=4`
  cross-backend gate; the runtime PR's incoming numerics gate.
- Apple Developer documentation — Metal-cpp,
  <https://developer.apple.com/metal/cpp/> (accessed 2026-05-09).
