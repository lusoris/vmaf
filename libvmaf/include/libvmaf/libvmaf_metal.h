/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Licensed under the BSD+Patent License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      https://opensource.org/licenses/BSDplusPatent
 */

/**
 * @file libvmaf_metal.h
 * @brief Metal (Apple Silicon) backend public API — ADR-0361 / T8-1 through T8-1d.
 *
 * **Status: live.** The runtime (T8-1b, ADR-0420) and the first kernel set
 * (T8-1c/d, ADR-0421 — `integer_motion_v2.metal` + 7 additional
 * feature-extractor MSL shaders) are fully shipped. All 8 `.mm` dispatch
 * translation units and 8 `.metal` shaders are compiled, linked, and
 * registered. Entry points return 0 on Apple-Family-7+ (M1 and later) or
 * -ENODEV on Intel Macs and non-Apple hosts.
 *
 * Mirrors the HIP scaffold (ADR-0212) and the Vulkan scaffold (ADR-0175) —
 * see ADR-0361 for the audit-first decision and rollout sequence.
 *
 * When libvmaf was built without `-Denable_metal=enabled` (or built on
 * a non-macOS host where the Metal framework auto-probe failed), every
 * entry point returns -ENOSYS unconditionally and the runtime treats
 * Metal as disabled.
 *
 * Header purity: the Metal runtime types (`id<MTLDevice>`,
 * `id<MTLCommandQueue>`, `id<MTLBuffer>`) cross the ABI as
 * `uintptr_t` to keep this header free of `<Metal/Metal.h>` /
 * `<Metal/Metal.hpp>`. Cast on the caller side. Same convention the
 * HIP backend uses for `hipStream_t` / `hipEvent_t` (per ADR-0212)
 * and the Vulkan backend uses for `VkDevice` / `VkQueue` (per
 * ADR-0184).
 *
 * Apple-platform-only: device selection is gated on `MTLGPUFamily.Apple7`
 * (M1 and later). Intel Macs and non-Apple hosts surface as -ENODEV from
 * `vmaf_metal_state_init`. See ADR-0361 §"Apple Silicon-only" for reasoning.
 * CLI exposure: `--metal_device <N>` / `--no_metal` / `--backend metal`
 * (ADR-0422).
 */

#ifndef LIBVMAF_METAL_H_
#define LIBVMAF_METAL_H_

#include <stddef.h>
#include <stdint.h>

#include "libvmaf.h"
#include "picture.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns 1 if libvmaf was built with Metal support
 * (-Denable_metal=enabled or auto-probe succeeded on macOS), 0
 * otherwise. Cheap to call; no Metal runtime is touched until @ref
 * vmaf_metal_state_init().
 */
VMAF_EXPORT int vmaf_metal_available(void);

/**
 * Opaque handle to a Metal-backed scoring state. One state pins one
 * Metal device + command queue; callers that want multi-GPU fan-out
 * create one state per device. Same lifetime model as
 * `VmafCudaState` / `VmafVulkanState` / `VmafHipState`.
 */
typedef struct VmafMetalState VmafMetalState;

typedef struct VmafMetalConfiguration {
    int device_index; /**< -1 = system default Metal device (typical Apple Silicon path) */
    int flags;        /**< reserved for future use; pass 0 */
} VmafMetalConfiguration;

/**
 * Allocate a VmafMetalState. Picks the device by index; -1 selects the
 * system default Metal device (`MTLCreateSystemDefaultDevice` on
 * Apple Silicon).
 *
 * @return 0 on success, -ENOSYS when built without Metal, -ENODEV when
 *         no Apple-Family-7+ device is available (Intel Mac, non-macOS
 *         host, or M-series device unavailable), -EINVAL on bad
 *         arguments.
 */
VMAF_EXPORT int vmaf_metal_state_init(VmafMetalState **out, VmafMetalConfiguration cfg);

/**
 * Hand the Metal state to a VmafContext. After import, the context
 * borrows the state pointer for the duration of its lifetime; the
 * caller still owns the state and must free it with
 * @ref vmaf_metal_state_free after vmaf_close(). Same lifetime model as
 * the SYCL + Vulkan + HIP backends.
 */
VMAF_EXPORT int vmaf_metal_import_state(VmafContext *ctx, VmafMetalState *state);

/**
 * Release a state previously allocated via @ref vmaf_metal_state_init.
 * Safe to pass `NULL` or a state that was never imported. After import
 * the caller is still responsible for freeing — call this after
 * vmaf_close() to avoid using a state the context still references.
 */
VMAF_EXPORT void vmaf_metal_state_free(VmafMetalState **state);

/**
 * Enumerate Apple-Family-7+ Metal devices visible to the runtime.
 * Prints one line per device with its ordinal, name, and GPU family.
 * Returns the device count or -ENOSYS when built without Metal.
 */
VMAF_EXPORT int vmaf_metal_list_devices(void);

/* -----------------------------------------------------------------
 * IOSurface zero-copy import — ADR-0423 scaffold (T8-IOS).
 *
 * Mirrors the Vulkan import surface (ADR-0184 / ADR-0186): caller
 * holds an external GPU-resident frame, hands its opaque handle to
 * libvmaf, and the Metal feature kernels read it without a host
 * round-trip. The Metal flavour consumes `IOSurfaceRef` because
 * Apple's VideoToolbox hwdec delivers frames as `CVPixelBufferRef`
 * whose backing store is always an `IOSurface` — the canonical
 * shared-GPU-memory primitive on macOS / iOS. FFmpeg surfaces it
 * as `AVFrame->data[3] -> CVPixelBufferRef` from
 * `AV_HWDEVICE_TYPE_VIDEOTOOLBOX`; the caller pulls the IOSurface
 * with `CVPixelBufferGetIOSurface` before handing it here.
 *
 * Same-device contract: source IOSurfaces are bound to whichever
 * MTLDevice rendered them. libvmaf compute must run on the same
 * device, hence @ref vmaf_metal_state_init_external — symmetric to
 * @ref vmaf_vulkan_state_init_external. On a single-GPU Apple
 * Silicon Mac (the common case) there is only one Apple-Family-7+
 * device and the constraint is trivially satisfied; the external-
 * init entry point still exists so multi-GPU Mac Pro hosts get a
 * deterministic device match.
 *
 * Status: T8-IOS scaffold landed under ADR-0423 — every entry
 * point in this block returns -ENOSYS until the implementation
 * PR (T8-IOS-b) replaces the stubs with
 * `[MTLDevice newTextureWithDescriptor:iosurface:plane:]` /
 * `CVMetalTextureCacheCreateTextureFromImage` wiring.
 * ----------------------------------------------------------------- */

/**
 * Pre-existing Metal handles supplied by the caller. Used by
 * @ref vmaf_metal_state_init_external so libvmaf compute runs on
 * the same MTLDevice as the source IOSurfaces (same constraint
 * the Vulkan import path enforces — see ADR-0184). Handles cross
 * the ABI as `uintptr_t` to keep this header free of
 * `<Metal/Metal.h>`; cast on the caller side.
 *
 * Lifetime: libvmaf does NOT take ownership. The caller (typically
 * FFmpeg's `AVHWDeviceContext` / `AVMetalDeviceContext` when the
 * MoltenVK or VideoToolbox bridge lands) keeps them alive at least
 * until @ref vmaf_metal_state_free returns.
 */
typedef struct VmafMetalExternalHandles {
    uintptr_t device;        /**< id<MTLDevice> */
    uintptr_t command_queue; /**< id<MTLCommandQueue> (optional; 0 = create internally) */
} VmafMetalExternalHandles;

/**
 * Allocate a VmafMetalState that adopts caller-supplied Metal
 * handles instead of creating its own MTLDevice / MTLCommandQueue.
 * Required when the caller will pass external IOSurface handles
 * via @ref vmaf_metal_picture_import — the IOSurface's backing
 * MTLTexture is only addressable on the device that mapped it.
 *
 * Mutually exclusive with @ref vmaf_metal_state_init in a single
 * process context: pick one. The Apple-Family-7+ gate still
 * applies — passing an Intel-Mac MTLDevice returns -ENODEV.
 *
 * @return 0 on success, -ENOSYS when built without Metal (or in
 *         the T8-IOS scaffold contract), -EINVAL on bad arguments,
 *         -ENODEV on a non-Apple-Family-7 device, -ENOMEM on
 *         allocation failure.
 */
VMAF_EXPORT int vmaf_metal_state_init_external(VmafMetalState **out,
                                               VmafMetalExternalHandles handles);

/**
 * Import an external IOSurface (typically pulled from a
 * `CVPixelBufferRef` via `CVPixelBufferGetIOSurface`) into the
 * libvmaf Metal compute pipeline. Caller retains ownership of the
 * underlying IOSurface; libvmaf reads it via a temporary
 * `id<MTLTexture>` materialised through
 * `[MTLDevice newTextureWithDescriptor:iosurface:plane:]`.
 *
 * @param state    Metal state handle.
 * @param iosurface IOSurfaceRef (cast to uintptr_t).
 * @param plane    Plane index (0 = luma; chroma planes via the
 *                 standard biplanar / triplanar layout).
 * @param w        Frame width.
 * @param h        Frame height.
 * @param bpc      Bits per component (8 / 10 / 12 / 16).
 * @param is_ref   1 = reference frame, 0 = distorted.
 * @param index    Frame index (matches the index passed to
 *                 @ref vmaf_metal_read_imported_pictures).
 *
 * @return 0 on success, -ENOSYS until T8-IOS-b lands, -EINVAL on
 *         bad arguments.
 */
VMAF_EXPORT int vmaf_metal_picture_import(VmafMetalState *state, uintptr_t iosurface,
                                          unsigned plane, unsigned w, unsigned h,
                                          unsigned bpc, int is_ref, unsigned index);

/**
 * Block until all previously-submitted Metal compute work on
 * `state` has finished. Mirrors @ref vmaf_vulkan_wait_compute.
 * Used by FFmpeg-side filters before reusing imported IOSurfaces
 * in the next frame.
 *
 * @return 0 on success, -ENOSYS until T8-IOS-b lands.
 */
VMAF_EXPORT int vmaf_metal_wait_compute(VmafMetalState *state);

/**
 * Trigger a libvmaf score read for the imported reference +
 * distorted IOSurfaces at `index`. Mirrors
 * @ref vmaf_vulkan_read_imported_pictures.
 *
 * @return 0 on success, -ENOSYS until T8-IOS-b lands.
 */
VMAF_EXPORT int vmaf_metal_read_imported_pictures(VmafContext *ctx, unsigned index);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_METAL_H_ */
