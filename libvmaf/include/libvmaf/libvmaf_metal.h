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

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_METAL_H_ */
