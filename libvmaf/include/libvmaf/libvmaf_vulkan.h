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
 * @file libvmaf_vulkan.h
 * @brief Vulkan backend public API — scaffolded by ADR-0175 / T5-1.
 *
 * **Status: scaffold only.** Every entry point currently returns -ENOSYS
 * pending a real implementation. The header lands so downstream consumers
 * can compile against the API surface; the kernels (ADM, VIF, motion)
 * arrive in follow-up PRs per ADR-0127's "VIF as pathfinder" sequence.
 *
 * When libvmaf was built without `-Denable_vulkan=enabled`, every entry
 * point returns -ENOSYS unconditionally and the runtime treats vulkan as
 * disabled.
 */

#ifndef LIBVMAF_VULKAN_H_
#define LIBVMAF_VULKAN_H_

#include <stddef.h>
#include <stdint.h>

#include "libvmaf.h"
#include "picture.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns 1 if libvmaf was built with Vulkan support
 * (-Denable_vulkan=enabled), 0 otherwise. Cheap to call; no Vulkan
 * runtime is touched until @ref vmaf_vulkan_state_init().
 */
int vmaf_vulkan_available(void);

/**
 * Opaque handle to a Vulkan-backed scoring state. One state pins one
 * VkInstance + VkDevice + compute queue; callers that want multi-GPU
 * fan-out create one state per device.
 */
typedef struct VmafVulkanState VmafVulkanState;

typedef struct VmafVulkanConfiguration {
    int device_index;      /**< -1 = first device with compute queue */
    int enable_validation; /**< non-zero: load VK_LAYER_KHRONOS_validation */
} VmafVulkanConfiguration;

/**
 * Allocate a VmafVulkanState. Picks the device by index; -1 selects the
 * first device that exposes a compute queue family.
 *
 * @return 0 on success, -ENOSYS when built without Vulkan, -ENODEV when
 *         no compatible device is found, -EINVAL on bad arguments.
 */
int vmaf_vulkan_state_init(VmafVulkanState **out, VmafVulkanConfiguration cfg);

/**
 * Hand the Vulkan state to a VmafContext. The context borrows the
 * state pointer for the duration of its lifetime; the caller still
 * owns the state and must free it with @ref vmaf_vulkan_state_free
 * after vmaf_close(). Same lifetime model as the SYCL backend.
 */
int vmaf_vulkan_import_state(VmafContext *ctx, VmafVulkanState *state);

/**
 * Release a state previously allocated via @ref vmaf_vulkan_state_init.
 * Safe to pass `NULL` or a state that was never imported. After import
 * the caller is still responsible for freeing — call this after
 * vmaf_close() to avoid using a state the context still references.
 */
void vmaf_vulkan_state_free(VmafVulkanState **state);

/**
 * Enumerate compute-capable Vulkan devices visible to the runtime.
 * Prints one line per device with its ordinal, name, and API version.
 * Returns the device count or -ENOSYS when built without Vulkan.
 */
int vmaf_vulkan_list_devices(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_VULKAN_H_ */
