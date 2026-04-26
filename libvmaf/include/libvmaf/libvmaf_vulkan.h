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

/**
 * Zero-copy frame import — T7-29 (ADR-0184).
 *
 * The next three entry points mirror the SYCL backend's
 * VAAPI/dmabuf import surface (see libvmaf_sycl.h). They let
 * an FFmpeg-side filter — or any direct C-API caller — hand
 * an externally-decoded VkImage straight to the libvmaf
 * Vulkan compute queue without a CPU readback round-trip.
 *
 * **Status: scaffold only (T7-29 part 1).** Every function
 * returns -ENOSYS pending the real implementation
 * (vkCmdCopyImageToBuffer + timeline-semaphore wait). The
 * declarations land so consumers can compile against the
 * stable contract.
 *
 * Header purity: Vulkan handles cross the ABI as `uintptr_t`
 * to keep this header usable from translation units that
 * don't have <vulkan/vulkan.h> in scope. Cast from VkImage /
 * VkSemaphore on the caller side.
 */

/**
 * Import an external VkImage into the libvmaf Vulkan compute
 * pipeline. Caller retains ownership of the underlying VkImage
 * and VkSemaphore.
 *
 * @param state              Vulkan state handle.
 * @param vk_image           VkImage handle (cast to uintptr_t).
 * @param vk_format          VkFormat enum value.
 * @param vk_layout          Current VkImageLayout enum value.
 * @param vk_semaphore       VkSemaphore handle (cast to uintptr_t).
 * @param vk_semaphore_value Wait value — libvmaf will wait until
 *                           the semaphore reaches this value
 *                           before reading the image.
 * @param w                  Frame width.
 * @param h                  Frame height.
 * @param bpc                Bits per component (8 / 10 / 12 / 16).
 * @param is_ref             1 = reference frame, 0 = distorted.
 * @param index              Frame index (matches the index
 *                           passed to vmaf_vulkan_read_imported_pictures).
 *
 * @return 0 on success, -ENOSYS until T7-29 part 2 lands,
 *         -EINVAL on bad args.
 */
int vmaf_vulkan_import_image(VmafVulkanState *state, uintptr_t vk_image, uint32_t vk_format,
                             uint32_t vk_layout, uintptr_t vk_semaphore,
                             uint64_t vk_semaphore_value, unsigned w, unsigned h, unsigned bpc,
                             int is_ref, unsigned index);

/**
 * Block until all previously-submitted compute work on `state`
 * has finished. Used by FFmpeg-side filters before reusing
 * imported images in the next frame.
 *
 * @return 0 on success, -ENOSYS until T7-29 part 2 lands.
 */
int vmaf_vulkan_wait_compute(VmafVulkanState *state);

/**
 * Trigger a libvmaf score read for the imported reference +
 * distorted images at `index`. Mirrors vmaf_read_pictures_sycl
 * but for Vulkan-imported frames.
 *
 * @return 0 on success, -ENOSYS until T7-29 part 2 lands.
 */
int vmaf_vulkan_read_imported_pictures(VmafContext *ctx, unsigned index);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_VULKAN_H_ */
