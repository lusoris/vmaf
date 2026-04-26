/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Internal-only Vulkan context layout. Kernel TUs in
 *  libvmaf/src/feature/vulkan/ include this header (NOT
 *  libvmaf/src/vulkan/vulkan_common.h alone) so they can read the
 *  device / queue / allocator handles. The public surface
 *  (vulkan_common.h) stays opaque so callers can't accidentally bind to
 *  Vulkan-specific types.
 */

#ifndef LIBVMAF_VULKAN_INTERNAL_H_
#define LIBVMAF_VULKAN_INTERNAL_H_

/* volk loads every Vulkan entry point at runtime; defining
 * VK_NO_PROTOTYPES is required so the system Vulkan headers do not
 * declare the static-link prototypes that would conflict with volk's
 * function pointers. The wrap's compile_args already injects
 * `-DVK_NO_PROTOTYPES` for users of `volk_dep`, so this `#define` is
 * defensive in case someone forgets. */
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vk_mem_alloc.h>

/* picture_vulkan.h declares the opaque VmafVulkanBuffer used by the
 * import slots below. Pulled in from the directory's local includes via
 * the meson include_directories list. */
#include "picture_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-state import slot for the VkImage zero-copy path (T7-29 part 2,
 * ADR-0186). One ref + one dis staging buffer per state, lazily
 * allocated on first import_image call and reused across frames. */
struct VmafVulkanImportSlots {
    /* Frame geometry pinned by the first import_image call. Subsequent
     * calls must match (or return -EINVAL) — same contract as the
     * SYCL `init_frame_buffers` model. Zero == not yet allocated. */
    unsigned w;
    unsigned h;
    unsigned bpc;
    /* Aligned row stride (bytes) of the staging buffers, matching the
     * DATA_ALIGN-rounded stride that vmaf_picture_alloc would produce.
     * Stored so vmaf_vulkan_read_imported_pictures can hand the buffers
     * to vmaf_read_pictures with correct geometry. */
    size_t stride_bytes;

    VmafVulkanBuffer *ref_buf;
    VmafVulkanBuffer *dis_buf;

    /* Per-buffer "data has been staged for `index`" flags. read_imported_pictures
     * verifies both are set for the requested index before triggering scoring. */
    int ref_pending;
    int dis_pending;
    unsigned ref_index;
    unsigned dis_index;

    /* Reusable transfer command buffer + completion fence for the
     * synchronous copy. Lazily created alongside the buffers. */
    VkCommandBuffer cmd;
    VkFence fence;
};

struct VmafVulkanState {
    VmafVulkanContext *ctx;
    struct VmafVulkanImportSlots import;
};

/* import.c — release any lazily-allocated import slot resources.
 * Safe to call on a state that never imported anything; the slot
 * fields are zero-initialised by calloc in state_init. */
void vmaf_vulkan_import_slots_free(struct VmafVulkanState *state);

struct VmafVulkanContext {
    /* Set to true after `volkInitialize()` succeeds the first time
     * any context is created. Subsequent contexts skip the global
     * init. */
    int volk_loaded;
    /* Non-zero when libvmaf created the VkInstance + VkDevice
     * itself (vmaf_vulkan_context_new). Zero when the caller
     * supplied them via vmaf_vulkan_state_init_external — in that
     * case context_destroy must NOT call vkDestroyDevice /
     * vkDestroyInstance. The VMA allocator and command pool are
     * always libvmaf-owned regardless. ADR-0186. */
    int owns_handles;

    int device_index; /* Resolved >=0 device ordinal (post auto-pick). */
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family_index;
    VkQueue queue;

    VmaAllocator allocator;
    VkCommandPool command_pool;

    /* Properties of the selected physical device — feature kernels
     * read these to pick group size, sub-group ops, etc. */
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
};

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_VULKAN_INTERNAL_H_ */
