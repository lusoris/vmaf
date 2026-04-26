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

#ifdef __cplusplus
extern "C" {
#endif

struct VmafVulkanContext {
    /* Set to true after `volkInitialize()` succeeds the first time
     * any context is created. Subsequent contexts skip the global
     * init. */
    int volk_loaded;

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
