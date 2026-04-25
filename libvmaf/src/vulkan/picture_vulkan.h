/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_VULKAN_PICTURE_VULKAN_H_
#define LIBVMAF_VULKAN_PICTURE_VULKAN_H_

#include <stddef.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

int vmaf_vulkan_picture_alloc(VmafVulkanContext *ctx, void **out, size_t size);
void vmaf_vulkan_picture_free(VmafVulkanContext *ctx, void *buf);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_VULKAN_PICTURE_VULKAN_H_ */
