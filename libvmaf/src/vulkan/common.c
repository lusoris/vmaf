/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <errno.h>
#include <stdlib.h>

#include "common.h"

struct VmafVulkanContext {
    int device_index;
    /* TODO: add queue / stream handle, allocator, etc. */
};

int vmaf_vulkan_context_new(VmafVulkanContext **out, int device_index)
{
    if (!out)
        return -EINVAL;
    VmafVulkanContext *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return -ENOMEM;
    ctx->device_index = device_index;
    /* TODO: initialize device/queue */
    *out = ctx;
    return 0;
}

void vmaf_vulkan_context_destroy(VmafVulkanContext *ctx)
{
    if (!ctx)
        return;
    /* TODO: release device/queue */
    free(ctx);
}

int vmaf_vulkan_device_count(void)
{
    /* TODO: probe vulkan runtime */
    return 0;
}
