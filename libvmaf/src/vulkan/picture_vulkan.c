/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Picture allocation / lifecycle for the Vulkan backend (T5-1b).
 *
 *  VMA-backed VkBuffer pairs sized for a single frame plane. The
 *  buffer is allocated `HOST_VISIBLE | HOST_COHERENT` where
 *  available so the host can `memcpy` into the persistent mapping
 *  without explicit `vkMapMemory` round-trips per frame.
 */

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "picture_vulkan.h"
#include "vulkan_internal.h"

struct VmafVulkanBuffer {
    VkBuffer vk_buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
    size_t size;
};

/* Bookkeeping for the legacy `void*` picture-alloc API used by the
 * T5-1 smoke test. We allocate at most a handful of pictures across
 * a session, so a flat append-only array with linear scan is fine.
 * If the count ever grows beyond ~32 entries we should swap this for
 * a hash table; for now keep it simple. */
typedef struct {
    void *host_ptr;
    VmafVulkanBuffer *buf;
} ShimEntry;

#define VMAF_VK_SHIM_MAX 32
static ShimEntry g_shim_entries[VMAF_VK_SHIM_MAX];
static size_t g_shim_count = 0;

int vmaf_vulkan_buffer_alloc(VmafVulkanContext *ctx, VmafVulkanBuffer **out_buf, size_t size)
{
    if (!ctx || !out_buf || size == 0)
        return -EINVAL;
    assert(ctx->allocator != VK_NULL_HANDLE);
    assert(ctx->device != VK_NULL_HANDLE);

    VmafVulkanBuffer *b = calloc(1, sizeof(*b));
    if (!b)
        return -ENOMEM;
    b->size = size;
    assert(b->vk_buffer == VK_NULL_HANDLE);

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo aci = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };
    VkResult vkr =
        vmaCreateBuffer(ctx->allocator, &bci, &aci, &b->vk_buffer, &b->allocation, &b->info);
    if (vkr != VK_SUCCESS) {
        free(b);
        return -ENOMEM;
    }
    *out_buf = b;
    return 0;
}

void *vmaf_vulkan_buffer_host(VmafVulkanBuffer *buf)
{
    return buf ? buf->info.pMappedData : NULL;
}

uintptr_t vmaf_vulkan_buffer_vkhandle(VmafVulkanBuffer *buf)
{
    return buf ? (uintptr_t)buf->vk_buffer : 0;
}

size_t vmaf_vulkan_buffer_size(VmafVulkanBuffer *buf)
{
    return buf ? buf->size : 0;
}

int vmaf_vulkan_buffer_flush(VmafVulkanContext *ctx, VmafVulkanBuffer *buf)
{
    if (!ctx || !buf)
        return -EINVAL;
    VkResult vkr = vmaFlushAllocation(ctx->allocator, buf->allocation, 0, VK_WHOLE_SIZE);
    return (vkr == VK_SUCCESS) ? 0 : -EIO;
}

void vmaf_vulkan_buffer_free(VmafVulkanContext *ctx, VmafVulkanBuffer *buf)
{
    if (!ctx || !buf)
        return;
    vmaDestroyBuffer(ctx->allocator, buf->vk_buffer, buf->allocation);
    free(buf);
}

/* Legacy void* shim used by the T5-1 smoke test — keep working. */
int vmaf_vulkan_picture_alloc(VmafVulkanContext *ctx, void **out, size_t size)
{
    if (!ctx || !out || size == 0)
        return -EINVAL;
    assert(g_shim_count <= VMAF_VK_SHIM_MAX);
    if (g_shim_count >= VMAF_VK_SHIM_MAX)
        return -ENOMEM;

    VmafVulkanBuffer *b = NULL;
    int err = vmaf_vulkan_buffer_alloc(ctx, &b, size);
    if (err)
        return err;
    assert(b != NULL);

    void *host = vmaf_vulkan_buffer_host(b);
    if (!host) {
        vmaf_vulkan_buffer_free(ctx, b);
        return -ENOMEM;
    }
    g_shim_entries[g_shim_count].host_ptr = host;
    g_shim_entries[g_shim_count].buf = b;
    g_shim_count++;
    *out = host;
    return 0;
}

void vmaf_vulkan_picture_free(VmafVulkanContext *ctx, void *host_ptr)
{
    if (!ctx || !host_ptr)
        return;
    for (size_t i = 0; i < g_shim_count; i++) {
        if (g_shim_entries[i].host_ptr != host_ptr)
            continue;
        vmaf_vulkan_buffer_free(ctx, g_shim_entries[i].buf);
        /* Compact in place — preserve insertion order isn't required. */
        g_shim_entries[i] = g_shim_entries[--g_shim_count];
        return;
    }
}
