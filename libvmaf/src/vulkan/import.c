/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  T7-29 part 2 — VkImage zero-copy import (ADR-0186).
 *
 *  Implements the libvmaf_vulkan.h "import" surface that PR #128
 *  scaffolded as -ENOSYS. Caller hands us an externally-decoded
 *  VkImage (e.g. AVVkFrame from FFmpeg's Vulkan hwframes); we wait
 *  the caller's timeline semaphore on the GPU, transition the
 *  image to TRANSFER_SRC_OPTIMAL, vkCmdCopyImageToBuffer the Y
 *  plane into a per-state staging VkBuffer (HOST_VISIBLE +
 *  COHERENT), and signal completion via a fence.
 *
 *  The staging buffers are allocated to match the DATA_ALIGN-rounded
 *  stride that vmaf_picture_alloc would produce, so
 *  vmaf_vulkan_read_imported_pictures can hand them straight to
 *  vmaf_read_pictures without an extra host memcpy.
 *
 *  Synchronisation model (v1): import_image submits and waits on
 *  the fence in-call. wait_compute is therefore a no-op on the
 *  current path but is kept in the public surface so the v2
 *  async-pending-fence model can drop in without an ABI change.
 */

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/libvmaf_vulkan.h"
#include "../picture.h" /* internal: VmafPicturePrivate, vmaf_picture_priv_init */
#include "../ref.h"     /* internal: VmafRef, vmaf_ref_init */
#include "import_picture.h"
#include "vulkan_internal.h"

static int noop_release_picture(VmafPicture *pic, void *cookie)
{
    (void)pic;
    (void)cookie;
    /* The pixel buffer is owned by the VmafVulkanState's import
     * slot, not by the picture pool. The state's
     * vmaf_vulkan_state_free() (or import_slots_free on
     * geometry change) is what actually releases it. This
     * callback exists so vmaf_picture_unref completes cleanly
     * after vmaf_read_pictures finishes consuming the picture. */
    return 0;
}

/* Mirrors libvmaf/src/picture.c's DATA_ALIGN. Kept local so this
 * file doesn't need to pull in the private "../picture.h" header
 * just for one constant. The two values must stay in sync — if
 * picture.c bumps DATA_ALIGN, this needs to follow. Asserted by
 * test_vulkan_import (compile-time check is awkward across a
 * directory boundary, so the test does a runtime assert). */
#define VMAF_VK_IMPORT_DATA_ALIGN 64u

static size_t aligned_stride_bytes(unsigned w, unsigned bpc)
{
    const unsigned bpp = (bpc > 8u) ? 2u : 1u;
    const unsigned aw = (w + (VMAF_VK_IMPORT_DATA_ALIGN - 1u)) & ~(VMAF_VK_IMPORT_DATA_ALIGN - 1u);
    return (size_t)aw * bpp;
}

static int alloc_command_resources(struct VmafVulkanState *state)
{
    VmafVulkanContext *ctx = state->ctx;
    struct VmafVulkanImportSlots *s = &state->import;

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(ctx->device, &cbai, &s->cmd) != VK_SUCCESS)
        return -ENOMEM;

    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    if (vkCreateFence(ctx->device, &fci, NULL, &s->fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &s->cmd);
        s->cmd = VK_NULL_HANDLE;
        return -ENOMEM;
    }
    return 0;
}

static int lazy_alloc_buffers(struct VmafVulkanState *state, unsigned w, unsigned h, unsigned bpc)
{
    struct VmafVulkanImportSlots *s = &state->import;
    if (s->w != 0u) {
        /* Already allocated — caller must keep the geometry stable. */
        if (s->w != w || s->h != h || s->bpc != bpc)
            return -EINVAL;
        return 0;
    }
    if (w == 0u || h == 0u)
        return -EINVAL;
    if (bpc != 8u && bpc != 10u && bpc != 12u && bpc != 16u)
        return -EINVAL;

    const size_t stride = aligned_stride_bytes(w, bpc);
    const size_t size = stride * (size_t)h;

    int err = vmaf_vulkan_buffer_alloc(state->ctx, &s->ref_buf, size);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(state->ctx, &s->dis_buf, size);
    if (err) {
        vmaf_vulkan_buffer_free(state->ctx, s->ref_buf);
        s->ref_buf = NULL;
        return err;
    }
    err = alloc_command_resources(state);
    if (err) {
        vmaf_vulkan_buffer_free(state->ctx, s->ref_buf);
        vmaf_vulkan_buffer_free(state->ctx, s->dis_buf);
        s->ref_buf = NULL;
        s->dis_buf = NULL;
        return err;
    }

    s->w = w;
    s->h = h;
    s->bpc = bpc;
    s->stride_bytes = stride;
    return 0;
}

void vmaf_vulkan_import_slots_free(struct VmafVulkanState *state)
{
    if (!state)
        return;
    struct VmafVulkanImportSlots *s = &state->import;
    VmafVulkanContext *ctx = state->ctx;
    if (s->fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx->device, s->fence, NULL);
        s->fence = VK_NULL_HANDLE;
    }
    if (s->cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &s->cmd);
        s->cmd = VK_NULL_HANDLE;
    }
    if (s->ref_buf) {
        vmaf_vulkan_buffer_free(ctx, s->ref_buf);
        s->ref_buf = NULL;
    }
    if (s->dis_buf) {
        vmaf_vulkan_buffer_free(ctx, s->dis_buf);
        s->dis_buf = NULL;
    }
    s->w = s->h = s->bpc = 0u;
    s->stride_bytes = 0u;
    s->ref_pending = s->dis_pending = 0;
    s->ref_index = s->dis_index = 0u;
}

static int record_image_to_buffer_copy(struct VmafVulkanState *state, VkImage image,
                                       VkImageLayout current_layout, VkBuffer dst, unsigned w,
                                       unsigned h)
{
    struct VmafVulkanImportSlots *s = &state->import;
    const uint32_t aligned_w = (uint32_t)(s->stride_bytes / ((s->bpc > 8u) ? 2u : 1u));

    if (vkResetCommandBuffer(s->cmd, 0) != VK_SUCCESS)
        return -EIO;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(s->cmd, &cbbi) != VK_SUCCESS)
        return -EIO;

    /* Layout transition into TRANSFER_SRC_OPTIMAL. We do not
     * transition back — caller-owned images are effectively
     * "read-once" between decoder signals, matching how AVVkFrame
     * is consumed. If a caller needs the original layout
     * preserved, they can re-transition on their side. */
    VkImageMemoryBarrier to_src = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = current_layout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(s->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_src);

    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = aligned_w,
        .bufferImageHeight = h,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {w, h, 1},
    };
    vkCmdCopyImageToBuffer(s->cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &region);

    if (vkEndCommandBuffer(s->cmd) != VK_SUCCESS)
        return -EIO;
    return 0;
}

static int submit_and_wait(struct VmafVulkanState *state, VkSemaphore wait_sem, uint64_t wait_value)
{
    struct VmafVulkanImportSlots *s = &state->import;
    VmafVulkanContext *ctx = state->ctx;

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo tsi = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_value,
    };
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &s->cmd,
    };
    if (wait_sem != VK_NULL_HANDLE) {
        si.pNext = &tsi;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &wait_sem;
        si.pWaitDstStageMask = &wait_stage;
    }

    if (vkResetFences(ctx->device, 1, &s->fence) != VK_SUCCESS)
        return -EIO;
    if (vkQueueSubmit(ctx->queue, 1, &si, s->fence) != VK_SUCCESS)
        return -EIO;
    if (vkWaitForFences(ctx->device, 1, &s->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return -EIO;
    return 0;
}

int vmaf_vulkan_import_image(VmafVulkanState *state, uintptr_t vk_image, uint32_t vk_format,
                             uint32_t vk_layout, uintptr_t vk_semaphore,
                             uint64_t vk_semaphore_value, unsigned w, unsigned h, unsigned bpc,
                             int is_ref, unsigned index)
{
    (void)vk_format; /* For v1, we trust the caller's bpc. Future
                      * versions may use vk_format to disambiguate
                      * planar layouts. */
    if (!state || !state->ctx)
        return -EINVAL;
    if (vk_image == 0u)
        return -EINVAL;

    int err = lazy_alloc_buffers(state, w, h, bpc);
    if (err)
        return err;

    struct VmafVulkanImportSlots *s = &state->import;
    VmafVulkanBuffer *dst = is_ref ? s->ref_buf : s->dis_buf;
    err = record_image_to_buffer_copy(state, (VkImage)vk_image, (VkImageLayout)vk_layout,
                                      (VkBuffer)vmaf_vulkan_buffer_vkhandle(dst), w, h);
    if (err)
        return err;

    err = submit_and_wait(state, (VkSemaphore)vk_semaphore, vk_semaphore_value);
    if (err)
        return err;

    /* HOST_VISIBLE | (potentially) NON_COHERENT memory needs an
     * explicit invalidate before the host reads back the GPU's
     * write. VMA's flush helper covers both directions on
     * coherent and non-coherent heaps. */
    err = vmaf_vulkan_buffer_flush(state->ctx, dst);
    if (err)
        return err;

    if (is_ref) {
        s->ref_pending = 1;
        s->ref_index = index;
    } else {
        s->dis_pending = 1;
        s->dis_index = index;
    }
    return 0;
}

int vmaf_vulkan_wait_compute(VmafVulkanState *state)
{
    if (!state || !state->ctx)
        return -EINVAL;
    /* v1 is synchronous — submit_and_wait already drained. Kept
     * in the surface for the v2 async-pending-fence model. */
    return 0;
}

static int build_one_picture(VmafPicture *out, VmafVulkanBuffer *src_buf, unsigned w, unsigned h,
                             unsigned bpc, size_t stride_bytes)
{
    memset(out, 0, sizeof(*out));
    out->pix_fmt = VMAF_PIX_FMT_YUV400P;
    out->bpc = bpc;
    out->w[0] = w;
    out->h[0] = h;
    out->stride[0] = (ptrdiff_t)stride_bytes;
    out->data[0] = vmaf_vulkan_buffer_host(src_buf);
    if (!out->data[0])
        return -EIO;

    int err = vmaf_picture_priv_init(out);
    if (err)
        return err;
    err = vmaf_picture_set_release_callback(out, NULL, noop_release_picture);
    if (err) {
        free(out->priv);
        out->priv = NULL;
        return err;
    }
    err = vmaf_ref_init(&out->ref);
    if (err) {
        free(out->priv);
        out->priv = NULL;
        return err;
    }
    return 0;
}

int vmaf_vulkan_state_build_pictures(VmafVulkanState *state, unsigned index, VmafPicture *out_ref,
                                     VmafPicture *out_dis)
{
    if (!state || !out_ref || !out_dis)
        return -EINVAL;
    struct VmafVulkanImportSlots *s = &state->import;
    if (!s->ref_pending || !s->dis_pending)
        return -EINVAL;
    if (s->ref_index != index || s->dis_index != index)
        return -EINVAL;
    if (s->w == 0u)
        return -EINVAL;

    int err = build_one_picture(out_ref, s->ref_buf, s->w, s->h, s->bpc, s->stride_bytes);
    if (err)
        return err;
    err = build_one_picture(out_dis, s->dis_buf, s->w, s->h, s->bpc, s->stride_bytes);
    if (err) {
        (void)vmaf_picture_unref(out_ref);
        return err;
    }

    s->ref_pending = 0;
    s->dis_pending = 0;
    return 0;
}
