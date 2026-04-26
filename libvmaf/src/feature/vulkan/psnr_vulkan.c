/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature kernel on the Vulkan backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1).
 *
 *  Per-pixel squared-error reduction → host-side log10 → score.
 *  Single dispatch per channel; this v1 emits luma-only (`psnr_y`).
 *  Chroma support (`psnr_cb` / `psnr_cr`) is a focused follow-up:
 *  the host loop here is plane-agnostic, but the
 *  `picture_vulkan` upload path is luma-only today.
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::extract):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;        (per channel)
 *      mse = sse / (w * h);
 *      psnr = (sse == 0)
 *             ? psnr_max
 *             : 10 * log10(peak * peak / mse);
 *  Bit-exactness contract: int64 SSE accumulation → places=4 vs CPU.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/motion_vulkan.c
 *  (single-dispatch + per-WG int64 reduction). PSNR is the simplest
 *  Vulkan extractor in the matrix — 1 dispatch/frame, no temporal
 *  state.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"

#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"

#include "psnr_spv.h" /* generated SPIR-V byte array */

#define PSNR_WG_X 16
#define PSNR_WG_Y 8

typedef struct {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    uint32_t peak;
    double psnr_max_y;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects. */
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader;
    VkPipeline pipeline;
    VkDescriptorPool desc_pool;

    /* Per-channel input buffers (host-mapped). v1 uses [0] only. */
    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *dis_in;

    /* Per-workgroup int64 SE partials. */
    VmafVulkanBuffer *se_partials;
    unsigned wg_count;

    VmafDictionary *feature_name_dict;
} PsnrVulkanState;

static const VmafOption options[] = {{0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} PsnrPushConsts;

static inline void psnr_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + PSNR_WG_X - 1u) / PSNR_WG_X;
    *gy = (h + PSNR_WG_Y - 1u) / PSNR_WG_Y;
}

static int create_pipeline(PsnrVulkanState *s)
{
    VkDevice dev = s->ctx->device;

    /* 3 storage-buffer bindings: ref, dis, SE partials. */
    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(dev, &dslci, NULL, &s->dsl) != VK_SUCCESS)
        return -ENOMEM;

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PsnrPushConsts),
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &s->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (vkCreatePipelineLayout(dev, &plci, NULL, &s->pipeline_layout) != VK_SUCCESS)
        return -ENOMEM;

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = psnr_spv_size,
        .pCode = psnr_spv,
    };
    if (vkCreateShaderModule(dev, &smci, NULL, &s->shader) != VK_SUCCESS)
        return -ENOMEM;

    struct {
        int32_t width;
        int32_t height;
        int32_t bpc;
        int32_t subgroup_size;
    } spec_data = {(int32_t)s->width, (int32_t)s->height, (int32_t)s->bpc, 32};

    VkSpecializationMapEntry spec_entries[4] = {
        {.constantID = 0,
         .offset = offsetof(__typeof__(spec_data), width),
         .size = sizeof(int32_t)},
        {.constantID = 1,
         .offset = offsetof(__typeof__(spec_data), height),
         .size = sizeof(int32_t)},
        {.constantID = 2, .offset = offsetof(__typeof__(spec_data), bpc), .size = sizeof(int32_t)},
        {.constantID = 3,
         .offset = offsetof(__typeof__(spec_data), subgroup_size),
         .size = sizeof(int32_t)},
    };
    VkSpecializationInfo spec_info = {
        .mapEntryCount = 4,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(spec_data),
        .pData = &spec_data,
    };
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = s->shader,
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
        .layout = s->pipeline_layout,
    };
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &s->pipeline) != VK_SUCCESS)
        return -ENOMEM;

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4 * 3,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 4,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(dev, &dpci, NULL, &s->desc_pool) != VK_SUCCESS)
        return -ENOMEM;

    return 0;
}

static int alloc_buffers(PsnrVulkanState *s)
{
    size_t bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bytes_per_pixel;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, in_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_in, in_bytes);
    if (err)
        return err;

    uint32_t gx = 0;
    uint32_t gy = 0;
    psnr_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t se_bytes = (size_t)s->wg_count * sizeof(int64_t);
    if (se_bytes == 0)
        se_bytes = sizeof(int64_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->se_partials, se_bytes);
    return err;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    PsnrVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;

    /* Match CPU integer_psnr.c psnr_max default: peak^2 / min_sse where
     * min_sse = 0.5 (one ULP for 8-bit). Simplified for Vulkan v1 — the
     * extractor only emits psnr_y so a single max suffices. */
    const double peak_d = (double)s->peak;
    s->psnr_max_y = 10.0 * log10((peak_d * peak_d) / 0.5);

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_vulkan: cannot create Vulkan context (%d)\n", err);
            return err;
        }
        s->owns_ctx = 1;
    }

    int err = create_pipeline(s);
    if (err)
        return err;

    err = alloc_buffers(s);
    if (err)
        return err;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

static int upload_plane(PsnrVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(dst_buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int write_descriptor_set(PsnrVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->se_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 3, writes, 0, NULL);
    return 0;
}

static double reduce_se_partials(const PsnrVulkanState *s)
{
    const int64_t *slots = vmaf_vulkan_buffer_host(s->se_partials);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += slots[i];
    return (double)total;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, s->ref_in, ref_pic);
    if (err)
        return err;
    err = upload_plane(s, s->dis_in, dist_pic);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->se_partials), 0, (size_t)s->wg_count * sizeof(int64_t));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->se_partials);
    if (err)
        return err;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &s->dsl,
    };
    if (vkAllocateDescriptorSets(s->ctx->device, &dsai, &set) != VK_SUCCESS)
        return -ENOMEM;
    write_descriptor_set(s, set);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = s->ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(s->ctx->device, &cbai, &cmd) != VK_SUCCESS) {
        err = -ENOMEM;
        goto cleanup;
    }
    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &cbbi);

    uint32_t gx = 0;
    uint32_t gy = 0;
    psnr_wg_dims(s->width, s->height, &gx, &gy);
    PsnrPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .num_workgroups_x = gx,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_layout, 0, 1, &set, 0,
                            NULL);
    vkCmdPushConstants(cmd, s->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, gx, gy, 1);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(s->ctx->device, &fci, NULL, &fence) != VK_SUCCESS) {
        err = -ENOMEM;
        goto cleanup;
    }
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    if (vkQueueSubmit(s->ctx->queue, 1, &si, fence) != VK_SUCCESS) {
        err = -EIO;
        goto cleanup;
    }
    vkWaitForFences(s->ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

    /* Host-side reduction + score emit. */
    const double sse = reduce_se_partials(s);
    const double n_pixels = (double)s->width * (double)s->height;
    const double mse = sse / n_pixels;
    double psnr_y = (sse <= 0.0) ? s->psnr_max_y : 10.0 * log10(((double)s->peak * s->peak) / mse);
    if (psnr_y > s->psnr_max_y)
        psnr_y = s->psnr_max_y;

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict, "psnr_y",
                                                  psnr_y, index);

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    if (set != VK_NULL_HANDLE)
        vkFreeDescriptorSets(s->ctx->device, s->desc_pool, 1, &set);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    PsnrVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    if (s->desc_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(dev, s->desc_pool, NULL);
    if (s->pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(dev, s->pipeline, NULL);
    if (s->shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(dev, s->shader, NULL);
    if (s->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(dev, s->pipeline_layout, NULL);
    if (s->dsl != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(dev, s->dsl, NULL);

    if (s->ref_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_in);
    if (s->dis_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_in);
    if (s->se_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->se_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* Provided features — luma-only v1. Chroma is a focused follow-up
 * (the `picture_vulkan` upload path is luma-only today). */
static const char *provided_features[] = {"psnr_y", NULL};

VmafFeatureExtractor vmaf_fex_psnr_vulkan = {
    .name = "psnr_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(PsnrVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 1 dispatch/frame, reduction-dominated; AUTO + 1080p area
     * matches motion's profile (see ADR-0181 / ADR-0182). */
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
