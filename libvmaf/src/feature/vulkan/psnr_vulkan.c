/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature kernel on the Vulkan backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1; chroma extension T3-15(b) / ADR-0210).
 *
 *  Per-pixel squared-error reduction → host-side log10 → score.
 *  One dispatch per plane (Y, Cb, Cr); the same `psnr.comp`
 *  pipeline is invoked three times per frame against per-plane
 *  buffers and per-plane (width, height) push-constants. Chroma
 *  buffers are sized for the active subsampling
 *  (4:2:0 → w/2 × h/2, 4:2:2 → w/2 × h, 4:4:4 → w × h); the
 *  shader is plane-agnostic and reads its dims out of push
 *  constants.
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::extract):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;        (per channel)
 *      mse = sse / (w_p * h_p);
 *      psnr = (sse == 0)
 *             ? psnr_max[p]
 *             : 10 * log10(peak * peak / mse);
 *  Bit-exactness contract: int64 SSE accumulation → places=4 vs CPU.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/motion_vulkan.c
 *  (single-dispatch + per-WG int64 reduction). PSNR remains the
 *  simplest Vulkan extractor in the matrix — 3 small dispatches/
 *  frame, no temporal state.
 *
 *  4:0:0 (YUV400) handling: chroma planes are absent, so only the
 *  luma plane is dispatched and only `psnr_y` is emitted. This
 *  matches CPU integer_psnr.c::init's `enable_chroma = false`
 *  branch.
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
#define PSNR_NUM_PLANES 3

typedef struct {
    /* Per-plane geometry: [0] luma (full), [1] Cb, [2] Cr (subsampled per pix_fmt). */
    unsigned width[PSNR_NUM_PLANES];
    unsigned height[PSNR_NUM_PLANES];
    unsigned bpc;
    uint32_t peak;
    double psnr_max[PSNR_NUM_PLANES];

    /* Number of active planes (1 for YUV400, 3 otherwise). */
    unsigned n_planes;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects. The shader is plane-agnostic — push constants
     * carry the per-plane width/height/num_workgroups_x. One pipeline
     * suffices for all three dispatches; spec-constants pin the *max*
     * (luma) frame dims for the shared-memory size, runtime guards on
     * push-constant width/height inside the shader. */
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader;
    VkPipeline pipeline;
    VkDescriptorPool desc_pool;

    /* Per-plane input buffers (host-mapped). */
    VmafVulkanBuffer *ref_in[PSNR_NUM_PLANES];
    VmafVulkanBuffer *dis_in[PSNR_NUM_PLANES];

    /* Per-plane per-workgroup int64 SE partials. */
    VmafVulkanBuffer *se_partials[PSNR_NUM_PLANES];
    unsigned wg_count[PSNR_NUM_PLANES];

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

    /* Spec constants pin the *max* per-plane dims (luma) for the
     * shared-memory size; runtime guards on push-constant width/height
     * inside the shader, so chroma's smaller dispatches reuse the same
     * pipeline safely. */
    struct {
        int32_t width;
        int32_t height;
        int32_t bpc;
        int32_t subgroup_size;
    } spec_data = {(int32_t)s->width[0], (int32_t)s->height[0], (int32_t)s->bpc, 32};

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

    /* 3 dispatches per frame (Y, Cb, Cr) × 3 storage buffers each =
     * 9 descriptors / 3 sets at peak; size pool for 4 frames in flight
     * with headroom (12 sets, 36 buffer descriptors). */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 12 * 3,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 12,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(dev, &dpci, NULL, &s->desc_pool) != VK_SUCCESS)
        return -ENOMEM;

    return 0;
}

static int alloc_buffers(PsnrVulkanState *s)
{
    const size_t bytes_per_pixel = (s->bpc <= 8) ? 1U : 2U;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const size_t in_bytes = (size_t)s->width[p] * s->height[p] * bytes_per_pixel;
        int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in[p], in_bytes);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_in[p], in_bytes);
        if (err)
            return err;

        uint32_t gx = 0;
        uint32_t gy = 0;
        psnr_wg_dims(s->width[p], s->height[p], &gx, &gy);
        s->wg_count[p] = gx * gy;
        size_t se_bytes = (size_t)s->wg_count[p] * sizeof(int64_t);
        if (se_bytes == 0)
            se_bytes = sizeof(int64_t);
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->se_partials[p], se_bytes);
        if (err)
            return err;
    }
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    PsnrVulkanState *s = fex->priv;
    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;

    /* Per-plane geometry derived from pix_fmt. CPU reference:
     * libvmaf/src/feature/integer_psnr.c::init computes the same
     * (ss_hor, ss_ver) split. YUV400 has chroma absent, so n_planes = 1. */
    s->width[0] = w;
    s->height[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1;
        s->width[1] = s->width[2] = 0;
        s->height[1] = s->height[2] = 0;
    } else {
        s->n_planes = PSNR_NUM_PLANES;
        const int ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P);
        const int ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P);
        const unsigned cw = ss_hor ? (w / 2U) : w;
        const unsigned ch = ss_ver ? (h / 2U) : h;
        s->width[1] = s->width[2] = cw;
        s->height[1] = s->height[2] = ch;
    }

    /* Match CPU integer_psnr.c::init's psnr_max default branch
     * (`min_sse == 0.0`): psnr_max[p] = (6 * bpc) + 12. The CPU path
     * uses a per-plane vector to leave room for the `min_sse`-driven
     * formula; we replicate the array even though all three entries
     * are identical in the default branch, so a future `min_sse`
     * option flip stays a one-line change. */
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++)
        s->psnr_max[p] = (double)(6U * bpc) + 12.0;

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

static int upload_plane(PsnrVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic,
                        unsigned plane)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(dst_buf);
    const uint8_t *src = (const uint8_t *)pic->data[plane];
    const size_t src_stride = (size_t)pic->stride[plane];
    const unsigned w = s->width[plane];
    const unsigned h = s->height[plane];
    const size_t dst_stride = (s->bpc <= 8) ? (size_t)w : (size_t)w * 2U;
    for (unsigned y = 0; y < h; y++)
        memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int write_descriptor_set(PsnrVulkanState *s, VkDescriptorSet set, unsigned plane)
{
    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->se_partials[plane]),
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

static double reduce_se_partials(const PsnrVulkanState *s, unsigned plane)
{
    const int64_t *slots = vmaf_vulkan_buffer_host(s->se_partials[plane]);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count[plane]; i++)
        total += slots[i];
    return (double)total;
}

/* psnr_name[p] — same array as the CPU path
 * (libvmaf/src/feature/integer_psnr.c::psnr_name). */
static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrVulkanState *s = fex->priv;
    int err = 0;
    VkDescriptorSet sets[PSNR_NUM_PLANES] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    unsigned sets_alloc = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    /* 1) Host → device upload + zero out partials, per active plane. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        err = upload_plane(s, s->ref_in[p], ref_pic, p);
        if (err)
            return err;
        err = upload_plane(s, s->dis_in[p], dist_pic, p);
        if (err)
            return err;

        memset(vmaf_vulkan_buffer_host(s->se_partials[p]), 0,
               (size_t)s->wg_count[p] * sizeof(int64_t));
        err = vmaf_vulkan_buffer_flush(s->ctx, s->se_partials[p]);
        if (err)
            return err;
    }

    /* 2) One descriptor set per plane (different SSBO bindings per
     * plane; the pipeline layout is shared). */
    for (unsigned p = 0; p < s->n_planes; p++) {
        VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = s->desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &s->dsl,
        };
        if (vkAllocateDescriptorSets(s->ctx->device, &dsai, &sets[p]) != VK_SUCCESS) {
            err = -ENOMEM;
            goto cleanup;
        }
        sets_alloc++;
        (void)write_descriptor_set(s, sets[p], p);
    }

    /* 3) One command buffer carrying n_planes back-to-back dispatches.
     * No barrier between dispatches — chroma writes go to a different
     * SE-partials SSBO than luma, so they're independent. */
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
    for (unsigned p = 0; p < s->n_planes; p++) {
        uint32_t gx = 0;
        uint32_t gy = 0;
        psnr_wg_dims(s->width[p], s->height[p], &gx, &gy);
        PsnrPushConsts pc = {
            .width = s->width[p],
            .height = s->height[p],
            .bpc = s->bpc,
            .num_workgroups_x = gx,
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_layout, 0, 1,
                                &sets[p], 0, NULL);
        vkCmdPushConstants(cmd, s->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdDispatch(cmd, gx, gy, 1);
    }
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

    /* 4) Host-side reduction + score emit, per active plane.
     * Mirrors integer_psnr.c::psnr emission semantics. */
    const double peak_sq = (double)s->peak * (double)s->peak;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double sse = reduce_se_partials(s, p);
        const double n_pixels = (double)s->width[p] * (double)s->height[p];
        const double mse = sse / n_pixels;
        double psnr = (sse <= 0.0) ? s->psnr_max[p] : 10.0 * log10(peak_sq / mse);
        if (psnr > s->psnr_max[p])
            psnr = s->psnr_max[p];

        const int e = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (e && !err)
            err = e;
    }

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    if (sets_alloc > 0)
        vkFreeDescriptorSets(s->ctx->device, s->desc_pool, sets_alloc, sets);
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

    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
        if (s->ref_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->ref_in[p]);
        if (s->dis_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->dis_in[p]);
        if (s->se_partials[p])
            vmaf_vulkan_buffer_free(s->ctx, s->se_partials[p]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* Provided features — full luma + chroma per T3-15(b) / ADR-0210.
 * For YUV400 sources `init` clamps `n_planes` to 1 and chroma
 * dispatches are skipped, so only `psnr_y` is emitted at runtime —
 * but the static list still claims chroma so the dispatcher routes
 * `psnr_cb` / `psnr_cr` requests through the Vulkan twin. */
static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

VmafFeatureExtractor vmaf_fex_psnr_vulkan = {
    .name = "psnr_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(PsnrVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 3 small dispatches/frame (Y + Cb + Cr in one command buffer),
     * reduction-dominated; AUTO + 1080p area matches motion's
     * profile (see ADR-0181 / ADR-0182 / ADR-0210). */
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
