/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_psnr_hvs feature kernel on the Vulkan backend (T7-23 /
 *  ADR-0188 / ADR-0191, GPU long-tail batch 2 part 3a). First
 *  DCT-based GPU kernel in the fork. Vulkan twin of the active
 *  CPU `psnr_hvs` extractor in
 *  libvmaf/src/feature/third_party/xiph/psnr_hvs.c.
 *
 *  Per-plane single-dispatch design (see ADR-0191):
 *    - One workgroup per output 8×8 block (step=7 sliding window).
 *    - 64 threads/WG (one per coefficient).
 *    - Cooperative load + per-quadrant reductions + integer DCT
 *      + masking + per-coefficient masked-error contribution +
 *      per-WG partial via subgroupAdd.
 *
 *  3 pipelines (one per plane, baked-in CSF + PLANE specialisation
 *  constant). picture_copy host-side normalises each plane's
 *  uint sample → float [0, 255] before upload. Host accumulates
 *  per-WG float partials in `double` per plane, applies
 *  `score / pixels / samplemax²` then `10·log10(1/score)` per
 *  plane. Combined `psnr_hvs = 0.8·Y + 0.1·(Cb + Cr)`.
 *
 *  Rejects YUV400P (no chroma) and `bpc > 12` (matches CPU).
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

#include "psnr_hvs_spv.h" /* generated SPIR-V byte array */
#include "../picture_copy.h"

#define PSNR_HVS_WG_X 8
#define PSNR_HVS_WG_Y 8
#define PSNR_HVS_BLOCK 8
#define PSNR_HVS_STEP 7
#define PSNR_HVS_NUM_PLANES 3
#define PSNR_HVS_NUM_BINDINGS 3 /* ref, dist, partials */

typedef struct {
    /* Frame geometry per plane. */
    unsigned width[PSNR_HVS_NUM_PLANES];
    unsigned height[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_x[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_y[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks[PSNR_HVS_NUM_PLANES];
    unsigned bpc;
    int32_t samplemax_sq;

    /* Vulkan context handle. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects. */
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader;
    VkPipeline pipeline[PSNR_HVS_NUM_PLANES]; /* one per plane (PLANE + BPC baked) */
    VkDescriptorPool desc_pool;

    /* Per-plane input buffers (host-mapped float). */
    VmafVulkanBuffer *ref_in[PSNR_HVS_NUM_PLANES];
    VmafVulkanBuffer *dist_in[PSNR_HVS_NUM_PLANES];
    VmafVulkanBuffer *partials[PSNR_HVS_NUM_PLANES];

    VmafDictionary *feature_name_dict;
} PsnrHvsVulkanState;

static const VmafOption options[] = {{0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t num_blocks_x;
    uint32_t num_blocks_y;
} PsnrHvsPushConsts;

static int build_pipeline_for_plane(PsnrHvsVulkanState *s, int plane, VkPipeline *out_pipeline)
{
    struct {
        int32_t bpc;
        int32_t plane;
        int32_t subgroup_size;
    } spec_data = {(int32_t)s->bpc, (int32_t)plane, 32};

    VkSpecializationMapEntry spec_entries[3] = {
        {.constantID = 0, .offset = offsetof(__typeof__(spec_data), bpc), .size = sizeof(int32_t)},
        {.constantID = 1,
         .offset = offsetof(__typeof__(spec_data), plane),
         .size = sizeof(int32_t)},
        {.constantID = 2,
         .offset = offsetof(__typeof__(spec_data), subgroup_size),
         .size = sizeof(int32_t)},
    };
    VkSpecializationInfo spec_info = {
        .mapEntryCount = 3,
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
    if (vkCreateComputePipelines(s->ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, out_pipeline) !=
        VK_SUCCESS)
        return -ENOMEM;
    return 0;
}

static int create_pipeline(PsnrHvsVulkanState *s)
{
    VkDevice dev = s->ctx->device;

    VkDescriptorSetLayoutBinding bindings[PSNR_HVS_NUM_BINDINGS] = {0};
    for (int i = 0; i < PSNR_HVS_NUM_BINDINGS; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = PSNR_HVS_NUM_BINDINGS,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(dev, &dslci, NULL, &s->dsl) != VK_SUCCESS)
        return -ENOMEM;

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PsnrHvsPushConsts),
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
        .codeSize = psnr_hvs_spv_size,
        .pCode = psnr_hvs_spv,
    };
    if (vkCreateShaderModule(dev, &smci, NULL, &s->shader) != VK_SUCCESS)
        return -ENOMEM;

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        int err = build_pipeline_for_plane(s, p, &s->pipeline[p]);
        if (err)
            return err;
    }

    /* Three descriptor sets — one per plane — each binding ref / dist /
     * partials of that plane. Allocated lazily per-extract. */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = (uint32_t)(2 * PSNR_HVS_NUM_PLANES * PSNR_HVS_NUM_BINDINGS),
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = (uint32_t)(2 * PSNR_HVS_NUM_PLANES),
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(dev, &dpci, NULL, &s->desc_pool) != VK_SUCCESS)
        return -ENOMEM;

    return 0;
}

static int alloc_buffers(PsnrHvsVulkanState *s)
{
    int err = 0;
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        const size_t plane_bytes = (size_t)s->width[p] * s->height[p] * sizeof(float);
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in[p], plane_bytes);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dist_in[p], plane_bytes);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->partials[p], partials_bytes);
    }
    return err ? -ENOMEM : 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    PsnrHvsVulkanState *s = fex->priv;

    if (bpc > 12) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_vulkan: invalid bitdepth (%u); bpc must be ≤ 12\n",
                 bpc);
        return -EINVAL;
    }
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "psnr_hvs_vulkan: YUV400P unsupported (psnr_hvs needs all 3 planes)\n");
        return -EINVAL;
    }
    if (w < (unsigned)PSNR_HVS_BLOCK || h < (unsigned)PSNR_HVS_BLOCK) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_vulkan: input %ux%u smaller than 8×8 block\n", w,
                 h);
        return -EINVAL;
    }

    s->bpc = bpc;
    const int32_t samplemax = (1 << bpc) - 1;
    s->samplemax_sq = samplemax * samplemax;

    /* Plane dims: 4:2:0 → chroma half. 4:2:2 → chroma half-width.
     * 4:4:4 → all three at full. Match VmafPicture's per-plane
     * w[i] / h[i] convention. */
    s->width[0] = w;
    s->height[0] = h;
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        s->width[1] = s->width[2] = w >> 1;
        s->height[1] = s->height[2] = h >> 1;
        break;
    case VMAF_PIX_FMT_YUV422P:
        s->width[1] = s->width[2] = w >> 1;
        s->height[1] = s->height[2] = h;
        break;
    case VMAF_PIX_FMT_YUV444P:
        s->width[1] = s->width[2] = w;
        s->height[1] = s->height[2] = h;
        break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_vulkan: unsupported pix_fmt\n");
        return -EINVAL;
    }

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        if (s->width[p] < (unsigned)PSNR_HVS_BLOCK || s->height[p] < (unsigned)PSNR_HVS_BLOCK) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "psnr_hvs_vulkan: plane %d dims %ux%u smaller than 8×8 block\n", p,
                     s->width[p], s->height[p]);
            return -EINVAL;
        }
        /* CPU loop: for (y = 0; y < H - 7; y += 7); same for x.
         * Number of blocks: floor((H - 7 + 6) / 7) = floor((H - 1) / 7).
         * Safer: count y starts in [0, H - 8] inclusive at step 7
         * = floor((H - 8) / 7) + 1. */
        s->num_blocks_x[p] = (s->width[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks_y[p] = (s->height[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks[p] = s->num_blocks_x[p] * s->num_blocks_y[p];
    }

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_vulkan: cannot create Vulkan context (%d)\n",
                     err);
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

static int upload_plane(PsnrHvsVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic,
                        int plane)
{
    /* picture_copy normalises samples → float [0, 255] regardless
     * of bpc (10/12-bit get divided by 4/16). The shader rescales
     * back to the original integer-domain via `sample_to_int`
     * so the integer DCT matches CPU bit-for-bit. */
    float *dst = vmaf_vulkan_buffer_host(dst_buf);
    if (!dst)
        return -EIO;
    /* picture_copy hardcodes plane 0 — we need a per-plane variant.
     * Use a small inline copy that handles the right plane. */
    const size_t dst_stride = (size_t)s->width[plane] * sizeof(float);
    if (pic->bpc <= 8) {
        const uint8_t *src = (const uint8_t *)pic->data[plane];
        for (unsigned y = 0; y < s->height[plane]; y++) {
            float *dst_row = dst + y * (dst_stride / sizeof(float));
            const uint8_t *src_row = src + y * (size_t)pic->stride[plane];
            for (unsigned x = 0; x < s->width[plane]; x++) {
                dst_row[x] = (float)src_row[x];
            }
        }
    } else {
        const float scaler = (pic->bpc == 10) ? 4.0f :
                             (pic->bpc == 12) ? 16.0f :
                             (pic->bpc == 16) ? 256.0f :
                                                1.0f;
        const uint16_t *src = (const uint16_t *)pic->data[plane];
        const size_t src_stride_words = (size_t)pic->stride[plane] / sizeof(uint16_t);
        for (unsigned y = 0; y < s->height[plane]; y++) {
            float *dst_row = dst + y * (dst_stride / sizeof(float));
            const uint16_t *src_row = src + y * src_stride_words;
            for (unsigned x = 0; x < s->width[plane]; x++) {
                dst_row[x] = (float)src_row[x] / scaler;
            }
        }
    }
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int write_descriptor_set(PsnrHvsVulkanState *s, VkDescriptorSet set, int plane)
{
    VkDescriptorBufferInfo dbi[PSNR_HVS_NUM_BINDINGS] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dist_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->partials[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[PSNR_HVS_NUM_BINDINGS];
    for (int i = 0; i < PSNR_HVS_NUM_BINDINGS; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, PSNR_HVS_NUM_BINDINGS, writes, 0, NULL);
    return 0;
}

static double convert_score_db(double score, double weight)
{
    return 10.0 * (-1.0 * log10(weight * score));
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrHvsVulkanState *s = fex->priv;
    int err = 0;

    /* Upload all three planes. */
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        err = upload_plane(s, s->ref_in[p], ref_pic, p);
        if (err)
            return err;
        err = upload_plane(s, s->dist_in[p], dist_pic, p);
        if (err)
            return err;
    }

    /* One descriptor set per plane. */
    VkDescriptorSet sets[PSNR_HVS_NUM_PLANES] = {VK_NULL_HANDLE};
    VkDescriptorSetLayout layouts[PSNR_HVS_NUM_PLANES] = {s->dsl, s->dsl, s->dsl};
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->desc_pool,
        .descriptorSetCount = (uint32_t)PSNR_HVS_NUM_PLANES,
        .pSetLayouts = layouts,
    };
    if (vkAllocateDescriptorSets(s->ctx->device, &dsai, sets) != VK_SUCCESS)
        return -ENOMEM;
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++)
        write_descriptor_set(s, sets[p], p);

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

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        PsnrHvsPushConsts pc = {
            .width = s->width[p],
            .height = s->height[p],
            .num_blocks_x = s->num_blocks_x[p],
            .num_blocks_y = s->num_blocks_y[p],
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_layout, 0, 1,
                                &sets[p], 0, NULL);
        vkCmdPushConstants(cmd, s->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline[p]);
        vkCmdDispatch(cmd, s->num_blocks_x[p], s->num_blocks_y[p], 1);
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

    /* Per-plane reduction + log10 transform. CPU `calc_psnrhvs`
     * accumulates `ret` in `float` over per-coefficient contribs
     * (line 360 of psnr_hvs.c), then `ret /= pixels` (float / int)
     * and `ret /= samplemax²` (float / int). Match that exactly:
     * sum the per-block partials in float in block iteration
     * order, divide by `pixels` (int) and `samplemax_sq` (int)
     * with implicit promotion. Promoting to double here would
     * be a more-precise-but-different value, surfacing as ~1e-4
     * dB drift vs CPU at places=4. */
    double plane_score[PSNR_HVS_NUM_PLANES];
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        const float *partials = vmaf_vulkan_buffer_host(s->partials[p]);
        float ret = 0.0f;
        for (unsigned i = 0; i < s->num_blocks[p]; i++)
            ret += partials[i];
        const int pixels = (int)(s->num_blocks[p] * 64u);
        ret /= (float)pixels;
        ret /= (float)s->samplemax_sq;
        plane_score[p] = (double)ret;
    }

    static const char *plane_features[PSNR_HVS_NUM_PLANES] = {"psnr_hvs_y", "psnr_hvs_cb",
                                                              "psnr_hvs_cr"};
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        err |= vmaf_feature_collector_append(feature_collector, plane_features[p],
                                             convert_score_db(plane_score[p], 1.0), index);
    }
    const double combined = 0.8 * plane_score[0] + 0.1 * (plane_score[1] + plane_score[2]);
    err |= vmaf_feature_collector_append(feature_collector, "psnr_hvs",
                                         convert_score_db(combined, 1.0), index);

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        if (sets[p] != VK_NULL_HANDLE)
            vkFreeDescriptorSets(s->ctx->device, s->desc_pool, 1, &sets[p]);
    }
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    PsnrHvsVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    if (s->desc_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(dev, s->desc_pool, NULL);
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        if (s->pipeline[p] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->pipeline[p], NULL);
    }
    if (s->shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(dev, s->shader, NULL);
    if (s->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(dev, s->pipeline_layout, NULL);
    if (s->dsl != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(dev, s->dsl, NULL);

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        if (s->ref_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->ref_in[p]);
        if (s->dist_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->dist_in[p]);
        if (s->partials[p])
            vmaf_vulkan_buffer_free(s->ctx, s->partials[p]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr", "psnr_hvs",
                                          NULL};

VmafFeatureExtractor vmaf_fex_psnr_hvs_vulkan = {
    .name = "psnr_hvs_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(PsnrHvsVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 3 dispatches/frame (one per plane). Each dispatch is per-WG
     * = one 8×8 block; per-block ALU is moderate (DCT scalar +
     * 64-thread mask reductions). 1080p has ~32K blocks per plane. */
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
