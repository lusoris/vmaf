/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature kernel on the Vulkan backend (T7-23 / batch 3
 *  part 5a — ADR-0192 / ADR-0197). Float twin of integer_vif's
 *  GPU kernels but the algorithm follows CPU `compute_vif`
 *  (libvmaf/src/feature/vif.c) — separable 4-scale pyramid with
 *  decimation between scales.
 *
 *  v1 scope: `vif_kernelscale = 1.0` only (the production default).
 *  Filter widths per scale: {17, 9, 5, 3}.
 *
 *  Per-frame flow (4 scales):
 *     scale 0 compute: read raw ref/dis            → (num0, den0)
 *     scale 1 decimate: read raw → buf_A           (filter+ds by 2)
 *     scale 1 compute:  read buf_A                  → (num1, den1)
 *     scale 2 decimate: read buf_A → buf_B
 *     scale 2 compute:  read buf_B                  → (num2, den2)
 *     scale 3 decimate: read buf_B → buf_A
 *     scale 3 compute:  read buf_A                  → (num3, den3)
 *
 *  Host: per-scale (num, den) double accumulation; final emit:
 *     float_vif_scaleN_score = num[N] / den[N]   (4 scores)
 *     debug:                                       vif, vif_num, vif_den, vif_num_scaleN/vif_den_scaleN
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

#include "float_vif_spv.h"

#define FVIF_WG_X 16
#define FVIF_WG_Y 16

/* Default vif_kernelscale=1.0 filter widths per scale. */
static const int FVIF_FW[4] = {17, 9, 5, 3};

typedef struct {
    bool debug;
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;

    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Per-scale dimensions (CPU-mirroring; scale 0 is full, each
     * subsequent scale is (prev - 2*HALF_FW(scale)) / 2). */
    unsigned scale_w[4];
    unsigned scale_h[4];

    VmafVulkanContext *ctx;
    int owns_ctx;

    VkDescriptorSetLayout dsl;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader;
    /* 7 pipelines: pipelines[mode][scale]. mode 0 = compute (4 scales),
     * mode 1 = decimate (3 scales — 1, 2, 3; index 0 unused). */
    VkPipeline pipelines[2][4];
    VkDescriptorPool desc_pool;

    /* Raw input buffers (uint8/16). */
    VmafVulkanBuffer *ref_raw;
    VmafVulkanBuffer *dis_raw;

    /* Two float ping-pong buffers (ref + dis each). buf_A reused for
     * scale 1 input and scale 3 input; buf_B for scale 2 input. */
    VmafVulkanBuffer *ref_buf[2];
    VmafVulkanBuffer *dis_buf[2];

    /* Per-scale (num, den) partials. Each pair sized to the scale's
     * WG count; we use the worst-case (scale 0) WG count for all four
     * to keep allocation trivial. */
    VmafVulkanBuffer *num_partials[4];
    VmafVulkanBuffer *den_partials[4];
    unsigned wg_count[4];

    VmafDictionary *feature_name_dict;
} FloatVifVulkanState;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
    uint32_t in_width;
    uint32_t in_height;
} FloatVifPushConsts;

static const VmafOption options[] = {{.name = "debug",
                                      .help = "debug mode: enable additional output",
                                      .offset = offsetof(FloatVifVulkanState, debug),
                                      .type = VMAF_OPT_TYPE_BOOL,
                                      .default_val.b = false},
                                     {.name = "vif_enhn_gain_limit",
                                      .alias = "egl",
                                      .help = "enhancement gain imposed on vif (>= 1.0)",
                                      .offset = offsetof(FloatVifVulkanState, vif_enhn_gain_limit),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 100.0,
                                      .min = 1.0,
                                      .max = 100.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {.name = "vif_kernelscale",
                                      .help = "scaling factor for the gaussian kernel",
                                      .offset = offsetof(FloatVifVulkanState, vif_kernelscale),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 1.0,
                                      .min = 0.1,
                                      .max = 4.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {.name = "vif_sigma_nsq",
                                      .alias = "snsq",
                                      .help = "neural noise variance",
                                      .offset = offsetof(FloatVifVulkanState, vif_sigma_nsq),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 2.0,
                                      .min = 0.0,
                                      .max = 5.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {0}};

static inline void wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + FVIF_WG_X - 1u) / FVIF_WG_X;
    *gy = (h + FVIF_WG_Y - 1u) / FVIF_WG_Y;
}

static int create_pipelines(FloatVifVulkanState *s)
{
    VkDevice dev = s->ctx->device;

    /* 8 storage-buffer bindings (some unused per pipeline; bound to
     * the same set so descriptor management stays simple). */
    VkDescriptorSetLayoutBinding bindings[8] = {0};
    for (int i = 0; i < 8; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 8,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(dev, &dslci, NULL, &s->dsl) != VK_SUCCESS)
        return -ENOMEM;

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(FloatVifPushConsts),
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
        .codeSize = float_vif_spv_size,
        .pCode = float_vif_spv,
    };
    if (vkCreateShaderModule(dev, &smci, NULL, &s->shader) != VK_SUCCESS)
        return -ENOMEM;

    for (int mode = 0; mode < 2; mode++) {
        for (int scale = 0; scale < 4; scale++) {
            if (mode == 1 && scale == 0)
                continue;
            struct {
                int32_t mode;
                int32_t scale;
                int32_t subgroup_size;
            } spec_data = {mode, scale, 32};
            VkSpecializationMapEntry spec_entries[3] = {
                {.constantID = 0,
                 .offset = offsetof(__typeof__(spec_data), mode),
                 .size = sizeof(int32_t)},
                {.constantID = 1,
                 .offset = offsetof(__typeof__(spec_data), scale),
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
                .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                          .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                          .module = s->shader,
                          .pName = "main",
                          .pSpecializationInfo = &spec_info},
                .layout = s->pipeline_layout,
            };
            if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                         &s->pipelines[mode][scale]) != VK_SUCCESS)
                return -ENOMEM;
        }
    }

    /* Up to 7 dispatches per frame → 7 descriptor sets. Round up to 8. */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 8 * 8,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 8,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(dev, &dpci, NULL, &s->desc_pool) != VK_SUCCESS)
        return -ENOMEM;
    return 0;
}

static void compute_per_scale_dims(FloatVifVulkanState *s)
{
    /* CPU `decimate_to_next_scale` runs with VIF_OPT_HANDLE_BORDERS
     * defined (vif_options.h) → buf_valid_{w,h} = full filtered dims,
     * mu_adj = b->mu (no border crop). vif_dec2_s then plain-subsamples
     * by 2 starting at (0,0). So each scale's output is simply
     * floor(prev_dim / 2) — no hfw_prev shrink. */
    s->scale_w[0] = s->width;
    s->scale_h[0] = s->height;
    for (int i = 1; i < 4; i++) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }
}

static int alloc_buffers(FloatVifVulkanState *s)
{
    size_t bpp = (s->bpc <= 8) ? 1 : 2;
    size_t raw_bytes = (size_t)s->width * s->height * bpp;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_raw, raw_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_raw, raw_bytes);
    if (err)
        return err;

    /* Float ping-pong: scale 1 input is the largest; allocate both at
     * scale-1 size + slack for scale 2/3 reuse. */
    size_t float_bytes = (size_t)s->scale_w[1] * s->scale_h[1] * sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[0], float_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_buf[0], float_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[1], float_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_buf[1], float_bytes);
    if (err)
        return err;

    /* Per-scale (num, den) partials, sized per scale's WG count. */
    for (int i = 0; i < 4; i++) {
        uint32_t gx = 0, gy = 0;
        wg_dims(s->scale_w[i], s->scale_h[i], &gx, &gy);
        s->wg_count[i] = gx * gy;
        size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);
        if (pbytes == 0)
            pbytes = sizeof(float);
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->num_partials[i], pbytes);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->den_partials[i], pbytes);
        if (err)
            return err;
    }
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    FloatVifVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    if (s->vif_kernelscale != 1.0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_vif_vulkan: only vif_kernelscale=1.0 is supported in v1\n");
        return -EINVAL;
    }
    compute_per_scale_dims(s);

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "float_vif_vulkan: cannot create Vulkan context (%d)\n",
                     err);
            return err;
        }
        s->owns_ctx = 1;
    }

    int err = create_pipelines(s);
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

static int upload_plane(FloatVifVulkanState *s, VmafPicture *pic, VmafVulkanBuffer *buf)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static int alloc_set_and_bind(FloatVifVulkanState *s, VkDescriptorSet *set,
                              VmafVulkanBuffer *bufs[8])
{
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &s->dsl,
    };
    if (vkAllocateDescriptorSets(s->ctx->device, &dsai, set) != VK_SUCCESS)
        return -ENOMEM;

    VkDescriptorBufferInfo dbi[8];
    VkWriteDescriptorSet writes[8];
    for (int i = 0; i < 8; i++) {
        dbi[i] = (VkDescriptorBufferInfo){
            .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(bufs[i]),
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = *set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 8, writes, 0, NULL);
    return 0;
}

static void cmd_storage_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static int dispatch_pass(FloatVifVulkanState *s, VkCommandBuffer cmd, int mode, int scale,
                         unsigned out_w, unsigned out_h, unsigned in_w, unsigned in_h,
                         int ref_in_idx, int dis_in_idx, int ref_out_idx, int dis_out_idx,
                         VkDescriptorSet *set_out)
{
    /* Build bindings array per the shader's binding ABI (see header). */
    VmafVulkanBuffer *bufs[8];
    bufs[0] = s->ref_raw;
    bufs[1] = s->dis_raw;
    bufs[2] = (ref_in_idx >= 0) ? s->ref_buf[ref_in_idx] : s->ref_buf[0];
    bufs[3] = (dis_in_idx >= 0) ? s->dis_buf[dis_in_idx] : s->dis_buf[0];
    bufs[4] = (ref_out_idx >= 0) ? s->ref_buf[ref_out_idx] : s->ref_buf[0];
    bufs[5] = (dis_out_idx >= 0) ? s->dis_buf[dis_out_idx] : s->dis_buf[0];
    bufs[6] = s->num_partials[scale];
    bufs[7] = s->den_partials[scale];

    int err = alloc_set_and_bind(s, set_out, bufs);
    if (err)
        return err;

    uint32_t gx = 0, gy = 0;
    wg_dims(out_w, out_h, &gx, &gy);
    FloatVifPushConsts pc = {
        .width = out_w,
        .height = out_h,
        .bpc = s->bpc,
        .num_workgroups_x = gx,
        .in_width = in_w,
        .in_height = in_h,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[mode][scale]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_layout, 0, 1, set_out,
                            0, NULL);
    vkCmdPushConstants(cmd, s->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, gx, gy, 1);
    cmd_storage_barrier(cmd);
    return 0;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    FloatVifVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, ref_pic, s->ref_raw);
    if (err)
        return err;
    err = upload_plane(s, dist_pic, s->dis_raw);
    if (err)
        return err;

    /* Reset partials. */
    for (int i = 0; i < 4; i++) {
        memset(vmaf_vulkan_buffer_host(s->num_partials[i]), 0,
               (size_t)s->wg_count[i] * sizeof(float));
        memset(vmaf_vulkan_buffer_host(s->den_partials[i]), 0,
               (size_t)s->wg_count[i] * sizeof(float));
        err = vmaf_vulkan_buffer_flush(s->ctx, s->num_partials[i]);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_flush(s->ctx, s->den_partials[i]);
        if (err)
            return err;
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSet sets[7] = {VK_NULL_HANDLE};
    int n_sets = 0;
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

    /* Scale 0 compute (read raw, write num/den 0). */
    err = dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/0, s->scale_w[0], s->scale_h[0],
                        s->scale_w[0], s->scale_h[0], -1, -1, -1, -1, &sets[n_sets++]);
    if (err)
        goto cleanup;

    /* Scale 1 decimate (raw → buf[0]). */
    err = dispatch_pass(s, cmd, /*mode=*/1, /*scale=*/1, s->scale_w[1], s->scale_h[1],
                        s->scale_w[0], s->scale_h[0], -1, -1, 0, 0, &sets[n_sets++]);
    if (err)
        goto cleanup;

    /* Scale 1 compute (read buf[0], write num/den 1). */
    err = dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/1, s->scale_w[1], s->scale_h[1],
                        s->scale_w[1], s->scale_h[1], 0, 0, -1, -1, &sets[n_sets++]);
    if (err)
        goto cleanup;

    /* Scale 2 decimate (buf[0] → buf[1]). */
    err = dispatch_pass(s, cmd, /*mode=*/1, /*scale=*/2, s->scale_w[2], s->scale_h[2],
                        s->scale_w[1], s->scale_h[1], 0, 0, 1, 1, &sets[n_sets++]);
    if (err)
        goto cleanup;

    /* Scale 2 compute. */
    err = dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/2, s->scale_w[2], s->scale_h[2],
                        s->scale_w[2], s->scale_h[2], 1, 1, -1, -1, &sets[n_sets++]);
    if (err)
        goto cleanup;

    /* Scale 3 decimate (buf[1] → buf[0]). */
    err = dispatch_pass(s, cmd, /*mode=*/1, /*scale=*/3, s->scale_w[3], s->scale_h[3],
                        s->scale_w[2], s->scale_h[2], 1, 1, 0, 0, &sets[n_sets++]);
    if (err)
        goto cleanup;

    /* Scale 3 compute. */
    err = dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/3, s->scale_w[3], s->scale_h[3],
                        s->scale_w[3], s->scale_h[3], 0, 0, -1, -1, &sets[n_sets++]);
    if (err)
        goto cleanup;

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

    /* Reduce per-scale partials in double, emit ratios + (debug) totals. */
    double scores[8];
    for (int i = 0; i < 4; i++) {
        const float *num_slots = vmaf_vulkan_buffer_host(s->num_partials[i]);
        const float *den_slots = vmaf_vulkan_buffer_host(s->den_partials[i]);
        double total_num = 0.0;
        double total_den = 0.0;
        for (unsigned j = 0; j < s->wg_count[i]; j++) {
            total_num += (double)num_slots[j];
            total_den += (double)den_slots[j];
        }
        scores[2 * i + 0] = total_num;
        scores[2 * i + 1] = total_den;
    }

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "VMAF_feature_vif_scale0_score",
                                                  scores[0] / scores[1], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_vif_scale1_score",
                                                      scores[2] / scores[3], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_vif_scale2_score",
                                                      scores[4] / scores[5], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_vif_scale3_score",
                                                      scores[6] / scores[7], index);

    if (s->debug && !err) {
        double score_num = scores[0] + scores[2] + scores[4] + scores[6];
        double score_den = scores[1] + scores[3] + scores[5] + scores[7];
        double score = score_den == 0.0 ? 1.0 : score_num / score_den;
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "vif", score, index);
        if (!err)
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "vif_num", score_num, index);
        if (!err)
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "vif_den", score_den, index);
        const char *names[8] = {"vif_num_scale0", "vif_den_scale0", "vif_num_scale1",
                                "vif_den_scale1", "vif_num_scale2", "vif_den_scale2",
                                "vif_num_scale3", "vif_den_scale3"};
        for (int i = 0; i < 8 && !err; i++) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          names[i], scores[i], index);
        }
    }

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    for (int i = 0; i < n_sets; i++) {
        if (sets[i] != VK_NULL_HANDLE)
            vkFreeDescriptorSets(s->ctx->device, s->desc_pool, 1, &sets[i]);
    }
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    FloatVifVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    if (s->desc_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(dev, s->desc_pool, NULL);
    for (int mode = 0; mode < 2; mode++) {
        for (int scale = 0; scale < 4; scale++) {
            if (s->pipelines[mode][scale] != VK_NULL_HANDLE)
                vkDestroyPipeline(dev, s->pipelines[mode][scale], NULL);
        }
    }
    if (s->shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(dev, s->shader, NULL);
    if (s->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(dev, s->pipeline_layout, NULL);
    if (s->dsl != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(dev, s->dsl, NULL);

    if (s->ref_raw)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_raw);
    if (s->dis_raw)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_raw);
    for (int i = 0; i < 2; i++) {
        if (s->ref_buf[i])
            vmaf_vulkan_buffer_free(s->ctx, s->ref_buf[i]);
        if (s->dis_buf[i])
            vmaf_vulkan_buffer_free(s->ctx, s->dis_buf[i]);
    }
    for (int i = 0; i < 4; i++) {
        if (s->num_partials[i])
            vmaf_vulkan_buffer_free(s->ctx, s->num_partials[i]);
        if (s->den_partials[i])
            vmaf_vulkan_buffer_free(s->ctx, s->den_partials[i]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"VMAF_feature_vif_scale0_score",
                                          "VMAF_feature_vif_scale1_score",
                                          "VMAF_feature_vif_scale2_score",
                                          "VMAF_feature_vif_scale3_score",
                                          "vif",
                                          "vif_num",
                                          "vif_den",
                                          "vif_num_scale0",
                                          "vif_den_scale0",
                                          "vif_num_scale1",
                                          "vif_den_scale1",
                                          "vif_num_scale2",
                                          "vif_den_scale2",
                                          "vif_num_scale3",
                                          "vif_den_scale3",
                                          NULL};

VmafFeatureExtractor vmaf_fex_float_vif_vulkan = {
    .name = "float_vif_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(FloatVifVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
