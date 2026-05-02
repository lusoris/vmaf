/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ansnr feature kernel on the Vulkan backend (T7-23 / batch 3
 *  part 2a — ADR-0192 / ADR-0194). Single-dispatch GLSL kernel that
 *  applies the 3x3 ref filter and 5x5 dis filter from
 *  libvmaf/src/feature/ansnr_tools.c, then reduces per-pixel
 *  `ref_filtr * ref_filtr` (sig) and `(ref_filtr - filtd)²` (noise)
 *  into per-WG float partials. Host reduces in `double` and emits
 *  the two CPU outputs:
 *
 *    float_ansnr  = 10 * log10(sig / noise)   (or psnr_max if noise == 0)
 *    float_anpsnr = MIN(10*log10(peak² · w · h / max(noise, 1e-10)), psnr_max)
 *
 *  Pattern reference: ciede_vulkan.c (single-dispatch float per-WG
 *  partials) + motion_v2_vulkan.c (raw-pixel ping-pong, but we don't
 *  need ping-pong here because ansnr is non-temporal).
 *
 *  Bit-exact disclaimer: the float convolution + per-WG reduction
 *  introduces tiny ULP drift vs the CPU's left-to-right accumulation
 *  order. Precision contract per ADR-0192: places=3.
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

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"

#include "float_ansnr_spv.h" /* generated SPIR-V byte array */

#define ANSNR_WG_X 16
#define ANSNR_WG_Y 16
#define ANSNR_NUM_BINDINGS 4

typedef struct {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    double peak;
    double psnr_max;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle, ADR-0221). */
    VmafVulkanKernelPipeline pl;

    /* Per-frame upload of ref + dis raw pixels. */
    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *dis_in;

    /* Per-WG float partials: sig and noise (one float per WG each). */
    VmafVulkanBuffer *sig_partials;
    VmafVulkanBuffer *noise_partials;
    unsigned wg_count;

    VmafDictionary *feature_name_dict;
} AnsnrVulkanState;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} AnsnrPushConsts;

static inline void ansnr_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + ANSNR_WG_X - 1u) / ANSNR_WG_X;
    *gy = (h + ANSNR_WG_Y - 1u) / ANSNR_WG_Y;
}

static int create_pipelines(AnsnrVulkanState *s)
{
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
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = (uint32_t)ANSNR_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(AnsnrPushConsts),
        .spv_bytes = float_ansnr_spv,
        .spv_size = float_ansnr_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = 4U,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
}

static int alloc_buffers(AnsnrVulkanState *s)
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
    ansnr_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t partial_bytes = (size_t)s->wg_count * sizeof(float);
    if (partial_bytes == 0)
        partial_bytes = sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->sig_partials, partial_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->noise_partials, partial_bytes);
    if (err)
        return err;
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    AnsnrVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    /* Match CPU init() peak / psnr_max table. */
    if (bpc == 8) {
        s->peak = 255.0;
        s->psnr_max = 60.0;
    } else if (bpc == 10) {
        s->peak = 255.75;
        s->psnr_max = 72.0;
    } else if (bpc == 12) {
        s->peak = 255.9375;
        s->psnr_max = 84.0;
    } else if (bpc == 16) {
        s->peak = 255.99609375;
        s->psnr_max = 108.0;
    } else
        return -EINVAL;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "float_ansnr_vulkan: cannot create Vulkan context (%d)\n", err);
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

static int upload_plane(AnsnrVulkanState *s, VmafPicture *pic, VmafVulkanBuffer *buf)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static int write_descriptor_set(AnsnrVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[4] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->sig_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->noise_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 4, writes, 0, NULL);
    return 0;
}

static void reduce_partials(const AnsnrVulkanState *s, double *sig_out, double *noise_out)
{
    const float *sig_slots = vmaf_vulkan_buffer_host(s->sig_partials);
    const float *noise_slots = vmaf_vulkan_buffer_host(s->noise_partials);
    double sig = 0.0;
    double noise = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++) {
        sig += (double)sig_slots[i];
        noise += (double)noise_slots[i];
    }
    *sig_out = sig;
    *noise_out = noise;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    AnsnrVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, ref_pic, s->ref_in);
    if (err)
        return err;
    err = upload_plane(s, dist_pic, s->dis_in);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->sig_partials), 0, (size_t)s->wg_count * sizeof(float));
    memset(vmaf_vulkan_buffer_host(s->noise_partials), 0, (size_t)s->wg_count * sizeof(float));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->sig_partials);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_flush(s->ctx, s->noise_partials);
    if (err)
        return err;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->pl.desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &s->pl.dsl,
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
    ansnr_wg_dims(s->width, s->height, &gx, &gy);
    AnsnrPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .num_workgroups_x = gx,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1, &set,
                            0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
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

    /* Final ANSNR / ANPSNR transforms — match ansnr.c::compute_ansnr. */
    double sig = 0.0;
    double noise = 0.0;
    reduce_partials(s, &sig, &noise);

    const double score = (noise == 0.0) ? s->psnr_max : 10.0 * log10(sig / noise);
    const double eps = 1e-10;
    const double n_pix = (double)s->width * (double)s->height;
    const double max_noise = noise > eps ? noise : eps;
    double score_psnr = 10.0 * log10(s->peak * s->peak * n_pix / max_noise);
    if (score_psnr > s->psnr_max)
        score_psnr = s->psnr_max;

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "float_ansnr", score, index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_anpsnr", score_psnr, index);

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    if (set != VK_NULL_HANDLE)
        vkFreeDescriptorSets(s->ctx->device, s->pl.desc_pool, 1, &set);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    AnsnrVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    vkDeviceWaitIdle(s->ctx->device);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_in);
    if (s->dis_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_in);
    if (s->sig_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->sig_partials);
    if (s->noise_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->noise_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"float_ansnr", "float_anpsnr", NULL};

VmafFeatureExtractor vmaf_fex_float_ansnr_vulkan = {
    .name = "float_ansnr_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .priv_size = sizeof(AnsnrVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
