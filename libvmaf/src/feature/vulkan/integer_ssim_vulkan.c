/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_ssim feature extractor on the Vulkan backend.
 *  Port of feature/cuda/integer_ssim_cuda.c to Vulkan, analogous
 *  to how ssim_vulkan.c (float path) was authored.
 *
 *  Unlike the float twin (ssim_vulkan.c) this extractor uploads raw
 *  integer luma samples (uint8 or uint16) directly into the SSBO and
 *  lets the GLSL shader (shaders/integer_ssim.comp) normalise them to
 *  float in [0, 255] — the same inline normalisation the CUDA twin
 *  performs via read_norm_8bpc / read_norm_16bpc.  The float twin used
 *  picture_copy() to pre-convert to float on the CPU before uploading;
 *  this approach avoids that staging cost.
 *
 *  Two-dispatch design (mirrors ssim_vulkan.c / ADR-0189):
 *    1. main_horiz:        horizontal 11-tap separable Gaussian over
 *                          raw integer samples → 5 float intermediate
 *                          buffers sized (W - 10) × H.
 *    2. main_vert_combine: vertical 11-tap on intermediates + per-pixel
 *                          SSIM combine + per-WG float partial sum.
 *
 *  Host: accumulates per-WG float partials in double, divides by
 *  (W - 10)·(H - 10) and emits `ssim` (the integer-path feature name
 *  consumed by the standard VMAF models).
 *
 *  v1: scale=1 only — same constraint as ssim_vulkan.c and
 *  integer_ssim_cuda.c. Auto-decimation rejects scale > 1 with -EINVAL.
 *
 *  Submit-pool model mirrors ssim_vulkan.c (T-GPU-OPT-VK-1 / ADR-0353):
 *    - Pre-allocated descriptor set with all 8 SSBO bindings written
 *      once at init() — no per-frame vkUpdateDescriptorSets.
 *    - Single-slot submit pool; both passes share one command buffer.
 */

#include <errno.h>
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
#include "../../vulkan/kernel_template.h"

#include "integer_ssim_spv.h" /* generated SPIR-V byte array */

#define ISSIM_WG_X 16
#define ISSIM_WG_Y 8
#define ISSIM_K 11
#define ISSIM_NUM_BINDINGS 8

typedef struct {
    unsigned width;
    unsigned height;
    unsigned bpc;
    int scale_override;

    unsigned w_horiz;
    unsigned h_horiz;
    unsigned w_final;
    unsigned h_final;
    unsigned wg_count_x;
    unsigned wg_count_y;
    unsigned wg_count;

    float c1;
    float c2;

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Two pipelines (pass=0 horizontal, pass=1 vertical+combine)
     * following the same pattern as ssim_vulkan.c. */
    VmafVulkanKernelPipeline pl;
    VkPipeline pipeline_vert;

    VmafVulkanKernelSubmitPool sub_pool;
    VkDescriptorSet pre_set;

    /* Raw integer input buffers — uploaded via memcpy without float
     * conversion (differs from float ssim_vulkan which uses picture_copy). */
    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *cmp_in;

    VmafVulkanBuffer *h_ref_mu;
    VmafVulkanBuffer *h_cmp_mu;
    VmafVulkanBuffer *h_ref_sq;
    VmafVulkanBuffer *h_cmp_sq;
    VmafVulkanBuffer *h_refcmp;
    VmafVulkanBuffer *partials;

    /* Bytes per input element for the memcpy upload. */
    size_t bytes_per_px;

    VmafDictionary *feature_name_dict;
} IntegerSsimVulkanState;

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 "
                "with -EINVAL.",
        .offset = offsetof(IntegerSsimVulkanState, scale_override),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {0},
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t w_horiz;
    uint32_t h_horiz;
    uint32_t w_final;
    uint32_t h_final;
    uint32_t num_workgroups_x;
    float c1;
    float c2;
} IntSsimPushConsts;

/* Spec-constant layout (matches integer_ssim.comp):
 *   0 WIDTH, 1 HEIGHT, 2 PASS, 3 BPC, 4 SUBGROUP_SIZE */
struct IntSsimSpecData {
    int32_t width;
    int32_t height;
    int32_t pass;
    int32_t bpc;
    int32_t subgroup_size;
};

static void issim_fill_spec(struct IntSsimSpecData *sd, VkSpecializationMapEntry *entries,
                            VkSpecializationInfo *info, const IntegerSsimVulkanState *s,
                            int pass_id)
{
    sd->width = (int32_t)s->width;
    sd->height = (int32_t)s->height;
    sd->pass = pass_id;
    sd->bpc = (int32_t)s->bpc;
    sd->subgroup_size = 32;

    entries[0] = (VkSpecializationMapEntry){.constantID = 0,
                                            .offset = offsetof(struct IntSsimSpecData, width),
                                            .size = sizeof(int32_t)};
    entries[1] = (VkSpecializationMapEntry){.constantID = 1,
                                            .offset = offsetof(struct IntSsimSpecData, height),
                                            .size = sizeof(int32_t)};
    entries[2] = (VkSpecializationMapEntry){
        .constantID = 2, .offset = offsetof(struct IntSsimSpecData, pass), .size = sizeof(int32_t)};
    entries[3] = (VkSpecializationMapEntry){
        .constantID = 3, .offset = offsetof(struct IntSsimSpecData, bpc), .size = sizeof(int32_t)};
    entries[4] =
        (VkSpecializationMapEntry){.constantID = 4,
                                   .offset = offsetof(struct IntSsimSpecData, subgroup_size),
                                   .size = sizeof(int32_t)};

    *info = (VkSpecializationInfo){
        .mapEntryCount = 5,
        .pMapEntries = entries,
        .dataSize = sizeof(*sd),
        .pData = sd,
    };
}

static int build_pipeline_for_pass(IntegerSsimVulkanState *s, int pass_id, VkPipeline *out_pipeline)
{
    struct IntSsimSpecData sd = {0};
    VkSpecializationMapEntry entries[5];
    VkSpecializationInfo info = {0};
    issim_fill_spec(&sd, entries, &info, s, pass_id);

    const VkComputePipelineCreateInfo cpci = {
        .stage = {.pName = "main", .pSpecializationInfo = &info},
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out_pipeline);
}

static int create_pipeline(IntegerSsimVulkanState *s)
{
    struct IntSsimSpecData sd = {0};
    VkSpecializationMapEntry entries[5];
    VkSpecializationInfo info = {0};
    issim_fill_spec(&sd, entries, &info, s, /*pass_id=*/0);

    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = (uint32_t)ISSIM_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(IntSsimPushConsts),
        .spv_bytes = integer_ssim_spv,
        .spv_size = integer_ssim_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &info,
                    },
            },
        .max_descriptor_sets = 4U,
    };
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;

    return build_pipeline_for_pass(s, /*pass_id=*/1, &s->pipeline_vert);
}

static int alloc_buffers(IntegerSsimVulkanState *s)
{
    /* Raw integer input: width × height × bytes_per_px. */
    const size_t input_bytes = (size_t)s->width * s->height * s->bytes_per_px;
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t partial_bytes = (size_t)s->wg_count * sizeof(float);

    int err = 0;
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, input_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->cmp_in, input_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_mu, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_mu, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_sq, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_sq, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_refcmp, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->partials, partial_bytes);
    return err ? -ENOMEM : 0;
}

static int round_to_int_issim(float x)
{
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}

static int min_int_issim(int a, int b)
{
    return a < b ? a : b;
}

static int compute_scale(unsigned w, unsigned h, int override)
{
    if (override > 0)
        return override;
    int scaled = round_to_int_issim((float)min_int_issim((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

static int write_descriptor_set(IntegerSsimVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[ISSIM_NUM_BINDINGS] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->cmp_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_ref_mu),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_cmp_mu),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_ref_sq),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_cmp_sq),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_refcmp),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[ISSIM_NUM_BINDINGS];
    for (int i = 0; i < ISSIM_NUM_BINDINGS; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, ISSIM_NUM_BINDINGS, writes, 0, NULL);
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    IntegerSsimVulkanState *s = fex->priv;

    int scale = compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_ssim_vulkan: v1 supports scale=1 only "
                 "(auto-detected scale=%d at %ux%u). "
                 "Pin --feature integer_ssim_vulkan:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < ISSIM_K || h < ISSIM_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_ssim_vulkan: input %ux%u smaller than 11x11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (ISSIM_K - 1);
    s->h_horiz = h;
    s->w_final = w - (ISSIM_K - 1);
    s->h_final = h - (ISSIM_K - 1);
    s->wg_count_x = (s->w_final + ISSIM_WG_X - 1) / ISSIM_WG_X;
    s->wg_count_y = (s->h_final + ISSIM_WG_Y - 1) / ISSIM_WG_Y;
    s->wg_count = s->wg_count_x * s->wg_count_y;
    s->bytes_per_px = (bpc <= 8) ? 1U : 2U;

    /* SSIM constants using L=255 for all bit-depths because the shader
     * normalises samples into [0, 255] (matches integer_ssim_cuda.c). */
    const float L = 255.0f;
    const float K1 = 0.01f;
    const float K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "integer_ssim_vulkan: cannot create Vulkan context (%d)\n", err);
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

    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   /*count=*/1, &s->pre_set);
    if (err)
        return err;
    (void)write_descriptor_set(s, s->pre_set);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

/* Upload raw integer luma samples (uint8 or uint16) row-by-row,
 * removing any stride padding.  The shader indexes the SSBO as a flat
 * array with stride = width * bytes_per_px.  This mirrors
 * psnr_vulkan.c::upload_plane (no float conversion). */
static int upload_pic(IntegerSsimVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(dst_buf);
    if (!dst)
        return -EIO;
    const uint8_t *src = (const uint8_t *)pic->data[0];
    const size_t src_stride = (size_t)pic->stride[0];
    const size_t dst_stride = (size_t)s->width * s->bytes_per_px;
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    IntegerSsimVulkanState *s = fex->priv;
    int err = 0;

    err = upload_pic(s, s->ref_in, ref_pic);
    if (err)
        return err;
    err = upload_pic(s, s->cmp_in, dist_pic);
    if (err)
        return err;

    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    IntSsimPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .w_horiz = s->w_horiz,
        .h_horiz = s->h_horiz,
        .w_final = s->w_final,
        .h_final = s->h_final,
        .num_workgroups_x = s->wg_count_x,
        .c1 = s->c1,
        .c2 = s->c2,
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                            &s->pre_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    /* Pass 0: horizontal — grid over (W - 10) × H. */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    const uint32_t gx_h = (s->w_horiz + ISSIM_WG_X - 1) / ISSIM_WG_X;
    const uint32_t gy_h = (s->h_horiz + ISSIM_WG_Y - 1) / ISSIM_WG_Y;
    vkCmdDispatch(cmd, gx_h, gy_h, 1);

    /* Storage-buffer barrier between passes. */
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    /* Pass 1: vertical + SSIM combine — grid over (W - 10) × (H - 10). */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_vert);
    vkCmdDispatch(cmd, s->wg_count_x, s->wg_count_y, 1);

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    {
        int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->partials);
        if (err_inv) {
            vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
            return err_inv;
        }
        const float *partials = vmaf_vulkan_buffer_host(s->partials);
        double total = 0.0;
        for (unsigned i = 0; i < s->wg_count; i++)
            total += (double)partials[i];
        const double n_pixels = (double)s->w_final * (double)s->h_final;
        const double score = total / n_pixels;

        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "ssim", score, index);
    }

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    IntegerSsimVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;

    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

    if (s->pipeline_vert != VK_NULL_HANDLE)
        vkDestroyPipeline(dev, s->pipeline_vert, NULL);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_in);
    if (s->cmp_in)
        vmaf_vulkan_buffer_free(s->ctx, s->cmp_in);
    if (s->h_ref_mu)
        vmaf_vulkan_buffer_free(s->ctx, s->h_ref_mu);
    if (s->h_cmp_mu)
        vmaf_vulkan_buffer_free(s->ctx, s->h_cmp_mu);
    if (s->h_ref_sq)
        vmaf_vulkan_buffer_free(s->ctx, s->h_ref_sq);
    if (s->h_cmp_sq)
        vmaf_vulkan_buffer_free(s->ctx, s->h_cmp_sq);
    if (s->h_refcmp)
        vmaf_vulkan_buffer_free(s->ctx, s->h_refcmp);
    if (s->partials)
        vmaf_vulkan_buffer_free(s->ctx, s->partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        (void)vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"ssim", NULL};

VmafFeatureExtractor vmaf_fex_integer_ssim_vulkan = {
    .name = "integer_ssim_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(IntegerSsimVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
