/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature kernel on the Vulkan backend (T7-23 /
 *  ADR-0188 / ADR-0189, GPU long-tail batch 2 part 1a). Vulkan
 *  twin of the active CPU `float_ssim` extractor in
 *  libvmaf/src/feature/float_ssim.c.
 *
 *  Two-dispatch design (see ADR-0189):
 *    1. main_horiz: separable horizontal 11-tap Gaussian over
 *       ref / cmp / ref² / cmp² / ref·cmp → 5 intermediate float
 *       buffers sized (W - 10) × H.
 *    2. main_vert_combine: vertical 11-tap on intermediates +
 *       per-pixel SSIM combine + per-WG float partial sums.
 *
 *  Host: divides the partial-sum total (in `double`) by
 *  (W - 10) × (H - 10) and emits `float_ssim`. Mirrors the CPU's
 *  averaging window (per ssim_compute_stats / iqa_ssim).
 *
 *  v1 does NOT support the CPU's auto-decimation path
 *  (scale > 1 → -EINVAL at init). The cross-backend gate
 *  fixture (576×324) auto-resolves to scale=1; production
 *  1080p use needs scale=1 pinned via
 *  `--feature float_ssim_vulkan:scale=1` (or smaller input).
 *  GPU-side decimation is a v2 follow-up.
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

#include "ssim_spv.h" /* generated SPIR-V byte array */
#include "../picture_copy.h"

#define SSIM_WG_X 16
#define SSIM_WG_Y 8
#define SSIM_K 11
#define SSIM_NUM_BINDINGS                                                                          \
    8 /* ref, cmp, h_ref_mu, h_cmp_mu, h_ref_sq, h_cmp_sq, h_refcmp, partials */

typedef struct {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    int scale_override;

    /* Output dims after the convolve "valid" reduction. */
    unsigned w_horiz; /* W - 10 */
    unsigned h_horiz; /* H */
    unsigned w_final; /* W - 10 */
    unsigned h_final; /* H - 10 */
    unsigned wg_count_x;
    unsigned wg_count_y;
    unsigned wg_count;

    /* SSIM constants — derived from L=255 (matches CPU). */
    float c1;
    float c2;

    /* Vulkan context handle (borrowed from imported state when
     * present, else lazy-created). */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle, ADR-0221).
     * `pl` carries the shared layout / shader / DSL / pool plus the
     * horizontal-pass pipeline (pass=0). `pipeline_vert` is a sibling
     * pipeline created via `vmaf_vulkan_kernel_pipeline_add_variant()`
     * — same layout + shader + DSL + pool, different spec-constant
     * (pass=1). */
    VmafVulkanKernelPipeline pl;
    VkPipeline pipeline_vert;

    /* Input float ref + cmp (host-mapped). */
    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *cmp_in;

    /* 5 intermediate buffers (W - 10) × H. */
    VmafVulkanBuffer *h_ref_mu;
    VmafVulkanBuffer *h_cmp_mu;
    VmafVulkanBuffer *h_ref_sq;
    VmafVulkanBuffer *h_cmp_sq;
    VmafVulkanBuffer *h_refcmp;

    /* Per-WG float partials. */
    VmafVulkanBuffer *partials;

    /* CPU-side staging for picture_copy (matches float_ssim.c
     * which also picture_copy()s into a float plane before
     * the SIMD compute). */
    size_t float_stride;

    VmafDictionary *feature_name_dict;
} SsimVulkanState;

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(SsimVulkanState, scale_override),
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
} SsimPushConsts;

/* Spec-constant payload shared between the horizontal (pass=0) and
 * vertical (pass=1) pipelines. Only spec-constant 2 (pass) varies. */
struct SsimSpecData {
    int32_t width;
    int32_t height;
    int32_t pass;
    int32_t subgroup_size;
};

static void ssim_fill_spec(struct SsimSpecData *spec_data, VkSpecializationMapEntry *entries,
                           VkSpecializationInfo *info, const SsimVulkanState *s, int pass_id)
{
    spec_data->width = (int32_t)s->width;
    spec_data->height = (int32_t)s->height;
    spec_data->pass = pass_id;
    spec_data->subgroup_size = 32;

    entries[0] = (VkSpecializationMapEntry){
        .constantID = 0, .offset = offsetof(struct SsimSpecData, width), .size = sizeof(int32_t)};
    entries[1] = (VkSpecializationMapEntry){
        .constantID = 1, .offset = offsetof(struct SsimSpecData, height), .size = sizeof(int32_t)};
    entries[2] = (VkSpecializationMapEntry){
        .constantID = 2, .offset = offsetof(struct SsimSpecData, pass), .size = sizeof(int32_t)};
    entries[3] = (VkSpecializationMapEntry){.constantID = 3,
                                            .offset = offsetof(struct SsimSpecData, subgroup_size),
                                            .size = sizeof(int32_t)};

    *info = (VkSpecializationInfo){
        .mapEntryCount = 4,
        .pMapEntries = entries,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_pipeline_for_pass(SsimVulkanState *s, int pass_id, VkPipeline *out_pipeline)
{
    struct SsimSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[4];
    VkSpecializationInfo spec_info = {0};
    ssim_fill_spec(&spec_data, spec_entries, &spec_info, s, pass_id);

    const VkComputePipelineCreateInfo cpci = {
        .stage = {.pName = "main", .pSpecializationInfo = &spec_info},
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out_pipeline);
}

static int create_pipeline(SsimVulkanState *s)
{
    /* Pass 0 (horizontal) is the base pipeline — the template owns
     * the shared layout / shader / DSL / pool plus the pass=0
     * pipeline. */
    struct SsimSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[4];
    VkSpecializationInfo spec_info = {0};
    ssim_fill_spec(&spec_data, spec_entries, &spec_info, s, /*pass_id=*/0);

    /* `vulkan/kernel_template.h` (ADR-0221) owns the descriptor-set
     * layout (SSIM_NUM_BINDINGS = 8 SSBO bindings), pipeline layout,
     * shader module, compute pipeline (pass=0), and descriptor pool
     * sizing (4 sets × N buffers). The vertical-pass pipeline is
     * created next as a sibling variant via _add_variant(). */
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = (uint32_t)SSIM_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(SsimPushConsts),
        .spv_bytes = ssim_spv,
        .spv_size = ssim_spv_size,
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
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;

    /* Pass 1 (vertical) — same layout/shader/DSL/pool, different
     * spec-constant. */
    return build_pipeline_for_pass(s, /*pass_id=*/1, &s->pipeline_vert);
}

static int alloc_buffers(SsimVulkanState *s)
{
    const size_t input_bytes = (size_t)s->width * s->height * sizeof(float);
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t partials_bytes = (size_t)s->wg_count * sizeof(float);

    int err = 0;
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, input_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->cmp_in, input_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_mu, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_mu, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_sq, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_sq, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_refcmp, horiz_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->partials, partials_bytes);
    return err ? -ENOMEM : 0;
}

static int round_to_int(float x)
{
    /* Match _round() in libvmaf/src/feature/iqa/math_utils.h. */
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}

static int min_int(int a, int b)
{
    return a < b ? a : b;
}

static int compute_scale(unsigned w, unsigned h, int override)
{
    if (override > 0)
        return override;
    int scaled = round_to_int((float)min_int((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    SsimVulkanState *s = fex->priv;

    /* v1 supports scale=1 only — auto-resolve and reject if larger. */
    int scale = compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_vulkan: v1 supports scale=1 only (auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_vulkan:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }

    if (w < SSIM_K || h < SSIM_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_vulkan: input %ux%u smaller than 11×11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (SSIM_K - 1);
    s->h_horiz = h;
    s->w_final = w - (SSIM_K - 1);
    s->h_final = h - (SSIM_K - 1);
    s->wg_count_x = (s->w_final + SSIM_WG_X - 1) / SSIM_WG_X;
    s->wg_count_y = (s->h_final + SSIM_WG_Y - 1) / SSIM_WG_Y;
    s->wg_count = s->wg_count_x * s->wg_count_y;
    s->float_stride = (size_t)w * sizeof(float);

    /* SSIM constants. picture_copy normalises samples into the
     * [0, 255] range regardless of bpc — same as float_ssim.c —
     * so L=255 is correct for every bit-depth. */
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
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "ssim_vulkan: cannot create Vulkan context (%d)\n", err);
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

static int upload_pic(SsimVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic)
{
    /* picture_copy normalises uint sample → float in [0, 255]
     * (matches float_ssim.c::extract). The destination is
     * tightly packed (no padding) at width*sizeof(float). */
    float *dst = vmaf_vulkan_buffer_host(dst_buf);
    if (!dst)
        return -EIO;
    picture_copy(dst, (ptrdiff_t)s->float_stride, pic, /*offset=*/0, pic->bpc, /*channel=*/0);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int write_descriptor_set(SsimVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[SSIM_NUM_BINDINGS] = {
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
    VkWriteDescriptorSet writes[SSIM_NUM_BINDINGS];
    for (int i = 0; i < SSIM_NUM_BINDINGS; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, SSIM_NUM_BINDINGS, writes, 0, NULL);
    return 0;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    SsimVulkanState *s = fex->priv;
    int err = 0;

    err = upload_pic(s, s->ref_in, ref_pic);
    if (err)
        return err;
    err = upload_pic(s, s->cmp_in, dist_pic);
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

    SsimPushConsts pc = {
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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1, &set,
                            0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    /* Pass 0: horizontal — grid sized over the (W-10) × H output. */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    const uint32_t gx_h = (s->w_horiz + SSIM_WG_X - 1) / SSIM_WG_X;
    const uint32_t gy_h = (s->h_horiz + SSIM_WG_Y - 1) / SSIM_WG_Y;
    vkCmdDispatch(cmd, gx_h, gy_h, 1);

    /* Storage-buffer barrier between the two passes. */
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    /* Pass 1: vertical + SSIM combine — grid sized over (W-10) × (H-10). */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline_vert);
    vkCmdDispatch(cmd, s->wg_count_x, s->wg_count_y, 1);

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

    /* Per-WG float partials → host double sum → mean SSIM over
     * (W - 10) × (H - 10) pixels (matches CPU's iqa_ssim
     * normalisation per line 371 of ssim_tools.c). */
    const float *partials = vmaf_vulkan_buffer_host(s->partials);
    double total = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += (double)partials[i];
    const double n_pixels = (double)s->w_final * (double)s->h_final;
    const double score = total / n_pixels;

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "float_ssim", score, index);

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
    SsimVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    /* Destroy the sibling variant first; the base pipeline and the
     * shared layout/shader/DSL/pool are owned by the template. */
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
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"float_ssim", NULL};

VmafFeatureExtractor vmaf_fex_float_ssim_vulkan = {
    .name = "float_ssim_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(SsimVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 2 dispatches/frame (horizontal pass + vertical-with-combine).
     * Per-pixel ALU is moderate (5×11 mac in horiz + 5×11 mac in
     * vert + a handful of ops in the SSIM combine). 1080p area
     * threshold matches motion's profile. */
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
