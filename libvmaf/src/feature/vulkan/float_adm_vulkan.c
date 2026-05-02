/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_adm feature kernel on the Vulkan backend (T7-23 / batch 3
 *  part 6 — ADR-0192 / ADR-0199). Sixth Group B float twin and the
 *  last metric in the GPU long-tail roadmap.
 *
 *  Float twin of integer_adm_vulkan (ADR-0178). Algorithm follows
 *  CPU compute_adm in libvmaf/src/feature/adm.c, with float (`_s`
 *  suffix) primitives from libvmaf/src/feature/adm_tools.c:
 *      adm_dwt2_s, adm_decouple_s, adm_csf_s, adm_csf_den_scale_s,
 *      adm_cm_s, adm_sum_cube_s.
 *
 *  Per-frame pipeline (per scale, 4 scales):
 *    Stage 0 — DWT vertical (ref+dis fused, dim_z=2)
 *    Stage 1 — DWT horizontal (ref+dis fused, dim_z=2) → 4 bands
 *    Stage 2 — Decouple + CSF (writes csf_a + csf_f)
 *    Stage 3 — CSF denominator + Contrast measure fused (1D dispatch
 *              over 3 bands × num_active_rows; per-WG float partials).
 *
 *  Stage 3 produces six float partials per workgroup
 *  (csf_h/v/d + cm_h/v/d). The host CPU promotes to double when
 *  reducing across WGs, then runs the same scoring helpers as the CPU
 *  reference (powf((double)accum, 1/3) + powf(area/32, 1/3)).
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/adm_vulkan.c —
 *  same lazy-or-borrow context, owns_ctx flag,
 *  VkSpecializationInfo-driven pipelines (one per (scale, stage)
 *  pair = 16 pipelines total).
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

#include "float_adm_spv.h" /* generated SPIR-V byte array */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FADM_NUM_SCALES 4
#define FADM_NUM_BANDS 3
#define FADM_NUM_STAGES 4
#define FADM_BORDER_FACTOR 0.1
#define FADM_ACCUM_SLOTS_PER_WG 6 /* csf_h,csf_v,csf_d,cm_h,cm_v,cm_d */

#define FADM_WG_X 16
#define FADM_WG_Y 16

/* DB2/CDF-9-7 wavelet noise model (matches dwt_7_9_YCbCr_threshold[0]
 * Y-plane row in adm_tools.h). */
typedef struct {
    float a, k, f0;
    float g[4];
} FadmDwtModelParams;

static const FadmDwtModelParams fadm_dwt_model_Y = {
    0.495f, 0.466f, 0.401f, {1.501f, 1.0f, 0.534f, 1.0f}};

static const float fadm_dwt_basis_amp[6][4] = {
    {0.62171f, 0.67234f, 0.72709f, 0.67234f},     {0.34537f, 0.41317f, 0.49428f, 0.41317f},
    {0.18004f, 0.22727f, 0.28688f, 0.22727f},     {0.091401f, 0.11792f, 0.15214f, 0.11792f},
    {0.045943f, 0.059758f, 0.077727f, 0.059758f}, {0.023013f, 0.030018f, 0.039156f, 0.030018f},
};

static float fadm_dwt_quant_step_host(int lambda, int theta, double view_dist, int display_h)
{
    /* Bit-for-bit replica of dwt_quant_step in adm_tools.h. The CPU
     * version is `static FORCE_INLINE float dwt_quant_step(...)` and
     * uses double-precision `pow` / `log10` with float-cast at the
     * final return. We do the same here to match rfactors exactly. */
    float r = (float)(view_dist * (double)display_h * M_PI / 180.0);
    float temp = (float)log10(pow(2.0, (double)(lambda + 1)) * (double)fadm_dwt_model_Y.f0 *
                              (double)fadm_dwt_model_Y.g[theta] / (double)r);
    float Q = (float)(2.0 * (double)fadm_dwt_model_Y.a *
                      pow(10.0, (double)fadm_dwt_model_Y.k * (double)temp * (double)temp) /
                      (double)fadm_dwt_basis_amp[lambda][theta]);
    return Q;
}

/* ------------------------------------------------------------------ */
/* Per-extractor state.                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Options. */
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    int adm_csf_mode; /* parsed but only mode 0 (default) supported */

    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned buf_stride; /* aligned float elements per band row */

    /* rfactors (host-side; used as push-constant floats). */
    float rfactor[12]; /* 4 scales × 3 bands (h, v, d) */

    /* Vulkan context. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipelines: one per (stage, scale) = 16. */
    /* `pl` carries the shared layout / shader / DSL / pool plus the
     * (stage=0, scale=0) base pipeline. The other 15 entries in
     * `pipelines[stage][scale]` are sibling pipelines created via
     * `vmaf_vulkan_kernel_pipeline_add_variant()`.
     * `pipelines[0][0]` aliases `pl.pipeline` so the dispatch path
     * keeps a clean 2-D lookup. (ADR-0221.) */
    VmafVulkanKernelPipeline pl;
    VkPipeline pipelines[FADM_NUM_STAGES][FADM_NUM_SCALES];

    /* GPU buffers. */
    VmafVulkanBuffer *src_ref; /* scale-0 host-uploaded source plane */
    VmafVulkanBuffer *src_dis;
    VmafVulkanBuffer *dwt_tmp_ref;
    VmafVulkanBuffer *dwt_tmp_dis;
    VmafVulkanBuffer *ref_band; /* 4 bands packed contiguous */
    VmafVulkanBuffer *dis_band;
    VmafVulkanBuffer *csf_a; /* 3 bands packed contiguous */
    VmafVulkanBuffer *csf_f; /* 3 bands packed contiguous */

    /* Per-scale accumulators (float, 6 slots per WG). */
    VmafVulkanBuffer *accum[FADM_NUM_SCALES];
    unsigned wg_count[FADM_NUM_SCALES];

    /* Per-scale dimensions cached. */
    unsigned scale_w[FADM_NUM_SCALES];
    unsigned scale_h[FADM_NUM_SCALES];
    unsigned scale_half_w[FADM_NUM_SCALES];
    unsigned scale_half_h[FADM_NUM_SCALES];

    VmafDictionary *feature_name_dict;
} FloatAdmVulkanState;

/* ------------------------------------------------------------------ */
/* Options — same shape as the CPU float_adm extractor.                */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {
    {.name = "debug",
     .help = "debug mode: enable additional output",
     .offset = offsetof(FloatAdmVulkanState, debug),
     .type = VMAF_OPT_TYPE_BOOL,
     .default_val.b = false},
    {.name = "adm_enhn_gain_limit",
     .alias = "egl",
     .help = "enhancement gain imposed on adm, must be >= 1.0",
     .offset = offsetof(FloatAdmVulkanState, adm_enhn_gain_limit),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 100.0,
     .min = 1.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_norm_view_dist",
     .alias = "nvd",
     .help = "normalized viewing distance",
     .offset = offsetof(FloatAdmVulkanState, adm_norm_view_dist),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 3.0,
     .min = 0.75,
     .max = 24.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_ref_display_height",
     .alias = "rdf",
     .help = "reference display height in pixels",
     .offset = offsetof(FloatAdmVulkanState, adm_ref_display_height),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 1080,
     .min = 1,
     .max = 4320,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_mode",
     .alias = "csf",
     .help = "contrast sensitivity function (mode 0 only on Vulkan)",
     .offset = offsetof(FloatAdmVulkanState, adm_csf_mode),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 0,
     .min = 0,
     .max = 9,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {0}};

/* ------------------------------------------------------------------ */
/* Push constants — must mirror `Params` in float_adm.comp.            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t cur_w;
    uint32_t cur_h;
    uint32_t half_w;
    uint32_t half_h;
    uint32_t buf_stride;
    float scaler;
    float pixel_offset;
    float rfactor_h;
    float rfactor_v;
    float rfactor_d;
    float gain_limit;
    int32_t active_left;
    int32_t active_top;
    int32_t active_right;
    int32_t active_bottom;
    uint32_t num_workgroups_x;
} FloatAdmPushConsts;

/* ------------------------------------------------------------------ */
/* Pipeline / descriptor-set layout creation.                          */
/* ------------------------------------------------------------------ */

struct FloatAdmSpecData {
    int32_t width;
    int32_t height;
    int32_t bpc;
    int32_t scale;
    int32_t stage;
};

static void float_adm_fill_spec(struct FloatAdmSpecData *spec_data, VkSpecializationMapEntry map[5],
                                VkSpecializationInfo *spec_info, const FloatAdmVulkanState *s,
                                int stage, int scale)
{
    spec_data->width = (int32_t)s->width;
    spec_data->height = (int32_t)s->height;
    spec_data->bpc = (int32_t)s->bpc;
    spec_data->scale = scale;
    spec_data->stage = stage;
    map[0] = (VkSpecializationMapEntry){.constantID = 0, .offset = 0, .size = sizeof(int32_t)};
    map[1] = (VkSpecializationMapEntry){.constantID = 1, .offset = 4, .size = sizeof(int32_t)};
    map[2] = (VkSpecializationMapEntry){.constantID = 2, .offset = 8, .size = sizeof(int32_t)};
    map[3] = (VkSpecializationMapEntry){.constantID = 3, .offset = 12, .size = sizeof(int32_t)};
    map[4] = (VkSpecializationMapEntry){.constantID = 4, .offset = 16, .size = sizeof(int32_t)};
    *spec_info = (VkSpecializationInfo){
        .mapEntryCount = 5,
        .pMapEntries = map,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_pipeline_for(FloatAdmVulkanState *s, int stage, int scale,
                              VkPipeline *out_pipeline)
{
    struct FloatAdmSpecData spec_data = {0};
    VkSpecializationMapEntry map[5];
    VkSpecializationInfo spec_info = {0};
    float_adm_fill_spec(&spec_data, map, &spec_info, s, stage, scale);

    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out_pipeline);
}

static int create_pipelines(FloatAdmVulkanState *s)
{
    /* (stage=0, scale=0) is the base pipeline. The template owns
     * the shared layout / shader / DSL / pool plus this base.
     * 9 SSBOs per set; pool sized for 4 frames × 16 (stage,scale)
     * pairs = 64 sets. */
    struct FloatAdmSpecData spec_data = {0};
    VkSpecializationMapEntry map[5];
    VkSpecializationInfo spec_info = {0};
    float_adm_fill_spec(&spec_data, map, &spec_info, s, /*stage=*/0, /*scale=*/0);

    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 9U,
        .push_constant_size = (uint32_t)sizeof(FloatAdmPushConsts),
        .spv_bytes = float_adm_spv,
        .spv_size = float_adm_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = (uint32_t)(FADM_NUM_SCALES * FADM_NUM_STAGES * 4),
    };
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;
    s->pipelines[0][0] = s->pl.pipeline;

    /* The remaining 15 (stage, scale) variants — same layout,
     * shader, DSL, pool, different stage/scale spec-constants. */
    for (int stage = 0; stage < FADM_NUM_STAGES; stage++) {
        for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
            if (stage == 0 && scale == 0)
                continue;
            err = build_pipeline_for(s, stage, scale, &s->pipelines[stage][scale]);
            if (err)
                return err;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Buffer allocation.                                                  */
/* ------------------------------------------------------------------ */

static int alloc_buffers(FloatAdmVulkanState *s)
{
    unsigned w = s->width;
    unsigned h = s->height;
    unsigned half_w0 = (w + 1) / 2;
    unsigned half_h0 = (h + 1) / 2;
    /* Align to 4 floats so the std430 layout is friendly across drivers. */
    s->buf_stride = (half_w0 + 3u) & ~3u;

    /* Source planes (scale 0 only). */
    size_t bpp = (s->bpc <= 8) ? 1 : 2;
    size_t src_bytes = (size_t)w * h * bpp;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->src_ref, src_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->src_dis, src_bytes);
    if (err)
        return err;

    /* DWT scratch: max size is at scale 0 (cur_w*2 × half_h elements,
     * float each). Subsequent scales fit since each is half-sized. */
    size_t dwt_bytes = (size_t)w * 2 * half_h0 * sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dwt_tmp_ref, dwt_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dwt_tmp_dis, dwt_bytes);
    if (err)
        return err;

    /* Band buffers: 4 bands × buf_stride × half_h0 floats. */
    size_t band_bytes = (size_t)4 * s->buf_stride * half_h0 * sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_band, band_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_band, band_bytes);
    if (err)
        return err;

    /* csf_a + csf_f: 3 bands × buf_stride × half_h0 floats each. */
    size_t csf_bytes = (size_t)3 * s->buf_stride * half_h0 * sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->csf_a, csf_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->csf_f, csf_bytes);
    if (err)
        return err;

    /* Per-scale accumulators (sized to 3 × num_active_rows × 6 floats). */
    unsigned cw = w, ch = h;
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        unsigned hw = (cw + 1) / 2;
        unsigned hh = (ch + 1) / 2;
        s->scale_w[scale] = cw;
        s->scale_h[scale] = ch;
        s->scale_half_w[scale] = hw;
        s->scale_half_h[scale] = hh;

        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (top < 0)
            top = 0;
        int bottom = (int)hh - top;
        unsigned num_rows = (bottom > top) ? (unsigned)(bottom - top) : 1u;
        unsigned wg_count = 3u * num_rows;
        s->wg_count[scale] = wg_count;
        size_t accum_bytes = (size_t)wg_count * FADM_ACCUM_SLOTS_PER_WG * sizeof(float);
        if (accum_bytes == 0)
            accum_bytes = sizeof(float);
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->accum[scale], accum_bytes);
        if (err)
            return err;

        cw = hw;
        ch = hh;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* init().                                                             */
/* ------------------------------------------------------------------ */

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    FloatAdmVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    if (s->adm_csf_mode != 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_adm_vulkan: only adm_csf_mode=0 is supported in v1\n");
        return -EINVAL;
    }

    /* rfactors per scale: 1 / dwt_quant_step at (lambda=scale, theta=1)
     * for h+v and (theta=2) for d — matches the CPU adm_csf_s /
     * adm_csf_den_scale_s / adm_cm_s in adm_tools.c. */
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        float f1 =
            fadm_dwt_quant_step_host(scale, 1, s->adm_norm_view_dist, s->adm_ref_display_height);
        float f2 =
            fadm_dwt_quant_step_host(scale, 2, s->adm_norm_view_dist, s->adm_ref_display_height);
        s->rfactor[scale * 3 + 0] = 1.0f / f1;
        s->rfactor[scale * 3 + 1] = 1.0f / f1;
        s->rfactor[scale * 3 + 2] = 1.0f / f2;
    }

    /* Borrow framework's imported context, fall back to lazy create. */
    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "float_adm_vulkan: cannot create Vulkan context (%d)\n",
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

/* ------------------------------------------------------------------ */
/* Per-frame helpers.                                                  */
/* ------------------------------------------------------------------ */

static int upload_plane(FloatAdmVulkanState *s, VmafPicture *pic, VmafVulkanBuffer *buf)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static void write_descriptor_set(FloatAdmVulkanState *s, VkDescriptorSet set, int scale)
{
    VkDescriptorBufferInfo dbi[9] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->src_ref),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dwt_tmp_ref),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dwt_tmp_dis),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_band),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_band),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->csf_a),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->csf_f),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->accum[scale]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->src_dis),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[9];
    for (int i = 0; i < 9; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 9, writes, 0, NULL);
}

static void issue_pipeline_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* ------------------------------------------------------------------ */
/* Score reduction (mirrors CPU's adm_csf_den_scale_s + adm_cm_s).     */
/* ------------------------------------------------------------------ */

static int reduce_and_emit(FloatAdmVulkanState *s, unsigned index, VmafFeatureCollector *fc)
{
    /* Per-scale double accumulation across WGs. The 6 slots per WG are
     * laid out [csf_h, csf_v, csf_d, cm_h, cm_v, cm_d]. Only band b's
     * (csf,cm) slots are non-zero in WG b — the others stay zero
     * because the accum buffer is host-cleared and band-specific
     * threads write only their own slots. So summing all 6 across
     * all WGs in scale s yields the totals per band. */
    double cm_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {0};
    double csf_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {0};

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const float *slots = vmaf_vulkan_buffer_host(s->accum[scale]);
        unsigned wg_count = s->wg_count[scale];
        for (unsigned wg = 0; wg < wg_count; wg++) {
            const float *p = slots + (size_t)wg * FADM_ACCUM_SLOTS_PER_WG;
            for (int b = 0; b < FADM_NUM_BANDS; b++) {
                csf_totals[scale][b] += (double)p[b];
                cm_totals[scale][b] += (double)p[3 + b];
            }
        }
    }

    double score_num = 0.0;
    double score_den = 0.0;
    double scores[8]; /* num_scale0, den_scale0, ..., num_scale3, den_scale3 */

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        int hw = (int)s->scale_half_w[scale];
        int hh = (int)s->scale_half_h[scale];
        int left = (int)((double)hw * FADM_BORDER_FACTOR - 0.5);
        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        int right = hw - left;
        int bottom = hh - top;
        float area_cbrt = powf((float)((bottom - top) * (right - left)) / 32.0f, 1.0f / 3.0f);

        /* num_scale = sum over 3 bands of (cbrt(cm_total) + area_cbrt). */
        float num_scale = 0.0f;
        for (int b = 0; b < FADM_NUM_BANDS; b++) {
            float ac = (float)cm_totals[scale][b];
            num_scale += powf(ac, 1.0f / 3.0f) + area_cbrt;
        }
        float den_scale = 0.0f;
        for (int b = 0; b < FADM_NUM_BANDS; b++) {
            float ac = (float)csf_totals[scale][b];
            den_scale += powf(ac, 1.0f / 3.0f) + area_cbrt;
        }
        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
        score_num += num_scale;
        score_den += den_scale;
    }

    /* numden_limit per ADM_OPT_SINGLE_PRECISION (matches adm.c line 88). */
    int w = (int)s->scale_w[0];
    int h = (int)s->scale_h[0];
    double numden_limit = 1e-2 * (double)(w * h) / (1920.0 * 1080.0);
    if (score_num < numden_limit)
        score_num = 0.0;
    if (score_den < numden_limit)
        score_den = 0.0;
    double score = (score_den == 0.0) ? 1.0 : score_num / score_den;

    int err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                      "VMAF_feature_adm2_score", score, index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                      "VMAF_feature_adm_scale0_score",
                                                      scores[0] / scores[1], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                      "VMAF_feature_adm_scale1_score",
                                                      scores[2] / scores[3], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                      "VMAF_feature_adm_scale2_score",
                                                      scores[4] / scores[5], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                      "VMAF_feature_adm_scale3_score",
                                                      scores[6] / scores[7], index);

    if (s->debug && !err) {
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm", score, index);
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm_num", score_num,
                                                index);
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm_den", score_den,
                                                index);
        const char *names[8] = {"adm_num_scale0", "adm_den_scale0", "adm_num_scale1",
                                "adm_den_scale1", "adm_num_scale2", "adm_den_scale2",
                                "adm_num_scale3", "adm_den_scale3"};
        for (int i = 0; i < 8 && !err; i++) {
            err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, names[i],
                                                          scores[i], index);
        }
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* extract().                                                          */
/* ------------------------------------------------------------------ */

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    FloatAdmVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, ref_pic, s->src_ref);
    if (err)
        return err;
    err = upload_plane(s, dist_pic, s->src_dis);
    if (err)
        return err;

    /* Zero per-scale accumulator buffers (host-mapped). */
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        size_t bytes = (size_t)s->wg_count[scale] * FADM_ACCUM_SLOTS_PER_WG * sizeof(float);
        memset(vmaf_vulkan_buffer_host(s->accum[scale]), 0, bytes);
        err = vmaf_vulkan_buffer_flush(s->ctx, s->accum[scale]);
        if (err)
            return err;
    }

    /* One descriptor set per scale (each binds the scale's accum). */
    VkDescriptorSet sets[FADM_NUM_SCALES] = {VK_NULL_HANDLE};
    VkDescriptorSetLayout layouts[FADM_NUM_SCALES];
    for (int i = 0; i < FADM_NUM_SCALES; i++)
        layouts[i] = s->pl.dsl;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->pl.desc_pool,
        .descriptorSetCount = FADM_NUM_SCALES,
        .pSetLayouts = layouts,
    };
    if (vkAllocateDescriptorSets(s->ctx->device, &dsai, sets) != VK_SUCCESS)
        return -ENOMEM;
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++)
        write_descriptor_set(s, sets[scale], scale);

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

    /* Picture-copy semantics for HBD. picture_copy in float_adm.c does
     * `(u16 / scaler) - 128`; for 8-bit it's `u8 - 128` directly. */
    float scaler = 1.0f;
    if (s->bpc == 10)
        scaler = 4.0f;
    else if (s->bpc == 12)
        scaler = 16.0f;
    else if (s->bpc == 16)
        scaler = 256.0f;

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        unsigned cw = s->scale_w[scale];
        unsigned ch = s->scale_h[scale];
        unsigned hw = s->scale_half_w[scale];
        unsigned hh = s->scale_half_h[scale];

        int left = (int)((double)hw * FADM_BORDER_FACTOR - 0.5);
        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        int right = (int)hw - left;
        int bottom = (int)hh - top;
        int active_h = bottom - top;

        FloatAdmPushConsts pc = {
            .cur_w = cw,
            .cur_h = ch,
            .half_w = hw,
            .half_h = hh,
            .buf_stride = s->buf_stride,
            .scaler = scaler,
            .pixel_offset = -128.0f,
            .rfactor_h = s->rfactor[scale * 3 + 0],
            .rfactor_v = s->rfactor[scale * 3 + 1],
            .rfactor_d = s->rfactor[scale * 3 + 2],
            .gain_limit = (float)s->adm_enhn_gain_limit,
            .active_left = left,
            .active_top = top,
            .active_right = right,
            .active_bottom = bottom,
            .num_workgroups_x = 1u,
        };

        /* Stage 0 — DWT vertical (z=2 fused ref+dis). */
        {
            uint32_t gx = (cw + FADM_WG_X - 1u) / FADM_WG_X;
            uint32_t gy = (hh + FADM_WG_Y - 1u) / FADM_WG_Y;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[0][scale]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0,
                                    1, &sets[scale], 0, NULL);
            vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pc), &pc);
            vkCmdDispatch(cmd, gx, gy, 2);
            issue_pipeline_barrier(cmd);
        }

        /* Stage 1 — DWT horizontal (z=2 fused ref+dis). */
        {
            uint32_t gx = (hw + FADM_WG_X - 1u) / FADM_WG_X;
            uint32_t gy = (hh + FADM_WG_Y - 1u) / FADM_WG_Y;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[1][scale]);
            vkCmdDispatch(cmd, gx, gy, 2);
            issue_pipeline_barrier(cmd);
        }

        /* Stage 2 — Decouple + CSF (writes csf_a + csf_f). */
        {
            uint32_t gx = (hw + FADM_WG_X - 1u) / FADM_WG_X;
            uint32_t gy = (hh + FADM_WG_Y - 1u) / FADM_WG_Y;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[2][scale]);
            vkCmdDispatch(cmd, gx, gy, 1);
            issue_pipeline_barrier(cmd);
        }

        /* Stage 3 — CSF denominator + CM fused. 1D dispatch over
         *   3 bands × num_active_rows. */
        {
            unsigned num_rows = (unsigned)(active_h > 0 ? active_h : 1);
            uint32_t gx = 3u * num_rows;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[3][scale]);
            vkCmdDispatch(cmd, gx, 1u, 1u);
            issue_pipeline_barrier(cmd);
        }
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

    err = reduce_and_emit(s, index, feature_collector);

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    if (sets[0] != VK_NULL_HANDLE)
        vkFreeDescriptorSets(s->ctx->device, s->pl.desc_pool, FADM_NUM_SCALES, sets);
    return err;
}

/* ------------------------------------------------------------------ */
/* close().                                                            */
/* ------------------------------------------------------------------ */

static int close_fex(VmafFeatureExtractor *fex)
{
    FloatAdmVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    /* Destroy the 15 sibling variants first; the (0, 0) base + the
     * shared layout / shader / DSL / pool are owned by the template.
     * (0, 0) is skipped because pipelines[0][0] aliases s->pl.pipeline. */
    for (int stage = 0; stage < FADM_NUM_STAGES; stage++) {
        for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
            if (stage == 0 && scale == 0)
                continue;
            if (s->pipelines[stage][scale] != VK_NULL_HANDLE)
                vkDestroyPipeline(dev, s->pipelines[stage][scale], NULL);
        }
    }
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->src_ref)
        vmaf_vulkan_buffer_free(s->ctx, s->src_ref);
    if (s->src_dis)
        vmaf_vulkan_buffer_free(s->ctx, s->src_dis);
    if (s->dwt_tmp_ref)
        vmaf_vulkan_buffer_free(s->ctx, s->dwt_tmp_ref);
    if (s->dwt_tmp_dis)
        vmaf_vulkan_buffer_free(s->ctx, s->dwt_tmp_dis);
    if (s->ref_band)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_band);
    if (s->dis_band)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_band);
    if (s->csf_a)
        vmaf_vulkan_buffer_free(s->ctx, s->csf_a);
    if (s->csf_f)
        vmaf_vulkan_buffer_free(s->ctx, s->csf_f);
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->accum[scale])
            vmaf_vulkan_buffer_free(s->ctx, s->accum[scale]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Provided features + registration. Matches CPU float_adm.            */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"VMAF_feature_adm2_score",
                                          "VMAF_feature_adm_scale0_score",
                                          "VMAF_feature_adm_scale1_score",
                                          "VMAF_feature_adm_scale2_score",
                                          "VMAF_feature_adm_scale3_score",
                                          "adm",
                                          "adm_num",
                                          "adm_den",
                                          "adm_num_scale0",
                                          "adm_den_scale0",
                                          "adm_num_scale1",
                                          "adm_den_scale1",
                                          "adm_num_scale2",
                                          "adm_den_scale2",
                                          "adm_num_scale3",
                                          "adm_den_scale3",
                                          NULL};

VmafFeatureExtractor vmaf_fex_float_adm_vulkan = {
    .name = "float_adm_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(FloatAdmVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
