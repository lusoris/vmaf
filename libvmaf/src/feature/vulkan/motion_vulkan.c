/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Motion feature kernel on the Vulkan backend (T5-1c-motion).
 *
 *  Replaces the dispatch-only stub from the T5-1 scaffold with a real
 *  `VmafFeatureExtractor` named "motion_vulkan" that emits the two
 *  `VMAF_integer_feature_motion{,2}_score` outputs identical to the
 *  CPU / SYCL / CUDA implementations on the Netflix golden gate.
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_motion.c):
 *    1. V->H separable 5-tap Gaussian blur of the reference plane
 *       (filter sum 65536, integer-only path with `>> bpc` then
 *       `>> 16` rounding) — implemented in shaders/motion.comp.
 *    2. SAD between current and previous blurred reference frames
 *       (per-WG int64 partial -> host-side scalar reduction).
 *    3. motion_score = SAD / 256.0 / (width * height)
 *       motion2_score = min(prev_motion_score, cur_motion_score),
 *       written at `index - 1` (delayed-by-one pattern from the CPU
 *       extractor — see CPU lines 510-560).
 *
 *  Scope note (T3-15(c) update — motion3_score, default 3-frame
 *  window): motion3 is now emitted on the Vulkan backend. The CPU
 *  extractor derives motion3 from motion2 via host-side scalar
 *  post-processing (`motion_blend` + clip-to-`motion_max_val` +
 *  optional moving-average) — none of this is on the GPU's critical
 *  path, so the GPU port is purely a host-side scalar add-on after
 *  the SAD reduction. See ADR-0219.
 *
 *  Still deferred: `motion_five_frame_window=true` mode (5-deep blur
 *  ring buffer + frame i-2↔i + i-2↔i-4 SAD pair). No shipped model
 *  uses it; init() rejects it with -ENOTSUP if a caller sets it.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/vif_vulkan.c (the
 *  canonical Vulkan-backend layout — same lazy-or-borrow context,
 *  owns_ctx flag, VkSpecializationInfo-driven pipelines, host-side
 *  reduction of per-WG int64 slots).
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../motion_blend_tools.h"

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"

#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"
#include "../../vulkan/kernel_template.h"

#include "motion_spv.h" /* generated SPIR-V byte array */

/* ------------------------------------------------------------------ */
/* Constants — must match motion.comp.                                 */
/* ------------------------------------------------------------------ */

#define MOTION_WG_X 32
#define MOTION_WG_Y 4

/* ------------------------------------------------------------------ */
/* Per-extractor state.                                                */
/* ------------------------------------------------------------------ */

/* Default upper clamp on motion / motion2 / motion3 — mirrors
 * DEFAULT_MOTION_MAX_VAL in libvmaf/src/feature/integer_motion.c. */
#define MOTION_VULKAN_DEFAULT_MAX_VAL (10000.0)

typedef struct {
    /* Options. */
    bool debug;
    bool motion_force_zero;
    bool motion_five_frame_window; /* rejected with -ENOTSUP — see init() */
    bool motion_moving_average;
    double motion_blend_factor;
    double motion_blend_offset;
    double motion_fps_weight;
    double motion_max_val;

    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Vulkan context handle. Borrow on imported state, lazy-create
     * otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle, ADR-0221).
     *
     * Note: prior versions stored `pipelines[2]` for "first frame" /
     * "subsequent frames" parity with the SYCL backend. Inspection
     * shows the COMPUTE_SAD flag is passed via *push constants*, not
     * spec-constants, so the two pipelines are functionally
     * identical. Collapsed to one. (T-GPU-DEDUP-7.) */
    VmafVulkanKernelPipeline pl;

    /* Source plane upload buffer (host-mapped). */
    VmafVulkanBuffer *ref_in;

    /* Ping-pong blurred-reference buffers. blur[cur_blur] receives
     * this frame's blurred plane; blur[1 - cur_blur] is "previous". */
    VmafVulkanBuffer *blur[2];
    int cur_blur;

    /* Per-workgroup int64 SAD partials. */
    VmafVulkanBuffer *sad_partials;
    unsigned wg_count;

    /* Frame state. */
    unsigned frame_index;
    double prev_motion_score;
    /* motion3 post-processing state — mirrors `previous_score` in
     * the CPU MotionState. Holds the last unaveraged blended
     * motion3 score so we can apply the moving-average rule on the
     * next frame. */
    double prev_motion3_blended;

    VmafDictionary *feature_name_dict;
} MotionVulkanState;

/* ------------------------------------------------------------------ */
/* Options table — mirrors libvmaf/src/feature/integer_motion.c.      */
/* The post-processing options drive motion3_score, derived on the    */
/* host from motion2 via `motion_blend(...)` (see extract()).         */
/* `motion_five_frame_window` is currently rejected — the kernel     */
/* still uses a 2-deep blur ring (T3-15(c) deferred sub-task).       */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionVulkanState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "force motion score to zero",
        .offset = offsetof(MotionVulkanState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionVulkanState, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionVulkanState, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 40.0,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionVulkanState, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .alias = "mmxv",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionVulkanState, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = MOTION_VULKAN_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_five_frame_window",
        .alias = "mffw",
        .help = "use five-frame temporal window (NOT YET SUPPORTED on Vulkan — T3-15(c) deferred)",
        .offset = offsetof(MotionVulkanState, motion_five_frame_window),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_moving_average",
        .alias = "mma",
        .help = "use moving average for motion3 scores after first frame",
        .offset = offsetof(MotionVulkanState, motion_moving_average),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

/* ------------------------------------------------------------------ */
/* Push constants — must mirror `Params` in motion.comp.              */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t compute_sad;
    uint32_t num_workgroups_x;
} MotionPushConsts;

/* ------------------------------------------------------------------ */
/* Helpers.                                                            */
/* ------------------------------------------------------------------ */

static inline void motion_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + MOTION_WG_X - 1u) / MOTION_WG_X;
    *gy = (h + MOTION_WG_Y - 1u) / MOTION_WG_Y;
}

/* ------------------------------------------------------------------ */
/* Pipeline / descriptor-set layout creation.                          */
/* ------------------------------------------------------------------ */

static int create_pipelines(MotionVulkanState *s)
{
    /* Spec-constants pin (width, height, bpc, subgroup_size) for the
     * shader's shared-memory sizing. COMPUTE_SAD is *not* a spec
     * constant — it's a runtime push-constant flag — so a single
     * pipeline is sufficient (collapsed from the prior `pipelines[2]`
     * "SYCL symmetry" layout per T-GPU-DEDUP-7). */
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

    /* `vulkan/kernel_template.h` (ADR-0221) owns the descriptor-set
     * layout (4 SSBO bindings: ref, prev_blur, blur, sad_partials),
     * pipeline layout, shader module, compute pipeline, and
     * descriptor pool sizing (4 sets × 4 buffers = 16 descriptors). */
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 4U,
        .push_constant_size = (uint32_t)sizeof(MotionPushConsts),
        .spv_bytes = motion_spv,
        .spv_size = motion_spv_size,
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

/* ------------------------------------------------------------------ */
/* Buffer allocation.                                                  */
/* ------------------------------------------------------------------ */

static int alloc_buffers(MotionVulkanState *s)
{
    size_t bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bytes_per_pixel;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, in_bytes);
    if (err)
        return err;

    /* Blurred reference: shader writes one uint per pixel (lower 16
     * bits payload). */
    size_t blur_bytes = (size_t)s->width * s->height * sizeof(uint32_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->blur[0], blur_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->blur[1], blur_bytes);
    if (err)
        return err;

    /* Per-WG SAD partials. */
    uint32_t gx = 0;
    uint32_t gy = 0;
    motion_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t sad_bytes = (size_t)s->wg_count * sizeof(int64_t);
    if (sad_bytes == 0)
        sad_bytes = sizeof(int64_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->sad_partials, sad_bytes);
    if (err)
        return err;

    return 0;
}

/* ------------------------------------------------------------------ */
/* init().                                                              */
/* ------------------------------------------------------------------ */

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    MotionVulkanState *s = fex->priv;

    /* Reject the 5-frame window mode explicitly. The CPU extractor
     * keeps a 5-deep blur ring and computes a second SAD pair (i-2 ↔
     * i-4); the GPU port ships only the 3-frame ring today. Failing
     * loud with -ENOTSUP keeps callers from silently falling back to
     * a wrong answer. See ADR-0219. */
    if (s->motion_five_frame_window) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "motion_vulkan: motion_five_frame_window=true is not yet supported on the "
                 "Vulkan backend (T3-15(c) deferred). Use the CPU extractor `motion` instead.\n");
        return -ENOTSUP;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->frame_index = 0;
    s->prev_motion_score = 0.0;
    s->prev_motion3_blended = 0.0;
    s->cur_blur = 0;

    /* Borrow the framework's imported context when available, fall
     * back to lazy creation. Same pattern as vif_vulkan. */
    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "motion_vulkan: cannot create Vulkan context (%d)\n",
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
/* Per-frame: upload + record + submit + wait + reduce.                */
/* ------------------------------------------------------------------ */

static int upload_ref_plane(MotionVulkanState *s, VmafPicture *pic)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(s->ref_in);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, s->ref_in);
}

static int write_descriptor_set(MotionVulkanState *s, VkDescriptorSet set)
{
    int cur = s->cur_blur;
    int prev = 1 - cur;

    VkDescriptorBufferInfo dbi[4] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->blur[cur]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->blur[prev]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->sad_partials),
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

static double reduce_sad_partials(const MotionVulkanState *s)
{
    const int64_t *slots = vmaf_vulkan_buffer_host(s->sad_partials);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += slots[i];
    return (double)total / 256.0 / ((double)s->width * s->height);
}

static int extract_force_zero(MotionVulkanState *s, unsigned index, VmafFeatureCollector *fc)
{
    int err = 0;
    if (s->frame_index > 0) {
        err |= vmaf_feature_collector_append_with_dict(
            fc, s->feature_name_dict, "VMAF_integer_feature_motion_score", 0.0, index);
    }
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_integer_feature_motion2_score", 0.0, index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_integer_feature_motion3_score", 0.0, index);
    s->frame_index++;
    return err;
}

/* ------------------------------------------------------------------ */
/* motion3 post-processing — pure host-side scalar work.              */
/*                                                                     */
/* Mirrors libvmaf/src/feature/integer_motion.c lines 510-560:        */
/*   processed = min(motion_blend(score2 * fps_weight, blend_factor,  */
/*                                blend_offset), max_val)              */
/*   if motion_moving_average:                                         */
/*       processed = (processed + previous_unaveraged) / 2             */
/*                                                                     */
/* `score2` is `motion2_score` (already clipped in our caller).       */
/* `prev_motion3_blended` keeps the *unaveraged* previous so the       */
/* moving-average rule cascades correctly across frames — exactly     */
/* the CPU pattern.                                                    */
/* ------------------------------------------------------------------ */
static double motion3_postprocess(MotionVulkanState *s, double score2)
{
    double const weighted = score2 * s->motion_fps_weight;
    double const blended = motion_blend(weighted, s->motion_blend_factor, s->motion_blend_offset);
    double const clipped = MIN(blended, s->motion_max_val);
    double const previous_unaveraged = s->prev_motion3_blended;
    s->prev_motion3_blended = clipped;
    if (s->motion_moving_average && s->frame_index > 1) {
        return (clipped + previous_unaveraged) / 2.0;
    }
    return clipped;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    MotionVulkanState *s = fex->priv;
    int err = 0;

    if (s->motion_force_zero)
        return extract_force_zero(s, index, feature_collector);

    err = upload_ref_plane(s, ref_pic);
    if (err)
        return err;

    /* Zero per-WG SAD partials before each dispatch (the shader
     * overwrites every slot, but defensive zero protects against
     * driver-managed memory not being initialised on the host side). */
    memset(vmaf_vulkan_buffer_host(s->sad_partials), 0, (size_t)s->wg_count * sizeof(int64_t));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->sad_partials);
    if (err)
        return err;

    /* Allocate one descriptor set, bind, record + submit. */
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
    motion_wg_dims(s->width, s->height, &gx, &gy);
    int compute_sad = (s->frame_index > 0) ? 1 : 0;
    MotionPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .compute_sad = (uint32_t)compute_sad,
        .num_workgroups_x = gx,
    };

    /* Use pipelines[1] when SAD is computed, pipelines[0] otherwise.
     * They share the same SPIR-V; the COMPUTE_SAD branch is gated by
     * the push constant. The 2-pipeline split is symmetric scaffolding
     * for a future spec-constant move (see create_pipelines comment). */
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

    /* ---------------- Host-side reduction + score emit ---------------- */
    double motion_score = 0.0;
    if (s->frame_index > 0)
        motion_score = reduce_sad_partials(s);

    /* Match CPU motion algorithm (integer_motion.c lines 480-560):
     *   index 0:        write motion2_score = 0 (and motion_score = 0
     *                   when debug). No prior frame.
     *   index 1:        write motion_score (debug). Don't write
     *                   motion2_score yet — CPU returns early.
     *   index >= 2:     write motion2_score[index-1] = min(prev, cur).
     */
    if (s->frame_index == 0) {
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_integer_feature_motion2_score", 0.0,
                                                      index);
        if (s->debug && !err) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_integer_feature_motion_score", 0.0,
                                                          index);
        }
    } else if (s->frame_index == 1) {
        /* Match CPU integer_motion.c lines 510-530: when the second
         * frame arrives, write motion3_score for index 0 (back-fill
         * the very first frame) using the current motion as the
         * substrate (no min(prev, cur) yet — there's no prev). */
        double const score_clipped = MIN(motion_score * s->motion_fps_weight, s->motion_max_val);
        double const motion3_score = motion3_postprocess(s, score_clipped);
        if (!err) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_integer_feature_motion3_score",
                                                          motion3_score, index - 1);
        }
        if (s->debug && !err) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_integer_feature_motion_score",
                                                          motion_score, index);
        }
    } else {
        double motion2 =
            (motion_score < s->prev_motion_score) ? motion_score : s->prev_motion_score;
        double const motion2_clipped = MIN(motion2 * s->motion_fps_weight, s->motion_max_val);
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_integer_feature_motion2_score", motion2,
                                                      index - 1);
        if (!err) {
            double const motion3_score = motion3_postprocess(s, motion2_clipped);
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_integer_feature_motion3_score",
                                                          motion3_score, index - 1);
        }
        if (s->debug && !err) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_integer_feature_motion_score",
                                                          motion_score, index);
        }
    }

    s->prev_motion_score = motion_score;
    s->cur_blur = 1 - s->cur_blur;
    s->frame_index++;

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    if (set != VK_NULL_HANDLE)
        vkFreeDescriptorSets(s->ctx->device, s->pl.desc_pool, 1, &set);
    return err;
}

/* ------------------------------------------------------------------ */
/* flush() — write the final motion2_score (delayed-by-one).          */
/* ------------------------------------------------------------------ */

static int flush(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    MotionVulkanState *s = fex->priv;
    int ret = 0;
    if (s->motion_force_zero)
        return 1;

    if (s->frame_index > 1) {
        double const last_motion2 =
            MIN(s->prev_motion_score * s->motion_fps_weight, s->motion_max_val);
        ret = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_integer_feature_motion2_score",
                                                      last_motion2, s->frame_index - 1);
        if (ret >= 0) {
            double const motion3_score = motion3_postprocess(s, last_motion2);
            int ret_m3 = vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "VMAF_integer_feature_motion3_score",
                motion3_score, s->frame_index - 1);
            if (ret_m3 < 0)
                ret = ret_m3;
        }
    }
    return (ret < 0) ? ret : !ret;
}

/* ------------------------------------------------------------------ */
/* close().                                                            */
/* ------------------------------------------------------------------ */

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    /* `vulkan/kernel_template.h` collapses the vkDeviceWaitIdle +
     * 5×vkDestroy* sweep into one call. */
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_in);
    if (s->blur[0])
        vmaf_vulkan_buffer_free(s->ctx, s->blur[0]);
    if (s->blur[1])
        vmaf_vulkan_buffer_free(s->ctx, s->blur[1]);
    if (s->sad_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->sad_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Provided features + registration.                                   */
/*                                                                      */
/* T3-15(c): motion3_score is now provided (3-frame window mode only). */
/* The 5-frame window mode (`motion_five_frame_window=true`) remains  */
/* deferred — init() rejects it with -ENOTSUP.                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"VMAF_integer_feature_motion_score",
                                          "VMAF_integer_feature_motion2_score",
                                          "VMAF_integer_feature_motion3_score",
                                          "integer_motion",
                                          "integer_motion2",
                                          "integer_motion3",
                                          NULL};

VmafFeatureExtractor vmaf_fex_integer_motion_vulkan = {
    .name = "motion_vulkan",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(MotionVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
