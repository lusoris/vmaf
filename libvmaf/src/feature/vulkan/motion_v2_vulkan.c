/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  motion_v2 feature kernel on the Vulkan backend (T7-23 / batch 3 part
 *  1a — ADR-0192 / ADR-0193).
 *
 *  Stateless variant of `motion_vulkan`: exploits convolution linearity
 *  (`SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)`) so we can
 *  compute the score in one dispatch over (prev_ref - cur_ref) without
 *  storing blurred frames across `extract` calls. Direct port of
 *  libvmaf/src/feature/integer_motion_v2.c.
 *
 *  Algorithm (mirrors CPU integer_motion_v2):
 *    1. V->H separable 5-tap Gaussian blur of (prev_ref - cur_ref)
 *       (signed; filter sum 65536; round 1<<(bpc-1) >> bpc, then
 *       round 1<<15 >> 16) — implemented in shaders/motion_v2.comp.
 *    2. Sum |blurred_diff| over the plane (per-WG int64 partial,
 *       host-side scalar reduction).
 *    3. motion_v2_sad_score = SAD / 256.0 / (width * height)
 *    4. motion2_v2_score = min(cur, next) emitted in flush(), reading
 *       back the collected sad scores (same shape as CPU's flush).
 *
 *  Pattern reference: motion_vulkan.c (the closest sibling — same
 *  separable filter, same int64 partials, same VkSpecializationInfo
 *  shape). motion_v2 simplifies by dropping the blur ping-pong (no
 *  blur output buffer to keep across frames) but adds a raw-pixel
 *  ping-pong (so we don't re-upload the previous frame each call).
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

#include "motion_v2_spv.h" /* generated SPIR-V byte array */

#define MOTION_V2_WG_X 32
#define MOTION_V2_WG_Y 4

typedef struct {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Single pipeline — kernel always computes the same SAD. Frame 0
     * is short-circuited host-side without a dispatch. */
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader;
    VkPipeline pipeline;
    VkDescriptorPool desc_pool;

    /* Ping-pong of raw ref planes. ref_buf[cur_ref_idx] = the frame
     * currently being processed; ref_buf[1 - cur_ref_idx] = previous. */
    VmafVulkanBuffer *ref_buf[2];
    int cur_ref_idx;

    /* Per-workgroup int64 SAD partials. */
    VmafVulkanBuffer *sad_partials;
    unsigned wg_count;

    unsigned frame_index;

    VmafDictionary *feature_name_dict;
} MotionV2VulkanState;

/* Push constants — must mirror `Params` in motion_v2.comp. */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} MotionV2PushConsts;

static inline void motion_v2_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + MOTION_V2_WG_X - 1u) / MOTION_V2_WG_X;
    *gy = (h + MOTION_V2_WG_Y - 1u) / MOTION_V2_WG_Y;
}

static int create_pipelines(MotionV2VulkanState *s)
{
    VkDevice dev = s->ctx->device;

    /* Three storage-buffer bindings, one per SSBO in motion_v2.comp. */
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
        .size = sizeof(MotionV2PushConsts),
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
        .codeSize = motion_v2_spv_size,
        .pCode = motion_v2_spv,
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
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = s->shader,
                  .pName = "main",
                  .pSpecializationInfo = &spec_info},
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

static int alloc_buffers(MotionV2VulkanState *s)
{
    size_t bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bytes_per_pixel;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[0], in_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[1], in_bytes);
    if (err)
        return err;

    uint32_t gx = 0;
    uint32_t gy = 0;
    motion_v2_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t sad_bytes = (size_t)s->wg_count * sizeof(int64_t);
    if (sad_bytes == 0)
        sad_bytes = sizeof(int64_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->sad_partials, sad_bytes);
    if (err)
        return err;

    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    MotionV2VulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->frame_index = 0;
    s->cur_ref_idx = 0;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "motion_v2_vulkan: cannot create Vulkan context (%d)\n",
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

static int upload_ref_plane(MotionV2VulkanState *s, VmafPicture *pic, int slot)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(s->ref_buf[slot]);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, s->ref_buf[slot]);
}

static int write_descriptor_set(MotionV2VulkanState *s, VkDescriptorSet set)
{
    int cur = s->cur_ref_idx;
    int prev = 1 - cur;

    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_buf[prev]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_buf[cur]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->sad_partials),
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

static double reduce_sad_partials(const MotionV2VulkanState *s)
{
    const int64_t *slots = vmaf_vulkan_buffer_host(s->sad_partials);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += slots[i];
    return (double)total / 256.0 / ((double)s->width * s->height);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    MotionV2VulkanState *s = fex->priv;
    int err = 0;

    /* Frame 0: store the ref pixels (so frame 1 has a "prev"); emit 0. */
    if (s->frame_index == 0) {
        err = upload_ref_plane(s, ref_pic, s->cur_ref_idx);
        if (err)
            return err;
        s->cur_ref_idx = 1 - s->cur_ref_idx;
        s->frame_index++;
        return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "VMAF_integer_feature_motion_v2_sad_score",
                                                       0.0, index);
    }

    /* Frame 1+: upload current ref into the slot we will treat as
     * "current"; previous frame's pixels are still in 1 - cur_ref_idx. */
    err = upload_ref_plane(s, ref_pic, s->cur_ref_idx);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->sad_partials), 0, (size_t)s->wg_count * sizeof(int64_t));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->sad_partials);
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
    motion_v2_wg_dims(s->width, s->height, &gx, &gy);
    MotionV2PushConsts pc = {
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

    double sad_score = reduce_sad_partials(s);
    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "VMAF_integer_feature_motion_v2_sad_score",
                                                  sad_score, index);

    s->cur_ref_idx = 1 - s->cur_ref_idx;
    s->frame_index++;

cleanup:
    if (fence != VK_NULL_HANDLE)
        vkDestroyFence(s->ctx->device, fence, NULL);
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(s->ctx->device, s->ctx->command_pool, 1, &cmd);
    if (set != VK_NULL_HANDLE)
        vkFreeDescriptorSets(s->ctx->device, s->desc_pool, 1, &set);
    return err;
}

static int flush(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;

    unsigned n_frames = 0;
    double dummy;
    while (!vmaf_feature_collector_get_score(
        feature_collector, "VMAF_integer_feature_motion_v2_sad_score", &dummy, n_frames))
        n_frames++;

    if (n_frames < 2)
        return 1;

    for (unsigned i = 0; i < n_frames; i++) {
        double score_cur;
        double score_next;
        vmaf_feature_collector_get_score(feature_collector,
                                         "VMAF_integer_feature_motion_v2_sad_score", &score_cur, i);

        double motion2;
        if (i + 1 < n_frames) {
            vmaf_feature_collector_get_score(
                feature_collector, "VMAF_integer_feature_motion_v2_sad_score", &score_next, i + 1);
            motion2 = score_cur < score_next ? score_cur : score_next;
        } else {
            motion2 = score_cur;
        }

        vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion2_v2_score",
                                      motion2, i);
    }

    return 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionV2VulkanState *s = fex->priv;
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

    if (s->ref_buf[0])
        vmaf_vulkan_buffer_free(s->ctx, s->ref_buf[0]);
    if (s->ref_buf[1])
        vmaf_vulkan_buffer_free(s->ctx, s->ref_buf[1]);
    if (s->sad_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->sad_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"VMAF_integer_feature_motion_v2_sad_score",
                                          "VMAF_integer_feature_motion2_v2_score", NULL};

VmafFeatureExtractor vmaf_fex_integer_motion_v2_vulkan = {
    .name = "motion_v2_vulkan",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(MotionV2VulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
