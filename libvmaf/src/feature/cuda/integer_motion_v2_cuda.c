/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  motion_v2 feature kernel on the CUDA backend (T7-23 / batch 3
 *  part 1b — ADR-0192 / ADR-0193). CUDA twin of motion_v2_vulkan
 *  (PR #146).
 *
 *  Stateless variant of `motion_cuda`: exploits convolution
 *  linearity (`SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)`)
 *  so each frame computes its score in one kernel launch over
 *  (prev_ref - cur_ref) without storing blurred frames across
 *  submits. Uses a raw-pixel ping-pong (`pix[2]`) instead of
 *  motion_cuda's blurred-frame ping-pong — one D2D copy per submit
 *  caches the current ref Y plane so the next frame can read it as
 *  "prev".
 *
 *  motion2_v2_score = min(score[i], score[i+1]) is emitted host-side
 *  in flush() (same shape as CPU integer_motion_v2.c::flush). No GPU
 *  work needed; the kernel only emits motion_v2_sad_score.
 */

#include <errno.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cuda/integer_motion_v2_cuda.h"
#include "cuda_helper.cuh"
#include "picture.h"
#include "picture_cuda.h"

typedef struct MotionV2StateCuda {
    CUevent event;
    CUevent finished;
    CUfunction funcbpc8;
    CUfunction funcbpc16;
    CUstream str;

    /* Ping-pong of raw ref Y planes (uint8 for bpc<=8, uint16 for
     * bpc>8 — bytes_per_pixel * w * h). pix[index%2] is the current
     * frame's slot; pix[(index+1)%2] is the previous frame's slot. */
    VmafCudaBuffer *pix[2];

    /* Single int64 atomic accumulator per frame. */
    VmafCudaBuffer *sad;
    uint64_t *sad_host;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    size_t plane_bytes;

    VmafDictionary *feature_name_dict;
} MotionV2StateCuda;

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    MotionV2StateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8 ? 1u : 2u);

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT), fail);

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, motion_v2_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "motion_v2_kernel_8bpc"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "motion_v2_kernel_16bpc"),
                    fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->pix[0], s->plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->pix[1], s->plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->sad, sizeof(uint64_t));
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->sad_host, sizeof(uint64_t));
    if (ret)
        goto free_buffers;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_buffers;

    return 0;

free_buffers:
    if (s->pix[0]) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->pix[0]);
        free(s->pix[0]);
    }
    if (s->pix[1]) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->pix[1]);
        free(s->pix[1]);
    }
    if (s->sad) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->sad);
        free(s->sad);
    }
    (void)vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)dist_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    MotionV2StateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    const unsigned cur_idx = index % 2u;
    const unsigned prev_idx = (index + 1u) % 2u;

    CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);

    /* Cache cur ref Y plane into pix[cur_idx] so the next frame can
     * read it as "prev". The picture's plane is itself on device — a
     * D2D copy of the contiguous Y plane (stride = w*bpp; libvmaf
     * picture allocator already packs tightly). */
    CHECK_CUDA_RETURN(cu_f,
                      cuStreamWaitEvent(pic_stream, vmaf_cuda_picture_get_ready_event(ref_pic),
                                        CU_EVENT_WAIT_DEFAULT));
    /* Source stride may exceed plane width — copy row by row with
     * cuMemcpy2D so we land a tightly-packed copy in pix[cur_idx].
     * Width of the copy = w * bpp; height = h. */
    CUDA_MEMCPY2D copy = {0};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = (CUdeviceptr)ref_pic->data[0];
    copy.srcPitch = ref_pic->stride[0];
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = (CUdeviceptr)s->pix[cur_idx]->data;
    copy.dstPitch = s->frame_w * (s->bpc <= 8u ? 1u : 2u);
    copy.WidthInBytes = copy.dstPitch;
    copy.Height = s->frame_h;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&copy, pic_stream));

    /* Frame 0: nothing more to do — emit 0 in collect. */
    if (index == 0) {
        CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->event, pic_stream));
        CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
        return 0;
    }

    /* Reset device SAD accumulator (single int64). */
    CHECK_CUDA_RETURN(cu_f, cuMemsetD8Async(s->sad->data, 0, sizeof(uint64_t), pic_stream));

    const unsigned block_dim_x = 16;
    const unsigned block_dim_y = 16;
    const unsigned grid_dim_x = DIV_ROUND_UP(s->frame_w, block_dim_x);
    const unsigned grid_dim_y = DIV_ROUND_UP(s->frame_h, block_dim_y);
    const ptrdiff_t plane_pitch = (ptrdiff_t)(s->frame_w * (s->bpc <= 8u ? 1u : 2u));

    if (s->bpc == 8u) {
        void *args[] = {
            &s->pix[prev_idx]->data, &s->pix[cur_idx]->data, (void *)&plane_pitch,
            (void *)&plane_pitch,    (void *)s->sad,         (void *)&s->frame_w,
            (void *)&s->frame_h,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->funcbpc8, grid_dim_x, grid_dim_y, 1, block_dim_x,
                                               block_dim_y, 1, 0, pic_stream, args, NULL));
    } else {
        void *args[] = {
            &s->pix[prev_idx]->data, &s->pix[cur_idx]->data, (void *)&plane_pitch,
            (void *)&plane_pitch,    (void *)s->sad,         (void *)&s->frame_w,
            (void *)&s->frame_h,     (void *)&s->bpc,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->funcbpc16, grid_dim_x, grid_dim_y, 1, block_dim_x,
                                               block_dim_y, 1, 0, pic_stream, args, NULL));
    }

    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->event, pic_stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));

    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->sad_host, (CUdeviceptr)s->sad->data,
                                              sizeof(*s->sad_host), s->str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->finished, s->str));
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    MotionV2StateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));

    if (index == 0) {
        return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "VMAF_integer_feature_motion_v2_sad_score",
                                                       0.0, index);
    }

    const double sad_score = (double)*s->sad_host / 256.0 / ((double)s->frame_w * s->frame_h);
    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_motion_v2_sad_score",
                                                   sad_score, index);
}

static int flush_fex_cuda(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
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

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    MotionV2StateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    int _cuda_err = 0;
    CHECK_CUDA_GOTO(cu_f, cuStreamSynchronize(s->str), after_stream_sync);
after_stream_sync:
    CHECK_CUDA_GOTO(cu_f, cuStreamDestroy(s->str), after_stream_destroy);
after_stream_destroy:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->event), after_event1_destroy);
after_event1_destroy:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->finished), after_event2_destroy);
after_event2_destroy:;

    int ret = _cuda_err;

    if (s->pix[0]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->pix[0]);
        free(s->pix[0]);
    }
    if (s->pix[1]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->pix[1]);
        free(s->pix[1]);
    }
    if (s->sad) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sad);
        free(s->sad);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"VMAF_integer_feature_motion_v2_sad_score",
                                          "VMAF_integer_feature_motion2_v2_score", NULL};

VmafFeatureExtractor vmaf_fex_integer_motion_v2_cuda = {
    .name = "motion_v2_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(MotionV2StateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_CUDA,
};
