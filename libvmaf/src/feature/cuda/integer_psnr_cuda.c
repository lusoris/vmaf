/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the CUDA backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1b).
 *
 *  Per-pixel squared-error reduction → host-side log10 → score.
 *  Mirrors the Vulkan psnr_vulkan.c host scaffolding shipped in
 *  PR #125 but uses CUDA's async submit/collect model (parallel
 *  with motion_cuda.c). Single dispatch per channel; this v1
 *  emits luma-only (`psnr_y`).
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/integer_psnr_cuda.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

typedef struct PsnrStateCuda {
    CUevent event;
    CUevent finished;
    CUfunction funcbpc8;
    CUfunction funcbpc16;
    CUstream str;
    VmafCudaBuffer *sse;
    uint64_t *sse_host;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    uint32_t peak;
    double psnr_max_y;
    VmafDictionary *feature_name_dict;
} PsnrStateCuda;

static const VmafOption options[] = {{0}};

static int psnr_cuda_dispatch(const VmafPicture *ref, const VmafPicture *dis, VmafCudaBuffer *sse,
                              unsigned width, unsigned height, unsigned bpc, CUfunction funcbpc8,
                              CUfunction funcbpc16, CudaFunctions *cu_f, CUstream stream)
{
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    const int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    void *kernelParams[] = {(void *)ref, (void *)dis, (void *)sse, &width, &height};
    CUfunction func = (bpc == 8) ? funcbpc8 : funcbpc16;
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(func, grid_dim_x, grid_dim_y, 1, block_dim_x,
                                           block_dim_y, 1, 0, stream, kernelParams, NULL));
    return 0;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)w;
    (void)h;
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT), fail);

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, psnr_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "calculate_psnr_kernel_8bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "calculate_psnr_kernel_16bpc"),
                    fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;
    const double peak_d = (double)s->peak;
    s->psnr_max_y = 10.0 * log10((peak_d * peak_d) / 0.5);

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->sse, sizeof(uint64_t));
    if (ret)
        goto free_ref;
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->sse_host, sizeof(uint64_t));
    if (ret)
        goto free_ref;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;

    return 0;

free_ref:
    if (s->sse) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sse);
        free(s->sse);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    (void)ret;
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
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Reset device SSE counter to zero. */
    CHECK_CUDA_RETURN(cu_f, cuMemsetD8Async(s->sse->data, 0, sizeof(uint64_t), s->str));

    /* Wait for the dist-side ready event on the ref-side stream
     * (motion_cuda.c uses the same trick to keep both planes
     * uploaded before the kernel reads). */
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                                              vmaf_cuda_picture_get_ready_event(dist_pic),
                                              CU_EVENT_WAIT_DEFAULT));

    int err =
        psnr_cuda_dispatch(ref_pic, dist_pic, s->sse, ref_pic->w[0], ref_pic->h[0], s->bpc,
                           s->funcbpc8, s->funcbpc16, cu_f, vmaf_cuda_picture_get_stream(ref_pic));
    if (err)
        return err;

    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->event, vmaf_cuda_picture_get_stream(ref_pic)));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));

    /* DtoH copy of the SSE counter on our private stream. */
    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->sse_host, (CUdeviceptr)s->sse->data,
                                              sizeof(*s->sse_host), s->str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->finished, s->str));
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));

    const double sse = (double)*s->sse_host;
    const double n_pixels = (double)s->frame_w * (double)s->frame_h;
    const double mse = sse / n_pixels;
    double psnr_y = (sse <= 0.0) ? s->psnr_max_y : 10.0 * log10(((double)s->peak * s->peak) / mse);
    if (psnr_y > s->psnr_max_y)
        psnr_y = s->psnr_max_y;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "psnr_y", psnr_y, index);
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    PsnrStateCuda *s = fex->priv;
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
    if (s->sse) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->sse);
        free(s->sse);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"psnr_y", NULL};

VmafFeatureExtractor vmaf_fex_psnr_cuda = {
    .name = "psnr_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(PsnrStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    /* 1 dispatch/frame, reduction-dominated; AUTO + 1080p area
     * matches motion's profile (see ADR-0181 / ADR-0182). */
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
