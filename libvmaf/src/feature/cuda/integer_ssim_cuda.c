/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature extractor on the CUDA backend
 *  (T7-23 / ADR-0188 / ADR-0189, GPU long-tail batch 2 part 1b).
 *  CUDA twin of ssim_vulkan (PR #139). Two-pass design mirrors
 *  the GLSL shader: horizontal 11-tap separable Gaussian over
 *  ref / cmp / ref² / cmp² / ref·cmp into 5 intermediate float
 *  buffers, then vertical 11-tap + per-pixel SSIM combine +
 *  per-block float partial sums. Host accumulates partials in
 *  `double`, divides by (W-10)·(H-10) and emits `float_ssim`.
 *
 *  Mirrors the psnr_cuda submit/collect scaffolding and the
 *  ciede_cuda per-block-partials precision pattern.
 *
 *  v1: scale=1 only — same constraint as ssim_vulkan. Auto-
 *  decimation is rejected at init with -EINVAL.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/integer_ssim_cuda.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

#define SSIM_BLOCK_X 16
#define SSIM_BLOCK_Y 8
#define SSIM_K 11

typedef struct SsimStateCuda {
    CUevent event;
    CUevent finished;
    CUfunction func_horiz_8;
    CUfunction func_horiz_16;
    CUfunction func_vert;
    CUstream str;
    int scale_override;

    /* 5 intermediate float buffers + per-block partials + host
     * pinned partials. Sized at init for the announced (w, h). */
    VmafCudaBuffer *h_ref_mu;
    VmafCudaBuffer *h_cmp_mu;
    VmafCudaBuffer *h_ref_sq;
    VmafCudaBuffer *h_cmp_sq;
    VmafCudaBuffer *h_refcmp;
    VmafCudaBuffer *partials;
    float *partials_host;
    unsigned partials_capacity;
    unsigned partials_count;

    unsigned width;
    unsigned height;
    unsigned w_horiz;
    unsigned h_horiz;
    unsigned w_final;
    unsigned h_final;
    unsigned bpc;
    float c1;
    float c2;

    unsigned index;
    VmafDictionary *feature_name_dict;
} SsimStateCuda;

static int round_to_int(float x)
{
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

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(SsimStateCuda, scale_override),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {0},
};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    SsimStateCuda *s = fex->priv;

    int scale = compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_cuda: v1 supports scale=1 only (auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_cuda:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < SSIM_K || h < SSIM_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_cuda: input %ux%u smaller than 11x11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }

    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT), fail);

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, ssim_score_ptx), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_horiz_8, module, "calculate_ssim_horiz_8bpc"), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_horiz_16, module, "calculate_ssim_horiz_16bpc"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_vert, module, "calculate_ssim_vert_combine"),
                    fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (SSIM_K - 1);
    s->h_horiz = h;
    s->w_final = w - (SSIM_K - 1);
    s->h_final = h - (SSIM_K - 1);
    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    const unsigned grid_x = (s->w_final + SSIM_BLOCK_X - 1) / SSIM_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_BLOCK_Y - 1) / SSIM_BLOCK_Y;
    s->partials_capacity = grid_x * grid_y;
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t partials_bytes = (size_t)s->partials_capacity * sizeof(float);

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_ref_mu, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_cmp_mu, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_ref_sq, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_cmp_sq, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_refcmp, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->partials, partials_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->partials_host, partials_bytes);
    if (ret)
        goto free_ref;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;
    return 0;

free_ref:
    if (s->h_ref_mu)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_mu);
    if (s->h_cmp_mu)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_mu);
    if (s->h_ref_sq)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_sq);
    if (s->h_cmp_sq)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_sq);
    if (s->h_refcmp)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_refcmp);
    if (s->partials)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->partials);
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
    (void)ref_pic_90;
    (void)dist_pic_90;
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;
    const unsigned grid_x = (s->w_final + SSIM_BLOCK_X - 1) / SSIM_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_BLOCK_Y - 1) / SSIM_BLOCK_Y;
    s->partials_count = grid_x * grid_y;

    /* Sync ref-side stream against dist's ready event (matches
     * psnr_cuda's pattern). */
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                                              vmaf_cuda_picture_get_ready_event(dist_pic),
                                              CU_EVENT_WAIT_DEFAULT));

    /* Pass 1 — horizontal. Grid sized over (W-10) × H. */
    const unsigned grid_horiz_x = (s->w_horiz + SSIM_BLOCK_X - 1) / SSIM_BLOCK_X;
    const unsigned grid_horiz_y = (s->h_horiz + SSIM_BLOCK_Y - 1) / SSIM_BLOCK_Y;
    CUstream stream = vmaf_cuda_picture_get_stream(ref_pic);
    if (s->bpc == 8) {
        unsigned width = s->width;
        void *params[] = {
            (void *)ref_pic,     (void *)dist_pic,
            (void *)s->h_ref_mu, (void *)s->h_cmp_mu,
            (void *)s->h_ref_sq, (void *)s->h_cmp_sq,
            (void *)s->h_refcmp, &s->w_horiz,
            &s->h_horiz,         &width,
        };
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_horiz_8, grid_horiz_x, grid_horiz_y, 1,
                                         SSIM_BLOCK_X, SSIM_BLOCK_Y, 1, 0, stream, params, NULL));
    } else {
        unsigned bpc = s->bpc;
        unsigned width = s->width;
        void *params[] = {
            (void *)ref_pic,
            (void *)dist_pic,
            (void *)s->h_ref_mu,
            (void *)s->h_cmp_mu,
            (void *)s->h_ref_sq,
            (void *)s->h_cmp_sq,
            (void *)s->h_refcmp,
            &s->w_horiz,
            &s->h_horiz,
            &bpc,
            &width,
        };
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_horiz_16, grid_horiz_x, grid_horiz_y, 1,
                                         SSIM_BLOCK_X, SSIM_BLOCK_Y, 1, 0, stream, params, NULL));
    }

    /* Pass 2 — vertical + SSIM combine. Grid sized over
     * (W-10) × (H-10). The horiz pass writes happen-before
     * the vert pass reads on the same stream — implicit
     * stream ordering, no extra event needed. */
    void *params2[] = {
        (void *)s->h_ref_mu,
        (void *)s->h_cmp_mu,
        (void *)s->h_ref_sq,
        (void *)s->h_cmp_sq,
        (void *)s->h_refcmp,
        (void *)s->partials,
        &s->w_horiz,
        &s->w_final,
        &s->h_final,
        &s->c1,
        &s->c2,
    };
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_vert, grid_x, grid_y, 1, SSIM_BLOCK_X,
                                           SSIM_BLOCK_Y, 1, 0, stream, params2, NULL));

    /* DtoH copy of the partials on our private stream. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->event, stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->partials_host, (CUdeviceptr)s->partials->data,
                                              (size_t)s->partials_count * sizeof(float), s->str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->finished, s->str));
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));

    double total = 0.0;
    for (unsigned i = 0; i < s->partials_count; i++)
        total += (double)s->partials_host[i];
    const double n_pixels = (double)s->w_final * (double)s->h_final;
    const double score = total / n_pixels;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_ssim", score, index);
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SsimStateCuda *s = fex->priv;
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
    if (s->h_ref_mu) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_mu);
        free(s->h_ref_mu);
    }
    if (s->h_cmp_mu) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_mu);
        free(s->h_cmp_mu);
    }
    if (s->h_ref_sq) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_sq);
        free(s->h_ref_sq);
    }
    if (s->h_cmp_sq) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_sq);
        free(s->h_cmp_sq);
    }
    if (s->h_refcmp) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_refcmp);
        free(s->h_refcmp);
    }
    if (s->partials) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->partials);
        free(s->partials);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"float_ssim", NULL};

VmafFeatureExtractor vmaf_fex_float_ssim_cuda = {
    .name = "float_ssim_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(SsimStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
