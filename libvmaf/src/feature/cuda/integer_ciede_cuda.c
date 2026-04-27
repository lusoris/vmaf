/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ciede2000 feature extractor on the CUDA backend (T7-23 /
 *  ADR-0182, GPU long-tail batch 1c part 2). CUDA twin of
 *  ciede_vulkan (PR #136 / ADR-0187); shares the float-precision
 *  contract — places=4 empirical floor on real hardware.
 *
 *  Single dispatch per frame; warp-shuffle reduces 32 thread
 *  contributions to one float, lane 0 atomicAdd's to a single
 *  device counter. Host divides the counter by w*h and applies
 *  the CPU's logarithmic transform `45 - 20*log10(mean_dE)` for
 *  the final `ciede2000` metric. Mirrors psnr_cuda's submit /
 *  collect scaffolding.
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
#include "cuda/integer_ciede_cuda.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

typedef struct CiedeStateCuda {
    CUevent event;
    CUevent finished;
    CUfunction funcbpc8;
    CUfunction funcbpc16;
    CUstream str;
    VmafCudaBuffer *partials; /* device: float[n_blocks] */
    float *partials_host;     /* host pinned: float[n_blocks] */
    unsigned partials_capacity;
    unsigned partials_count;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    unsigned ss_hor;
    unsigned ss_ver;
    VmafDictionary *feature_name_dict;
} CiedeStateCuda;

static const VmafOption options[] = {{0}};

static int ciede_cuda_dispatch(const VmafPicture *ref, const VmafPicture *dis,
                               VmafCudaBuffer *partials, unsigned width, unsigned height,
                               unsigned bpc, unsigned ss_hor, unsigned ss_ver, CUfunction funcbpc8,
                               CUfunction funcbpc16, CudaFunctions *cu_f, CUstream stream)
{
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    const int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    void *kernelParams[] = {(void *)ref, (void *)dis, (void *)partials, &width,
                            &height,     &bpc,        &ss_hor,          &ss_ver};
    CUfunction func = (bpc == 8) ? funcbpc8 : funcbpc16;
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(func, grid_dim_x, grid_dim_y, 1, block_dim_x,
                                           block_dim_y, 1, 0, stream, kernelParams, NULL));
    return 0;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;
    (void)w;
    (void)h;
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->event, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT), fail);

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, ciede_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "calculate_ciede_kernel_8bpc"),
                    fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->funcbpc16, module, "calculate_ciede_kernel_16bpc"), fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    s->bpc = bpc;
    s->ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P) ? 1u : 0u;
    s->ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P) ? 1u : 0u;

    /* Pre-size partials for the announced (w, h). Submit reuses if
     * the picture is the same size; reallocates if a larger picture
     * arrives (rare, but the API doesn't pin geometry). */
    const unsigned grid_x = (w + 15u) / 16u;
    const unsigned grid_y = (h + 15u) / 16u;
    s->partials_capacity = grid_x * grid_y;

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->partials,
                                  (size_t)s->partials_capacity * sizeof(float));
    if (ret)
        goto free_ref;
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->partials_host,
                                       (size_t)s->partials_capacity * sizeof(float));
    if (ret)
        goto free_ref;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;

    return 0;

free_ref:
    if (s->partials) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->partials);
        free(s->partials);
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
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    const unsigned grid_x = (s->frame_w + 15u) / 16u;
    const unsigned grid_y = (s->frame_h + 15u) / 16u;
    s->partials_count = grid_x * grid_y;

    /* The kernel writes one float per block (no atomic), so we
     * don't need a device-side memset before launch. */

    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                                              vmaf_cuda_picture_get_ready_event(dist_pic),
                                              CU_EVENT_WAIT_DEFAULT));

    int err = ciede_cuda_dispatch(ref_pic, dist_pic, s->partials, ref_pic->w[0], ref_pic->h[0],
                                  s->bpc, s->ss_hor, s->ss_ver, s->funcbpc8, s->funcbpc16, cu_f,
                                  vmaf_cuda_picture_get_stream(ref_pic));
    if (err)
        return err;

    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->event, vmaf_cuda_picture_get_stream(ref_pic)));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->str, s->event, CU_EVENT_WAIT_DEFAULT));

    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->partials_host, (CUdeviceptr)s->partials->data,
                                              (size_t)s->partials_count * sizeof(float), s->str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->finished, s->str));
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    CiedeStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));

    /* Per-block partials → host accumulation in double. Same
     * precision argument as ciede_vulkan (ADR-0187): per-block
     * sums fit in float7 precision (max ~5000 magnitude); the
     * cross-block reduction across thousands of partials needs
     * double to retain places=4. */
    double total = 0.0;
    for (unsigned i = 0; i < s->partials_count; i++)
        total += (double)s->partials_host[i];
    const double n_pixels = (double)s->frame_w * (double)s->frame_h;
    const double mean_de = total / n_pixels;
    const double score = 45.0 - 20.0 * log10(mean_de);

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "ciede2000", score, index);
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    CiedeStateCuda *s = fex->priv;
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
    if (s->partials) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->partials);
        free(s->partials);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"ciede2000", NULL};

VmafFeatureExtractor vmaf_fex_ciede_cuda = {
    .name = "ciede_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(CiedeStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
