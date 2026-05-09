/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the CUDA backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1b; chroma extension T3-15(b) / ADR-0351).
 *
 *  Per-pixel squared-error reduction -> host-side log10 -> score.
 *  One dispatch per active plane (Y, Cb, Cr); the same kernel pair
 *  (`calculate_psnr_kernel_{8,16}bpc`) is launched up to three times
 *  per frame against per-plane data[] / stride[] entries on the
 *  uploaded VmafPicture. Chroma buffers are sized for the active
 *  subsampling (4:2:0 -> w/2 x h/2, 4:2:2 -> w/2 x h, 4:4:4 -> w x h);
 *  the kernel is plane-agnostic and reads its dims out of the launch
 *  arguments. Upload coverage: chroma planes are uploaded for any
 *  non-YUV400P input by `libvmaf.c::translate_picture_host` (since
 *  T7-23 batch 1c, the ciede_cuda landing).
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::extract):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;        (per channel)
 *      mse = sse / (w_p * h_p);
 *      psnr = (sse == 0) ? psnr_max[p] : 10 * log10(peak * peak / mse);
 *  Bit-exactness contract: int64 SSE accumulation -> places=4 vs CPU.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/psnr_vulkan.c
 *  (chroma extension twin, ADR-0216) and motion_cuda.c
 *  (single-dispatch async lifecycle).
 *
 *  4:0:0 (YUV400) handling: chroma planes are absent on the source
 *  picture, so only the luma plane is dispatched and only `psnr_y`
 *  is emitted. Mirrors CPU integer_psnr.c::init's `enable_chroma =
 *  false` branch.
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
#include "kernel_template.h"

#define PSNR_NUM_PLANES 3

typedef struct PsnrStateCuda {
    /* Lifecycle (private stream + submit/finished event pair) is
     * shared across all per-plane dispatches — a single readback
     * stream drains the per-plane SSE accumulators in one host wait
     * point at collect() time. */
    VmafCudaKernelLifecycle lc;

    /* One (device SSE accumulator, pinned host readback slot) pair
     * per plane. Chroma slots are unused on YUV400P inputs (and
     * still allocated; the per-plane allocation is uniform to keep
     * close + error-unwind simple). */
    VmafCudaKernelReadback rb[PSNR_NUM_PLANES];

    CUfunction funcbpc8;
    CUfunction funcbpc16;
    unsigned index;
    unsigned plane_w[PSNR_NUM_PLANES];
    unsigned plane_h[PSNR_NUM_PLANES];
    unsigned n_planes; /* 1 for YUV400P, 3 otherwise. */
    unsigned bpc;
    uint32_t peak;
    /* Per-plane psnr_max (default branch: (6*bpc)+12, identical for
     * all three planes — mirrors integer_psnr.c::init). The array
     * layout makes the min_sse-driven per-plane formula a one-line
     * change if a future opts row enables it. */
    double psnr_max[PSNR_NUM_PLANES];
    VmafDictionary *feature_name_dict;
} PsnrStateCuda;

static const VmafOption options[] = {{0}};

static int psnr_cuda_dispatch(const VmafPicture *ref, const VmafPicture *dis, VmafCudaBuffer *sse,
                              unsigned width, unsigned height, unsigned plane, unsigned bpc,
                              CUfunction funcbpc8, CUfunction funcbpc16, CudaFunctions *cu_f,
                              CUstream stream)
{
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    const int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    void *kernelParams[] = {(void *)ref, (void *)dis, (void *)sse, &width, &height, &plane};
    CUfunction func = (bpc == 8) ? funcbpc8 : funcbpc16;
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(func, grid_dim_x, grid_dim_y, 1, block_dim_x,
                                           block_dim_y, 1, 0, stream, kernelParams, NULL));
    return 0;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)w;
    (void)h;
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Per-plane geometry derived from pix_fmt (mirrors CPU
     * integer_psnr.c::init's (ss_hor, ss_ver) logic). */
    s->plane_w[0] = w;
    s->plane_h[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1;
        s->plane_w[1] = s->plane_w[2] = 0;
        s->plane_h[1] = s->plane_h[2] = 0;
    } else {
        s->n_planes = PSNR_NUM_PLANES;
        const int ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P);
        const int ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P);
        s->plane_w[1] = s->plane_w[2] = w >> ss_hor;
        s->plane_h[1] = s->plane_h[2] = h >> ss_ver;
    }

    /* Stream + event pair via the template — replaces the
     * cuCtxPushCurrent -> cuStreamCreateWithPriority -> cuEventCreate x2
     * -> cuCtxPopCurrent block every CUDA feature kernel hand-rolled. */
    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    /* Module load + function lookups stay per-feature (each metric
     * has its own .ptx blob and entry-point names). */
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, psnr_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "calculate_psnr_kernel_8bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "calculate_psnr_kernel_16bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;
    /* psnr_max per plane: default branch (6*bpc)+12 (no min_sse opt
     * shipped on this extractor today). Same value across planes;
     * the array layout matches Vulkan/CPU and lets a future
     * min_sse-aware variant fill in plane-specific bounds. */
    const double psnr_max = (double)((6u * bpc) + 12u);
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
        s->psnr_max[p] = psnr_max;
    }

    /* Per-plane readback pair (device SSE accumulator + pinned host
     * slot) via the template. Uniform allocation across all 3 slots
     * keeps the close path symmetric; chroma slots stay quiescent on
     * YUV400P inputs (no dispatch, no DtoH). */
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
        err = vmaf_cuda_kernel_readback_alloc(&s->rb[p], fex->cu_state, sizeof(uint64_t));
        if (err)
            goto free_ref;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;

    return 0;

free_ref:
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
        (void)vmaf_cuda_kernel_readback_free(&s->rb[p], fex->cu_state);
    }
    if (s->feature_name_dict) {
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    }
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return -ENOMEM;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
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
    /* Refresh per-plane geometry from the live picture (init() set
     * dimensions based on declared pix_fmt; the picture is the
     * authoritative source for actual w/h per plane). */
    for (unsigned p = 0; p < s->n_planes; p++) {
        s->plane_w[p] = ref_pic->w[p];
        s->plane_h[p] = ref_pic->h[p];
    }

    /* Pre-launch boilerplate (per plane): zero the device accumulator
     * on `lc.str` and wait for the dist-side ready event on the
     * picture stream. The wait is idempotent across planes; the
     * memset writes distinct buffers. */
    CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CUevent dist_ready = vmaf_cuda_picture_get_ready_event(dist_pic);
    for (unsigned p = 0; p < s->n_planes; p++) {
        int err = vmaf_cuda_kernel_submit_pre_launch(&s->lc, fex->cu_state, &s->rb[p], pic_stream,
                                                     dist_ready);
        if (err)
            return err;
    }

    /* One dispatch per active plane on the picture stream — the
     * SSBOs are independent across planes, so no inter-dispatch
     * barrier is needed. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        int err =
            psnr_cuda_dispatch(ref_pic, dist_pic, s->rb[p].device, s->plane_w[p], s->plane_h[p], p,
                               s->bpc, s->funcbpc8, s->funcbpc16, cu_f, pic_stream);
        if (err)
            return err;
    }

    /* Post-launch readback: record submit on the picture stream once
     * (covers all per-plane dispatches), wait on the readback stream,
     * DtoH copy the per-plane accumulators in series on `lc.str`,
     * record `finished`. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, pic_stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    for (unsigned p = 0; p < s->n_planes; p++) {
        CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->rb[p].host_pinned,
                                                  (CUdeviceptr)s->rb[p].device->data,
                                                  s->rb[p].bytes, s->lc.str));
    }
    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;

    /* Drain the private readback stream so all per-plane host pinned
     * buffers are safe to read. */
    int err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (err)
        return err;

    int rc = 0;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double sse = (double)*(uint64_t *)s->rb[p].host_pinned;
        const double n_pixels = (double)s->plane_w[p] * (double)s->plane_h[p];
        const double mse = sse / n_pixels;
        double psnr =
            (sse <= 0.0) ? s->psnr_max[p] : 10.0 * log10(((double)s->peak * s->peak) / mse);
        if (psnr > s->psnr_max[p])
            psnr = s->psnr_max[p];

        const int e = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (e && rc == 0)
            rc = e;
    }

    return rc;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    PsnrStateCuda *s = fex->priv;

    /* Lifecycle teardown via the template (sync -> destroy stream ->
     * destroy events). Best-effort error aggregation matches the
     * old hand-rolled CHECK_CUDA_GOTO chain. */
    int rc = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
        int err = vmaf_cuda_kernel_readback_free(&s->rb[p], fex->cu_state);
        if (err && rc == 0)
            rc = err;
    }
    int err = vmaf_dictionary_free(&s->feature_name_dict);
    if (err && rc == 0)
        rc = err;
    return rc;
}

static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

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
    /* Up to 3 dispatches/frame (one per plane), reduction-dominated;
     * AUTO + 1080p area matches motion's profile (see ADR-0181 /
     * ADR-0182). Chroma dispatches at 4:2:0 each cover ~25 % of luma
     * area, so the wall-time impact is sub-linear in plane count. */
    .chars =
        {
            .n_dispatches_per_frame = PSNR_NUM_PLANES,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
