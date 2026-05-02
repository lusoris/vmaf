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
 *
 *  Reference consumer of `cuda/kernel_template.h` (ADR-0246) — the
 *  per-frame async lifecycle (private stream + submit/finished event
 *  pair) and the (device, pinned-host) readback pair are dispensed
 *  by the template instead of being open-coded here. This was
 *  documented as the migration target in `kernel_template.h`'s
 *  docstring; the migration lands the first consumer (T-GPU-DEDUP-4).
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

typedef struct PsnrStateCuda {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device SSE accumulator, pinned host readback slot) pair are
     * managed by `cuda/kernel_template.h` (ADR-0246). */
    VmafCudaKernelLifecycle lc;
    VmafCudaKernelReadback rb;
    CUfunction funcbpc8;
    CUfunction funcbpc16;
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

    /* Stream + event pair via the template — replaces the
     * cuCtxPushCurrent → cuStreamCreateWithPriority → cuEventCreate ×2
     * → cuCtxPopCurrent block every CUDA feature kernel hand-rolled. */
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
    const double peak_d = (double)s->peak;
    s->psnr_max_y = 10.0 * log10((peak_d * peak_d) / 0.5);

    /* Readback pair (device SSE accumulator + pinned host slot) via
     * the template. */
    err = vmaf_cuda_kernel_readback_alloc(&s->rb, fex->cu_state, sizeof(uint64_t));
    if (err)
        goto free_ref;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;

    return 0;

free_ref:
    (void)vmaf_cuda_kernel_readback_free(&s->rb, fex->cu_state);
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
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Pre-launch boilerplate: zero the device accumulator on `lc.str`
     * and wait for the dist-side ready event on the picture stream. */
    int err = vmaf_cuda_kernel_submit_pre_launch(&s->lc, fex->cu_state, &s->rb,
                                                 vmaf_cuda_picture_get_stream(ref_pic),
                                                 vmaf_cuda_picture_get_ready_event(dist_pic));
    if (err)
        return err;

    err =
        psnr_cuda_dispatch(ref_pic, dist_pic, s->rb.device, ref_pic->w[0], ref_pic->h[0], s->bpc,
                           s->funcbpc8, s->funcbpc16, cu_f, vmaf_cuda_picture_get_stream(ref_pic));
    if (err)
        return err;

    /* Post-launch readback: record submit on the picture stream, wait
     * for it on the private readback stream, DtoH copy + record
     * `finished`. The template documents this exact sequence in its
     * docstring; left inline for clarity since the kernel
     * launch + ref_pic stream are inherently per-feature. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, vmaf_cuda_picture_get_stream(ref_pic)));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->rb.host_pinned, (CUdeviceptr)s->rb.device->data,
                                              s->rb.bytes, s->lc.str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.finished, s->lc.str));
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. */
    int err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (err)
        return err;

    const double sse = (double)*(uint64_t *)s->rb.host_pinned;
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

    /* Lifecycle teardown via the template (sync → destroy stream →
     * destroy events). Best-effort error aggregation matches the
     * old hand-rolled CHECK_CUDA_GOTO chain. */
    int rc = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    int err = vmaf_cuda_kernel_readback_free(&s->rb, fex->cu_state);
    if (err && rc == 0)
        rc = err;
    err = vmaf_dictionary_free(&s->feature_name_dict);
    if (err && rc == 0)
        rc = err;
    return rc;
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
