/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  psnr_hvs feature extractor on the CUDA backend
 *  (T7-23 / ADR-0188 / ADR-0191, GPU long-tail batch 2 part 3b).
 *  CUDA twin of psnr_hvs_vulkan (PR #143).
 *
 *  Per-plane single-dispatch design — one CUDA block per output
 *  8×8 image block (step=7), 64 threads per block. Cooperative
 *  load + thread-0-serial reductions matching CPU's exact i,j
 *  summation order (same precision strategy as psnr_hvs.comp).
 *  Picture_copy host-side normalises uint sample → float in
 *  [0, 255] before D2H roundtrip with `cuMemcpy2DAsync` to honour
 *  the device pitch (matches the ms_ssim_cuda fix in PR #142).
 *
 *  3 dispatches per frame (Y, Cb, Cr). Combined `psnr_hvs =
 *  0.8·Y + 0.1·(Cb + Cr)` on the host.
 *
 *  Rejects YUV400P (no chroma) and `bpc > 12` (matches CPU).
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
#include "cuda/integer_psnr_hvs_cuda.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "picture_copy.h"
#include "cuda_helper.cuh"

#define PSNR_HVS_BLOCK 8
#define PSNR_HVS_STEP 7
#define PSNR_HVS_NUM_PLANES 3
#define PSNR_HVS_BLOCK_DIM 8

typedef struct PsnrHvsStateCuda {
    CUevent event;
    CUevent finished;
    CUfunction func_psnr_hvs;
    CUstream str;

    unsigned width[PSNR_HVS_NUM_PLANES];
    unsigned height[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_x[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_y[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks[PSNR_HVS_NUM_PLANES];
    unsigned bpc;
    int32_t samplemax_sq;

    /* Per-plane ref / dist float buffers (picture_copy normalised). */
    VmafCudaBuffer *d_ref[PSNR_HVS_NUM_PLANES];
    VmafCudaBuffer *d_dist[PSNR_HVS_NUM_PLANES];
    /* Per-plane block partials. */
    VmafCudaBuffer *d_partials[PSNR_HVS_NUM_PLANES];

    /* Pinned host staging. */
    float *h_ref[PSNR_HVS_NUM_PLANES];
    float *h_dist[PSNR_HVS_NUM_PLANES];
    float *h_partials[PSNR_HVS_NUM_PLANES];

    unsigned index;
    VmafDictionary *feature_name_dict;
} PsnrHvsStateCuda;

static const VmafOption options[] = {{0}};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    PsnrHvsStateCuda *s = fex->priv;

    if (bpc > 12) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_cuda: invalid bitdepth (%u); bpc must be ≤ 12\n",
                 bpc);
        return -EINVAL;
    }
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "psnr_hvs_cuda: YUV400P unsupported (psnr_hvs needs all 3 planes)\n");
        return -EINVAL;
    }
    if (w < (unsigned)PSNR_HVS_BLOCK || h < (unsigned)PSNR_HVS_BLOCK) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_cuda: input %ux%u smaller than 8×8 block\n", w, h);
        return -EINVAL;
    }

    s->bpc = bpc;
    const int32_t samplemax = (1 << bpc) - 1;
    s->samplemax_sq = samplemax * samplemax;

    s->width[0] = w;
    s->height[0] = h;
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        s->width[1] = s->width[2] = w >> 1;
        s->height[1] = s->height[2] = h >> 1;
        break;
    case VMAF_PIX_FMT_YUV422P:
        s->width[1] = s->width[2] = w >> 1;
        s->height[1] = s->height[2] = h;
        break;
    case VMAF_PIX_FMT_YUV444P:
        s->width[1] = s->width[2] = w;
        s->height[1] = s->height[2] = h;
        break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_cuda: unsupported pix_fmt\n");
        return -EINVAL;
    }

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        if (s->width[p] < (unsigned)PSNR_HVS_BLOCK || s->height[p] < (unsigned)PSNR_HVS_BLOCK) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "psnr_hvs_cuda: plane %d dims %ux%u smaller than 8×8 block\n", p, s->width[p],
                     s->height[p]);
            return -EINVAL;
        }
        s->num_blocks_x[p] = (s->width[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks_y[p] = (s->height[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks[p] = s->num_blocks_x[p] * s->num_blocks_y[p];
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
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, psnr_hvs_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_psnr_hvs, module, "psnr_hvs"), fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    int ret = 0;
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        const size_t plane_bytes = (size_t)s->width[p] * s->height[p] * sizeof(float);
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_ref[p], plane_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_dist[p], plane_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_partials[p], partials_bytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_ref[p], plane_bytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_dist[p], plane_bytes);
        ret |=
            vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_partials[p], partials_bytes);
    }
    if (ret)
        return -ENOMEM;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

/* picture_copy-style upload: read the device-side pic plane into
 * a contiguous host buffer (cuMemcpy2DAsync honours the device
 * pitch from cuMemAllocPitch — see PR #142 commit message for
 * the bug this avoids), normalise uint → float in [0, 255], and
 * upload back as float to s->d_ref[plane] / s->d_dist[plane]. */
static int upload_plane_cuda(PsnrHvsStateCuda *s, VmafFeatureExtractor *fex, VmafPicture *pic,
                             VmafCudaBuffer *dst_buf, float *h_buf, int plane)
{
    CudaFunctions *cu_f = fex->cu_state->f;
    const unsigned bpc_bytes = (s->bpc <= 8 ? 1u : 2u);
    const size_t input_bytes_uint = (size_t)s->width[plane] * s->height[plane] * bpc_bytes;
    void *tmp_uint = NULL;
    int ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, &tmp_uint, input_bytes_uint);
    if (ret)
        return -ENOMEM;
    CUstream stream = vmaf_cuda_picture_get_stream(pic);

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)pic->data[plane];
    m.srcPitch = (size_t)pic->stride[plane];
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstHost = tmp_uint;
    m.dstPitch = (size_t)s->width[plane] * bpc_bytes;
    m.WidthInBytes = (size_t)s->width[plane] * bpc_bytes;
    m.Height = s->height[plane];
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&m, stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(stream));

    /* Inline picture_copy-equivalent on the per-plane buffer
     * (matching the libvmaf picture_copy.c arithmetic exactly). */
    if (pic->bpc <= 8) {
        const uint8_t *src = (const uint8_t *)tmp_uint;
        for (unsigned y = 0; y < s->height[plane]; y++) {
            for (unsigned x = 0; x < s->width[plane]; x++) {
                h_buf[y * s->width[plane] + x] = (float)src[y * s->width[plane] + x];
            }
        }
    } else {
        const float scaler = (pic->bpc == 10) ? 4.0f :
                             (pic->bpc == 12) ? 16.0f :
                             (pic->bpc == 16) ? 256.0f :
                                                1.0f;
        const uint16_t *src = (const uint16_t *)tmp_uint;
        for (unsigned y = 0; y < s->height[plane]; y++) {
            for (unsigned x = 0; x < s->width[plane]; x++) {
                h_buf[y * s->width[plane] + x] = (float)src[y * s->width[plane] + x] / scaler;
            }
        }
    }
    (void)vmaf_cuda_buffer_host_free(fex->cu_state, tmp_uint);

    const size_t plane_bytes = (size_t)s->width[plane] * s->height[plane] * sizeof(float);
    CHECK_CUDA_RETURN(cu_f,
                      cuMemcpyHtoDAsync((CUdeviceptr)dst_buf->data, h_buf, plane_bytes, s->str));
    return 0;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrHvsStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    s->index = index;

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        int err = upload_plane_cuda(s, fex, ref_pic, s->d_ref[p], s->h_ref[p], p);
        if (err)
            return err;
        err = upload_plane_cuda(s, fex, dist_pic, s->d_dist[p], s->h_dist[p], p);
        if (err)
            return err;
    }

    /* Dispatch one kernel per plane. Per-plane PLANE + BPC are
     * passed as runtime kernel args (same kernel handles all 3
     * planes via the CSF_TABLES[plane] lookup). */
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        int plane_arg = p;
        int bpc_arg = (int)s->bpc;
        unsigned width = s->width[p];
        unsigned height = s->height[p];
        unsigned nbx = s->num_blocks_x[p];
        unsigned nby = s->num_blocks_y[p];
        void *params[] = {
            (void *)s->d_ref[p], (void *)s->d_dist[p], (void *)s->d_partials[p],
            (void *)&width,      (void *)&height,      (void *)&nbx,
            (void *)&nby,        (void *)&plane_arg,   (void *)&bpc_arg,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_psnr_hvs, nbx, nby, 1, PSNR_HVS_BLOCK_DIM,
                                               PSNR_HVS_BLOCK_DIM, 1, 0, s->str, params, NULL));
    }
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrHvsStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* D2H readback all 3 planes' partials, then sync. */
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->h_partials[p], (CUdeviceptr)s->d_partials[p]->data,
                                            partials_bytes, s->str));
    }
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));

    /* Per-plane reduction matching CPU's float `ret` register
     * semantics (see psnr_hvs_vulkan.c for the rationale). */
    double plane_score[PSNR_HVS_NUM_PLANES];
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        float ret = 0.0f;
        for (unsigned i = 0; i < s->num_blocks[p]; i++)
            ret += s->h_partials[p][i];
        const int pixels = (int)(s->num_blocks[p] * 64u);
        ret /= (float)pixels;
        ret /= (float)s->samplemax_sq;
        plane_score[p] = (double)ret;
    }

    int err = 0;
    static const char *plane_features[PSNR_HVS_NUM_PLANES] = {"psnr_hvs_y", "psnr_hvs_cb",
                                                              "psnr_hvs_cr"};
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        const double db = 10.0 * (-1.0 * log10(plane_score[p]));
        err |= vmaf_feature_collector_append(feature_collector, plane_features[p], db, index);
    }
    const double combined = 0.8 * plane_score[0] + 0.1 * (plane_score[1] + plane_score[2]);
    const double db_combined = 10.0 * (-1.0 * log10(combined));
    err |= vmaf_feature_collector_append(feature_collector, "psnr_hvs", db_combined, index);
    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    PsnrHvsStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    CHECK_CUDA_GOTO(cu_f, cuStreamSynchronize(s->str), after_stream_sync);
after_stream_sync:
    CHECK_CUDA_GOTO(cu_f, cuStreamDestroy(s->str), after_stream_destroy);
after_stream_destroy:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->event), after_event1);
after_event1:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->finished), after_event2);
after_event2:;

    int ret = _cuda_err;
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; p++) {
        if (s->d_ref[p]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->d_ref[p]);
            free(s->d_ref[p]);
        }
        if (s->d_dist[p]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->d_dist[p]);
            free(s->d_dist[p]);
        }
        if (s->d_partials[p]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->d_partials[p]);
            free(s->d_partials[p]);
        }
        if (s->h_ref[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_ref[p]);
        if (s->h_dist[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_dist[p]);
        if (s->h_partials[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_partials[p]);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr", "psnr_hvs",
                                          NULL};

VmafFeatureExtractor vmaf_fex_psnr_hvs_cuda = {
    .name = "psnr_hvs_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(PsnrHvsStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
