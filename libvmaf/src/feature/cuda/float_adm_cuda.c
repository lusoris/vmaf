/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_adm feature kernel on the CUDA backend (T7-23 / batch 3
 *  part 6b — ADR-0192 / ADR-0202). CUDA twin of float_adm_vulkan
 *  (PR #154 / ADR-0199). Same four pipeline stages, same `-1` mirror
 *  form, same fused stage 3 with cross-band CM threshold.
 *
 *  Per-frame flow: 16 launches (4 stages × 4 scales) + a pinned-host
 *  D2H copy of the per-scale (csf, cm) partial buffers. Reduction
 *  across WGs happens on the host in double precision — same trick
 *  as the Vulkan host wrapper, matches CPU adm_csf_den_scale_s /
 *  adm_cm_s row-by-row order to keep the places=4 contract.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cuda/float_adm_cuda.h"
#include "cuda/kernel_template.h"
#include "cuda_helper.cuh"
#include "picture.h"
#include "picture_cuda.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FADM_NUM_SCALES 4
#define FADM_NUM_BANDS 3
#define FADM_BX 16
#define FADM_BY 16
#define FADM_BORDER_FACTOR 0.1
#define FADM_ACCUM_SLOTS 6

typedef struct {
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    int adm_csf_mode;

    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned buf_stride;

    float rfactor[12];

    /* Stream + event pair owned by `cuda/kernel_template.h` lifecycle
     * (ADR-0221). Multi-stage DWT + CSF pipeline state stays outside
     * the template's single-pair readback bundle. */
    VmafCudaKernelLifecycle lc;
    CUfunction func_dwt_vert;
    CUfunction func_dwt_hori;
    CUfunction func_decouple_csf;
    CUfunction func_csf_cm;

    VmafCudaBuffer *src_ref;
    VmafCudaBuffer *src_dis;
    VmafCudaBuffer *dwt_tmp_ref;
    VmafCudaBuffer *dwt_tmp_dis;
    VmafCudaBuffer *ref_band[FADM_NUM_SCALES];
    VmafCudaBuffer *dis_band[FADM_NUM_SCALES];
    VmafCudaBuffer *csf_a;
    VmafCudaBuffer *csf_f;
    VmafCudaBuffer *accum[FADM_NUM_SCALES];
    float *accum_host[FADM_NUM_SCALES];

    unsigned wg_count[FADM_NUM_SCALES];
    unsigned scale_w[FADM_NUM_SCALES];
    unsigned scale_h[FADM_NUM_SCALES];
    unsigned scale_half_w[FADM_NUM_SCALES];
    unsigned scale_half_h[FADM_NUM_SCALES];

    VmafDictionary *feature_name_dict;
} FloatAdmStateCuda;

static const VmafOption options[] = {
    {.name = "debug",
     .help = "debug mode: enable additional output",
     .offset = offsetof(FloatAdmStateCuda, debug),
     .type = VMAF_OPT_TYPE_BOOL,
     .default_val.b = false},
    {.name = "adm_enhn_gain_limit",
     .alias = "egl",
     .help = "enhancement gain imposed on adm, must be >= 1.0",
     .offset = offsetof(FloatAdmStateCuda, adm_enhn_gain_limit),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 100.0,
     .min = 1.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_norm_view_dist",
     .alias = "nvd",
     .help = "normalized viewing distance",
     .offset = offsetof(FloatAdmStateCuda, adm_norm_view_dist),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 3.0,
     .min = 0.75,
     .max = 24.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_ref_display_height",
     .alias = "rdf",
     .help = "reference display height in pixels",
     .offset = offsetof(FloatAdmStateCuda, adm_ref_display_height),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 1080,
     .min = 1,
     .max = 4320,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_mode",
     .alias = "csf",
     .help = "contrast sensitivity function (mode 0 only on CUDA v1)",
     .offset = offsetof(FloatAdmStateCuda, adm_csf_mode),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 0,
     .min = 0,
     .max = 9,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {0}};

/* DB2/CDF-9-7 wavelet noise model — matches dwt_7_9_YCbCr_threshold[0]
 * (Y-plane row) in adm_tools.h. */
static const float fadm_dwt_basis_amp[6][4] = {
    {0.62171f, 0.67234f, 0.72709f, 0.67234f},     {0.34537f, 0.41317f, 0.49428f, 0.41317f},
    {0.18004f, 0.22727f, 0.28688f, 0.22727f},     {0.091401f, 0.11792f, 0.15214f, 0.11792f},
    {0.045943f, 0.059758f, 0.077727f, 0.059758f}, {0.023013f, 0.030018f, 0.039156f, 0.030018f},
};
static const float fadm_dwt_a_Y = 0.495f;
static const float fadm_dwt_k_Y = 0.466f;
static const float fadm_dwt_f0_Y = 0.401f;
static const float fadm_dwt_g_Y[4] = {1.501f, 1.0f, 0.534f, 1.0f};

static float fadm_dwt_quant_step(int lambda, int theta, double view_dist, int display_h)
{
    /* Bit-for-bit replica of dwt_quant_step in adm_tools.h. */
    const float r = (float)(view_dist * (double)display_h * M_PI / 180.0);
    const float temp = (float)log10(pow(2.0, (double)(lambda + 1)) * (double)fadm_dwt_f0_Y *
                                    (double)fadm_dwt_g_Y[theta] / (double)r);
    const float Q = (float)(2.0 * (double)fadm_dwt_a_Y *
                            pow(10.0, (double)fadm_dwt_k_Y * (double)temp * (double)temp) /
                            (double)fadm_dwt_basis_amp[lambda][theta]);
    return Q;
}

static void compute_per_scale_dims(FloatAdmStateCuda *s)
{
    unsigned cw = s->width;
    unsigned ch = s->height;
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const unsigned hw = (cw + 1u) / 2u;
        const unsigned hh = (ch + 1u) / 2u;
        s->scale_w[scale] = cw;
        s->scale_h[scale] = ch;
        s->scale_half_w[scale] = hw;
        s->scale_half_h[scale] = hh;
        cw = hw;
        ch = hh;
    }
    /* buf_stride sized to scale-0 half_w0 so a single stride works at
     * every scale (parent's stride read at scale s+1 still aligns
     * because the host-side buffer was allocated with this stride). */
    s->buf_stride = (s->scale_half_w[0] + 3u) & ~3u;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatAdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (s->adm_csf_mode != 0)
        return -EINVAL;

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    compute_per_scale_dims(s);

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const float f1 =
            fadm_dwt_quant_step(scale, 1, s->adm_norm_view_dist, s->adm_ref_display_height);
        const float f2 =
            fadm_dwt_quant_step(scale, 2, s->adm_norm_view_dist, s->adm_ref_display_height);
        s->rfactor[scale * 3 + 0] = 1.0f / f1;
        s->rfactor[scale * 3 + 1] = 1.0f / f1;
        s->rfactor[scale * 3 + 2] = 1.0f / f2;
    }

    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, float_adm_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_dwt_vert, module, "float_adm_dwt_vert"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_dwt_hori, module, "float_adm_dwt_hori"),
                    fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_decouple_csf, module, "float_adm_decouple_csf"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_csf_cm, module, "float_adm_csf_cm"), fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t raw_bytes = (size_t)w * h * bpp;
    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->src_ref, raw_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->src_dis, raw_bytes);

    /* DWT scratch sized at scale 0 (worst case). */
    const size_t dwt_bytes = (size_t)s->width * 2u * s->scale_half_h[0] * sizeof(float);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dwt_tmp_ref, dwt_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dwt_tmp_dis, dwt_bytes);

    /* Per-scale band buffers — 4 bands × buf_stride × half_h. The
     * scale-(s+1) DWT vert kernel reads scale-s's LL band, so each
     * scale needs its own ref_band/dis_band buffer. */
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const size_t band_bytes =
            (size_t)4u * s->buf_stride * s->scale_half_h[scale] * sizeof(float);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->ref_band[scale], band_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->dis_band[scale], band_bytes);
    }

    /* csf_a + csf_f reused per-scale (sized to scale 0 worst case). */
    const size_t csf_bytes =
        (size_t)FADM_NUM_BANDS * s->buf_stride * s->scale_half_h[0] * sizeof(float);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->csf_a, csf_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->csf_f, csf_bytes);

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const int hh = (int)s->scale_half_h[scale];
        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (top < 0)
            top = 0;
        const int bottom = hh - top;
        const unsigned num_rows = (bottom > top) ? (unsigned)(bottom - top) : 1u;
        const unsigned wg_count = 3u * num_rows;
        s->wg_count[scale] = wg_count;
        const size_t accum_bytes = (size_t)wg_count * FADM_ACCUM_SLOTS * sizeof(float);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->accum[scale], accum_bytes);
        ret |=
            vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->accum_host[scale], accum_bytes);
    }
    if (ret)
        goto free_buffers;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_buffers;
    return 0;

free_buffers:
    if (s->src_ref) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->src_ref);
        free(s->src_ref);
    }
    if (s->src_dis) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->src_dis);
        free(s->src_dis);
    }
    if (s->dwt_tmp_ref) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->dwt_tmp_ref);
        free(s->dwt_tmp_ref);
    }
    if (s->dwt_tmp_dis) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->dwt_tmp_dis);
        free(s->dwt_tmp_dis);
    }
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->ref_band[scale]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->ref_band[scale]);
            free(s->ref_band[scale]);
        }
        if (s->dis_band[scale]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->dis_band[scale]);
            free(s->dis_band[scale]);
        }
    }
    if (s->csf_a) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->csf_a);
        free(s->csf_a);
    }
    if (s->csf_f) {
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->csf_f);
        free(s->csf_f);
    }
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->accum[scale]) {
            (void)vmaf_cuda_buffer_free(fex->cu_state, s->accum[scale]);
            free(s->accum[scale]);
        }
    }
    (void)vmaf_dictionary_free(&s->feature_name_dict);
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
    (void)index;
    FloatAdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t raw_stride = (ptrdiff_t)(s->width * bpp);

    CUstream pic_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CHECK_CUDA_RETURN(cu_f,
                      cuStreamWaitEvent(pic_stream, vmaf_cuda_picture_get_ready_event(dist_pic),
                                        CU_EVENT_WAIT_DEFAULT));

    CUDA_MEMCPY2D cpy = {0};
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice = (CUdeviceptr)ref_pic->data[0];
    cpy.srcPitch = ref_pic->stride[0];
    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstDevice = (CUdeviceptr)s->src_ref->data;
    cpy.dstPitch = raw_stride;
    cpy.WidthInBytes = raw_stride;
    cpy.Height = s->height;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&cpy, pic_stream));

    CUDA_MEMCPY2D cpy_d = cpy;
    cpy_d.srcDevice = (CUdeviceptr)dist_pic->data[0];
    cpy_d.srcPitch = dist_pic->stride[0];
    cpy_d.dstDevice = (CUdeviceptr)s->src_dis->data;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&cpy_d, pic_stream));

    /* Reset accumulator buffers — the kernel only writes 2/6 slots
     * per WG (its band's csf + cm); the others must stay zero. */
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        CHECK_CUDA_RETURN(
            cu_f, cuMemsetD8Async(s->accum[scale]->data, 0,
                                  (size_t)s->wg_count[scale] * FADM_ACCUM_SLOTS * sizeof(float),
                                  pic_stream));
    }

    float scaler = 1.0f;
    float pixel_offset = -128.0f;
    if (s->bpc == 10u)
        scaler = 4.0f;
    else if (s->bpc == 12u)
        scaler = 16.0f;
    else if (s->bpc == 16u)
        scaler = 256.0f;

    CUdeviceptr ref_raw_d = (CUdeviceptr)s->src_ref->data;
    CUdeviceptr dis_raw_d = (CUdeviceptr)s->src_dis->data;
    CUdeviceptr dwt_ref_d = (CUdeviceptr)s->dwt_tmp_ref->data;
    CUdeviceptr dwt_dis_d = (CUdeviceptr)s->dwt_tmp_dis->data;
    CUdeviceptr csf_a_d = (CUdeviceptr)s->csf_a->data;
    CUdeviceptr csf_f_d = (CUdeviceptr)s->csf_f->data;

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const int cur_w = (int)s->scale_w[scale];
        const int cur_h = (int)s->scale_h[scale];
        const int half_w = (int)s->scale_half_w[scale];
        const int half_h = (int)s->scale_half_h[scale];

        /* Parent LL band dimensions = scale_w/h[scale] (per
         * `compute_per_scale_dims`: scale_w[s] is the *input* dim at
         * scale s, which equals the parent's LL output dim).
         * Mirror reads in stage 0 must clamp against these, NOT
         * scale_w[scale-1] (full parent image dims).  This matches
         * the Vulkan kernel's `read_band_a_at`, which uses
         * `pc.cur_w/cur_h` = the same thing. */
        const int parent_w = (scale > 0) ? (int)s->scale_w[scale] : 0;
        const int parent_h = (scale > 0) ? (int)s->scale_h[scale] : 0;
        const int parent_half_h = (scale > 0) ? (int)s->scale_half_h[scale - 1] : 0;
        const int parent_buf_stride = (int)s->buf_stride;

        CUdeviceptr ref_band_d = (CUdeviceptr)s->ref_band[scale]->data;
        CUdeviceptr dis_band_d = (CUdeviceptr)s->dis_band[scale]->data;
        CUdeviceptr parent_ref_band_d =
            (scale > 0) ? (CUdeviceptr)s->ref_band[scale - 1]->data : (CUdeviceptr)0;
        CUdeviceptr parent_dis_band_d =
            (scale > 0) ? (CUdeviceptr)s->dis_band[scale - 1]->data : (CUdeviceptr)0;

        int top = (int)((double)half_h * FADM_BORDER_FACTOR - 0.5);
        int left = (int)((double)half_w * FADM_BORDER_FACTOR - 0.5);
        if (top < 0)
            top = 0;
        if (left < 0)
            left = 0;
        const int bottom = half_h - top;
        const int right = half_w - left;
        const int active_h = bottom - top;

        const float rfactor_h = s->rfactor[scale * 3 + 0];
        const float rfactor_v = s->rfactor[scale * 3 + 1];
        const float rfactor_d = s->rfactor[scale * 3 + 2];
        const float gain_limit = (float)s->adm_enhn_gain_limit;

        /* Stage 0 — DWT vertical (z=2 fused ref+dis). */
        {
            const unsigned gx = ((unsigned)cur_w + FADM_BX - 1u) / FADM_BX;
            const unsigned gy = ((unsigned)half_h + FADM_BY - 1u) / FADM_BY;
            int scale_arg = scale;
            int half_h_arg = half_h;
            int parent_half_h_arg = parent_half_h;
            int parent_buf_stride_arg = parent_buf_stride;
            int parent_w_arg = parent_w;
            int parent_h_arg = parent_h;
            int cur_w_arg = cur_w;
            int cur_h_arg = cur_h;
            unsigned bpc_arg = s->bpc;
            float scaler_arg = scaler;
            float pixel_offset_arg = pixel_offset;
            void *args[] = {&scale_arg,
                            &ref_raw_d,
                            &dis_raw_d,
                            (void *)&raw_stride,
                            &parent_ref_band_d,
                            &parent_dis_band_d,
                            &parent_buf_stride_arg,
                            &parent_half_h_arg,
                            &parent_w_arg,
                            &parent_h_arg,
                            &dwt_ref_d,
                            &dwt_dis_d,
                            &cur_w_arg,
                            &cur_h_arg,
                            &half_h_arg,
                            &bpc_arg,
                            &scaler_arg,
                            &pixel_offset_arg};
            CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_dwt_vert, gx, gy, 2, FADM_BX, FADM_BY, 1,
                                                   0, pic_stream, args, NULL));
        }

        /* Stage 1 — DWT horizontal. */
        {
            const unsigned gx = ((unsigned)half_w + FADM_BX - 1u) / FADM_BX;
            const unsigned gy = ((unsigned)half_h + FADM_BY - 1u) / FADM_BY;
            int scale_arg = scale;
            int cur_w_arg = cur_w;
            int half_w_arg = half_w;
            int half_h_arg = half_h;
            int buf_stride_arg = (int)s->buf_stride;
            void *args[] = {&scale_arg, &dwt_ref_d,  &dwt_dis_d,  &ref_band_d,    &dis_band_d,
                            &cur_w_arg, &half_w_arg, &half_h_arg, &buf_stride_arg};
            CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_dwt_hori, gx, gy, 2, FADM_BX, FADM_BY, 1,
                                                   0, pic_stream, args, NULL));
        }

        /* Stage 2 — Decouple + CSF. */
        {
            const unsigned gx = ((unsigned)half_w + FADM_BX - 1u) / FADM_BX;
            const unsigned gy = ((unsigned)half_h + FADM_BY - 1u) / FADM_BY;
            int half_w_arg = half_w;
            int half_h_arg = half_h;
            int buf_stride_arg = (int)s->buf_stride;
            float rfh = rfactor_h;
            float rfv = rfactor_v;
            float rfd = rfactor_d;
            float gl = gain_limit;
            void *args[] = {&ref_band_d, &dis_band_d,     &csf_a_d, &csf_f_d, &half_w_arg,
                            &half_h_arg, &buf_stride_arg, &rfh,     &rfv,     &rfd,
                            &gl};
            CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_decouple_csf, gx, gy, 1, FADM_BX,
                                                   FADM_BY, 1, 0, pic_stream, args, NULL));
        }

        /* Stage 3 — CSF denominator + CM fused (1D dispatch over 3
         * bands × num_active_rows). */
        {
            const unsigned num_rows = (unsigned)(active_h > 0 ? active_h : 1);
            const unsigned gx = 3u * num_rows;
            int half_w_arg = half_w;
            int half_h_arg = half_h;
            int buf_stride_arg = (int)s->buf_stride;
            int active_left_arg = left;
            int active_top_arg = top;
            int active_right_arg = right;
            int active_bottom_arg = bottom;
            float rfh = rfactor_h;
            float rfv = rfactor_v;
            float rfd = rfactor_d;
            float gl = gain_limit;
            CUdeviceptr accum_d = (CUdeviceptr)s->accum[scale]->data;
            void *args[] = {&ref_band_d,
                            &dis_band_d,
                            &csf_a_d,
                            &csf_f_d,
                            &accum_d,
                            &half_w_arg,
                            &half_h_arg,
                            &buf_stride_arg,
                            &active_left_arg,
                            &active_top_arg,
                            &active_right_arg,
                            &active_bottom_arg,
                            &rfh,
                            &rfv,
                            &rfd,
                            &gl};
            CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_csf_cm, gx, 1u, 1u, FADM_BX, FADM_BY, 1,
                                                   0, pic_stream, args, NULL));
        }
    }

    /* Sync over to the secondary stream + D2H copy partials. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, pic_stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        CHECK_CUDA_RETURN(
            cu_f, cuMemcpyDtoHAsync(s->accum_host[scale], (CUdeviceptr)s->accum[scale]->data,
                                    (size_t)s->wg_count[scale] * FADM_ACCUM_SLOTS * sizeof(float),
                                    s->lc.str));
    }
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.finished, s->lc.str));
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index, VmafFeatureCollector *fc)
{
    FloatAdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->lc.str));

    /* Per-scale double accumulation across WGs, mirroring the Vulkan
     * host wrapper's reduce_and_emit. */
    double cm_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {{0.0}};
    double csf_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {{0.0}};
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const float *slots = s->accum_host[scale];
        const unsigned wg_count = s->wg_count[scale];
        for (unsigned wg = 0u; wg < wg_count; wg++) {
            const float *p = slots + (size_t)wg * FADM_ACCUM_SLOTS;
            for (int b = 0; b < FADM_NUM_BANDS; b++) {
                csf_totals[scale][b] += (double)p[b];
                cm_totals[scale][b] += (double)p[3 + b];
            }
        }
    }

    double score_num = 0.0;
    double score_den = 0.0;
    double scores[8];
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const int hw = (int)s->scale_half_w[scale];
        const int hh = (int)s->scale_half_h[scale];
        int left = (int)((double)hw * FADM_BORDER_FACTOR - 0.5);
        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        const int right = hw - left;
        const int bottom = hh - top;
        const float area_cbrt = powf((float)((bottom - top) * (right - left)) / 32.0f, 1.0f / 3.0f);
        float num_scale = 0.0f;
        float den_scale = 0.0f;
        for (int b = 0; b < FADM_NUM_BANDS; b++) {
            num_scale += powf((float)cm_totals[scale][b], 1.0f / 3.0f) + area_cbrt;
            den_scale += powf((float)csf_totals[scale][b], 1.0f / 3.0f) + area_cbrt;
        }
        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
        score_num += num_scale;
        score_den += den_scale;
    }

    /* numden_limit per ADM_OPT_SINGLE_PRECISION (matches adm.c L88). */
    const int w = (int)s->scale_w[0];
    const int h = (int)s->scale_h[0];
    const double numden_limit = 1e-2 * (double)(w * h) / (1920.0 * 1080.0);
    if (score_num < numden_limit)
        score_num = 0.0;
    if (score_den < numden_limit)
        score_den = 0.0;
    const double score = (score_den == 0.0) ? 1.0 : score_num / score_den;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                   "VMAF_feature_adm2_score", score, index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale0_score", scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale1_score", scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale2_score", scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale3_score", scores[6] / scores[7], index);

    if (s->debug && !err) {
        err |=
            vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm", score, index);
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm_num",
                                                       score_num, index);
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm_den",
                                                       score_den, index);
        const char *names[8] = {"adm_num_scale0", "adm_den_scale0", "adm_num_scale1",
                                "adm_den_scale1", "adm_num_scale2", "adm_den_scale2",
                                "adm_num_scale3", "adm_den_scale3"};
        for (int i = 0; i < 8 && !err; i++) {
            err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, names[i],
                                                           scores[i], index);
        }
    }
    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    FloatAdmStateCuda *s = fex->priv;
    int ret = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    if (s->src_ref) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->src_ref);
        free(s->src_ref);
    }
    if (s->src_dis) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->src_dis);
        free(s->src_dis);
    }
    if (s->dwt_tmp_ref) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->dwt_tmp_ref);
        free(s->dwt_tmp_ref);
    }
    if (s->dwt_tmp_dis) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->dwt_tmp_dis);
        free(s->dwt_tmp_dis);
    }
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->ref_band[scale]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->ref_band[scale]);
            free(s->ref_band[scale]);
        }
        if (s->dis_band[scale]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->dis_band[scale]);
            free(s->dis_band[scale]);
        }
    }
    if (s->csf_a) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->csf_a);
        free(s->csf_a);
    }
    if (s->csf_f) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->csf_f);
        free(s->csf_f);
    }
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->accum[scale]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->accum[scale]);
            free(s->accum[scale]);
        }
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

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

VmafFeatureExtractor vmaf_fex_float_adm_cuda = {
    .name = "float_adm_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(FloatAdmStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};
