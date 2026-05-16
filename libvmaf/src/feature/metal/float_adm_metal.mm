/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_adm feature extractor — Metal backend (T8-2a / ADR-0424).
 *  Direct port of the CUDA twin (feature/cuda/float_adm_cuda.c,
 *  ADR-0192 / ADR-0202). Same four pipeline stages, same 16-launch
 *  per-frame loop (4 stages × 4 scales), same host-side double-
 *  precision accumulation.
 *
 *  Pipeline stages (one kernel function each):
 *    0 — float_adm_dwt_vert   (grid: cur_w/16 × half_h/16 × 2)
 *    1 — float_adm_dwt_hori   (grid: half_w/16 × half_h/16 × 2)
 *    2 — float_adm_decouple_csf (grid: half_w/16 × half_h/16 × 1)
 *    3 — float_adm_csf_cm     (grid: 3*num_active_rows × 1 × 1,
 *                               threadgroup: 256 × 1 × 1)
 *
 *  On Apple Silicon (unified memory) the per-scale accum buffers are
 *  MTLResourceStorageModeShared so the host can read [contents]
 *  directly without a blit copy.
 *
 *  Numerical invariants: same places=4 contract as the CUDA twin
 *  (see float_adm.metal for the kernel-level invariants).
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" {
#include "dict.h"
#include "feature/adm_options.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../metal/common.h"
#include "../../metal/kernel_template.h"
}

extern "C" {
extern const unsigned char libvmaf_metallib_start[] __asm("section$start$__TEXT$__metallib");
extern const unsigned char libvmaf_metallib_end[]   __asm("section$end$__TEXT$__metallib");
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FADM_NUM_SCALES  4
#define FADM_NUM_BANDS   3
#define FADM_ACCUM_SLOTS 6
#define FADM_BORDER_FACTOR 0.1
#define FADM_BX 16
#define FADM_BY 16
#define FADM_STAGE3_TG 256

/* ------------------------------------------------------------------ */
/*  DWT quant-step table (same constants as float_adm_cuda.c)          */
/* ------------------------------------------------------------------ */

static const float fadm_dwt_basis_amp[6][4] = {
    {0.62171f, 0.67234f, 0.72709f, 0.67234f},
    {0.34537f, 0.41317f, 0.49428f, 0.41317f},
    {0.18004f, 0.22727f, 0.28688f, 0.22727f},
    {0.091401f, 0.11792f, 0.15214f, 0.11792f},
    {0.045943f, 0.059758f, 0.077727f, 0.059758f},
    {0.023013f, 0.030018f, 0.039156f, 0.030018f},
};
static const float fadm_dwt_a_Y  = 0.495f;
static const float fadm_dwt_k_Y  = 0.466f;
static const float fadm_dwt_f0_Y = 0.401f;
static const float fadm_dwt_g_Y[4] = {1.501f, 1.0f, 0.534f, 1.0f};

static float fadm_dwt_quant_step(int lambda, int theta, double view_dist, int display_h)
{
    const float r = (float)(view_dist * (double)display_h * M_PI / 180.0);
    const float temp = (float)log10(pow(2.0, (double)(lambda + 1)) *
                                    (double)fadm_dwt_f0_Y *
                                    (double)fadm_dwt_g_Y[theta] / (double)r);
    const float Q = (float)(2.0 * (double)fadm_dwt_a_Y *
                            pow(10.0, (double)fadm_dwt_k_Y * (double)temp * (double)temp) /
                            (double)fadm_dwt_basis_amp[lambda][theta]);
    return Q;
}

/* ------------------------------------------------------------------ */
/*  State struct                                                        */
/* ------------------------------------------------------------------ */

typedef struct FloatAdmStateMetal {
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int    adm_ref_display_height;
    int    adm_csf_mode;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_noise_weight;

    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned buf_stride;

    float rfactor[12];   /* 4 scales × 3 bands (h, v, d) */

    VmafMetalContext *ctx;

    /* Four pipeline state objects, one per stage. */
    void *pso_dwt_vert;       /* float_adm_dwt_vert    */
    void *pso_dwt_hori;       /* float_adm_dwt_hori    */
    void *pso_decouple_csf;   /* float_adm_decouple_csf */
    void *pso_csf_cm;         /* float_adm_csf_cm      */

    /* Raw source buffers (packed, stride = width * bpp). */
    void *src_ref_buf;   /* id<MTLBuffer> */
    void *src_dis_buf;   /* id<MTLBuffer> */

    /* DWT scratch: 2 × (cur_w * 2) × half_h floats (ref+dis separate). */
    void *dwt_tmp_ref_buf;
    void *dwt_tmp_dis_buf;

    /* Per-scale ref/dis band buffers: 4 bands × buf_stride × half_h. */
    void *ref_band_buf[FADM_NUM_SCALES];
    void *dis_band_buf[FADM_NUM_SCALES];

    /* Per-scale CSF intermediate buffers (3 bands × buf_stride × half_h). */
    void *csf_a_buf;
    void *csf_f_buf;

    /* Per-scale accumulator buffers (wg_count × 6 floats, Shared). */
    void *accum_buf[FADM_NUM_SCALES];

    unsigned wg_count[FADM_NUM_SCALES];
    unsigned scale_w[FADM_NUM_SCALES];
    unsigned scale_h[FADM_NUM_SCALES];
    unsigned scale_half_w[FADM_NUM_SCALES];
    unsigned scale_half_h[FADM_NUM_SCALES];

    VmafDictionary *feature_name_dict;
} FloatAdmStateMetal;

/* ------------------------------------------------------------------ */
/*  Options                                                             */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {
    {.name = "debug",
     .help = "debug mode: enable additional output",
     .offset = offsetof(FloatAdmStateMetal, debug),
     .type = VMAF_OPT_TYPE_BOOL,
     .default_val.b = false},
    {.name = "adm_enhn_gain_limit",
     .alias = "egl",
     .help = "enhancement gain imposed on adm, must be >= 1.0",
     .offset = offsetof(FloatAdmStateMetal, adm_enhn_gain_limit),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 100.0,
     .min = 1.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_norm_view_dist",
     .alias = "nvd",
     .help = "normalized viewing distance",
     .offset = offsetof(FloatAdmStateMetal, adm_norm_view_dist),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 3.0,
     .min = 0.75,
     .max = 24.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_ref_display_height",
     .alias = "rdf",
     .help = "reference display height in pixels",
     .offset = offsetof(FloatAdmStateMetal, adm_ref_display_height),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 1080,
     .min = 1,
     .max = 4320,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_mode",
     .alias = "csf",
     .help = "contrast sensitivity function (mode 0 only on Metal v1)",
     .offset = offsetof(FloatAdmStateMetal, adm_csf_mode),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 0,
     .min = 0,
     .max = 9,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_scale",
     .alias = "cs",
     .help = "CSF band-scale multiplier for h/v bands (default 1.0)",
     .offset = offsetof(FloatAdmStateMetal, adm_csf_scale),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = DEFAULT_ADM_CSF_SCALE,
     .min = 0.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_diag_scale",
     .alias = "cds",
     .help = "CSF band-scale multiplier for diagonal bands (default 1.0)",
     .offset = offsetof(FloatAdmStateMetal, adm_csf_diag_scale),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
     .min = 0.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_noise_weight",
     .alias = "nw",
     .help = "noise floor weight for CM numerator (default 0.03125 = 1/32)",
     .offset = offsetof(FloatAdmStateMetal, adm_noise_weight),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
     .min = 0.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {0}};

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */

static void compute_per_scale_dims(FloatAdmStateMetal *s)
{
    unsigned cw = s->width;
    unsigned ch = s->height;
    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        const unsigned hw = (cw + 1u) / 2u;
        const unsigned hh = (ch + 1u) / 2u;
        s->scale_w[sc]      = cw;
        s->scale_h[sc]      = ch;
        s->scale_half_w[sc] = hw;
        s->scale_half_h[sc] = hh;
        cw = hw;
        ch = hh;
    }
    s->buf_stride = (s->scale_half_w[0] + 3u) & ~3u;
}

static int build_pipelines(FloatAdmStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size =
        (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn_vert = [lib newFunctionWithName:@"float_adm_dwt_vert"];
    id<MTLFunction> fn_hori = [lib newFunctionWithName:@"float_adm_dwt_hori"];
    id<MTLFunction> fn_csf  = [lib newFunctionWithName:@"float_adm_decouple_csf"];
    id<MTLFunction> fn_cm   = [lib newFunctionWithName:@"float_adm_csf_cm"];
    if (fn_vert == nil || fn_hori == nil || fn_csf == nil || fn_cm == nil) {
        return -ENODEV;
    }

    id<MTLComputePipelineState> pso_v =
        [device newComputePipelineStateWithFunction:fn_vert error:&err];
    id<MTLComputePipelineState> pso_h =
        [device newComputePipelineStateWithFunction:fn_hori error:&err];
    id<MTLComputePipelineState> pso_c =
        [device newComputePipelineStateWithFunction:fn_csf  error:&err];
    id<MTLComputePipelineState> pso_m =
        [device newComputePipelineStateWithFunction:fn_cm   error:&err];
    if (pso_v == nil || pso_h == nil || pso_c == nil || pso_m == nil) {
        return -ENODEV;
    }

    s->pso_dwt_vert     = (__bridge_retained void *)pso_v;
    s->pso_dwt_hori     = (__bridge_retained void *)pso_h;
    s->pso_decouple_csf = (__bridge_retained void *)pso_c;
    s->pso_csf_cm       = (__bridge_retained void *)pso_m;
    return 0;
}

static void release_buffer(void **buf_ptr)
{
    if (*buf_ptr != NULL) {
        (void)(__bridge_transfer id<MTLBuffer>)*buf_ptr;
        *buf_ptr = NULL;
    }
}

static void release_pso(void **pso_ptr)
{
    if (*pso_ptr != NULL) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)*pso_ptr;
        *pso_ptr = NULL;
    }
}

static id<MTLBuffer> new_shared_buf(id<MTLDevice> dev, size_t bytes)
{
    return [dev newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
}

/* ------------------------------------------------------------------ */
/*  init                                                                */
/* ------------------------------------------------------------------ */

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatAdmStateMetal *s = (FloatAdmStateMetal *)fex->priv;

    if (s->adm_csf_mode != 0) { return -EINVAL; }

    s->width  = w;
    s->height = h;
    s->bpc    = bpc;
    compute_per_scale_dims(s);

    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        const float f1 = fadm_dwt_quant_step(
            sc, 1, s->adm_norm_view_dist, s->adm_ref_display_height);
        const float f2 = fadm_dwt_quant_step(
            sc, 2, s->adm_norm_view_dist, s->adm_ref_display_height);
        s->rfactor[sc * 3 + 0] = (float)s->adm_csf_scale      / f1;
        s->rfactor[sc * 3 + 1] = (float)s->adm_csf_scale      / f1;
        s->rfactor[sc * 3 + 2] = (float)s->adm_csf_diag_scale / f2;
    }

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    if (dh == NULL) { err = -ENODEV; goto fail_ctx; }
    id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

    err = build_pipelines(s, device);
    if (err != 0) { goto fail_ctx; }

    {
        const size_t bpp      = (bpc <= 8u) ? 1u : 2u;
        const size_t raw_bytes = (size_t)w * h * bpp;

        s->src_ref_buf = (__bridge_retained void *)new_shared_buf(device, raw_bytes);
        s->src_dis_buf = (__bridge_retained void *)new_shared_buf(device, raw_bytes);
        if (s->src_ref_buf == NULL || s->src_dis_buf == NULL) {
            err = -ENOMEM; goto fail_pso;
        }

        /* DWT scratch: (cur_w * 2) × half_h × sizeof(float).  Scale-0
         * half_w is the widest; size on that. */
        const size_t dwt_bytes =
            (size_t)s->scale_w[0] * 2u * s->scale_half_h[0] * sizeof(float);
        s->dwt_tmp_ref_buf = (__bridge_retained void *)new_shared_buf(device, dwt_bytes);
        s->dwt_tmp_dis_buf = (__bridge_retained void *)new_shared_buf(device, dwt_bytes);
        if (s->dwt_tmp_ref_buf == NULL || s->dwt_tmp_dis_buf == NULL) {
            err = -ENOMEM; goto fail_raw;
        }

        for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
            const size_t band_bytes =
                (size_t)4u * s->buf_stride * s->scale_half_h[sc] * sizeof(float);
            s->ref_band_buf[sc] = (__bridge_retained void *)new_shared_buf(device, band_bytes);
            s->dis_band_buf[sc] = (__bridge_retained void *)new_shared_buf(device, band_bytes);
            if (s->ref_band_buf[sc] == NULL || s->dis_band_buf[sc] == NULL) {
                err = -ENOMEM; goto fail_dwt;
            }
        }

        const size_t csf_bytes =
            (size_t)FADM_NUM_BANDS * s->buf_stride * s->scale_half_h[0] * sizeof(float);
        s->csf_a_buf = (__bridge_retained void *)new_shared_buf(device, csf_bytes);
        s->csf_f_buf = (__bridge_retained void *)new_shared_buf(device, csf_bytes);
        if (s->csf_a_buf == NULL || s->csf_f_buf == NULL) {
            err = -ENOMEM; goto fail_bands;
        }

        for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
            const int hh    = (int)s->scale_half_h[sc];
            int top         = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
            if (top < 0) { top = 0; }
            const int bottom    = hh - top;
            const unsigned nrows = (bottom > top) ? (unsigned)(bottom - top) : 1u;
            s->wg_count[sc]      = 3u * nrows;
            const size_t accum_bytes =
                (size_t)s->wg_count[sc] * FADM_ACCUM_SLOTS * sizeof(float);
            s->accum_buf[sc] =
                (__bridge_retained void *)new_shared_buf(device, accum_bytes);
            if (s->accum_buf[sc] == NULL) {
                err = -ENOMEM; goto fail_csf;
            }
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_accum; }
    return 0;

fail_accum:
    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) { release_buffer(&s->accum_buf[sc]); }
fail_csf:
    release_buffer(&s->csf_a_buf);
    release_buffer(&s->csf_f_buf);
fail_bands:
    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        release_buffer(&s->ref_band_buf[sc]);
        release_buffer(&s->dis_band_buf[sc]);
    }
fail_dwt:
    release_buffer(&s->dwt_tmp_ref_buf);
    release_buffer(&s->dwt_tmp_dis_buf);
fail_raw:
    release_buffer(&s->src_ref_buf);
    release_buffer(&s->src_dis_buf);
fail_pso:
    release_pso(&s->pso_dwt_vert);
    release_pso(&s->pso_dwt_hori);
    release_pso(&s->pso_decouple_csf);
    release_pso(&s->pso_csf_cm);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* ------------------------------------------------------------------ */
/*  submit                                                              */
/* ------------------------------------------------------------------ */

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    FloatAdmStateMetal *s = (FloatAdmStateMetal *)fex->priv;

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>      device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)qh;

    /* Upload pixels into shared MTLBuffers row-by-row. */
    const size_t bpp        = (s->bpc <= 8u) ? 1u : 2u;
    const size_t raw_stride = (size_t)s->width * bpp;

    {
        uint8_t *rdst = (uint8_t *)[(id<MTLBuffer>)(__bridge id<MTLBuffer>)s->src_ref_buf contents];
        uint8_t *ddst = (uint8_t *)[(id<MTLBuffer>)(__bridge id<MTLBuffer>)s->src_dis_buf contents];
        for (unsigned y = 0; y < s->height; y++) {
            const uint8_t *rs = (const uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0];
            const uint8_t *ds = (const uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0];
            memcpy(rdst + y * raw_stride, rs, raw_stride);
            memcpy(ddst + y * raw_stride, ds, raw_stride);
        }
    }

    float scaler       = 1.0f;
    float pixel_offset = -128.0f;
    if (s->bpc == 10u)      { scaler = 4.0f; }
    else if (s->bpc == 12u) { scaler = 16.0f; }
    else if (s->bpc == 16u) { scaler = 256.0f; }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    /* Zero accumulator buffers via blit. */
    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
            id<MTLBuffer> ab = (__bridge id<MTLBuffer>)s->accum_buf[sc];
            const size_t ab_bytes = (size_t)s->wg_count[sc] * FADM_ACCUM_SLOTS * sizeof(float);
            [blit fillBuffer:ab range:NSMakeRange(0, ab_bytes) value:0];
        }
        [blit endEncoding];
    }

    /* Per-scale pipeline: 4 stages. */
    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        const int cur_w  = (int)s->scale_w[sc];
        const int cur_h  = (int)s->scale_h[sc];
        const int half_w = (int)s->scale_half_w[sc];
        const int half_h = (int)s->scale_half_h[sc];

        const int par_w      = (sc > 0) ? (int)s->scale_w[sc]         : 0;
        const int par_h      = (sc > 0) ? (int)s->scale_h[sc]         : 0;
        const int par_half_h = (sc > 0) ? (int)s->scale_half_h[sc-1]  : 0;
        const int par_stride = (int)s->buf_stride;

        id<MTLBuffer> ref_band_b = (__bridge id<MTLBuffer>)s->ref_band_buf[sc];
        id<MTLBuffer> dis_band_b = (__bridge id<MTLBuffer>)s->dis_band_buf[sc];
        id<MTLBuffer> par_ref_b  = (sc > 0) ?
            (__bridge id<MTLBuffer>)s->ref_band_buf[sc - 1] : nil;
        id<MTLBuffer> par_dis_b  = (sc > 0) ?
            (__bridge id<MTLBuffer>)s->dis_band_buf[sc - 1] : nil;
        id<MTLBuffer> src_ref_b  = (__bridge id<MTLBuffer>)s->src_ref_buf;
        id<MTLBuffer> src_dis_b  = (__bridge id<MTLBuffer>)s->src_dis_buf;
        id<MTLBuffer> tmp_ref_b  = (__bridge id<MTLBuffer>)s->dwt_tmp_ref_buf;
        id<MTLBuffer> tmp_dis_b  = (__bridge id<MTLBuffer>)s->dwt_tmp_dis_buf;
        id<MTLBuffer> csf_a_b    = (__bridge id<MTLBuffer>)s->csf_a_buf;
        id<MTLBuffer> csf_f_b    = (__bridge id<MTLBuffer>)s->csf_f_buf;
        id<MTLBuffer> accum_b    = (__bridge id<MTLBuffer>)s->accum_buf[sc];

        const float rfh  = s->rfactor[sc * 3 + 0];
        const float rfv  = s->rfactor[sc * 3 + 1];
        const float rfd  = s->rfactor[sc * 3 + 2];
        const float gl   = (float)s->adm_enhn_gain_limit;

        int top  = (int)((double)half_h * FADM_BORDER_FACTOR - 0.5);
        int left = (int)((double)half_w * FADM_BORDER_FACTOR - 0.5);
        if (top  < 0) { top  = 0; }
        if (left < 0) { left = 0; }
        const int bottom   = half_h - top;
        const int right    = half_w - left;
        const int active_h = bottom - top;

        /* ---- Stage 0: DWT vertical ---- */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:
                (__bridge id<MTLComputePipelineState>)s->pso_dwt_vert];

            uint32_t p_u[9] = {
                (uint32_t)sc,      (uint32_t)cur_w,      (uint32_t)cur_h,
                (uint32_t)half_h,  (uint32_t)s->bpc,
                (uint32_t)par_stride, (uint32_t)par_half_h,
                (uint32_t)par_w,   (uint32_t)par_h
            };
            float p_f[2] = {scaler, pixel_offset};
            [enc setBytes:p_u length:sizeof(p_u) atIndex:0];
            [enc setBytes:p_f length:sizeof(p_f) atIndex:1];
            [enc setBuffer:src_ref_b offset:0 atIndex:2];
            [enc setBuffer:src_dis_b offset:0 atIndex:3];
            uint32_t raw_s = (uint32_t)raw_stride;
            [enc setBytes:&raw_s length:sizeof(raw_s) atIndex:4];
            /* Null buffers at scale 0 — kernel gates on `scale==0`. */
            if (par_ref_b != nil) {
                [enc setBuffer:par_ref_b offset:0 atIndex:5];
                [enc setBuffer:par_dis_b offset:0 atIndex:6];
            } else {
                /* Bind the src buffers as placeholders; kernel ignores them. */
                [enc setBuffer:src_ref_b offset:0 atIndex:5];
                [enc setBuffer:src_dis_b offset:0 atIndex:6];
            }
            [enc setBuffer:tmp_ref_b offset:0 atIndex:7];
            [enc setBuffer:tmp_dis_b offset:0 atIndex:8];

            MTLSize tg   = MTLSizeMake(FADM_BX, FADM_BY, 1);
            MTLSize grid = MTLSizeMake(
                ((NSUInteger)cur_w  + FADM_BX - 1) / FADM_BX,
                ((NSUInteger)half_h + FADM_BY - 1) / FADM_BY,
                2);  /* z=0 ref, z=1 dis */
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        /* ---- Stage 1: DWT horizontal ---- */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:
                (__bridge id<MTLComputePipelineState>)s->pso_dwt_hori];

            uint32_t p_u[4] = {(uint32_t)cur_w, (uint32_t)half_w,
                                (uint32_t)half_h, (uint32_t)s->buf_stride};
            [enc setBytes:p_u  length:sizeof(p_u) atIndex:0];
            [enc setBuffer:tmp_ref_b  offset:0 atIndex:1];
            [enc setBuffer:tmp_dis_b  offset:0 atIndex:2];
            [enc setBuffer:ref_band_b offset:0 atIndex:3];
            [enc setBuffer:dis_band_b offset:0 atIndex:4];

            MTLSize tg   = MTLSizeMake(FADM_BX, FADM_BY, 1);
            MTLSize grid = MTLSizeMake(
                ((NSUInteger)half_w + FADM_BX - 1) / FADM_BX,
                ((NSUInteger)half_h + FADM_BY - 1) / FADM_BY,
                2);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        /* ---- Stage 2: Decouple + CSF ---- */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:
                (__bridge id<MTLComputePipelineState>)s->pso_decouple_csf];

            uint32_t p_u[3] = {(uint32_t)half_w, (uint32_t)half_h,
                                (uint32_t)s->buf_stride};
            float p_f[4] = {rfh, rfv, rfd, gl};
            [enc setBytes:p_u      length:sizeof(p_u) atIndex:0];
            [enc setBytes:p_f      length:sizeof(p_f) atIndex:1];
            [enc setBuffer:ref_band_b offset:0 atIndex:2];
            [enc setBuffer:dis_band_b offset:0 atIndex:3];
            [enc setBuffer:csf_a_b    offset:0 atIndex:4];
            [enc setBuffer:csf_f_b    offset:0 atIndex:5];

            MTLSize tg   = MTLSizeMake(FADM_BX, FADM_BY, 1);
            MTLSize grid = MTLSizeMake(
                ((NSUInteger)half_w + FADM_BX - 1) / FADM_BX,
                ((NSUInteger)half_h + FADM_BY - 1) / FADM_BY,
                1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        /* ---- Stage 3: CSF-den + CM fused ---- */
        if (active_h > 0) {
            const unsigned nrows = (unsigned)active_h;
            const unsigned n_wg  = 3u * nrows;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:
                (__bridge id<MTLComputePipelineState>)s->pso_csf_cm];

            uint32_t p_u[7] = {
                (uint32_t)half_w,  (uint32_t)half_h, (uint32_t)s->buf_stride,
                (uint32_t)left,    (uint32_t)top,
                (uint32_t)right,   (uint32_t)bottom
            };
            float p_f[4] = {rfh, rfv, rfd, gl};
            [enc setBytes:p_u      length:sizeof(p_u) atIndex:0];
            [enc setBytes:p_f      length:sizeof(p_f) atIndex:1];
            [enc setBuffer:ref_band_b offset:0 atIndex:2];
            [enc setBuffer:dis_band_b offset:0 atIndex:3];
            [enc setBuffer:csf_a_b    offset:0 atIndex:4];
            [enc setBuffer:csf_f_b    offset:0 atIndex:5];
            [enc setBuffer:accum_b    offset:0 atIndex:6];

            MTLSize tg   = MTLSizeMake(FADM_STAGE3_TG, 1, 1);
            MTLSize grid = MTLSizeMake(n_wg, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    } /* per-scale loop */

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

/* ------------------------------------------------------------------ */
/*  collect                                                             */
/* ------------------------------------------------------------------ */

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *fc)
{
    FloatAdmStateMetal *s = (FloatAdmStateMetal *)fex->priv;

    double cm_totals[FADM_NUM_SCALES][FADM_NUM_BANDS]  = {{0.0}};
    double csf_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {{0.0}};

    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        const float *slots = (const float *)
            [(id<MTLBuffer>)(__bridge id<MTLBuffer>)s->accum_buf[sc] contents];
        const unsigned wg_count = s->wg_count[sc];
        for (unsigned wg = 0u; wg < wg_count; wg++) {
            const float *p = slots + (size_t)wg * FADM_ACCUM_SLOTS;
            for (int b = 0; b < FADM_NUM_BANDS; b++) {
                csf_totals[sc][b] += (double)p[b];
                cm_totals[sc][b]  += (double)p[3 + b];
            }
        }
    }

    double score_num = 0.0;
    double score_den = 0.0;
    double scores[8];

    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        const int hw  = (int)s->scale_half_w[sc];
        const int hh  = (int)s->scale_half_h[sc];
        int left = (int)((double)hw * FADM_BORDER_FACTOR - 0.5);
        int top  = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (left < 0) { left = 0; }
        if (top  < 0) { top  = 0; }
        const int right  = hw - left;
        const int bottom = hh - top;
        const float area_cbrt = powf(
            (float)((bottom - top) * (right - left)) * (float)s->adm_noise_weight,
            1.0f / 3.0f);
        float num_scale = 0.0f;
        float den_scale = 0.0f;
        for (int b = 0; b < FADM_NUM_BANDS; b++) {
            num_scale += powf((float)cm_totals[sc][b],  1.0f / 3.0f) + area_cbrt;
            den_scale += powf((float)csf_totals[sc][b], 1.0f / 3.0f) + area_cbrt;
        }
        scores[2 * sc + 0] = num_scale;
        scores[2 * sc + 1] = den_scale;
        score_num += num_scale;
        score_den += den_scale;
    }

    const int w = (int)s->scale_w[0];
    const int h = (int)s->scale_h[0];
    const double numden_limit = 1e-2 * (double)(w * h) / (1920.0 * 1080.0);
    if (score_num < numden_limit) { score_num = 0.0; }
    if (score_den < numden_limit) { score_den = 0.0; }
    const double score = (score_den == 0.0) ? 1.0 : score_num / score_den;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm2_score", score, index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale0_score",
        scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale1_score",
        scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale2_score",
        scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale3_score",
        scores[6] / scores[7], index);

    if (s->debug && (err == 0)) {
        err |= vmaf_feature_collector_append_with_dict(
            fc, s->feature_name_dict, "adm", score, index);
        err |= vmaf_feature_collector_append_with_dict(
            fc, s->feature_name_dict, "adm_num", score_num, index);
        err |= vmaf_feature_collector_append_with_dict(
            fc, s->feature_name_dict, "adm_den", score_den, index);
        const char *names[8] = {
            "adm_num_scale0", "adm_den_scale0",
            "adm_num_scale1", "adm_den_scale1",
            "adm_num_scale2", "adm_den_scale2",
            "adm_num_scale3", "adm_den_scale3",
        };
        for (int i = 0; i < 8 && (err == 0); i++) {
            err |= vmaf_feature_collector_append_with_dict(
                fc, s->feature_name_dict, names[i], scores[i], index);
        }
    }
    return err;
}

/* ------------------------------------------------------------------ */
/*  close                                                               */
/* ------------------------------------------------------------------ */

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatAdmStateMetal *s = (FloatAdmStateMetal *)fex->priv;

    for (int sc = 0; sc < FADM_NUM_SCALES; sc++) {
        release_buffer(&s->accum_buf[sc]);
        release_buffer(&s->ref_band_buf[sc]);
        release_buffer(&s->dis_band_buf[sc]);
    }
    release_buffer(&s->csf_a_buf);
    release_buffer(&s->csf_f_buf);
    release_buffer(&s->dwt_tmp_ref_buf);
    release_buffer(&s->dwt_tmp_dis_buf);
    release_buffer(&s->src_ref_buf);
    release_buffer(&s->src_dis_buf);

    release_pso(&s->pso_dwt_vert);
    release_pso(&s->pso_dwt_hori);
    release_pso(&s->pso_decouple_csf);
    release_pso(&s->pso_csf_cm);

    if (s->feature_name_dict != NULL) {
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    }
    if (s->ctx != NULL) {
        vmaf_metal_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Registration                                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {
    "VMAF_feature_adm2_score",
    "VMAF_feature_adm_scale0_score",
    "VMAF_feature_adm_scale1_score",
    "VMAF_feature_adm_scale2_score",
    "VMAF_feature_adm_scale3_score",
    "adm",
    "adm_num",
    "adm_den",
    "adm_num_scale0", "adm_den_scale0",
    "adm_num_scale1", "adm_den_scale1",
    "adm_num_scale2", "adm_den_scale2",
    "adm_num_scale3", "adm_den_scale3",
    NULL,
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_adm_metal = {
    .name              = "float_adm_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(FloatAdmStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 16,
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
