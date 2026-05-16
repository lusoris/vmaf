/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature extractor on the Metal backend (T8-1k / ADR-0462).
 *  Port pattern mirrors float_ssim_metal.mm (T8-1j / ADR-0421).
 *
 *  VIF is computed across 4 dyadic scales using separable Gaussian
 *  filters (widths 17, 9, 5, 3) with mirror padding (VIF_OPT_HANDLE_BORDERS).
 *
 *  Per-frame pipeline:
 *    Scale 0: Pass A (compute) on raw pixels → (num0, den0)
 *             Pass B (decimate) on raw pixels → scale-1 float buffers
 *    Scale 1: Pass A on scale-1 float buffers → (num1, den1)
 *             Pass B (decimate)               → scale-2 float buffers
 *    Scale 2: Pass A on scale-2 float buffers → (num2, den2)
 *             Pass B (decimate)               → scale-3 float buffers
 *    Scale 3: Pass A on scale-3 float buffers → (num3, den3)
 *
 *  Output scores (matching float_vif.c):
 *    VMAF_feature_vif_scale{N}_score = num[N] / den[N]  (4 features)
 *
 *  Pixel convention: same as float_vif.c / picture_copy with offset=-128:
 *    8 bpc:  float = pixel - 128
 *   10 bpc:  float = pixel/4 - 128
 *   12 bpc:  float = pixel/16 - 128
 *   16 bpc:  float = pixel/256 - 128
 *  VIF sigma_nsq = 2.0 (default); enhn_gain_limit = 100.0 (default).
 *
 *  Kernel: float_vif_compute  (Pass A)
 *          float_vif_decimate (Pass B)
 *  Both in libvmaf's embedded __TEXT,__metallib.
 *
 *  Apple-Family-7 runtime check: vmaf_metal_context_new() returns
 *  -ENODEV on any device below Apple-Family-7 (Intel Mac, etc.).
 *
 *  ARC: all .mm files in this directory compile under -fobjc-arc.
 *  No manual retain/release.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" {
#include "dict.h"
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

#define FVIF_NUM_SCALES  4
#define FVIF_TG          16  /* threadgroup side length */

typedef struct FloatVifStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    void *pso_compute;   /* float_vif_compute  */
    void *pso_decimate;  /* float_vif_decimate */

    /* Raw-pixel upload buffers (shared, set once per frame). */
    void *raw_ref_buf;
    void *raw_dis_buf;

    /* Intermediate float planes for scales 1–3.
     * f_buf[s] holds the scale-(s+1) input, size scale_w[s+1]*scale_h[s+1]*4. */
    void *f_ref_buf[3];
    void *f_dis_buf[3];

    /* Per-scale partials buffers: 2*grid_w*grid_h floats (num, den interleaved). */
    void *partials_buf[FVIF_NUM_SCALES];

    unsigned scale_w[FVIF_NUM_SCALES];
    unsigned scale_h[FVIF_NUM_SCALES];
    unsigned grid_w [FVIF_NUM_SCALES];
    unsigned grid_h [FVIF_NUM_SCALES];
    size_t   partials_count[FVIF_NUM_SCALES];

    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;
    bool   debug;

    VmafDictionary *feature_name_dict;
} FloatVifStateMetal;

static const VmafOption options[] = {
    {
        .name        = "debug",
        .help        = "debug mode: enable additional output",
        .offset      = offsetof(FloatVifStateMetal, debug),
        .type        = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
    },
    {
        .name        = "vif_enhn_gain_limit",
        .alias       = "egl",
        .help        = "enhancement gain imposed on vif (>= 1.0)",
        .offset      = offsetof(FloatVifStateMetal, vif_enhn_gain_limit),
        .type        = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 100.0},
        .min         = 1.0,
        .max         = 100.0,
        .flags       = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name        = "vif_kernelscale",
        .help        = "scaling factor for the gaussian kernel (must be 1.0)",
        .offset      = offsetof(FloatVifStateMetal, vif_kernelscale),
        .type        = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 1.0},
        .min         = 0.1,
        .max         = 4.0,
        .flags       = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name        = "vif_sigma_nsq",
        .alias       = "snsq",
        .help        = "neural noise variance",
        .offset      = offsetof(FloatVifStateMetal, vif_sigma_nsq),
        .type        = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 2.0},
        .min         = 0.0,
        .max         = 5.0,
        .flags       = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}
};

/* Build both compute pipeline state objects from the embedded metallib. */
static int build_pipelines(FloatVifStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size =
        (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0u) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0u),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn_cmp = [lib newFunctionWithName:@"float_vif_compute"];
    id<MTLFunction> fn_dec = [lib newFunctionWithName:@"float_vif_decimate"];
    if (fn_cmp == nil || fn_dec == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_c =
        [device newComputePipelineStateWithFunction:fn_cmp error:&err];
    id<MTLComputePipelineState> pso_d =
        [device newComputePipelineStateWithFunction:fn_dec error:&err];
    if (pso_c == nil || pso_d == nil) { return -ENODEV; }

    s->pso_compute  = (__bridge_retained void *)pso_c;
    s->pso_decimate = (__bridge_retained void *)pso_d;
    return 0;
}

/* Release all MTLBuffer and PSO handles. */
static void release_buffers(FloatVifStateMetal *s)
{
    if (s->pso_compute) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_compute;
        s->pso_compute = NULL;
    }
    if (s->pso_decimate) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_decimate;
        s->pso_decimate = NULL;
    }
    if (s->raw_ref_buf) {
        (void)(__bridge_transfer id<MTLBuffer>)s->raw_ref_buf;
        s->raw_ref_buf = NULL;
    }
    if (s->raw_dis_buf) {
        (void)(__bridge_transfer id<MTLBuffer>)s->raw_dis_buf;
        s->raw_dis_buf = NULL;
    }
    for (int i = 0; i < 3; ++i) {
        if (s->f_ref_buf[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->f_ref_buf[i];
            s->f_ref_buf[i] = NULL;
        }
        if (s->f_dis_buf[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->f_dis_buf[i];
            s->f_dis_buf[i] = NULL;
        }
    }
    for (int i = 0; i < FVIF_NUM_SCALES; ++i) {
        if (s->partials_buf[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->partials_buf[i];
            s->partials_buf[i] = NULL;
        }
    }
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;

    /* Only kernelscale=1.0 is implemented (matches CUDA parity). */
    if (s->vif_kernelscale != 1.0) { return -EINVAL; }

    s->frame_w = w;
    s->frame_h = h;
    s->bpc     = bpc;

    /* Dyadic scale dimensions (VIF_OPT_HANDLE_BORDERS: each scale halves). */
    s->scale_w[0] = w;    s->scale_h[0] = h;
    for (int i = 1; i < FVIF_NUM_SCALES; ++i) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }
    for (int i = 0; i < FVIF_NUM_SCALES; ++i) {
        s->grid_w[i] = (s->scale_w[i] + FVIF_TG - 1u) / FVIF_TG;
        s->grid_h[i] = (s->scale_h[i] + FVIF_TG - 1u) / FVIF_TG;
        s->partials_count[i] = (size_t)s->grid_w[i] * s->grid_h[i];
    }

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        /* Raw-pixel upload buffers (scale 0). */
        const size_t bytes_per_px = (bpc <= 8u) ? 1u : 2u;
        const size_t raw_size = (size_t)w * h * bytes_per_px;
        id<MTLBuffer> rr = [device newBufferWithLength:raw_size
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> dr = [device newBufferWithLength:raw_size
                                               options:MTLResourceStorageModeShared];
        if (rr == nil || dr == nil) { err = -ENOMEM; goto fail_lc; }
        s->raw_ref_buf = (__bridge_retained void *)rr;
        s->raw_dis_buf = (__bridge_retained void *)dr;

        /* Intermediate float buffers for scales 1–3. */
        for (int i = 0; i < 3; ++i) {
            const size_t float_sz =
                (size_t)s->scale_w[i + 1] * s->scale_h[i + 1] * sizeof(float);
            id<MTLBuffer> fr =
                [device newBufferWithLength:float_sz options:MTLResourceStorageModeShared];
            id<MTLBuffer> fd =
                [device newBufferWithLength:float_sz options:MTLResourceStorageModeShared];
            if (fr == nil || fd == nil) { err = -ENOMEM; goto fail_lc; }
            s->f_ref_buf[i] = (__bridge_retained void *)fr;
            s->f_dis_buf[i] = (__bridge_retained void *)fd;
        }

        /* Partials buffers: 2 floats per threadgroup. */
        for (int i = 0; i < FVIF_NUM_SCALES; ++i) {
            const size_t pb_sz = 2u * s->partials_count[i] * sizeof(float);
            id<MTLBuffer> pb =
                [device newBufferWithLength:pb_sz options:MTLResourceStorageModeShared];
            if (pb == nil) { err = -ENOMEM; goto fail_lc; }
            s->partials_buf[i] = (__bridge_retained void *)pb;
        }

        err = build_pipelines(s, device);
    }
    if (err != 0) { goto fail_lc; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    release_buffers(s);
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* -----------------------------------------------------------------------
 *  Submit: encode one full-frame VIF pipeline into a command buffer.
 * --------------------------------------------------------------------- */
static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>        device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>   queue = (__bridge id<MTLCommandQueue>)qh;
    (void)device;  /* used only for buffer alloc, already done in init */

    /* Upload raw luma plane. */
    {
        const size_t bytes_per_px = (s->bpc <= 8u) ? 1u : 2u;
        const size_t raw_size = (size_t)s->frame_w * s->frame_h * bytes_per_px;
        uint8_t *rr = (uint8_t *)[(__bridge id<MTLBuffer>)s->raw_ref_buf contents];
        uint8_t *dr = (uint8_t *)[(__bridge id<MTLBuffer>)s->raw_dis_buf contents];

        for (unsigned y = 0; y < s->frame_h; ++y) {
            const uint8_t *rs =
                (const uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0];
            const uint8_t *ds =
                (const uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0];
            memcpy(rr + y * s->frame_w * bytes_per_px, rs,
                   s->frame_w * bytes_per_px);
            memcpy(dr + y * s->frame_w * bytes_per_px, ds,
                   s->frame_w * bytes_per_px);
        }
        (void)raw_size;
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    /* Zero the partials buffers. */
    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        for (int sc = 0; sc < FVIF_NUM_SCALES; ++sc) {
            id<MTLBuffer> pb = (__bridge id<MTLBuffer>)s->partials_buf[sc];
            [blit fillBuffer:pb
                       range:NSMakeRange(0, 2u * s->partials_count[sc] * sizeof(float))
                       value:0];
        }
        [blit endEncoding];
    }

    /* VIF parameters. */
    const float sigma_nsq     = (float)s->vif_sigma_nsq;
    const float enhn_gain_lim = (float)s->vif_enhn_gain_limit;
    /* sigma_max_inv = sigma_nsq^2 / (255^2) */
    const float sigma_max_inv =
        (sigma_nsq * sigma_nsq) / (255.0f * 255.0f);

    /* Helper PSO casts. */
    id<MTLComputePipelineState> pso_cmp =
        (__bridge id<MTLComputePipelineState>)s->pso_compute;
    id<MTLComputePipelineState> pso_dec =
        (__bridge id<MTLComputePipelineState>)s->pso_decimate;

    /* Raw-plane buffers. */
    id<MTLBuffer> raw_ref_buf = (__bridge id<MTLBuffer>)s->raw_ref_buf;
    id<MTLBuffer> raw_dis_buf = (__bridge id<MTLBuffer>)s->raw_dis_buf;

    const uint32_t raw_stride = (uint32_t)(s->frame_w *
        (s->bpc <= 8u ? 1u : 2u));

    /* Dummy 1-byte buffers for unused raw_ref/dis at scale > 0 and for
     * unused ref_f/dis_f at scale 0 (the kernel ignores them). */
    id<MTLBuffer> dummy = [device newBufferWithLength:4u
                                              options:MTLResourceStorageModeShared];
    if (dummy == nil) { return -ENOMEM; }

    for (int sc = 0; sc < FVIF_NUM_SCALES; ++sc) {
        const unsigned W = s->scale_w[sc];
        const unsigned H = s->scale_h[sc];
        const unsigned gw = s->grid_w[sc];
        const unsigned gh = s->grid_h[sc];

        id<MTLBuffer> ref_f = (sc == 0) ? dummy
                            : (__bridge id<MTLBuffer>)s->f_ref_buf[sc - 1];
        id<MTLBuffer> dis_f = (sc == 0) ? dummy
                            : (__bridge id<MTLBuffer>)s->f_dis_buf[sc - 1];
        id<MTLBuffer> pb    = (__bridge id<MTLBuffer>)s->partials_buf[sc];

        /* Pass A — compute. */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_cmp];
            [enc setBuffer:ref_f       offset:0 atIndex:0];
            [enc setBuffer:dis_f       offset:0 atIndex:1];
            [enc setBuffer:pb          offset:0 atIndex:2];
            uint32_t par[4] = {W, H, s->bpc, gw};
            [enc setBytes:par          length:sizeof(par) atIndex:3];
            float cst[4] = {sigma_nsq, enhn_gain_lim, sigma_max_inv,
                            (float)sc};
            [enc setBytes:cst          length:sizeof(cst) atIndex:4];
            [enc setBuffer:raw_ref_buf offset:0 atIndex:5];
            [enc setBuffer:raw_dis_buf offset:0 atIndex:6];
            uint32_t rpar[4] = {raw_stride, 0u, 0u, 0u};
            [enc setBytes:rpar         length:sizeof(rpar) atIndex:7];
            MTLSize tg   = MTLSizeMake(FVIF_TG, FVIF_TG, 1);
            MTLSize grid = MTLSizeMake(gw, gh, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        /* Pass B — decimate (skip for scale 3). */
        if (sc < FVIF_NUM_SCALES - 1) {
            const unsigned ow = s->scale_w[sc + 1];
            const unsigned oh = s->scale_h[sc + 1];
            id<MTLBuffer> ref_out = (__bridge id<MTLBuffer>)s->f_ref_buf[sc];
            id<MTLBuffer> dis_out = (__bridge id<MTLBuffer>)s->f_dis_buf[sc];

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dec];
            [enc setBuffer:ref_f       offset:0 atIndex:0];
            [enc setBuffer:dis_f       offset:0 atIndex:1];
            [enc setBuffer:ref_out     offset:0 atIndex:2];
            [enc setBuffer:dis_out     offset:0 atIndex:3];
            uint32_t par[4] = {ow, oh, W, H};
            [enc setBytes:par          length:sizeof(par) atIndex:4];
            /* scale_idx for decimate: the filter used to compute this
             * scale's downsampled output is the NEXT scale's filter.
             * Matches CUDA: decimate(scale) uses scale+1 coefficients
             * (vif_get_filter(scale+1, kernelscale=1.0)). */
            uint32_t cst[4] = {s->bpc, (uint32_t)(sc + 1u), 0u, 0u};
            [enc setBytes:cst          length:sizeof(cst) atIndex:5];
            [enc setBuffer:raw_ref_buf offset:0 atIndex:6];
            [enc setBuffer:raw_dis_buf offset:0 atIndex:7];
            uint32_t rpar[4] = {raw_stride, 0u, 0u, 0u};
            [enc setBytes:rpar         length:sizeof(rpar) atIndex:8];
            MTLSize tg   = MTLSizeMake(FVIF_TG, FVIF_TG, 1);
            MTLSize grid = MTLSizeMake((ow + FVIF_TG - 1u) / FVIF_TG,
                                       (oh + FVIF_TG - 1u) / FVIF_TG, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;
    int err = 0;

    static const char *scale_names[FVIF_NUM_SCALES] = {
        "VMAF_feature_vif_scale0_score",
        "VMAF_feature_vif_scale1_score",
        "VMAF_feature_vif_scale2_score",
        "VMAF_feature_vif_scale3_score",
    };

    double scores[FVIF_NUM_SCALES * 2u];
    for (int sc = 0; sc < FVIF_NUM_SCALES; ++sc) {
        const float *parts =
            (const float *)[(__bridge id<MTLBuffer>)s->partials_buf[sc] contents];
        double sum_num = 0.0, sum_den = 0.0;
        for (size_t i = 0; i < s->partials_count[sc]; ++i) {
            sum_num += (double)parts[2u * i];
            sum_den += (double)parts[2u * i + 1u];
        }
        scores[2 * sc]     = sum_num;
        scores[2 * sc + 1] = sum_den;
        const double score = (sum_den > 0.0) ? (sum_num / sum_den) : 1.0;
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            scale_names[sc], score, index);
    }

    if (s->debug && !err) {
        const double score_num =
            scores[0] + scores[2] + scores[4] + scores[6];
        const double score_den =
            scores[1] + scores[3] + scores[5] + scores[7];
        const double score =
            (score_den == 0.0) ? 1.0 : (score_num / score_den);
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "vif", score, index);
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "vif_num", score_num, index);
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "vif_den", score_den, index);
        static const char *dbg_names[8] = {
            "vif_num_scale0", "vif_den_scale0",
            "vif_num_scale1", "vif_den_scale1",
            "vif_num_scale2", "vif_den_scale2",
            "vif_num_scale3", "vif_den_scale3",
        };
        for (int i = 0; i < 8; ++i) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                dbg_names[i], scores[i], index);
        }
    }

    return err;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
    release_buffers(s);
    if (s->feature_name_dict) {
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    }
    if (s->ctx) {
        vmaf_metal_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {
    "VMAF_feature_vif_scale0_score",
    "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score",
    "VMAF_feature_vif_scale3_score",
    "vif",
    "vif_num",
    "vif_den",
    "vif_num_scale0",
    "vif_den_scale0",
    "vif_num_scale1",
    "vif_den_scale1",
    "vif_num_scale2",
    "vif_den_scale2",
    "vif_num_scale3",
    "vif_den_scale3",
    NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_vif_metal = {
    .name              = "float_vif_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(FloatVifStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 7,   /* 4 compute + 3 decimate */
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
