/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_moment feature extractor on the Metal backend (T8-1e / ADR-0421).
 *  Dispatches `float_moment_kernel_{8,16}bpc` from float_moment.metal.
 *
 *  Reduction layout (integer-exact, mirroring integer_psnr_metal pattern):
 *    Eight uint32 buffers, two per accumulator (lo+hi):
 *      r1_lo/r1_hi, d1_lo/d1_hi, r2_lo/r2_hi, d2_lo/d2_hi
 *    Each buffer holds grid_w × grid_h uint32 WG partials.
 *    Host reconstructs: val_u64 = ((uint64)hi << 32) | lo, sums, then
 *    divides by (W * H * scaler) for 1st moment and
 *    (W * H * scaler^2) for 2nd moment.
 *    For 8bpc: scaler = 1. For >8bpc: scaler = 1 << (bpc - 8).
 *
 *  Feature names: float_moment_ref1st, float_moment_dis1st,
 *                 float_moment_ref2nd, float_moment_dis2nd.
 */

#include <errno.h>
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

/* Indices into rb[]: r1_lo=0, r1_hi=1, d1_lo=2, d1_hi=3,
 *                    r2_lo=4, r2_hi=5, d2_lo=6, d2_hi=7. */
#define FM_NUM_BUFS 8u

typedef struct FloatMomentStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb[FM_NUM_BUFS]; /* uint32 lo/hi partials */
    VmafMetalContext *ctx;
    void *pso_8bpc;
    void *pso_16bpc;

    size_t plane_bytes;
    size_t partials_count;   /* grid_w × grid_h */
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} FloatMomentStateMetal;

static const VmafOption options[] = {{0}};

static int build_pipelines(FloatMomentStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size = (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn8  = [lib newFunctionWithName:@"float_moment_kernel_8bpc"];
    id<MTLFunction> fn16 = [lib newFunctionWithName:@"float_moment_kernel_16bpc"];
    if (fn8 == nil || fn16 == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso8  = [device newComputePipelineStateWithFunction:fn8  error:&err];
    id<MTLComputePipelineState> pso16 = [device newComputePipelineStateWithFunction:fn16 error:&err];
    if (pso8 == nil || pso16 == nil) { return -ENODEV; }

    s->pso_8bpc  = (__bridge_retained void *)pso8;
    s->pso_16bpc = (__bridge_retained void *)pso16;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatMomentStateMetal *s = (FloatMomentStateMetal *)fex->priv;

    s->frame_w     = w;
    s->frame_h     = h;
    s->bpc         = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8u ? 1u : 2u);

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        const size_t grid_w   = (w + 15) / 16;
        const size_t grid_h   = (h + 15) / 16;
        s->partials_count     = grid_w * grid_h;
        const size_t par_size = s->partials_count * sizeof(uint32_t);
        for (unsigned b = 0u; b < FM_NUM_BUFS; ++b) {
            err = vmaf_metal_kernel_buffer_alloc(&s->rb[b], s->ctx, par_size);
            if (err != 0) {
                for (unsigned q = 0u; q < b; ++q) {
                    (void)vmaf_metal_kernel_buffer_free(&s->rb[q], s->ctx);
                }
                goto fail_lc;
            }
        }
    }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_rb; }
        err = build_pipelines(s, (__bridge id<MTLDevice>)dh);
    }
    if (err != 0) { goto fail_rb; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }
    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
fail_rb:
    for (unsigned b = 0u; b < FM_NUM_BUFS; ++b) {
        (void)vmaf_metal_kernel_buffer_free(&s->rb[b], s->ctx);
    }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    FloatMomentStateMetal *s = (FloatMomentStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    const size_t row_bytes = (size_t)s->frame_w * (s->bpc <= 8u ? 1u : 2u);

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>      device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)qh;
    id<MTLComputePipelineState> pso = (s->bpc <= 8u)
        ? (__bridge id<MTLComputePipelineState>)s->pso_8bpc
        : (__bridge id<MTLComputePipelineState>)s->pso_16bpc;

    id<MTLBuffer> ref_buf = [device newBufferWithLength:s->plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_buf = [device newBufferWithLength:s->plane_bytes options:MTLResourceStorageModeShared];
    if (ref_buf == nil || dis_buf == nil) { return -ENOMEM; }
    {
        uint8_t *rd = (uint8_t *)[ref_buf contents];
        uint8_t *dd = (uint8_t *)[dis_buf contents];
        for (unsigned y = 0; y < s->frame_h; y++) {
            memcpy(rd + y * row_bytes, (uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0], row_bytes);
            memcpy(dd + y * row_bytes, (uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0], row_bytes);
        }
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    /* Zero all partial buffers before dispatch. */
    const size_t par_size = s->partials_count * sizeof(uint32_t);
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    for (unsigned b = 0u; b < FM_NUM_BUFS; ++b) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void *)s->rb[b].buffer;
        [blit fillBuffer:buf range:NSMakeRange(0, par_size) value:0];
    }
    [blit endEncoding];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:ref_buf offset:0 atIndex:0];
    [enc setBuffer:dis_buf offset:0 atIndex:1];
    for (unsigned b = 0u; b < FM_NUM_BUFS; ++b) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void *)s->rb[b].buffer;
        [enc setBuffer:buf offset:0 atIndex:(NSUInteger)(b + 2u)];
    }
    if (s->bpc <= 8u) {
        uint32_t st[2] = {(uint32_t)row_bytes, (uint32_t)row_bytes};
        [enc setBytes:st length:sizeof(st) atIndex:10];
    } else {
        uint32_t st[4] = {(uint32_t)row_bytes, (uint32_t)row_bytes, (uint32_t)s->bpc, 0};
        [enc setBytes:st length:sizeof(st) atIndex:10];
    }
    uint32_t dim[2] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h};
    [enc setBytes:dim length:sizeof(dim) atIndex:11];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((s->frame_w + 15) / 16, (s->frame_h + 15) / 16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    FloatMomentStateMetal *s = (FloatMomentStateMetal *)fex->priv;

    /*
     * Reconstruct uint64 sums from lo/hi partial buffers.
     * Buffer layout: [r1_lo=0, r1_hi=1, d1_lo=2, d1_hi=3,
     *                 r2_lo=4, r2_hi=5, d2_lo=6, d2_hi=7].
     * sum[0]=ref1, sum[1]=dis1, sum[2]=ref2, sum[3]=dis2.
     */
    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    {
        const uint32_t *r1_lo = (const uint32_t *)s->rb[0].host_view;
        const uint32_t *r1_hi = (const uint32_t *)s->rb[1].host_view;
        const uint32_t *d1_lo = (const uint32_t *)s->rb[2].host_view;
        const uint32_t *d1_hi = (const uint32_t *)s->rb[3].host_view;
        const uint32_t *r2_lo = (const uint32_t *)s->rb[4].host_view;
        const uint32_t *r2_hi = (const uint32_t *)s->rb[5].host_view;
        const uint32_t *d2_lo = (const uint32_t *)s->rb[6].host_view;
        const uint32_t *d2_hi = (const uint32_t *)s->rb[7].host_view;

        if (r1_lo != NULL && r1_hi != NULL &&
            d1_lo != NULL && d1_hi != NULL &&
            r2_lo != NULL && r2_hi != NULL &&
            d2_lo != NULL && d2_hi != NULL) {
            for (size_t i = 0; i < s->partials_count; ++i) {
                sum[0] += (double)(((uint64_t)r1_hi[i] << 32u) | (uint64_t)r1_lo[i]);
                sum[1] += (double)(((uint64_t)d1_hi[i] << 32u) | (uint64_t)d1_lo[i]);
                sum[2] += (double)(((uint64_t)r2_hi[i] << 32u) | (uint64_t)r2_lo[i]);
                sum[3] += (double)(((uint64_t)d2_hi[i] << 32u) | (uint64_t)d2_lo[i]);
            }
        }
    }

    /*
     * Divide by (W * H * scaler) for 1st moment and
     * (W * H * scaler^2) for 2nd moment.
     * For 8bpc scaler=1, so both denominators equal W*H.
     * This matches the CPU float_moment.c behaviour where pixels are
     * pre-divided by scaler before accumulation; here we defer the
     * division to the host for integer-exact GPU accumulation.
     */
    const double n_pix   = (double)s->frame_w * (double)s->frame_h;
    double scaler        = 1.0;
    if (s->bpc > 8u) {
        scaler = (double)(1u << (s->bpc - 8u));
    }
    const double denom1 = n_pix * scaler;
    const double denom2 = n_pix * scaler * scaler;

    const double ref1 = (denom1 > 0.0) ? (sum[0] / denom1) : 0.0;
    const double dis1 = (denom1 > 0.0) ? (sum[1] / denom1) : 0.0;
    const double ref2 = (denom2 > 0.0) ? (sum[2] / denom2) : 0.0;
    const double dis2 = (denom2 > 0.0) ? (sum[3] / denom2) : 0.0;

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_ref1st", ref1, index);
    if (err != 0) { return err; }
    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_dis1st", dis1, index);
    if (err != 0) { return err; }
    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_ref2nd", ref2, index);
    if (err != 0) { return err; }
    return vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_dis2nd", dis2, index);
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatMomentStateMetal *s = (FloatMomentStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }

    for (unsigned b = 0u; b < FM_NUM_BUFS; ++b) {
        int err = vmaf_metal_kernel_buffer_free(&s->rb[b], s->ctx);
        if (err != 0 && rc == 0) { rc = err; }
    }
    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {
    "float_moment_ref1st", "float_moment_dis1st",
    "float_moment_ref2nd", "float_moment_dis2nd",
    NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_moment_metal = {
    .name              = "float_moment_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(FloatMomentStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 1,
        .is_reduction_only      = true,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
