/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Float SSIM feature extractor on the Metal backend — third scaffold
 *  consumer in the T8-1 batch (ADR-0361).
 *
 *  Mirrors `libvmaf/src/feature/hip/float_ssim_hip.c` (ADR-0274)
 *  call-graph-for-call-graph.  Two-dispatch separable Gaussian + per-block
 *  float partial readback shape (same as the CUDA / HIP twins).  The
 *  kernel-template helpers in `libvmaf/src/metal/kernel_template.h`
 *  currently return -ENOSYS; `init()` therefore propagates -ENOSYS
 *  so the feature engine reports "float_ssim_metal extractor found but
 *  its runtime is not implemented" rather than "float_ssim_metal extractor
 *  not found".
 *
 *  Provided feature: `float_ssim`.
 *  No temporal flag — SSIM is per-frame.
 *  v1 is scale=1 only (consistent with the HIP / CUDA twins at
 *  scaffold stage). Runtime PR (T8-1b) replaces -ENOSYS template bodies
 *  with real Metal calls (MetalCpp wrapper).
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../metal/common.h"
#include "../../metal/kernel_template.h"

typedef struct FloatSsimStateMetal {
    /* Lifecycle (private command queue + submit/finished event pair)
     * and the float partial-result readback buffer are managed by
     * `metal/kernel_template.h` (T8-1 / ADR-0361).
     * Two-dispatch shape mirrors the HIP twin (ADR-0274):
     *   dispatch 1 — separable Gaussian (intermediate float buffers),
     *   dispatch 2 — per-block SSIM reduction into readback buffer.
     * The intermediate buffers (five, matching the CUDA / HIP twins)
     * will be allocated by the runtime PR (T8-1b). */
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;
    VmafMetalContext *ctx;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} FloatSsimStateMetal;

static const VmafOption options[] = {
    {
        .name = "scale",
        .alias = "s",
        .help = "scale factor (integer, ≥ 1). v1 supports scale=1 only.",
        .offset = 0, /* no runtime field in the scaffold struct */
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 1,
        .min = 1,
        .max = 1,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0},
};

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatSsimStateMetal *s = fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    /* Readback buffer: one float per 8×8 block (worst-case 1080p → ≤32 400 blocks).
     * The runtime PR (T8-1b) determines the exact block count and may resize. */
    unsigned n_blocks = ((w + 7u) / 8u) * ((h + 7u) / 8u);
    err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx, (size_t)n_blocks * sizeof(float));
    if (err != 0) {
        goto fail_after_lc;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
        goto fail_after_rb;
    }

    return 0;

fail_after_rb:
    (void)vmaf_metal_kernel_buffer_free(&s->rb, s->ctx);
fail_after_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_after_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)ref_pic;
    (void)dist_pic;
    FloatSsimStateMetal *s = fex->priv;
    s->index = index;
    /* Runtime PR (T8-1b) replaces this with the live two-dispatch Metal chain. */
    return -ENOSYS;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    FloatSsimStateMetal *s = fex->priv;

    int err = vmaf_metal_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score aggregation (sum of per-block SSIM / n_blocks) lands
     * with the runtime PR. */
    return -ENOSYS;
}

static int flush_fex_metal(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;
    /* SSIM has no tail frame — return 1 (done) immediately. */
    return 1;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatSsimStateMetal *s = fex->priv;

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    int err = vmaf_metal_kernel_buffer_free(&s->rb, s->ctx);
    if (err != 0 && rc == 0) {
        rc = err;
    }
    if (s->feature_name_dict != NULL) {
        err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0) {
            rc = err;
        }
    }
    if (s->ctx != NULL) {
        vmaf_metal_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"float_ssim", NULL};

/* Load-bearing: registered via `extern VmafFeatureExtractor
 * vmaf_fex_float_ssim_metal;` in feature_extractor.c's
 * feature_extractor_list[]. Same pattern as every CUDA / SYCL /
 * Vulkan / HIP feature extractor (e.g. `vmaf_fex_float_ssim_hip`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ssim_metal = {
    .name = "float_ssim_metal",
    .init = init_fex_metal,
    .submit = submit_fex_metal,
    .collect = collect_fex_metal,
    .flush = flush_fex_metal,
    .close = close_fex_metal,
    .options = options,
    .priv_size = sizeof(FloatSsimStateMetal),
    .provided_features = provided_features,
    /* No TEMPORAL flag: SSIM is per-frame, no previous-frame carry. */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
