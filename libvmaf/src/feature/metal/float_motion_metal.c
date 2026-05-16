/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Float Motion feature extractor on the Metal backend — T8-1 batch-2
 *  scaffold (ADR-0361).
 *
 *  Mirrors `libvmaf/src/feature/hip/float_motion_hip.c` call-graph-for-call-graph.
 *  The kernel-template helpers in `libvmaf/src/metal/kernel_template.h`
 *  currently return -ENOSYS; `init()` therefore propagates -ENOSYS so
 *  the feature engine reports "float_motion_metal extractor found but its
 *  runtime is not implemented" rather than "float_motion_metal extractor
 *  not found".
 *
 *  Provided feature: `float_motion`.
 *  TEMPORAL flag set — float Motion carries the prior frame's reference Y plane.
 *  Algorithm parity: per-frame double-precision L2-distance accumulation
 *  between adjacent reference Y planes (10-tap Lanczos downsampling).
 *  Runtime PR (T8-1b) replaces the -ENOSYS template bodies with real Metal
 *  calls (MetalCpp wrapper).
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

typedef struct FloatMotionStateMetal {
    /* Lifecycle (private command queue + submit/finished event pair)
     * and the distance-accumulator buffer slot are managed by
     * `metal/kernel_template.h` (T8-1 / ADR-0361). */
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;
    VmafMetalContext *ctx;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} FloatMotionStateMetal;

static const VmafOption options[] = {{0}};

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatMotionStateMetal *s = fex->priv;

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

    /* One double-precision L2-distance accumulator for downsampled Y planes. */
    err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx, sizeof(double));
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
    FloatMotionStateMetal *s = fex->priv;
    s->index = index;
    /* Runtime PR (T8-1b) replaces this with the live Metal dispatch. */
    return -ENOSYS;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    FloatMotionStateMetal *s = fex->priv;

    int err = vmaf_metal_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR. */
    return -ENOSYS;
}

static int flush_fex_metal(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;
    /* float_motion has no tail frame — return 1 (done) immediately. */
    return 1;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatMotionStateMetal *s = fex->priv;

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

static const char *provided_features[] = {"float_motion", NULL};

/* Load-bearing: registered via `extern VmafFeatureExtractor
 * vmaf_fex_float_motion_metal;` in feature_extractor.c's
 * feature_extractor_list[]. Same pattern as every CUDA / SYCL /
 * Vulkan / HIP feature extractor (e.g. `vmaf_fex_float_motion_hip`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_motion_metal = {
    .name = "float_motion_metal",
    .init = init_fex_metal,
    .submit = submit_fex_metal,
    .collect = collect_fex_metal,
    .flush = flush_fex_metal,
    .close = close_fex_metal,
    .options = options,
    .priv_size = sizeof(FloatMotionStateMetal),
    .provided_features = provided_features,
    /* TEMPORAL flag: carries prior reference Y plane for L2-distance. */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
