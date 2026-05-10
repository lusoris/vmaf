/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Float Moment feature extractor on the Metal backend — T8-1 batch-2
 *  scaffold (ADR-0361).
 *
 *  Mirrors `libvmaf/src/feature/hip/float_moment_hip.c` call-graph-for-call-graph.
 *  The kernel-template helpers in `libvmaf/src/metal/kernel_template.h`
 *  currently return -ENOSYS; `init()` therefore propagates -ENOSYS so
 *  the feature engine reports "float_moment_metal extractor found but its
 *  runtime is not implemented" rather than "float_moment_metal extractor
 *  not found".
 *
 *  Provided features: `float_moment1`, `float_moment2`, `float_std`.
 *  No temporal flag — float Moment is per-frame, no previous-frame carry.
 *  Algorithm parity: per-frame double-precision first and second moment
 *  accumulation; std derived from moments on the host.  Runtime PR
 *  (T8-1b) replaces the -ENOSYS template bodies with real Metal calls
 *  (MetalCpp wrapper).
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

typedef struct FloatMomentStateMetal {
    /* Lifecycle (private command queue + submit/finished event pair)
     * and the moment-accumulator buffer slot are managed by
     * `metal/kernel_template.h` (T8-1 / ADR-0361). */
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;
    VmafMetalContext *ctx;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} FloatMomentStateMetal;

static const VmafOption options[] = {{0}};

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatMomentStateMetal *s = fex->priv;

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

    /* Two double-precision moment accumulators (first + second). */
    err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx, 2u * sizeof(double));
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
    FloatMomentStateMetal *s = fex->priv;
    s->index = index;
    /* Runtime PR (T8-1b) replaces this with the live Metal dispatch. */
    return -ENOSYS;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    FloatMomentStateMetal *s = fex->priv;

    int err = vmaf_metal_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission (moment -> std conversion) lands with the runtime PR. */
    return -ENOSYS;
}

static int flush_fex_metal(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;
    /* float_moment has no tail frame — return 1 (done) immediately. */
    return 1;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatMomentStateMetal *s = fex->priv;

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

static const char *provided_features[] = {"float_moment1", "float_moment2", "float_std", NULL};

/* Load-bearing: registered via `extern VmafFeatureExtractor
 * vmaf_fex_float_moment_metal;` in feature_extractor.c's
 * feature_extractor_list[]. Same pattern as every CUDA / SYCL /
 * Vulkan / HIP feature extractor (e.g. `vmaf_fex_float_moment_hip`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_moment_metal = {
    .name = "float_moment_metal",
    .init = init_fex_metal,
    .submit = submit_fex_metal,
    .collect = collect_fex_metal,
    .flush = flush_fex_metal,
    .close = close_fex_metal,
    .options = options,
    .priv_size = sizeof(FloatMomentStateMetal),
    .provided_features = provided_features,
    /* No TEMPORAL flag: float_moment is per-frame, no previous-frame carry. */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
