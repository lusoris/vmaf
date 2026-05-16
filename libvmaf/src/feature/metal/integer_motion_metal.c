/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Motion (v1) feature extractor on the Metal backend — fourth scaffold
 *  consumer in the T8-1 batch (ADR-0361).
 *
 *  Mirrors the CPU `integer_motion.c` / HIP `integer_motion_v2_hip.c`
 *  temporal structure call-graph-for-call-graph.  The kernel-template
 *  helpers in `libvmaf/src/metal/kernel_template.h` currently return
 *  -ENOSYS; `init()` therefore propagates -ENOSYS so the feature engine
 *  reports "motion_metal extractor found but its runtime is not
 *  implemented" rather than "motion_metal extractor not found".
 *
 *  Provided features: `VMAF_integer_feature_motion_score`,
 *  `VMAF_integer_feature_motion2_score`, `VMAF_integer_feature_motion3_score`.
 *  TEMPORAL flag required: motion v1 needs the previous-frame carry
 *  (SAD between consecutive frames).  Runtime PR (T8-1b) replaces the
 *  -ENOSYS template bodies with real Metal calls (MetalCpp wrapper).
 *
 *  Relationship to motion_v2_metal: v1 emits three features (motion,
 *  motion2, motion3) vs v2's two (motion_v2_sad, motion2_v2); v1 uses
 *  a SAD-over-blurred-frames algorithm while v2 exploits convolution
 *  linearity to avoid storing blurred frames across submits.  The struct
 *  shape therefore keeps a `prev_ref` slot for the unblurred ref carry
 *  (like the CPU twin's raw-pixel ping-pong), distinct from v2's
 *  blurred-result slot.
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

typedef struct MotionStateMetal {
    /* Lifecycle (private command queue + submit/finished event pair)
     * and the SAD-accumulator buffer are managed by
     * `metal/kernel_template.h` (T8-1 / ADR-0361). */
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;
    VmafMetalContext *ctx;

    /* Previous-frame ref Y buffer (raw, pre-blur) — unified-memory slot.
     * The runtime PR (T8-1b) will allocate a plane_bytes-sized MTLBuffer
     * here and host-write the cur ref Y plane into it on each submit.
     * Mirrors the `prev_ref` slot in `integer_motion_v2_metal.c` modulo
     * the v1/v2 algorithm difference noted in the file header. */
    uintptr_t prev_ref;
    size_t plane_bytes;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} MotionStateMetal;

static const VmafOption options[] = {{0}};

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    (void)pix_fmt;
    MotionStateMetal *s = fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8u ? 1u : 2u);

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    /* Single uint64 SAD accumulator (same shape as motion_v2_metal). */
    err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx, sizeof(uint64_t));
    if (err != 0) {
        goto fail_after_lc;
    }

    s->prev_ref = 0;

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
    (void)dist_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)ref_pic;
    MotionStateMetal *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Runtime PR (T8-1b) replaces this with the live Metal dispatch. */
    return -ENOSYS;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    MotionStateMetal *s = fex->priv;

    int err = vmaf_metal_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission (motion, motion2, motion3 from uint64 SAD sum) lands
     * with the runtime PR. */
    return -ENOSYS;
}

static int flush_fex_metal(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;
    /* Tail-frame motion3 window post-pass lands with the runtime PR.
     * Until then, return 1 (done) so the feature engine does not loop. */
    return 1;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    MotionStateMetal *s = fex->priv;

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    s->prev_ref = 0;

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

static const char *provided_features[] = {"VMAF_integer_feature_motion_score",
                                          "VMAF_integer_feature_motion2_score",
                                          "VMAF_integer_feature_motion3_score", NULL};

/* Load-bearing: registered via `extern VmafFeatureExtractor
 * vmaf_fex_integer_motion_metal;` in feature_extractor.c's
 * feature_extractor_list[]. Same pattern as every CUDA / SYCL /
 * Vulkan / HIP feature extractor. */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_motion_metal = {
    .name = "motion_metal",
    .init = init_fex_metal,
    .submit = submit_fex_metal,
    .collect = collect_fex_metal,
    .flush = flush_fex_metal,
    .close = close_fex_metal,
    .options = options,
    .priv_size = sizeof(MotionStateMetal),
    .provided_features = provided_features,
    /* TEMPORAL flag mandatory: motion v1 needs the previous-frame carry. */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
