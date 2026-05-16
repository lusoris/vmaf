/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  motion_v2 feature extractor on the Metal backend — first
 *  kernel-template consumer (T8-1 / ADR-0361).
 *
 *  This TU mirrors `libvmaf/src/feature/hip/integer_motion_v2_hip.c`
 *  (ADR-0267) call-graph-for-call-graph, modulo the unified-memory
 *  buffer simplification: where the HIP twin tracks a `pix[2]`
 *  ping-pong of opaque `uintptr_t` slots (separate device buffers
 *  for ref Y plane carry-over), the Metal twin tracks a single
 *  `prev_ref` MTLBuffer — Apple Silicon's unified memory removes
 *  the need to stage the previous frame across a device boundary.
 *  The runtime PR (T8-1b) will allocate one `MTLResourceStorageModeShared`
 *  buffer sized to the ref Y plane and reuse it as the "previous
 *  frame" slot the kernel reads on each submit.
 *
 *  Stateless variant of the classic motion kernel: exploits
 *  convolution linearity (`SAD(blur(prev), blur(cur)) ==
 *  sum(|blur(prev - cur)|)`) so each frame computes its score in
 *  one kernel dispatch over (prev_ref - cur_ref) without storing
 *  blurred frames across submits.
 *
 *  Unique posture vs the prior (HIP / CUDA) consumers: a **temporal**
 *  extractor (`VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag) but with one
 *  fewer ping-pong slot than the HIP twin thanks to unified memory.
 *  Pinning the field shape now makes the runtime PR's diff a content
 *  swap rather than a struct redesign.
 *
 *  The kernel-template helpers in `libvmaf/src/metal/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports
 *  "motion_v2_metal extractor found but its runtime is not implemented"
 *  rather than "motion_v2_metal extractor not found". The smoke test
 *  pins this registration-shape contract.
 *
 *  When the runtime PR (T8-1b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real Metal calls (MetalCpp wrapper) and *this* TU
 *  keeps its current shape verbatim. That's the load-bearing
 *  invariant: the consumer is written against the template's
 *  contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): single int64 atomic
 *  SAD accumulator per frame, divided by 256.0 / (w*h) on the host
 *  to recover `VMAF_integer_feature_motion_v2_sad_score`.
 *  `VMAF_integer_feature_motion2_v2_score = min(score[i], score[i+1])`
 *  is emitted host-side in `flush()` (same shape as the HIP / CUDA
 *  twins). Mirrors the HIP reference exactly modulo the
 *  unified-memory buffer collapse.
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

typedef struct MotionV2StateMetal {
    /* Lifecycle (private command queue + submit/finished event pair)
     * and the SAD-accumulator buffer slot are managed by
     * `metal/kernel_template.h` (T8-1 first consumer / ADR-0361).
     * The struct shape mirrors the HIP twin's `MotionV2StateHip`
     * (ADR-0267) modulo the ping-pong collapse documented below. */
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;
    VmafMetalContext *ctx;

    /* Single previous-frame ref Y buffer slot — unified-memory
     * collapse of the HIP twin's `pix[2]` ping-pong. Apple Silicon's
     * `MTLResourceStorageModeShared` means the host can write the
     * cur ref Y plane directly into the buffer's contents pointer
     * (no D2D copy needed) and the kernel can read the prev frame
     * from the same allocation on the next submit. The runtime PR
     * (T8-1b) will replace the `uintptr_t` slot with a real
     * MTLBuffer handle and flip `plane_bytes`-sized host stores
     * through it. */
    uintptr_t prev_ref;
    size_t plane_bytes;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} MotionV2StateMetal;

static const VmafOption options[] = {{0}};

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    (void)pix_fmt;
    MotionV2StateMetal *s = fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8u ? 1u : 2u);

    /* Allocate a Metal context — the scaffold's `vmaf_metal_context_new`
     * succeeds today (calloc + struct init); the runtime PR will swap
     * in `MTLCreateSystemDefaultDevice` + command-queue creation via
     * MetalCpp. */
    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    /* Command queue + event pair via the template. Scaffold returns
     * -ENOSYS unconditionally; the runtime PR replaces the helper
     * body, this call site stays. */
    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    /* SAD-accumulator buffer (single int64). The runtime kernel uses
     * atomic adds, so the runtime PR's `submit` will fillBuffer the
     * accumulator to zero before each dispatch (via the template's
     * `submit_pre_launch` helper). */
    err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx, sizeof(uint64_t));
    if (err != 0) {
        goto fail_after_lc;
    }

    /* Previous-frame slot stays zero in the scaffold — the runtime
     * PR (T8-1b) will allocate a `plane_bytes`-sized MTLBuffer here
     * and host-write the cur ref Y plane into it on each submit so
     * the next frame's kernel can read it as "prev". Pinning the
     * field shape now keeps the runtime PR's diff focused on
     * content, not struct layout. */
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
    MotionV2StateMetal *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Mirrors the HIP / CUDA twins' submit body. The runtime PR
     * (T8-1b) replaces this -ENOSYS return with the live chain:
     *   1. wait on ref_pic ready event,
     *   2. host-write cur ref Y plane into prev_ref's [contents]
     *      pointer (zero-copy on unified memory; replaces the HIP
     *      twin's D2D copy),
     *   3. early return if `index == 0` — first frame has no "prev",
     *      collect emits 0,
     *   4. fillBuffer the device int64 SAD accumulator to zero on
     *      the picture command buffer (intentionally bypasses
     *      `vmaf_metal_kernel_submit_pre_launch` because the wait
     *      was already issued in step 1; the template's helper
     *      would double-issue it),
     *   5. dispatch motion_v2 compute kernel for the active bpc
     *      over (prev_ref, cur_ref),
     *   6. record submit + finished events. No DtoH copy needed
     *      under unified memory — collect reads the accumulator
     *      via `[buffer contents]` directly after the wait. */
    return -ENOSYS;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    MotionV2StateMetal *s = fex->priv;

    /* Drain the private command queue so `[buffer contents]` is safe
     * to read. Mirrors the HIP / CUDA twins. */
    int err = vmaf_metal_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same uint64 SAD
     * sum -> divide by 256.0 / (w*h) chain as the HIP / CUDA
     * references, with the `index == 0` short-circuit emitting 0.0
     * directly (no kernel ran on the first frame). */
    return -ENOSYS;
}

static int flush_fex_metal(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;

    /* Mirrors the HIP / CUDA twins' `flush_fex_*`. The host-only
     * post-pass (`min(score[i], score[i+1])` window) lands with the
     * runtime PR — until then the consumer surfaces 1 ("done, no
     * more flushes") so the feature engine doesn't loop forever
     * when this extractor is enabled in a runtime-not-ready build. */
    return 1;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    MotionV2StateMetal *s = fex->priv;

    /* Lifecycle teardown via the template (drain -> release command
     * queue -> release events). Best-effort error aggregation
     * matches the HIP twin's close path. */
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    /* Previous-frame slot: scaffold has nothing allocated; the
     * runtime PR (T8-1b) will release the MTLBuffer here. */
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

static const char *provided_features[] = {"VMAF_integer_feature_motion_v2_sad_score",
                                          "VMAF_integer_feature_motion2_v2_score", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_metal;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan / HIP feature extractor uses
 * (see e.g. `vmaf_fex_integer_motion_v2_hip` in
 * `libvmaf/src/feature/hip/integer_motion_v2_hip.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_motion_v2_metal = {
    .name = "motion_v2_metal",
    .init = init_fex_metal,
    .submit = submit_fex_metal,
    .collect = collect_fex_metal,
    .flush = flush_fex_metal,
    .close = close_fex_metal,
    .options = options,
    .priv_size = sizeof(MotionV2StateMetal),
    .provided_features = provided_features,
    /* TEMPORAL flag is mandatory: motion_v2 needs the previous-frame
     * carry, so the feature engine has to drive collect *before* the
     * next submit. Mirrors the HIP / CUDA twins verbatim.
     *
     * Intentionally no VMAF_FEATURE_EXTRACTOR_METAL flag yet — the
     * picture buffer-type plumbing for Metal lands with the runtime
     * PR (T8-1b). Until then the consumer registers as a
     * TEMPORAL-only extractor whose `init()` returns -ENOSYS, so any
     * caller asking for `motion_v2_metal` gets a clean "runtime not
     * ready" surface. Same posture as the HIP first consumer
     * (ADR-0241) and the HIP sixth consumer (ADR-0267). */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
