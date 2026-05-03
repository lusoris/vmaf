/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  motion_v2 feature extractor on the HIP backend — sixth consumer
 *  of `libvmaf/src/hip/kernel_template.h` (T7-10b follow-up /
 *  ADR-0267).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_motion_v2_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations, same `flush()` host-only post-processing. Stateless
 *  variant of the classic motion kernel: exploits convolution
 *  linearity (`SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)`)
 *  so each frame computes its score in one kernel launch over
 *  (prev_ref - cur_ref) without storing blurred frames across submits.
 *
 *  Unique posture vs the prior consumers: a **temporal** extractor
 *  (`VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag) carrying a raw-pixel
 *  ping-pong buffer pair (`pix[2]`) outside the kernel template's
 *  readback bundle — the template models a single device+host pair,
 *  not a ping-pong of device-only buffers. The CUDA twin makes the
 *  same call. Until the runtime PR (T7-10b) lands a HIP buffer-alloc
 *  helper, the ping-pong slots are tracked as opaque `uintptr_t` in
 *  the state struct; the runtime PR will swap them for real
 *  device-buffer handles. Pinning the field shape now makes the
 *  runtime PR's diff a content swap rather than a struct redesign.
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports
 *  "motion_v2_hip extractor found but its runtime is not implemented"
 *  rather than "motion_v2_hip extractor not found". The smoke test
 *  pins this registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): single int64 atomic
 *  SAD accumulator per frame, divided by 256.0 / (w*h) on the host
 *  to recover `VMAF_integer_feature_motion_v2_sad_score`.
 *  `VMAF_integer_feature_motion2_v2_score = min(score[i], score[i+1])`
 *  is emitted host-side in `flush()` (same shape as
 *  `python/test/integer_motion_v2.c::flush`). Mirrors the CUDA
 *  reference exactly.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"

typedef struct MotionV2StateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device int64 SAD accumulator, pinned host readback slot) pair
     * are managed by `hip/kernel_template.h` (T7-10b sixth consumer /
     * ADR-0267). The struct shape mirrors the CUDA twin's
     * `MotionV2StateCuda` — same fields in the same order, modulo the
     * `*_hip` -> `*_cuda` type names and the absence of the
     * CUDA-driver-table function-pointer slots (`funcbpc8`,
     * `funcbpc16`), which the runtime PR (T7-10b) will reintroduce as
     * their HIP equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;

    /* Ping-pong of raw ref Y planes — `pix[index%2]` is the current
     * frame's slot, `pix[(index+1)%2]` is the previous frame's slot.
     * Tracked as `uintptr_t` slots because the HIP scaffold has no
     * device-buffer allocator yet; the runtime PR (T7-10b) will swap
     * each slot for a real `VmafHipBuffer *` (or equivalent handle)
     * and flip `plane_bytes`-sized D2D copies through them. The
     * ping-pong lives outside the template's readback bundle for
     * the same reason the CUDA twin keeps it separate: the template
     * models a single device+host pair, not a ping-pong of
     * device-only buffers. */
    uintptr_t pix[2];
    size_t plane_bytes;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} MotionV2StateHip;

static const VmafOption options[] = {{0}};

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    MotionV2StateHip *s = fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8u ? 1u : 2u);

    /* Allocate a HIP context — the scaffold's `vmaf_hip_context_new`
     * succeeds today (calloc + struct init); the runtime PR will swap
     * in `hipSetDevice` + handle creation. */
    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    /* Stream + event pair via the template. Scaffold returns -ENOSYS
     * unconditionally; the runtime PR replaces the helper body, this
     * call site stays. */
    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    /* Readback pair (single int64 SAD accumulator + pinned host slot).
     * The runtime kernel uses atomic adds, so the runtime PR's
     * `submit` will memset the accumulator to zero before each
     * dispatch (via the template's `submit_pre_launch` helper). */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, sizeof(uint64_t));
    if (err != 0) {
        goto fail_after_lc;
    }

    /* Ping-pong slots stay zero in the scaffold — the runtime PR
     * (T7-10b) will land a HIP device-buffer allocator and replace
     * these with real handles sized at `plane_bytes`. Pinning the
     * field shape now keeps the runtime PR's diff focused on
     * content, not struct layout. */
    s->pix[0] = 0;
    s->pix[1] = 0;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
        goto fail_after_rb;
    }

    return 0;

fail_after_rb:
    (void)vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
fail_after_lc:
    (void)vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
fail_after_ctx:
    vmaf_hip_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int submit_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                          VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)dist_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    MotionV2StateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Mirrors the CUDA twin's submit body. The runtime PR (T7-10b)
     * replaces this -ENOSYS return with the live chain:
     *   1. wait on ref_pic ready event,
     *   2. D2D copy of cur ref Y plane into pix[index % 2] (so the
     *      next frame can read it as "prev"),
     *   3. early return if `index == 0` — first frame has no "prev",
     *      collect emits 0,
     *   4. memset device int64 SAD accumulator to zero on the
     *      picture stream (intentionally bypasses
     *      `vmaf_hip_kernel_submit_pre_launch` because the wait was
     *      already issued in step 1; the template's helper would
     *      double-issue it),
     *   5. launch motion_v2 kernel for the active bpc over
     *      (pix[prev_idx], pix[cur_idx]),
     *   6. record submit + finished events, DtoH copy of the
     *      accumulator. */
    return -ENOSYS;
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    MotionV2StateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same uint64 SAD
     * sum -> divide by 256.0 / (w*h) chain as the CUDA reference,
     * with the `index == 0` short-circuit emitting 0.0 directly
     * (no kernel ran on the first frame). */
    return -ENOSYS;
}

static int flush_fex_hip(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;

    /* Mirrors the CUDA twin's `flush_fex_cuda`. The host-only
     * post-pass (`min(score[i], score[i+1])` window) lands with the
     * runtime PR — until then the consumer surfaces 1 ("done, no
     * more flushes") so the feature engine doesn't loop forever
     * when this extractor is enabled in a runtime-not-ready build.
     * The CUDA twin returns 1 unconditionally after its loop; the
     * scaffold's degenerate "0-frames-collected" early return
     * (`n_frames < 2`) is preserved in shape, with the loop body
     * replaced by the -ENOSYS-equivalent posture (no scores were
     * collected, so there's nothing to fold). */
    return 1;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    MotionV2StateHip *s = fex->priv;

    /* Lifecycle teardown via the template (sync -> destroy stream ->
     * destroy events). Best-effort error aggregation matches the
     * CUDA twin's close path. */
    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

    /* Ping-pong slots: scaffold has nothing allocated; the runtime
     * PR (T7-10b) will free real device handles here in the same
     * order the CUDA twin uses (pix[0] then pix[1], best-effort). */
    s->pix[0] = 0;
    s->pix[1] = 0;

    int err = vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
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
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"VMAF_integer_feature_motion_v2_sad_score",
                                          "VMAF_integer_feature_motion2_v2_score", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_integer_motion_v2_cuda` in
 * `libvmaf/src/feature/cuda/integer_motion_v2_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_motion_v2_hip = {
    .name = "motion_v2_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .flush = flush_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(MotionV2StateHip),
    .provided_features = provided_features,
    /* TEMPORAL flag is mandatory: motion_v2 needs the previous-frame
     * carry, so the feature engine has to drive collect *before* the
     * next submit. Mirrors the CUDA twin verbatim.
     *
     * Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime PR
     * (T7-10b). Until then the consumer registers as a TEMPORAL-only
     * extractor whose `init()` returns -ENOSYS, so any caller asking
     * for `motion_v2_hip` gets a clean "runtime not ready" surface.
     * Same posture as the first / second / third / fourth / fifth
     * consumers (ADR-0241 / ADR-0254 / ADR-0259 / ADR-0260 /
     * ADR-0266). */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
