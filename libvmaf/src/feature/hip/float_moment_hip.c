/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_moment feature extractor on the HIP backend — fourth consumer
 *  of `libvmaf/src/hip/kernel_template.h` (T7-10b follow-up /
 *  ADR-0260).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_moment_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations. Single dispatch per frame; the runtime PR will emit
 *  all four metrics (`float_moment_ref{1st,2nd}`,
 *  `float_moment_dis{1st,2nd}`) in one kernel pass via four uint64
 *  atomic counters — that's the precision posture this consumer pins
 *  (int64 atomic accumulators, distinct from the float partials of
 *  the third consumer `ciede_hip` and the float partials of the
 *  second consumer `float_psnr_hip`).
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports
 *  "float_moment_hip extractor found but its runtime is not
 *  implemented" rather than "float_moment_hip extractor not found".
 *  The smoke test pins this registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): four uint64 atomic
 *  counters [ref1, dis1, ref2, dis2] reduced per-frame, divided by
 *  `w*h` on the host to recover four mean / second-moment metrics.
 *  Mirrors the CUDA reference exactly.
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

typedef struct MomentStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device four-uint64 accumulator, pinned host readback slot) pair
     * are managed by `hip/kernel_template.h` (T7-10b fourth consumer /
     * ADR-0260). The struct shape mirrors the CUDA twin's
     * `MomentStateCuda` — same fields in the same order, modulo the
     * `*_hip` -> `*_cuda` type names and the absence of the
     * CUDA-driver-table function-pointer slots (`funcbpc8`,
     * `funcbpc16`), which the runtime PR (T7-10b) will reintroduce
     * as their HIP equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    VmafDictionary *feature_name_dict;
} MomentStateHip;

/* Number of uint64 atomic counters the runtime kernel emits per
 * frame: [ref1st, dis1st, ref2nd, dis2nd]. Pinned by the CUDA twin's
 * `cuda/integer_moment_cuda.c` so the eventual cross-backend numeric
 * gate has nothing fork-specific to track. */
#define MOMENT_HIP_COUNTERS 4u

static const VmafOption options[] = {{0}};

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)w;
    (void)h;
    MomentStateHip *s = fex->priv;

    /* Allocate a HIP context — the scaffold's `vmaf_hip_context_new`
     * succeeds today (calloc + struct init); the runtime PR will swap
     * in `hipSetDevice` + handle creation. */
    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    /* Stream + event pair via the template — scaffold body returns
     * -ENOSYS unconditionally. The consumer surfaces that verbatim so
     * a caller running `vmaf --feature float_moment_hip` sees a
     * runtime-not-ready signal at init time. The runtime PR replaces
     * the helper body; this call site stays. */
    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    s->bpc = bpc;

    /* Readback pair (device four-uint64 accumulator + pinned host
     * slot). The runtime kernel uses atomic adds, so the runtime PR's
     * `submit` will memset the accumulator to zero before each
     * dispatch (via the template's `submit_pre_launch` helper). */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx,
                                         (size_t)MOMENT_HIP_COUNTERS * sizeof(uint64_t));
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
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)dist_pic;
    MomentStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Mirrors the CUDA twin's pre-launch boilerplate: zero the four
     * device counters and wait for the dist-side ready event. The
     * kernel uses atomic adds, so the memset is mandatory. The
     * runtime PR will replace the scaffold-only -ENOSYS return with
     * a real dispatch + readback chain (memset accumulator, wait on
     * dist ready event, launch moment kernel for the active bpc,
     * record submit/finished events, DtoH copy). The picture stream
     * + ready-event handles are threaded through as `uintptr_t` to
     * match the kernel template's header-pure ABI. */
    int err = vmaf_hip_kernel_submit_pre_launch(&s->lc, s->ctx, &s->rb,
                                                /* picture_stream */ 0,
                                                /* dist_ready_event */ 0);
    if (err != 0) {
        return err;
    }

    /* Launch + post-launch event sequence land with the runtime PR
     * (T7-10b). Mirrors the CUDA twin's bpc8/bpc16 `cuLaunchKernel`
     * dispatch + `hipEventRecord(submit)` +
     * `hipStreamWaitEvent(str, submit)` + `hipMemcpyDtoHAsync` +
     * `hipEventRecord(finished)` chain. */
    return -ENOSYS;
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    MomentStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same uint64 atomic
     * sums → divide by w*h chain as the CUDA reference, emitting
     * four features (`float_moment_ref1st`, `float_moment_dis1st`,
     * `float_moment_ref2nd`, `float_moment_dis2nd`) per frame. */
    return -ENOSYS;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    MomentStateHip *s = fex->priv;

    /* Lifecycle teardown via the template (sync → destroy stream →
     * destroy events). Best-effort error aggregation matches the
     * CUDA twin's close path. */
    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
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

static const char *provided_features[] = {
    "float_moment_ref1st",
    "float_moment_dis1st",
    "float_moment_ref2nd",
    "float_moment_dis2nd",
    NULL,
};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_moment_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_moment_cuda` in
 * `libvmaf/src/feature/cuda/integer_moment_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_moment_hip = {
    .name = "float_moment_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(MomentStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * "CPU-flagged" extractor whose `init()` returns -ENOSYS, so
     * any caller asking for `float_moment_hip` gets a clean
     * "runtime not ready" surface. Same posture as the first /
     * second / third consumers (ADR-0241 / ADR-0254 / ADR-0259). */
    .flags = 0,
    /* 1 dispatch/frame, reduction-dominated; AUTO + 1080p area
     * matches the CUDA twin's profile. */
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
