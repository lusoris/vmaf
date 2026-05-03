/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_motion feature extractor on the HIP backend — seventh
 *  consumer of `libvmaf/src/hip/kernel_template.h` (T7-10b
 *  follow-up / ADR-0273).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/float_motion_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations, same `flush()` host-only post-processing tail, and
 *  the same `motion_force_zero` short-circuit posture.
 *
 *  Posture vs prior consumers: a **temporal** extractor with a
 *  blurred-frame ping-pong (`blur[2]`) plus a separate raw-pixel
 *  cache (`ref_in`). The kernel-template models a single device+host
 *  pair, so the three device-only buffers are tracked outside the
 *  template's readback bundle — same shape the CUDA twin uses. Until
 *  the runtime PR (T7-10b) lands a HIP buffer-alloc helper, all
 *  three slots are tracked as opaque `uintptr_t` in the state
 *  struct; the runtime PR will swap them for real device-buffer
 *  handles. Pinning the field shape now makes the runtime PR's
 *  diff a content swap rather than a struct redesign.
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports
 *  "float_motion_hip extractor found but its runtime is not
 *  implemented" rather than "float_motion_hip extractor not found".
 *  The smoke test pins this registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): per-WG SAD float
 *  partials reduced to a double on the host, divided by `w*h` for
 *  the per-frame motion score. `VMAF_feature_motion2_score` is
 *  emitted at index-1 with `min(prev_motion_score, motion_score)`,
 *  with a final tail-frame emission in `flush()`. Mirrors the CUDA
 *  reference exactly.
 */

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"

/* Block dimensions mirror the CUDA twin's `FM_BX`/`FM_BY`. The
 * runtime PR (T7-10b) uses these to size the dispatch grid; pinning
 * them in the scaffold so the cross-backend numeric gate has a
 * single value to compare against. */
#define FMH_BX 16u
#define FMH_BY 16u

typedef struct FloatMotionStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and
     * the (device per-WG SAD float partials, pinned host readback
     * slot) pair are managed by `hip/kernel_template.h` (T7-10b
     * seventh consumer / ADR-0273). The struct shape mirrors the
     * CUDA twin's `FloatMotionStateCuda` — same fields in the same
     * order, modulo the `*_hip` -> `*_cuda` type names and the
     * absence of the CUDA-driver-table function-pointer slots
     * (`funcbpc8`, `funcbpc16`), which the runtime PR (T7-10b) will
     * reintroduce as their HIP equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;

    /* Raw-pixel cache (`ref_in`) plus blurred-frame ping-pong
     * (`blur[cur_blur]`, `blur[1 - cur_blur]`). All three are
     * device-only buffers tracked outside the template's readback
     * bundle for the same reason `motion_v2_hip` keeps its
     * `pix[2]` slots separate: the template models a single
     * device+host pair only. Tracked as `uintptr_t` slots because
     * the HIP scaffold has no device-buffer allocator yet; the
     * runtime PR (T7-10b) will swap each slot for a real
     * `VmafHipBuffer *` (or equivalent handle). The CUDA twin
     * carries `VmafCudaBuffer *ref_in` + `VmafCudaBuffer *blur[2]`
     * field-for-field. */
    uintptr_t ref_in;
    uintptr_t blur[2];
    int cur_blur;
    unsigned wg_count;

    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    double prev_motion_score;
    bool debug;
    bool motion_force_zero;

    VmafDictionary *feature_name_dict;
} FloatMotionStateHip;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FloatMotionStateHip, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .help = "force motion score to zero",
        .offset = offsetof(FloatMotionStateHip, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0},
};

static int extract_force_zero(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                              VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                              VmafPicture *dist_pic_90, unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    FloatMotionStateHip *s = fex->priv;

    int err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_motion2_score", 0.0, index);
    if (s->debug && err == 0) {
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_motion_score", 0.0, index);
    }
    return err;
}

/* Extracted from init: motion_force_zero short-circuit. */
static int init_force_zero_hip(VmafFeatureExtractor *fex, FloatMotionStateHip *s)
{
    fex->extract = extract_force_zero;
    fex->submit = NULL;
    fex->collect = NULL;
    fex->flush = NULL;
    fex->close = NULL;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        return -ENOMEM;
    }
    return 0;
}

/* Extracted from init: per-frame state defaults + WG-count math. */
static void init_state_hip(FloatMotionStateHip *s, unsigned w, unsigned h, unsigned bpc)
{
    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->index = 0;
    s->prev_motion_score = 0.0;
    s->cur_blur = 0;
    const unsigned gx = (w + FMH_BX - 1u) / FMH_BX;
    const unsigned gy = (h + FMH_BY - 1u) / FMH_BY;
    s->wg_count = gx * gy;
    s->ref_in = 0;
    s->blur[0] = 0;
    s->blur[1] = 0;
}

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatMotionStateHip *s = fex->priv;

    init_state_hip(s, w, h, bpc);

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }
    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    if (s->motion_force_zero) {
        err = init_force_zero_hip(fex, s);
        if (err != 0) {
            goto fail_after_lc;
        }
        return 0;
    }

    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, (size_t)s->wg_count * sizeof(float));
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
    (void)dist_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    FloatMotionStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Mirrors the CUDA twin's submit body. The runtime PR (T7-10b)
     * replaces this -ENOSYS return with the live chain:
     *   1. wait on ref_pic ready event,
     *   2. D2D copy of cur ref Y plane into `ref_in` (raw-pixel
     *      cache for the next frame's "prev"),
     *   3. launch float_motion kernel for the active bpc — writes
     *      blurred output to `blur[cur_blur]`, reads
     *      `blur[1 - cur_blur]` as "prev", emits per-WG SAD float
     *      partials,
     *   4. record submit + finished events, DtoH copy of partials
     *      (only when `index > 0`; first frame has no "prev" so
     *      no SAD computation).
     * Submit intentionally bypasses
     * `vmaf_hip_kernel_submit_pre_launch` because the kernel writes
     * per-WG partials directly — no atomic, no memset. Same posture
     * as `ciede_hip` (ADR-0259) and `float_ansnr_hip` (ADR-0266). */
    return -ENOSYS;
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    FloatMotionStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer
     * is safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same per-WG float
     * partials -> double reduction -> divide by `w*h` chain as the
     * CUDA reference. The first-frame (`index == 0`) emits 0.0 for
     * both motion / motion2; index 1 emits motion only (no prev
     * yet); index >= 2 emits `motion2 = min(prev, cur)` at
     * `index - 1`. Tail frame's motion2 is emitted in `flush()`. */
    s->cur_blur = 1 - s->cur_blur;
    return -ENOSYS;
}

static int flush_fex_hip(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)feature_collector;

    /* Mirrors the CUDA twin's `flush_fex_cuda` shape. The host-only
     * post-pass (final-frame motion2 emission with the cached
     * `prev_motion_score`) lands with the runtime PR — until then
     * the consumer surfaces 1 ("done, no more flushes") so the
     * feature engine doesn't loop forever when this extractor is
     * enabled in a runtime-not-ready build. The CUDA twin returns
     * `(ret < 0) ? ret : !ret`; the scaffold's degenerate
     * "no-frames-collected" early return is preserved in shape,
     * with the body replaced by the -ENOSYS-equivalent posture
     * (no scores were collected, so there's nothing to fold). */
    return 1;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatMotionStateHip *s = fex->priv;

    /* Lifecycle teardown via the template (sync -> destroy stream
     * -> destroy events). Best-effort error aggregation matches the
     * CUDA twin's close path. */
    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

    /* Raw-pixel cache + blurred-frame ping-pong slots: scaffold has
     * nothing allocated; the runtime PR (T7-10b) will free real
     * device handles here in the same order the CUDA twin uses
     * (ref_in then blur[0] then blur[1], best-effort). */
    s->ref_in = 0;
    s->blur[0] = 0;
    s->blur[1] = 0;

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

static const char *provided_features[] = {"VMAF_feature_motion_score", "VMAF_feature_motion2_score",
                                          NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_motion_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_motion_cuda` in
 * `libvmaf/src/feature/cuda/float_motion_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_motion_hip = {
    .name = "float_motion_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .flush = flush_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatMotionStateHip),
    .provided_features = provided_features,
    /* TEMPORAL flag is mandatory: float_motion needs the
     * previous-frame blur carry, so the feature engine has to drive
     * collect *before* the next submit. Mirrors the CUDA twin
     * verbatim.
     *
     * Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * TEMPORAL-only extractor whose `init()` returns -ENOSYS, so
     * any caller asking for `float_motion_hip` gets a clean
     * "runtime not ready" surface. Same posture as the first /
     * second / third / fourth / fifth / sixth consumers (ADR-0241
     * / ADR-0254 / ADR-0259 / ADR-0260 / ADR-0266 / ADR-0267). */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
