/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ciede2000 feature extractor on the HIP backend — third consumer of
 *  `libvmaf/src/hip/kernel_template.h` (T7-10b follow-up / ADR-0259).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_ciede_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations. Like the CUDA twin, the submit path intentionally
 *  inlines the pre-launch wait rather than calling
 *  `vmaf_hip_kernel_submit_pre_launch` — ciede's kernel writes one
 *  float per block (no atomic), so the template's memset is
 *  unnecessary. The lifecycle / readback / collect helpers still
 *  apply.
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports "ciede_hip
 *  extractor found but its runtime is not implemented" rather than
 *  "ciede_hip extractor not found". The smoke test pins this
 *  registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): per-block float
 *  partials reduced to a single mean ΔE on the host, then the CPU's
 *  logarithmic transform `45 - 20*log10(mean_dE)` produces the
 *  `ciede2000` metric. v1 emits one feature (`ciede2000`) just like
 *  the CUDA reference.
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

typedef struct CiedeStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device per-block float partials, pinned host readback slot)
     * pair are managed by `hip/kernel_template.h` (T7-10b third
     * consumer / ADR-0259). The struct shape mirrors the CUDA twin's
     * `CiedeStateCuda` — same fields in the same order, modulo the
     * `*_hip` -> `*_cuda` type names and the absence of the
     * CUDA-driver-table function-pointer slots (`funcbpc8`,
     * `funcbpc16`), which the runtime PR (T7-10b) will reintroduce as
     * their HIP equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;
    unsigned partials_capacity;
    unsigned partials_count;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    unsigned ss_hor;
    unsigned ss_ver;
    VmafDictionary *feature_name_dict;
} CiedeStateHip;

/* Mirrors the CUDA twin's 16×16 workgroup tile. Kept verbatim so the
 * runtime PR's `partials_count` math agrees with the CUDA reference. */
#define CIEDE_HIP_BX 16
#define CIEDE_HIP_BY 16

static const VmafOption options[] = {{0}};

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        return -EINVAL;
    }
    CiedeStateHip *s = fex->priv;

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

    s->bpc = bpc;
    s->ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P) ? 1u : 0u;
    s->ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P) ? 1u : 0u;

    /* Pre-size partials for the announced (w, h). Submit reuses if
     * the picture is the same size; the runtime PR will reallocate
     * if a larger picture arrives (rare, but the API doesn't pin
     * geometry). */
    const unsigned grid_x = (w + (CIEDE_HIP_BX - 1u)) / CIEDE_HIP_BX;
    const unsigned grid_y = (h + (CIEDE_HIP_BY - 1u)) / CIEDE_HIP_BY;
    s->partials_capacity = grid_x * grid_y;

    /* Readback pair (device float partials + pinned host slot) sized
     * per-WG so T7-10b's host reduction sees identical partial counts
     * to the CUDA twin. */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx,
                                         (size_t)s->partials_capacity * sizeof(float));
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
    CiedeStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    const unsigned grid_x = (s->frame_w + (CIEDE_HIP_BX - 1u)) / CIEDE_HIP_BX;
    const unsigned grid_y = (s->frame_h + (CIEDE_HIP_BY - 1u)) / CIEDE_HIP_BY;
    s->partials_count = grid_x * grid_y;

    /* Mirrors the CUDA twin's intentionally-inlined pre-launch wait —
     * ciede's kernel writes one float per block (no atomic), so the
     * template's memset is unnecessary. Until the runtime PR lands,
     * there is no real `hipStreamWaitEvent` to call, so the consumer
     * surfaces -ENOSYS verbatim. The runtime PR (T7-10b) replaces
     * this body with the live `hipStreamWaitEvent` + dispatch +
     * `hipEventRecord(submit)` + `hipStreamWaitEvent(str, submit)` +
     * `hipMemcpyDtoHAsync` + `hipEventRecord(finished)` chain. */
    return -ENOSYS;
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    CiedeStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same per-block float
     * partials → host accumulation in double → mean ΔE → 45 - 20 *
     * log10(mean_dE) chain as the CUDA reference (which carries the
     * same precision argument as ciede_vulkan / ADR-0187: per-block
     * sums fit in float7 precision; the cross-block reduction across
     * thousands of partials needs double to retain places=4). */
    return -ENOSYS;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    CiedeStateHip *s = fex->priv;

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

static const char *provided_features[] = {"ciede2000", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_ciede_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_ciede_cuda` in
 * `libvmaf/src/feature/cuda/integer_ciede_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_ciede_hip = {
    .name = "ciede_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(CiedeStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * "CPU-flagged" extractor whose `init()` returns -ENOSYS, so
     * any caller asking for `ciede_hip` gets a clean "runtime not
     * ready" surface. Same posture as the first/second consumers
     * (ADR-0241 / ADR-0254). */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
