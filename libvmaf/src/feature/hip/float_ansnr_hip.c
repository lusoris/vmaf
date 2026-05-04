/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ansnr feature extractor on the HIP backend — fifth consumer
 *  of `libvmaf/src/hip/kernel_template.h` (T7-10b follow-up /
 *  ADR-0266).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/float_ansnr_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations. Single-dispatch kernel; the runtime PR will produce
 *  per-block (sig, noise) interleaved float partials and reduce them
 *  on the host in `double` to recover `float_ansnr` and
 *  `float_anpsnr` — the same precision posture as the CUDA twin
 *  (per-block sums in float, cross-block reduction in double).
 *
 *  Like the third consumer (`ciede_hip`), the submit path
 *  intentionally bypasses `vmaf_hip_kernel_submit_pre_launch` because
 *  the kernel writes per-block partials directly (no atomic, no
 *  memset prerequisite). The CUDA twin makes the same choice; the
 *  bypass is the load-bearing artefact this consumer pins.
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports
 *  "float_ansnr_hip extractor found but its runtime is not
 *  implemented" rather than "float_ansnr_hip extractor not found".
 *  The smoke test pins this registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): per-block (sig, noise)
 *  float partials reduced to two doubles on the host, then the CPU
 *  formulas:
 *    float_ansnr  = 10 * log10(sig / noise)              (or psnr_max if noise == 0)
 *    float_anpsnr = MIN(10 * log10(peak^2 * w * h / max(noise, 1e-10)), psnr_max)
 *  v1 emits two features (`float_ansnr`, `float_anpsnr`) just like
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

typedef struct FloatAnsnrStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device per-block (sig, noise) float partials, pinned host
     * readback slot) pair are managed by `hip/kernel_template.h`
     * (T7-10b fifth consumer / ADR-0266). The struct shape mirrors
     * the CUDA twin's `FloatAnsnrStateCuda` — same fields in the same
     * order, modulo the `*_hip` -> `*_cuda` type names and the
     * absence of the CUDA-driver-table function-pointer slots
     * (`funcbpc8`, `funcbpc16`), which the runtime PR (T7-10b) will
     * reintroduce as their HIP equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;
    unsigned wg_count;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    double peak;
    double psnr_max;
    VmafDictionary *feature_name_dict;
} FloatAnsnrStateHip;

/* Mirrors the CUDA twin's 16x16 workgroup tile. Kept verbatim so the
 * runtime PR's `wg_count` math agrees with the CUDA reference. */
#define ANSNR_HIP_BX 16
#define ANSNR_HIP_BY 16

static const VmafOption options[] = {{0}};

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatAnsnrStateHip *s = fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;

    /* peak / psnr_max table mirrors the CUDA twin verbatim. */
    if (bpc == 8u) {
        s->peak = 255.0;
        s->psnr_max = 60.0;
    } else if (bpc == 10u) {
        s->peak = 255.75;
        s->psnr_max = 72.0;
    } else if (bpc == 12u) {
        s->peak = 255.9375;
        s->psnr_max = 84.0;
    } else if (bpc == 16u) {
        s->peak = 255.99609375;
        s->psnr_max = 108.0;
    } else {
        return -EINVAL;
    }

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

    const unsigned grid_x = (w + (ANSNR_HIP_BX - 1u)) / ANSNR_HIP_BX;
    const unsigned grid_y = (h + (ANSNR_HIP_BY - 1u)) / ANSNR_HIP_BY;
    s->wg_count = grid_x * grid_y;

    /* Readback pair (device interleaved (sig, noise) float partials +
     * pinned host slot) sized at 2 floats per workgroup so T7-10b's
     * host reduction sees identical partial counts to the CUDA twin. */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, (size_t)s->wg_count * 2u * sizeof(float));
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
    FloatAnsnrStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Mirrors the CUDA twin's intentionally-inlined pre-launch path —
     * the kernel writes interleaved (sig, noise) float partials per
     * block (no atomic), so the template's accumulator memset is
     * unnecessary. Until the runtime PR lands, there is no real
     * `hipStreamWaitEvent` to call, so the consumer surfaces -ENOSYS
     * verbatim. The runtime PR (T7-10b) replaces this body with the
     * live `cuMemcpy2D` ref/dis upload + `cuMemsetD8Async` partials
     * reset + dispatch + `hipEventRecord(submit)` +
     * `hipStreamWaitEvent(str, submit)` + `hipMemcpyDtoHAsync` +
     * `hipEventRecord(finished)` chain (see CUDA twin). */
    return -ENOSYS;
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    FloatAnsnrStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same per-block
     * (sig, noise) float partials -> double host reduction ->
     * 10 * log10(sig / noise) (and the anpsnr peak^2 form) chain as
     * the CUDA reference. The cross-block reduction across thousands
     * of partials needs double precision to retain places=4 in the
     * eventual cross-backend numeric gate. */
    return -ENOSYS;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatAnsnrStateHip *s = fex->priv;

    /* Lifecycle teardown via the template (sync -> destroy stream ->
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

static const char *provided_features[] = {"float_ansnr", "float_anpsnr", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_ansnr_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_ansnr_cuda` in
 * `libvmaf/src/feature/cuda/float_ansnr_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ansnr_hip = {
    .name = "float_ansnr_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatAnsnrStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * "CPU-flagged" extractor whose `init()` returns -ENOSYS, so
     * any caller asking for `float_ansnr_hip` gets a clean "runtime
     * not ready" surface. Same posture as the first / second /
     * third / fourth consumers (ADR-0241 / ADR-0254 / ADR-0259 /
     * ADR-0260). */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
