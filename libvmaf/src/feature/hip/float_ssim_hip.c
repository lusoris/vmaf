/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature extractor on the HIP backend — eighth
 *  consumer of `libvmaf/src/hip/kernel_template.h` (T7-10b
 *  follow-up / ADR-0274).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_ssim_cuda.c`
 *  (which despite its filename registers `vmaf_fex_float_ssim_cuda`
 *  and emits `float_ssim`) call-graph-for-call-graph: same
 *  private-state struct shape, same init/submit/collect/close
 *  lifecycle, same template helper invocations, same two-pass
 *  separable-Gaussian dispatch posture (horiz then vert+combine on
 *  the picture stream, implicit happens-before on the same stream).
 *
 *  Posture vs prior consumers: a **two-dispatch-per-frame**
 *  extractor with **five intermediate float buffers** (`h_ref_mu`,
 *  `h_cmp_mu`, `h_ref_sq`, `h_cmp_sq`, `h_refcmp`) tracked outside
 *  the kernel-template's readback bundle — same reasoning as
 *  `motion_v2_hip` and `float_motion_hip`: the template models a
 *  single device+host pair, not a multi-buffer pyramid. Until the
 *  runtime PR (T7-10b) lands a HIP buffer-alloc helper, all five
 *  slots are tracked as opaque `uintptr_t` in the state struct;
 *  the runtime PR will swap them for real device-buffer handles.
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports
 *  "float_ssim_hip extractor found but its runtime is not
 *  implemented" rather than "float_ssim_hip extractor not found".
 *  The smoke test pins this registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): per-block float
 *  partials reduced to a double on the host, divided by
 *  `(W-10)*(H-10)` for the per-frame `float_ssim` score. Mirrors
 *  the CUDA reference exactly (host accumulates partials in
 *  `double`).
 *
 *  v1: scale=1 only — same constraint as ssim_vulkan / ssim_cuda.
 *  Auto-decimation is rejected at init with -EINVAL via the
 *  `compute_scale` helper. Pinning this constraint in the scaffold
 *  so a caller asking for `float_ssim_hip:scale=2` gets a clean
 *  -EINVAL instead of an unimplemented dispatch attempt.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"
#include "log.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"

/* Block dimensions + Gaussian footprint mirror the CUDA twin's
 * `SSIM_BLOCK_X` / `SSIM_BLOCK_Y` / `SSIM_K`. The runtime PR
 * (T7-10b) uses these to size the dispatch grid; pinning them in
 * the scaffold so the cross-backend numeric gate has a single
 * value to compare against. */
#define SSIM_HIP_BLOCK_X 16u
#define SSIM_HIP_BLOCK_Y 8u
#define SSIM_HIP_K 11u

typedef struct SsimStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and
     * the (device per-block float partials, pinned host readback
     * slot) pair are managed by `hip/kernel_template.h` (T7-10b
     * eighth consumer / ADR-0274). The struct shape mirrors the
     * CUDA twin's `SsimStateCuda` — same fields in the same order,
     * modulo the `*_hip` -> `*_cuda` type names and the absence of
     * the CUDA-driver-table function-pointer slots
     * (`func_horiz_8`, `func_horiz_16`, `func_vert`), which the
     * runtime PR (T7-10b) will reintroduce as their HIP
     * equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;

    int scale_override;

    /* Five intermediate float buffers — kept outside the template's
     * readback bundle since the bundle models a single device+host
     * pair, not a 5-buffer pyramid. Tracked as `uintptr_t` slots
     * because the HIP scaffold has no device-buffer allocator yet;
     * the runtime PR (T7-10b) will swap each slot for a real
     * `VmafHipBuffer *` (or equivalent handle) sized at
     * `(W - K + 1) * H * sizeof(float)`. The CUDA twin carries
     * `VmafCudaBuffer *h_{ref_mu,cmp_mu,ref_sq,cmp_sq,refcmp}`
     * field-for-field. */
    uintptr_t h_ref_mu;
    uintptr_t h_cmp_mu;
    uintptr_t h_ref_sq;
    uintptr_t h_cmp_sq;
    uintptr_t h_refcmp;
    unsigned partials_capacity;
    unsigned partials_count;

    unsigned width;
    unsigned height;
    unsigned w_horiz;
    unsigned h_horiz;
    unsigned w_final;
    unsigned h_final;
    unsigned bpc;
    float c1;
    float c2;

    unsigned index;
    VmafDictionary *feature_name_dict;
} SsimStateHip;

static int round_to_int_hip(float x)
{
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}

static int min_int_hip(int a, int b)
{
    return a < b ? a : b;
}

static int compute_scale_hip(unsigned w, unsigned h, int override_val)
{
    if (override_val > 0) {
        return override_val;
    }
    int scaled = round_to_int_hip((float)min_int_hip((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(SsimStateHip, scale_override),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {0},
};

/* Extracted from `init_fex_hip` to keep its branch budget under
 * `readability-function-size`. Mirrors the v1 constraint shape of
 * `integer_ssim_cuda.c::init_fex_cuda`: scale=1 only, footprint at
 * least 11x11. */
static int validate_dims_hip(const SsimStateHip *s, unsigned w, unsigned h)
{
    int scale = compute_scale_hip(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_hip: v1 supports scale=1 only (auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_hip:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < SSIM_HIP_K || h < SSIM_HIP_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_hip: input %ux%u smaller than 11x11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }
    return 0;
}

/* Extracted from `init_fex_hip` to keep its branch budget under
 * `readability-function-size`. Populates the geometry / SSIM-constant
 * fields in the order the CUDA twin does. */
static void init_dims_hip(SsimStateHip *s, unsigned w, unsigned h, unsigned bpc)
{
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (SSIM_HIP_K - 1u);
    s->h_horiz = h;
    s->w_final = w - (SSIM_HIP_K - 1u);
    s->h_final = h - (SSIM_HIP_K - 1u);

    /* SSIM constants: L = (1 << bpc) - 1 in the integer reference;
     * the float CUDA twin pins L = 255.0 / K1 = 0.01 / K2 = 0.03.
     * Mirror those values verbatim so the cross-backend numeric
     * gate (places=4) has nothing fork-specific to track. */
    const float L = 255.0f;
    const float K1 = 0.01f;
    const float K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    const unsigned grid_x = (s->w_final + SSIM_HIP_BLOCK_X - 1u) / SSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_HIP_BLOCK_Y - 1u) / SSIM_HIP_BLOCK_Y;
    s->partials_capacity = grid_x * grid_y;

    /* Five intermediate float buffer slots stay zero in the
     * scaffold — the runtime PR (T7-10b) will land a HIP
     * device-buffer allocator and replace these with real handles
     * sized at `w_horiz * h_horiz * sizeof(float)`. */
    s->h_ref_mu = 0;
    s->h_cmp_mu = 0;
    s->h_ref_sq = 0;
    s->h_cmp_sq = 0;
    s->h_refcmp = 0;
}

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    SsimStateHip *s = fex->priv;

    /* v1 constraint mirror of the CUDA twin: scale=1 only + at least
     * an 11x11 Gaussian footprint. Extracted into `validate_dims_hip`
     * for the readability-function-size budget. */
    int err = validate_dims_hip(s, w, h);
    if (err != 0) {
        return err;
    }

    /* Allocate a HIP context — the scaffold's `vmaf_hip_context_new`
     * succeeds today (calloc + struct init); the runtime PR will
     * swap in `hipSetDevice` + handle creation. */
    err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    /* Stream + event pair via the template — scaffold body returns
     * -ENOSYS unconditionally. The runtime PR replaces the helper
     * body; this call site stays. */
    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    init_dims_hip(s, w, h, bpc);

    /* Readback pair (per-block float partials + pinned host slot).
     * The runtime kernel writes per-block partials directly (no
     * atomic, no memset prerequisite) — same posture as
     * `ciede_hip` (ADR-0259), `float_ansnr_hip` (ADR-0266), and
     * `float_motion_hip` (ADR-0273). Submit will bypass
     * `vmaf_hip_kernel_submit_pre_launch`. */
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
    SsimStateHip *s = fex->priv;

    s->index = index;
    (void)ref_pic;
    const unsigned grid_x = (s->w_final + SSIM_HIP_BLOCK_X - 1u) / SSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_HIP_BLOCK_Y - 1u) / SSIM_HIP_BLOCK_Y;
    s->partials_count = grid_x * grid_y;

    /* Mirrors the CUDA twin's submit body. The runtime PR (T7-10b)
     * replaces this -ENOSYS return with the live two-pass dispatch
     * chain:
     *   1. wait for dist-side ready event on the picture stream,
     *   2. Pass 1 — horizontal 11-tap separable Gaussian over
     *      ref / cmp / ref^2 / cmp^2 / ref*cmp into the five
     *      intermediate float buffers (grid sized over (W-10) x H),
     *   3. Pass 2 — vertical 11-tap + per-pixel SSIM combine +
     *      per-block float partial sum (grid sized over
     *      (W-10) x (H-10), implicit happens-before on the same
     *      picture stream),
     *   4. record submit + finished events, DtoH copy of partials.
     * Submit intentionally bypasses
     * `vmaf_hip_kernel_submit_pre_launch` because the kernel writes
     * per-block partials directly — no atomic, no memset. Same
     * posture as `ciede_hip` (ADR-0259), `float_ansnr_hip`
     * (ADR-0266), and `float_motion_hip` (ADR-0273). */
    return -ENOSYS;
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    (void)index;
    SsimStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer
     * is safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same per-block
     * float partials -> double reduction -> divide by
     * `(W-10)*(H-10)` chain as the CUDA reference, emitting one
     * feature (`float_ssim`). */
    return -ENOSYS;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    SsimStateHip *s = fex->priv;

    /* Lifecycle teardown via the template (sync -> destroy stream
     * -> destroy events). Best-effort error aggregation matches the
     * CUDA twin's close path. */
    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

    /* Five intermediate float buffer slots: scaffold has nothing
     * allocated; the runtime PR (T7-10b) will free real device
     * handles here in the same order the CUDA twin uses. */
    s->h_ref_mu = 0;
    s->h_cmp_mu = 0;
    s->h_ref_sq = 0;
    s->h_cmp_sq = 0;
    s->h_refcmp = 0;

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

static const char *provided_features[] = {"float_ssim", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_ssim_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_ssim_cuda` in
 * `libvmaf/src/feature/cuda/integer_ssim_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ssim_hip = {
    .name = "float_ssim_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(SsimStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * "CPU-flagged" extractor whose `init()` returns -ENOSYS, so
     * any caller asking for `float_ssim_hip` gets a clean
     * "runtime not ready" surface. Same posture as the first /
     * second / third / fourth / fifth / sixth / seventh consumers
     * (ADR-0241 / ADR-0254 / ADR-0259 / ADR-0260 / ADR-0266 /
     * ADR-0267 / ADR-0273). */
    .flags = 0,
    /* 2 dispatches/frame (horiz + vert+combine), reduction-only
     * after the second pass; AUTO + 1080p area matches the CUDA
     * twin's profile. */
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
