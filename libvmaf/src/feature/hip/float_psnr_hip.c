/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_psnr feature extractor on the HIP backend — second consumer
 *  of `libvmaf/src/hip/kernel_template.h` (T7-10 follow-up /
 *  ADR-0254).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/float_psnr_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations. The only deviation from the CUDA twin is the precision
 *  posture — float per-WG partials instead of `uint64` SSE — which
 *  surfaces only in the eventual runtime PR; the scaffold-only
 *  registration shape is identical to `integer_psnr_hip.c` (the first
 *  consumer, ADR-0241).
 *
 *  The kernel-template helpers in `libvmaf/src/hip/kernel_template.c`
 *  currently return -ENOSYS; the consumer's `init` therefore returns
 *  -ENOSYS up the stack, so the feature engine reports "float_psnr_hip
 *  extractor found but its runtime is not implemented" rather than
 *  "float_psnr_hip extractor not found". The smoke test pins this
 *  registration-shape contract.
 *
 *  When the runtime PR (T7-10b) ships, the kernel-template bodies flip
 *  from -ENOSYS to real `hipStreamCreate` / `hipMemcpyAsync` / ...
 *  calls and *this* TU keeps its current shape verbatim. That's the
 *  load-bearing invariant: the consumer is written against the
 *  template's contract, not against the absent runtime.
 *
 *  Algorithm parity (when the kernel arrives): per-pixel float
 *  squared-error reduction on the float-converted luma plane → host-
 *  side log10 → score, with the bit-depth-aware peak/clamp formula the
 *  CUDA twin uses (peak = (1<<(bpc-1)) - 0.25 / 0.0625 / ..., clamp at
 *  60/72/84/108 dB). v1 emits luma-only `float_psnr` just like the
 *  CUDA reference.
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

typedef struct FloatPsnrStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device float-partials accumulator, pinned host readback slot)
     * pair are managed by `hip/kernel_template.h` (T7-10 second
     * consumer / ADR-0254). The struct shape mirrors the CUDA twin's
     * `FloatPsnrStateCuda` — same fields in the same order, modulo the
     * `*_hip` -> `*_cuda` type names and the absence of the
     * CUDA-driver-table function-pointer slots (`funcbpc8`,
     * `funcbpc16`) and the `VmafCudaBuffer` ref/dis staging buffers,
     * which the runtime PR (T7-10b) will reintroduce as their HIP
     * equivalents. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    unsigned wg_count;
    double peak;
    double psnr_max;
    VmafDictionary *feature_name_dict;
} FloatPsnrStateHip;

/* Mirrors the CUDA twin's `FPSNR_BX` / `FPSNR_BY` workgroup tile size.
 * Kept verbatim so the eventual runtime PR's `wg_count` math agrees
 * with the CUDA reference. */
#define FPSNR_HIP_BX 16
#define FPSNR_HIP_BY 16

static const VmafOption options[] = {{0}};

/* Bit-depth → peak/clamp formula matches the CUDA reference
 * (`libvmaf/src/feature/cuda/float_psnr_cuda.c`). The runtime PR
 * (T7-10b) will exercise these values in the per-frame log10 step;
 * until then they're inert state, but we set them up at init() time
 * so the eventual cross-backend numeric gate has nothing fork-
 * specific to track. Extracted into its own function to keep
 * `init_fex_hip` under the `readability-function-size` 60-line
 * threshold without sacrificing the intent-comment density of the
 * mirrored CUDA twin. */
static int float_psnr_hip_resolve_peak_clamp(FloatPsnrStateHip *s, unsigned bpc)
{
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
    return 0;
}

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatPsnrStateHip *s = fex->priv;

    int err = float_psnr_hip_resolve_peak_clamp(s, bpc);
    if (err != 0) {
        return err;
    }

    s->bpc = bpc;
    s->frame_w = w;
    s->frame_h = h;
    /* Workgroup count derived the same way the CUDA twin does. */
    const unsigned gx = (w + FPSNR_HIP_BX - 1u) / FPSNR_HIP_BX;
    const unsigned gy = (h + FPSNR_HIP_BY - 1u) / FPSNR_HIP_BY;
    s->wg_count = gx * gy;

    /* Scaffold's `vmaf_hip_context_new` succeeds today (calloc + struct
     * init); T7-10b will swap in `hipSetDevice` + handle creation. */
    err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    /* Stream + event pair via the template. Scaffold returns -ENOSYS
     * unconditionally so a caller running `vmaf --feature
     * float_psnr_hip` sees a runtime-not-ready signal at init time;
     * T7-10b replaces the helper body, this call site stays. */
    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    /* Readback pair (device float-partials + pinned host slot) sized
     * per-WG so T7-10b's host reduction sees identical partial counts
     * to the CUDA twin. */
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
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)dist_pic;
    FloatPsnrStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    /* Mirrors the CUDA twin's pre-launch boilerplate. The runtime PR
     * will replace the scaffold-only -ENOSYS return with a real
     * dispatch + readback chain (zero accumulator, wait on dist
     * ready event, two `hipMemcpy2DAsync` ref/dis stages, launch
     * float_psnr kernel for the active bpc, record submit/finished
     * events, DtoH copy). The picture stream + ready-event handles
     * are threaded through as `uintptr_t` to match the kernel
     * template's header-pure ABI. */
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
    FloatPsnrStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

    /* Score emission lands with the runtime PR — same float-partials
     * → noise → 10*log10((peak^2)/max(noise, eps)) clamped to
     * `psnr_max` chain as the CUDA reference. */
    return -ENOSYS;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatPsnrStateHip *s = fex->priv;

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

static const char *provided_features[] = {"float_psnr", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_psnr_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_psnr_cuda` in
 * `libvmaf/src/feature/cuda/float_psnr_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_psnr_hip = {
    .name = "float_psnr_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatPsnrStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * "CPU-flagged" extractor whose `init()` returns -ENOSYS, so
     * any caller asking for `float_psnr_hip` gets a clean "runtime
     * not ready" surface. The flag bit is reserved in
     * `feature_extractor.h` so the runtime PR can adopt it without
     * an enum reshuffle. Same posture as the first consumer
     * (`vmaf_fex_psnr_hip`, ADR-0241). */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
