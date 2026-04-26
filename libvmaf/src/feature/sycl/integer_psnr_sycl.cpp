/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the SYCL backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1b part 2). SYCL twin of psnr_vulkan
 *  (PR #125) and psnr_cuda (PR #129).
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::sse_line_{8,16}):
 *      diff = (int64)ref - (int64)dis;     (per pixel)
 *      sse  += diff * diff;                (atomic int64 reduction)
 *
 *  Single kernel per frame, atomic-int64 reduction into a shared
 *  device counter; host downloads + computes PSNR. Simpler than
 *  motion_sycl because there's no temporal state, no ping-pong,
 *  no separable filter.
 *
 *  Pattern: register with vmaf_sycl_graph_register and ride the
 *  combined-graph submit/wait machinery just like motion_sycl.
 *
 *  v1: luma-only ("psnr_y"), matching the CUDA + Vulkan twins'
 *  scope. The shared frame buffer that vmaf_sycl_shared_frame_init
 *  sets up is luma-only today.
 */

#include <sycl/sycl.hpp>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "picture.h"
#include "sycl/common.h"
}

namespace
{

struct PsnrStateSycl {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    uint32_t peak;
    double psnr_max_y;

    /* SYCL state back-pointer. */
    VmafSyclState *sycl_state;

    /* Device + host accumulators. */
    int64_t *d_sse;
    int64_t *h_sse;

    /* Submit/collect plumbing. */
    bool has_pending;
    unsigned pending_index;

    VmafDictionary *feature_name_dict;
};

/* Per-pixel SSE kernel. Reads the shared ref/dis frame buffers
 * — set up by vmaf_sycl_shared_frame_init at uint8 packing for
 * ≤8bpc and uint16 packing for ≥10bpc, tightly packed at
 * `width * bytes_per_pixel`. Atomic-adds each pixel's int64
 * squared error to the device accumulator. */
static void launch_sse(sycl::queue &q, void *shared_ref, void *shared_dis, int64_t *d_sse,
                       unsigned width, unsigned height, unsigned bpc)
{
    sycl::range<2> global{(size_t)height, (size_t)width};
    const unsigned e_w = width;
    const unsigned e_bpc = bpc;
    void *ref_in = shared_ref;
    void *dis_in = shared_dis;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(global, [=](sycl::id<2> id) {
            const size_t y = id[0];
            const size_t x = id[1];
            const size_t off = y * (size_t)e_w + x;
            int64_t r;
            int64_t d;
            if (e_bpc <= 8) {
                r = (int64_t)static_cast<const uint8_t *>(ref_in)[off];
                d = (int64_t)static_cast<const uint8_t *>(dis_in)[off];
            } else {
                r = (int64_t)static_cast<const uint16_t *>(ref_in)[off];
                d = (int64_t)static_cast<const uint16_t *>(dis_in)[off];
            }
            const int64_t diff = r - d;
            const int64_t se = diff * diff;
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                accum(*d_sse);
            accum.fetch_add(se);
        });
    });
}

/* Pre-graph: zero the SSE accumulator (direct enqueue, outside graph). */
static void psnr_pre_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<PsnrStateSycl *>(priv);
    q.memset(s->d_sse, 0, sizeof(int64_t));
}

/* Graph-recorded: the per-pixel reduction kernel. */
static void enqueue_psnr_work(void *queue_ptr, void *priv, void *shared_ref, void *shared_dis)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<PsnrStateSycl *>(priv);
    launch_sse(q, shared_ref, shared_dis, s->d_sse, s->width, s->height, s->bpc);
}

/* Post-graph: D2H copy of the SSE accumulator (direct enqueue, outside graph). */
static void psnr_post_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<PsnrStateSycl *>(priv);
    q.memcpy(s->h_sse, s->d_sse, sizeof(int64_t));
}

/* No per-slot config — psnr is stateless across frames. */
static void config_psnr_slot(void *priv, int slot)
{
    (void)priv;
    (void)slot;
}

} /* anonymous namespace */

extern "C" {

static const VmafOption options_psnr_sycl[] = {{0}};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<PsnrStateSycl *>(fex->priv);

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;
    const double peak_d = (double)s->peak;
    s->psnr_max_y = 10.0 * std::log10((peak_d * peak_d) / 0.5);

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_sycl: no SYCL state\n");
        return -EINVAL;
    }

    VmafSyclState *state = fex->sycl_state;
    s->sycl_state = state;

    int const err = vmaf_sycl_shared_frame_init(state, w, h, bpc);
    if (err)
        return err;

    s->d_sse = static_cast<int64_t *>(vmaf_sycl_malloc_device(state, sizeof(int64_t)));
    s->h_sse = static_cast<int64_t *>(vmaf_sycl_malloc_host(state, sizeof(int64_t)));
    if (!s->d_sse || !s->h_sse) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_sycl: device memory allocation failed\n");
        return -ENOMEM;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    s->has_pending = false;

    int const err2 = vmaf_sycl_graph_register(state, enqueue_psnr_work, psnr_pre_graph,
                                              psnr_post_graph, config_psnr_slot, s, "PSNR");
    if (err2)
        return err2;

    return 0;
}

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;

    auto *s = static_cast<PsnrStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    int const err = vmaf_sycl_graph_submit(state);
    if (err)
        return err;

    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<PsnrStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    vmaf_sycl_graph_wait(state);

    const double sse = (double)*s->h_sse;
    const double n_pixels = (double)s->width * (double)s->height;
    const double mse = sse / n_pixels;
    double psnr_y =
        (sse <= 0.0) ? s->psnr_max_y : 10.0 * std::log10(((double)s->peak * s->peak) / mse);
    if (psnr_y > s->psnr_max_y)
        psnr_y = s->psnr_max_y;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "psnr_y", psnr_y, index);
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<PsnrStateSycl *>(fex->priv);
    if (s->sycl_state) {
        if (s->d_sse)
            vmaf_sycl_free(s->sycl_state, s->d_sse);
        if (s->h_sse)
            vmaf_sycl_free(s->sycl_state, s->h_sse);
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_psnr_sycl[] = {"psnr_y", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_psnr_sycl = {
    .name = "psnr_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_psnr_sycl,
    .priv_size = sizeof(PsnrStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_psnr_sycl,
    /* 1 dispatch/frame, reduction-dominated; AUTO + 1080p area
     * matches motion's profile (see ADR-0181 / ADR-0182). */
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};

} /* extern "C" */
