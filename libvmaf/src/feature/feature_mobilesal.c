/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  MobileSal saliency feature extractor (T6-2a). No-reference scoring-side
 *  surface that runs a tiny ONNX saliency model over the distorted frame
 *  and emits the mean of its per-pixel saliency map as a scalar feature.
 *
 *  Wire-up:
 *
 *    YUV420P/422P/444P 8-bit (distorted)
 *       → upsample chroma to 4:4:4 (shared pattern with feature_lpips.c)
 *       → BT.709 limited-range YUV → RGB (uint8 planes)
 *       → vmaf_tensor_from_rgb_imagenet()  (single input)
 *       → vmaf_dnn_session_run() with named binding "input" → "saliency_map"
 *       → reduce mean over the [1,1,H,W] map → scalar "saliency_mean"
 *
 *  Output contract (mirrors the upstream MobileSal paper, "MobileSal:
 *  Extremely Efficient RGB-D Salient Object Detection"):
 *
 *    input         float32[1, 3, H, W]   ImageNet-normalised RGB, NCHW
 *    saliency_map  float32[1, 1, H, W]   per-pixel saliency in [0, 1]
 *
 *  Scope (T6-2a)
 *  -------------
 *  This is the *scoring-side* extractor only — it emits ``saliency_mean``
 *  (a single scalar per frame) so downstream tools can correlate
 *  saliency mass against quality. The full saliency-weighted residual
 *  pooling (`saliency_weighted_l1`) and the `tools/vmaf-roi` per-CTU
 *  QP-offset sidecar are deliberate follow-ups in T6-2b.
 *
 *  Model resolution
 *  ----------------
 *  The ONNX path is provided via the feature option ``model_path``; if
 *  unset, the env var ``VMAF_MOBILESAL_MODEL_PATH`` is consulted as a
 *  fallback. Missing model → init() returns -EINVAL so the pipeline
 *  cleanly declines instead of running on a stub.
 *
 *  When libvmaf is built with -Denable_dnn=false the session-open call
 *  returns -ENOSYS and init() propagates that, so this extractor simply
 *  cannot be instantiated without a real ORT build.
 *
 *  Checkpoint provenance: the in-tree ``model/tiny/mobilesal.onnx`` is a
 *  smoke-only synthetic placeholder (3→1 Conv + Sigmoid) that matches
 *  the I/O contract. Real upstream MobileSal weights are tracked as the
 *  T6-2a-followup task. See ``docs/ai/models/mobilesal.md`` and
 *  ``docs/adr/0218-mobilesal-saliency-extractor.md``.
 */

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/dnn.h"
#include "libvmaf/picture.h"

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "log.h"
#include "mem.h"
#include "opt.h"

#include "dnn/tensor_io.h"

typedef struct MobilesalState {
    char *model_path; /**< feature option, owned by opt.c */
    VmafDnnSession *sess;
    unsigned w, h;
    /* Scratch RGB planes (uint8) + ImageNet-normalised float tensor,
     * sized once at init and reused per-frame. The saliency output map
     * is also sized once — [1, 1, H, W] floats. */
    uint8_t *rgb8[3];
    float *tensor_in; /**< 3 * w * h floats, NCHW */
    float *sal_map;   /**< 1 * w * h floats, NCHW [1,1,H,W] */
} MobilesalState;

/* BT.709 limited-range YUV → RGB in [0, 255]. Mirrors feature_lpips.c so
 * the two extractors stay numerically comparable on identical inputs. */
static inline void yuv_bt709_to_rgb8_pixel(int y, int u, int v, uint8_t *r, uint8_t *g, uint8_t *b)
{
    const float yn = ((float)y - 16.0f) * (1.0f / 219.0f);
    const float un = ((float)u - 128.0f) * (1.0f / 224.0f);
    const float vn = ((float)v - 128.0f) * (1.0f / 224.0f);

    float rf = yn + 1.28033f * vn;
    float gf = yn - 0.21482f * un - 0.38059f * vn;
    float bf = yn + 2.12798f * un;

    if (rf < 0.0f) {
        rf = 0.0f;
    } else if (rf > 1.0f) {
        rf = 1.0f;
    }
    if (gf < 0.0f) {
        gf = 0.0f;
    } else if (gf > 1.0f) {
        gf = 1.0f;
    }
    if (bf < 0.0f) {
        bf = 0.0f;
    } else if (bf > 1.0f) {
        bf = 1.0f;
    }

    *r = (uint8_t)(rf * 255.0f + 0.5f);
    *g = (uint8_t)(gf * 255.0f + 0.5f);
    *b = (uint8_t)(bf * 255.0f + 0.5f);
}

/* Expand planar YUV (4:2:0 / 4:2:2 / 4:4:4, 8-bit) to RGB planes — same
 * nearest-neighbour chroma upsample as feature_lpips.c / ciede.c. */
static int yuv8_to_rgb8_planes(const VmafPicture *pic, uint8_t *dst_r, uint8_t *dst_g,
                               uint8_t *dst_b)
{
    if (pic->bpc != 8) {
        return -ENOTSUP;
    }
    const int ss_hor = (pic->pix_fmt != VMAF_PIX_FMT_YUV444P) ? 1 : 0;
    const int ss_ver = (pic->pix_fmt == VMAF_PIX_FMT_YUV420P) ? 1 : 0;
    const unsigned w = pic->w[0];
    const unsigned h = pic->h[0];
    const uint8_t *Y = pic->data[0];
    const uint8_t *U = pic->data[1];
    const uint8_t *V = pic->data[2];
    const size_t sy = pic->stride[0];
    const size_t su = pic->stride[1];
    const size_t sv = pic->stride[2];

    for (unsigned i = 0; i < h; ++i) {
        const uint8_t *yrow = Y + (size_t)i * sy;
        const unsigned ci = ss_ver ? (i >> 1) : i;
        const uint8_t *urow = U + (size_t)ci * su;
        const uint8_t *vrow = V + (size_t)ci * sv;
        uint8_t *rrow = dst_r + (size_t)i * w;
        uint8_t *grow = dst_g + (size_t)i * w;
        uint8_t *brow = dst_b + (size_t)i * w;
        for (unsigned j = 0; j < w; ++j) {
            const unsigned cj = ss_hor ? (j >> 1) : j;
            yuv_bt709_to_rgb8_pixel(yrow[j], urow[cj], vrow[cj], rrow + j, grow + j, brow + j);
        }
    }
    return 0;
}

static void mobilesal_release(MobilesalState *s)
{
    if (!s) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        if (s->rgb8[i]) {
            aligned_free(s->rgb8[i]);
            s->rgb8[i] = NULL;
        }
    }
    if (s->tensor_in) {
        aligned_free(s->tensor_in);
        s->tensor_in = NULL;
    }
    if (s->sal_map) {
        aligned_free(s->sal_map);
        s->sal_map = NULL;
    }
    if (s->sess) {
        (void)vmaf_dnn_session_close(s->sess);
        s->sess = NULL;
    }
}

static int mobilesal_init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    MobilesalState *s = fex->priv;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        return -EINVAL;
    }
    if (bpc != 8) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "mobilesal: bpc=%u not supported yet (8-bit only in this build)\n", bpc);
        return -ENOTSUP;
    }

    const char *path = s->model_path;
    if (!path || !*path) {
        const char *env = getenv("VMAF_MOBILESAL_MODEL_PATH");
        if (env && *env) {
            path = env;
        }
    }
    if (!path || !*path) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "mobilesal: no model path (set feature option model_path or env "
                 "VMAF_MOBILESAL_MODEL_PATH)\n");
        return -EINVAL;
    }

    int rc = vmaf_dnn_session_open(&s->sess, path, NULL);
    if (rc < 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "mobilesal: vmaf_dnn_session_open(%s) failed: %d\n", path,
                 rc);
        return rc;
    }
    assert(s->sess != NULL);

    s->w = w;
    s->h = h;
    const size_t plane = (size_t)w * (size_t)h;

    for (int i = 0; i < 3; ++i) {
        s->rgb8[i] = (uint8_t *)aligned_malloc(plane, 32);
        if (!s->rgb8[i]) {
            goto oom;
        }
    }
    s->tensor_in = (float *)aligned_malloc(3u * plane * sizeof(float), 32);
    if (!s->tensor_in) {
        goto oom;
    }
    s->sal_map = (float *)aligned_malloc(plane * sizeof(float), 32);
    if (!s->sal_map) {
        goto oom;
    }

    return 0;

oom:
    mobilesal_release(s);
    return -ENOMEM;
}

static int mobilesal_extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                             VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                             VmafPicture *dist_pic_90, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    MobilesalState *s = fex->priv;
    assert(s != NULL);
    assert(s->sess != NULL);

    if (dist_pic->w[0] != s->w || dist_pic->h[0] != s->h) {
        return -ERANGE;
    }

    int rc = yuv8_to_rgb8_planes(dist_pic, s->rgb8[0], s->rgb8[1], s->rgb8[2]);
    if (rc < 0) {
        return rc;
    }

    rc = vmaf_tensor_from_rgb_imagenet(s->rgb8[0], s->w, s->rgb8[1], s->w, s->rgb8[2], s->w,
                                       (int)s->w, (int)s->h, s->tensor_in);
    if (rc < 0) {
        return rc;
    }

    const int64_t in_shape[4] = {1, 3, (int64_t)s->h, (int64_t)s->w};
    const size_t plane = (size_t)s->w * (size_t)s->h;
    VmafDnnInput inputs[1] = {
        {.name = "input", .data = s->tensor_in, .shape = in_shape, .rank = 4u},
    };
    VmafDnnOutput outputs[1] = {
        {.name = "saliency_map", .data = s->sal_map, .capacity = plane, .written = 0u},
    };

    rc = vmaf_dnn_session_run(s->sess, inputs, 1u, outputs, 1u);
    if (rc < 0) {
        return rc;
    }
    if (outputs[0].written == 0u) {
        return -EIO;
    }

    /* Reduce-mean over the saliency map. The session writes ``written``
     * elements; clamp to the allocated plane size to be defensive
     * against a model returning a partial tile (CERT INT30-C). */
    size_t n = outputs[0].written;
    if (n > plane) {
        n = plane;
    }
    if (n == 0u) {
        return -EIO;
    }
    double sum = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        sum += (double)s->sal_map[i];
    }
    const double mean = sum / (double)n;

    return vmaf_feature_collector_append(feature_collector, "saliency_mean", mean, index);
}

static int mobilesal_close(VmafFeatureExtractor *fex)
{
    MobilesalState *s = fex->priv;
    if (!s) {
        return 0;
    }
    mobilesal_release(s);
    memset(s, 0, sizeof(*s));
    return 0;
}

static const VmafOption mobilesal_options[] = {
    {
        .name = "model_path",
        .help = "Filesystem path to the MobileSal saliency ONNX model (single input "
                "'input', single output 'saliency_map'). Overrides the "
                "VMAF_MOBILESAL_MODEL_PATH env var.",
        .offset = offsetof(MobilesalState, model_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {NULL},
};

static const char *mobilesal_provided_features[] = {"saliency_mean", NULL};

VmafFeatureExtractor vmaf_fex_mobilesal = {
    .name = "mobilesal",
    .init = mobilesal_init,
    .extract = mobilesal_extract,
    .close = mobilesal_close,
    .options = mobilesal_options,
    .priv_size = sizeof(MobilesalState),
    .provided_features = mobilesal_provided_features,
    /* Explicit zero-init so GCC LTO sees the full struct layout across
     * TUs (silences -Wlto-type-mismatch — see ADR-0181). */
    .chars = {0},
};
