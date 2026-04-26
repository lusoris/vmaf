/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  LPIPS (Learned Perceptual Image Patch Similarity) full-reference
 *  extractor. Backed by a two-input ONNX model (SqueezeNet backbone) that
 *  accepts ImageNet-normalised RGB planes and emits a scalar perceptual
 *  distance. Wire-up:
 *
 *    YUV420P/422P/444P 8-bit
 *       → upsample chroma to 4:4:4 (shared pattern with ciede.c)
 *       → BT.709 limited-range YUV→RGB (float) → round to uint8 planes
 *       → vmaf_tensor_from_rgb_imagenet()  (ref and dist)
 *       → vmaf_dnn_session_run() with named bindings "ref" / "dist"
 *       → scalar "lpips" feature
 *
 *  Model resolution
 *  ----------------
 *  The ONNX path is provided via the feature option ``model_path``; if
 *  unset, the env var ``VMAF_LPIPS_MODEL_PATH`` is consulted as a
 *  fallback. Missing model → init() returns -EINVAL so the pipeline
 *  cleanly declines instead of running silently on dummy weights.
 *
 *  When libvmaf is built with -Denable_dnn=false the session-open call
 *  returns -ENOSYS and init() propagates that, so this extractor simply
 *  cannot be instantiated without a real ORT build.
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

typedef struct LpipsState {
    char *model_path; /**< feature option, owned by opt.c */
    VmafDnnSession *sess;
    unsigned w, h;
    /* Scratch RGB planes (uint8) + ImageNet-normalised float tensors,
     * sized once at init and reused per-frame. */
    uint8_t *rgb8_ref[3];
    uint8_t *rgb8_dist[3];
    float *tensor_ref;  /**< 3 * w * h floats, NCHW */
    float *tensor_dist; /**< 3 * w * h floats, NCHW */
} LpipsState;

/* BT.709 limited-range YUV → RGB in [0, 255]. Writes one output row of
 * three uint8 channels given three Y/U/V pixels. u444 / v444 are the
 * chroma samples already upsampled to 4:4:4. */
static inline void yuv_bt709_to_rgb8_pixel(int y, int u, int v, uint8_t *r, uint8_t *g, uint8_t *b)
{
    /* Studio-swing normalisation: Y∈[16,235], UV∈[16,240]. */
    const float yn = ((float)y - 16.0f) * (1.0f / 219.0f);
    const float un = ((float)u - 128.0f) * (1.0f / 224.0f);
    const float vn = ((float)v - 128.0f) * (1.0f / 224.0f);

    float rf = yn + 1.28033f * vn;
    float gf = yn - 0.21482f * un - 0.38059f * vn;
    float bf = yn + 2.12798f * un;

    /* Saturating clamp to [0, 1] then scale to [0, 255] with round-to-nearest. */
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

/* Expand planar YUV (4:2:0 / 4:2:2 / 4:4:4, 8-bit) to packed RGB planes.
 * Chroma is upsampled by nearest-neighbour replication — the same
 * convention used by ciede.c, so LPIPS scores stay comparable with the
 * rest of the CPU pipeline under identical inputs. */
static int yuv8_to_rgb8_planes(const VmafPicture *pic, uint8_t *dst_r, uint8_t *dst_g,
                               uint8_t *dst_b)
{
    if (pic->bpc != 8)
        return -ENOTSUP;
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

static int lpips_init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                      unsigned w, unsigned h)
{
    LpipsState *s = fex->priv;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;
    if (bpc != 8) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "lpips: bpc=%u not supported yet (8-bit only in this build)\n", bpc);
        return -ENOTSUP;
    }

    const char *path = s->model_path;
    if (!path || !*path) {
        const char *env = getenv("VMAF_LPIPS_MODEL_PATH");
        if (env && *env)
            path = env;
    }
    if (!path || !*path) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "lpips: no model path (set feature option model_path or env "
                                       "VMAF_LPIPS_MODEL_PATH)\n");
        return -EINVAL;
    }

    int rc = vmaf_dnn_session_open(&s->sess, path, NULL);
    if (rc < 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "lpips: vmaf_dnn_session_open(%s) failed: %d\n", path, rc);
        return rc;
    }
    assert(s->sess != NULL);

    s->w = w;
    s->h = h;
    const size_t plane = (size_t)w * (size_t)h;

    for (int i = 0; i < 3; ++i) {
        s->rgb8_ref[i] = (uint8_t *)aligned_malloc(plane, 32);
        s->rgb8_dist[i] = (uint8_t *)aligned_malloc(plane, 32);
        if (!s->rgb8_ref[i] || !s->rgb8_dist[i])
            goto oom;
    }
    s->tensor_ref = (float *)aligned_malloc(3u * plane * sizeof(float), 32);
    s->tensor_dist = (float *)aligned_malloc(3u * plane * sizeof(float), 32);
    if (!s->tensor_ref || !s->tensor_dist)
        goto oom;

    return 0;

oom:
    for (int i = 0; i < 3; ++i) {
        if (s->rgb8_ref[i]) {
            aligned_free(s->rgb8_ref[i]);
            s->rgb8_ref[i] = NULL;
        }
        if (s->rgb8_dist[i]) {
            aligned_free(s->rgb8_dist[i]);
            s->rgb8_dist[i] = NULL;
        }
    }
    if (s->tensor_ref) {
        aligned_free(s->tensor_ref);
        s->tensor_ref = NULL;
    }
    if (s->tensor_dist) {
        aligned_free(s->tensor_dist);
        s->tensor_dist = NULL;
    }
    if (s->sess) {
        vmaf_dnn_session_close(s->sess);
        s->sess = NULL;
    }
    return -ENOMEM;
}

static int lpips_extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                         VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                         VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    LpipsState *s = fex->priv;
    assert(s != NULL);
    assert(s->sess != NULL);

    if (ref_pic->w[0] != s->w || ref_pic->h[0] != s->h)
        return -ERANGE;
    if (dist_pic->w[0] != s->w || dist_pic->h[0] != s->h)
        return -ERANGE;

    int rc = yuv8_to_rgb8_planes(ref_pic, s->rgb8_ref[0], s->rgb8_ref[1], s->rgb8_ref[2]);
    if (rc < 0)
        return rc;
    rc = yuv8_to_rgb8_planes(dist_pic, s->rgb8_dist[0], s->rgb8_dist[1], s->rgb8_dist[2]);
    if (rc < 0)
        return rc;

    rc = vmaf_tensor_from_rgb_imagenet(s->rgb8_ref[0], s->w, s->rgb8_ref[1], s->w, s->rgb8_ref[2],
                                       s->w, (int)s->w, (int)s->h, s->tensor_ref);
    if (rc < 0)
        return rc;
    rc = vmaf_tensor_from_rgb_imagenet(s->rgb8_dist[0], s->w, s->rgb8_dist[1], s->w,
                                       s->rgb8_dist[2], s->w, (int)s->w, (int)s->h, s->tensor_dist);
    if (rc < 0)
        return rc;

    const int64_t shape[4] = {1, 3, (int64_t)s->h, (int64_t)s->w};
    VmafDnnInput inputs[2] = {
        {.name = "ref", .data = s->tensor_ref, .shape = shape, .rank = 4},
        {.name = "dist", .data = s->tensor_dist, .shape = shape, .rank = 4},
    };
    float score_buf[1] = {0.0f};
    VmafDnnOutput outputs[1] = {
        {.name = "score", .data = score_buf, .capacity = 1, .written = 0},
    };

    rc = vmaf_dnn_session_run(s->sess, inputs, 2u, outputs, 1u);
    if (rc < 0)
        return rc;
    if (outputs[0].written == 0u)
        return -EIO;

    return vmaf_feature_collector_append(feature_collector, "lpips", (double)score_buf[0], index);
}

static int lpips_close(VmafFeatureExtractor *fex)
{
    LpipsState *s = fex->priv;
    if (!s)
        return 0;
    for (int i = 0; i < 3; ++i) {
        if (s->rgb8_ref[i])
            aligned_free(s->rgb8_ref[i]);
        if (s->rgb8_dist[i])
            aligned_free(s->rgb8_dist[i]);
    }
    if (s->tensor_ref)
        aligned_free(s->tensor_ref);
    if (s->tensor_dist)
        aligned_free(s->tensor_dist);
    if (s->sess)
        vmaf_dnn_session_close(s->sess);
    memset(s, 0, sizeof(*s));
    return 0;
}

static const VmafOption lpips_options[] = {
    {
        .name = "model_path",
        .help = "Filesystem path to the LPIPS ONNX model (two-input, 'ref'/'dist'). "
                "Overrides the VMAF_LPIPS_MODEL_PATH env var.",
        .offset = offsetof(LpipsState, model_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {NULL},
};

static const char *lpips_provided_features[] = {"lpips", NULL};

VmafFeatureExtractor vmaf_fex_lpips = {
    .name = "lpips",
    .init = lpips_init,
    .extract = lpips_extract,
    .close = lpips_close,
    .options = lpips_options,
    .priv_size = sizeof(LpipsState),
    .provided_features = lpips_provided_features,
    /* Explicit zero-init so GCC LTO sees the full struct layout
     * across TUs (silences -Wlto-type-mismatch after the chars
     * field landed; see ADR-0181). */
    .chars = {0},
};
