/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <errno.h>
#include <math.h>
#include <string.h>
#include <stddef.h>

#include "dict.h"
#include "feature_collector.h"
#include "framesync.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "adm.h"
#include "adm_options.h"
#include "mem.h"
#include "picture_copy.h"

/* Number of DWT decomposition scales. */
#define ADM_NUM_SCALES (4)

typedef struct AdmState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    /* adm_csf_diag_scale / adm_csf_scale: CSF band-scale multipliers.
     * These affect the feature-name suffix (option alias scfd/scf) but
     * are not consumed by the float adm_csf_s path (no-op in computation). */
    double adm_csf_diag_scale;
    int adm_csf_mode;
    double adm_csf_scale;
    double adm_dlm_weight;
    double adm_min_val;
    bool adm_skip_scale0;
    /* Per-scale DLM noise-floor weights (f1s[0..3], one per DWT scale). */
    double adm_f1s0;
    double adm_f1s1;
    double adm_f1s2;
    double adm_f1s3;
    /* Per-scale AIM noise-floor weights (f2s[0..3], one per DWT scale). */
    double adm_f2s0;
    double adm_f2s1;
    double adm_f2s2;
    double adm_f2s3;
    VmafDictionary *feature_name_dict;
} AdmState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdf",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        /* scfd is registered before csf so it appears first in option-suffixed feature names
         * (float_adm does not apply this scale to its CSF computation; it only affects naming). */
        .name = "adm_csf_diag_scale",
        .alias = "scfd",
        .help = "scale coefficient for the diagonal direction term of CSF (float path: name-only)",
        .offset = offsetof(AdmState, adm_csf_diag_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(AdmState, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 9,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        /* scf is registered after csf so it appears after csf in option-suffixed feature names
         * (float_adm does not apply this scale to its CSF computation; it only affects naming). */
        .name = "adm_csf_scale",
        .alias = "scf",
        .help =
            "scale coefficient for horizontal/vertical direction terms of CSF (float path: name-only)",
        .offset = offsetof(AdmState, adm_csf_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_dlm_weight",
        .alias = "dlmw",
        .help = "weight of the DLM component in the adm3 blend "
                "(adm3 = dlm_weight * dlm + (1 - dlm_weight) * (1 - aim))",
        .offset = offsetof(AdmState, adm_dlm_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.5,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_min_val",
        .alias = "min",
        .help = "minimum value for adm3; values below this are clipped to adm_min_val",
        .offset = offsetof(AdmState, adm_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_skip_scale0",
        .alias = "ssz",
        .help = "when set, skip scale 0 in ADM calculation (sets scale0 score to 0)",
        .offset = offsetof(AdmState, adm_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    /* Per-scale DLM noise-floor weights (f1s[0..3]). */
    {
        .name = "adm_f1s0",
        .alias = "f1s0",
        .help = "per-scale DLM noise-floor weight for scale 0",
        .offset = offsetof(AdmState, adm_f1s0),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s1",
        .alias = "f1s1",
        .help = "per-scale DLM noise-floor weight for scale 1",
        .offset = offsetof(AdmState, adm_f1s1),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s2",
        .alias = "f1s2",
        .help = "per-scale DLM noise-floor weight for scale 2",
        .offset = offsetof(AdmState, adm_f1s2),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s3",
        .alias = "f1s3",
        .help = "per-scale DLM noise-floor weight for scale 3",
        .offset = offsetof(AdmState, adm_f1s3),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    /* Per-scale AIM noise-floor weights (f2s[0..3]). */
    {
        .name = "adm_f2s0",
        .alias = "f2s0",
        .help = "per-scale AIM noise-floor weight for scale 0",
        .offset = offsetof(AdmState, adm_f2s0),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s1",
        .alias = "f2s1",
        .help = "per-scale AIM noise-floor weight for scale 1",
        .offset = offsetof(AdmState, adm_f2s1),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s2",
        .alias = "f2s2",
        .help = "per-scale AIM noise-floor weight for scale 2",
        .offset = offsetof(AdmState, adm_f2s2),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s3",
        .alias = "f2s3",
        .help = "per-scale AIM noise-floor weight for scale 3",
        .offset = offsetof(AdmState, adm_f2s3),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    AdmState *s = fex->priv;
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref)
        goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist)
        goto fail;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto fail;

    return 0;

fail:
    if (s->ref)
        aligned_free(s->ref);
    if (s->dist)
        aligned_free(s->dist);
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

/* Append the per-scale debug features. Called only when s->debug is set;
 * preserves the exact append order of the historical inline block.
 */
static int append_debug_features(VmafFeatureCollector *feature_collector,
                                 VmafDictionary *feature_name_dict, double score, double score_num,
                                 double score_den, const double scores[8], unsigned index)
{
    int err = 0;

    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict, "adm",
                                                   score, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict, "adm_num",
                                                   score_num, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict, "adm_den",
                                                   score_den, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_num_scale0", scores[0], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_den_scale0", scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_num_scale1", scores[2], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_den_scale1", scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_num_scale2", scores[4], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_den_scale2", scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_num_scale3", scores[6], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, feature_name_dict,
                                                   "adm_den_scale3", scores[7], index);

    return err;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    AdmState *s = fex->priv;
    int err = 0;

    (void)ref_pic_90;
    (void)dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc, 0);
    picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc, 0);

    double score;
    double score_num;
    double score_den;
    double score_aim;
    double scores[8];

    /* Build per-scale noise-weight arrays from the per-field options. */
    const double adm_f1s[ADM_NUM_SCALES] = {s->adm_f1s0, s->adm_f1s1, s->adm_f1s2, s->adm_f1s3};
    const double adm_f2s[ADM_NUM_SCALES] = {s->adm_f2s0, s->adm_f2s1, s->adm_f2s2, s->adm_f2s3};

    err = compute_adm(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0], s->float_stride,
                      s->float_stride, &score, &score_num, &score_den, scores, ADM_BORDER_FACTOR,
                      s->adm_enhn_gain_limit, s->adm_norm_view_dist, s->adm_ref_display_height,
                      s->adm_csf_mode, &score_aim, adm_f1s, adm_f2s, s->adm_skip_scale0,
                      s->adm_csf_scale, s->adm_csf_diag_scale);
    if (err)
        return err;

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_adm2_score", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_aim_score", score_aim, index);

    /* adm3 = dlm_weight * dlm + (1 - dlm_weight) * (1 - aim), clipped to [adm_min_val, 1]. */
    double score_adm3 = s->adm_dlm_weight * score + (1.0 - s->adm_dlm_weight) * (1.0 - score_aim);
    if (score_adm3 < s->adm_min_val)
        score_adm3 = s->adm_min_val;
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_adm3_score", score_adm3, index);

    /* When adm_skip_scale0 is set, compute_adm() already zeroed num and set den to 1e-10.
     * Convert to the -1.0 published sentinel (matching integer_adm.c convention) and
     * publish the per-scale scale0 score as 0.0. */
    double adm_scale0_score;
    if (s->adm_skip_scale0) {
        scores[0] = 0.0;  /* num already 0 from compute_adm; restate for clarity */
        scores[1] = -1.0; /* convert 1e-10 sentinel to -1.0 for XML output */
        adm_scale0_score = 0.0;
    } else {
        adm_scale0_score = scores[0] / scores[1];
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_adm_scale0_score",
                                                   adm_scale0_score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_adm_scale1_score",
                                                   scores[2] / scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_adm_scale2_score",
                                                   scores[4] / scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_adm_scale3_score",
                                                   scores[6] / scores[7], index);

    if (!s->debug)
        return err;

    err |= append_debug_features(feature_collector, s->feature_name_dict, score, score_num,
                                 score_den, scores, index);

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;
    if (s->ref)
        aligned_free(s->ref);
    if (s->dist)
        aligned_free(s->dist);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"VMAF_feature_adm2_score",
                                          "VMAF_feature_aim_score",
                                          "VMAF_feature_adm3_score",
                                          "VMAF_feature_adm_scale0_score",
                                          "VMAF_feature_adm_scale1_score",
                                          "VMAF_feature_adm_scale2_score",
                                          "VMAF_feature_adm_scale3_score",
                                          "adm_num",
                                          "adm_den",
                                          "adm_scale0",
                                          "adm_num_scale0",
                                          "adm_den_scale0",
                                          "adm_num_scale1",
                                          "adm_den_scale1",
                                          "adm_num_scale2",
                                          "adm_den_scale2",
                                          "adm_num_scale3",
                                          "adm_den_scale3",
                                          NULL};

// Registered via extern in src/feature/feature_extractor.c — must keep
// external linkage despite no in-TU users (misc-use-internal-linkage
// is a false positive for the feature-registry pattern).
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_adm = {
    .name = "float_adm",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(AdmState),
    .provided_features = provided_features,
};
