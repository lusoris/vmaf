/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Dispatch support table for the Metal backend (ADR-0421 / T8-1c-k).
 *  Mirrors the CUDA / Vulkan dispatch predicates: callers ask whether
 *  a feature can route to this backend before they bind GPU pictures.
 */

#include "dispatch_strategy.h"

#include <string.h>

static const char *const g_metal_features[] = {
    /* integer_motion_v2_metal provided_features[] */
    "VMAF_integer_feature_motion_v2_sad_score",
    "VMAF_integer_feature_motion2_v2_score", /* was "motion2_v2_score" — wrong key */
    /* float_psnr_metal provided_features[] */
    "float_psnr",
    /* float_moment_metal provided_features[] */
    "float_moment_ref1st",
    "float_moment_dis1st",
    "float_moment_ref2nd",
    "float_moment_dis2nd",
    /* float_ansnr_metal provided_features[] */
    "float_ansnr",
    /* integer_psnr_metal provided_features[] */
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    /* float_motion_metal provided_features[] */
    "VMAF_feature_motion_score", /* was "float_motion" — wrong key */
    "VMAF_feature_motion2_score",
    /* integer_motion_metal provided_features[] */
    "VMAF_integer_feature_motion_y_score",
    "VMAF_integer_feature_motion2_score", /* was "motion2_score" — wrong key */
    /* "motion3_score" removed — no Metal extractor provides this feature */
    /* float_ssim_metal provided_features[] */
    "float_ssim",
    /* "float_ms_ssim" removed — no Metal extractor provides this feature */
    NULL,
};

int vmaf_metal_dispatch_supports(const VmafMetalContext *ctx, const char *feature)
{
    if (!ctx || !feature)
        return 0;

    for (size_t i = 0; g_metal_features[i]; ++i) {
        if (strcmp(feature, g_metal_features[i]) == 0)
            return 1;
    }
    return 0;
}
