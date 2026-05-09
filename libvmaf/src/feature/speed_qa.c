/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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

/*
 * SpEED-QA NR metric scaffold — ADR-0253.
 * Reference: Bampis, Gupta, Soundararajan and Bovik, IEEE SPL 24(9), 2017.
 *
 * TODO ADR-0253: real SpEED-QA spatial+temporal entropic-difference algorithm
 *                — scaffold only
 */

#include "config.h"
#include "feature_extractor.h"

static const char *const provided_features[] = {"speed_qa", NULL};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)fex;
    (void)pix_fmt;
    (void)bpc;
    (void)w;
    (void)h;

    return 0;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)fex;
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;

    /* Placeholder score — real algorithm deferred per ADR-0253. */
    return vmaf_feature_collector_append(feature_collector, "speed_qa", 0.0, index);
}

static int close(VmafFeatureExtractor *fex)
{
    (void)fex;
    return 0;
}

VmafFeatureExtractor vmaf_fex_speed_qa = {
    .name = "speed_qa",
    .init = init,
    .extract = extract,
    .close = close,
    .provided_features = provided_features,
    .chars = {0},
};
