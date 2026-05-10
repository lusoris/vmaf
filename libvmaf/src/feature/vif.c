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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "mem.h"
#include "vif.h"
#include "vif_options.h"
#include "vif_tools.h"

#define VIF_BUF_CNT 8

/* The eight contiguous working buffers sliced out of the one big
 * allocation made in `compute_vif`. Lifetime is a single call. */
typedef struct VifBuffers {
    float *ref_scale;
    float *dis_scale;
    float *mu1;
    float *mu2;
    float *ref_sq_filt;
    float *dis_sq_filt;
    float *ref_dis_filt;
    float *tmpbuf;
} VifBuffers;

/* Mutable per-scale state threaded through the per-scale driver so the
 * outer `compute_vif` loop can update the "current scale" pointers +
 * dimensions after each decimation step. */
typedef struct VifScaleCtx {
    const float *curr_ref_scale;
    const float *curr_dis_scale;
    int curr_ref_stride;
    int curr_dis_stride;
    int w;
    int h;
} VifScaleCtx;

/* No filter-table index needed: filters are computed on-the-fly via
 * vif_get_filter() / vif_get_filter_size(), matching Netflix upstream
 * bf9ad333. Validation still uses vif_validate_kernelscale(). */

/* Slice a single `VIF_BUF_CNT * buf_sz_one` allocation into the 8 named
 * buffers. The tmpbuf is intentionally the tail — no trailing `+=` offset
 * (upstream had a dead-store there). */
static void slice_vif_buffers(void *data_buf, size_t buf_sz_one, VifBuffers *b)
{
    char *top = (char *)data_buf;
    b->ref_scale = (float *)(void *)top;
    top += buf_sz_one;
    b->dis_scale = (float *)(void *)top;
    top += buf_sz_one;
    b->mu1 = (float *)(void *)top;
    top += buf_sz_one;
    b->mu2 = (float *)(void *)top;
    top += buf_sz_one;
    b->ref_sq_filt = (float *)(void *)top;
    top += buf_sz_one;
    b->dis_sq_filt = (float *)(void *)top;
    top += buf_sz_one;
    b->ref_dis_filt = (float *)(void *)top;
    top += buf_sz_one;
    b->tmpbuf = (float *)(void *)top;
}

/* Apply the filter/decimate step that turns scale `scale-1`'s mu images
 * into scale `scale`'s ref/dis. Updates the context's "current" pointers +
 * dimensions so the subsequent convolve + statistic operate on the new
 * scale. */
static void decimate_to_next_scale(VifScaleCtx *st, const VifBuffers *b, const float *filter,
                                   int filter_width, int buf_stride)
{
    vif_filter1d_s(filter, st->curr_ref_scale, b->mu1, b->tmpbuf, st->w, st->h, st->curr_ref_stride,
                   buf_stride, filter_width);
    vif_filter1d_s(filter, st->curr_dis_scale, b->mu2, b->tmpbuf, st->w, st->h, st->curr_dis_stride,
                   buf_stride, filter_width);

#ifdef VIF_OPT_HANDLE_BORDERS
    const int buf_valid_w = st->w;
    const int buf_valid_h = st->h;
    float *const mu1_adj = b->mu1;
    float *const mu2_adj = b->mu2;
#else
    const int filter_adj = filter_width / 2;
    const int buf_valid_w = st->w - filter_adj * 2;
    const int buf_valid_h = st->h - filter_adj * 2;
    float *const mu1_adj =
        (float *)((char *)b->mu1 + (ptrdiff_t)filter_adj * buf_stride + filter_adj * sizeof(float));
    float *const mu2_adj =
        (float *)((char *)b->mu2 + (ptrdiff_t)filter_adj * buf_stride + filter_adj * sizeof(float));
#endif

    vif_dec2_s(mu1_adj, b->ref_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);
    vif_dec2_s(mu2_adj, b->dis_scale, buf_valid_w, buf_valid_h, buf_stride, buf_stride);

    st->w = buf_valid_w / 2;
    st->h = buf_valid_h / 2;
    st->curr_ref_scale = b->ref_scale;
    st->curr_dis_scale = b->dis_scale;
    st->curr_ref_stride = buf_stride;
    st->curr_dis_stride = buf_stride;
}

/* Run one pyramid scale: convolve + covariance filters + statistic. Writes
 * (num, den) into the two `scores[]` slots for this scale.
 * bf9ad333: filters are generated on-the-fly via vif_get_filter() so that
 * Gaussian coefficients are computed identically to Netflix upstream. */
static void compute_vif_at_scale(int scale, float vif_kernelscale, int buf_stride, VifScaleCtx *st,
                                 const VifBuffers *b, double vif_enhn_gain_limit,
                                 double vif_sigma_nsq, double *scores)
{
    /* Filters will never be larger than 128 elements (largest known: 65). */
    float filter[128];
    vif_get_filter(filter, scale, vif_kernelscale);
    const int filter_width = vif_get_filter_size(scale, vif_kernelscale);

    if (scale > 0)
        decimate_to_next_scale(st, b, filter, filter_width, buf_stride);

    vif_filter1d_s(filter, st->curr_ref_scale, b->mu1, b->tmpbuf, st->w, st->h, st->curr_ref_stride,
                   buf_stride, filter_width);
    vif_filter1d_s(filter, st->curr_dis_scale, b->mu2, b->tmpbuf, st->w, st->h, st->curr_dis_stride,
                   buf_stride, filter_width);
    vif_filter1d_sq_s(filter, st->curr_ref_scale, b->ref_sq_filt, b->tmpbuf, st->w, st->h,
                      st->curr_ref_stride, buf_stride, filter_width);
    vif_filter1d_sq_s(filter, st->curr_dis_scale, b->dis_sq_filt, b->tmpbuf, st->w, st->h,
                      st->curr_dis_stride, buf_stride, filter_width);
    vif_filter1d_xy_s(filter, st->curr_ref_scale, st->curr_dis_scale, b->ref_dis_filt, b->tmpbuf,
                      st->w, st->h, st->curr_ref_stride, st->curr_dis_stride, buf_stride,
                      filter_width);

    float num;
    float den;
    vif_statistic_s(b->mu1, b->mu2, b->ref_sq_filt, b->dis_sq_filt, b->ref_dis_filt, &num, &den,
                    st->w, st->h, buf_stride, buf_stride, buf_stride, buf_stride, buf_stride,
                    vif_enhn_gain_limit, vif_sigma_nsq);
    scores[(ptrdiff_t)2 * scale] = num;
    scores[(ptrdiff_t)2 * scale + 1] = den;
}

/* Sum the per-scale num/den pairs and produce the final VIF score. */
static void finalize_vif_score(const double *scores, double *score, double *score_num,
                               double *score_den)
{
    *score_num = 0.0;
    *score_den = 0.0;
    for (int scale = 0; scale < 4; ++scale) {
        *score_num += scores[(ptrdiff_t)2 * scale];
        *score_den += scores[(ptrdiff_t)2 * scale + 1];
    }
    *score = (*score_den == 0.0) ? 1.0 : (*score_num) / (*score_den);
}

int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride,
                double *score, double *score_num, double *score_den, double *scores,
                double vif_enhn_gain_limit, double vif_kernelscale, double vif_sigma_nsq)
{
    /* bf9ad333: validate via the shared allow-list then narrow to float. */
    const float kernelscale_f = (float)vif_kernelscale;
    if (!vif_validate_kernelscale(kernelscale_f)) {
        printf("error: vif_kernelscale %f is not a supported value\n", vif_kernelscale);
        (void)fflush(stdout);
        return 1;
    }

    const int buf_stride = ALIGN_CEIL(w * sizeof(float));
    const size_t buf_sz_one = (size_t)buf_stride * h;
    if (SIZE_MAX / buf_sz_one < VIF_BUF_CNT) {
        printf("error: SIZE_MAX / buf_sz_one < VIF_BUF_CNT, buf_sz_one = %zu.\n", buf_sz_one);
        (void)fflush(stdout);
        return 1;
    }

    float *const data_buf = aligned_malloc(buf_sz_one * VIF_BUF_CNT, MAX_ALIGN);
    if (!data_buf) {
        printf("error: aligned_malloc failed for data_buf.\n");
        (void)fflush(stdout);
        return 1;
    }

    VifBuffers b;
    slice_vif_buffers(data_buf, buf_sz_one, &b);

    VifScaleCtx st = {
        .curr_ref_scale = ref,
        .curr_dis_scale = dis,
        .curr_ref_stride = ref_stride,
        .curr_dis_stride = dis_stride,
        .w = w,
        .h = h,
    };

    for (int scale = 0; scale < 4; ++scale) {
        compute_vif_at_scale(scale, kernelscale_f, buf_stride, &st, &b, vif_enhn_gain_limit,
                             vif_sigma_nsq, scores);
    }

    finalize_vif_score(scores, score, score_num, score_den);
    aligned_free(data_buf);
    return 0;
}
