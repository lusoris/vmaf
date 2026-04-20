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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mem.h"
#include "iqa/math_utils.h"
#include "iqa/decimate.h"
#include "iqa/ssim_tools.h"
#include "ms_ssim_decimate.h"

/*
 * MS-SSIM 9-tap 9/7 biorthogonal wavelet LPF coefficients moved to
 * libvmaf/src/feature/ms_ssim_decimate.c (separable form). The 2-D
 * `g_lpf` array in upstream Netflix/vmaf ms_ssim.c is no longer used
 * on this fork because the decimate path switched from
 * `_iqa_decimate(..., 2, &lpf_2d, ...)` to the separable scalar-FMA
 * path `ms_ssim_decimate`. See ADR-0125.
 *
 * REBASE-SENSITIVE INVARIANT: if Netflix upstream modifies `g_lpf`,
 * `g_lpf_h`, or `g_lpf_v` in this file during a sync, mirror the
 * change to `ms_ssim_lpf_h` / `ms_ssim_lpf_v` in ms_ssim_decimate.c.
 * See `libvmaf/src/feature/AGENTS.md` and docs/rebase-notes.md.
 */
#define LPF_LEN 9

/* Alpha, beta, and gamma values for each scale */
static float g_alphas[] = {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f};
static float g_betas[] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static float g_gammas[] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

struct _context {
    double l; /* Luminance */
    double c; /* Contrast */
    double s; /* Structure */
    float alpha;
    float beta;
    float gamma;
};

/* Called for each pixel */
int _ms_ssim_map(const struct _ssim_int *si, void *ctx)
{
    struct _context *ms_ctx = (struct _context *)ctx;
    ms_ctx->l += si->l;
    ms_ctx->c += si->c;
    ms_ctx->s += si->s;
    return 0;
}

/* Called to calculate the final result */
float _ms_ssim_reduce(int w, int h, void *ctx)
{
    double size = (double)(w * h);
    struct _context *ms_ctx = (struct _context *)ctx;
    ms_ctx->l = pow(ms_ctx->l / size, (double)ms_ctx->alpha);
    ms_ctx->c = pow(ms_ctx->c / size, (double)ms_ctx->beta);
    ms_ctx->s = pow(fabs(ms_ctx->s / size), (double)ms_ctx->gamma);
    return (float)(ms_ctx->l * ms_ctx->c * ms_ctx->s);
}

/* Releases the scaled buffers */
void _free_buffers(float **buf, int scales)
{
    int idx;
    for (idx = 0; idx < scales; ++idx)
        free(buf[idx]);
}

/* Allocates the scaled buffers. If error, all buffers are free'd */
int _alloc_buffers(float **buf, int w, int h, int scales)
{
    int idx;
    int cur_w = w;
    int cur_h = h;
    for (idx = 0; idx < scales; ++idx) {
        buf[idx] = (float *)malloc(cur_w * cur_h * sizeof(float));
        if (!buf[idx]) {
            _free_buffers(buf, idx);
            return 1;
        }
        cur_w = cur_w / 2 + (cur_w & 1);
        cur_h = cur_h / 2 + (cur_h & 1);
    }
    return 0;
}

int compute_ms_ssim(const float *ref, const float *cmp, int w, int h, int ref_stride,
                    int cmp_stride, double *score, double *l_scores, double *c_scores,
                    double *s_scores)
{

    int ret = 1;

    int wang = 1; /* set default to wang's ms_ssim */
    int scales = SCALES;
    int gauss = 1;
    const float *alphas = g_alphas, *betas = g_betas, *gammas = g_gammas;
    int idx, x, y, cur_w, cur_h;
    int offset, src_offset;
    float **ref_imgs, **cmp_imgs; /* Array of pointers to scaled images */
    double msssim;
    float l, c, s;
    struct _kernel window;
    struct iqa_ssim_args s_args;
    struct _map_reduce mr;
    struct _context ms_ctx;

    /* check stride */
    int stride = ref_stride; /* stride in bytes */
    if (stride != cmp_stride) {
        printf("error: for ms_ssim, ref_stride (%d) != dis_stride (%d) bytes.\n", ref_stride,
               cmp_stride);
        fflush(stdout);
        goto fail_or_end;
    }
    stride /= sizeof(float); /* stride_ in pixels */

    /* specify some default parameters */
    const struct iqa_ms_ssim_args *args = 0; /* 0 for default */

    /* initialize algorithm parameters */
    if (args) {
        wang = args->wang;
        gauss = args->gaussian;
        scales = args->scales;
        if (args->alphas)
            alphas = args->alphas;
        if (args->betas)
            betas = args->betas;
        if (args->gammas)
            gammas = args->gammas;
    }

    /* make sure we won't scale below 1x1 */
    cur_w = w;
    cur_h = h;
    for (idx = 0; idx < scales; ++idx) {
        if (gauss ? cur_w < GAUSSIAN_LEN || cur_h < GAUSSIAN_LEN :
                    cur_w < LPF_LEN || cur_h < LPF_LEN) {
            printf("error: scale below 1x1!\n");
            goto fail_or_end;
        }
        cur_w /= 2;
        cur_h /= 2;
    }

    window.kernel = (float *)g_square_window;
    window.kernel_h = (float *)g_square_window_h; /* zli-nflx */
    window.kernel_v = (float *)g_square_window_v; /* zli-nflx */
    window.w = window.h = SQUARE_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;
    if (gauss) {
        window.kernel = (float *)g_gaussian_window;
        window.kernel_h = (float *)g_gaussian_window_h; /* zli-nflx */
        window.kernel_v = (float *)g_gaussian_window_v; /* zli-nflx */
        window.w = window.h = GAUSSIAN_LEN;
    }

    mr.map = _ms_ssim_map;
    mr.reduce = _ms_ssim_reduce;

    /* allocate the scaled image buffers */
    ref_imgs = (float **)malloc(scales * sizeof(float *));
    cmp_imgs = (float **)malloc(scales * sizeof(float *));
    if (!ref_imgs || !cmp_imgs) {
        if (ref_imgs)
            free(ref_imgs);
        if (cmp_imgs)
            free(cmp_imgs);
        printf("error: unable to malloc ref_imgs or cmp_imgs.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (_alloc_buffers(ref_imgs, w, h, scales)) {
        free(ref_imgs);
        free(cmp_imgs);
        printf("error: unable to _alloc_buffers on ref_imgs.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    if (_alloc_buffers(cmp_imgs, w, h, scales)) {
        _free_buffers(ref_imgs, scales);
        free(ref_imgs);
        free(cmp_imgs);
        printf("error: unable to _alloc_buffers on cmp_imgs.\n");
        fflush(stdout);
        goto fail_or_end;
    }

    /* copy original images into first scale buffer, forcing stride = width. */
    for (y = 0; y < h; ++y) {
        src_offset = y * stride;
        offset = y * w;
        for (x = 0; x < w; ++x, ++offset, ++src_offset) {
            ref_imgs[0][offset] = (float)ref[src_offset];
            cmp_imgs[0][offset] = (float)cmp[src_offset];
        }
    }

    /*
     * Create scaled versions of the images via the separable decimate
     * path (see ADR-0125). Replaces upstream's
     * `_iqa_decimate(..., 2, &lpf_2d, ...)` 2-D 9x9 kernel.
     */
    cur_w = w;
    cur_h = h;
    for (idx = 1; idx < scales; ++idx) {
        if (ms_ssim_decimate(ref_imgs[idx - 1], cur_w, cur_h, ref_imgs[idx], 0, 0) ||
            ms_ssim_decimate(cmp_imgs[idx - 1], cur_w, cur_h, cmp_imgs[idx], &cur_w, &cur_h)) {
            _free_buffers(ref_imgs, scales);
            _free_buffers(cmp_imgs, scales);
            free(ref_imgs);
            free(cmp_imgs);
            printf("error: decimation fails on ref_imgs or cmp_imgs.\n");
            fflush(stdout);
            goto fail_or_end;
        }
    }

    cur_w = w;
    cur_h = h;
    msssim = 1.0;
    for (idx = 0; idx < scales; ++idx) {

        ms_ctx.l = 0;
        ms_ctx.c = 0;
        ms_ctx.s = 0;
        ms_ctx.alpha = alphas[idx];
        ms_ctx.beta = betas[idx];
        ms_ctx.gamma = gammas[idx];

        if (!wang) {
            /* MS-SSIM* (Rouse/Hemami) */
            s_args.alpha = 1.0f;
            s_args.beta = 1.0f;
            s_args.gamma = 1.0f;
            s_args.K1 = 0.0f; /* Force stabilization constants to 0 */
            s_args.K2 = 0.0f;
            s_args.L = 255;
            s_args.f = 1; /* Don't resize */
            mr.context = &ms_ctx;
            _iqa_ssim(ref_imgs[idx], cmp_imgs[idx], cur_w, cur_h, &window, &mr, &s_args, &l, &c,
                      &s);
        } else {
            /* MS-SSIM (Wang) */
            /*
            s_args.alpha = 1.0f;
            s_args.beta  = 1.0f;
            s_args.gamma = 1.0f;
            s_args.K1 = 0.01f;
            s_args.K2 = 0.03f;
            s_args.L  = 255;
            s_args.f  = 1; // Don't resize
            mr.context = &ms_ctx;
            msssim *= _iqa_ssim(ref_imgs[idx], cmp_imgs[idx], cur_w, cur_h, &window, &mr, &s_args, &l, &c, &s);
            */

            /* above is equivalent to passing default parameter: */
            _iqa_ssim(ref_imgs[idx], cmp_imgs[idx], cur_w, cur_h, &window, NULL, NULL, &l, &c, &s);
        }

        msssim *= pow(l, alphas[idx]) * pow(c, betas[idx]) * pow(s, gammas[idx]);
        l_scores[idx] = l;
        c_scores[idx] = c;
        s_scores[idx] = s;

        if (msssim == INFINITY) {
            _free_buffers(ref_imgs, scales);
            _free_buffers(cmp_imgs, scales);
            free(ref_imgs);
            free(cmp_imgs);
            printf("error: ms_ssim is INFINITY.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        cur_w = cur_w / 2 + (cur_w & 1);
        cur_h = cur_h / 2 + (cur_h & 1);
    }

    _free_buffers(ref_imgs, scales);
    _free_buffers(cmp_imgs, scales);
    free(ref_imgs);
    free(cmp_imgs);

    *score = msssim;

    ret = 0;
fail_or_end:
    return ret;
}
