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

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "libvmaf/vmaf_assert.h"
#include "ms_ssim_decimate.h"

/*
 * 9-tap 9/7 biorthogonal wavelet LPF, separable form.
 *
 * REBASE-SENSITIVE INVARIANT: these coefficients MUST match
 * `g_lpf_h` / `g_lpf_v` in libvmaf/src/feature/ms_ssim.c. If Netflix
 * upstream changes the coefficients there, mirror the change here.
 * See docs/adr/0125-ms-ssim-decimate-simd.md.
 */
#define MS_SSIM_DECIMATE_LPF_LEN 9
#define MS_SSIM_DECIMATE_LPF_HALF 4

static const float ms_ssim_lpf_h[MS_SSIM_DECIMATE_LPF_LEN] = {
    0.026727f, -0.016828f, -0.078201f, 0.266846f, 0.602914f,
    0.266846f, -0.078201f, -0.016828f, 0.026727f,
};

static const float ms_ssim_lpf_v[MS_SSIM_DECIMATE_LPF_LEN] = {
    0.026727f, -0.016828f, -0.078201f, 0.266846f, 0.602914f,
    0.266846f, -0.078201f, -0.016828f, 0.026727f,
};

/*
 * KBND_SYMMETRIC mirror: indices outside [0, n) reflect around the
 * nearest border. This matches `KBND_SYMMETRIC` in
 * libvmaf/src/feature/iqa/convolve.c so the boundary behaviour is
 * identical to the vendored 2-D path.
 */
static inline int ms_ssim_decimate_mirror(int idx, int n)
{
    if (idx < 0) {
        return -1 - idx;
    }
    if (idx >= n) {
        return (n - (idx - n)) - 1;
    }
    return idx;
}

int ms_ssim_decimate_scalar(const float *src, int w, int h, float *dst, int *rw, int *rh)
{
    VMAF_ASSERT_DEBUG(src != NULL);
    VMAF_ASSERT_DEBUG(dst != NULL);
    VMAF_ASSERT_DEBUG(src != dst);
    VMAF_ASSERT_DEBUG(w > 0);
    VMAF_ASSERT_DEBUG(h > 0);

    const int w_out = (w / 2) + (w & 1);
    const int h_out = (h / 2) + (h & 1);
    const int half = MS_SSIM_DECIMATE_LPF_HALF;

    /*
     * Horizontal pass scratch: w_out columns per row, all h rows
     * (factor-2 subsampling is applied in the horizontal pass; the
     * vertical pass still needs all source rows because the 9-tap
     * vertical kernel at y_out*2 reads rows y_out*2-4..y_out*2+4 which
     * spans both even and odd source indices).
     */
    float *tmp = (float *)malloc((size_t)w_out * (size_t)h * sizeof(float));
    if (tmp == NULL) {
        return -1;
    }

    /* Horizontal pass: src[h x w] -> tmp[h x w_out] via g_lpf_h. */
    for (int y = 0; y < h; ++y) {
        const float *src_row = &src[y * w];
        float *tmp_row = &tmp[y * w_out];
        for (int x_out = 0; x_out < w_out; ++x_out) {
            const int x_src = x_out * 2;
            float acc = 0.0f;
            for (int k = 0; k < MS_SSIM_DECIMATE_LPF_LEN; ++k) {
                const int xi = ms_ssim_decimate_mirror(x_src + k - half, w);
                acc = fmaf(src_row[xi], ms_ssim_lpf_h[k], acc);
            }
            tmp_row[x_out] = acc;
        }
    }

    /* Vertical pass: tmp[h x w_out] -> dst[h_out x w_out] via g_lpf_v. */
    for (int y_out = 0; y_out < h_out; ++y_out) {
        const int y_src = y_out * 2;
        float *dst_row = &dst[y_out * w_out];
        for (int x_out = 0; x_out < w_out; ++x_out) {
            float acc = 0.0f;
            for (int k = 0; k < MS_SSIM_DECIMATE_LPF_LEN; ++k) {
                const int yi = ms_ssim_decimate_mirror(y_src + k - half, h);
                acc = fmaf(tmp[yi * w_out + x_out], ms_ssim_lpf_v[k], acc);
            }
            dst_row[x_out] = acc;
        }
    }

    free(tmp);

    if (rw != NULL) {
        *rw = w_out;
    }
    if (rh != NULL) {
        *rh = h_out;
    }
    return 0;
}
