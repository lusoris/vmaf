/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include <immintrin.h>
#include <math.h>

#include "ssim_avx2.h"

void ssim_precompute_avx2(const float *ref, const float *cmp, float *ref_sq, float *cmp_sq,
                          float *ref_cmp, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 r = _mm256_loadu_ps(ref + i);
        __m256 c = _mm256_loadu_ps(cmp + i);
        _mm256_storeu_ps(ref_sq + i, _mm256_mul_ps(r, r));
        _mm256_storeu_ps(cmp_sq + i, _mm256_mul_ps(c, c));
        _mm256_storeu_ps(ref_cmp + i, _mm256_mul_ps(r, c));
    }
    for (; i < n; i++) {
        ref_sq[i] = ref[i] * ref[i];
        cmp_sq[i] = cmp[i] * cmp[i];
        ref_cmp[i] = ref[i] * cmp[i];
    }
}

void ssim_variance_avx2(float *ref_sigma_sqd, float *cmp_sigma_sqd, float *sigma_both,
                        const float *ref_mu, const float *cmp_mu, int n)
{
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 rm = _mm256_loadu_ps(ref_mu + i);
        __m256 cm = _mm256_loadu_ps(cmp_mu + i);

        __m256 rs = _mm256_loadu_ps(ref_sigma_sqd + i);
        rs = _mm256_sub_ps(rs, _mm256_mul_ps(rm, rm));
        rs = _mm256_max_ps(rs, zero);
        _mm256_storeu_ps(ref_sigma_sqd + i, rs);

        __m256 cs = _mm256_loadu_ps(cmp_sigma_sqd + i);
        cs = _mm256_sub_ps(cs, _mm256_mul_ps(cm, cm));
        cs = _mm256_max_ps(cs, zero);
        _mm256_storeu_ps(cmp_sigma_sqd + i, cs);

        __m256 sb = _mm256_loadu_ps(sigma_both + i);
        sb = _mm256_sub_ps(sb, _mm256_mul_ps(rm, cm));
        _mm256_storeu_ps(sigma_both + i, sb);
    }
    for (; i < n; i++) {
        ref_sigma_sqd[i] -= ref_mu[i] * ref_mu[i];
        if (ref_sigma_sqd[i] < 0.0f)
            ref_sigma_sqd[i] = 0.0f;
        cmp_sigma_sqd[i] -= cmp_mu[i] * cmp_mu[i];
        if (cmp_sigma_sqd[i] < 0.0f)
            cmp_sigma_sqd[i] = 0.0f;
        sigma_both[i] -= ref_mu[i] * cmp_mu[i];
    }
}

void ssim_accumulate_avx2(const float *ref_mu, const float *cmp_mu, const float *ref_sigma_sqd,
                          const float *cmp_sigma_sqd, const float *sigma_both, int n, float C1,
                          float C2, float C3, double *ssim_sum, double *l_sum, double *c_sum,
                          double *s_sum)
{
    __m256 vC1 = _mm256_set1_ps(C1);
    __m256 vC2 = _mm256_set1_ps(C2);
    __m256 vC3 = _mm256_set1_ps(C3);
    __m256 v2 = _mm256_set1_ps(2.0f);
    __m256 vzero = _mm256_setzero_ps();

    double local_ssim = 0.0;
    double local_l = 0.0;
    double local_c = 0.0;
    double local_s = 0.0;

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 rm = _mm256_loadu_ps(ref_mu + i);
        __m256 cm = _mm256_loadu_ps(cmp_mu + i);
        __m256 rs = _mm256_loadu_ps(ref_sigma_sqd + i);
        __m256 cs = _mm256_loadu_ps(cmp_sigma_sqd + i);
        __m256 sb = _mm256_loadu_ps(sigma_both + i);

        __m256 srsc = _mm256_sqrt_ps(_mm256_mul_ps(rs, cs));

        __m256 l_num = _mm256_add_ps(_mm256_mul_ps(v2, _mm256_mul_ps(rm, cm)), vC1);
        __m256 l_den =
            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rm, rm), _mm256_mul_ps(cm, cm)), vC1);
        __m256 l = _mm256_div_ps(l_num, l_den);

        __m256 c_num = _mm256_add_ps(_mm256_mul_ps(v2, srsc), vC2);
        __m256 c_den = _mm256_add_ps(_mm256_add_ps(rs, cs), vC2);
        __m256 c = _mm256_div_ps(c_num, c_den);

        __m256 sb_neg = _mm256_cmp_ps(sb, vzero, _CMP_LT_OQ);
        __m256 srsc_le0 = _mm256_cmp_ps(srsc, vzero, _CMP_LE_OQ);
        __m256 clamp_mask = _mm256_and_ps(sb_neg, srsc_le0);
        __m256 clamped_sb = _mm256_blendv_ps(sb, vzero, clamp_mask);

        __m256 s_num = _mm256_add_ps(clamped_sb, vC3);
        __m256 s_den = _mm256_add_ps(srsc, vC3);
        __m256 s = _mm256_div_ps(s_num, s_den);

        __m256 ssim_val = _mm256_mul_ps(_mm256_mul_ps(l, c), s);

        _Alignas(32) float t_ssim[8];
        _Alignas(32) float t_l[8];
        _Alignas(32) float t_c[8];
        _Alignas(32) float t_s[8];
        _mm256_store_ps(t_ssim, ssim_val);
        _mm256_store_ps(t_l, l);
        _mm256_store_ps(t_c, c);
        _mm256_store_ps(t_s, s);
        for (int k = 0; k < 8; k++) {
            local_ssim += (double)t_ssim[k];
            local_l += (double)t_l[k];
            local_c += (double)t_c[k];
            local_s += (double)t_s[k];
        }
    }

    /* Scalar tail */
    for (; i < n; i++) {
        float srsc = sqrtf(ref_sigma_sqd[i] * cmp_sigma_sqd[i]);
        double lv = (2.0 * ref_mu[i] * cmp_mu[i] + C1) /
                    (ref_mu[i] * ref_mu[i] + cmp_mu[i] * cmp_mu[i] + C1);
        double cv = (2.0 * srsc + C2) / (ref_sigma_sqd[i] + cmp_sigma_sqd[i] + C2);
        float csb = (sigma_both[i] < 0.0f && srsc <= 0.0f) ? 0.0f : sigma_both[i];
        double sv = (csb + C3) / (srsc + C3);
        local_ssim += lv * cv * sv;
        local_l += lv;
        local_c += cv;
        local_s += sv;
    }

    *ssim_sum += local_ssim;
    *l_sum += local_l;
    *c_sum += local_c;
    *s_sum += local_s;
}
