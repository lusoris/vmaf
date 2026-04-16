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

#include <immintrin.h>
#include "ansnr_avx512.h"

void ansnr_mse_line_avx512(const float *ref, const float *dis, float *sig_accum, float *noise_accum,
                           int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    __m512d sig_dsum0 = _mm512_setzero_pd();
    __m512d sig_dsum1 = _mm512_setzero_pd();
    __m512d noise_dsum0 = _mm512_setzero_pd();
    __m512d noise_dsum1 = _mm512_setzero_pd();
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        __m512 r = _mm512_loadu_ps(ref + j);
        __m512 d = _mm512_loadu_ps(dis + j);
        __m512 diff = _mm512_sub_ps(r, d);
        __m512 sig_val = _mm512_mul_ps(r, r);
        __m512 noise_val = _mm512_mul_ps(diff, diff);

        __m256 sig_lo = _mm512_castps512_ps256(sig_val);
        __m256 sig_hi = _mm512_extractf32x8_ps(sig_val, 1);
        sig_dsum0 = _mm512_add_pd(sig_dsum0, _mm512_cvtps_pd(sig_lo));
        sig_dsum1 = _mm512_add_pd(sig_dsum1, _mm512_cvtps_pd(sig_hi));

        __m256 noise_lo = _mm512_castps512_ps256(noise_val);
        __m256 noise_hi = _mm512_extractf32x8_ps(noise_val, 1);
        noise_dsum0 = _mm512_add_pd(noise_dsum0, _mm512_cvtps_pd(noise_lo));
        noise_dsum1 = _mm512_add_pd(noise_dsum1, _mm512_cvtps_pd(noise_hi));
    }

    /* Explicit horizontal reduce matching AVX2 accumulation order */
    __m512d sig_combined = _mm512_add_pd(sig_dsum0, sig_dsum1);
    __m256d sig_lo4 = _mm512_castpd512_pd256(sig_combined);
    __m256d sig_hi4 = _mm512_extractf64x4_pd(sig_combined, 1);
    __m256d sig_t4 = _mm256_add_pd(sig_lo4, sig_hi4);
    __m128d sig_tlo = _mm256_castpd256_pd128(sig_t4);
    __m128d sig_thi = _mm256_extractf128_pd(sig_t4, 1);
    __m128d sig_s = _mm_add_pd(sig_tlo, sig_thi);
    sig_s = _mm_add_sd(sig_s, _mm_unpackhi_pd(sig_s, sig_s));
    float sig_result = (float)_mm_cvtsd_f64(sig_s);

    __m512d noise_combined = _mm512_add_pd(noise_dsum0, noise_dsum1);
    __m256d noise_lo4 = _mm512_castpd512_pd256(noise_combined);
    __m256d noise_hi4 = _mm512_extractf64x4_pd(noise_combined, 1);
    __m256d noise_t4 = _mm256_add_pd(noise_lo4, noise_hi4);
    __m128d noise_tlo = _mm256_castpd256_pd128(noise_t4);
    __m128d noise_thi = _mm256_extractf128_pd(noise_t4, 1);
    __m128d noise_s = _mm_add_pd(noise_tlo, noise_thi);
    noise_s = _mm_add_sd(noise_s, _mm_unpackhi_pd(noise_s, noise_s));
    float noise_result = (float)_mm_cvtsd_f64(noise_s);

    for (; j < w; j++) {
        float r = ref[j];
        float d = dis[j];
        float diff = r - d;
        sig_result += r * r;
        noise_result += diff * diff;
    }

    *sig_accum += sig_result;
    *noise_accum += noise_result;
}
