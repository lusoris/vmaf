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
#include "float_psnr_avx512.h"

double float_psnr_noise_line_avx512(const float *ref, const float *dis, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    __m512d dsum0 = _mm512_setzero_pd();
    __m512d dsum1 = _mm512_setzero_pd();
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        __m512 r = _mm512_loadu_ps(ref + j);
        __m512 d = _mm512_loadu_ps(dis + j);
        __m512 diff = _mm512_sub_ps(r, d);
        __m512 sq = _mm512_mul_ps(diff, diff);

        __m256 lo = _mm512_castps512_ps256(sq);
        __m256 hi = _mm512_extractf32x8_ps(sq, 1);
        dsum0 = _mm512_add_pd(dsum0, _mm512_cvtps_pd(lo));
        dsum1 = _mm512_add_pd(dsum1, _mm512_cvtps_pd(hi));
    }

    __m512d total = _mm512_add_pd(dsum0, dsum1);
    __m256d tlo = _mm512_castpd512_pd256(total);
    __m256d thi = _mm512_extractf64x4_pd(total, 1);
    __m256d t4 = _mm256_add_pd(tlo, thi);
    __m128d t2lo = _mm256_castpd256_pd128(t4);
    __m128d t2hi = _mm256_extractf128_pd(t4, 1);
    __m128d s = _mm_add_pd(t2lo, t2hi);
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    double result = _mm_cvtsd_f64(s);

    for (; j < w; j++) {
        float diff = ref[j] - dis[j];
        result += (double)(diff * diff);
    }

    return result;
}
