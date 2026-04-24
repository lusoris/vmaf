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
 * AVX-512 port of the SSIMULACRA 2 SIMD kernels. Mechanical 16-wide
 * widening of the AVX2 sister TU (x86/ssimulacra2_avx2.c, ADR-0161).
 * Same bit-exactness contract: byte-for-byte identical output to the
 * scalar reference in feature/ssimulacra2.c under FLT_EVAL_METHOD == 0.
 *
 * The shape of the ops is identical — only the lane count changes.
 * IEEE-754 lane-commutative adds/muls + per-lane scalar libm for
 * `cbrtf` preserve the summation tree byte-for-byte.
 *
 * Downsample_2x2's deinterleave uses AVX-512's `vpermt2ps` with
 * explicit index vectors, providing cleaner cross-lane rearrangement
 * than the AVX2 `vshufps + vpermpd` chain.
 */

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ssimulacra2_avx512.h"

#pragma STDC FP_CONTRACT OFF

static const float kM00 = 0.30f;
static const float kM02 = 0.078f;
static const float kM10 = 0.23f;
static const float kM12 = 0.078f;
static const float kM20 = 0.24342268924547819f;
static const float kM21 = 0.20476744424496821f;
static const float kOpsinBias = 0.0037930732552754493f;
static const float kC2 = 0.0009f;

static inline __m512 cbrtf_lane_avx512(__m512 v)
{
    float tmp[16] __attribute__((aligned(64)));
    _mm512_store_ps(tmp, v);
    for (int k = 0; k < 16; k++) {
        tmp[k] = cbrtf(tmp[k]);
    }
    return _mm512_load_ps(tmp);
}

static inline double quartic_d(double x)
{
    x *= x;
    return x * x;
}

void ssimulacra2_multiply_3plane_avx512(const float *a, const float *b, float *mul, unsigned w,
                                        unsigned h)
{
    const size_t n = 3u * (size_t)w * (size_t)h;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(mul + i, _mm512_mul_ps(va, vb));
    }
    for (; i < n; i++) {
        mul[i] = a[i] * b[i];
    }
}

// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
void ssimulacra2_linear_rgb_to_xyb_avx512(const float *lin, float *xyb, unsigned w, unsigned h)
{
    assert(lin != NULL);
    assert(xyb != NULL);
    assert(w > 0 && h > 0);
    const size_t plane_sz = (size_t)w * (size_t)h;
    const float *rp = lin;
    const float *gp = lin + plane_sz;
    const float *bp = lin + 2 * plane_sz;
    float *xp = xyb;
    float *yp = xyb + plane_sz;
    float *bxp = xyb + 2 * plane_sz;

    const float m01 = 1.0f - kM00 - kM02;
    const float m11 = 1.0f - kM10 - kM12;
    const float m22 = 1.0f - kM20 - kM21;
    const float cbrt_bias = cbrtf(kOpsinBias);

    const __m512 vm00 = _mm512_set1_ps(kM00);
    const __m512 vm01 = _mm512_set1_ps(m01);
    const __m512 vm02 = _mm512_set1_ps(kM02);
    const __m512 vm10 = _mm512_set1_ps(kM10);
    const __m512 vm11 = _mm512_set1_ps(m11);
    const __m512 vm12 = _mm512_set1_ps(kM12);
    const __m512 vm20 = _mm512_set1_ps(kM20);
    const __m512 vm21 = _mm512_set1_ps(kM21);
    const __m512 vm22 = _mm512_set1_ps(m22);
    const __m512 vbias = _mm512_set1_ps(kOpsinBias);
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vcbrt_bias = _mm512_set1_ps(cbrt_bias);
    const __m512 vhalf = _mm512_set1_ps(0.5f);
    const __m512 v14 = _mm512_set1_ps(14.0f);
    const __m512 v42 = _mm512_set1_ps(0.42f);
    const __m512 v55 = _mm512_set1_ps(0.55f);
    const __m512 v01 = _mm512_set1_ps(0.01f);

    size_t i = 0;
    for (; i + 16 <= plane_sz; i += 16) {
        const __m512 r = _mm512_loadu_ps(rp + i);
        const __m512 g = _mm512_loadu_ps(gp + i);
        const __m512 b = _mm512_loadu_ps(bp + i);
        __m512 l = _mm512_add_ps(_mm512_mul_ps(vm00, r), _mm512_mul_ps(vm01, g));
        l = _mm512_add_ps(l, _mm512_mul_ps(vm02, b));
        l = _mm512_add_ps(l, vbias);
        __m512 m = _mm512_add_ps(_mm512_mul_ps(vm10, r), _mm512_mul_ps(vm11, g));
        m = _mm512_add_ps(m, _mm512_mul_ps(vm12, b));
        m = _mm512_add_ps(m, vbias);
        __m512 sv = _mm512_add_ps(_mm512_mul_ps(vm20, r), _mm512_mul_ps(vm21, g));
        sv = _mm512_add_ps(sv, _mm512_mul_ps(vm22, b));
        sv = _mm512_add_ps(sv, vbias);
        l = _mm512_max_ps(l, vzero);
        m = _mm512_max_ps(m, vzero);
        sv = _mm512_max_ps(sv, vzero);
        const __m512 L = _mm512_sub_ps(cbrtf_lane_avx512(l), vcbrt_bias);
        const __m512 M = _mm512_sub_ps(cbrtf_lane_avx512(m), vcbrt_bias);
        const __m512 S = _mm512_sub_ps(cbrtf_lane_avx512(sv), vcbrt_bias);
        const __m512 X = _mm512_mul_ps(vhalf, _mm512_sub_ps(L, M));
        const __m512 Y = _mm512_mul_ps(vhalf, _mm512_add_ps(L, M));
        const __m512 B = S;
        const __m512 Bfinal = _mm512_add_ps(_mm512_sub_ps(B, Y), v55);
        const __m512 Xfinal = _mm512_add_ps(_mm512_mul_ps(X, v14), v42);
        const __m512 Yfinal = _mm512_add_ps(Y, v01);
        _mm512_storeu_ps(xp + i, Xfinal);
        _mm512_storeu_ps(yp + i, Yfinal);
        _mm512_storeu_ps(bxp + i, Bfinal);
    }

    for (; i < plane_sz; i++) {
        float r = rp[i];
        float g = gp[i];
        float bb = bp[i];
        float l = kM00 * r + m01 * g + kM02 * bb + kOpsinBias;
        float m = kM10 * r + m11 * g + kM12 * bb + kOpsinBias;
        float s = kM20 * r + kM21 * g + m22 * bb + kOpsinBias;
        if (l < 0.0f)
            l = 0.0f;
        if (m < 0.0f)
            m = 0.0f;
        if (s < 0.0f)
            s = 0.0f;
        float L = cbrtf(l) - cbrt_bias;
        float M = cbrtf(m) - cbrt_bias;
        float S = cbrtf(s) - cbrt_bias;
        float X = 0.5f * (L - M);
        float Y = 0.5f * (L + M);
        float B = S;
        B = (B - Y) + 0.55f;
        X = X * 14.0f + 0.42f;
        Y = Y + 0.01f;
        xp[i] = X;
        yp[i] = Y;
        bxp[i] = B;
    }
}

void ssimulacra2_downsample_2x2_avx512(const float *in, unsigned iw, unsigned ih, float *out,
                                       unsigned *ow_out, unsigned *oh_out)
{
    const unsigned ow = (iw + 1) / 2;
    const unsigned oh = (ih + 1) / 2;
    *ow_out = ow;
    *oh_out = oh;

    const size_t in_plane = (size_t)iw * (size_t)ih;
    const size_t out_plane = (size_t)ow * (size_t)oh;

    /* Index vectors for `vpermt2ps` even-/odd-lane deinterleave across
     * two __m512 source vectors (32 consecutive floats). */
    const __m512i idx_even =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_odd =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    const __m512 vquarter = _mm512_set1_ps(0.25f);

    for (int c = 0; c < 3; c++) {
        const float *ip = in + (size_t)c * in_plane;
        float *op = out + (size_t)c * out_plane;
        for (unsigned oy = 0; oy < oh; oy++) {
            const unsigned iy0 = oy * 2;
            const unsigned iy1 = (iy0 + 1 < ih) ? iy0 + 1 : ih - 1;
            const float *row0 = ip + (size_t)iy0 * iw;
            const float *row1 = ip + (size_t)iy1 * iw;
            float *orow = op + (size_t)oy * ow;
            unsigned ox = 0;
            const unsigned interior_end = (ow > 0 && iw >= 2) ? ((ow - 1) / 16) * 16 : 0;
            for (; ox < interior_end; ox += 16) {
                const size_t base = (size_t)ox * 2u;
                const __m512 r00 = _mm512_loadu_ps(row0 + base);
                const __m512 r01 = _mm512_loadu_ps(row0 + base + 16);
                const __m512 r10 = _mm512_loadu_ps(row1 + base);
                const __m512 r11 = _mm512_loadu_ps(row1 + base + 16);
                const __m512 r0e = _mm512_permutex2var_ps(r00, idx_even, r01);
                const __m512 r0o = _mm512_permutex2var_ps(r00, idx_odd, r01);
                const __m512 r1e = _mm512_permutex2var_ps(r10, idx_even, r11);
                const __m512 r1o = _mm512_permutex2var_ps(r10, idx_odd, r11);
                __m512 acc = _mm512_add_ps(r0e, r0o);
                acc = _mm512_add_ps(acc, r1e);
                acc = _mm512_add_ps(acc, r1o);
                _mm512_storeu_ps(orow + ox, _mm512_mul_ps(acc, vquarter));
            }
            for (; ox < ow; ox++) {
                unsigned ix0 = ox * 2;
                unsigned ix1 = (ix0 + 1 < iw) ? ix0 + 1 : iw - 1;
                float sum = row0[ix0] + row0[ix1] + row1[ix0] + row1[ix1];
                orow[ox] = sum * 0.25f;
            }
        }
    }
}

// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
void ssimulacra2_ssim_map_avx512(const float *m1, const float *m2, const float *s11,
                                 const float *s22, const float *s12, unsigned w, unsigned h,
                                 double plane_averages[6])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;
    const __m512 vc2 = _mm512_set1_ps(kC2);
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vtwo = _mm512_set1_ps(2.0f);

    for (int c = 0; c < 3; c++) {
        double sum_l1 = 0.0;
        double sum_l4 = 0.0;
        const float *rm1 = m1 + (size_t)c * plane;
        const float *rm2 = m2 + (size_t)c * plane;
        const float *rs11 = s11 + (size_t)c * plane;
        const float *rs22 = s22 + (size_t)c * plane;
        const float *rs12 = s12 + (size_t)c * plane;

        size_t i = 0;
        for (; i + 16 <= plane; i += 16) {
            const __m512 mu1 = _mm512_loadu_ps(rm1 + i);
            const __m512 mu2 = _mm512_loadu_ps(rm2 + i);
            const __m512 mu11 = _mm512_mul_ps(mu1, mu1);
            const __m512 mu22 = _mm512_mul_ps(mu2, mu2);
            const __m512 mu12 = _mm512_mul_ps(mu1, mu2);
            const __m512 diff = _mm512_sub_ps(mu1, mu2);
            const __m512 num_m = _mm512_sub_ps(vone, _mm512_mul_ps(diff, diff));
            const __m512 num_s = _mm512_add_ps(
                _mm512_mul_ps(vtwo, _mm512_sub_ps(_mm512_loadu_ps(rs12 + i), mu12)), vc2);
            const __m512 denom_s =
                _mm512_add_ps(_mm512_add_ps(_mm512_sub_ps(_mm512_loadu_ps(rs11 + i), mu11),
                                            _mm512_sub_ps(_mm512_loadu_ps(rs22 + i), mu22)),
                              vc2);
            float num_m_f[16] __attribute__((aligned(64)));
            float num_s_f[16] __attribute__((aligned(64)));
            float denom_s_f[16] __attribute__((aligned(64)));
            _mm512_store_ps(num_m_f, num_m);
            _mm512_store_ps(num_s_f, num_s);
            _mm512_store_ps(denom_s_f, denom_s);
            for (int k = 0; k < 16; k++) {
                double d = 1.0 - ((double)num_m_f[k] * (double)num_s_f[k] / (double)denom_s_f[k]);
                if (d < 0.0)
                    d = 0.0;
                sum_l1 += d;
                sum_l4 += quartic_d(d);
            }
        }
        for (; i < plane; i++) {
            float mu1 = rm1[i];
            float mu2 = rm2[i];
            float mu11 = mu1 * mu1;
            float mu22 = mu2 * mu2;
            float mu12 = mu1 * mu2;
            float num_m = 1.0f - (mu1 - mu2) * (mu1 - mu2);
            float num_s = 2.0f * (rs12[i] - mu12) + kC2;
            float denom_s = (rs11[i] - mu11) + (rs22[i] - mu22) + kC2;
            double d = 1.0 - ((double)num_m * (double)num_s / (double)denom_s);
            if (d < 0.0)
                d = 0.0;
            sum_l1 += d;
            sum_l4 += quartic_d(d);
        }
        plane_averages[c * 2 + 0] = one_per_pixels * sum_l1;
        plane_averages[c * 2 + 1] = sqrt(sqrt(one_per_pixels * sum_l4));
    }
}

void ssimulacra2_edge_diff_map_avx512(const float *img1, const float *mu1, const float *img2,
                                      const float *mu2, unsigned w, unsigned h,
                                      double plane_averages[12])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;
    const __m512 vsignmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

    for (int c = 0; c < 3; c++) {
        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
        const float *r1 = img1 + (size_t)c * plane;
        const float *rm1 = mu1 + (size_t)c * plane;
        const float *r2 = img2 + (size_t)c * plane;
        const float *rm2 = mu2 + (size_t)c * plane;

        size_t i = 0;
        for (; i + 16 <= plane; i += 16) {
            const __m512 a1 = _mm512_loadu_ps(r1 + i);
            const __m512 a2 = _mm512_loadu_ps(r2 + i);
            const __m512 am1 = _mm512_loadu_ps(rm1 + i);
            const __m512 am2 = _mm512_loadu_ps(rm2 + i);
            const __m512 d1 = _mm512_and_ps(vsignmask, _mm512_sub_ps(a1, am1));
            const __m512 d2 = _mm512_and_ps(vsignmask, _mm512_sub_ps(a2, am2));
            float d1f[16] __attribute__((aligned(64)));
            float d2f[16] __attribute__((aligned(64)));
            _mm512_store_ps(d1f, d1);
            _mm512_store_ps(d2f, d2);
            for (int k = 0; k < 16; k++) {
                double ed1 = (double)d1f[k];
                double ed2 = (double)d2f[k];
                double d = (1.0 + ed2) / (1.0 + ed1) - 1.0;
                double art = d > 0.0 ? d : 0.0;
                double det = d < 0.0 ? -d : 0.0;
                s0 += art;
                s1 += quartic_d(art);
                s2 += det;
                s3 += quartic_d(det);
            }
        }
        for (; i < plane; i++) {
            double ed1 = fabs((double)r1[i] - (double)rm1[i]);
            double ed2 = fabs((double)r2[i] - (double)rm2[i]);
            double d = (1.0 + ed2) / (1.0 + ed1) - 1.0;
            double art = d > 0.0 ? d : 0.0;
            double det = d < 0.0 ? -d : 0.0;
            s0 += art;
            s1 += quartic_d(art);
            s2 += det;
            s3 += quartic_d(det);
        }
        plane_averages[c * 4 + 0] = one_per_pixels * s0;
        plane_averages[c * 4 + 1] = sqrt(sqrt(one_per_pixels * s1));
        plane_averages[c * 4 + 2] = one_per_pixels * s2;
        plane_averages[c * 4 + 3] = sqrt(sqrt(one_per_pixels * s3));
    }
}
