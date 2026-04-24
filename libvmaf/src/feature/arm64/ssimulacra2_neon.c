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
 * aarch64 NEON port of the SSIMULACRA 2 SIMD kernels. Structural
 * mirror of the AVX2 / AVX-512 TUs (ADR-0161) — 4-wide float lanes.
 *
 * `cbrtf` applied per-lane via scalar libm. The 2x2 downsample's
 * deinterleave uses `vuzp1q_f32` / `vuzp2q_f32` to pull even / odd
 * positions into separate vectors; sequential adds preserve the
 * scalar left-to-right summation order.
 */

#include <arm_neon.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "ssimulacra2_neon.h"

#pragma STDC FP_CONTRACT OFF

static const float kM00 = 0.30f;
static const float kM02 = 0.078f;
static const float kM10 = 0.23f;
static const float kM12 = 0.078f;
static const float kM20 = 0.24342268924547819f;
static const float kM21 = 0.20476744424496821f;
static const float kOpsinBias = 0.0037930732552754493f;
static const float kC2 = 0.0009f;

static inline float32x4_t cbrtf_lane_neon(float32x4_t v)
{
    float tmp[4] __attribute__((aligned(16)));
    vst1q_f32(tmp, v);
    for (int k = 0; k < 4; k++) {
        tmp[k] = cbrtf(tmp[k]);
    }
    return vld1q_f32(tmp);
}

static inline double quartic_d(double x)
{
    x *= x;
    return x * x;
}

void ssimulacra2_multiply_3plane_neon(const float *a, const float *b, float *mul, unsigned w,
                                      unsigned h)
{
    const size_t n = 3u * (size_t)w * (size_t)h;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(mul + i, vmulq_f32(va, vb));
    }
    for (; i < n; i++) {
        mul[i] = a[i] * b[i];
    }
}

// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
void ssimulacra2_linear_rgb_to_xyb_neon(const float *lin, float *xyb, unsigned w, unsigned h)
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

    const float32x4_t vm00 = vdupq_n_f32(kM00);
    const float32x4_t vm01 = vdupq_n_f32(m01);
    const float32x4_t vm02 = vdupq_n_f32(kM02);
    const float32x4_t vm10 = vdupq_n_f32(kM10);
    const float32x4_t vm11 = vdupq_n_f32(m11);
    const float32x4_t vm12 = vdupq_n_f32(kM12);
    const float32x4_t vm20 = vdupq_n_f32(kM20);
    const float32x4_t vm21 = vdupq_n_f32(kM21);
    const float32x4_t vm22 = vdupq_n_f32(m22);
    const float32x4_t vbias = vdupq_n_f32(kOpsinBias);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vcbrt_bias = vdupq_n_f32(cbrt_bias);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const float32x4_t v14 = vdupq_n_f32(14.0f);
    const float32x4_t v42 = vdupq_n_f32(0.42f);
    const float32x4_t v55 = vdupq_n_f32(0.55f);
    const float32x4_t v01 = vdupq_n_f32(0.01f);

    size_t i = 0;
    for (; i + 4 <= plane_sz; i += 4) {
        const float32x4_t r = vld1q_f32(rp + i);
        const float32x4_t g = vld1q_f32(gp + i);
        const float32x4_t b = vld1q_f32(bp + i);
        float32x4_t l = vaddq_f32(vmulq_f32(vm00, r), vmulq_f32(vm01, g));
        l = vaddq_f32(l, vmulq_f32(vm02, b));
        l = vaddq_f32(l, vbias);
        float32x4_t m = vaddq_f32(vmulq_f32(vm10, r), vmulq_f32(vm11, g));
        m = vaddq_f32(m, vmulq_f32(vm12, b));
        m = vaddq_f32(m, vbias);
        float32x4_t sv = vaddq_f32(vmulq_f32(vm20, r), vmulq_f32(vm21, g));
        sv = vaddq_f32(sv, vmulq_f32(vm22, b));
        sv = vaddq_f32(sv, vbias);
        l = vmaxq_f32(l, vzero);
        m = vmaxq_f32(m, vzero);
        sv = vmaxq_f32(sv, vzero);
        const float32x4_t L = vsubq_f32(cbrtf_lane_neon(l), vcbrt_bias);
        const float32x4_t M = vsubq_f32(cbrtf_lane_neon(m), vcbrt_bias);
        const float32x4_t S = vsubq_f32(cbrtf_lane_neon(sv), vcbrt_bias);
        const float32x4_t X = vmulq_f32(vhalf, vsubq_f32(L, M));
        const float32x4_t Y = vmulq_f32(vhalf, vaddq_f32(L, M));
        const float32x4_t B = S;
        const float32x4_t Bfinal = vaddq_f32(vsubq_f32(B, Y), v55);
        const float32x4_t Xfinal = vaddq_f32(vmulq_f32(X, v14), v42);
        const float32x4_t Yfinal = vaddq_f32(Y, v01);
        vst1q_f32(xp + i, Xfinal);
        vst1q_f32(yp + i, Yfinal);
        vst1q_f32(bxp + i, Bfinal);
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

/* ADR-0141 carve-out: the outer loop iterates per-plane × per-row ×
 * per-tile and keeps the SIMD-deinterleave + scalar-tail together for
 * the line-for-line scalar diff audit. Splitting mid-iteration would
 * duplicate the bounds-clamp logic. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
void ssimulacra2_downsample_2x2_neon(const float *in, unsigned iw, unsigned ih, float *out,
                                     unsigned *ow_out, unsigned *oh_out)
{
    const unsigned ow = (iw + 1) / 2;
    const unsigned oh = (ih + 1) / 2;
    *ow_out = ow;
    *oh_out = oh;

    const size_t in_plane = (size_t)iw * (size_t)ih;
    const size_t out_plane = (size_t)ow * (size_t)oh;
    const float32x4_t vquarter = vdupq_n_f32(0.25f);

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
            const unsigned interior_end = (ow > 0 && iw >= 2) ? ((ow - 1) / 4) * 4 : 0;
            for (; ox < interior_end; ox += 4) {
                const size_t base = (size_t)ox * 2u;
                const float32x4_t r00 = vld1q_f32(row0 + base);
                const float32x4_t r01 = vld1q_f32(row0 + base + 4);
                const float32x4_t r10 = vld1q_f32(row1 + base);
                const float32x4_t r11 = vld1q_f32(row1 + base + 4);
                const float32x4_t r0e = vuzp1q_f32(r00, r01);
                const float32x4_t r0o = vuzp2q_f32(r00, r01);
                const float32x4_t r1e = vuzp1q_f32(r10, r11);
                const float32x4_t r1o = vuzp2q_f32(r10, r11);
                float32x4_t acc = vaddq_f32(r0e, r0o);
                acc = vaddq_f32(acc, r1e);
                acc = vaddq_f32(acc, r1o);
                vst1q_f32(orow + ox, vmulq_f32(acc, vquarter));
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
void ssimulacra2_ssim_map_neon(const float *m1, const float *m2, const float *s11, const float *s22,
                               const float *s12, unsigned w, unsigned h, double plane_averages[6])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;
    const float32x4_t vc2 = vdupq_n_f32(kC2);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    const float32x4_t vtwo = vdupq_n_f32(2.0f);

    for (int c = 0; c < 3; c++) {
        double sum_l1 = 0.0;
        double sum_l4 = 0.0;
        const float *rm1 = m1 + (size_t)c * plane;
        const float *rm2 = m2 + (size_t)c * plane;
        const float *rs11 = s11 + (size_t)c * plane;
        const float *rs22 = s22 + (size_t)c * plane;
        const float *rs12 = s12 + (size_t)c * plane;

        size_t i = 0;
        for (; i + 4 <= plane; i += 4) {
            const float32x4_t mu1 = vld1q_f32(rm1 + i);
            const float32x4_t mu2 = vld1q_f32(rm2 + i);
            const float32x4_t mu11 = vmulq_f32(mu1, mu1);
            const float32x4_t mu22 = vmulq_f32(mu2, mu2);
            const float32x4_t mu12 = vmulq_f32(mu1, mu2);
            const float32x4_t diff = vsubq_f32(mu1, mu2);
            const float32x4_t num_m = vsubq_f32(vone, vmulq_f32(diff, diff));
            const float32x4_t num_s =
                vaddq_f32(vmulq_f32(vtwo, vsubq_f32(vld1q_f32(rs12 + i), mu12)), vc2);
            const float32x4_t denom_s = vaddq_f32(vaddq_f32(vsubq_f32(vld1q_f32(rs11 + i), mu11),
                                                            vsubq_f32(vld1q_f32(rs22 + i), mu22)),
                                                  vc2);
            float num_m_f[4] __attribute__((aligned(16)));
            float num_s_f[4] __attribute__((aligned(16)));
            float denom_s_f[4] __attribute__((aligned(16)));
            vst1q_f32(num_m_f, num_m);
            vst1q_f32(num_s_f, num_s);
            vst1q_f32(denom_s_f, denom_s);
            for (int k = 0; k < 4; k++) {
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

void ssimulacra2_edge_diff_map_neon(const float *img1, const float *mu1, const float *img2,
                                    const float *mu2, unsigned w, unsigned h,
                                    double plane_averages[12])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;

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
        for (; i + 4 <= plane; i += 4) {
            const float32x4_t a1 = vld1q_f32(r1 + i);
            const float32x4_t a2 = vld1q_f32(r2 + i);
            const float32x4_t am1 = vld1q_f32(rm1 + i);
            const float32x4_t am2 = vld1q_f32(rm2 + i);
            const float32x4_t d1 = vabsq_f32(vsubq_f32(a1, am1));
            const float32x4_t d2 = vabsq_f32(vsubq_f32(a2, am2));
            float d1f[4] __attribute__((aligned(16)));
            float d2f[4] __attribute__((aligned(16)));
            vst1q_f32(d1f, d1);
            vst1q_f32(d2f, d2);
            for (int k = 0; k < 4; k++) {
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
