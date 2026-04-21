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

#include <arm_neon.h>
#include <math.h>

#include "../simd_dx.h"
#include "ssim_neon.h"

/*
 * Bit-exact NEON SSIM SIMD helpers — mirror the scalar reference in
 * libvmaf/src/feature/iqa/ssim_tools.c byte-for-byte under
 * FLT_EVAL_METHOD == 0. See ADR-0139. `precompute` and `variance` are
 * pure elementwise float ops and are bit-exact to scalar by
 * construction. `accumulate` is the tricky one: scalar computes l and c
 * in double (because of the `2.0 * ...` literal) and does the final
 * l*c*s product in double, so we keep SIMD for the float-valued
 * intermediates (srsc, denominators, sv) and do the double-precision
 * numerator + division + product per-lane in scalar double, preserving
 * both the type promotions and the reduction order.
 */

void ssim_precompute_neon(const float *ref, const float *cmp, float *ref_sq, float *cmp_sq,
                          float *ref_cmp, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t r = vld1q_f32(ref + i);
        float32x4_t c = vld1q_f32(cmp + i);
        vst1q_f32(ref_sq + i, vmulq_f32(r, r));
        vst1q_f32(cmp_sq + i, vmulq_f32(c, c));
        vst1q_f32(ref_cmp + i, vmulq_f32(r, c));
    }
    for (; i < n; i++) {
        ref_sq[i] = ref[i] * ref[i];
        cmp_sq[i] = cmp[i] * cmp[i];
        ref_cmp[i] = ref[i] * cmp[i];
    }
}

void ssim_variance_neon(float *ref_sigma_sqd, float *cmp_sigma_sqd, float *sigma_both,
                        const float *ref_mu, const float *cmp_mu, int n)
{
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t rm = vld1q_f32(ref_mu + i);
        float32x4_t cm = vld1q_f32(cmp_mu + i);

        float32x4_t rs = vld1q_f32(ref_sigma_sqd + i);
        rs = vsubq_f32(rs, vmulq_f32(rm, rm));
        rs = vmaxq_f32(rs, zero);
        vst1q_f32(ref_sigma_sqd + i, rs);

        float32x4_t cs = vld1q_f32(cmp_sigma_sqd + i);
        cs = vsubq_f32(cs, vmulq_f32(cm, cm));
        cs = vmaxq_f32(cs, zero);
        vst1q_f32(cmp_sigma_sqd + i, cs);

        float32x4_t sb = vld1q_f32(sigma_both + i);
        sb = vsubq_f32(sb, vmulq_f32(rm, cm));
        vst1q_f32(sigma_both + i, sb);
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

void ssim_accumulate_neon(const float *ref_mu, const float *cmp_mu, const float *ref_sigma_sqd,
                          const float *cmp_sigma_sqd, const float *sigma_both, int n, float C1,
                          float C2, float C3, double *ssim_sum, double *l_sum, double *c_sum,
                          double *s_sum)
{
    const float32x4_t vC1 = vdupq_n_f32(C1);
    const float32x4_t vC2 = vdupq_n_f32(C2);
    const float32x4_t vC3 = vdupq_n_f32(C3);
    const float32x4_t vzero = vdupq_n_f32(0.0f);

    double local_ssim = 0.0;
    double local_l = 0.0;
    double local_c = 0.0;
    double local_s = 0.0;

    int i = 0;
    for (; i + SIMD_LANES_NEON <= n; i += SIMD_LANES_NEON) {
        const float32x4_t rm = vld1q_f32(ref_mu + i);
        const float32x4_t cm = vld1q_f32(cmp_mu + i);
        const float32x4_t rs = vld1q_f32(ref_sigma_sqd + i);
        const float32x4_t cs = vld1q_f32(cmp_sigma_sqd + i);
        const float32x4_t sb = vld1q_f32(sigma_both + i);

        /* srsc = sqrtf(rs * cs) — float mul + float sqrt. */
        const float32x4_t srsc = vsqrtq_f32(vmulq_f32(rs, cs));

        /* Float denominators, matching scalar order: ((a+b)+C). */
        const float32x4_t l_den = vaddq_f32(vaddq_f32(vmulq_f32(rm, rm), vmulq_f32(cm, cm)), vC1);
        const float32x4_t c_den = vaddq_f32(vaddq_f32(rs, cs), vC2);

        /* s numerator: clamp sb to 0 iff sb<0 AND srsc<=0, add C3. */
        const uint32x4_t sb_neg = vcltq_f32(sb, vzero);
        const uint32x4_t srsc_le0 = vcleq_f32(srsc, vzero);
        const uint32x4_t clamp_mask = vandq_u32(sb_neg, srsc_le0);
        const float32x4_t clamped_sb = vbslq_f32(clamp_mask, vzero, sb);
        /* sv = (csb + C3) / (srsc + C3) — all float; promoted to double
         * at per-lane load below. */
        const float32x4_t sv_f = vdivq_f32(vaddq_f32(clamped_sb, vC3), vaddq_f32(srsc, vC3));

        SIMD_ALIGNED_F32_BUF_NEON(t_rm);
        SIMD_ALIGNED_F32_BUF_NEON(t_cm);
        SIMD_ALIGNED_F32_BUF_NEON(t_srsc);
        SIMD_ALIGNED_F32_BUF_NEON(t_l_den);
        SIMD_ALIGNED_F32_BUF_NEON(t_c_den);
        SIMD_ALIGNED_F32_BUF_NEON(t_sv);
        vst1q_f32(t_rm, rm);
        vst1q_f32(t_cm, cm);
        vst1q_f32(t_srsc, srsc);
        vst1q_f32(t_l_den, l_den);
        vst1q_f32(t_c_den, c_den);
        vst1q_f32(t_sv, sv_f);

        /* Per-lane scalar-double reduction. Matches scalar:
         *   lv = (2.0 * rm * cm + C1) / (rm*rm + cm*cm + C1)     [double]
         *   cv = (2.0 * srsc + C2)    / (rs + cs + C2)           [double]
         *   sv = (csb + C3) / (srsc + C3)                        [float→double]
         *   ssim += lv * cv * sv
         */
        for (int k = 0; k < SIMD_LANES_NEON; k++) {
            const double lv = (2.0 * t_rm[k] * t_cm[k] + C1) / t_l_den[k];
            const double cv = (2.0 * t_srsc[k] + C2) / t_c_den[k];
            const double sv = t_sv[k];
            local_ssim += lv * cv * sv;
            local_l += lv;
            local_c += cv;
            local_s += sv;
        }
    }

    /* Scalar tail */
    for (; i < n; i++) {
        const float srsc = sqrtf(ref_sigma_sqd[i] * cmp_sigma_sqd[i]);
        const double lv = (2.0 * ref_mu[i] * cmp_mu[i] + C1) /
                          (ref_mu[i] * ref_mu[i] + cmp_mu[i] * cmp_mu[i] + C1);
        const double cv = (2.0 * srsc + C2) / (ref_sigma_sqd[i] + cmp_sigma_sqd[i] + C2);
        const float csb = (sigma_both[i] < 0.0f && srsc <= 0.0f) ? 0.0f : sigma_both[i];
        const double sv = (csb + C3) / (srsc + C3);
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
