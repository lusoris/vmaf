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
 * Bit-exactness contract test for the SSIMULACRA 2 SIMD kernels.
 *
 * Each vectorised kernel must produce byte-for-byte identical output
 * to its scalar counterpart on reproducible pseudo-random inputs
 * (ADR-0161). Scalar references are duplicated in-file because the
 * extractor keeps them `static` — the test TU is the audit surface
 * for the bit-exact contract.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "test.h"

#if ARCH_X86
#include "feature/x86/ssimulacra2_avx2.h"
#if HAVE_AVX512
#include "feature/x86/ssimulacra2_avx512.h"
#endif
#include "x86/cpu.h"
#endif
#if ARCH_AARCH64
#include "feature/arm64/ssimulacra2_neon.h"
#endif

static const float kM00 = 0.30f;
static const float kM02 = 0.078f;
static const float kM10 = 0.23f;
static const float kM12 = 0.078f;
static const float kM20 = 0.24342268924547819f;
static const float kM21 = 0.20476744424496821f;
static const float kOpsinBias = 0.0037930732552754493f;
static const float kC2 = 0.0009f;

/* xorshift32 for reproducible float input generation. */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    return s;
}

/* Fill `n` floats with pseudo-random values in [lo, hi). */
static void fill_random(float *buf, size_t n, float lo, float hi, uint32_t seed)
{
    uint32_t state = seed;
    const float range = hi - lo;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32(&state);
        buf[i] = lo + range * ((float)(r & 0x00ffffff) / (float)0x01000000);
    }
}

/* Scalar reference: multiply_3plane */
static void ref_multiply_3plane(const float *a, const float *b, float *mul, unsigned w, unsigned h)
{
    const size_t n = 3u * (size_t)w * (size_t)h;
    for (size_t i = 0; i < n; i++) {
        mul[i] = a[i] * b[i];
    }
}

/* Scalar reference: linear_rgb_to_xyb */
static void ref_linear_rgb_to_xyb(const float *lin, float *xyb, unsigned w, unsigned h)
{
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
    for (size_t i = 0; i < plane_sz; i++) {
        float r = rp[i];
        float g = gp[i];
        float b = bp[i];
        float l = kM00 * r + m01 * g + kM02 * b + kOpsinBias;
        float m = kM10 * r + m11 * g + kM12 * b + kOpsinBias;
        float s = kM20 * r + kM21 * g + m22 * b + kOpsinBias;
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

/* Scalar reference: downsample_2x2. Kept as a line-for-line copy of the
 * extractor's scalar reference for bit-exactness auditability. */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
static void ref_downsample_2x2(const float *in, unsigned iw, unsigned ih, float *out,
                               unsigned *ow_out, unsigned *oh_out)
{
    const unsigned ow = (iw + 1) / 2;
    const unsigned oh = (ih + 1) / 2;
    *ow_out = ow;
    *oh_out = oh;
    const size_t in_plane = (size_t)iw * (size_t)ih;
    const size_t out_plane = (size_t)ow * (size_t)oh;
    for (int c = 0; c < 3; c++) {
        const float *ip = in + (size_t)c * in_plane;
        float *op = out + (size_t)c * out_plane;
        for (unsigned oy = 0; oy < oh; oy++) {
            for (unsigned ox = 0; ox < ow; ox++) {
                float sum = 0.0f;
                for (unsigned dy = 0; dy < 2; dy++) {
                    for (unsigned dx = 0; dx < 2; dx++) {
                        unsigned ix = ox * 2 + dx;
                        unsigned iy = oy * 2 + dy;
                        if (ix >= iw)
                            ix = iw - 1;
                        if (iy >= ih)
                            iy = ih - 1;
                        sum += ip[(size_t)iy * iw + ix];
                    }
                }
                op[(size_t)oy * ow + ox] = sum * 0.25f;
            }
        }
    }
}

static inline double quartic_d(double x)
{
    x *= x;
    return x * x;
}

/* Scalar reference: ssim_map */
static void ref_ssim_map(const float *m1, const float *m2, const float *s11, const float *s22,
                         const float *s12, unsigned w, unsigned h, double plane_averages[6])
{
    const size_t plane = (size_t)w * (size_t)h;
    const double one_per_pixels = 1.0 / (double)plane;
    for (int c = 0; c < 3; c++) {
        double sum_l1 = 0.0;
        double sum_l4 = 0.0;
        const float *rm1 = m1 + (size_t)c * plane;
        const float *rm2 = m2 + (size_t)c * plane;
        const float *rs11 = s11 + (size_t)c * plane;
        const float *rs22 = s22 + (size_t)c * plane;
        const float *rs12 = s12 + (size_t)c * plane;
        for (size_t i = 0; i < plane; i++) {
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

/* Scalar reference: edge_diff_map */
static void ref_edge_diff_map(const float *img1, const float *mu1, const float *img2,
                              const float *mu2, unsigned w, unsigned h, double plane_averages[12])
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
        for (size_t i = 0; i < plane; i++) {
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

#define W 33 /* deliberately non-multiple of 8 / 16 to exercise tails */
#define H 21
#define PLANE_SZ ((size_t)W * (size_t)H)
#define RGB_SZ (3u * PLANE_SZ)

/* SIMD dispatcher: pick AVX2 on x86, NEON on aarch64. Returns NULL on
 * platforms where no SIMD variant is available (the test will skip). */
typedef void (*mul3_fn_t)(const float *, const float *, float *, unsigned, unsigned);
typedef void (*xyb_fn_t)(const float *, float *, unsigned, unsigned);
typedef void (*down_fn_t)(const float *, unsigned, unsigned, float *, unsigned *, unsigned *);
typedef void (*ssim_fn_t)(const float *, const float *, const float *, const float *, const float *,
                          unsigned, unsigned, double[6]);
typedef void (*edge_fn_t)(const float *, const float *, const float *, const float *, unsigned,
                          unsigned, double[12]);

static mul3_fn_t pick_mul3(void)
{
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags_x86();
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        return ssimulacra2_multiply_3plane_avx512;
#endif
    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        return ssimulacra2_multiply_3plane_avx2;
#endif
#if ARCH_AARCH64
    return ssimulacra2_multiply_3plane_neon;
#endif
    return NULL;
}
static xyb_fn_t pick_xyb(void)
{
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags_x86();
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        return ssimulacra2_linear_rgb_to_xyb_avx512;
#endif
    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        return ssimulacra2_linear_rgb_to_xyb_avx2;
#endif
#if ARCH_AARCH64
    return ssimulacra2_linear_rgb_to_xyb_neon;
#endif
    return NULL;
}
static down_fn_t pick_down(void)
{
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags_x86();
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        return ssimulacra2_downsample_2x2_avx512;
#endif
    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        return ssimulacra2_downsample_2x2_avx2;
#endif
#if ARCH_AARCH64
    return ssimulacra2_downsample_2x2_neon;
#endif
    return NULL;
}
static ssim_fn_t pick_ssim(void)
{
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags_x86();
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        return ssimulacra2_ssim_map_avx512;
#endif
    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        return ssimulacra2_ssim_map_avx2;
#endif
#if ARCH_AARCH64
    return ssimulacra2_ssim_map_neon;
#endif
    return NULL;
}
static edge_fn_t pick_edge(void)
{
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags_x86();
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        return ssimulacra2_edge_diff_map_avx512;
#endif
    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        return ssimulacra2_edge_diff_map_avx2;
#endif
#if ARCH_AARCH64
    return ssimulacra2_edge_diff_map_neon;
#endif
    return NULL;
}

static char *test_multiply(void)
{
    mul3_fn_t fn = pick_mul3();
    if (!fn)
        return NULL;
    float *a = malloc(RGB_SZ * sizeof(float));
    float *b = malloc(RGB_SZ * sizeof(float));
    float *out_ref = malloc(RGB_SZ * sizeof(float));
    float *out_simd = malloc(RGB_SZ * sizeof(float));
    fill_random(a, RGB_SZ, -2.0f, 2.0f, 0xdeadbeefu);
    fill_random(b, RGB_SZ, -2.0f, 2.0f, 0x12345678u);
    ref_multiply_3plane(a, b, out_ref, W, H);
    fn(a, b, out_simd, W, H);
    /* memcmp on float buffers is deliberate: ADR-0161's SIMD contract
     * requires byte-for-byte equality under FLT_EVAL_METHOD == 0. */
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison,cert-exp42-c,cert-flp37-c)
    int match = memcmp(out_ref, out_simd, RGB_SZ * sizeof(float)) == 0;
    free(a);
    free(b);
    free(out_ref);
    free(out_simd);
    mu_assert("multiply_3plane SIMD not bit-identical to scalar", match);
    return NULL;
}

static char *test_xyb(void)
{
    xyb_fn_t fn = pick_xyb();
    if (!fn)
        return NULL;
    float *lin = malloc(RGB_SZ * sizeof(float));
    float *out_ref = malloc(RGB_SZ * sizeof(float));
    float *out_simd = malloc(RGB_SZ * sizeof(float));
    fill_random(lin, RGB_SZ, 0.0f, 1.0f, 0xabcdef01u);
    ref_linear_rgb_to_xyb(lin, out_ref, W, H);
    fn(lin, out_simd, W, H);
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison,cert-exp42-c,cert-flp37-c) ADR-0161 byte-exact
    int match = memcmp(out_ref, out_simd, RGB_SZ * sizeof(float)) == 0;
    free(lin);
    free(out_ref);
    free(out_simd);
    mu_assert("linear_rgb_to_xyb SIMD not bit-identical to scalar", match);
    return NULL;
}

static char *test_downsample(void)
{
    down_fn_t fn = pick_down();
    if (!fn)
        return NULL;
    float *in = malloc(RGB_SZ * sizeof(float));
    const size_t out_plane = (size_t)((W + 1) / 2) * (size_t)((H + 1) / 2);
    float *out_ref = malloc(3 * out_plane * sizeof(float));
    float *out_simd = malloc(3 * out_plane * sizeof(float));
    fill_random(in, RGB_SZ, -1.0f, 1.0f, 0x5a5a5a5au);
    unsigned ow_r = 0;
    unsigned oh_r = 0;
    unsigned ow_s = 0;
    unsigned oh_s = 0;
    ref_downsample_2x2(in, W, H, out_ref, &ow_r, &oh_r);
    fn(in, W, H, out_simd, &ow_s, &oh_s);
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison,cert-exp42-c,cert-flp37-c) ADR-0161 byte-exact
    const int mem_eq = memcmp(out_ref, out_simd, 3 * out_plane * sizeof(float)) == 0;
    int match = (ow_r == ow_s) && (oh_r == oh_s) && mem_eq;
    free(in);
    free(out_ref);
    free(out_simd);
    mu_assert("downsample_2x2 SIMD not bit-identical to scalar", match);
    return NULL;
}

static char *test_ssim(void)
{
    ssim_fn_t fn = pick_ssim();
    if (!fn)
        return NULL;
    float *m1 = malloc(RGB_SZ * sizeof(float));
    float *m2 = malloc(RGB_SZ * sizeof(float));
    float *s11 = malloc(RGB_SZ * sizeof(float));
    float *s22 = malloc(RGB_SZ * sizeof(float));
    float *s12 = malloc(RGB_SZ * sizeof(float));
    fill_random(m1, RGB_SZ, -0.5f, 0.5f, 0x11111111u);
    fill_random(m2, RGB_SZ, -0.5f, 0.5f, 0x22222222u);
    fill_random(s11, RGB_SZ, 0.0f, 1.0f, 0x33333333u);
    fill_random(s22, RGB_SZ, 0.0f, 1.0f, 0x44444444u);
    fill_random(s12, RGB_SZ, 0.0f, 1.0f, 0x55555555u);
    double ref[6] = {0};
    double simd[6] = {0};
    ref_ssim_map(m1, m2, s11, s22, s12, W, H, ref);
    fn(m1, m2, s11, s22, s12, W, H, simd);
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison,cert-exp42-c,cert-flp37-c) ADR-0161 byte-exact
    int match = memcmp(ref, simd, sizeof(ref)) == 0;
    free(m1);
    free(m2);
    free(s11);
    free(s22);
    free(s12);
    mu_assert("ssim_map SIMD averages not bit-identical to scalar", match);
    return NULL;
}

static char *test_edge(void)
{
    edge_fn_t fn = pick_edge();
    if (!fn)
        return NULL;
    float *img1 = malloc(RGB_SZ * sizeof(float));
    float *mu1 = malloc(RGB_SZ * sizeof(float));
    float *img2 = malloc(RGB_SZ * sizeof(float));
    float *mu2 = malloc(RGB_SZ * sizeof(float));
    fill_random(img1, RGB_SZ, -1.0f, 1.0f, 0x66666666u);
    fill_random(mu1, RGB_SZ, -1.0f, 1.0f, 0x77777777u);
    fill_random(img2, RGB_SZ, -1.0f, 1.0f, 0x88888888u);
    fill_random(mu2, RGB_SZ, -1.0f, 1.0f, 0x99999999u);
    double ref[12] = {0};
    double simd[12] = {0};
    ref_edge_diff_map(img1, mu1, img2, mu2, W, H, ref);
    fn(img1, mu1, img2, mu2, W, H, simd);
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison,cert-exp42-c,cert-flp37-c) ADR-0161 byte-exact
    int match = memcmp(ref, simd, sizeof(ref)) == 0;
    free(img1);
    free(mu1);
    free(img2);
    free(mu2);
    mu_assert("edge_diff_map SIMD averages not bit-identical to scalar", match);
    return NULL;
}

char *run_tests(void)
{
#if ARCH_X86 || ARCH_AARCH64
    mu_run_test(test_multiply);
    mu_run_test(test_xyb);
    mu_run_test(test_downsample);
    mu_run_test(test_ssim);
    mu_run_test(test_edge);
#else
    (void)fprintf(stderr, "skipping: arch without ssimulacra2 SIMD\n");
#endif
    return NULL;
}
