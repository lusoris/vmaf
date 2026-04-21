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
 * Bit-exactness contract (ADR-0138, ADR-0140): the SIMD `iqa_convolve_*`
 * variants — AVX2, AVX-512, and NEON — produce byte-for-byte the same
 * output as the scalar reference `_iqa_convolve` under IQA_CONVOLVE_1D,
 * for any (src, w, h, kernel) in the SSIM/MS-SSIM supported space:
 *   - 11-tap Gaussian (odd kernel, kw_even == 0)
 *   - 8-tap box       (even kernel, kw_even == 1)
 *   - image dimensions where `w >= kw` and `h >= kh`
 *
 * Coverage:
 *   - SIMD inner region (1920x1080, 576x324).
 *   - Tail sizes that hit the masked 4-lane AVX2 tail (1..3 cols) and
 *     the masked 8-lane AVX-512 tail (1..7 cols).
 *   - Odd dimensions (33x17, 61x41) to exercise off-stride tails.
 *   - Minimum-size cases equal to the kernel footprint (11x11, 8x8).
 *
 * The assertion is strict byte-equality via memcmp.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "feature/iqa/convolve.h"
#if ARCH_X86
#include "feature/x86/convolve_avx2.h"
#if HAVE_AVX512
#include "feature/x86/convolve_avx512.h"
#endif
#endif
#if ARCH_AARCH64
#include "cpu.h"
#include "feature/arm64/convolve_neon.h"
#endif
#include "test.h"
#if ARCH_X86
#include "x86/cpu.h"
#endif

#if ARCH_X86
static int g_has_avx2 = 0;
static int g_has_avx512 = 0;
#endif
#if ARCH_AARCH64
static int g_has_neon = 0;
#endif

/* 11-tap Gaussian — matches g_gaussian_window_{h,v} in ssim_tools.h. */
static const float kernel_gauss11[11] = {0.001028f, 0.007599f, 0.036001f, 0.109361f,
                                         0.213006f, 0.266012f, 0.213006f, 0.109361f,
                                         0.036001f, 0.007599f, 0.001028f};

/* 8-tap box — matches g_square_window_{h,v} in ssim_tools.h. */
static const float kernel_box8[8] = {0.125f, 0.125f, 0.125f, 0.125f,
                                     0.125f, 0.125f, 0.125f, 0.125f};

/* Deterministic pseudo-random fill — reproducible across runs. */
static void fill_pattern(float *buf, size_t n, uint32_t seed)
{
    uint32_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        buf[i] = ((float)(int32_t)state) / (float)INT32_MAX;
    }
}

static int compare_bitexact(const float *a, const float *b, size_t n)
{
    return memcmp(a, b, n * sizeof(float)) == 0;
}

#if ARCH_X86 || ARCH_AARCH64
typedef void (*convolve_simd_fn)(float *img, int w, int h, const float *kernel_h,
                                 const float *kernel_v, int kw, int kh, int normalized,
                                 float *workspace, float *result, int *rw, int *rh);

static char *check_simd_variant(const float *src, int w, int h, const float *kernel_h,
                                const float *kernel_v, int kw, int kh, const float *dst_scalar,
                                size_t dst_n, convolve_simd_fn fn, int poison, char *fail_cmp)
{
    float *src_copy = (float *)malloc((size_t)w * (size_t)h * sizeof(float));
    float *dst = (float *)malloc(dst_n * sizeof(float));
    float *workspace = (float *)malloc((size_t)w * (size_t)h * sizeof(float));
    mu_assert("simd malloc failed", src_copy && dst && workspace);
    memcpy(src_copy, src, (size_t)w * (size_t)h * sizeof(float));
    memset(dst, poison, dst_n * sizeof(float));

    int rw = 0;
    int rh = 0;
    fn(src_copy, w, h, kernel_h, kernel_v, kw, kh, /*normalized=*/1, workspace, dst, &rw, &rh);
    mu_assert("simd rw mismatch", rw == w - kw + 1);
    mu_assert("simd rh mismatch", rh == h - kh + 1);
    mu_assert(fail_cmp, compare_bitexact(dst_scalar, dst, dst_n));

    free(src_copy);
    free(dst);
    free(workspace);
    return NULL;
}
#endif /* ARCH_X86 || ARCH_AARCH64 */

static char *check_case(int w, int h, int kw, const float *kernel_h, const float *kernel_v,
                        uint32_t seed)
{
    const int dst_w = w - kw + 1;
    const int dst_h = h - kw + 1;
    const size_t src_n = (size_t)w * (size_t)h;
    const size_t dst_n = (size_t)dst_w * (size_t)dst_h;

    float *src = (float *)malloc(src_n * sizeof(float));
    float *dst_scalar = (float *)malloc(dst_n * sizeof(float));
    mu_assert("malloc failed", src && dst_scalar);
    fill_pattern(src, src_n, seed);
    memset(dst_scalar, 0xAA, dst_n * sizeof(float));

    /* Build the _kernel descriptor for the scalar reference. */
    struct _kernel k;
    k.kernel = NULL;
    k.kernel_h = (float *)kernel_h;
    k.kernel_v = (float *)kernel_v;
    k.w = kw;
    k.h = kw;
    k.normalized = 1;
    k.bnd_opt = NULL;
    k.bnd_const = 0.0f;

    /* _iqa_convolve mutates `src` in-place when result is non-NULL only
     * for the cache; the result goes into dst_scalar. Make a copy to be
     * safe against any mutation. */
    float *src_scalar_copy = (float *)malloc(src_n * sizeof(float));
    mu_assert("malloc failed", src_scalar_copy != NULL);
    memcpy(src_scalar_copy, src, src_n * sizeof(float));
    _iqa_convolve(src_scalar_copy, w, h, &k, dst_scalar, NULL, NULL);
    free(src_scalar_copy);

    char *msg = NULL;
#if ARCH_X86
    if (g_has_avx2) {
        msg = check_simd_variant(src, w, h, kernel_h, kernel_v, kw, kw, dst_scalar, dst_n,
                                 iqa_convolve_avx2, 0x55,
                                 "avx2 convolve output not bit-identical to scalar");
        if (msg)
            goto done;
    }
#if HAVE_AVX512
    if (g_has_avx512) {
        msg = check_simd_variant(src, w, h, kernel_h, kernel_v, kw, kw, dst_scalar, dst_n,
                                 iqa_convolve_avx512, 0x33,
                                 "avx512 convolve output not bit-identical to scalar");
        if (msg)
            goto done;
    }
#endif
#endif
#if ARCH_AARCH64
    if (g_has_neon) {
        msg = check_simd_variant(src, w, h, kernel_h, kernel_v, kw, kw, dst_scalar, dst_n,
                                 iqa_convolve_neon, 0x77,
                                 "neon convolve output not bit-identical to scalar");
        if (msg)
            goto done;
    }
#endif

done:
    free(src);
    free(dst_scalar);
    (void)dst_h;
    return msg;
}

/* Gaussian (11-tap, kw_even=0) cases. */
static char *test_gauss_11x11(void)
{
    return check_case(11, 11, 11, kernel_gauss11, kernel_gauss11, 0x11111111u);
}
static char *test_gauss_12x12(void)
{
    return check_case(12, 12, 11, kernel_gauss11, kernel_gauss11, 0x22222222u);
}
/* AVX-512 tail sizes 1..7: dst_w = w - 10; we want dst_w % 8 ∈ {1..7}. */
static char *test_gauss_19x19(void) /* dst_w=9 -> tail 1 (8-lane) / 1 (4-lane) */
{
    return check_case(19, 19, 11, kernel_gauss11, kernel_gauss11, 0x33333333u);
}
static char *test_gauss_20x20(void) /* dst_w=10 -> tail 2 */
{
    return check_case(20, 20, 11, kernel_gauss11, kernel_gauss11, 0x44444444u);
}
static char *test_gauss_25x25(void) /* dst_w=15 -> tail 7 (8-lane) / 3 (4-lane) */
{
    return check_case(25, 25, 11, kernel_gauss11, kernel_gauss11, 0x55555555u);
}
static char *test_gauss_33x17(void) /* odd, dst_w=23 -> tail 7 / 3 */
{
    return check_case(33, 17, 11, kernel_gauss11, kernel_gauss11, 0x66666666u);
}
static char *test_gauss_61x41(void)
{
    return check_case(61, 41, 11, kernel_gauss11, kernel_gauss11, 0x77777777u);
}
static char *test_gauss_576x324(void)
{
    return check_case(576, 324, 11, kernel_gauss11, kernel_gauss11, 0x11112222u);
}
static char *test_gauss_1920x1080(void)
{
    return check_case(1920, 1080, 11, kernel_gauss11, kernel_gauss11, 0x33334444u);
}

/* Box (8-tap, kw_even=1) cases. */
static char *test_box_8x8(void)
{
    return check_case(8, 8, 8, kernel_box8, kernel_box8, 0x88888888u);
}
static char *test_box_16x16(void)
{
    return check_case(16, 16, 8, kernel_box8, kernel_box8, 0x99999999u);
}
static char *test_box_21x13(void) /* dst_w=14 -> tail 6/2 */
{
    return check_case(21, 13, 8, kernel_box8, kernel_box8, 0xaaaaaaaau);
}
static char *test_box_576x324(void)
{
    return check_case(576, 324, 8, kernel_box8, kernel_box8, 0xbbbbbbbbu);
}

char *run_tests(void)
{
#if ARCH_X86
    const unsigned cpu_flags = vmaf_get_cpu_flags_x86();
    g_has_avx2 = (cpu_flags & VMAF_X86_CPU_FLAG_AVX2) ? 1 : 0;
    g_has_avx512 = (cpu_flags & VMAF_X86_CPU_FLAG_AVX512) ? 1 : 0;
    if (!g_has_avx2 && !g_has_avx512) {
        (void)fprintf(stderr, "skipping: CPU has neither AVX2 nor AVX-512\n");
        return NULL;
    }
#elif ARCH_AARCH64
    const unsigned cpu_flags = vmaf_get_cpu_flags();
    g_has_neon = (cpu_flags & VMAF_ARM_CPU_FLAG_NEON) ? 1 : 0;
    if (!g_has_neon) {
        (void)fprintf(stderr, "skipping: aarch64 CPU lacks NEON\n");
        return NULL;
    }
#else
    (void)fprintf(stderr, "skipping: non-x86, non-aarch64 arch\n");
    return NULL;
#endif

    mu_run_test(test_gauss_11x11);
    mu_run_test(test_gauss_12x12);
    mu_run_test(test_gauss_19x19);
    mu_run_test(test_gauss_20x20);
    mu_run_test(test_gauss_25x25);
    mu_run_test(test_gauss_33x17);
    mu_run_test(test_gauss_61x41);
    mu_run_test(test_gauss_576x324);
    mu_run_test(test_gauss_1920x1080);

    mu_run_test(test_box_8x8);
    mu_run_test(test_box_16x16);
    mu_run_test(test_box_21x13);
    mu_run_test(test_box_576x324);
    return NULL;
}
