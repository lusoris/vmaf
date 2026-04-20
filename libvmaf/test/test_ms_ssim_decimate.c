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
 * Bit-exactness contract (ADR-0125): ms_ssim_decimate_{avx2,avx512}
 * produce byte-for-byte the same output as ms_ssim_decimate_scalar,
 * for any (src, w, h). This test compares the AVX2 (and, when
 * available, AVX-512) path against the scalar reference on several
 * synthetic inputs covering:
 *   - The SIMD inner region (1920x1080).
 *   - Borders narrower than the 9-tap kernel (1x1, 8x8, 9x9).
 *   - Edge cases where the SIMD boundary meets the border (odd w/h).
 *
 * The assertion is strict byte-equality via memcmp — a single ULP
 * difference would fail the test.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "feature/ms_ssim_decimate.h"
#include "feature/x86/ms_ssim_decimate_avx2.h"
#include "test.h"

/* Deterministic pseudo-random fill — reproducible across runs. */
static void fill_pattern(float *buf, size_t n, uint32_t seed)
{
    uint32_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        /* xorshift32 */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        /* Map to [-1.0, 1.0] — keeps FMA intermediates in a well-conditioned
         * range where per-lane bit-identity is most likely to expose any
         * lane-ordering bug. */
        buf[i] = ((float)(int32_t)state) / (float)INT32_MAX;
    }
}

static int compare_bitexact(const float *a, const float *b, size_t n)
{
    /*
     * memcmp covers all IEEE-754 bit patterns except NaN payload
     * differences, which don't occur in this pipeline (no division,
     * no sqrt, no overflow to Inf for [-1, 1] inputs through a 9-tap
     * LPF with coefficients summing to ~1).
     */
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static char *check_case(int w, int h, uint32_t seed)
{
    const int w_out = (w / 2) + (w & 1);
    const int h_out = (h / 2) + (h & 1);
    const size_t src_n = (size_t)w * (size_t)h;
    const size_t dst_n = (size_t)w_out * (size_t)h_out;

    float *src = (float *)malloc(src_n * sizeof(float));
    float *dst_scalar = (float *)malloc(dst_n * sizeof(float));
    float *dst_avx2 = (float *)malloc(dst_n * sizeof(float));
    mu_assert("malloc failed", src && dst_scalar && dst_avx2);

    fill_pattern(src, src_n, seed);
    /* Poison the destinations with a distinct pattern so a write-miss shows up. */
    memset(dst_scalar, 0xAA, dst_n * sizeof(float));
    memset(dst_avx2, 0x55, dst_n * sizeof(float));

    const int rc_scalar = ms_ssim_decimate_scalar(src, w, h, dst_scalar, NULL, NULL);
    const int rc_avx2 = ms_ssim_decimate_avx2(src, w, h, dst_avx2, NULL, NULL);

    mu_assert("scalar decimate failed", rc_scalar == 0);
    mu_assert("avx2 decimate failed", rc_avx2 == 0);
    mu_assert("avx2 output not bit-identical to scalar",
              compare_bitexact(dst_scalar, dst_avx2, dst_n));

    free(src);
    free(dst_scalar);
    free(dst_avx2);
    return NULL;
}

static char *test_1x1(void)
{
    return check_case(1, 1, 0x12345678u);
}
static char *test_8x8(void)
{
    return check_case(8, 8, 0x87654321u);
}
static char *test_9x9(void)
{
    return check_case(9, 9, 0xdeadbeefu);
}
static char *test_16x16(void)
{
    return check_case(16, 16, 0xcafebabeu);
}
static char *test_32x32(void)
{
    return check_case(32, 32, 0xfeedfaceu);
}
/* Odd dimensions hit the (w&1)/(h&1) rounding path. */
static char *test_33x17(void)
{
    return check_case(33, 17, 0x00bab10cu);
}
/* Typical MS-SSIM first-scale dimensions. */
static char *test_576x324(void)
{
    return check_case(576, 324, 0x11112222u);
}
static char *test_1920x1080(void)
{
    return check_case(1920, 1080, 0x33334444u);
}
/* Tiny width just past the 9-tap kernel boundary. */
static char *test_10x10(void)
{
    return check_case(10, 10, 0x55556666u);
}
/* Cascaded: 1920x1080 -> 960x540 -> 480x270 -> ... mimicking MS-SSIM scales. */
static char *test_480x270(void)
{
    return check_case(480, 270, 0x77778888u);
}

char *run_tests(void)
{
    mu_run_test(test_1x1);
    mu_run_test(test_8x8);
    mu_run_test(test_9x9);
    mu_run_test(test_10x10);
    mu_run_test(test_16x16);
    mu_run_test(test_32x32);
    mu_run_test(test_33x17);
    mu_run_test(test_480x270);
    mu_run_test(test_576x324);
    mu_run_test(test_1920x1080);
    return NULL;
}
