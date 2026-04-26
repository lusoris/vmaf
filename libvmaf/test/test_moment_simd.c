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
 * Numerical-parity contract test for the float_moment SIMD kernels
 * (T7-19, ADR-0179).
 *
 * The contract per `moment_avx2.c` / `moment_neon.c` headers is
 * tolerance-bounded, not bit-exact: the lane-widening + per-row
 * tail divergence yields residuals well inside the snapshot gate's
 * tolerance but not byte-for-byte equal to the scalar reference.
 * The scalar TU's auto-vectorisation (and any compiler-driven
 * precision behaviour) further removes the bit-exact guarantee.
 *
 * Tolerance: 1e-9 absolute on the post-normalisation score (range
 * 0..2^16 for 8-bit pixel squares), which is ~5 orders of magnitude
 * tighter than the snapshot gate's `places=4`.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "test.h"

#include "feature/moment.h"

#if ARCH_X86
#include "feature/x86/moment_avx2.h"
#include "x86/cpu.h"
#endif
#if ARCH_AARCH64
#include "feature/arm64/moment_neon.h"
#endif

#define ALIGN_BYTES 32
#define TEST_W 73 /* not a multiple of 4 or 8 — exercises tail */
#define TEST_H 17

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

static void fill_random(float *buf, size_t n_floats, uint32_t seed)
{
    uint32_t state = seed;
    for (size_t i = 0; i < n_floats; ++i) {
        const uint32_t r = xorshift32(&state) & 0x00ffffffu;
        /* values in [0, 256) — matches the post-`picture_copy`
         * 8-bit pixel range of the float_moment extractor. */
        buf[i] = (float)r * (256.0f / (float)0x01000000);
    }
}

static int alloc_aligned_frame(float **out, size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, ALIGN_BYTES, bytes) != 0) {
        return -1;
    }
    *out = (float *)p;
    return 0;
}

/* Relative tolerance: 1e-7 of the scalar score. Residual sources:
 * per-row tail order, lane-pair cross-add precision, scalar-TU
 * auto-vectorisation. Still ~500× tighter than the production
 * snapshot gate (`places=4` ⇒ |abs| < 5e-5 on a normalised score). */
#define MOMENT_REL_TOL 1e-7

static char *check_within_tolerance(double s_scalar, double s_simd, double t_scalar, double t_simd)
{
    const double rel1 = fabs(s_simd - s_scalar) / (fabs(s_scalar) + 1e-30);
    const double rel2 = fabs(t_simd - t_scalar) / (fabs(t_scalar) + 1e-30);
    mu_assert("compute_1st_moment SIMD outside relative tolerance", rel1 < MOMENT_REL_TOL);
    mu_assert("compute_2nd_moment SIMD outside relative tolerance", rel2 < MOMENT_REL_TOL);
    return NULL;
}

#if ARCH_X86

static int g_has_avx2;

static int detect_avx2(void)
{
    const unsigned cpu_flags = vmaf_get_cpu_flags_x86();
    g_has_avx2 = (cpu_flags & VMAF_X86_CPU_FLAG_AVX2) ? 1 : 0;
    if (!g_has_avx2) {
        (void)fprintf(stderr, "skipping: CPU lacks AVX2\n");
    }
    return g_has_avx2;
}

static char *check_avx2(uint32_t seed, int w, int h)
{
    const int stride_floats = (w + 7) & ~7;
    const size_t bytes = (size_t)stride_floats * (size_t)h * sizeof(float);
    const int stride_bytes = stride_floats * (int)sizeof(float);

    float *buf = NULL;
    if (alloc_aligned_frame(&buf, bytes) != 0) {
        return "posix_memalign failed";
    }
    fill_random(buf, (size_t)stride_floats * (size_t)h, seed);

    double s_scalar = 0.0;
    double s_avx2 = 0.0;
    (void)compute_1st_moment(buf, w, h, stride_bytes, &s_scalar);
    (void)compute_1st_moment_avx2(buf, w, h, stride_bytes, &s_avx2);

    double t_scalar = 0.0;
    double t_avx2 = 0.0;
    (void)compute_2nd_moment(buf, w, h, stride_bytes, &t_scalar);
    (void)compute_2nd_moment_avx2(buf, w, h, stride_bytes, &t_avx2);

    free(buf);
    return check_within_tolerance(s_scalar, s_avx2, t_scalar, t_avx2);
}

static char *test_avx2_seed_a(void)
{
    return check_avx2(0xdeadbeefu, TEST_W, TEST_H);
}
static char *test_avx2_seed_b(void)
{
    return check_avx2(0x12345678u, TEST_W, TEST_H);
}
static char *test_avx2_aligned_w(void)
{
    return check_avx2(0xabcdef01u, 64, 16);
}
static char *test_avx2_tiny(void)
{
    return check_avx2(0xfeedface, 9, 1);
}

#endif /* ARCH_X86 */

#if ARCH_AARCH64

static char *check_neon(uint32_t seed, int w, int h)
{
    const int stride_floats = (w + 3) & ~3;
    const size_t bytes = (size_t)stride_floats * (size_t)h * sizeof(float);
    const int stride_bytes = stride_floats * (int)sizeof(float);

    float *buf = NULL;
    if (alloc_aligned_frame(&buf, bytes) != 0) {
        return "posix_memalign failed";
    }
    fill_random(buf, (size_t)stride_floats * (size_t)h, seed);

    double s_scalar = 0.0;
    double s_neon = 0.0;
    (void)compute_1st_moment(buf, w, h, stride_bytes, &s_scalar);
    (void)compute_1st_moment_neon(buf, w, h, stride_bytes, &s_neon);

    double t_scalar = 0.0;
    double t_neon = 0.0;
    (void)compute_2nd_moment(buf, w, h, stride_bytes, &t_scalar);
    (void)compute_2nd_moment_neon(buf, w, h, stride_bytes, &t_neon);

    free(buf);
    return check_within_tolerance(s_scalar, s_neon, t_scalar, t_neon);
}

static char *test_neon_seed_a(void)
{
    return check_neon(0xdeadbeefu, TEST_W, TEST_H);
}
static char *test_neon_seed_b(void)
{
    return check_neon(0x12345678u, TEST_W, TEST_H);
}
static char *test_neon_aligned_w(void)
{
    return check_neon(0xabcdef01u, 64, 16);
}
static char *test_neon_tiny(void)
{
    return check_neon(0xfeedface, 5, 1);
}

#endif /* ARCH_AARCH64 */

char *run_tests(void)
{
#if ARCH_X86
    if (!detect_avx2()) {
        return NULL;
    }
    mu_run_test(test_avx2_seed_a);
    mu_run_test(test_avx2_seed_b);
    mu_run_test(test_avx2_aligned_w);
    mu_run_test(test_avx2_tiny);
#elif ARCH_AARCH64
    mu_run_test(test_neon_seed_a);
    mu_run_test(test_neon_seed_b);
    mu_run_test(test_neon_aligned_w);
    mu_run_test(test_neon_tiny);
#else
    (void)fprintf(stderr, "skipping: arch lacks moment SIMD\n");
#endif
    return NULL;
}
