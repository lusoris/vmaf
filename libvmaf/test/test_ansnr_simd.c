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
 * Numerical-parity test for the ansnr MSE SIMD line kernels
 * (ADR-0245, coverage gap identified in simd-coverage-audit-2026-05-15).
 *
 * Contract: tolerance-bounded, not bit-exact.
 *
 * The per-ISA kernels accumulate each 8- or 16-lane batch into a
 * double-precision running sum, then narrow the per-row result back
 * to float before adding to the inter-row float accumulator:
 *
 *   per-line:   sig_result (double) = sum_j (ref[j] * ref[j])
 *               *sig_accum (float) += (float)sig_result;   // narrow
 *
 * The scalar reference accumulates entirely in float. Across H rows
 * the float accumulator drifts from scalar by the precision lost at
 * each `(float)` narrowing, giving residuals that are bounded but not
 * zero. The tolerance below (1e-5 relative) is ~5 orders of magnitude
 * tighter than the snapshot gate's `places=4` threshold (5e-5 abs on
 * normalised scores) while leaving a comfortable margin for the
 * per-row narrowing residual.
 *
 * Pixel-value range: [0, 255] float (8-bit luma after picture_copy).
 * Worst-case accumulator for 1920 pixels: 255^2 * 1920 * 1080 ~= 1.35e11,
 * well inside float's ~3.4e38 range.
 *
 * Test cases:
 *   1. 64x64  — smaller than one AVX-512 row body; exercises tail handling.
 *   2. 1920x1080 — production size; exercises the full inner loop.
 *   3. Random seed A (0xdeadbeef).
 *   4. Random seed B (0x12345678) — independent draw.
 *   5. Identity fixture: ref == dis — expected noise_accum == 0 (up to rounding).
 *
 * Boilerplate: xorshift PRNG, portable aligned alloc, CPU-flag gate.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "test.h"
/* clang-format off — test.h has no header guard; must precede
 * simd_bitexact_test.h to avoid a mu_report redefinition clash. */
#include "simd_bitexact_test.h"
/* clang-format on */

#if ARCH_X86
#include "feature/x86/ansnr_avx2.h"
#if HAVE_AVX512
#include "feature/x86/ansnr_avx512.h"
#endif
#endif

#if ARCH_AARCH64
#include "feature/arm64/ansnr_neon.h"
#endif

/* -------------------------------------------------------------------
 * Constants
 * ----------------------------------------------------------------- */

#define ALIGN_BYTES 64

/* Pixel range: 8-bit luma after picture_copy ([0, 255] as float). */
#define ANSNR_FILL_LO 0.0f
#define ANSNR_FILL_HI 255.0f

/* Relative tolerance: 1e-5 of the scalar value.
 * Sources of residual:
 *   - Per-row narrowing: (float)double_sum loses ~7 significant bits
 *     per row relative to a pure-double accumulator.
 *   - With 1080 rows and pixel^2 ~ 65025, the per-row residual
 *     is ~65025 * 1920 * eps_float ~ 1.5e6 * 1.2e-7 ~ 0.18 per row;
 *     summed over 1080 rows that is ~200 absolute, against a total
 *     signal of ~1.35e11 -> relative ~1.5e-9. A 1e-5 gate gives
 *     >4 orders of margin.
 *
 * Still ~500x tighter than the production snapshot gate (places=4). */
#define ANSNR_REL_TOL 1e-5

/* -------------------------------------------------------------------
 * Scalar per-line reference (mirrors the scalar branch of
 * ansnr_tools.c:ansnr_mse_s without CPU-flag dispatch).
 * ----------------------------------------------------------------- */

static void ansnr_mse_line_scalar(const float *ref, const float *dis, float *sig_accum,
                                  float *noise_accum, int w)
{
    float sig_inner = 0.0f;
    float noise_inner = 0.0f;
    for (int j = 0; j < w; ++j) {
        const float r = ref[j];
        const float d = dis[j];
        const float diff = r - d;
        sig_inner += r * r;
        noise_inner += diff * diff;
    }
    *sig_accum += sig_inner;
    *noise_accum += noise_inner;
}

/* -------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------- */

/* Allocate and fill two aligned float planes (ref and dis) with
 * reproducible random values in [ANSNR_FILL_LO, ANSNR_FILL_HI).
 * Returns 0 on success, -1 on allocation failure. */
static int alloc_planes(int w, int h, uint32_t seed_ref, uint32_t seed_dis, float **ref_out,
                        float **dis_out, int *stride_px_out)
{
    const int stride_px = (w + 15) & ~15; /* 16-float alignment */
    const size_t bytes = (size_t)stride_px * (size_t)h * sizeof(float);

    float *ref_buf = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    float *dis_buf = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);

    if (!ref_buf || !dis_buf) {
        simd_test_aligned_free(ref_buf);
        simd_test_aligned_free(dis_buf);
        return -1;
    }

    simd_test_fill_random_f32(ref_buf, (size_t)stride_px * (size_t)h, ANSNR_FILL_LO, ANSNR_FILL_HI,
                              seed_ref);
    simd_test_fill_random_f32(dis_buf, (size_t)stride_px * (size_t)h, ANSNR_FILL_LO, ANSNR_FILL_HI,
                              seed_dis);

    *ref_out = ref_buf;
    *dis_out = dis_buf;
    *stride_px_out = stride_px;
    return 0;
}

/* Run scalar reference over all H rows; return accumulated sig and noise. */
static void run_scalar(const float *ref, const float *dis, int w, int h, int stride_px,
                       float *sig_out, float *noise_out)
{
    float sig_accum = 0.0f;
    float noise_accum = 0.0f;
    for (int i = 0; i < h; ++i) {
        ansnr_mse_line_scalar(ref + (size_t)i * (size_t)stride_px,
                              dis + (size_t)i * (size_t)stride_px, &sig_accum, &noise_accum, w);
    }
    *sig_out = sig_accum;
    *noise_out = noise_accum;
}

/* -------------------------------------------------------------------
 * AVX2 tests
 * ----------------------------------------------------------------- */

#if ARCH_X86

static char *check_avx2(uint32_t seed_ref, uint32_t seed_dis, int w, int h)
{
    float *ref = NULL;
    float *dis = NULL;
    int stride_px = 0;

    if (alloc_planes(w, h, seed_ref, seed_dis, &ref, &dis, &stride_px) != 0) {
        return "aligned_malloc failed (AVX2 check)";
    }

    float sig_scalar = 0.0f;
    float noise_scalar = 0.0f;
    run_scalar(ref, dis, w, h, stride_px, &sig_scalar, &noise_scalar);

    float sig_avx2 = 0.0f;
    float noise_avx2 = 0.0f;
    for (int i = 0; i < h; ++i) {
        ansnr_mse_line_avx2(ref + (size_t)i * (size_t)stride_px,
                            dis + (size_t)i * (size_t)stride_px, &sig_avx2, &noise_avx2, w);
    }

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    SIMD_BITEXACT_ASSERT_RELATIVE(sig_scalar, sig_avx2, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx2 sig outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE(noise_scalar, noise_avx2, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx2 noise outside relative tolerance");
    return NULL;
}

static char *test_avx2_64x64_seed_a(void)
{
    return check_avx2(0xdeadbeefu, 0xcafe1234u, 64, 64);
}

static char *test_avx2_1920x1080_seed_a(void)
{
    return check_avx2(0xdeadbeefu, 0xcafe1234u, 1920, 1080);
}

static char *test_avx2_1920x1080_seed_b(void)
{
    return check_avx2(0x12345678u, 0xabcdef01u, 1920, 1080);
}

static char *test_avx2_identity(void)
{
    /* ref == dis: noise_accum should be 0 (or float-close to 0). */
    int stride_px = 0;
    float *ref = NULL;
    float *dis = NULL;

    if (alloc_planes(64, 64, 0xf00df00du, 0xf00df00du, &ref, &dis, &stride_px) != 0) {
        return "aligned_malloc failed (AVX2 identity)";
    }
    /* Make dis identical to ref. */
    (void)memcpy(dis, ref, (size_t)stride_px * 64u * sizeof(float));

    float sig_avx2 = 0.0f;
    float noise_avx2 = 0.0f;
    for (int i = 0; i < 64; ++i) {
        ansnr_mse_line_avx2(ref + (size_t)i * (size_t)stride_px,
                            dis + (size_t)i * (size_t)stride_px, &sig_avx2, &noise_avx2, 64);
    }

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    /* noise must be exactly 0 when ref == dis (diff is always 0). */
    if (noise_avx2 != 0.0f) {
        (void)fprintf(stderr, "  ansnr_mse_line_avx2 identity: noise_avx2=%.17g (expected 0)\n",
                      (double)noise_avx2);
        return "ansnr_mse_line_avx2 identity noise != 0";
    }
    return NULL;
}

#if HAVE_AVX512

static char *check_avx512(uint32_t seed_ref, uint32_t seed_dis, int w, int h)
{
    float *ref = NULL;
    float *dis = NULL;
    int stride_px = 0;

    if (alloc_planes(w, h, seed_ref, seed_dis, &ref, &dis, &stride_px) != 0) {
        return "aligned_malloc failed (AVX-512 check)";
    }

    float sig_scalar = 0.0f;
    float noise_scalar = 0.0f;
    run_scalar(ref, dis, w, h, stride_px, &sig_scalar, &noise_scalar);

    float sig_avx512 = 0.0f;
    float noise_avx512 = 0.0f;
    for (int i = 0; i < h; ++i) {
        ansnr_mse_line_avx512(ref + (size_t)i * (size_t)stride_px,
                              dis + (size_t)i * (size_t)stride_px, &sig_avx512, &noise_avx512, w);
    }

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    SIMD_BITEXACT_ASSERT_RELATIVE(sig_scalar, sig_avx512, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx512 sig outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE(noise_scalar, noise_avx512, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx512 noise outside relative tolerance");
    return NULL;
}

static char *test_avx512_64x64_seed_a(void)
{
    return check_avx512(0xdeadbeefu, 0xcafe1234u, 64, 64);
}

static char *test_avx512_1920x1080_seed_a(void)
{
    return check_avx512(0xdeadbeefu, 0xcafe1234u, 1920, 1080);
}

static char *test_avx512_1920x1080_seed_b(void)
{
    return check_avx512(0x12345678u, 0xabcdef01u, 1920, 1080);
}

static char *test_avx512_identity(void)
{
    int stride_px = 0;
    float *ref = NULL;
    float *dis = NULL;

    if (alloc_planes(64, 64, 0xf00df00du, 0xf00df00du, &ref, &dis, &stride_px) != 0) {
        return "aligned_malloc failed (AVX-512 identity)";
    }
    (void)memcpy(dis, ref, (size_t)stride_px * 64u * sizeof(float));

    float sig_avx512 = 0.0f;
    float noise_avx512 = 0.0f;
    for (int i = 0; i < 64; ++i) {
        ansnr_mse_line_avx512(ref + (size_t)i * (size_t)stride_px,
                              dis + (size_t)i * (size_t)stride_px, &sig_avx512, &noise_avx512, 64);
    }

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    if (noise_avx512 != 0.0f) {
        (void)fprintf(stderr, "  ansnr_mse_line_avx512 identity: noise_avx512=%.17g (expected 0)\n",
                      (double)noise_avx512);
        return "ansnr_mse_line_avx512 identity noise != 0";
    }
    return NULL;
}

#endif /* HAVE_AVX512 */

#endif /* ARCH_X86 */

/* -------------------------------------------------------------------
 * NEON tests (aarch64)
 * ----------------------------------------------------------------- */

#if ARCH_AARCH64

static char *check_neon(uint32_t seed_ref, uint32_t seed_dis, int w, int h)
{
    float *ref = NULL;
    float *dis = NULL;
    int stride_px = 0;

    if (alloc_planes(w, h, seed_ref, seed_dis, &ref, &dis, &stride_px) != 0) {
        return "aligned_malloc failed (NEON check)";
    }

    float sig_scalar = 0.0f;
    float noise_scalar = 0.0f;
    run_scalar(ref, dis, w, h, stride_px, &sig_scalar, &noise_scalar);

    float sig_neon = 0.0f;
    float noise_neon = 0.0f;
    for (int i = 0; i < h; ++i) {
        ansnr_mse_line_neon(ref + (size_t)i * (size_t)stride_px,
                            dis + (size_t)i * (size_t)stride_px, &sig_neon, &noise_neon, w);
    }

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    SIMD_BITEXACT_ASSERT_RELATIVE(sig_scalar, sig_neon, ANSNR_REL_TOL,
                                  "ansnr_mse_line_neon sig outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE(noise_scalar, noise_neon, ANSNR_REL_TOL,
                                  "ansnr_mse_line_neon noise outside relative tolerance");
    return NULL;
}

static char *test_neon_64x64_seed_a(void)
{
    return check_neon(0xdeadbeefu, 0xcafe1234u, 64, 64);
}

static char *test_neon_1920x1080_seed_a(void)
{
    return check_neon(0xdeadbeefu, 0xcafe1234u, 1920, 1080);
}

static char *test_neon_1920x1080_seed_b(void)
{
    return check_neon(0x12345678u, 0xabcdef01u, 1920, 1080);
}

static char *test_neon_identity(void)
{
    int stride_px = 0;
    float *ref = NULL;
    float *dis = NULL;

    if (alloc_planes(64, 64, 0xf00df00du, 0xf00df00du, &ref, &dis, &stride_px) != 0) {
        return "aligned_malloc failed (NEON identity)";
    }
    (void)memcpy(dis, ref, (size_t)stride_px * 64u * sizeof(float));

    float sig_neon = 0.0f;
    float noise_neon = 0.0f;
    for (int i = 0; i < 64; ++i) {
        ansnr_mse_line_neon(ref + (size_t)i * (size_t)stride_px,
                            dis + (size_t)i * (size_t)stride_px, &sig_neon, &noise_neon, 64);
    }

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    if (noise_neon != 0.0f) {
        (void)fprintf(stderr, "  ansnr_mse_line_neon identity: noise_neon=%.17g (expected 0)\n",
                      (double)noise_neon);
        return "ansnr_mse_line_neon identity noise != 0";
    }
    return NULL;
}

#endif /* ARCH_AARCH64 */

/* -------------------------------------------------------------------
 * Test runner helpers.
 *
 * Separate helpers per ISA keep run_tests() below the
 * readability-function-size branch threshold (15). Each helper owns
 * exactly the 4 mu_run_test calls for one ISA.
 * ----------------------------------------------------------------- */

#if ARCH_X86

static char *run_avx2_tests(void)
{
    mu_run_test(test_avx2_64x64_seed_a);
    mu_run_test(test_avx2_1920x1080_seed_a);
    mu_run_test(test_avx2_1920x1080_seed_b);
    mu_run_test(test_avx2_identity);
    return NULL;
}

#if HAVE_AVX512

static char *run_avx512_tests(void)
{
    mu_run_test(test_avx512_64x64_seed_a);
    mu_run_test(test_avx512_1920x1080_seed_a);
    mu_run_test(test_avx512_1920x1080_seed_b);
    mu_run_test(test_avx512_identity);
    return NULL;
}

#endif /* HAVE_AVX512 */

#elif ARCH_AARCH64

static char *run_neon_tests(void)
{
    mu_run_test(test_neon_64x64_seed_a);
    mu_run_test(test_neon_1920x1080_seed_a);
    mu_run_test(test_neon_1920x1080_seed_b);
    mu_run_test(test_neon_identity);
    return NULL;
}

#endif /* ARCH_X86 / ARCH_AARCH64 */

/* -------------------------------------------------------------------
 * Test runner
 * ----------------------------------------------------------------- */

char *run_tests(void)
{
#if ARCH_X86
    if (!simd_test_have_avx2()) {
        return NULL;
    }
    {
        char *r = run_avx2_tests();
        if (r) {
            return r;
        }
    }
#if HAVE_AVX512
    {
        char *r = run_avx512_tests();
        if (r) {
            return r;
        }
    }
#endif /* HAVE_AVX512 */
#elif ARCH_AARCH64
    {
        char *r = run_neon_tests();
        if (r) {
            return r;
        }
    }
#else
    (void)fprintf(stderr, "skipping: arch lacks ansnr SIMD\n");
#endif
    return NULL;
}
