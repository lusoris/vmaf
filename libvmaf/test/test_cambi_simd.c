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
 * Cambi `calculate_c_values_row` SIMD parity test + microbenchmark.
 *
 * Companion to ADR-0328's status update for the AVX-512 + NEON twins.
 * Asserts byte-identical output for the scalar reference, AVX-2 twin,
 * AVX-512 twin (AVX-512 hosts only), and NEON twin (aarch64 only).
 *
 * The microbenchmark prints per-variant ms / row and the AVX-512-vs-AVX-2
 * speedup at the end of the test run. The harness is informational —
 * it does not assert on the timing (microbench numbers are noisy on
 * shared CI hosts), but on a typical AVX-512 host we expect the AVX-512
 * row stage at 1.5–2x the AVX-2 row stage, per ADR-0328 update.
 *
 * Bit-exactness invariants: ADR-0138 / ADR-0139 (cambi SIMD twins keep
 * IEEE-754 rounding aligned across variants by reproducing the scalar
 * `c_value_pixel` arithmetic per lane, in identical order, with
 * identical operand widths).
 */

#include "test.h"
#include "simd_bitexact_test.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpu.h"
#include "feature/cambi_reciprocal_lut.h"
#if ARCH_X86
#include "feature/x86/cambi_avx2.h"
#include "feature/x86/cambi_avx512.h"
#endif
#if ARCH_AARCH64
#include "feature/arm64/cambi_neon.h"
#endif

/* Scalar reference — duplicated here (rather than including cambi.c) because
 * cambi.c brings the entire feature-extractor surface and conflicts with
 * symbols already in libvmaf_feature_static_lib. The arithmetic mirrors
 * `calculate_c_values_row` and `c_value_pixel` from cambi.c verbatim. */
static void cambi_calc_c_values_row_scalar(float *c_values, const uint16_t *histograms,
                                           const uint16_t *image, const uint16_t *mask, int row,
                                           int width, ptrdiff_t stride, const uint16_t num_diffs,
                                           const uint16_t *tvi_thresholds, uint16_t vlt_luma,
                                           const int *diff_weights, const int *all_diffs,
                                           const float *reciprocal_lut)
{
    int v_lo_signed = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
    uint16_t v_band_size = tvi_thresholds[num_diffs - 1] + 1 - v_band_base;

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row = &mask[row * stride];
    float *c_row = &c_values[row * width];

    for (int col = 0; col < width; col++) {
        if (mask_row[col]) {
            uint16_t value = (uint16_t)(image_row[col] + num_diffs);
            int compact_v_signed = (int)image_row[col] - (int)v_band_base;
            if ((unsigned)compact_v_signed >= v_band_size) {
                c_row[col] = 0.0f;
                continue;
            }
            uint16_t compact_v_sc = (uint16_t)compact_v_signed;
            uint16_t p_0 = histograms[compact_v_sc * width + col];
            float c_v = 0.0f;
            for (int d = 0; d < num_diffs; d++) {
                if ((value <= tvi_thresholds[d]) &&
                    ((value + all_diffs[num_diffs + d + 1]) > vlt_luma)) {
                    int idx1 = compact_v_signed + all_diffs[num_diffs + d + 1];
                    int idx2 = compact_v_signed + all_diffs[num_diffs - d - 1];
                    uint16_t p_1 = histograms[idx1 * width + col];
                    uint16_t p_2 = (idx2 >= 0) ? histograms[idx2 * width + col] : 0;
                    uint16_t p_max = (p_1 > p_2) ? p_1 : p_2;
                    float val =
                        (float)(diff_weights[d] * p_0 * p_max) * reciprocal_lut[p_max + p_0];
                    if (val > c_v)
                        c_v = val;
                }
            }
            c_row[col] = c_v;
        } else {
            c_row[col] = 0.0f;
        }
    }
}

/* Realistic cambi parameter set, lifted from `set_contrast_arrays` /
 * `tvi_for_diff` initialisation in cambi.c for `num_diffs = 4`. */
enum {
    NUM_DIFFS = 4,
    /* `v_band_size` for `vlt_luma=0, tvi_for_diff[3]=559, num_diffs=4` is
     * `559 + 1 - 0 = 560`. We pad to 1032 to match the in-tree
     * `histograms[4 * 1032]` allocations for safety. */
    BAND_SIZE = 560,
};

static const uint16_t TVI_THRESHOLDS[NUM_DIFFS] = {178, 305, 432, 559};
static const int DIFF_WEIGHTS[NUM_DIFFS] = {1, 2, 3, 4};
static const int ALL_DIFFS[2 * NUM_DIFFS + 1] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

/* Build a deterministic histograms / image / mask fixture for the row.
 *
 * The image generator clumps `image[col]` values in a small range so the
 * gather addresses across the SIMD lane group cluster on a few histogram
 * rows. This mirrors natural-image intensity locality (the real cambi
 * c-value driver sees gather access patterns dominated by L1 / L2
 * residency, not main memory). A pure-uniform fixture would punish
 * gather-based SIMD paths far worse than real workloads do. */
static void cambi_simd_fill_fixture(uint16_t *histograms, uint16_t *image, uint16_t *mask,
                                    int width, int band_size, uint32_t seed)
{
    uint32_t state = seed;
    for (int i = 0; i < band_size; i++) {
        for (int j = 0; j < width; j++) {
            histograms[i * width + j] = (uint16_t)(simd_test_xorshift32(&state) & 0x1F);
        }
    }
    /* Generate a slowly-varying intensity field: pick a base value once, drift
     * by ±2 per column. Range stays in [0, BAND_SIZE) so the c-value gate is
     * exercised. */
    int base = (int)(simd_test_xorshift32(&state) % 200) + 100;
    for (int j = 0; j < width; j++) {
        int drift = (int)(simd_test_xorshift32(&state) % 5) - 2;
        base += drift;
        if (base < 0)
            base = 0;
        if (base > 540)
            base = 540;
        image[j] = (uint16_t)base;
    }
    for (int j = 0; j < width; j++) {
        /* Sparsify the mask so the `if (mask_row[col])` early-out path also
         * gets exercised (mask=0 lanes must produce 0). */
        mask[j] = (simd_test_xorshift32(&state) & 0x3) == 0 ? 0 : 1;
    }
}

static int memcmp_floats_bitexact(const float *a, const float *b, size_t n)
{
    return memcmp(a, b, n * sizeof(float));
}

static double now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One-row bit-exactness gate. Returns NULL on success, message on failure. */
static char *cambi_simd_check_row(int width, uint32_t seed)
{
    const int band_size = BAND_SIZE;
    const ptrdiff_t stride = width;
    const uint16_t vlt_luma = 0;

    size_t hist_bytes = (size_t)width * band_size * sizeof(uint16_t);
    uint16_t *histograms = simd_test_aligned_malloc(hist_bytes, 64);
    uint16_t *image = simd_test_aligned_malloc((size_t)width * sizeof(uint16_t), 64);
    uint16_t *mask = simd_test_aligned_malloc((size_t)width * sizeof(uint16_t), 64);
    float *out_scalar = simd_test_aligned_malloc((size_t)width * sizeof(float), 64);
    float *out_avx2 = simd_test_aligned_malloc((size_t)width * sizeof(float), 64);
    float *out_avx512 = simd_test_aligned_malloc((size_t)width * sizeof(float), 64);
    float *out_neon = simd_test_aligned_malloc((size_t)width * sizeof(float), 64);
    mu_assert("alloc fail",
              histograms && image && mask && out_scalar && out_avx2 && out_avx512 && out_neon);

    cambi_simd_fill_fixture(histograms, image, mask, width, band_size, seed);

    cambi_calc_c_values_row_scalar(out_scalar, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                   TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                   reciprocal_lut);

#if ARCH_X86
    unsigned cpu_flags = vmaf_get_cpu_flags();
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2) {
        calculate_c_values_row_avx2(out_avx2, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                    TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                    reciprocal_lut);
        mu_assert("AVX-2 cambi row diverges from scalar (bit-exact gate)",
                  memcmp_floats_bitexact(out_scalar, out_avx2, (size_t)width) == 0);
    }
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512) {
        calculate_c_values_row_avx512(out_avx512, histograms, image, mask, 0, width, stride,
                                      NUM_DIFFS, TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                      reciprocal_lut);
        mu_assert("AVX-512 cambi row diverges from scalar (bit-exact gate)",
                  memcmp_floats_bitexact(out_scalar, out_avx512, (size_t)width) == 0);
    }
#endif
#if ARCH_AARCH64
    /* NEON is always available on aarch64. */
    calculate_c_values_row_neon(out_neon, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS, reciprocal_lut);
    mu_assert("NEON cambi row diverges from scalar (bit-exact gate)",
              memcmp_floats_bitexact(out_scalar, out_neon, (size_t)width) == 0);
#endif

    simd_test_aligned_free(histograms);
    simd_test_aligned_free(image);
    simd_test_aligned_free(mask);
    simd_test_aligned_free(out_scalar);
    simd_test_aligned_free(out_avx2);
    simd_test_aligned_free(out_avx512);
    simd_test_aligned_free(out_neon);
    return NULL;
}

/* Sweep a few row widths to cover both the chunk path and the scalar tail. */
static char *test_cambi_simd_row_bitexact(void)
{
    const int widths[] = {64, 96, 256, 1920};
    const int n = (int)(sizeof(widths) / sizeof(widths[0]));
    for (int i = 0; i < n; i++) {
        char *err = cambi_simd_check_row(widths[i], 0xC0FFEE00u + (uint32_t)widths[i]);
        if (err)
            return err;
    }
    return NULL;
}

/* Microbenchmark — informational, never asserts.
 *
 * The AVX-512 path is expected to deliver 1.5x+ over AVX-2 on this kernel,
 * driven by the wider gathers and the natural mmask predicate handling.
 * NEON's expected speedup over scalar is ~1.5x (no vector gather forces
 * scalar histogram loads, but the surrounding arithmetic vectorises). */
static char *test_cambi_simd_row_microbench(void)
{
    /* Skip the bench when the harness is run with `MESON_SKIP_LARGE_TESTS=1`
     * (CI subset gate). */
    const char *skip = getenv("MESON_SKIP_LARGE_TESTS");
    if (skip && skip[0] == '1')
        return NULL;

    const int width = 1920;
    const int band_size = BAND_SIZE;
    const ptrdiff_t stride = width;
    const uint16_t vlt_luma = 0;
    const int repeats = 2000;

    size_t hist_bytes = (size_t)width * band_size * sizeof(uint16_t);
    uint16_t *histograms = simd_test_aligned_malloc(hist_bytes, 64);
    uint16_t *image = simd_test_aligned_malloc((size_t)width * sizeof(uint16_t), 64);
    uint16_t *mask = simd_test_aligned_malloc((size_t)width * sizeof(uint16_t), 64);
    float *out = simd_test_aligned_malloc((size_t)width * sizeof(float), 64);
    mu_assert("microbench alloc fail", histograms && image && mask && out);

    cambi_simd_fill_fixture(histograms, image, mask, width, band_size, 0xBADBEEFu);

    /* Warm-up. */
    for (int r = 0; r < 50; r++) {
        cambi_calc_c_values_row_scalar(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                       TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                       reciprocal_lut);
    }

    double t0 = now_seconds();
    for (int r = 0; r < repeats; r++) {
        cambi_calc_c_values_row_scalar(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                       TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                       reciprocal_lut);
    }
    double t_scalar = now_seconds() - t0;

    double t_avx2 = 0.0;
    double t_avx512 = 0.0;
    double t_neon = 0.0;
    (void)t_avx2;
    (void)t_avx512;
    (void)t_neon;

#if ARCH_X86
    unsigned cpu_flags = vmaf_get_cpu_flags();
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2) {
        for (int r = 0; r < 50; r++) {
            calculate_c_values_row_avx2(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                        TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                        reciprocal_lut);
        }
        t0 = now_seconds();
        for (int r = 0; r < repeats; r++) {
            calculate_c_values_row_avx2(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                        TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                        reciprocal_lut);
        }
        t_avx2 = now_seconds() - t0;
    }
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512) {
        for (int r = 0; r < 50; r++) {
            calculate_c_values_row_avx512(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                          TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                          reciprocal_lut);
        }
        t0 = now_seconds();
        for (int r = 0; r < repeats; r++) {
            calculate_c_values_row_avx512(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                          TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                          reciprocal_lut);
        }
        t_avx512 = now_seconds() - t0;
    }
#endif
#if ARCH_AARCH64
    for (int r = 0; r < 50; r++) {
        calculate_c_values_row_neon(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                    TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                    reciprocal_lut);
    }
    t0 = now_seconds();
    for (int r = 0; r < repeats; r++) {
        calculate_c_values_row_neon(out, histograms, image, mask, 0, width, stride, NUM_DIFFS,
                                    TVI_THRESHOLDS, vlt_luma, DIFF_WEIGHTS, ALL_DIFFS,
                                    reciprocal_lut);
    }
    t_neon = now_seconds() - t0;
#endif

    fprintf(stderr,
            "\ncambi_calc_c_values_row microbench (width=%d, band=%d, repeats=%d):\n"
            "  scalar : %8.3f ms total, %7.3f us/row\n",
            width, band_size, repeats, t_scalar * 1000.0, t_scalar * 1e6 / (double)repeats);
#if ARCH_X86
    if (t_avx2 > 0.0) {
        fprintf(stderr, "  AVX-2  : %8.3f ms total, %7.3f us/row, speedup vs scalar = %.2fx\n",
                t_avx2 * 1000.0, t_avx2 * 1e6 / (double)repeats, t_scalar / t_avx2);
    }
    if (t_avx512 > 0.0) {
        fprintf(stderr,
                "  AVX-512: %8.3f ms total, %7.3f us/row, speedup vs scalar = %.2fx, "
                "speedup vs AVX-2 = %.2fx\n",
                t_avx512 * 1000.0, t_avx512 * 1e6 / (double)repeats, t_scalar / t_avx512,
                t_avx2 > 0.0 ? t_avx2 / t_avx512 : 0.0);
    }
#endif
#if ARCH_AARCH64
    if (t_neon > 0.0) {
        fprintf(stderr, "  NEON   : %8.3f ms total, %7.3f us/row, speedup vs scalar = %.2fx\n",
                t_neon * 1000.0, t_neon * 1e6 / (double)repeats, t_scalar / t_neon);
    }
#endif

    simd_test_aligned_free(histograms);
    simd_test_aligned_free(image);
    simd_test_aligned_free(mask);
    simd_test_aligned_free(out);
    return NULL;
}

char *run_tests(void)
{
    /* Required so `vmaf_get_cpu_flags()` reports AVX-2 / AVX-512 / NEON
     * (without this it returns 0 and the SIMD branches are skipped). */
    vmaf_init_cpu();
    mu_run_test(test_cambi_simd_row_bitexact);
    mu_run_test(test_cambi_simd_row_microbench);
    return NULL;
}
