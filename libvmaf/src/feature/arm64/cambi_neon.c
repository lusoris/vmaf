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
#include <stdbool.h>
#include <stdint.h>

void cambi_increment_range_neon(uint16_t *arr, int left, int right)
{
    uint16x8_t one = vdupq_n_u16(1);
    int col = left;
    for (; col + 8 <= right; col += 8) {
        uint16x8_t data = vld1q_u16(&arr[col]);
        data = vaddq_u16(data, one);
        vst1q_u16(&arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    }
}

void cambi_decrement_range_neon(uint16_t *arr, int left, int right)
{
    uint16x8_t one = vdupq_n_u16(1);
    int col = left;
    for (; col + 8 <= right; col += 8) {
        uint16x8_t data = vld1q_u16(&arr[col]);
        data = vsubq_u16(data, one);
        vst1q_u16(&arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    }
}

void get_derivative_data_for_row_neon(const uint16_t *image_data, uint16_t *derivative_buffer,
                                      int width, int height, int row, int stride)
{
    uint16x8_t ones = vdupq_n_u16(1);

    if (row == height - 1) {
        /* Last row: only horizontal derivatives */
        int col = 0;
        for (; col + 8 <= width - 1; col += 8) {
            uint16x8_t vals1 = vld1q_u16(&image_data[row * stride + col]);
            uint16x8_t vals2 = vld1q_u16(&image_data[row * stride + col + 1]);
            /* cmpeq returns 0xFFFF for equal, 0 for not */
            uint16x8_t eq = vceqq_u16(vals1, vals2);
            /* AND with 1 to get 1/0 instead of 0xFFFF/0 */
            vst1q_u16(&derivative_buffer[col], vandq_u16(ones, eq));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] =
                (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    } else {
        /* Interior rows: horizontal AND vertical derivatives */
        int col = 0;
        for (; col + 8 <= width - 1; col += 8) {
            uint16x8_t h1 = vld1q_u16(&image_data[row * stride + col]);
            uint16x8_t h2 = vld1q_u16(&image_data[row * stride + col + 1]);
            uint16x8_t horiz_eq = vandq_u16(ones, vceqq_u16(h1, h2));

            uint16x8_t v1 = vld1q_u16(&image_data[row * stride + col]);
            uint16x8_t v2 = vld1q_u16(&image_data[(row + 1) * stride + col]);
            uint16x8_t vert_eq = vandq_u16(ones, vceqq_u16(v1, v2));

            vst1q_u16(&derivative_buffer[col], vandq_u16(horiz_eq, vert_eq));
        }
        for (; col < width; col++) {
            bool horizontal_derivative =
                (col == width - 1 ||
                 image_data[row * stride + col] == image_data[row * stride + col + 1]);
            bool vertical_derivative =
                image_data[row * stride + col] == image_data[(row + 1) * stride + col];
            derivative_buffer[col] = horizontal_derivative && vertical_derivative;
        }
    }
}

/* NEON twin of `calculate_c_values_row_avx2` from `x86/cambi_avx2.c`.
 *
 * NEON has no general-purpose gather instruction, so the four histogram
 * accesses per lane (`p_0` plus per-diff `p_1` / `p_2`) are emitted as scalar
 * loads from the lane indices. The arithmetic above the gather (mask check,
 * predicate composition, weighted product, reciprocal-LUT multiplication, and
 * running max) is vectorised across four int32 lanes (`uint32x4_t`,
 * `float32x4_t`).
 *
 * Bit-exactness vs. the scalar reference is preserved: each lane independently
 * reproduces the scalar `c_value_pixel` arithmetic in the same order with the
 * same int32 / float32 widths. (Cf. ADR-0138 / ADR-0139 — the fork forbids any
 * vectorised reduction reordering that would alter IEEE-754 rounding for cambi.)
 */
void calculate_c_values_row_neon(float *c_values, const uint16_t *histograms, const uint16_t *image,
                                 const uint16_t *mask, int row, int width, ptrdiff_t stride,
                                 const uint16_t num_diffs, const uint16_t *tvi_thresholds,
                                 uint16_t vlt_luma, const int *diff_weights, const int *all_diffs,
                                 const float *reciprocal_lut)
{
    int v_lo_signed_sc = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed_sc > 0 ? (uint16_t)v_lo_signed_sc : 0;
    uint16_t v_band_size = tvi_thresholds[num_diffs - 1] + 1 - v_band_base;

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row = &mask[row * stride];
    float *c_row = &c_values[row * width];

    const int32x4_t num_diffs_v = vdupq_n_s32(num_diffs);
    const int32x4_t vlt_luma_v = vdupq_n_s32(vlt_luma);
    const int32x4_t band_offset_v = vdupq_n_s32((int)num_diffs + (int)v_band_base);
    const int32x4_t band_max_v = vdupq_n_s32((int)v_band_size - 1);
    const int32x4_t zero_i = vdupq_n_s32(0);
    const float32x4_t zero_f = vdupq_n_f32(0.0f);

    int col = 0;
    /* Vector loop: same `col + 16 < width` invariant as the AVX-2 / AVX-512
     * paths — keep one column of safety margin past the chunk so any clamped
     * gather index never overruns the histogram buffer. We process 4 lanes per
     * inner step but iterate over groups of 16 to share the predicate-skip and
     * the per-chunk early-out. */
    for (; col + 16 < width; col += 4) {
        /* Load 4 mask uint16 values, widen to int32, build active mask. */
        uint16x4_t mask16 = vld1_u16(&mask_row[col]);
        uint32x4_t mask_active = vmovl_u16(mask16);
        /* Skip the chunk if no lane is active. */
        if (vmaxvq_u32(mask_active) == 0) {
            vst1q_f32(&c_row[col], zero_f);
            continue;
        }
        /* Materialise the predicate as 0xFFFFFFFF / 0 lanes (lane > 0). */
        uint32x4_t mask_active_pred = vcgtq_u32(mask_active, vreinterpretq_u32_s32(zero_i));

        /* value = image[col + lane] + num_diffs (used for TVI / vlt checks). */
        uint16x4_t img16 = vld1_u16(&image_row[col]);
        int32x4_t value_v = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(img16)), num_diffs_v);

        /* compact_v = value_v - band_offset, clamped to [0, band_max] for safe access. */
        int32x4_t compact_v = vsubq_s32(value_v, band_offset_v);
        compact_v = vmaxq_s32(compact_v, zero_i);
        compact_v = vminq_s32(compact_v, band_max_v);

        /* Scalar gather of p_0 (NEON has no vector gather; the lane count is 4
         * so the unrolled loads are cheap). */
        int32_t cv_lanes[4];
        vst1q_s32(cv_lanes, compact_v);
        uint32_t p0_lanes[4];
        for (int lane = 0; lane < 4; lane++) {
            p0_lanes[lane] = histograms[cv_lanes[lane] * width + col + lane];
        }
        uint32x4_t p0 = vld1q_u32(p0_lanes);

        float32x4_t c_value = zero_f;

        for (int d = 0; d < num_diffs; d++) {
            int delta_plus = all_diffs[num_diffs + d + 1];
            int delta_minus = all_diffs[num_diffs - d - 1];
            int weight = diff_weights[d];
            int tvi_thresh = tvi_thresholds[d];

            /* pred_a = (value <= tvi_thresh). NEON's `vcleq_s32` returns
             * 0xFFFFFFFF / 0. */
            uint32x4_t pred_a = vcleq_s32(value_v, vdupq_n_s32(tvi_thresh));

            int32x4_t value_plus = vaddq_s32(value_v, vdupq_n_s32(delta_plus));
            uint32x4_t pred_b = vcgtq_s32(value_plus, vlt_luma_v);
            uint32x4_t predicate = vandq_u32(pred_a, pred_b);
            if (vmaxvq_u32(predicate) == 0)
                continue;

            /* compact p1/p2 indices, clamped + tracked-OOB for the minus side. */
            int32x4_t compact_plus_raw = vaddq_s32(compact_v, vdupq_n_s32(delta_plus));
            int32x4_t compact_plus = vminq_s32(compact_plus_raw, band_max_v);

            int32x4_t compact_minus_raw = vaddq_s32(compact_v, vdupq_n_s32(delta_minus));
            uint32x4_t p2_inbounds = vcgeq_s32(compact_minus_raw, zero_i);
            int32x4_t compact_minus = vmaxq_s32(compact_minus_raw, zero_i);

            /* Scalar gather of p_1 / p_2 (4 lanes each). */
            int32_t cp_lanes[4], cm_lanes[4];
            vst1q_s32(cp_lanes, compact_plus);
            vst1q_s32(cm_lanes, compact_minus);
            uint32_t p1_lanes[4], p2_lanes[4];
            for (int lane = 0; lane < 4; lane++) {
                p1_lanes[lane] = histograms[cp_lanes[lane] * width + col + lane];
                p2_lanes[lane] = histograms[cm_lanes[lane] * width + col + lane];
            }
            uint32x4_t p1 = vld1q_u32(p1_lanes);
            uint32x4_t p2 = vld1q_u32(p2_lanes);
            /* Lanes with negative minus-index read 0 (matches the scalar branch
             * `(idx2 >= 0) ? histograms[...] : 0`). */
            p2 = vandq_u32(p2, p2_inbounds);

            uint32x4_t p_max = vmaxq_u32(p1, p2);
            uint32x4_t denom = vaddq_u32(p_max, p0);

            /* num = (float)(weight * p_0 * p_max). All terms uint16-bounded so
             * the int32 mul fits without overflow (matches the AVX-2 / scalar
             * arithmetic). */
            uint32x4_t weight_v = vdupq_n_u32((uint32_t)weight);
            uint32x4_t num_int = vmulq_u32(weight_v, vmulq_u32(p0, p_max));
            float32x4_t num_f = vcvtq_f32_s32(vreinterpretq_s32_u32(num_int));

            /* rcp = reciprocal_lut[denom], scalar gather over 4 lanes. */
            uint32_t denom_lanes[4];
            vst1q_u32(denom_lanes, denom);
            float rcp_lanes[4];
            for (int lane = 0; lane < 4; lane++) {
                rcp_lanes[lane] = reciprocal_lut[denom_lanes[lane]];
            }
            float32x4_t rcp = vld1q_f32(rcp_lanes);

            float32x4_t val = vmulq_f32(num_f, rcp);
            /* Mask off lanes where predicate is false (val := 0 for those),
             * then take the running max. */
            val = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(val), predicate));
            c_value = vmaxq_f32(c_value, val);
        }

        /* Apply mask: lanes with mask == 0 keep 0. */
        c_value =
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(c_value), mask_active_pred));
        vst1q_f32(&c_row[col], c_value);
    }

    /* Scalar tail (bit-identical with the scalar reference). */
    for (; col < width; col++) {
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
