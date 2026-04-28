/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CUDA kernel for the ssimulacra2 separable FastGaussian IIR blur.
 *  Mirrors the libjxl Charalampidis 2016 3-pole recursive Gaussian
 *  (k = {1, 3, 5}, sigma=1.5, zero-padded boundaries) implemented
 *  in libvmaf/src/feature/ssimulacra2.c::fast_gaussian_1d.
 *
 *  The IIR is sequential along the scan axis. Separability lets us
 *  parallelise across the orthogonal axis:
 *    H pass: one thread per row, scans left to right.
 *    V pass: one thread per column, scans top to bottom.
 *
 *  Bit-exact-with-CPU strategy (per ADR-0192 / ADR-0201): the
 *  CPU writes `o = n2 * sum - d1 * prev1 - prev2` as separate
 *  FMUL/FSUB ops under -ffp-contract=off. Compile this fatbin
 *  with --fmad=false so NVCC does NOT fuse those into FMAs;
 *  otherwise the IIR pole tracking compounds an FMA-rounding
 *  delta across the radius and the 6-scale pyramid into a
 *  ~1e-3 pooled-score drift (places=1).
 */

#include "cuda_helper.cuh"

extern "C" {

/* H pass: one thread per row.
 *  in_buf / out_buf are full 3-plane buffers (channel offsets
 *  passed via in_offset / out_offset).
 *
 *  Bit-identical control flow with CPU `fast_gaussian_1d`. */
__global__ void ssimulacra2_blur_h(const float *__restrict__ in_buf, float *__restrict__ out_buf,
                                   unsigned width, unsigned height, float n2_0, float n2_1,
                                   float n2_2, float d1_0, float d1_1, float d1_2, int radius,
                                   unsigned in_offset, unsigned out_offset)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height)
        return;

    const int xsize = (int)width;
    const int N = radius;

    float prev1_0 = 0.f, prev1_1 = 0.f, prev1_2 = 0.f;
    float prev2_0 = 0.f, prev2_1 = 0.f, prev2_2 = 0.f;

    const unsigned in_base = in_offset + row * width;
    const unsigned out_base = out_offset + row * width;

    for (int n = -N + 1; n < xsize; ++n) {
        const int left = n - N - 1;
        const int right = n + N - 1;
        const float lv = (left >= 0) ? in_buf[in_base + (unsigned)left] : 0.f;
        const float rv = (right < xsize) ? in_buf[in_base + (unsigned)right] : 0.f;
        const float sum = lv + rv;

        /* Match CPU expression `n2*sum - d1*prev1 - prev2` ordering
         * exactly. Explicit temporaries + --fmad=false at the
         * fatbin level keep this as separate FMUL/FSUB ops. */
        const float ns0 = n2_0 * sum;
        const float dp0 = d1_0 * prev1_0;
        const float t0 = ns0 - dp0;
        const float o0 = t0 - prev2_0;
        const float ns1 = n2_1 * sum;
        const float dp1 = d1_1 * prev1_1;
        const float t1 = ns1 - dp1;
        const float o1 = t1 - prev2_1;
        const float ns2 = n2_2 * sum;
        const float dp2 = d1_2 * prev1_2;
        const float t2 = ns2 - dp2;
        const float o2 = t2 - prev2_2;
        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            const float s01 = o0 + o1;
            const float s_total = s01 + o2;
            out_buf[out_base + (unsigned)n] = s_total;
        }
    }
}

/* V pass: one thread per column.  */
__global__ void ssimulacra2_blur_v(const float *__restrict__ in_buf, float *__restrict__ out_buf,
                                   unsigned width, unsigned height, float n2_0, float n2_1,
                                   float n2_2, float d1_0, float d1_1, float d1_2, int radius,
                                   unsigned in_offset, unsigned out_offset)
{
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
        return;

    const int ysize = (int)height;
    const int N = radius;

    float prev1_0 = 0.f, prev1_1 = 0.f, prev1_2 = 0.f;
    float prev2_0 = 0.f, prev2_1 = 0.f, prev2_2 = 0.f;

    for (int n = -N + 1; n < ysize; ++n) {
        const int left = n - N - 1;
        const int right = n + N - 1;
        const float lv = (left >= 0) ? in_buf[in_offset + (unsigned)left * width + col] : 0.f;
        const float rv = (right < ysize) ? in_buf[in_offset + (unsigned)right * width + col] : 0.f;
        const float sum = lv + rv;

        const float ns0 = n2_0 * sum;
        const float dp0 = d1_0 * prev1_0;
        const float t0 = ns0 - dp0;
        const float o0 = t0 - prev2_0;
        const float ns1 = n2_1 * sum;
        const float dp1 = d1_1 * prev1_1;
        const float t1 = ns1 - dp1;
        const float o1 = t1 - prev2_1;
        const float ns2 = n2_2 * sum;
        const float dp2 = d1_2 * prev1_2;
        const float t2 = ns2 - dp2;
        const float o2 = t2 - prev2_2;
        prev2_0 = prev1_0;
        prev2_1 = prev1_1;
        prev2_2 = prev1_2;
        prev1_0 = o0;
        prev1_1 = o1;
        prev1_2 = o2;

        if (n >= 0) {
            const float s01 = o0 + o1;
            const float s_total = s01 + o2;
            out_buf[out_offset + (unsigned)n * width + col] = s_total;
        }
    }
}

} /* extern "C" */
