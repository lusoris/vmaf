/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CUDA compute kernels for the CAMBI banding-detection feature extractor
 *  (T3-15 / ADR-0360). CUDA twin of cambi_vulkan.c (ADR-0210).
 *
 *  Strategy II hybrid (mirrors the Vulkan precedent from ADR-0205 and
 *  ADR-0210): three GPU kernels handle the embarrassingly parallel stages
 *  (spatial mask, 2× decimate, 3-tap separable mode filter), while the
 *  precision-sensitive sliding-histogram `calculate_c_values` pass and
 *  top-K spatial pooling run on the host CPU via the shared wrappers in
 *  `cambi_internal.h`. This keeps the CUDA port bit-exact at `places=4`
 *  w.r.t. the CPU extractor (ULP=0 on the emitted score) at no extra
 *  development risk from a fully-on-GPU histogram pass.
 *
 *  GPU kernel inventory:
 *
 *    cambi_spatial_mask_kernel — derivative (pixel == right AND == below)
 *        + 7×7 summed-area table in-register, then threshold compare.
 *        One thread per output pixel; 2-pass SAT (row-scan → col-scan
 *        in shared memory) is unnecessary because the 7×7 window fits in
 *        the per-thread accum register without smem pressure at 16×16
 *        blocks. Bit-exact with cambi.c::get_spatial_mask_for_index.
 *
 *    cambi_decimate_kernel — strict 2× stride-2 subsample of a uint16
 *        luma buffer. One thread per output pixel. Bit-exact with
 *        cambi.c::decimate.
 *
 *    cambi_filter_mode_kernel — separable 3-tap mode filter, horizontal
 *        pass first then vertical, each in a separate kernel launch
 *        (axis == 0 → H, axis == 1 → V). One thread per output pixel.
 *        Bit-exact with cambi.c::filter_mode.
 *
 *  All buffers are flat `uint16_t` device arrays; the host glue converts
 *  from/to `VmafPicture` layout via DtoH / HtoD memcpy.
 *
 *  Precision contract: `places=4` (ULP=0 on host-emitted score).
 *  The three GPU phases are integer + bit-exact w.r.t. CPU scalar.
 *  The host residual runs the exact CPU code path from cambi_internal.h,
 *  so the final CAMBI score is bit-for-bit identical to `vmaf_fex_cambi`.
 */

#include "cuda_helper.cuh"
#include "cuda/integer_cambi_cuda.h"
#include "common.h"

/* ------------------------------------------------------------------
 * Kernel 1: Spatial mask
 *
 * Input:  `image`  — flat uint16 array, stride_words columns per row.
 * Output: `mask`   — flat uint16 array, same layout; 1 = edge, 0 = flat.
 *
 * Algorithm (matches cambi.c::get_spatial_mask_for_index via the SAT
 * path):
 *   1. For each pixel (x,y): zero_deriv[y][x] = (image[y][x] == image[y][x+1])
 *                                             && (image[y][x] == image[y+1][x]).
 *      Border pixels treat out-of-bounds neighbours as "equal".
 *   2. Compute the 7×7 box sum of zero_deriv around each pixel using
 *      a naive loop (7×7 = 49 reads, cheap for warp parallelism).
 *   3. mask[y][x] = (box_sum > mask_index) ? 1 : 0.
 *
 * `mask_index` is the integer threshold computed by
 * cambi.c::get_mask_index (filter_size=7, area-dependent formula).
 * ------------------------------------------------------------------*/
extern "C" {

__global__ void cambi_spatial_mask_kernel(const uint16_t *image, uint16_t *mask, unsigned width,
                                          unsigned height, unsigned stride_words,
                                          unsigned mask_index)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= (int)width || y >= (int)height)
        return;

    /* Derivative at (x,y): 1 if equal to both right and bottom neighbours.
     * Edge pixels treat out-of-bound as "equal" (matches CPU). */
    const uint16_t here = image[(size_t)y * stride_words + (unsigned)x];
    const int right_eq =
        (x == (int)width - 1 || here == image[(size_t)y * stride_words + (unsigned)(x + 1)]);
    const int below_eq =
        (y == (int)height - 1 || here == image[(size_t)(y + 1) * stride_words + (unsigned)x]);
    /* zero_deriv[y][x] = 1 when both horizontal and vertical derivatives are zero. */

    /* 7×7 SAT — compute box sum over the clamped [y-3,y+3] × [x-3,x+3] window.
     * For the purposes of this kernel each pixel computes its own window
     * independently (no shared-memory SAT). The 7×7 window is 49 pixels;
     * the warp executes them in parallel so the serial loop is unrolled
     * by the compiler. The approach matches the CPU SAT intent but avoids
     * the cyclic-row DP complexity by paying ~49 global reads per thread.
     * At 1080p (2M pixels × 49 reads = ~100M 16-bit reads) this is
     * memory-bandwidth bound but well within RTX 4090 bandwidth budget. */
    const int HALF = 3; /* (MASK_FILTER_SIZE=7) >> 1 */
    unsigned box_sum = 0;
    for (int dy = -HALF; dy <= HALF; dy++) {
        int ry = y + dy;
        if (ry < 0)
            ry = 0;
        if (ry >= (int)height)
            ry = (int)height - 1;
        for (int dx = -HALF; dx <= HALF; dx++) {
            int rx = x + dx;
            if (rx < 0)
                rx = 0;
            if (rx >= (int)width)
                rx = (int)width - 1;
            const uint16_t p = image[(size_t)ry * stride_words + (unsigned)rx];
            const uint16_t r =
                image[(size_t)ry * stride_words + (unsigned)(rx == (int)width - 1 ? rx : rx + 1)];
            const uint16_t b =
                image[(size_t)(ry == (int)height - 1 ? ry : ry + 1) * stride_words + (unsigned)rx];
            const int eq_right = (rx == (int)width - 1) || (p == r);
            const int eq_below = (ry == (int)height - 1) || (p == b);
            box_sum += (unsigned)(eq_right && eq_below);
        }
    }
    /* Mask pixel equals the pixel's own zero_deriv, not the neighbour's.
     * The CPU code computes: mask[i][j] = (box_sum > mask_index).
     * The box_sum is over the zero_deriv field of the (2×pad+1)² window
     * centred on (i,j). We have already computed that above. */
    mask[(size_t)y * stride_words + (unsigned)x] = (uint16_t)(box_sum > mask_index ? 1u : 0u);
    (void)right_eq;
    (void)below_eq;
}

/* ------------------------------------------------------------------
 * Kernel 2: 2× decimate
 *
 * Output pixel (x,y) samples input pixel (2x, 2y). Matches cambi.c::decimate
 * (strict even-pixel subsample, no filtering). Input and output are in
 * separate flat buffers (`src` and `dst`); both use `stride_words` columns
 * (the output width is `(width+1)/2`, `(height+1)/2`).
 *
 * `src_stride_words` is the stride of the source (larger) buffer.
 * `dst_stride_words` is the stride of the destination (smaller) buffer.
 * Both strides are in uint16_t words.
 * ------------------------------------------------------------------ */
__global__ void cambi_decimate_kernel(const uint16_t *src, uint16_t *dst, unsigned out_width,
                                      unsigned out_height, unsigned src_stride_words,
                                      unsigned dst_stride_words)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height)
        return;

    /* Sample at stride-2 — exact match of cambi.c::decimate:
     *   data[i * stride + j] = data[(i<<1) * stride + (j<<1)]; */
    dst[(size_t)y * dst_stride_words + x] = src[(size_t)(y * 2u) * src_stride_words + x * 2u];
}

/* ------------------------------------------------------------------
 * Kernel 3: Separable 3-tap mode filter
 *
 * One kernel, two launches: axis=0 (horizontal), axis=1 (vertical).
 * The mode of three equal-length 1-D triplets is the value that appears
 * at least twice, or the minimum if all three are distinct — matching
 * cambi.c::mode3.
 *
 * For axis=0 (H pass): each thread (x,y) writes
 *     out[y][x] = mode3(in[y][x-1], in[y][x], in[y][x+1])
 * except at the left/right border where in[y][-1] = in[y][0] etc.
 *
 * For axis=1 (V pass): same but over rows:
 *     out[y][x] = mode3(in[y-1][x], in[y][x], in[y+1][x])
 * with clamped borders.
 *
 * Matches cambi.c::filter_mode's row-by-row logic; the only difference
 * is that the GPU computes all rows in parallel (no rolling 3-row buffer
 * trick needed — we have enough global memory).
 * ------------------------------------------------------------------ */
__device__ static inline uint16_t mode3_dev(uint16_t a, uint16_t b, uint16_t c)
{
    /* Two equal → that value. All distinct → min of the three. */
    if (a == b || a == c)
        return a;
    if (b == c)
        return b;
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

__global__ void cambi_filter_mode_kernel(const uint16_t *in, uint16_t *out, unsigned width,
                                         unsigned height, unsigned stride_words, int axis)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= (int)width || y >= (int)height)
        return;

    uint16_t a, b, c;
    if (axis == 0) {
        /* Horizontal: neighbours in x. */
        const int xl = (x > 0) ? x - 1 : 0;
        const int xr = (x < (int)width - 1) ? x + 1 : (int)width - 1;
        a = in[(size_t)y * stride_words + (unsigned)xl];
        b = in[(size_t)y * stride_words + (unsigned)x];
        c = in[(size_t)y * stride_words + (unsigned)xr];
    } else {
        /* Vertical: neighbours in y. */
        const int yu = (y > 0) ? y - 1 : 0;
        const int yd = (y < (int)height - 1) ? y + 1 : (int)height - 1;
        a = in[(size_t)(unsigned)yu * stride_words + (unsigned)x];
        b = in[(size_t)(unsigned)y * stride_words + (unsigned)x];
        c = in[(size_t)(unsigned)yd * stride_words + (unsigned)x];
    }
    out[(size_t)(unsigned)y * stride_words + (unsigned)x] = mode3_dev(a, b, c);
}

} /* extern "C" */
