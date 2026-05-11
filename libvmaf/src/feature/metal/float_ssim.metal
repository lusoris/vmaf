/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for float_ssim (T8-1j / ADR-0421).
 *  Direct port of `libvmaf/src/feature/vulkan/shaders/ssim.comp`.
 *
 *  Algorithm: separable 11-tap Gaussian convolution on five SSIM
 *  statistics (μ_r, μ_d, σ_r², σ_d², σ_rd), then per-pixel SSIM
 *  combine using the standard formula:
 *
 *    SSIM(x,y) = (2·μ_r·μ_d + C1)·(2·σ_rd + C2)
 *              / ((μ_r² + μ_d² + C1)·(σ_r² + σ_d² + C2))
 *
 *  where σ² = E[x²] − μ², σ_rd = E[rd] − μ_r·μ_d.
 *  Constants: C1 = (0.01·255)², C2 = (0.03·255)² (L = 255, fixed).
 *
 *  Inputs are float planes pre-normalized to [0, 255] by the host
 *  (applies regardless of bpc — same as float_ssim.c::extract via
 *  picture_copy).
 *
 *  "Valid" convolution — no mirror padding:
 *    horiz output:  w_h × H   where w_h = W - 10
 *    vert output:   w_h × h_v where h_v = H - 10
 *  SSIM is averaged over w_h × h_v pixels.
 *
 *  Two kernel functions, dispatched by the .mm host in order:
 *
 *  1. `float_ssim_horiz` — horizontal 11-tap on ref and dis → 5
 *     intermediate float planes in hbuf (stride = w_h).
 *     Grid: ceil(w_h / 16) × ceil(H / 8)   [16×8 threadgroup]
 *
 *  2. `float_ssim_vert_combine` — vertical 11-tap on hbuf → per-pixel
 *     SSIM → per-WG float partial.
 *     Grid: ceil(w_h / 16) × ceil(h_v / 8)
 *
 *  Buffer bindings for `float_ssim_horiz`:
 *   [[buffer(0)]] ref_f   — float * (full W×H reference, row-major)
 *   [[buffer(1)]] dis_f   — float * (full W×H distorted)
 *   [[buffer(2)]] hbuf    — float * (5 × w_h × H interleaved:
 *                            stride per stat = w_h × H;
 *                            stat layout: [mu_r | mu_d | sq_r | sq_d | rd])
 *   [[buffer(3)]] params  — uint4 (.x=W, .y=H, .z=w_h, .w=0)
 *
 *  Buffer bindings for `float_ssim_vert_combine`:
 *   [[buffer(0)]] hbuf    — float * (5 × w_h × H from pass 0)
 *   [[buffer(1)]] partials — float * (grid_w × grid_h per-WG partial)
 *   [[buffer(2)]] params  — uint4 (.x=W, .y=H, .z=w_h, .w=h_v)
 *   [[buffer(3)]] consts  — float2 (.x=c1, .y=c2)
 *   [[buffer(4)]] grid_dim — uint2 (.x=grid_w, .y=grid_h) for partial index
 */

#include <metal_stdlib>
using namespace metal;

constant float G[11] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f,
    0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f
};

#define K       11
#define K_HALF  5

/* ------------------------------------------------------------------ */
/*  Pass 0: horizontal convolution                                      */
/* ------------------------------------------------------------------ */
kernel void float_ssim_horiz(
    const device float  *ref_f  [[buffer(0)]],
    const device float  *dis_f  [[buffer(1)]],
    device       float  *hbuf   [[buffer(2)]],
    constant     uint4  &params [[buffer(3)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint W   = params.x;
    const uint H   = params.y;
    const uint w_h = params.z;  /* W - 10 */

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= w_h || y >= H) { return; }

    float mu_r = 0.0f, mu_d = 0.0f;
    float sq_r = 0.0f, sq_d = 0.0f, rd = 0.0f;

    /* Convolve columns [x, x + K - 1] — "valid" (no padding). */
    for (int u = 0; u < K; ++u) {
        const uint src_x   = x + (uint)u;
        const uint src_idx = y * W + src_x;
        const float r = ref_f[src_idx];
        const float d = dis_f[src_idx];
        const float w = G[u];
        mu_r += w * r;
        mu_d += w * d;
        sq_r += w * (r * r);
        sq_d += w * (d * d);
        rd   += w * (r * d);
    }

    /* hbuf layout: [mu_r plane | mu_d plane | sq_r plane | sq_d plane | rd plane]
     * Each plane has stride w_h per row, total w_h × H floats. */
    const uint plane_size = w_h * H;
    const uint dst_idx    = y * w_h + x;
    hbuf[0 * plane_size + dst_idx] = mu_r;
    hbuf[1 * plane_size + dst_idx] = mu_d;
    hbuf[2 * plane_size + dst_idx] = sq_r;
    hbuf[3 * plane_size + dst_idx] = sq_d;
    hbuf[4 * plane_size + dst_idx] = rd;
}

/* ------------------------------------------------------------------ */
/*  Pass 1: vertical convolution + per-pixel SSIM + per-WG reduction   */
/* ------------------------------------------------------------------ */
kernel void float_ssim_vert_combine(
    const device float   *hbuf     [[buffer(0)]],
    device       float   *partials [[buffer(1)]],
    constant     uint4   &params   [[buffer(2)]],
    constant     float2  &consts   [[buffer(3)]],
    constant     uint2   &grid_dim [[buffer(4)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const uint w_h = params.z;  /* W - 10 */
    const uint h_v = params.w;  /* H - 10 */

    const float c1 = consts.x;
    const float c2 = consts.y;

    const uint x = gid.x;
    const uint y = gid.y;

    float my_ssim = 0.0f;
    if (x < w_h && y < h_v) {
        const uint plane_size = w_h * params.y;  /* w_h × H */

        float mu_r = 0.0f, mu_d = 0.0f;
        float sq_r = 0.0f, sq_d = 0.0f, rd = 0.0f;

        /* Convolve rows [y, y + K - 1] — "valid" (no padding). */
        for (int v = 0; v < K; ++v) {
            const uint src_y   = y + (uint)v;
            const uint src_idx = src_y * w_h + x;
            const float w = G[v];
            mu_r += w * hbuf[0 * plane_size + src_idx];
            mu_d += w * hbuf[1 * plane_size + src_idx];
            sq_r += w * hbuf[2 * plane_size + src_idx];
            sq_d += w * hbuf[3 * plane_size + src_idx];
            rd   += w * hbuf[4 * plane_size + src_idx];
        }

        /* SSIM combine — matches ssim_tools.c::ssim_map_s. */
        const float var_r  = sq_r - mu_r * mu_r;
        const float var_d  = sq_d - mu_d * mu_d;
        const float cov_rd = rd   - mu_r * mu_d;
        const float num = (2.0f * mu_r * mu_d + c1) * (2.0f * cov_rd + c2);
        const float den = (mu_r * mu_r + mu_d * mu_d + c1) *
                          (var_r + var_d + c2);
        my_ssim = num / den;
    }

    /* Per-WG float partial (host sums and divides by w_h × h_v). */
    threadgroup float sg_sums[8];
    const float lane_sum = simd_sum(my_ssim);
    if (simd_lane == 0) { sg_sums[simd_id] = lane_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float gs = 0.0f;
        for (uint i = 0; i < simd_count; ++i) { gs += sg_sums[i]; }
        partials[bid.y * grid_dim.x + bid.x] = gs;
    }
}
