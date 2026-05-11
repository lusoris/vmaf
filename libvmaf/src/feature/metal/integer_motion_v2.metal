/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernel for integer_motion_v2 (T8-1c / ADR-0421).
 *  Direct translation of `libvmaf/src/feature/cuda/integer_motion_v2/
 *  motion_v2_score.cu` (ADR-0192 / ADR-0193) — same algorithm,
 *  threadgroup-shared tile, separable filter, atomic SAD reduction.
 *
 *  Algorithm parity (must match scalar `integer_motion_v2.c`):
 *    1. Per pixel: diff = prev[i,j] - cur[i,j]   (signed)
 *    2. Vertical filter on diff:
 *         v[i,j] = (sum_k filter[k] * diff[mirror(i-2+k), j]
 *                    + (1 << (bpc - 1))) >> bpc
 *    3. Horizontal filter on v:
 *         h[i,j] = (sum_k filter[k] * v[i, mirror(j-2+k)]
 *                    + 32768) >> 16
 *    4. SAD: atomic-add |h[i,j]| into a single int64 accumulator.
 *
 *  Mirror padding: edge-replicating reflective mirror
 *  (`2 * size - idx - 1` for idx >= size), matching the CUDA twin.
 *  DIFFERS from motion_v1's `2 * size - idx - 2`; see CUDA file
 *  header for the bring-up note.
 *
 *  Threadgroup layout: 16 × 16 threads per group, +2 pixel halo on
 *  each side → 20 × 20 shared tile. Inner pitch padded to 21 to
 *  break SIMD-group bank conflicts (analogous to the CUDA bank-
 *  conflict mitigation; Apple GPUs have a 32-bank threadgroup
 *  memory similar to NVIDIA).
 *
 *  Bit-exactness gate: `places=4` cross-backend-diff against scalar
 *  (per ADR-0214). Validation pending Apple Silicon hardware run.
 */

#include <metal_stdlib>
using namespace metal;

constant constexpr int MV2_FILTER[5] = {3571, 16004, 26386, 16004, 3571};
constant constexpr int MV2_RADIUS    = 2;
constant constexpr int MV2_BLOCK_X   = 16;
constant constexpr int MV2_BLOCK_Y   = 16;
constant constexpr int MV2_TILE_W    = MV2_BLOCK_X + 2 * MV2_RADIUS; /* 20 */
constant constexpr int MV2_TILE_H    = MV2_BLOCK_Y + 2 * MV2_RADIUS; /* 20 */
constant constexpr int MV2_TILE_PITCH = MV2_TILE_W + 1;              /* 21 */

inline int mv2_mirror(int idx, int sup)
{
    if (idx < 0)         return -idx;
    if (idx >= sup)      return 2 * sup - idx - 1;
    return idx;
}

/* SIMD-group reduction. Apple GPUs use a fixed SIMD width of 32
 * threads (Apple7+) — same width as NVIDIA warps, so the CUDA
 * reduction shape ports one-to-one. `simd_shuffle_down` is the
 * MSL equivalent of `__shfl_down_sync`. */
inline long mv2_simd_sum(long v)
{
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v,  8);
    v += simd_shuffle_down(v,  4);
    v += simd_shuffle_down(v,  2);
    v += simd_shuffle_down(v,  1);
    return v;
}

/*
 * 8 bpc kernel — packed `uint8_t` ref + dist samples, single
 * `atomic_long` accumulator on the host side.
 *
 * Buffer bindings:
 *   [[buffer(0)]] prev       — const uint8_t * (planar Y, row pitch in bytes via `strides.x`)
 *   [[buffer(1)]] cur        — const uint8_t * (planar Y, row pitch via `strides.y`)
 *   [[buffer(2)]] sad        — atomic_long, single accumulator
 *   [[buffer(3)]] strides    — packed (prev_stride, cur_stride) in bytes (uint2)
 *   [[buffer(4)]] dim        — packed (width, height) (uint2)
 */
kernel void motion_v2_kernel_8bpc(
    const device uchar       *prev    [[buffer(0)]],
    const device uchar       *cur     [[buffer(1)]],
    device   atomic_long     &sad     [[buffer(2)]],
    constant uint2           &strides [[buffer(3)]],
    constant uint2           &dim     [[buffer(4)]],
    threadgroup int          *s_diff_raw [[threadgroup(0)]],
    uint2  gid               [[thread_position_in_grid]],
    uint2  tid               [[thread_position_in_threadgroup]],
    uint2  bid               [[threadgroup_position_in_grid]],
    uint   lid               [[thread_index_in_threadgroup]],
    uint   simd_lane         [[thread_index_in_simdgroup]])
{
    /* Cast the flat threadgroup buffer to a 2-D view via index math
     * — MSL doesn't accept multi-dimensional threadgroup arrays as
     * kernel params, so the host passes a flat `int *` of size
     * MV2_TILE_H * MV2_TILE_PITCH. */
    threadgroup int *s_diff = s_diff_raw; /* s_diff[ty * MV2_TILE_PITCH + tx] */

    const int width  = (int)dim.x;
    const int height = (int)dim.y;
    const int prev_stride = (int)strides.x;
    const int cur_stride  = (int)strides.y;

    constant constexpr int shift_y = 8;
    constant constexpr int round_y = 1 << 7;
    constant constexpr int shift_x = 16;
    constant constexpr int round_x = 1 << 15;

    const int tile_origin_x = (int)bid.x * MV2_BLOCK_X - MV2_RADIUS;
    const int tile_origin_y = (int)bid.y * MV2_BLOCK_Y - MV2_RADIUS;
    const int tile_elems    = MV2_TILE_W * MV2_TILE_H;
    const int wg_size       = MV2_BLOCK_X * MV2_BLOCK_Y;

    for (int i = (int)lid; i < tile_elems; i += wg_size) {
        const int ty = i / MV2_TILE_W;
        const int tx = i % MV2_TILE_W;
        const int gx = mv2_mirror(tile_origin_x + tx, width);
        const int gy = mv2_mirror(tile_origin_y + ty, height);
        const int p  = (int)prev[gy * prev_stride + gx];
        const int c  = (int)cur [gy * cur_stride  + gx];
        s_diff[ty * MV2_TILE_PITCH + tx] = p - c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    long abs_h = 0;
    if ((int)gid.x < width && (int)gid.y < height) {
        const int lx = (int)tid.x + MV2_RADIUS;
        const int ly = (int)tid.y + MV2_RADIUS;

        long blurred = 0;
        for (int xf = 0; xf < 5; ++xf) {
            int blurred_y = 0;
            for (int yf = 0; yf < 5; ++yf) {
                blurred_y += MV2_FILTER[yf] *
                             s_diff[(ly - MV2_RADIUS + yf) * MV2_TILE_PITCH +
                                    (lx - MV2_RADIUS + xf)];
            }
            const int v = (blurred_y + round_y) >> shift_y;
            blurred += (long)MV2_FILTER[xf] * (long)v;
        }
        const long h = (blurred + round_x) >> shift_x;
        abs_h = h < 0 ? -h : h;
    }

    /* SIMD-group reduction → lane 0 carries the partial; one atomic-add
     * per simdgroup keeps contention on the single accumulator low. */
    const long lane_sum = mv2_simd_sum(abs_h);
    if (simd_lane == 0) {
        atomic_fetch_add_explicit(&sad, lane_sum, memory_order_relaxed);
    }
}

/*
 * 16 bpc kernel — `uint16_t` samples (each row a multiple of 2
 * bytes; strides are in bytes). The inner accumulator stays
 * `long` from the start because 26386 × 65535 × 5 ≈ 8.6e9
 * overflows int32; see the CUDA twin's 16bpc kernel for the
 * same comment.
 *
 * `bpc` arrives via the strides buffer (z component reuse) to
 * avoid a separate setBytes call on the host. Strides type is
 * promoted to `uint4`: (prev_stride, cur_stride, bpc, reserved).
 */
kernel void motion_v2_kernel_16bpc(
    const device uchar       *prev    [[buffer(0)]],
    const device uchar       *cur     [[buffer(1)]],
    device   atomic_long     &sad     [[buffer(2)]],
    constant uint4           &strides [[buffer(3)]],
    constant uint2           &dim     [[buffer(4)]],
    threadgroup int          *s_diff_raw [[threadgroup(0)]],
    uint2  gid               [[thread_position_in_grid]],
    uint2  tid               [[thread_position_in_threadgroup]],
    uint2  bid               [[threadgroup_position_in_grid]],
    uint   lid               [[thread_index_in_threadgroup]],
    uint   simd_lane         [[thread_index_in_simdgroup]])
{
    threadgroup int *s_diff = s_diff_raw;

    const int width  = (int)dim.x;
    const int height = (int)dim.y;
    const int prev_stride = (int)strides.x; /* bytes */
    const int cur_stride  = (int)strides.y; /* bytes */
    const int bpc         = (int)strides.z;

    const int shift_y = bpc;
    const int round_y = 1 << (bpc - 1);
    constant constexpr int shift_x = 16;
    constant constexpr int round_x = 1 << 15;

    const int tile_origin_x = (int)bid.x * MV2_BLOCK_X - MV2_RADIUS;
    const int tile_origin_y = (int)bid.y * MV2_BLOCK_Y - MV2_RADIUS;
    const int tile_elems    = MV2_TILE_W * MV2_TILE_H;
    const int wg_size       = MV2_BLOCK_X * MV2_BLOCK_Y;

    for (int i = (int)lid; i < tile_elems; i += wg_size) {
        const int ty = i / MV2_TILE_W;
        const int tx = i % MV2_TILE_W;
        const int gx = mv2_mirror(tile_origin_x + tx, width);
        const int gy = mv2_mirror(tile_origin_y + ty, height);
        /* uint16_t fetch via reinterpret + byte stride. */
        const device ushort *prev_row =
            (const device ushort *)(prev + gy * prev_stride);
        const device ushort *cur_row  =
            (const device ushort *)(cur  + gy * cur_stride);
        const int p = (int)prev_row[gx];
        const int c = (int)cur_row [gx];
        s_diff[ty * MV2_TILE_PITCH + tx] = p - c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    long abs_h = 0;
    if ((int)gid.x < width && (int)gid.y < height) {
        const int lx = (int)tid.x + MV2_RADIUS;
        const int ly = (int)tid.y + MV2_RADIUS;

        long blurred = 0;
        for (int xf = 0; xf < 5; ++xf) {
            long blurred_y = 0;
            for (int yf = 0; yf < 5; ++yf) {
                blurred_y += (long)MV2_FILTER[yf] *
                             (long)s_diff[(ly - MV2_RADIUS + yf) * MV2_TILE_PITCH +
                                          (lx - MV2_RADIUS + xf)];
            }
            const int v = (int)((blurred_y + (long)round_y) >> shift_y);
            blurred += (long)MV2_FILTER[xf] * (long)v;
        }
        const long h = (blurred + (long)round_x) >> shift_x;
        abs_h = h < 0 ? -h : h;
    }

    const long lane_sum = mv2_simd_sum(abs_h);
    if (simd_lane == 0) {
        atomic_fetch_add_explicit(&sad, lane_sum, memory_order_relaxed);
    }
}
