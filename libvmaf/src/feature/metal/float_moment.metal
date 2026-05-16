/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernel for float_moment (T8-1e / ADR-0421).
 *  Emits four per-frame statistics across ref and dis planes:
 *  float_moment_ref1st, float_moment_dis1st,
 *  float_moment_ref2nd, float_moment_dis2nd.
 *
 *  Algorithm (must match CPU libvmaf/src/feature/float_moment.c and
 *  all other GPU backends — CUDA/SYCL/Vulkan/HIP):
 *    For each plane in {ref, dis}:
 *      1st moment: sum  += raw_val              (integer pixel, pre-scale)
 *      2nd moment: sum2 += raw_val * raw_val    (integer, pre-scale)
 *    Host: reconstruct uint64, cast to double,
 *          divide by (W * H * scaler) for 1st and (W * H * scaler^2) for 2nd.
 *
 *  Reduction: MSL lacks atomic_ulong, so each WG emits lo+hi uint32 slots.
 *  Host reconstructs: val_u64 = ((uint64)hi << 32) | lo.
 *  For 8bpc: scaler = 1. For >8bpc: scaler = 1 << (bpc - 8).
 *
 *  Eight uint partials per WG (two uint32 slots per accumulator):
 *    idx = bid.y * grid_groups.x + bid.x
 *    r1_lo[idx], r1_hi[idx]  — ref 1st moment lo/hi
 *    d1_lo[idx], d1_hi[idx]  — dis 1st moment lo/hi
 *    r2_lo[idx], r2_hi[idx]  — ref 2nd moment lo/hi
 *    d2_lo[idx], d2_hi[idx]  — dis 2nd moment lo/hi
 *
 *  Buffer bindings (8bpc):
 *   [[buffer(0)]]  ref    — const uchar *
 *   [[buffer(1)]]  dis    — const uchar *
 *   [[buffer(2)]]  r1_lo  — uint * (grid_w × grid_h)
 *   [[buffer(3)]]  r1_hi  — uint *
 *   [[buffer(4)]]  d1_lo  — uint *
 *   [[buffer(5)]]  d1_hi  — uint *
 *   [[buffer(6)]]  r2_lo  — uint *
 *   [[buffer(7)]]  r2_hi  — uint *
 *   [[buffer(8)]]  d2_lo  — uint *
 *   [[buffer(9)]]  d2_hi  — uint *
 *   [[buffer(10)]] strides — uint2 (.x=ref_stride, .y=dis_stride)
 *   [[buffer(11)]] dim     — uint2 (width, height)
 *
 *  Buffer bindings (16bpc): same layout but strides is uint4
 *   (.x=ref_stride, .y=dis_stride, .z=bpc, .w=0).
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  Internal helper: write WG uint64 into lo/hi output slots           */
/* ------------------------------------------------------------------ */
static void write_u64(device uint *lo_buf, device uint *hi_buf,
                      uint idx, ulong val)
{
    lo_buf[idx] = (uint)(val & 0xFFFFFFFFuL);
    hi_buf[idx] = (uint)(val >> 32uL);
}

/* ------------------------------------------------------------------ */
/*  8 bpc kernel                                                        */
/* ------------------------------------------------------------------ */
kernel void float_moment_kernel_8bpc(
    const device uchar  *ref    [[buffer(0)]],
    const device uchar  *dis    [[buffer(1)]],
    device       uint   *r1_lo  [[buffer(2)]],
    device       uint   *r1_hi  [[buffer(3)]],
    device       uint   *d1_lo  [[buffer(4)]],
    device       uint   *d1_hi  [[buffer(5)]],
    device       uint   *r2_lo  [[buffer(6)]],
    device       uint   *r2_hi  [[buffer(7)]],
    device       uint   *d2_lo  [[buffer(8)]],
    device       uint   *d2_hi  [[buffer(9)]],
    constant     uint2  &strides [[buffer(10)]],
    constant     uint2  &dim     [[buffer(11)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  grid_groups [[threadgroups_per_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const int width  = (int)dim.x;
    const int height = (int)dim.y;

    /* Per-thread integer accumulators (raw pixel values, no float). */
    ulong my_r1 = 0uL, my_d1 = 0uL, my_r2 = 0uL, my_d2 = 0uL;
    if ((int)gid.x < width && (int)gid.y < height) {
        const ulong rv = (ulong)ref[(int)gid.y * (int)strides.x + (int)gid.x];
        const ulong dv = (ulong)dis[(int)gid.y * (int)strides.y + (int)gid.x];
        my_r1 = rv;
        my_d1 = dv;
        my_r2 = rv * rv;
        my_d2 = dv * dv;
    }

    /*
     * Two-level reduction: SIMD group then threadgroup.
     * Each accumulator is split into lo/hi uint32 to stay within
     * simd_sum(uint) range (MSL lacks simd_sum for ulong).
     *
     * 8bpc max per pixel: r1/d1 ≤ 255, r2/d2 ≤ 65025.
     * Per SIMD group (32 threads): r2 ≤ 32 × 65025 ≈ 2.08M — fits uint32.
     * Per WG (up to 8 SIMD groups, 256 threads): r2 ≤ 256 × 65025 ≈ 16.6M
     * — also fits uint32, but we always reconstruct via uint64 for safety.
     */
    threadgroup uint sg_r1_lo[8], sg_r1_hi[8];
    threadgroup uint sg_d1_lo[8], sg_d1_hi[8];
    threadgroup uint sg_r2_lo[8], sg_r2_hi[8];
    threadgroup uint sg_d2_lo[8], sg_d2_hi[8];

    const uint lane_r1_lo = simd_sum((uint)(my_r1 & 0xFFFFFFFFuL));
    const uint lane_r1_hi = simd_sum((uint)(my_r1 >> 32uL));
    const uint lane_d1_lo = simd_sum((uint)(my_d1 & 0xFFFFFFFFuL));
    const uint lane_d1_hi = simd_sum((uint)(my_d1 >> 32uL));
    const uint lane_r2_lo = simd_sum((uint)(my_r2 & 0xFFFFFFFFuL));
    const uint lane_r2_hi = simd_sum((uint)(my_r2 >> 32uL));
    const uint lane_d2_lo = simd_sum((uint)(my_d2 & 0xFFFFFFFFuL));
    const uint lane_d2_hi = simd_sum((uint)(my_d2 >> 32uL));

    if (simd_lane == 0) {
        sg_r1_lo[simd_id] = lane_r1_lo;
        sg_r1_hi[simd_id] = lane_r1_hi;
        sg_d1_lo[simd_id] = lane_d1_lo;
        sg_d1_hi[simd_id] = lane_d1_hi;
        sg_r2_lo[simd_id] = lane_r2_lo;
        sg_r2_hi[simd_id] = lane_r2_hi;
        sg_d2_lo[simd_id] = lane_d2_lo;
        sg_d2_hi[simd_id] = lane_d2_hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        ulong wg_r1 = 0uL, wg_d1 = 0uL, wg_r2 = 0uL, wg_d2 = 0uL;
        for (uint i = 0; i < simd_count; ++i) {
            wg_r1 += ((ulong)sg_r1_hi[i] << 32uL) | (ulong)sg_r1_lo[i];
            wg_d1 += ((ulong)sg_d1_hi[i] << 32uL) | (ulong)sg_d1_lo[i];
            wg_r2 += ((ulong)sg_r2_hi[i] << 32uL) | (ulong)sg_r2_lo[i];
            wg_d2 += ((ulong)sg_d2_hi[i] << 32uL) | (ulong)sg_d2_lo[i];
        }
        const uint idx = bid.y * grid_groups.x + bid.x;
        write_u64(r1_lo, r1_hi, idx, wg_r1);
        write_u64(d1_lo, d1_hi, idx, wg_d1);
        write_u64(r2_lo, r2_hi, idx, wg_r2);
        write_u64(d2_lo, d2_hi, idx, wg_d2);
    }
}

/* ------------------------------------------------------------------ */
/*  16 bpc kernel                                                       */
/* ------------------------------------------------------------------ */
kernel void float_moment_kernel_16bpc(
    const device uchar  *ref    [[buffer(0)]],
    const device uchar  *dis    [[buffer(1)]],
    device       uint   *r1_lo  [[buffer(2)]],
    device       uint   *r1_hi  [[buffer(3)]],
    device       uint   *d1_lo  [[buffer(4)]],
    device       uint   *d1_hi  [[buffer(5)]],
    device       uint   *r2_lo  [[buffer(6)]],
    device       uint   *r2_hi  [[buffer(7)]],
    device       uint   *d2_lo  [[buffer(8)]],
    device       uint   *d2_hi  [[buffer(9)]],
    constant     uint4  &strides [[buffer(10)]],
    constant     uint2  &dim     [[buffer(11)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  grid_groups [[threadgroups_per_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const int width  = (int)dim.x;
    const int height = (int)dim.y;

    /*
     * 16bpc: accumulate raw integer pixel values; host divides by scaler.
     * Max raw pixel = 65535 (10bpc: 1023, 12bpc: 4095).
     * r2 per pixel: 65535^2 = 4.29e9 — overflows uint32 per SIMD group
     * (32 * 4.29e9 = 1.37e11 > 2^32). Must use ulong in both SIMD lanes
     * and WG accumulation — achieved via lo/hi split before simd_sum.
     */
    ulong my_r1 = 0uL, my_d1 = 0uL, my_r2 = 0uL, my_d2 = 0uL;
    if ((int)gid.x < width && (int)gid.y < height) {
        const device ushort *ref_row =
            (const device ushort *)(ref + (int)gid.y * (int)strides.x);
        const device ushort *dis_row =
            (const device ushort *)(dis + (int)gid.y * (int)strides.y);
        const ulong rv = (ulong)ref_row[(int)gid.x];
        const ulong dv = (ulong)dis_row[(int)gid.x];
        my_r1 = rv;
        my_d1 = dv;
        my_r2 = rv * rv;
        my_d2 = dv * dv;
    }

    threadgroup uint sg_r1_lo[8], sg_r1_hi[8];
    threadgroup uint sg_d1_lo[8], sg_d1_hi[8];
    threadgroup uint sg_r2_lo[8], sg_r2_hi[8];
    threadgroup uint sg_d2_lo[8], sg_d2_hi[8];

    const uint lane_r1_lo = simd_sum((uint)(my_r1 & 0xFFFFFFFFuL));
    const uint lane_r1_hi = simd_sum((uint)(my_r1 >> 32uL));
    const uint lane_d1_lo = simd_sum((uint)(my_d1 & 0xFFFFFFFFuL));
    const uint lane_d1_hi = simd_sum((uint)(my_d1 >> 32uL));
    const uint lane_r2_lo = simd_sum((uint)(my_r2 & 0xFFFFFFFFuL));
    const uint lane_r2_hi = simd_sum((uint)(my_r2 >> 32uL));
    const uint lane_d2_lo = simd_sum((uint)(my_d2 & 0xFFFFFFFFuL));
    const uint lane_d2_hi = simd_sum((uint)(my_d2 >> 32uL));

    if (simd_lane == 0) {
        sg_r1_lo[simd_id] = lane_r1_lo;
        sg_r1_hi[simd_id] = lane_r1_hi;
        sg_d1_lo[simd_id] = lane_d1_lo;
        sg_d1_hi[simd_id] = lane_d1_hi;
        sg_r2_lo[simd_id] = lane_r2_lo;
        sg_r2_hi[simd_id] = lane_r2_hi;
        sg_d2_lo[simd_id] = lane_d2_lo;
        sg_d2_hi[simd_id] = lane_d2_hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        ulong wg_r1 = 0uL, wg_d1 = 0uL, wg_r2 = 0uL, wg_d2 = 0uL;
        for (uint i = 0; i < simd_count; ++i) {
            wg_r1 += ((ulong)sg_r1_hi[i] << 32uL) | (ulong)sg_r1_lo[i];
            wg_d1 += ((ulong)sg_d1_hi[i] << 32uL) | (ulong)sg_d1_lo[i];
            wg_r2 += ((ulong)sg_r2_hi[i] << 32uL) | (ulong)sg_r2_lo[i];
            wg_d2 += ((ulong)sg_d2_hi[i] << 32uL) | (ulong)sg_d2_lo[i];
        }
        const uint idx = bid.y * grid_groups.x + bid.x;
        write_u64(r1_lo, r1_hi, idx, wg_r1);
        write_u64(d1_lo, d1_hi, idx, wg_d1);
        write_u64(r2_lo, r2_hi, idx, wg_r2);
        write_u64(d2_lo, d2_hi, idx, wg_d2);
    }
}
