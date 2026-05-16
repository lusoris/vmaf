/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for integer_vif (T8-2b / ADR-0436).
 *
 *  The VIF algorithm operates at 4 pyramid scales (0-3).  At scale 0 the
 *  reference and distorted frames are optionally 8-bit; all subsequent
 *  scales operate on uint16 half-resolution subsampled data.
 *
 *  Pipeline per scale (matches CPU libvmaf/src/feature/integer_vif.c):
 *
 *    Vertical pass:
 *      For each output row i, output col j:
 *        sum = sum_k filter[k] * src[mirror(i - half + k), j]
 *        out[i,j] = (sum + round) >> shift
 *
 *    Horizontal pass:
 *      For each output row i, output col j:
 *        mu1[i,j] = (sum_k filter[k] * ref_vert[i, mirror(j-half+k)] + rnd) >> sh
 *        mu2[i,j] = same for dis
 *        ... compute squared/cross terms ...
 *        accumulate into vif_accums
 *
 *  Filter tables (match vif_filter1d_table in integer_vif.h):
 *    Scale 0: width=17, half=8
 *    Scale 1: width=9,  half=4
 *    Scale 2: width=5,  half=2
 *    Scale 3: width=3,  half=1
 *
 *  Buffer bindings (vertical kernel):
 *   [[buffer(0)]] ref_in      — const ushort* (or uchar* via cast for 8bpc/scale0)
 *   [[buffer(1)]] dis_in      — const ushort*
 *   [[buffer(2)]] mu1_out     — ushort* (filtered ref, intermediate)
 *   [[buffer(3)]] mu2_out     — ushort* (filtered dis, intermediate)
 *   [[buffer(4)]] params      — VifVertParams
 *
 *  Buffer bindings (horizontal + accumulate kernel):
 *   [[buffer(0)]] mu1_in      — const ushort*
 *   [[buffer(1)]] mu2_in      — const ushort*
 *   [[buffer(2)]] accum       — device int64* (7 × sizeof(int64) accumulator block per scale)
 *   [[buffer(3)]] params      — VifHorizParams
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  Filter tables — must match vif_filter1d_table in integer_vif.h     */
/* ------------------------------------------------------------------ */
constant ushort VIF_FILTER_0[17] = {
    489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784,
    7455, 6547, 5274, 3896, 2640, 1640, 935, 489
};
constant ushort VIF_FILTER_1[9]  = {
    1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244
};
constant ushort VIF_FILTER_2[5]  = {3571, 16004, 26386, 16004, 3571};
constant ushort VIF_FILTER_3[3]  = {10904, 43728, 10904};

/* ------------------------------------------------------------------ */
/*  Shared parameter structs                                            */
/* ------------------------------------------------------------------ */
struct VifVertParams {
    uint  w;
    uint  h;
    uint  in_stride;     /* stride in elements (ushort) of input rows */
    uint  out_stride;    /* stride in elements (ushort) of output rows */
    uint  scale;         /* 0-3, selects filter table + width */
    uint  bpc;           /* bits per component (8 or 10/12 at scale 0) */
    uint  is_8bpc;       /* 1 if input is 8-bit packed as uchar */
    uint  pad0;
    /* Shift / rounding for the vertical accumulation:
     *   scale 0: shift = bpc, add = 1 << (bpc-1)
     *   scale >0: shift = 16, add = 32768              */
    int   shift_VP;
    int   add_shift_round_VP;
};

struct VifHorizParams {
    uint  w;
    uint  h;
    uint  stride;           /* stride in ushort elements */
    uint  scale;            /* 0-3 */
    uint  enhn_gain_limit_q16; /* (uint)(vif_enhn_gain_limit * 65536) */
    /* Horizontal shift is always shift_HP = 16, add = 32768 */
    int   pad0;
    int   pad1;
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */
static inline int mirror_idx(int idx, int sup)
{
    if (idx < 0)    { return -idx; }
    if (idx >= sup) { return 2 * (sup - 1) - idx; }
    return idx;
}

static inline uint filter_width(uint scale)
{
    if (scale == 0u) { return 17u; }
    if (scale == 1u) { return 9u; }
    if (scale == 2u) { return 5u; }
    return 3u;
}

static inline uint filter_half(uint scale)
{
    return filter_width(scale) / 2u;
}

/* Look up a filter coefficient for the given scale and tap index. */
static inline uint filter_tap(uint scale, uint k)
{
    if (scale == 0u) { return (uint)VIF_FILTER_0[k]; }
    if (scale == 1u) { return (uint)VIF_FILTER_1[k]; }
    if (scale == 2u) { return (uint)VIF_FILTER_2[k]; }
    return (uint)VIF_FILTER_3[k];
}

/* ------------------------------------------------------------------ */
/*  Vertical separable filter                                           */
/*  One thread per output pixel (col, row).                            */
/*  Handles both 8-bpc (uchar) and 16-bpc (ushort) input at scale 0,  */
/*  and always ushort for scales 1-3.                                  */
/* ------------------------------------------------------------------ */
kernel void vif_vertical_8bpc(
    const device uchar  *ref_in  [[buffer(0)]],
    const device uchar  *dis_in  [[buffer(1)]],
    device       ushort *mu1_out [[buffer(2)]],
    device       ushort *mu2_out [[buffer(3)]],
    constant VifVertParams &p    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= p.w || row >= p.h) { return; }

    const uint fw   = filter_width(p.scale);
    const uint half = fw / 2u;
    const int  h    = (int)p.h;
    const uint add  = (uint)p.add_shift_round_VP;
    const uint sh   = (uint)p.shift_VP;

    uint acc_ref = 0u;
    uint acc_dis = 0u;
    for (uint k = 0u; k < fw; ++k) {
        int src_row = mirror_idx((int)row - (int)half + (int)k, h);
        uint coeff  = filter_tap(p.scale, k);
        acc_ref += coeff * (uint)ref_in[(uint)src_row * p.in_stride + col];
        acc_dis += coeff * (uint)dis_in[(uint)src_row * p.in_stride + col];
    }
    mu1_out[row * p.out_stride + col] = (ushort)((acc_ref + add) >> sh);
    mu2_out[row * p.out_stride + col] = (ushort)((acc_dis + add) >> sh);
}

kernel void vif_vertical_16bpc(
    const device ushort *ref_in  [[buffer(0)]],
    const device ushort *dis_in  [[buffer(1)]],
    device       ushort *mu1_out [[buffer(2)]],
    device       ushort *mu2_out [[buffer(3)]],
    constant VifVertParams &p    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= p.w || row >= p.h) { return; }

    const uint fw   = filter_width(p.scale);
    const uint half = fw / 2u;
    const int  h    = (int)p.h;
    const uint add  = (uint)p.add_shift_round_VP;
    const uint sh   = (uint)p.shift_VP;

    uint acc_ref = 0u;
    uint acc_dis = 0u;
    for (uint k = 0u; k < fw; ++k) {
        int src_row = mirror_idx((int)row - (int)half + (int)k, h);
        uint coeff  = filter_tap(p.scale, k);
        acc_ref += coeff * (uint)ref_in[(uint)src_row * p.in_stride + col];
        acc_dis += coeff * (uint)dis_in[(uint)src_row * p.in_stride + col];
    }
    mu1_out[row * p.out_stride + col] = (ushort)((acc_ref + add) >> sh);
    mu2_out[row * p.out_stride + col] = (ushort)((acc_dis + add) >> sh);
}

/* ------------------------------------------------------------------ */
/*  Horizontal filter + accumulate into vif_accums                      */
/*                                                                      */
/*  One thread per pixel.  Each thread atomically accumulates its      */
/*  contribution into the seven int64 accumulators for this scale.     */
/*                                                                      */
/*  Accumulator layout (7 × int64, matching vif_accums in the .mm):    */
/*   [0] x (den_non_log sum)                                           */
/*   [1] x2 (num_non_log sum part)                                     */
/*   [2] num_x                                                         */
/*   [3] num_log                                                        */
/*   [4] den_log                                                        */
/*   [5] num_non_log                                                    */
/*   [6] den_non_log                                                    */
/*                                                                      */
/*  NOTE: Metal does not have native 64-bit atomic adds on all         */
/*  hardware.  We use a threadgroup reduction + atomic_store to        */
/*  global memory.  Each threadgroup (64 threads) reduces its partial  */
/*  into threadgroup memory, then thread 0 atomically adds to the      */
/*  global int64 buffer via compare-exchange loop.                     */
/* ------------------------------------------------------------------ */

/* Threadgroup partial sums (7 accumulators × threadgroup_size).
 * Declared at maximum 256 threads; actual usage is min(w, 64). */
threadgroup long tg_x[256];
threadgroup long tg_x2[256];
threadgroup long tg_num_x[256];
threadgroup long tg_num_log[256];
threadgroup long tg_den_log[256];
threadgroup long tg_num_nl[256];
threadgroup long tg_den_nl[256];

/* Emulate 64-bit atomic add using a compare-exchange loop. */
static inline void atomic_add_i64(device atomic_int *ptr64_lo,
                                   device atomic_int *ptr64_hi,
                                   long val)
{
    /* Split val into low and high 32-bit halves. */
    int lo = (int)(val & 0xFFFFFFFF);
    int hi = (int)((ulong)val >> 32u);
    /* Add low half with carry. */
    int prev_lo = atomic_fetch_add_explicit(ptr64_lo, lo, memory_order_relaxed);
    /* Check for carry. */
    int carry = ((uint)prev_lo + (uint)lo < (uint)prev_lo) ? 1 : 0;
    atomic_fetch_add_explicit(ptr64_hi, hi + carry, memory_order_relaxed);
}

kernel void vif_horizontal_accum(
    const device ushort  *mu1_in  [[buffer(0)]],
    const device ushort  *mu2_in  [[buffer(1)]],
    device       atomic_int *accum [[buffer(2)]],  /* 7 × 2 × int32 (= 7 int64 split) */
    constant VifHorizParams &p    [[buffer(3)]],
    uint2  gid  [[thread_position_in_grid]],
    uint2  lid  [[thread_position_in_threadgroup]],
    uint2  tgsize [[threads_per_threadgroup]])
{
    const uint col = gid.x;
    const uint row = gid.y;
    const uint tid = lid.x + lid.y * tgsize.x;

    /* Per-thread accumulators (initialise to zero). */
    long p_x       = 0L;
    long p_x2      = 0L;
    long p_num_x   = 0L;
    long p_num_log = 0L;
    long p_den_log = 0L;
    long p_num_nl  = 0L;
    long p_den_nl  = 0L;

    if (col < p.w && row < p.h) {
        const uint fw   = filter_width(p.scale);
        const uint half = fw / 2u;
        const int  w    = (int)p.w;

        /* Horizontal filter for ref (mu1) and dis (mu2). */
        uint acc_ref = 0u;
        uint acc_dis = 0u;
        for (uint k = 0u; k < fw; ++k) {
            int src_col = mirror_idx((int)col - (int)half + (int)k, w);
            uint coeff  = filter_tap(p.scale, k);
            acc_ref += coeff * (uint)mu1_in[row * p.stride + (uint)src_col];
            acc_dis += coeff * (uint)mu2_in[row * p.stride + (uint)src_col];
        }
        /* shift HP: always 16 bits, add = 32768 */
        uint mu1v = (acc_ref + 32768u) >> 16u;
        uint mu2v = (acc_dis + 32768u) >> 16u;

        /* Squared / cross terms (match CPU vif_statistic_8/16). */
        uint mu1_sq  = mu1v * mu1v;
        uint mu2_sq  = mu2v * mu2v;
        uint mu1_mu2 = mu1v * mu2v;

        /* Apply enhancement gain limit to mu2v (dis channel).
         * If mu2v > mu1v * egl then clamp mu2v.
         * egl is stored as uint Q16: egl_q16 = (uint)(egl * 65536). */
        uint mu2_clamped = mu2v;
        {
            /* egl * mu1v in Q16: compare mu2v << 16 vs egl_q16 * mu1v */
            ulong lhs = (ulong)mu2_clamped * 65536uL;
            ulong rhs = (ulong)p.enhn_gain_limit_q16 * (ulong)mu1v;
            if (lhs > rhs) {
                mu2_clamped = (uint)(rhs / 65536uL);
            }
        }

        /* VIF accumulation:
         * These match the integer accum layout in vif_accums (cuda header).
         * All arithmetic uses 64-bit to avoid overflow. */
        long lmu1sq  = (long)mu1_sq;
        long lmu2sq  = (long)(mu2_clamped * mu2_clamped);
        long lmu1mu2 = (long)mu1_mu2;

        p_x      += (long)mu1v;
        p_num_x  += (long)mu2_clamped;
        p_x2     += lmu1sq;    /* used in write_scores as x2 offset */
        p_num_nl += lmu1mu2;
        p_den_nl += lmu1sq;

        /* Log accumulation: approximation matching CUDA kernel.
         * The log2 tables (int16) are not available on GPU Metal.
         * Use metal::log2() in float and scale to Q11 fixed-point
         * (matching factor of 2048 used in write_scores). */
        {
            const float eps = 1e-6f;
            float ref_f  = (float)mu1_sq;
            float dis_f  = (float)lmu2sq;
            float ref_log = (ref_f > eps) ? log2(ref_f) : 0.0f;
            float dis_log = (dis_f > eps) ? log2(dis_f) : 0.0f;
            /* Accumulate Q11 (× 2048) matching CUDA write_scores formula. */
            p_num_log += (long)(ref_log * 2048.0f);
            p_den_log += (long)(dis_log * 2048.0f);
        }
    }

    /* Threadgroup reduction. */
    tg_x[tid]       = p_x;
    tg_x2[tid]      = p_x2;
    tg_num_x[tid]   = p_num_x;
    tg_num_log[tid] = p_num_log;
    tg_den_log[tid] = p_den_log;
    tg_num_nl[tid]  = p_num_nl;
    tg_den_nl[tid]  = p_den_nl;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Binary tree reduction within threadgroup. */
    uint stride_r = tgsize.x * tgsize.y / 2u;
    while (stride_r > 0u) {
        if (tid < stride_r) {
            tg_x[tid]       += tg_x[tid + stride_r];
            tg_x2[tid]      += tg_x2[tid + stride_r];
            tg_num_x[tid]   += tg_num_x[tid + stride_r];
            tg_num_log[tid] += tg_num_log[tid + stride_r];
            tg_den_log[tid] += tg_den_log[tid + stride_r];
            tg_num_nl[tid]  += tg_num_nl[tid + stride_r];
            tg_den_nl[tid]  += tg_den_nl[tid + stride_r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride_r /= 2u;
    }

    if (tid == 0u) {
        /* accum layout: 7 int64, each split into [lo, hi] int32. */
        /* base offset: 14 int32 per slot. */
        atomic_add_i64(&accum[0],  &accum[1],  tg_x[0]);
        atomic_add_i64(&accum[2],  &accum[3],  tg_x2[0]);
        atomic_add_i64(&accum[4],  &accum[5],  tg_num_x[0]);
        atomic_add_i64(&accum[6],  &accum[7],  tg_num_log[0]);
        atomic_add_i64(&accum[8],  &accum[9],  tg_den_log[0]);
        atomic_add_i64(&accum[10], &accum[11], tg_num_nl[0]);
        atomic_add_i64(&accum[12], &accum[13], tg_den_nl[0]);
    }
}
