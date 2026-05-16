/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for float_vif (T8-1k / ADR-0462).
 *  Port of the CUDA twin at `cuda/float_vif/float_vif_score.cu`.
 *
 *  Algorithm overview
 *  ------------------
 *  VIF is computed across 4 dyadic scales.  Each scale uses a
 *  scale-specific Gaussian filter (widths 17, 9, 5, 3) with mirror
 *  padding at frame borders (`VIF_OPT_HANDLE_BORDERS`).
 *
 *  Per scale the pipeline is two passes:
 *
 *    Pass A — `float_vif_compute`:
 *      Separable V → H Gaussian filter on ref + dis.  Accumulates five
 *      per-pixel statistics:
 *        mu1, mu2       — filtered ref, filtered dis
 *        xx, yy, xy     — filtered ref², dis², ref*dis
 *      Then applies the VIF statistic per pixel and reduces (num, den)
 *      to one float partial per threadgroup.
 *
 *    Pass B — `float_vif_decimate`:
 *      For scales 0–2, apply this scale's filter at the full input
 *      dimensions and sample at (2*gx, 2*gy), producing the
 *      half-resolution input for the next scale.  Scale 3 skips Pass B
 *      (the host does not dispatch it for scale 3).
 *
 *  Buffer layout for `float_vif_compute` (scale > 0):
 *   [[buffer(0)]] ref_f   — float * (current scale, row-major, W × H)
 *   [[buffer(1)]] dis_f   — float * (current scale)
 *   [[buffer(2)]] partials — float * (grid_w × grid_h; two floats per
 *                            threadgroup: [num, den] interleaved as
 *                            partials[2*(bid.y*grid_w+bid.x) + 0/1])
 *   [[buffer(3)]] params  — uint4 (.x=W, .y=H, .z=bpc, .w=grid_w)
 *   [[buffer(4)]] consts  — float4 (.x=sigma_nsq, .y=enhn_gain_limit,
 *                            .z=sigma_max_inv, .w=scale_idx)
 *
 *  Buffer layout for scale-0 `float_vif_compute`:
 *   Identical, but .z of params carries bpc and the kernel reads
 *   uint8/uint16 pixels inline and converts to float in [-128, peak-128].
 *   ref_f / dis_f are unused (raw plane pointer convention: the host
 *   passes the raw uint8/uint16 plane cast to float* — only the lower
 *   bits are meaningful at bpc ≤ 8; for bpc > 8 the uint16 elements
 *   are reinterpreted from the float* pointer).
 *   To keep MSL clean we pass a fifth "is_raw" parameter instead and
 *   use [[buffer(5)]] raw_ref / [[buffer(6)]] raw_dis as uchar* with
 *   [[buffer(7)]] raw_stride (uint).  Scale > 0 ignores buffers 5–7.
 *
 *  Buffer layout for `float_vif_decimate`:
 *   [[buffer(0)]] ref_in  — float * (current scale input, or raw for s0)
 *   [[buffer(1)]] dis_in  — float * (current scale input)
 *   [[buffer(2)]] ref_out — float * (half-res output for next scale)
 *   [[buffer(3)]] dis_out — float * (half-res output)
 *   [[buffer(4)]] params  — uint4 (.x=out_w, .y=out_h,
 *                                   .z=in_w,  .w=in_h)
 *   [[buffer(5)]] consts  — uint4 (.x=bpc, .y=scale_idx, .z=0, .w=0)
 *   [[buffer(6)]] raw_ref — uchar * (scale-0 raw plane; ignored s>0)
 *   [[buffer(7)]] raw_dis — uchar * (scale-0 raw plane; ignored s>0)
 *   [[buffer(8)]] raw_params — uint4 (.x=stride_bytes, others 0)
 *
 *  VIF statistic (`matching_matlab` mode, matches vif_tools.c)
 *  -----------------------------------------------------------
 *    sigma1_sq = E[ref²] - mu1²      (clamped ≥ 0)
 *    sigma2_sq = E[dis²] - mu2²      (clamped ≥ 0)
 *    sigma12   = E[ref*dis] - mu1*mu2
 *    eps       = 1e-10
 *
 *    g     = sigma12 / (sigma1_sq + eps)
 *    sv_sq = sigma2_sq - g * sigma12
 *    (various corner-case clamps on g and sv_sq)
 *    g     = min(g, vif_enhn_gain_limit)
 *
 *    if sigma1_sq >= sigma_nsq:
 *      num_val = log2(1 + g²·sigma1_sq / (sv_sq + sigma_nsq))
 *      den_val = log2(1 + sigma1_sq / sigma_nsq)
 *      if sigma12 < 0: num_val = 0
 *    else:
 *      num_val = 1 - sigma2_sq * sigma_max_inv
 *      den_val = 1
 *
 *  Threadgroup: 16 × 16 = 256 threads.
 *  Grid for Pass A: ceil(W/16) × ceil(H/16).
 *  Grid for Pass B: ceil(out_w/16) × ceil(out_h/16).
 *
 *  Per-WG float partial: two floats [num, den] interleaved.
 *  Host sums partials over the grid and writes per-scale scores.
 *
 *  Filter coefficients (normalised Gaussian, matches vif_tools.c):
 *    Scale 0 (fw=17): FVIF_COEFF_S0
 *    Scale 1 (fw=9):  FVIF_COEFF_S1
 *    Scale 2 (fw=5):  FVIF_COEFF_S2
 *    Scale 3 (fw=3):  FVIF_COEFF_S3
 */

#include <metal_stdlib>
using namespace metal;

/* Per-scale Gaussian filter coefficients (matches float_vif_score.cu). */
constant float FVIF_COEFF_S0[17] = {
    0.00745626912f, 0.0142655009f, 0.0250313189f, 0.0402820669f,
    0.0594526194f,  0.0804751068f, 0.0999041125f, 0.113746084f,
    0.118773937f,   0.113746084f,  0.0999041125f, 0.0804751068f,
    0.0594526194f,  0.0402820669f, 0.0250313189f, 0.0142655009f,
    0.00745626912f
};
constant float FVIF_COEFF_S1[9] = {
    0.0189780835f, 0.0558981746f, 0.120920904f, 0.192116052f,
    0.224173605f,  0.192116052f,  0.120920904f, 0.0558981746f,
    0.0189780835f
};
constant float FVIF_COEFF_S2[5] = {
    0.054488685f, 0.244201347f, 0.402619958f, 0.244201347f, 0.054488685f
};
constant float FVIF_COEFF_S3[3] = {
    0.166378498f, 0.667243004f, 0.166378498f
};

/* Mirror-pad index (matches CUDA fvif_mirror_v / fvif_mirror_h). */
static inline int fvif_mirror(int idx, int sup)
{
    if (idx < 0)          { return -idx; }
    if (idx >= sup)       { return 2 * sup - idx - 2; }
    return idx;
}

/* Read a coefficient for a given scale and tap index. */
static inline float fvif_coeff(uint scale_idx, int k)
{
    if (scale_idx == 0u) { return FVIF_COEFF_S0[k]; }
    if (scale_idx == 1u) { return FVIF_COEFF_S1[k]; }
    if (scale_idx == 2u) { return FVIF_COEFF_S2[k]; }
    return FVIF_COEFF_S3[k];
}

static inline int fvif_fw(uint scale_idx)
{
    if (scale_idx == 0u) { return 17; }
    if (scale_idx == 1u) { return 9;  }
    if (scale_idx == 2u) { return 5;  }
    return 3;
}

/* -----------------------------------------------------------------------
 *  Shared-tile dimensions: worst-case at scale 0 (fw=17, hfw=8).
 *  Tile = 16 + 2*8 = 32 in each dimension.
 *  We allocate 32*32 = 1024 for both ref and dis.
 * --------------------------------------------------------------------- */
#define FVIF_TG    16
#define FVIF_HALO  8   /* max half-filter-width (scale 0, fw=17, hfw=8) */
#define FVIF_TILE  (FVIF_TG + 2 * FVIF_HALO)  /* 32 */

/* -----------------------------------------------------------------------
 *  Pass A — compute: separable filter + VIF statistic + per-WG reduction.
 * --------------------------------------------------------------------- */
kernel void float_vif_compute(
    const device float   *ref_f      [[buffer(0)]],
    const device float   *dis_f      [[buffer(1)]],
    device       float   *partials   [[buffer(2)]],
    constant     uint4   &params     [[buffer(3)]],
    constant     float4  &consts     [[buffer(4)]],
    /* Scale-0 raw planes (uint8/uint16 packed); ignored for scale > 0. */
    const device uchar   *raw_ref    [[buffer(5)]],
    const device uchar   *raw_dis    [[buffer(6)]],
    constant     uint4   &raw_params [[buffer(7)]],

    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  lid2        [[thread_position_in_threadgroup]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const uint W         = params.x;
    const uint H         = params.y;
    const uint bpc       = params.z;
    const uint grid_w    = params.w;
    const uint scale_idx = uint(consts.w);
    const float sigma_nsq     = consts.x;
    const float enhn_gain_lim = consts.y;
    const float sigma_max_inv = consts.z;

    const int fw  = fvif_fw(scale_idx);
    const int hfw = fw / 2;

    /* ---------------------------------------------------------------
     * Shared-memory tile: ref + dis (32×32 each).
     * We declare worst-case size and only use the relevant sub-tile.
     * ------------------------------------------------------------- */
    threadgroup float s_ref[FVIF_TILE * FVIF_TILE];
    threadgroup float s_dis[FVIF_TILE * FVIF_TILE];

    /* Vertical-pass intermediates for the 16 output rows × tile_w cols. */
    const int tile_w = FVIF_TG + 2 * hfw;
    const int tile_h = FVIF_TG + 2 * hfw;

    threadgroup float s_v_mu1[FVIF_TG * FVIF_TILE];
    threadgroup float s_v_mu2[FVIF_TG * FVIF_TILE];
    threadgroup float s_v_xx [FVIF_TG * FVIF_TILE];
    threadgroup float s_v_yy [FVIF_TG * FVIF_TILE];
    threadgroup float s_v_xy [FVIF_TG * FVIF_TILE];

    const uint lx = lid2.x;
    const uint ly = lid2.y;
    const uint wg_size = FVIF_TG * FVIF_TG;

    /* ---------------------------------------------------------------
     * Phase 1: cooperative tile load with mirror padding.
     * ------------------------------------------------------------- */
    const int tile_oy = int(bid.y) * FVIF_TG - hfw;
    const int tile_ox = int(bid.x) * FVIF_TG - hfw;
    const int tile_elems = tile_h * tile_w;
    const bool is_raw = (scale_idx == 0u);
    const uint raw_stride = raw_params.x;

    for (uint i = lid; i < uint(tile_elems); i += wg_size) {
        const int tr = int(i) / tile_w;
        const int tc = int(i) - tr * tile_w;
        const int py = fvif_mirror(tile_oy + tr, int(H));
        const int px = fvif_mirror(tile_ox + tc, int(W));

        float r, d;
        if (is_raw) {
            if (bpc <= 8u) {
                r = float(raw_ref[py * int(raw_stride) + px]) - 128.0f;
                d = float(raw_dis[py * int(raw_stride) + px]) - 128.0f;
            } else {
                float scaler = (bpc == 10u) ? 4.0f : ((bpc == 12u) ? 16.0f : 256.0f);
                const device ushort *rr = (const device ushort *)(raw_ref
                    + py * int(raw_stride));
                const device ushort *dr = (const device ushort *)(raw_dis
                    + py * int(raw_stride));
                r = float(rr[px]) / scaler - 128.0f;
                d = float(dr[px]) / scaler - 128.0f;
            }
        } else {
            r = ref_f[py * int(W) + px];
            d = dis_f[py * int(W) + px];
        }
        s_ref[tr * FVIF_TILE + tc] = r;
        s_dis[tr * FVIF_TILE + tc] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---------------------------------------------------------------
     * Phase 2: vertical filter → s_v_mu1/mu2/xx/yy/xy
     * Each thread handles one (col × row) output cell of the vert pass.
     * Total cells = FVIF_TG × tile_w; spread across wg_size threads.
     * ------------------------------------------------------------- */
    const int vert_total = FVIF_TG * tile_w;
    for (uint i = lid; i < uint(vert_total); i += wg_size) {
        const int r = int(i) / tile_w;
        const int c = int(i) - r * tile_w;
        float a_mu1 = 0.0f, a_mu2 = 0.0f;
        float a_xx  = 0.0f, a_yy  = 0.0f, a_xy = 0.0f;
        for (int k = 0; k < fw; ++k) {
            const float ck    = fvif_coeff(scale_idx, k);
            const float ref_v = s_ref[(r + k) * FVIF_TILE + c];
            const float dis_v = s_dis[(r + k) * FVIF_TILE + c];
            a_mu1 += ck * ref_v;
            a_mu2 += ck * dis_v;
            a_xx  += ck * (ref_v * ref_v);
            a_yy  += ck * (dis_v * dis_v);
            a_xy  += ck * (ref_v * dis_v);
        }
        s_v_mu1[r * FVIF_TILE + c] = a_mu1;
        s_v_mu2[r * FVIF_TILE + c] = a_mu2;
        s_v_xx [r * FVIF_TILE + c] = a_xx;
        s_v_yy [r * FVIF_TILE + c] = a_yy;
        s_v_xy [r * FVIF_TILE + c] = a_xy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---------------------------------------------------------------
     * Phase 3: horizontal filter + VIF statistic.
     * ------------------------------------------------------------- */
    const bool valid = (gid.x < W && gid.y < H);
    float my_num = 0.0f, my_den = 0.0f;

    if (valid) {
        float mu1 = 0.0f, mu2 = 0.0f;
        float xx  = 0.0f, yy  = 0.0f, xy = 0.0f;
        for (int k = 0; k < fw; ++k) {
            const float ck = fvif_coeff(scale_idx, k);
            mu1 += ck * s_v_mu1[ly * FVIF_TILE + (lx + k)];
            mu2 += ck * s_v_mu2[ly * FVIF_TILE + (lx + k)];
            xx  += ck * s_v_xx [ly * FVIF_TILE + (lx + k)];
            yy  += ck * s_v_yy [ly * FVIF_TILE + (lx + k)];
            xy  += ck * s_v_xy [ly * FVIF_TILE + (lx + k)];
        }

        const float eps = 1.0e-10f;
        float sigma1_sq = xx - mu1 * mu1;
        float sigma2_sq = yy - mu2 * mu2;
        float sigma12   = xy - mu1 * mu2;
        sigma1_sq = max(sigma1_sq, 0.0f);
        sigma2_sq = max(sigma2_sq, 0.0f);

        float g     = sigma12 / (sigma1_sq + eps);
        float sv_sq = sigma2_sq - g * sigma12;

        if (sigma1_sq < eps) {
            g        = 0.0f;
            sv_sq    = sigma2_sq;
            sigma1_sq = 0.0f;
        }
        if (sigma2_sq < eps) {
            g     = 0.0f;
            sv_sq = 0.0f;
        }
        if (g < 0.0f) {
            sv_sq = sigma2_sq;
            g     = 0.0f;
        }
        sv_sq = max(sv_sq, eps);
        g     = min(g, enhn_gain_lim);

        if (sigma1_sq >= sigma_nsq) {
            my_num = log2(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
            my_den = log2(1.0f + sigma1_sq / sigma_nsq);
            if (sigma12 < 0.0f) { my_num = 0.0f; }
        } else {
            my_num = 1.0f - sigma2_sq * sigma_max_inv;
            my_den = 1.0f;
        }
    }

    /* ---------------------------------------------------------------
     * Phase 4: SIMD + cross-SIMD reduction → per-WG partials.
     * ------------------------------------------------------------- */
    threadgroup float sg_num[8];
    threadgroup float sg_den[8];
    const float lane_num = simd_sum(my_num);
    const float lane_den = simd_sum(my_den);
    if (simd_lane == 0u) {
        sg_num[simd_id] = lane_num;
        sg_den[simd_id] = lane_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        float gn = 0.0f, gd = 0.0f;
        for (uint i = 0u; i < simd_count; ++i) {
            gn += sg_num[i];
            gd += sg_den[i];
        }
        const uint slot = bid.y * grid_w + bid.x;
        partials[2u * slot]      = gn;
        partials[2u * slot + 1u] = gd;
    }
}

/* -----------------------------------------------------------------------
 *  Pass B — decimate: apply scale's filter at full input dims, sample
 *  at (2*gx, 2*gy) → half-res output.  Mirror padding at borders.
 * --------------------------------------------------------------------- */
kernel void float_vif_decimate(
    const device float   *ref_in    [[buffer(0)]],
    const device float   *dis_in    [[buffer(1)]],
    device       float   *ref_out   [[buffer(2)]],
    device       float   *dis_out   [[buffer(3)]],
    constant     uint4   &params    [[buffer(4)]],
    constant     uint4   &consts    [[buffer(5)]],
    const device uchar   *raw_ref   [[buffer(6)]],
    const device uchar   *raw_dis   [[buffer(7)]],
    constant     uint4   &raw_params[[buffer(8)]],

    uint2  gid [[thread_position_in_grid]])
{
    const uint out_w     = params.x;
    const uint out_h     = params.y;
    const uint in_w      = params.z;
    const uint in_h      = params.w;
    const uint bpc       = consts.x;
    const uint scale_idx = consts.y;
    const uint raw_stride = raw_params.x;

    if (gid.x >= out_w || gid.y >= out_h) { return; }

    const int fw  = fvif_fw(scale_idx);
    const int hfw = fw / 2;
    const int in_x = int(gid.x) * 2;
    const int in_y = int(gid.y) * 2;
    const bool is_raw = (scale_idx == 1u);  /* decimate called with scale=1 reads scale-0 raw */

    /* V-inner / H-outer: matches vif_filter1d_s loop ordering. */
    float acc_ref = 0.0f, acc_dis = 0.0f;
    for (int kj = 0; kj < fw; ++kj) {
        const float cj = fvif_coeff(scale_idx, kj);
        const int px   = fvif_mirror(in_x - hfw + kj, int(in_w));
        float v_ref = 0.0f, v_dis = 0.0f;
        for (int ki = 0; ki < fw; ++ki) {
            const float ci = fvif_coeff(scale_idx, ki);
            const int py   = fvif_mirror(in_y - hfw + ki, int(in_h));
            float r, d;
            if (is_raw) {
                if (bpc <= 8u) {
                    r = float(raw_ref[py * int(raw_stride) + px]) - 128.0f;
                    d = float(raw_dis[py * int(raw_stride) + px]) - 128.0f;
                } else {
                    float scaler = (bpc == 10u) ? 4.0f :
                                   ((bpc == 12u) ? 16.0f : 256.0f);
                    const device ushort *rr =
                        (const device ushort *)(raw_ref + py * int(raw_stride));
                    const device ushort *dr =
                        (const device ushort *)(raw_dis + py * int(raw_stride));
                    r = float(rr[px]) / scaler - 128.0f;
                    d = float(dr[px]) / scaler - 128.0f;
                }
            } else {
                r = ref_in[py * int(in_w) + px];
                d = dis_in[py * int(in_w) + px];
            }
            v_ref += ci * r;
            v_dis += ci * d;
        }
        acc_ref += cj * v_ref;
        acc_dis += cj * v_dis;
    }

    ref_out[gid.y * out_w + gid.x] = acc_ref;
    dis_out[gid.y * out_w + gid.x] = acc_dis;
}
