/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for float_adm (T8-2a / ADR-0424).
 *  Direct port of the CUDA kernels in feature/cuda/float_adm/float_adm_score.cu
 *  (ADR-0202 / ADR-0192). Four pipeline stages, identical semantics.
 *
 *  Stages:
 *    0 — float_adm_dwt_vert   : 2-D vertical DWT pass (ref+dis in z=0,1)
 *    1 — float_adm_dwt_hori   : 2-D horizontal DWT pass → 4 sub-bands
 *    2 — float_adm_decouple_csf : decouple + CSF weighting
 *    3 — float_adm_csf_cm     : CSF-den + CM fused row reduction
 *
 *  Buffer bindings are documented per kernel below.
 *  The host dispatch order (16 launches: 4 stages × 4 scales) is in
 *  the companion .mm file (float_adm_metal.mm).
 *
 *  Numerical invariants (places=4 contract):
 *    - Mirror form: same `2*sup - idx - 1` form as the CUDA twin.
 *    - Decouple parenthesisation matches the CUDA `__forceinline__`
 *      helper (angle-flag inner products use explicit paren groups to
 *      suppress FMA reordering — Metal's default is `fast-math` OFF for
 *      compute kernels, so the explicit order is sufficient).
 *    - CM threshold uses all 3 bands' 8-neighbours (cross-band aggregate
 *      matches CPU ADM_CM_THRESH_S_I_J macro).
 *    - Per-WG reduction via simd_sum (simd-width 32 on Apple Silicon).
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  Shared constants (wavelet filter, decouple, CM)                    */
/* ------------------------------------------------------------------ */

constant float FADM_LO0 =  0.482962913144690f;
constant float FADM_LO1 =  0.836516303737469f;
constant float FADM_LO2 =  0.224143868041857f;
constant float FADM_LO3 = -0.129409522550921f;
constant float FADM_HI0 = -0.129409522550921f;
constant float FADM_HI1 = -0.224143868041857f;
constant float FADM_HI2 =  0.836516303737469f;
constant float FADM_HI3 = -0.482962913144690f;

constant float FADM_ONE_BY_30    = 0.0333333351f;
constant float FADM_ONE_BY_15    = 0.0666666701f;
constant float FADM_COS_1DEG_SQ  = 0.99969541789740297f;
constant float FADM_EPS          = 1e-30f;

/* ------------------------------------------------------------------ */
/*  Shared helpers                                                      */
/* ------------------------------------------------------------------ */

static inline int fadm_mirror(int idx, int sup)
{
    if (idx < 0)   return -idx;
    if (idx >= sup) return 2 * sup - idx - 1;
    return idx;
}

/* ------------------------------------------------------------------ */
/*  Stage 0 — DWT vertical pass                                        */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] params_u32 — uint[18]:                              */
/*     [0] scale, [1] cur_w, [2] cur_h, [3] half_h,                   */
/*     [4] bpc,   [5] parent_buf_stride, [6] parent_half_h,            */
/*     [7] parent_w, [8] parent_h                                      */
/*   [[buffer(1)]] params_f32 — float[2]:                              */
/*     [0] scaler, [1] pixel_offset                                    */
/*   [[buffer(2)]] ref_raw   — device uint8_t * (stride = raw_stride)  */
/*   [[buffer(3)]] dis_raw   — device uint8_t *                        */
/*   [[buffer(4)]] raw_stride — uint (bytes per row in raw planes)     */
/*   [[buffer(5)]] parent_ref_band — device float * (or NULL at scale 0) */
/*   [[buffer(6)]] parent_dis_band — device float *                    */
/*   [[buffer(7)]] dwt_tmp_ref — device float *                        */
/*   [[buffer(8)]] dwt_tmp_dis — device float *                        */
/*                                                                      */
/*  Grid: ceil(cur_w/16) × ceil(half_h/16) × 2  (z=0 ref, z=1 dis)   */
/* ------------------------------------------------------------------ */

kernel void float_adm_dwt_vert(
    constant uint   *params_u32    [[buffer(0)]],
    constant float  *params_f32    [[buffer(1)]],
    device const uint8_t  *ref_raw [[buffer(2)]],
    device const uint8_t  *dis_raw [[buffer(3)]],
    constant uint          &raw_stride [[buffer(4)]],
    device const float    *parent_ref_band [[buffer(5)]],
    device const float    *parent_dis_band [[buffer(6)]],
    device float           *dwt_tmp_ref    [[buffer(7)]],
    device float           *dwt_tmp_dis    [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x;
    const int gy = (int)gid.y;
    const int plane_is_dis = (int)gid.z;  /* 0=ref, 1=dis */

    const uint scale      = params_u32[0];
    const int  cur_w      = (int)params_u32[1];
    const int  cur_h      = (int)params_u32[2];
    const int  half_h     = (int)params_u32[3];
    const uint bpc        = params_u32[4];
    const int  par_stride = (int)params_u32[5];  /* parent_buf_stride */
    const int  par_half_h = (int)params_u32[6];  /* parent_half_h     */
    const int  par_w      = (int)params_u32[7];
    const int  par_h      = (int)params_u32[8];

    const float scaler       = params_f32[0];
    const float pixel_offset = params_f32[1];

    if (gx >= cur_w || gy >= half_h) { return; }

    const int row_start = 2 * gy - 1;
    float s[4];

    for (int k = 0; k < 4; k++) {
        if (scale == 0u) {
            /* Read from raw u8/u16 plane. */
            device const uint8_t *plane = (plane_is_dis == 0) ? ref_raw : dis_raw;
            int rx = gx;
            int ry = fadm_mirror(row_start + k, cur_h);
            if (rx < 0)      rx = 0;
            if (rx >= cur_w) rx = cur_w - 1;
            if (bpc <= 8u) {
                s[k] = (float)plane[(uint)ry * raw_stride + (uint)rx] + pixel_offset;
            } else {
                device const uint16_t *p16 =
                    (device const uint16_t *)(plane + (uint)ry * raw_stride);
                s[k] = (float)p16[rx] / scaler + pixel_offset;
            }
        } else {
            /* Read from parent LL band (band 0 = LL, packed at offset 0). */
            device const float *band =
                (plane_is_dis == 0) ? parent_ref_band : parent_dis_band;
            int bx = gx;
            int by = fadm_mirror(row_start + k, par_h);
            if (bx < 0)       bx = 0;
            if (bx >= par_w)  bx = par_w - 1;
            (void)par_half_h;
            s[k] = band[by * par_stride + bx];
        }
    }

    const float lo = FADM_LO0 * s[0] + FADM_LO1 * s[1] + FADM_LO2 * s[2] + FADM_LO3 * s[3];
    const float hi = FADM_HI0 * s[0] + FADM_HI1 * s[1] + FADM_HI2 * s[2] + FADM_HI3 * s[3];

    const int out_stride = cur_w * 2;
    device float *dst = (plane_is_dis == 0) ? dwt_tmp_ref : dwt_tmp_dis;
    dst[gy * out_stride + gx]         = lo;
    dst[gy * out_stride + cur_w + gx] = hi;
}

/* ------------------------------------------------------------------ */
/*  Stage 1 — DWT horizontal pass → 4 sub-bands                       */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] params_u32 — uint[4]: cur_w, half_w, half_h, buf_stride */
/*   [[buffer(1)]] dwt_tmp_ref — device float *                        */
/*   [[buffer(2)]] dwt_tmp_dis — device float *                        */
/*   [[buffer(3)]] ref_band    — device float *                        */
/*   [[buffer(4)]] dis_band    — device float *                        */
/*                                                                      */
/*  Grid: ceil(half_w/16) × ceil(half_h/16) × 2  (z=0 ref, z=1 dis)  */
/* ------------------------------------------------------------------ */

kernel void float_adm_dwt_hori(
    constant uint *params_u32      [[buffer(0)]],
    device const float *dwt_tmp_ref [[buffer(1)]],
    device const float *dwt_tmp_dis [[buffer(2)]],
    device float       *ref_band    [[buffer(3)]],
    device float       *dis_band    [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x;
    const int gy = (int)gid.y;
    const int plane_is_dis = (int)gid.z;

    const int cur_w     = (int)params_u32[0];
    const int half_w    = (int)params_u32[1];
    const int half_h    = (int)params_u32[2];
    const int buf_stride = (int)params_u32[3];

    if (gx >= half_w || gy >= half_h) { return; }

    device const float *src = (plane_is_dis == 0) ? dwt_tmp_ref : dwt_tmp_dis;
    device float       *dst = (plane_is_dis == 0) ? ref_band     : dis_band;

    /* Helper: read lo/hi sub-row with mirror on x. */
    auto read_tmp = [&](int x_sub, int half_off) -> float {
        int xs = fadm_mirror(x_sub, cur_w);
        return src[gy * (cur_w * 2) + half_off + xs];
    };

    const int base_x = 2 * gx;

    /* lo sub-row taps (half_off = 0). */
    const float l0 = read_tmp(base_x - 1, 0);
    const float l1 = read_tmp(base_x + 0, 0);
    const float l2 = read_tmp(base_x + 1, 0);
    const float l3 = read_tmp(base_x + 2, 0);
    const float a_val = FADM_LO0 * l0 + FADM_LO1 * l1 + FADM_LO2 * l2 + FADM_LO3 * l3;
    const float v_val = FADM_HI0 * l0 + FADM_HI1 * l1 + FADM_HI2 * l2 + FADM_HI3 * l3;

    /* hi sub-row taps (half_off = cur_w). */
    const float h0 = read_tmp(base_x - 1, cur_w);
    const float h1 = read_tmp(base_x + 0, cur_w);
    const float h2 = read_tmp(base_x + 1, cur_w);
    const float h3 = read_tmp(base_x + 2, cur_w);
    const float h_val = FADM_LO0 * h0 + FADM_LO1 * h1 + FADM_LO2 * h2 + FADM_LO3 * h3;
    const float d_val = FADM_HI0 * h0 + FADM_HI1 * h1 + FADM_HI2 * h2 + FADM_HI3 * h3;

    const int slice = buf_stride * half_h;
    dst[0 * slice + gy * buf_stride + gx] = a_val;
    dst[1 * slice + gy * buf_stride + gx] = h_val;
    dst[2 * slice + gy * buf_stride + gx] = v_val;
    dst[3 * slice + gy * buf_stride + gx] = d_val;
}

/* ------------------------------------------------------------------ */
/*  Stage 2 — Decouple + CSF                                           */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] params_u32 — uint[3]: half_w, half_h, buf_stride   */
/*   [[buffer(1)]] params_f32 — float[4]: rfactor_h, rfactor_v, rfactor_d, gain_limit */
/*   [[buffer(2)]] ref_band   — device float *                         */
/*   [[buffer(3)]] dis_band   — device float *                         */
/*   [[buffer(4)]] csf_a      — device float *                         */
/*   [[buffer(5)]] csf_f      — device float *                         */
/*                                                                      */
/*  Grid: ceil(half_w/16) × ceil(half_h/16) × 1                       */
/* ------------------------------------------------------------------ */

kernel void float_adm_decouple_csf(
    constant uint  *params_u32 [[buffer(0)]],
    constant float *params_f32 [[buffer(1)]],
    device const float *ref_band [[buffer(2)]],
    device const float *dis_band [[buffer(3)]],
    device float       *csf_a    [[buffer(4)]],
    device float       *csf_f    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x;
    const int gy = (int)gid.y;

    const int half_w    = (int)params_u32[0];
    const int half_h    = (int)params_u32[1];
    const int buf_stride = (int)params_u32[2];

    const float rfactor_h  = params_f32[0];
    const float rfactor_v  = params_f32[1];
    const float rfactor_d  = params_f32[2];
    const float gain_limit = params_f32[3];

    if (gx >= half_w || gy >= half_h) { return; }

    const int slice = buf_stride * half_h;

    auto read_b = [&](int band, int y, int x) -> float {
        return ref_band[band * slice + y * buf_stride + x];
    };
    auto read_d = [&](int band, int y, int x) -> float {
        return dis_band[band * slice + y * buf_stride + x];
    };

    const float oh = read_b(1, gy, gx);
    const float ov = read_b(2, gy, gx);
    const float od = read_b(3, gy, gx);
    const float th = read_d(1, gy, gx);
    const float tv = read_d(2, gy, gx);
    const float td = read_d(3, gy, gx);

    /* Angle flag: explicit parens suppress FMA reordering (matches CUDA). */
    const float ot_dp = (oh * th) + (ov * tv);
    const float o_mag = (oh * oh) + (ov * ov);
    const float t_mag = (th * th) + (tv * tv);
    const float lhs = ot_dp * ot_dp;
    const float rhs = FADM_COS_1DEG_SQ * (o_mag * t_mag);
    const bool angle_flag = (ot_dp >= 0.0f) && (lhs >= rhs);

    const float oarr[3] = {oh, ov, od};
    const float tarr[3] = {th, tv, td};
    const float rfac[3] = {rfactor_h, rfactor_v, rfactor_d};

    for (int b = 0; b < 3; b++) {
        float k = tarr[b] / (oarr[b] + FADM_EPS);
        k = fmax(0.0f, fmin(k, 1.0f));
        float rst = k * oarr[b];
        if (angle_flag && rst > 0.0f)
            rst = fmin(rst * gain_limit, tarr[b]);
        else if (angle_flag && rst < 0.0f)
            rst = fmax(rst * gain_limit, tarr[b]);
        const float a_val     = tarr[b] - rst;
        const float csf_a_val = rfac[b] * a_val;
        csf_a[b * slice + gy * buf_stride + gx] = csf_a_val;
        csf_f[b * slice + gy * buf_stride + gx] = FADM_ONE_BY_30 * fabs(csf_a_val);
    }
}

/* ------------------------------------------------------------------ */
/*  Stage 3 — CSF-den + CM fused row reduction                        */
/*                                                                      */
/*  1-D dispatch: one threadgroup per (band, active-row) pair.         */
/*  wg_id = band_idx * num_active_rows + row_idx                       */
/*  Accumulator: 6 floats per WG: [csf_h csf_v csf_d cm_h cm_v cm_d] */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] params_u32 — uint[7]:                               */
/*     half_w, half_h, buf_stride,                                     */
/*     active_left, active_top, active_right, active_bottom            */
/*   [[buffer(1)]] params_f32 — float[4]:                              */
/*     rfactor_h, rfactor_v, rfactor_d, gain_limit                    */
/*   [[buffer(2)]] ref_band   — device float *                         */
/*   [[buffer(3)]] dis_band   — device float *                         */
/*   [[buffer(4)]] csf_a      — device float *                         */
/*   [[buffer(5)]] csf_f      — device float *                         */
/*   [[buffer(6)]] accum_out  — device float *  (wg_count × 6 floats) */
/*                                                                      */
/*  Grid: (3 * num_active_rows, 1, 1);  threadgroup: (256, 1, 1)      */
/* ------------------------------------------------------------------ */

kernel void float_adm_csf_cm(
    constant uint  *params_u32  [[buffer(0)]],
    constant float *params_f32  [[buffer(1)]],
    device const float *ref_band [[buffer(2)]],
    device const float *dis_band [[buffer(3)]],
    device const float *csf_a    [[buffer(4)]],
    device const float *csf_f    [[buffer(5)]],
    device float       *accum_out [[buffer(6)]],
    uint  wg_id [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  simd_lane  [[thread_index_in_simdgroup]],
    uint  simd_id    [[simdgroup_index_in_threadgroup]],
    uint  simd_count [[simdgroups_per_threadgroup]])
{
    const int half_w     = (int)params_u32[0];
    const int half_h     = (int)params_u32[1];
    const int buf_stride = (int)params_u32[2];
    const int act_left   = (int)params_u32[3];
    const int act_top    = (int)params_u32[4];
    const int act_right  = (int)params_u32[5];
    const int act_bottom = (int)params_u32[6];

    const float rfactor_h  = params_f32[0];
    const float rfactor_v  = params_f32[1];
    const float rfactor_d  = params_f32[2];
    const float gain_limit = params_f32[3];

    const int active_h = act_bottom - act_top;
    const int active_w = act_right  - act_left;
    if (active_h <= 0 || active_w <= 0) { return; }

    const uint num_rows = (uint)active_h;
    const uint band_idx = wg_id / num_rows;
    const uint row_idx  = wg_id - band_idx * num_rows;
    const int  row      = act_top + (int)row_idx;

    const float rfactor_band = (band_idx == 0u) ? rfactor_h :
                               (band_idx == 1u) ? rfactor_v :
                                                  rfactor_d;

    const int slice = buf_stride * half_h;

    /* Helper: read ref_band at (band+1, row, col) clamped to edges. */
    auto read_ref = [&](int b1, int y, int x) -> float {
        int cx = clamp(x, 0, half_w - 1);
        int cy = clamp(y, 0, half_h - 1);
        return ref_band[b1 * slice + cy * buf_stride + cx];
    };

    /* csf_f mirror (matches CUDA `fadm_read_csf_f_at`). */
    auto read_csf_f = [&](int b, int y, int x) -> float {
        int cx = x, cy = y;
        if (cx < 0)       cx = -cx;
        if (cx >= half_w) cx = 2 * half_w - cx - 2;
        if (cy < 0)       cy = -cy;
        if (cy >= half_h) cy = 2 * half_h - cy - 2;
        cx = clamp(cx, 0, half_w - 1);
        cy = clamp(cy, 0, half_h - 1);
        return csf_f[b * slice + cy * buf_stride + cx];
    };

    /* csf_a clamped read. */
    auto read_csf_a = [&](int b, int y, int x) -> float {
        int cx = clamp(x, 0, half_w - 1);
        int cy = clamp(y, 0, half_h - 1);
        return csf_a[b * slice + cy * buf_stride + cx];
    };

    /* Thread-serial accumulation across the active row. */
    float local_csf = 0.0f;
    float local_cm  = 0.0f;

    const uint WG_SIZE = 256u;
    for (int col = act_left + (int)lid; col < act_right; col += (int)WG_SIZE) {
        /* CSF denominator: (|rfactor * ref_band|)^3. */
        const float src_ref = read_ref((int)band_idx + 1, row, col);
        const float csf_o = fabs(rfactor_band * src_ref);
        local_csf += csf_o * csf_o * csf_o;

        /* Re-derive decoupled-r inline (same closed form as CUDA stage 3). */
        const float oh = read_ref(1, row, col);
        const float ov = read_ref(2, row, col);
        const float od = read_ref(3, row, col);
        const float th = dis_band[1 * slice + row * buf_stride + clamp(col, 0, half_w - 1)];
        const float tv = dis_band[2 * slice + row * buf_stride + clamp(col, 0, half_w - 1)];
        const float td = dis_band[3 * slice + row * buf_stride + clamp(col, 0, half_w - 1)];

        const float ot_dp    = (oh * th) + (ov * tv);
        const float o_mag    = (oh * oh) + (ov * ov);
        const float t_mag    = (th * th) + (tv * tv);
        const float lhs      = ot_dp * ot_dp;
        const float rhs      = FADM_COS_1DEG_SQ * (o_mag * t_mag);
        const bool  angle_flag = (ot_dp >= 0.0f) && (lhs >= rhs);

        const float oarr[3] = {oh, ov, od};
        const float tarr[3] = {th, tv, td};
        float k = tarr[band_idx] / (oarr[band_idx] + FADM_EPS);
        k = fmax(0.0f, fmin(k, 1.0f));
        float r_val = k * oarr[band_idx];
        if (angle_flag && r_val > 0.0f)
            r_val = fmin(r_val * gain_limit, tarr[band_idx]);
        else if (angle_flag && r_val < 0.0f)
            r_val = fmax(r_val * gain_limit, tarr[band_idx]);

        /* CM threshold: cross-band 8-neighbour sum + (1/15)|csf_a centre|. */
        float thr = 0.0f;
        for (int b = 0; b < 3; b++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) { continue; }
                    thr += read_csf_f(b, row + dy, col + dx);
                }
            }
        }
        thr += FADM_ONE_BY_15 * fabs(read_csf_a(0, row, col));
        thr += FADM_ONE_BY_15 * fabs(read_csf_a(1, row, col));
        thr += FADM_ONE_BY_15 * fabs(read_csf_a(2, row, col));

        const float x_val = rfactor_band * r_val;
        float xa = fabs(x_val) - thr;
        if (xa < 0.0f) { xa = 0.0f; }
        local_cm += xa * xa * xa;
    }

    /* Simd-group + threadgroup reduction → one partial per WG. */
    threadgroup float sg_csf[8];
    threadgroup float sg_cm[8];

    const float lane_csf = simd_sum(local_csf);
    const float lane_cm  = simd_sum(local_cm);
    if (simd_lane == 0u) {
        sg_csf[simd_id] = lane_csf;
        sg_cm[simd_id]  = lane_cm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        float total_csf = 0.0f;
        float total_cm  = 0.0f;
        for (uint i = 0u; i < simd_count; i++) {
            total_csf += sg_csf[i];
            total_cm  += sg_cm[i];
        }
        const uint slot_base = wg_id * 6u;
        accum_out[slot_base + band_idx]      = total_csf;
        accum_out[slot_base + 3u + band_idx] = total_cm;
    }
}
