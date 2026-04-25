/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tensor_io.h"

/* IEEE-754 binary16 <-> binary32 conversion without hw intrinsics. Round-to-
 * nearest-even on the down-convert; exact on the up-convert. */

static uint16_t f32_to_f16_one(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (exp >= 31) {
        /* overflow → inf, or propagate NaN */
        uint16_t nan_mant = (mant != 0u) ? 0x200u : 0u;
        return (uint16_t)((sign << 15) | (0x1fu << 10) | nan_mant);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)(sign << 15);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        uint32_t round = (mant & 0x1000u) != 0u ? 1u : 0u;
        return (uint16_t)((sign << 15) | ((mant >> 13) + round));
    }
    uint32_t round = (mant & 0x1000u) != 0u ? 1u : 0u;
    uint32_t h = (sign << 15) | ((uint32_t)exp << 10) | (mant >> 13);
    return (uint16_t)(h + round);
}

static float f16_to_f32_one(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign << 31;
        } else {
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                exp -= 1u;
            }
            exp += 1u;
            mant &= 0x3ffu;
            out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

void vmaf_f32_to_f16(const float *src, uint16_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f32_to_f16_one(src[i]);
    }
}

void vmaf_f16_to_f32(const uint16_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f16_to_f32_one(src[i]);
    }
}

int vmaf_tensor_from_luma(const uint8_t *src, size_t stride_src, int width, int height,
                          VmafTensorLayout layout, VmafTensorDType dtype, const float *mean,
                          const float *std, void *dst)
{
    if (!src || !dst || width <= 0 || height <= 0 || stride_src < (size_t)width) {
        return -EINVAL;
    }
    /* Luma is single-channel; NCHW vs NHWC layout is equivalent for C=1. */
    (void)layout;

    const float m = mean ? mean[0] : 0.0f;
    const float s = std ? std[0] : 1.0f;
    if (s == 0.0f) {
        return -EINVAL;
    }
    const float inv_s = 1.0f / s;

    const size_t n = (size_t)width * (size_t)height;
    if (dtype == VMAF_TENSOR_DTYPE_F32) {
        float *out = (float *)dst;
        for (int y = 0; y < height; ++y) {
            const uint8_t *row = src + (size_t)y * stride_src;
            for (int x = 0; x < width; ++x) {
                out[(size_t)y * (size_t)width + (size_t)x] =
                    (((float)row[x] * (1.0f / 255.0f)) - m) * inv_s;
            }
        }
    } else if (dtype == VMAF_TENSOR_DTYPE_F16) {
        uint16_t *out = (uint16_t *)dst;
        for (int y = 0; y < height; ++y) {
            const uint8_t *row = src + (size_t)y * stride_src;
            for (int x = 0; x < width; ++x) {
                float v = (((float)row[x] * (1.0f / 255.0f)) - m) * inv_s;
                out[(size_t)y * (size_t)width + (size_t)x] = f32_to_f16_one(v);
            }
        }
    } else {
        return -EINVAL;
    }
    (void)n;
    return 0;
}

/* ImageNet / torchvision normalization constants. Keep in sync with the
 * export pipeline in ai/ — any divergence silently biases every learned
 * RGB model (LPIPS, MobileSal, TransNet-V2). */
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

int vmaf_tensor_from_rgb_imagenet(const uint8_t *src_r, size_t stride_r, const uint8_t *src_g,
                                  size_t stride_g, const uint8_t *src_b, size_t stride_b, int width,
                                  int height, float *dst)
{
    if (!src_r || !src_g || !src_b || !dst || width <= 0 || height <= 0)
        return -EINVAL;
    if (stride_r < (size_t)width || stride_g < (size_t)width || stride_b < (size_t)width)
        return -EINVAL;

    const size_t plane = (size_t)width * (size_t)height;
    const uint8_t *const srcs[3] = {src_r, src_g, src_b};
    const size_t strides[3] = {stride_r, stride_g, stride_b};

    for (int c = 0; c < 3; ++c) {
        const float m = IMAGENET_MEAN[c];
        const float inv_s = 1.0f / IMAGENET_STD[c];
        float *out_c = dst + (size_t)c * plane;
        for (int y = 0; y < height; ++y) {
            const uint8_t *row = srcs[c] + (size_t)y * strides[c];
            float *out_row = out_c + (size_t)y * (size_t)width;
            for (int x = 0; x < width; ++x) {
                out_row[x] = (((float)row[x] * (1.0f / 255.0f)) - m) * inv_s;
            }
        }
    }
    return 0;
}

int vmaf_tensor_from_plane16(const uint16_t *src, size_t stride_src, int width, int height, int bpc,
                             VmafTensorLayout layout, VmafTensorDType dtype, const float *mean,
                             const float *std, void *dst)
{
    if (!src || !dst || width <= 0 || height <= 0 || bpc < 9 || bpc > 16 ||
        stride_src < (size_t)width * sizeof(uint16_t)) {
        return -EINVAL;
    }
    (void)layout; /* C=1 → NCHW and NHWC coincide. */

    const float m = mean ? mean[0] : 0.0f;
    const float s = std ? std[0] : 1.0f;
    if (s == 0.0f) {
        return -EINVAL;
    }
    const float inv_s = 1.0f / s;
    const float inv_max = 1.0f / (float)((1u << bpc) - 1u);

    if (dtype == VMAF_TENSOR_DTYPE_F32) {
        float *out = (float *)dst;
        for (int y = 0; y < height; ++y) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)src + (size_t)y * stride_src);
            for (int x = 0; x < width; ++x) {
                out[(size_t)y * (size_t)width + (size_t)x] =
                    (((float)row[x] * inv_max) - m) * inv_s;
            }
        }
    } else if (dtype == VMAF_TENSOR_DTYPE_F16) {
        uint16_t *out = (uint16_t *)dst;
        for (int y = 0; y < height; ++y) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)src + (size_t)y * stride_src);
            for (int x = 0; x < width; ++x) {
                float v = (((float)row[x] * inv_max) - m) * inv_s;
                out[(size_t)y * (size_t)width + (size_t)x] = f32_to_f16_one(v);
            }
        }
    } else {
        return -EINVAL;
    }
    return 0;
}

int vmaf_tensor_to_plane16(const void *src, VmafTensorLayout layout, VmafTensorDType dtype,
                           int width, int height, int bpc, const float *mean, const float *std,
                           uint16_t *dst, size_t stride_dst)
{
    if (!src || !dst || width <= 0 || height <= 0 || bpc < 9 || bpc > 16 ||
        stride_dst < (size_t)width * sizeof(uint16_t)) {
        return -EINVAL;
    }
    (void)layout;

    const float m = mean ? mean[0] : 0.0f;
    const float s = std ? std[0] : 1.0f;
    const float max_val = (float)((1u << bpc) - 1u);
    const int max_int = (int)((1u << bpc) - 1u);

    for (int y = 0; y < height; ++y) {
        uint16_t *row = (uint16_t *)((uint8_t *)dst + (size_t)y * stride_dst);
        for (int x = 0; x < width; ++x) {
            float v;
            if (dtype == VMAF_TENSOR_DTYPE_F32) {
                v = ((const float *)src)[(size_t)y * (size_t)width + (size_t)x];
            } else if (dtype == VMAF_TENSOR_DTYPE_F16) {
                v = f16_to_f32_one(((const uint16_t *)src)[(size_t)y * (size_t)width + (size_t)x]);
            } else {
                return -EINVAL;
            }
            float denorm = ((v * s) + m) * max_val;
            float rounded = nearbyintf(denorm);
            int ri = (int)rounded;
            if (ri < 0)
                ri = 0;
            if (ri > max_int)
                ri = max_int;
            row[x] = (uint16_t)ri;
        }
    }
    return 0;
}

int vmaf_tensor_to_luma(const void *src, VmafTensorLayout layout, VmafTensorDType dtype, int width,
                        int height, const float *mean, const float *std, uint8_t *dst,
                        size_t stride_dst)
{
    if (!src || !dst || width <= 0 || height <= 0 || stride_dst < (size_t)width) {
        return -EINVAL;
    }
    (void)layout;

    const float m = mean ? mean[0] : 0.0f;
    const float s = std ? std[0] : 1.0f;

    for (int y = 0; y < height; ++y) {
        uint8_t *row = dst + (size_t)y * stride_dst;
        for (int x = 0; x < width; ++x) {
            float v;
            if (dtype == VMAF_TENSOR_DTYPE_F32) {
                v = ((const float *)src)[(size_t)y * (size_t)width + (size_t)x];
            } else if (dtype == VMAF_TENSOR_DTYPE_F16) {
                v = f16_to_f32_one(((const uint16_t *)src)[(size_t)y * (size_t)width + (size_t)x]);
            } else {
                return -EINVAL;
            }
            float denorm = ((v * s) + m) * 255.0f;
            float rounded = nearbyintf(denorm);
            int ri = (int)rounded;
            if (ri < 0)
                ri = 0;
            if (ri > 255)
                ri = 255;
            row[x] = (uint8_t)ri;
        }
    }
    return 0;
}
