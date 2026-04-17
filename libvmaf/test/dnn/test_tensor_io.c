/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "dnn/tensor_io.h"

static char *test_luma_to_f32_unnormalized(void)
{
    uint8_t src[16] = {0, 64, 128, 192, 255, 0, 128, 255, 10, 20, 30, 40, 50, 60, 70, 80};
    float dst[16];
    int err = vmaf_tensor_from_luma(src, 16, 16, 1, VMAF_TENSOR_LAYOUT_NCHW, VMAF_TENSOR_DTYPE_F32,
                                    NULL, NULL, dst);
    mu_assert("vmaf_tensor_from_luma failed", err == 0);
    mu_assert("0 did not map to 0", dst[0] == 0.0f);
    mu_assert("255 did not map to 1", fabsf(dst[4] - 1.0f) < 1e-6f);
    mu_assert("128 did not round correctly", fabsf(dst[2] - (128.0f / 255.0f)) < 1e-6f);
    return NULL;
}

static char *test_f16_roundtrip(void)
{
    const float inputs[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.5f, 0.123456f, -7.25f};
    const size_t n = sizeof(inputs) / sizeof(inputs[0]);
    uint16_t h[16];
    float back[16];
    vmaf_f32_to_f16(inputs, h, n);
    vmaf_f16_to_f32(h, back, n);
    for (size_t i = 0; i < n; ++i) {
        float tol = fabsf(inputs[i]) * 1e-3f + 1e-3f;
        mu_assert("f16 roundtrip exceeded tolerance", fabsf(inputs[i] - back[i]) <= tol);
    }
    return NULL;
}

static char *test_luma_roundtrip(void)
{
    uint8_t src[64], dst[64];
    float tensor[64];
    for (int i = 0; i < 64; ++i)
        src[i] = (uint8_t)(i * 4 % 256);
    int err = vmaf_tensor_from_luma(src, 8, 8, 8, VMAF_TENSOR_LAYOUT_NCHW, VMAF_TENSOR_DTYPE_F32,
                                    NULL, NULL, tensor);
    mu_assert("luma→tensor failed", err == 0);
    err = vmaf_tensor_to_luma(tensor, VMAF_TENSOR_LAYOUT_NCHW, VMAF_TENSOR_DTYPE_F32, 8, 8, NULL,
                              NULL, dst, 8);
    mu_assert("tensor→luma failed", err == 0);
    for (int i = 0; i < 64; ++i) {
        mu_assert("luma roundtrip differed", src[i] == dst[i]);
    }
    return NULL;
}

static char *test_rejects_bad_args(void)
{
    uint8_t src[4];
    float dst[4];
    int err = vmaf_tensor_from_luma(NULL, 2, 2, 2, VMAF_TENSOR_LAYOUT_NCHW, VMAF_TENSOR_DTYPE_F32,
                                    NULL, NULL, dst);
    mu_assert("expected -EINVAL on NULL src", err < 0);
    err = vmaf_tensor_from_luma(src, 1, 2, 2, VMAF_TENSOR_LAYOUT_NCHW, VMAF_TENSOR_DTYPE_F32, NULL,
                                NULL, dst);
    mu_assert("expected -EINVAL on stride < width", err < 0);
    return NULL;
}

static char *test_rgb_imagenet_known_values(void)
{
    /* 2x2 frame, each channel a constant. ImageNet mean/std:
     *   R: (v/255 - 0.485) / 0.229
     *   G: (v/255 - 0.456) / 0.224
     *   B: (v/255 - 0.406) / 0.225
     * For v=0 this is the classic "most-negative ImageNet value" per channel. */
    uint8_t r[4] = {0, 0, 0, 0};
    uint8_t g[4] = {0, 0, 0, 0};
    uint8_t b[4] = {0, 0, 0, 0};
    float dst[12];
    int err = vmaf_tensor_from_rgb_imagenet(r, 2, g, 2, b, 2, 2, 2, dst);
    mu_assert("rgb imagenet failed for zero input", err == 0);
    const float expected_r = (0.0f - 0.485f) / 0.229f;
    const float expected_g = (0.0f - 0.456f) / 0.224f;
    const float expected_b = (0.0f - 0.406f) / 0.225f;
    for (int i = 0; i < 4; ++i) {
        mu_assert("R plane mismatch at zero", fabsf(dst[i] - expected_r) < 1e-5f);
        mu_assert("G plane mismatch at zero", fabsf(dst[4 + i] - expected_g) < 1e-5f);
        mu_assert("B plane mismatch at zero", fabsf(dst[8 + i] - expected_b) < 1e-5f);
    }

    /* v=255 → (1 - mean) / std */
    uint8_t r2[4] = {255, 255, 255, 255};
    uint8_t g2[4] = {255, 255, 255, 255};
    uint8_t b2[4] = {255, 255, 255, 255};
    err = vmaf_tensor_from_rgb_imagenet(r2, 2, g2, 2, b2, 2, 2, 2, dst);
    mu_assert("rgb imagenet failed for 255 input", err == 0);
    const float r255 = (1.0f - 0.485f) / 0.229f;
    const float g255 = (1.0f - 0.456f) / 0.224f;
    const float b255 = (1.0f - 0.406f) / 0.225f;
    for (int i = 0; i < 4; ++i) {
        mu_assert("R plane mismatch at 255", fabsf(dst[i] - r255) < 1e-5f);
        mu_assert("G plane mismatch at 255", fabsf(dst[4 + i] - g255) < 1e-5f);
        mu_assert("B plane mismatch at 255", fabsf(dst[8 + i] - b255) < 1e-5f);
    }
    return NULL;
}

static char *test_rgb_imagenet_nchw_layout(void)
{
    /* Verify planes are written contiguously in NCHW order (R first,
     * then G, then B). Distinct per-channel values should land in
     * distinct 1/3 segments of the destination buffer. */
    uint8_t r[6] = {10, 20, 30, 40, 50, 60};
    uint8_t g[6] = {70, 80, 90, 100, 110, 120};
    uint8_t b[6] = {130, 140, 150, 160, 170, 180};
    float dst[3 * 6];
    int err = vmaf_tensor_from_rgb_imagenet(r, 3, g, 3, b, 3, 3, 2, dst);
    mu_assert("rgb imagenet 3x2 failed", err == 0);
    /* The R plane's first element should reflect r[0]=10: (10/255 - 0.485) / 0.229. */
    const float first_r = ((10.0f / 255.0f) - 0.485f) / 0.229f;
    const float first_g = ((70.0f / 255.0f) - 0.456f) / 0.224f;
    const float first_b = ((130.0f / 255.0f) - 0.406f) / 0.225f;
    mu_assert("R plane not first in NCHW order", fabsf(dst[0] - first_r) < 1e-5f);
    mu_assert("G plane not second in NCHW order", fabsf(dst[6] - first_g) < 1e-5f);
    mu_assert("B plane not third in NCHW order", fabsf(dst[12] - first_b) < 1e-5f);
    return NULL;
}

static char *test_rgb_imagenet_rejects_bad_args(void)
{
    uint8_t c[4] = {0};
    float dst[12];
    int err = vmaf_tensor_from_rgb_imagenet(NULL, 2, c, 2, c, 2, 2, 2, dst);
    mu_assert("expected -EINVAL on NULL R", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, NULL, 2, c, 2, 2, 2, dst);
    mu_assert("expected -EINVAL on NULL G", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, c, 2, NULL, 2, 2, 2, dst);
    mu_assert("expected -EINVAL on NULL B", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, c, 2, c, 2, 2, 2, NULL);
    mu_assert("expected -EINVAL on NULL dst", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 1, c, 2, c, 2, 2, 2, dst);
    mu_assert("expected -EINVAL on stride_r < width", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, c, 1, c, 2, 2, 2, dst);
    mu_assert("expected -EINVAL on stride_g < width", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, c, 2, c, 1, 2, 2, dst);
    mu_assert("expected -EINVAL on stride_b < width", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, c, 2, c, 2, 0, 2, dst);
    mu_assert("expected -EINVAL on zero width", err < 0);
    err = vmaf_tensor_from_rgb_imagenet(c, 2, c, 2, c, 2, 2, 0, dst);
    mu_assert("expected -EINVAL on zero height", err < 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_luma_to_f32_unnormalized);
    mu_run_test(test_f16_roundtrip);
    mu_run_test(test_luma_roundtrip);
    mu_run_test(test_rejects_bad_args);
    mu_run_test(test_rgb_imagenet_known_values);
    mu_run_test(test_rgb_imagenet_nchw_layout);
    mu_run_test(test_rgb_imagenet_rejects_bad_args);
    return NULL;
}
