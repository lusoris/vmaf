/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Tests for VmafDnnConfig.device execution-provider selection and
 *  VmafDnnConfig.fp16_io round-trip conversion. Runs only when libvmaf
 *  was built with -Denable_dnn=enabled; otherwise the asserts collapse
 *  to stub-semantics checks.
 *
 *  EP selection is verified via vmaf_dnn_session_attached_ep(), which
 *  returns the human-readable name of the EP that actually bound. On a
 *  stock CI ORT build (no CUDA/OpenVINO/ROCm compiled in) every AUTO
 *  and explicit-device selection must fall through to "CPU" without
 *  erroring — that's the behaviour this gate locks in.
 *
 *  fp16_io is verified with model/tiny/smoke_fp16_v0.onnx, a 2x2 Identity
 *  model with FLOAT16 inputs and outputs. The test feeds fp32 values,
 *  libvmaf casts to fp16 host-side, ORT runs Identity, libvmaf casts
 *  back — output must equal input within fp16-rounding tolerance.
 */

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/dnn.h"

#define SMOKE_FP32_MODEL "model/tiny/smoke_v0.onnx"
#define SMOKE_FP16_MODEL "model/tiny/smoke_fp16_v0.onnx"

static char *test_auto_falls_through_to_cpu(void)
{
    if (!vmaf_dnn_available())
        return NULL;

    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = {.device = VMAF_DNN_DEVICE_AUTO};
    int rc = vmaf_dnn_session_open(&sess, SMOKE_FP32_MODEL, &cfg);
    if (rc == -ENOENT) {
        /* Test data not present — working tree checkout only. */
        return NULL;
    }
    mu_assert("AUTO open succeeds", rc == 0);
    mu_assert("session is non-NULL", sess != NULL);

    const char *ep = vmaf_dnn_session_attached_ep(sess);
    mu_assert("attached_ep is non-NULL", ep != NULL);
    /* CI ORT ships CPU EP only. If the runtime happens to have CUDA or
     * OpenVINO compiled in that's fine — any named EP is acceptable so
     * long as it's a known stable string. */
    mu_assert("EP name matches known set", strcmp(ep, "CPU") == 0 || strcmp(ep, "CUDA") == 0 ||
                                               strcmp(ep, "OpenVINO:GPU") == 0 ||
                                               strcmp(ep, "OpenVINO:CPU") == 0 ||
                                               strcmp(ep, "ROCm") == 0);

    vmaf_dnn_session_close(sess);
    return NULL;
}

static char *test_explicit_openvino_graceful_fallback(void)
{
    if (!vmaf_dnn_available())
        return NULL;

    /* Requesting OpenVINO on a CPU-only ORT build must silently fall
     * back to CPU rather than failing the open. This is the
     * "accepted-but-ignored" behaviour turning into "accepted-and-
     * honoured-or-logged-fallback". */
    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = {.device = VMAF_DNN_DEVICE_OPENVINO};
    int rc = vmaf_dnn_session_open(&sess, SMOKE_FP32_MODEL, &cfg);
    if (rc == -ENOENT)
        return NULL;
    mu_assert("OPENVINO request does not fail open", rc == 0);

    const char *ep = vmaf_dnn_session_attached_ep(sess);
    mu_assert("EP is reported", ep != NULL);
    /* Will be CPU on stock CI ORT; OpenVINO:* if DPC++/OV-enabled build. */

    vmaf_dnn_session_close(sess);
    return NULL;
}

static char *test_explicit_cuda_graceful_fallback(void)
{
    if (!vmaf_dnn_available())
        return NULL;

    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = {.device = VMAF_DNN_DEVICE_CUDA};
    int rc = vmaf_dnn_session_open(&sess, SMOKE_FP32_MODEL, &cfg);
    if (rc == -ENOENT)
        return NULL;
    mu_assert("CUDA request does not fail open", rc == 0);

    const char *ep = vmaf_dnn_session_attached_ep(sess);
    mu_assert("EP is reported", ep != NULL);

    vmaf_dnn_session_close(sess);
    return NULL;
}

static char *test_fp16_io_round_trip(void)
{
    if (!vmaf_dnn_available())
        return NULL;

    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = {.device = VMAF_DNN_DEVICE_CPU, .fp16_io = true};
    int rc = vmaf_dnn_session_open(&sess, SMOKE_FP16_MODEL, &cfg);
    if (rc == -ENOENT)
        return NULL;
    mu_assert("fp16 session open succeeds", rc == 0);

    /* Identity model: output must equal input within fp16 rounding
     * error. Values chosen to stay exactly representable in fp16 so the
     * assertion is tight. Non-exact values would introduce ±0.5 ULP of
     * fp16 precision on top — still detectable as a bug but a looser
     * tolerance would mask real cast-direction errors. */
    const float in_data[4] = {0.0f, 1.0f, -0.5f, 256.0f};
    float out_data[4] = {9999.f, 9999.f, 9999.f, 9999.f};
    const int64_t shape[4] = {1, 1, 2, 2};

    VmafDnnInput in = {.name = "x", .data = in_data, .shape = shape, .rank = 4};
    VmafDnnOutput out = {.name = "y", .data = out_data, .capacity = 4, .written = 0};

    rc = vmaf_dnn_session_run(sess, &in, 1, &out, 1);
    mu_assert("fp16 Identity run succeeds", rc == 0);
    mu_assert("output element count is 4", out.written == 4);

    for (size_t i = 0; i < 4; ++i) {
        const float diff = fabsf(out_data[i] - in_data[i]);
        mu_assert("fp16 round-trip matches input exactly", diff == 0.0f);
    }

    vmaf_dnn_session_close(sess);
    return NULL;
}

static char *test_fp16_model_rejects_fp32_config(void)
{
    if (!vmaf_dnn_available())
        return NULL;

    /* Opening a FLOAT16-typed model with fp16_io=false: the session
     * still opens (ORT accepts the graph), but a run with fp32 input
     * buffers must fail because we hand ORT a FLOAT-typed tensor for a
     * FLOAT16-declared input slot. We assert the run failure, not the
     * open failure — see ADR-0102 rationale. */
    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = {.device = VMAF_DNN_DEVICE_CPU, .fp16_io = false};
    int rc = vmaf_dnn_session_open(&sess, SMOKE_FP16_MODEL, &cfg);
    if (rc == -ENOENT)
        return NULL;
    mu_assert("fp16 model opens under fp16_io=false", rc == 0);

    const float in_data[4] = {0.f, 1.f, 2.f, 3.f};
    float out_data[4] = {0.f, 0.f, 0.f, 0.f};
    const int64_t shape[4] = {1, 1, 2, 2};
    VmafDnnInput in = {.name = "x", .data = in_data, .shape = shape, .rank = 4};
    VmafDnnOutput out = {.name = "y", .data = out_data, .capacity = 4, .written = 0};

    rc = vmaf_dnn_session_run(sess, &in, 1, &out, 1);
    mu_assert("fp16 model rejects fp32 tensor when fp16_io=false", rc < 0);

    vmaf_dnn_session_close(sess);
    return NULL;
}

static char *test_stub_attached_ep_returns_null(void)
{
    if (vmaf_dnn_available())
        return NULL;
    /* When DNN is disabled, the accessor is defined but returns NULL for
     * any session (which can only be NULL via the stub path anyway). */
    mu_assert("stub attached_ep returns NULL", vmaf_dnn_session_attached_ep(NULL) == NULL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_auto_falls_through_to_cpu);
    mu_run_test(test_explicit_openvino_graceful_fallback);
    mu_run_test(test_explicit_cuda_graceful_fallback);
    mu_run_test(test_fp16_io_round_trip);
    mu_run_test(test_fp16_model_rejects_fp32_config);
    mu_run_test(test_stub_attached_ep_returns_null);
    return NULL;
}
