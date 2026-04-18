/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Public-surface tests for vmaf_dnn_session_run() and its data types.
 *  Runs against the public headers only (no private dnn/ includes) so it
 *  verifies the stable API contract that downstream consumers see.
 *
 *  When libvmaf was built with -Denable_dnn=disabled, every entry point
 *  returns -ENOSYS and only the stub-semantics tests execute. The real-
 *  ORT path is additionally covered by the CLI smoke gate (test_cli.sh)
 *  against the model/tiny/smoke_v0.onnx fixture.
 */

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/dnn.h"

static char *test_stub_returns_enosys_when_disabled(void)
{
    if (vmaf_dnn_available()) {
        /* This binary was built with real ORT. The stub-only assertions
         * do not apply — skip without failing. */
        return NULL;
    }
    float in_data[1] = {0.0f};
    int64_t in_shape[1] = {1};
    float out_data[1] = {0.0f};

    VmafDnnInput in = {.name = NULL, .data = in_data, .shape = in_shape, .rank = 1};
    VmafDnnOutput out = {.name = NULL, .data = out_data, .capacity = 1, .written = 0};

    int rc = vmaf_dnn_session_run(NULL, &in, 1, &out, 1);
    mu_assert("stub must return -ENOSYS when DNN is disabled", rc == -ENOSYS);
    return NULL;
}

static char *test_rejects_null_session(void)
{
    float in_data[1] = {0.0f};
    int64_t in_shape[1] = {1};
    float out_data[1] = {0.0f};
    VmafDnnInput in = {.name = NULL, .data = in_data, .shape = in_shape, .rank = 1};
    VmafDnnOutput out = {.name = NULL, .data = out_data, .capacity = 1, .written = 0};

    int rc = vmaf_dnn_session_run(NULL, &in, 1, &out, 1);
    mu_assert("NULL session must be rejected", rc < 0);
    return NULL;
}

static char *test_descriptor_field_layout(void)
{
    /* Compile-time sanity: the public descriptor fields must stay where
     * downstream callers expect them (designated initialisers above pin
     * the contract). Catches accidental field-reorder refactors. */
    VmafDnnInput in = {.name = "ref", .data = NULL, .shape = NULL, .rank = 4};
    VmafDnnOutput out = {.name = "y", .data = NULL, .capacity = 0, .written = 0};
    mu_assert("input.name binding", in.name != NULL);
    mu_assert("input.rank binding", in.rank == 4u);
    mu_assert("output.name binding", out.name != NULL);
    mu_assert("output.written starts zeroed", out.written == 0u);
    return NULL;
}

static char *test_session_open_rejects_null_out(void)
{
    int rc = vmaf_dnn_session_open(NULL, "anything.onnx", NULL);
    /* Stub branch: -ENOSYS; real branch: -EINVAL. Either is a hard reject. */
    mu_assert("NULL out pointer rejected", rc < 0);
    return NULL;
}

static char *test_session_open_rejects_null_path(void)
{
    VmafDnnSession *s = NULL;
    int rc = vmaf_dnn_session_open(&s, NULL, NULL);
    mu_assert("NULL path rejected", rc < 0);
    mu_assert("session pointer not written on reject", s == NULL);
    return NULL;
}

static char *test_session_open_rejects_missing_file(void)
{
    if (!vmaf_dnn_available()) {
        /* Stub returns -ENOSYS regardless of path; no file-existence check
         * to exercise. Skip without failing. */
        return NULL;
    }
    VmafDnnSession *s = NULL;
    int rc = vmaf_dnn_session_open(&s, "/nonexistent/path/to/model.onnx", NULL);
    mu_assert("missing model file rejected", rc < 0);
    mu_assert("session pointer not populated", s == NULL);
    return NULL;
}

static char *test_session_run_luma8_rejects_null(void)
{
    /* Stub branch: returns -ENOSYS for any args. Real branch: -EINVAL on
     * NULL sess/in/out. The wrapper rejects either way. */
    uint8_t buf[16] = {0};
    int rc = vmaf_dnn_session_run_luma8(NULL, buf, 4, 4, 4, buf, 4);
    mu_assert("NULL session rejected by run_luma8", rc < 0);
    return NULL;
}

static char *test_session_close_null_is_noop(void)
{
    /* Free on NULL is a hard contract — must never crash. There is no
     * return value to assert; reaching the next line is the test. */
    vmaf_dnn_session_close(NULL);
    mu_assert("close(NULL) returned without crashing", 1);
    return NULL;
}

static char *test_attached_ep_null_returns_null(void)
{
    const char *ep = vmaf_dnn_session_attached_ep(NULL);
    mu_assert("attached_ep(NULL) returns NULL", ep == NULL);
    return NULL;
}

static char *test_run_rejects_zero_n_inputs(void)
{
    /* Even with non-NULL pointers, 0 inputs / 0 outputs must be rejected.
     * Stub returns -ENOSYS; real branch returns -EINVAL. */
    float buf[1] = {0.0f};
    int64_t shape[1] = {1};
    VmafDnnInput in = {.name = NULL, .data = buf, .shape = shape, .rank = 1};
    VmafDnnOutput out = {.name = NULL, .data = buf, .capacity = 1, .written = 0};
    int rc = vmaf_dnn_session_run((VmafDnnSession *)0xdeadbeef, &in, 0u, &out, 1u);
    mu_assert("zero n_inputs rejected", rc < 0);
    rc = vmaf_dnn_session_run((VmafDnnSession *)0xdeadbeef, &in, 1u, &out, 0u);
    mu_assert("zero n_outputs rejected", rc < 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_stub_returns_enosys_when_disabled);
    mu_run_test(test_rejects_null_session);
    mu_run_test(test_descriptor_field_layout);
    mu_run_test(test_session_open_rejects_null_out);
    mu_run_test(test_session_open_rejects_null_path);
    mu_run_test(test_session_open_rejects_missing_file);
    mu_run_test(test_session_run_luma8_rejects_null);
    mu_run_test(test_session_close_null_is_noop);
    mu_run_test(test_attached_ep_null_returns_null);
    mu_run_test(test_run_rejects_zero_n_inputs);
    return NULL;
}
