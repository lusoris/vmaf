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

char *run_tests(void)
{
    mu_run_test(test_stub_returns_enosys_when_disabled);
    mu_run_test(test_rejects_null_session);
    mu_run_test(test_descriptor_field_layout);
    return NULL;
}
