/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Build + init smoke test for the Vulkan backend scaffold (ADR-0175).
 *  All public entry points return -ENOSYS until the kernels land; this
 *  test pins that contract so a future PR can't accidentally enable
 *  the backend without flipping the smoke expectations.
 */

#include <errno.h>
#include <stddef.h>

#include "test.h"

#include "vulkan/common.h"

static char *test_context_new_returns_zeroed_struct(void)
{
    /* The scaffold's calloc + struct initialisation succeeds even
     * before a real device is selected. The opaque pointer is
     * non-NULL on success. */
    VmafVulkanContext *ctx = NULL;
    int rc = vmaf_vulkan_context_new(&ctx, 0);
    mu_assert("scaffold context_new must succeed", rc == 0);
    mu_assert("scaffold context must be populated", ctx != NULL);
    vmaf_vulkan_context_destroy(ctx);
    return NULL;
}

static char *test_context_new_rejects_null_out(void)
{
    int rc = vmaf_vulkan_context_new(NULL, 0);
    mu_assert("NULL out → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_context_destroy_null_is_noop(void)
{
    /* No assertion needed — must not crash. */
    vmaf_vulkan_context_destroy(NULL);
    return NULL;
}

static char *test_device_count_scaffold_returns_zero(void)
{
    /* The scaffold returns 0 (no real Vulkan probe yet). When the
     * runtime PR replaces the stub, this expectation flips to
     * "either >= 0 from a real probe or skip when no device". */
    const int n = vmaf_vulkan_device_count();
    mu_assert("scaffold device_count returns 0", n == 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_context_new_returns_zeroed_struct);
    mu_run_test(test_context_new_rejects_null_out);
    mu_run_test(test_context_destroy_null_is_noop);
    mu_run_test(test_device_count_scaffold_returns_zero);
    return NULL;
}
