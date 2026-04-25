/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Build + init smoke test for the Vulkan backend (T5-1b runtime).
 *  Verifies the runtime API surface: device_count returns >= 0 from
 *  a real volk + Vulkan probe, context_new succeeds when at least
 *  one compute-capable device is present (skipped on hosts with
 *  none — e.g. lavapipe-less CI containers), and context_destroy is
 *  null-safe. The contract was flipped from the T5-1 scaffold's
 *  `-ENOSYS` once volk + VMA + glslc plumbing landed.
 */

#include <errno.h>
#include <stddef.h>

#include "test.h"

#include "vulkan/common.h"

static char *test_context_destroy_null_is_noop(void)
{
    /* No assertion needed — must not crash. */
    vmaf_vulkan_context_destroy(NULL);
    return NULL;
}

static char *test_context_new_rejects_null_out(void)
{
    int rc = vmaf_vulkan_context_new(NULL, 0);
    mu_assert("NULL out → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_device_count_is_nonnegative(void)
{
    /* device_count returns 0 when no Vulkan loader / no compute device
     * (e.g. headless CI without lavapipe). On a host with a real GPU
     * exposing VK_QUEUE_COMPUTE_BIT, returns the count. Negative is
     * an enumeration error. */
    const int n = vmaf_vulkan_device_count();
    mu_assert("device_count returns >= 0", n >= 0);
    return NULL;
}

static char *test_context_new_or_skip(void)
{
    /* If at least one compute device is present, context_new must
     * succeed with auto-pick (-1) and the context must be non-NULL.
     * If zero devices, context_new returns -ENOSYS / -ENODEV; we
     * skip the assertion (smoke test is not required to provision
     * a GPU). */
    if (vmaf_vulkan_device_count() <= 0)
        return NULL;

    VmafVulkanContext *ctx = NULL;
    int rc = vmaf_vulkan_context_new(&ctx, -1);
    mu_assert("auto-pick context_new must succeed when devices >= 1", rc == 0);
    mu_assert("context must be non-NULL", ctx != NULL);
    vmaf_vulkan_context_destroy(ctx);
    return NULL;
}

static char *test_context_new_rejects_out_of_range_index(void)
{
    /* Skip when no devices — there's no valid range to probe. */
    if (vmaf_vulkan_device_count() <= 0)
        return NULL;

    VmafVulkanContext *ctx = NULL;
    /* 4096 is far above any plausible device count. */
    int rc = vmaf_vulkan_context_new(&ctx, 4096);
    mu_assert("out-of-range device_index → -EINVAL", rc == -EINVAL);
    mu_assert("context must remain NULL on error", ctx == NULL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_context_destroy_null_is_noop);
    mu_run_test(test_context_new_rejects_null_out);
    mu_run_test(test_device_count_is_nonnegative);
    mu_run_test(test_context_new_or_skip);
    mu_run_test(test_context_new_rejects_out_of_range_index);
    return NULL;
}
