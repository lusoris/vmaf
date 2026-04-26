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

#include "libvmaf/libvmaf_vulkan.h"
#include "vulkan/vulkan_common.h"

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

/* T7-29 part 2 (ADR-0186): contract checks for the VkImage import
 * surface. The end-to-end GPU plumbing is validated downstream by
 * part 3 (the libvmaf_vulkan FFmpeg filter); these unit tests
 * cover the public-API error paths so callers cannot misuse the
 * surface silently. */

static char *test_import_image_rejects_null_state(void)
{
    int rc = vmaf_vulkan_import_image(NULL, /*vk_image=*/0x1u, /*vk_format=*/0,
                                      /*vk_layout=*/0,
                                      /*vk_semaphore=*/0, /*vk_semaphore_value=*/0,
                                      /*w=*/16, /*h=*/16, /*bpc=*/8, /*is_ref=*/1, /*index=*/0);
    mu_assert("NULL state → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_import_image_rejects_zero_handle(void)
{
    if (vmaf_vulkan_device_count() <= 0)
        return NULL;
    VmafVulkanConfiguration cfg = {.device_index = -1, .enable_validation = 0};
    VmafVulkanState *state = NULL;
    mu_assert("state_init", vmaf_vulkan_state_init(&state, cfg) == 0);

    int rc = vmaf_vulkan_import_image(state, /*vk_image=*/0u, 0, 0, 0, 0, 16, 16, 8, 1, 0);
    mu_assert("vk_image == 0 → -EINVAL", rc == -EINVAL);

    vmaf_vulkan_state_free(&state);
    return NULL;
}

static char *test_wait_compute_null_state(void)
{
    int rc = vmaf_vulkan_wait_compute(NULL);
    mu_assert("NULL state → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_wait_compute_idle_is_zero(void)
{
    if (vmaf_vulkan_device_count() <= 0)
        return NULL;
    VmafVulkanConfiguration cfg = {.device_index = -1, .enable_validation = 0};
    VmafVulkanState *state = NULL;
    mu_assert("state_init", vmaf_vulkan_state_init(&state, cfg) == 0);

    /* No imports yet — wait_compute on an idle state must not
     * touch any uninitialised fence and must return success. */
    mu_assert("idle wait_compute returns 0", vmaf_vulkan_wait_compute(state) == 0);

    vmaf_vulkan_state_free(&state);
    return NULL;
}

static char *test_read_imported_pictures_without_imports(void)
{
    int rc = vmaf_vulkan_read_imported_pictures(NULL, 0);
    mu_assert("NULL ctx → -EINVAL", rc == -EINVAL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_context_destroy_null_is_noop);
    mu_run_test(test_context_new_rejects_null_out);
    mu_run_test(test_device_count_is_nonnegative);
    mu_run_test(test_context_new_or_skip);
    mu_run_test(test_context_new_rejects_out_of_range_index);
    mu_run_test(test_import_image_rejects_null_state);
    mu_run_test(test_import_image_rejects_zero_handle);
    mu_run_test(test_wait_compute_null_state);
    mu_run_test(test_wait_compute_idle_is_zero);
    mu_run_test(test_read_imported_pictures_without_imports);
    return NULL;
}
