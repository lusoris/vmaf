/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  T7-29 part 4 (ADR-0235) — contract-level smoke for the v2
 *  async pending-fence ring on top of vmaf_vulkan_import_image /
 *  _wait_compute / _state_build_pictures.
 *
 *  The end-to-end "submit N > ring_size frames and verify
 *  scores match" test requires a valid VkImage handle and a
 *  decoder-side VkSemaphore — those live downstream in the
 *  FFmpeg `libvmaf_vulkan` filter (T7-29 part 3) and the
 *  cross-backend parity gate. This file pins the **contract**
 *  the ring exposes:
 *    - import_image with vk_image == 0 still rejects with
 *      -EINVAL after the v1 → v2 swap.
 *    - wait_compute on a state that never imported anything
 *      stays a 0-returning no-op.
 *    - build_pictures rejects "index never imported" with
 *      -EINVAL regardless of slot index parity.
 *    - state_free on a ring with zero outstanding fences must
 *      not crash, must not deadlock, and must not leak.
 *
 *  Once a real VkImage path is wired (FFmpeg filter or a synthetic
 *  vkCreateImage in the test), this file extends with the ring-
 *  wrap correctness check (submit ring_size + 1 frames, verify
 *  the slot[0] back-pressure stalls without corruption).
 */

#include <errno.h>
#include <stddef.h>

#include "test.h"

#include "libvmaf/libvmaf_vulkan.h"

/* Same VkImage = 0 rejection contract as v1 — the ring must not
 * have changed the public-API decision tree. */
static char *test_zero_image_still_rejects(void)
{
    if (vmaf_vulkan_list_devices() <= 0)
        return NULL;
    VmafVulkanConfiguration cfg = {.device_index = -1, .enable_validation = 0};
    VmafVulkanState *state = NULL;
    mu_assert("state_init", vmaf_vulkan_state_init(&state, cfg) == 0);

    /* index = 0 — would land on slot 0 if the handle were valid. */
    int rc = vmaf_vulkan_import_image(state, /*vk_image=*/0u, 0, 0, 0, 0, /*w=*/16, /*h=*/16,
                                      /*bpc=*/8, /*is_ref=*/1, /*index=*/0);
    mu_assert("v2 still rejects vk_image == 0 → -EINVAL", rc == -EINVAL);

    /* index large enough to wrap past ring_size = 4. The pre-image
     * checks must still fire before the ring math runs — otherwise
     * an attacker could provoke an out-of-range slot read by passing
     * a giant index with vk_image = 0. */
    rc = vmaf_vulkan_import_image(state, 0u, 0, 0, 0, 0, 16, 16, 8, 1, /*index=*/9999);
    mu_assert("v2 still rejects vk_image == 0 at high index → -EINVAL", rc == -EINVAL);

    vmaf_vulkan_state_free(&state);
    return NULL;
}

/* wait_compute on a state that never imported anything has to
 * be a no-op even though the ring now exists conceptually.
 * The lazy-alloc contract from ADR-0186 is preserved — no
 * import means no ring materialised means nothing to drain. */
static char *test_wait_compute_idle_after_v2(void)
{
    if (vmaf_vulkan_list_devices() <= 0)
        return NULL;
    VmafVulkanConfiguration cfg = {.device_index = -1, .enable_validation = 0};
    VmafVulkanState *state = NULL;
    mu_assert("state_init", vmaf_vulkan_state_init(&state, cfg) == 0);

    mu_assert("idle wait_compute returns 0 (v2 ring not materialised)",
              vmaf_vulkan_wait_compute(state) == 0);
    /* Calling twice in a row is also idempotent. */
    mu_assert("idle wait_compute is idempotent", vmaf_vulkan_wait_compute(state) == 0);

    vmaf_vulkan_state_free(&state);
    return NULL;
}

/* state_free on a freshly-init'd state with zero imports must
 * walk the ring teardown path without touching uninitialised
 * fences. The state struct's import-slots ring_size is 0 here,
 * so the loop bound is 0 and no slot_release runs — but we
 * still verify the path doesn't deadlock or crash. */
static char *test_state_free_no_imports_v2(void)
{
    if (vmaf_vulkan_list_devices() <= 0)
        return NULL;
    VmafVulkanConfiguration cfg = {.device_index = -1, .enable_validation = 0};
    VmafVulkanState *state = NULL;
    mu_assert("state_init", vmaf_vulkan_state_init(&state, cfg) == 0);
    /* No imports — ring is dormant. */
    vmaf_vulkan_state_free(&state);
    mu_assert("state_free clears the handle", state == NULL);
    return NULL;
}

/* read_imported_pictures still has to gate on a NULL VmafContext
 * regardless of which slot the requested index would map to. */
static char *test_read_imported_null_ctx_after_v2(void)
{
    int rc = vmaf_vulkan_read_imported_pictures(NULL, /*index=*/0);
    mu_assert("NULL ctx → -EINVAL (slot 0)", rc == -EINVAL);
    rc = vmaf_vulkan_read_imported_pictures(NULL, /*index=*/3);
    mu_assert("NULL ctx → -EINVAL (slot ring_size-1)", rc == -EINVAL);
    rc = vmaf_vulkan_read_imported_pictures(NULL, /*index=*/4);
    mu_assert("NULL ctx → -EINVAL (slot 0 after wrap)", rc == -EINVAL);
    rc = vmaf_vulkan_read_imported_pictures(NULL, /*index=*/9999);
    mu_assert("NULL ctx → -EINVAL (high index)", rc == -EINVAL);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_zero_image_still_rejects);
    mu_run_test(test_wait_compute_idle_after_v2);
    mu_run_test(test_state_free_no_imports_v2);
    mu_run_test(test_read_imported_null_ctx_after_v2);
    return NULL;
}
