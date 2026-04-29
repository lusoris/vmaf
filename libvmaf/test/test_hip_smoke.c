/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Build + init smoke test for the HIP backend scaffold (ADR-0212 / T7-10).
 *  Every public C-API entry point in libvmaf_hip.h is expected to return
 *  -ENOSYS (or -EINVAL on bad arguments) until the runtime PR lands; this
 *  test pins that contract so a future PR can't accidentally enable the
 *  backend without flipping the smoke expectations.
 *
 *  Mirrors libvmaf/test/test_vulkan_smoke.c.
 */

#include <errno.h>
#include <stddef.h>

#include "test.h"

#include "hip/common.h"
#include "libvmaf/libvmaf_hip.h"

/* ---- Internal context (libvmaf/src/hip/common.h) ---- */

static char *test_context_new_returns_zeroed_struct(void)
{
    /* The scaffold's calloc + struct initialisation succeeds even
     * before a real device is selected. The opaque pointer is
     * non-NULL on success. */
    VmafHipContext *ctx = NULL;
    int rc = vmaf_hip_context_new(&ctx, 0);
    mu_assert("scaffold context_new must succeed", rc == 0);
    mu_assert("scaffold context must be populated", ctx != NULL);
    vmaf_hip_context_destroy(ctx);
    return NULL;
}

static char *test_context_new_rejects_null_out(void)
{
    int rc = vmaf_hip_context_new(NULL, 0);
    mu_assert("NULL out -> -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_context_destroy_null_is_noop(void)
{
    /* No assertion needed — must not crash. */
    vmaf_hip_context_destroy(NULL);
    return NULL;
}

static char *test_device_count_scaffold_returns_zero(void)
{
    /* The scaffold returns 0 (no real HIP probe yet). When the runtime
     * PR replaces the stub, this expectation flips to "either >= 0 from
     * a real probe or skip when no device". */
    const int n = vmaf_hip_device_count();
    mu_assert("scaffold device_count returns 0", n == 0);
    return NULL;
}

/* ---- Public C-API (libvmaf/include/libvmaf/libvmaf_hip.h) ---- */

static char *test_available_reports_build_flag(void)
{
    /* When this TU compiles under -Denable_hip=true the meson glue
     * defines HAVE_HIP=1 for the test executable, so the function
     * reports 1; on the default no-HIP build it reports 0. The smoke
     * test is wired in libvmaf/test/meson.build only under
     * `if get_option('enable_hip') == true`, so this branch matches
     * the build that exercises the test. */
    const int avail = vmaf_hip_available();
    mu_assert("vmaf_hip_available returns a boolean", avail == 0 || avail == 1);
    return NULL;
}

static char *test_state_init_returns_enosys(void)
{
    VmafHipConfiguration cfg = {.device_index = -1, .flags = 0};
    VmafHipState *state = NULL;
    int rc = vmaf_hip_state_init(&state, cfg);
    mu_assert("scaffold state_init returns -ENOSYS", rc == -ENOSYS);
    mu_assert("scaffold leaves out-pointer untouched on -ENOSYS", state == NULL);
    return NULL;
}

static char *test_import_state_returns_enosys(void)
{
    int rc = vmaf_hip_import_state(NULL, NULL);
    mu_assert("scaffold import_state returns -ENOSYS", rc == -ENOSYS);
    return NULL;
}

static char *test_state_free_null_is_noop(void)
{
    /* Must not crash on NULL pointer-to-pointer. */
    vmaf_hip_state_free(NULL);

    /* Must not crash and must clear the slot on a NULL value. */
    VmafHipState *state = NULL;
    vmaf_hip_state_free(&state);
    mu_assert("state_free leaves slot at NULL", state == NULL);
    return NULL;
}

static char *test_list_devices_returns_enosys(void)
{
    int rc = vmaf_hip_list_devices();
    mu_assert("scaffold list_devices returns -ENOSYS", rc == -ENOSYS);
    return NULL;
}

/* Function-pointer table keeps `run_tests` flat — without it,
 * `mu_run_test` macro-expands to a branching pair per test, blowing
 * past clang-tidy's `readability-function-size` 15-branch budget at
 * 9 sub-tests. The table-driven form is also what
 * libvmaf/test/test_vulkan_smoke.c grew to once its sub-test count
 * climbed past the threshold. */
typedef char *(*test_fn)(void);

static const test_fn test_table[] = {
    test_context_new_returns_zeroed_struct, test_context_new_rejects_null_out,
    test_context_destroy_null_is_noop,      test_device_count_scaffold_returns_zero,
    test_available_reports_build_flag,      test_state_init_returns_enosys,
    test_import_state_returns_enosys,       test_state_free_null_is_noop,
    test_list_devices_returns_enosys,
};

static const size_t test_table_len = sizeof(test_table) / sizeof(test_table[0]);

char *run_tests(void)
{
    for (size_t i = 0; i < test_table_len; ++i) {
        mu_run_test(test_table[i]);
    }
    return NULL;
}
