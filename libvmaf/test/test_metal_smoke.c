/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Build + init smoke test for the Metal backend scaffold (ADR-0361 / T8-1).
 *  Every public C-API entry point in libvmaf_metal.h is expected to
 *  return -ENOSYS (or -EINVAL on bad arguments) until the runtime PR
 *  lands; this test pins that contract so a future PR can't
 *  accidentally enable the backend without flipping the smoke
 *  expectations.
 *
 *  Mirrors libvmaf/test/test_hip_smoke.c.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "test.h"

#include "feature/feature_extractor.h"
#include "libvmaf/libvmaf_metal.h"
#include "metal/common.h"
#include "metal/kernel_template.h"

/* ---- Internal context (libvmaf/src/metal/common.h) ---- */

static char *test_context_new_returns_zeroed_struct(void)
{
    /* The scaffold's calloc + struct initialisation succeeds even
     * before a real device is selected. The opaque pointer is
     * non-NULL on success. */
    VmafMetalContext *ctx = NULL;
    int rc = vmaf_metal_context_new(&ctx, 0);
    mu_assert("scaffold context_new must succeed", rc == 0);
    mu_assert("scaffold context must be populated", ctx != NULL);
    vmaf_metal_context_destroy(ctx);
    return NULL;
}

static char *test_context_new_rejects_null_out(void)
{
    int rc = vmaf_metal_context_new(NULL, 0);
    mu_assert("NULL out -> -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_context_destroy_null_is_noop(void)
{
    /* No assertion needed — must not crash. */
    vmaf_metal_context_destroy(NULL);
    return NULL;
}

static char *test_device_count_scaffold_returns_zero(void)
{
    /* The scaffold returns 0 (no real Metal probe yet). When the
     * runtime PR replaces the stub, this expectation flips to
     * "either >= 0 from a real probe or skip when no device". */
    const int n = vmaf_metal_device_count();
    mu_assert("scaffold device_count returns 0", n == 0);
    return NULL;
}

/* ---- Public C-API (libvmaf/include/libvmaf/libvmaf_metal.h) ---- */

static char *test_available_reports_build_flag(void)
{
    /* When this TU compiles under -Denable_metal=enabled the meson
     * glue defines HAVE_METAL=1 for the test executable, so the
     * function reports 1; on the non-enabled build it reports 0.
     * The smoke test is wired in libvmaf/test/meson.build only
     * under `if get_option('enable_metal').enabled() or auto-resolve`,
     * so this branch matches the build that exercises the test. */
    const int avail = vmaf_metal_available();
    mu_assert("vmaf_metal_available returns a boolean", avail == 0 || avail == 1);
    return NULL;
}

static char *test_state_init_returns_enosys(void)
{
    VmafMetalConfiguration cfg = {.device_index = -1, .flags = 0};
    VmafMetalState *state = NULL;
    int rc = vmaf_metal_state_init(&state, cfg);
    mu_assert("scaffold state_init returns -ENOSYS", rc == -ENOSYS);
    mu_assert("scaffold leaves out-pointer untouched on -ENOSYS", state == NULL);
    return NULL;
}

static char *test_import_state_returns_enosys(void)
{
    int rc = vmaf_metal_import_state(NULL, NULL);
    mu_assert("scaffold import_state returns -ENOSYS", rc == -ENOSYS);
    return NULL;
}

static char *test_state_free_null_is_noop(void)
{
    /* Must not crash on NULL pointer-to-pointer. */
    vmaf_metal_state_free(NULL);

    /* Must not crash and must clear the slot on a NULL value. */
    VmafMetalState *state = NULL;
    vmaf_metal_state_free(&state);
    mu_assert("state_free leaves slot at NULL", state == NULL);
    return NULL;
}

static char *test_list_devices_returns_enosys(void)
{
    int rc = vmaf_metal_list_devices();
    mu_assert("scaffold list_devices returns -ENOSYS", rc == -ENOSYS);
    return NULL;
}

/* ---- Kernel-template helpers (T8-1 first consumer / ADR-0361) ---- */
/*
 * Pin the scaffold contract for `metal/kernel_template.h`: every
 * helper returns -ENOSYS while the runtime PR (T8-1b) is pending.
 * The runtime PR flips these expectations to "0 on success /
 * negative errno on a real Metal failure"; the test then exercises
 * the full lifecycle against a real device. Until then the -ENOSYS
 * pin is the bit-rot guard.
 */

static char *test_kernel_lifecycle_init_returns_enosys(void)
{
    VmafMetalKernelLifecycle lc = {0};
    const int rc = vmaf_metal_kernel_lifecycle_init(&lc, NULL);
    mu_assert("scaffold kernel_lifecycle_init returns -ENOSYS", rc == -ENOSYS);
    mu_assert("scaffold leaves cmd_queue handle at 0", lc.cmd_queue == 0);
    mu_assert("scaffold leaves submit event at 0", lc.submit == 0);
    mu_assert("scaffold leaves finished event at 0", lc.finished == 0);
    return NULL;
}

static char *test_kernel_buffer_alloc_returns_enosys(void)
{
    VmafMetalKernelBuffer buf = {0};
    const int rc = vmaf_metal_kernel_buffer_alloc(&buf, NULL, sizeof(uint64_t));
    mu_assert("scaffold kernel_buffer_alloc returns -ENOSYS", rc == -ENOSYS);
    mu_assert("scaffold leaves MTLBuffer slot at 0", buf.buffer == 0);
    mu_assert("scaffold leaves host_view at NULL", buf.host_view == NULL);
    mu_assert("scaffold records requested byte count", buf.bytes == sizeof(uint64_t));
    return NULL;
}

static char *test_kernel_lifecycle_close_is_noop(void)
{
    /* Close on an all-zero lifecycle must succeed (mirrors the HIP
     * twin's "safe to call on a partially-initialised lifecycle"
     * contract). The scaffold body has nothing to release; the
     * runtime PR will sequence waitUntilCompleted / release queue /
     * release events. */
    VmafMetalKernelLifecycle lc = {0};
    const int rc = vmaf_metal_kernel_lifecycle_close(&lc, NULL);
    mu_assert("scaffold kernel_lifecycle_close returns 0 on zero handles", rc == 0);
    return NULL;
}

static char *test_kernel_buffer_free_is_noop(void)
{
    /* Same partially-allocated contract as the lifecycle close. */
    VmafMetalKernelBuffer buf = {0};
    const int rc = vmaf_metal_kernel_buffer_free(&buf, NULL);
    mu_assert("scaffold kernel_buffer_free returns 0 on zero handles", rc == 0);
    return NULL;
}

/* ---- First consumer extractor registration (T8-1 / ADR-0361) ---- */

static char *test_motion_v2_metal_extractor_registered(void)
{
    /* The first-consumer scaffold (T8-1 / ADR-0361) registers
     * `motion_v2_metal` so callers asking by name get a clean
     * "found but runtime not ready" surface rather than "no such
     * extractor". The runtime PR (T8-1b) keeps this assertion green
     * and tightens it to "init returns 0 with a real device". The
     * TEMPORAL flag is load-bearing for the feature engine's
     * collect-before-next-submit scheduling that motion-class
     * metrics rely on. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion_v2_metal");
    mu_assert("motion_v2_metal extractor must be registered", fex != NULL);
    mu_assert("motion_v2_metal extractor name matches", strcmp(fex->name, "motion_v2_metal") == 0);
    mu_assert("motion_v2_metal extractor must carry the TEMPORAL flag",
              (fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL) != 0);
    return NULL;
}

/* Function-pointer table keeps `run_tests` flat — same pattern as
 * `test_hip_smoke.c`; without it `mu_run_test` macro-expands to a
 * branching pair per test, blowing past clang-tidy's
 * `readability-function-size` 15-branch budget once the sub-test
 * count climbs. */
typedef char *(*test_fn)(void);

static const test_fn test_table[] = {
    test_context_new_returns_zeroed_struct,
    test_context_new_rejects_null_out,
    test_context_destroy_null_is_noop,
    test_device_count_scaffold_returns_zero,
    test_available_reports_build_flag,
    test_state_init_returns_enosys,
    test_import_state_returns_enosys,
    test_state_free_null_is_noop,
    test_list_devices_returns_enosys,
    /* T8-1 kernel-template (ADR-0361) */
    test_kernel_lifecycle_init_returns_enosys,
    test_kernel_buffer_alloc_returns_enosys,
    test_kernel_lifecycle_close_is_noop,
    test_kernel_buffer_free_is_noop,
    /* T8-1 first consumer (ADR-0361) */
    test_motion_v2_metal_extractor_registered,
};

static const size_t test_table_len = sizeof(test_table) / sizeof(test_table[0]);

char *run_tests(void)
{
    for (size_t i = 0; i < test_table_len; ++i) {
        mu_run_test(test_table[i]);
    }
    return NULL;
}
