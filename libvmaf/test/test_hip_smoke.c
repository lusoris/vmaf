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
#include <stdint.h>
#include <string.h>

#include "test.h"

#include "feature/feature_extractor.h"
#include "hip/common.h"
#include "hip/kernel_template.h"
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

/* ---- Kernel-template helpers (T7-10 first consumer / ADR-0241) ---- */
/*
 * Pin the scaffold contract for `hip/kernel_template.h`: every helper
 * returns -ENOSYS while the runtime PR (T7-10b) is pending. The
 * runtime PR flips these expectations to "0 on success / negative
 * errno on a real HIP failure"; the test then exercises the full
 * lifecycle against a real device. Until then the -ENOSYS pin is
 * the bit-rot guard.
 */

static char *test_kernel_lifecycle_init_returns_enosys(void)
{
    VmafHipKernelLifecycle lc = {0};
    const int rc = vmaf_hip_kernel_lifecycle_init(&lc, NULL);
    mu_assert("scaffold kernel_lifecycle_init returns -ENOSYS", rc == -ENOSYS);
    mu_assert("scaffold leaves stream handle at 0", lc.str == 0);
    mu_assert("scaffold leaves submit event at 0", lc.submit == 0);
    mu_assert("scaffold leaves finished event at 0", lc.finished == 0);
    return NULL;
}

static char *test_kernel_readback_alloc_returns_enosys(void)
{
    VmafHipKernelReadback rb = {0};
    const int rc = vmaf_hip_kernel_readback_alloc(&rb, NULL, sizeof(uint64_t));
    mu_assert("scaffold kernel_readback_alloc returns -ENOSYS", rc == -ENOSYS);
    mu_assert("scaffold leaves device pointer at NULL", rb.device == NULL);
    mu_assert("scaffold leaves host_pinned pointer at NULL", rb.host_pinned == NULL);
    mu_assert("scaffold records requested byte count", rb.bytes == sizeof(uint64_t));
    return NULL;
}

static char *test_kernel_lifecycle_close_is_noop(void)
{
    /* Close on an all-zero lifecycle must succeed (mirrors the CUDA
     * twin's "safe to call on a partially-initialised lifecycle"
     * contract). The scaffold body has nothing to release; the
     * runtime PR will sequence hipStreamDestroy / hipEventDestroy. */
    VmafHipKernelLifecycle lc = {0};
    const int rc = vmaf_hip_kernel_lifecycle_close(&lc, NULL);
    mu_assert("scaffold kernel_lifecycle_close returns 0 on zero handles", rc == 0);
    return NULL;
}

static char *test_kernel_readback_free_is_noop(void)
{
    /* Same partially-allocated contract as the lifecycle close. */
    VmafHipKernelReadback rb = {0};
    const int rc = vmaf_hip_kernel_readback_free(&rb, NULL);
    mu_assert("scaffold kernel_readback_free returns 0 on zero handles", rc == 0);
    return NULL;
}

/* ---- First consumer extractor registration (T7-10 / ADR-0241) ---- */

static char *test_psnr_hip_extractor_registered(void)
{
    /* The first-consumer PR (T7-10 / ADR-0241) flips the registration
     * posture: previously the docs noted "kernel stubs intentionally do
     * not register with the feature registry"; for `psnr_hip` we now
     * register the extractor so callers asking by name get a clean
     * "found but runtime not ready" surface. The runtime PR (T7-10b)
     * keeps this assertion green and tightens it to "init returns 0
     * with a real device". */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("psnr_hip");
    mu_assert("psnr_hip extractor must be registered", fex != NULL);
    mu_assert("psnr_hip extractor name matches", strcmp(fex->name, "psnr_hip") == 0);
    return NULL;
}

static char *test_ciede_hip_extractor_registered(void)
{
    /* Third-consumer PR (ADR-0257) extends the same registration
     * contract to `ciede_hip`: extractor is found by name, with the
     * matching `.name` string. `init()` is not invoked here — the
     * scaffold returns -ENOSYS at that layer; the registration smoke
     * test only pins the lookup contract. The runtime PR (T7-10b)
     * keeps this assertion green and tightens it to "init returns 0
     * with a real device". */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("ciede_hip");
    mu_assert("ciede_hip extractor must be registered", fex != NULL);
    mu_assert("ciede_hip extractor name matches", strcmp(fex->name, "ciede_hip") == 0);
    return NULL;
}

static char *test_float_moment_hip_extractor_registered(void)
{
    /* Fourth-consumer PR (ADR-0258) extends the same registration
     * contract to `float_moment_hip`: extractor is found by name with
     * the matching `.name` string. Pins the registration shape — the
     * runtime PR (T7-10b) keeps the assertion green. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("float_moment_hip");
    mu_assert("float_moment_hip extractor must be registered", fex != NULL);
    mu_assert("float_moment_hip extractor name matches",
              strcmp(fex->name, "float_moment_hip") == 0);
    return NULL;
}

/* ---- Fifth/sixth consumer extractor registration (T7-10b /
 * ADR-0266 / ADR-0267) ---- */

static char *test_float_ansnr_hip_extractor_registered(void)
{
    /* Fifth-consumer PR (ADR-0266) extends the same registration
     * contract to `float_ansnr_hip`: extractor is found by name,
     * with the matching `.name` string. `init()` is not invoked
     * here — the scaffold returns -ENOSYS at that layer; the
     * registration smoke test only pins the lookup contract. The
     * runtime PR (T7-10b) keeps this assertion green and tightens
     * it to "init returns 0 with a real device". */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("float_ansnr_hip");
    mu_assert("float_ansnr_hip extractor must be registered", fex != NULL);
    mu_assert("float_ansnr_hip extractor name matches", strcmp(fex->name, "float_ansnr_hip") == 0);
    return NULL;
}

static char *test_motion_v2_hip_extractor_registered(void)
{
    /* Sixth-consumer PR (ADR-0267) extends the same registration
     * contract to `motion_v2_hip`. The CUDA twin's name is
     * `motion_v2_cuda` (without the `integer_` prefix on the
     * extractor name even though the source file is
     * `integer_motion_v2_cuda.c`); the HIP twin keeps the same
     * naming choice. Smoke test only pins the lookup contract;
     * the runtime PR (T7-10b) wires the kernel and tightens this
     * assertion to "init returns 0 with a real device". */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion_v2_hip");
    mu_assert("motion_v2_hip extractor must be registered", fex != NULL);
    mu_assert("motion_v2_hip extractor name matches", strcmp(fex->name, "motion_v2_hip") == 0);
    return NULL;
}

static char *test_float_motion_hip_extractor_registered(void)
{
    /* Seventh-consumer PR (ADR-0273) extends the same registration
     * contract to `float_motion_hip`: extractor is found by name with
     * the matching `.name` string, and carries the TEMPORAL flag bit
     * (load-bearing for the feature engine's collect-before-next-submit
     * scheduling that motion-class metrics rely on). The runtime PR
     * (T7-10b) keeps these assertions green. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("float_motion_hip");
    mu_assert("float_motion_hip extractor must be registered", fex != NULL);
    mu_assert("float_motion_hip extractor name matches",
              strcmp(fex->name, "float_motion_hip") == 0);
    mu_assert("float_motion_hip extractor must carry the TEMPORAL flag",
              (fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL) != 0);
    return NULL;
}

static char *test_float_ssim_hip_extractor_registered(void)
{
    /* Eighth-consumer PR (ADR-0274) extends the same registration
     * contract to `float_ssim_hip`: extractor is found by name with
     * the matching `.name` string. Pins the two-dispatch shape via
     * `chars.n_dispatches_per_frame == 2` so the runtime PR's
     * dispatch-counter accounting (and the `places=4` cross-backend
     * gate's per-frame-cost model) inherits a consumer that exercises
     * the multi-dispatch path. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("float_ssim_hip");
    mu_assert("float_ssim_hip extractor must be registered", fex != NULL);
    mu_assert("float_ssim_hip extractor name matches", strcmp(fex->name, "float_ssim_hip") == 0);
    mu_assert("float_ssim_hip extractor reports two dispatches per frame",
              fex->chars.n_dispatches_per_frame == 2);
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
    test_context_new_returns_zeroed_struct,
    test_context_new_rejects_null_out,
    test_context_destroy_null_is_noop,
    test_device_count_scaffold_returns_zero,
    test_available_reports_build_flag,
    test_state_init_returns_enosys,
    test_import_state_returns_enosys,
    test_state_free_null_is_noop,
    test_list_devices_returns_enosys,
    /* T7-10 first consumer (ADR-0241) */
    test_kernel_lifecycle_init_returns_enosys,
    test_kernel_readback_alloc_returns_enosys,
    test_kernel_lifecycle_close_is_noop,
    test_kernel_readback_free_is_noop,
    test_psnr_hip_extractor_registered,
    /* T7-10b third + fourth consumers (ADR-0257 / ADR-0258) */
    test_ciede_hip_extractor_registered,
    test_float_moment_hip_extractor_registered,
    /* T7-10b fifth/sixth consumers (ADR-0266 / ADR-0267) */
    test_float_ansnr_hip_extractor_registered,
    test_motion_v2_hip_extractor_registered,
    /* T7-10b seventh + eighth consumers (ADR-0273 / ADR-0274) */
    test_float_motion_hip_extractor_registered,
    test_float_ssim_hip_extractor_registered,
};

static const size_t test_table_len = sizeof(test_table) / sizeof(test_table[0]);

char *run_tests(void)
{
    for (size_t i = 0; i < test_table_len; ++i) {
        mu_run_test(test_table[i]);
    }
    return NULL;
}
