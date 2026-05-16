/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  T3-15(c) / ADR-0219: cross-backend parity gate for motion3_score on
 *  the Vulkan backend.
 *
 *  Verifies that `motion_vulkan` emits `VMAF_integer_feature_motion3_score`
 *  values that match the CPU `motion` extractor at `places=4` (|delta| <
 *  5e-5) across a 6-frame sequence of synthetic YUV420P frames.
 *
 *  Design:
 *  - Runs two independent VmafContext instances side-by-side: one with the
 *    CPU `motion` extractor and one with the Vulkan `motion_vulkan`
 *    extractor.  Both receive byte-identical frame pairs.
 *  - Frame content is deterministic pseudo-random (XOR-shift), chosen so
 *    that inter-frame pixel changes drive non-trivial motion scores (not
 *    all-zero or all-identical).
 *  - Gracefully skips all device-dependent assertions when the runtime has
 *    no Vulkan-capable device (e.g. lavapipe-less CI containers) — the same
 *    skip-on-no-GPU pattern used by test_vulkan_smoke.c and
 *    test_vulkan_pic_preallocation.c.
 *  - motion3_score at frame 0 is always 0.0 by the delayed-write contract
 *    (the extractor writes frame n-1's score when it processes frame n);
 *    frames 1..4 exercise the moving-average / blend post-processing path.
 *    Frame 5 is written by flush().
 *
 *  The 5-frame window mode (`motion_five_frame_window=true`) is NOT tested
 *  here — both GPU and CPU gate it at init() with -ENOTSUP/-EINVAL
 *  respectively (ADR-0219 §Decision).
 *
 *  The places=4 contract matches the integer pipeline tolerance from
 *  cross_backend_parity_gate.py (FEATURE_TOLERANCE["motion"] = 5e-5).
 *
 *  See:
 *   - docs/adr/0219-motion3-gpu-coverage.md (design decision)
 *   - docs/adr/0177-vulkan-motion-kernel.md (Vulkan motion kernel)
 *   - scripts/ci/cross_backend_parity_gate.py (CI gate; also covers motion3)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_vulkan.h"
#include "libvmaf/picture.h"

/* Internal header for VmafFeatureExtractor + vmaf_get_feature_extractor_by_name.
 * Other GPU backend tests (test_sycl.c, test_vulkan_smoke.c) use the same
 * include pattern — the internal API is not exposed in the public headers.  */
#include "feature/feature_extractor.h"

/* --------------------------------------------------------------------- */
/* Tuneable constants                                                      */
/* --------------------------------------------------------------------- */

enum {
    T_WIDTH = 64, /* small enough to run fast on any GPU                */
    T_HEIGHT = 64,
    T_BPC = 8,
    T_FRAMES = 6, /* ≥ 3 to exercise the moving-average path             */
};

/* Absolute tolerance — same as cross_backend_parity_gate.py               */
static const double MOTION3_TOL = 5e-5; /* places=4 */

/* The motion3 score at index 0 is always 0.0 by the delayed-write          */
/* contract (the extractor back-fills index n-1 when it processes frame n). */
static const unsigned FRAME_ZERO_SKIP = 1u;

/* --------------------------------------------------------------------- */
/* Helpers                                                                 */
/* --------------------------------------------------------------------- */

/* XOR-shift PRNG — no stdlib rand() to avoid locale / seed coupling.     */
static uint32_t xorshift(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *s = x;
    return x;
}

/* Fill a Y plane with pseudo-random 8-bit values derived from `seed`.
 * Each call advances the seed state so consecutive frames differ.        */
static void fill_plane(uint8_t *plane, size_t stride, unsigned w, unsigned h, uint32_t *seed)
{
    for (unsigned y = 0; y < h; ++y) {
        for (unsigned x = 0; x < w; ++x)
            plane[(ptrdiff_t)y * (ptrdiff_t)stride + x] = (uint8_t)(xorshift(seed) & 0xFFu);
    }
}

/* Allocate and fill one YUV420P frame.  seed is advanced in-place.       */
static int make_frame(VmafPicture *pic, unsigned w, unsigned h, uint32_t *seed)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, T_BPC, w, h);
    if (err)
        return err;
    fill_plane((uint8_t *)pic->data[0], pic->stride[0], w, h, seed);
    fill_plane((uint8_t *)pic->data[1], pic->stride[1], w / 2u, h / 2u, seed);
    fill_plane((uint8_t *)pic->data[2], pic->stride[2], w / 2u, h / 2u, seed);
    return 0;
}

/* --------------------------------------------------------------------- */
/* CPU reference run                                                       */
/* --------------------------------------------------------------------- */

/* Run the CPU `motion` extractor on T_FRAMES frame pairs and collect
 * per-frame motion3_score into out_scores[0..T_FRAMES-1].
 * Frames are generated with a deterministic seed sequence.               */
static char *run_cpu_motion3(double out_scores[T_FRAMES])
{
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 1,
    };
    VmafContext *vmaf = NULL;
    mu_assert("cpu: vmaf_init", vmaf_init(&vmaf, cfg) == 0);
    mu_assert("cpu: vmaf_use_feature motion", vmaf_use_feature(vmaf, "motion", NULL) == 0);

    uint32_t seed_ref = 0xDEAD1234u;
    uint32_t seed_dist = 0xBEEF5678u;

    for (unsigned i = 0; i < T_FRAMES; ++i) {
        VmafPicture ref = {0};
        VmafPicture dist = {0};
        mu_assert("cpu: alloc ref", make_frame(&ref, T_WIDTH, T_HEIGHT, &seed_ref) == 0);
        mu_assert("cpu: alloc dist", make_frame(&dist, T_WIDTH, T_HEIGHT, &seed_dist) == 0);
        mu_assert("cpu: vmaf_read_pictures", vmaf_read_pictures(vmaf, &ref, &dist, i) == 0);
    }
    mu_assert("cpu: flush", vmaf_read_pictures(vmaf, NULL, NULL, 0) == 0);

    for (unsigned i = 0; i < T_FRAMES; ++i) {
        double sc = 0.0;
        int err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion3_score", &sc, i);
        mu_assert("cpu: score_at_index", err == 0);
        out_scores[i] = sc;
    }

    mu_assert("cpu: vmaf_close", vmaf_close(vmaf) == 0);
    return NULL;
}

/* --------------------------------------------------------------------- */
/* Vulkan run + comparison                                                 */
/* --------------------------------------------------------------------- */

static char *test_vulkan_motion3_parity(void)
{
    /* Skip on hosts with no Vulkan-capable compute device.               */
    if (vmaf_vulkan_list_devices() <= 0) {
        (void)fprintf(stderr, "  [SKIP] test_vulkan_motion3_parity: "
                              "no Vulkan compute device available\n");
        return NULL;
    }

    /* --- CPU reference ------------------------------------------------ */
    double cpu_scores[T_FRAMES];
    char *msg = run_cpu_motion3(cpu_scores);
    if (msg)
        return msg;

    /* --- Vulkan run ---------------------------------------------------- */
    VmafVulkanConfiguration vk_cfg = {.device_index = -1, .enable_validation = 0};
    VmafVulkanState *vk_state = NULL;
    mu_assert("vulkan: state_init", vmaf_vulkan_state_init(&vk_state, vk_cfg) == 0);

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 1,
    };
    VmafContext *vmaf = NULL;
    mu_assert("vulkan: vmaf_init", vmaf_init(&vmaf, cfg) == 0);
    mu_assert("vulkan: import_state", vmaf_vulkan_import_state(vmaf, vk_state) == 0);
    mu_assert("vulkan: vmaf_use_feature motion_vulkan",
              vmaf_use_feature(vmaf, "motion_vulkan", NULL) == 0);

    /* Feed byte-identical frames (same seed sequence as the CPU run).    */
    uint32_t seed_ref = 0xDEAD1234u;
    uint32_t seed_dist = 0xBEEF5678u;

    for (unsigned i = 0; i < T_FRAMES; ++i) {
        VmafPicture ref = {0};
        VmafPicture dist = {0};
        mu_assert("vk: alloc ref", make_frame(&ref, T_WIDTH, T_HEIGHT, &seed_ref) == 0);
        mu_assert("vk: alloc dist", make_frame(&dist, T_WIDTH, T_HEIGHT, &seed_dist) == 0);
        mu_assert("vk: vmaf_read_pictures", vmaf_read_pictures(vmaf, &ref, &dist, i) == 0);
    }
    mu_assert("vk: flush", vmaf_read_pictures(vmaf, NULL, NULL, 0) == 0);

    /* --- Compare per-frame motion3_score at places=4 ------------------- */
    for (unsigned i = FRAME_ZERO_SKIP; i < T_FRAMES; ++i) {
        double vk_sc = 0.0;
        int err =
            vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion3_score", &vk_sc, i);
        mu_assert("vk: score_at_index", err == 0);

        const double delta = fabs(vk_sc - cpu_scores[i]);
        if (delta > MOTION3_TOL) {
            (void)fprintf(stderr, "  motion3_score[%u]: cpu=%.8f vk=%.8f delta=%.3e (tol %.3e)\n",
                          i, cpu_scores[i], vk_sc, delta, MOTION3_TOL);
            mu_assert("motion3_score mismatch vs CPU (places=4)", delta <= MOTION3_TOL);
        }
    }

    mu_assert("vk: vmaf_close", vmaf_close(vmaf) == 0);
    vmaf_vulkan_state_free(&vk_state);
    return NULL;
}

/* --------------------------------------------------------------------- */
/* Registration-only tests (no GPU required)                               */
/* --------------------------------------------------------------------- */

static char *test_motion_vulkan_extractor_registered(void)
{
    /* Verify the extractor is registered at compile time — doesn't need
     * a live GPU device.                                                  */
#if HAVE_VULKAN
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion_vulkan");
    mu_assert("motion_vulkan extractor not registered", fex != NULL);
    mu_assert("wrong name", fex && strcmp(fex->name, "motion_vulkan") == 0);
    mu_assert("TEMPORAL flag", fex && (fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL));
    mu_assert("VULKAN flag", fex && (fex->flags & VMAF_FEATURE_EXTRACTOR_VULKAN));
#else
    (void)fprintf(stderr, "  [SKIP] Vulkan not enabled\n");
#endif
    return NULL;
}

/* --------------------------------------------------------------------- */
/* Test runner                                                             */
/* --------------------------------------------------------------------- */

char *run_tests(void)
{
    mu_run_test(test_motion_vulkan_extractor_registered);
    mu_run_test(test_vulkan_motion3_parity);
    return NULL;
}
