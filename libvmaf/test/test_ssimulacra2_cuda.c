/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

/*
 * CUDA parity test for ssimulacra2_cuda.
 *
 * Goals (ADR-0214 / ADR-0138 / ADR-0139):
 *   1. Verify vmaf_fex_ssimulacra2_cuda is discoverable by name.
 *   2. Init the CUDA extractor on a synthetic 256×144 YUV420P 8-bpc frame.
 *   3. Assert that "ssimulacra2" is finite. Note: SSIMULACRA2 scores
 *      can be negative for very distorted content; for identical frames
 *      the score approaches 100. We only gate on isfinite().
 *   4. Skip cleanly when no CUDA device is present.
 *
 * CPU↔CUDA bit-exactness (places=4) is enforced by the cross-backend
 * scoring gate (validate-scores / ADR-0214). A full golden assertion
 * belongs in python/test/ per CLAUDE.md §8.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "config.h"
#include "test.h"

#if HAVE_CUDA

#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/libvmaf.h"
#include "feature/feature_extractor.h"

static VmafCudaState *cuda_state = NULL;
static int cuda_init_failed = 0;

/* ------------------------------------------------------------------ */
/* Setup: initialise CUDA state (skip if no GPU present).              */
/* ------------------------------------------------------------------ */
static char *test_ssimulacra2_cuda_setup(void)
{
    VmafCudaConfiguration cfg = {.cu_ctx = NULL};
    int err = vmaf_cuda_state_init(&cuda_state, cfg);
    if (err) {
        (void)fprintf(stderr,
                      "  [SKIP] CUDA state init failed (err=%d), "
                      "no GPU available — skipping ssimulacra2_cuda tests\n",
                      err);
        cuda_init_failed = 1;
        cuda_state = NULL;
        return NULL;
    }
    mu_assert("cuda state should be non-NULL", cuda_state != NULL);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Test 1: registration — ssimulacra2_cuda is discoverable.            */
/* ------------------------------------------------------------------ */
static char *test_ssimulacra2_cuda_registration(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("ssimulacra2_cuda");
    mu_assert("vmaf_fex_ssimulacra2_cuda should be findable by name", fex != NULL);
    if (fex) {
        mu_assert("fex name should be ssimulacra2_cuda",
                  strcmp(fex->name, "ssimulacra2_cuda") == 0);
        mu_assert("fex flags should include VMAF_FEATURE_EXTRACTOR_CUDA",
                  (fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA) != 0);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Test 2: end-to-end smoke — init/submit/collect/close without crash. */
/*                                                                      */
/* Identical flat-grey frames are used. SSIMULACRA2 scores can range   */
/* well outside [0, 100] for pathological input; the only contract      */
/* asserted here is finiteness (not a sign or magnitude bound).        */
/* ------------------------------------------------------------------ */
static char *test_ssimulacra2_cuda_smoke(void)
{
    if (cuda_init_failed) {
        (void)fprintf(stderr, "  [SKIP] test_ssimulacra2_cuda_smoke (no GPU)\n");
        return NULL;
    }

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init should succeed", err == 0);
    if (err)
        return NULL;

    err = vmaf_cuda_import_state(vmaf, cuda_state);
    mu_assert("vmaf_cuda_import_state should succeed", err == 0);
    if (err) {
        (void)vmaf_close(vmaf);
        return NULL;
    }

    err = vmaf_use_feature(vmaf, "ssimulacra2_cuda", NULL);
    mu_assert("vmaf_use_feature(ssimulacra2_cuda) should succeed", err == 0);
    if (err) {
        (void)vmaf_close(vmaf);
        return NULL;
    }

    /* 256×144 YUV420P 8-bpc — small enough to stay <5 s on any device. */
    static const unsigned W = 256u;
    static const unsigned H = 144u;
    VmafPicture ref_pic, dis_pic;
    err = vmaf_picture_alloc(&ref_pic, VMAF_PIX_FMT_YUV420P, 8, W, H);
    mu_assert("ref picture alloc", err == 0);
    err = vmaf_picture_alloc(&dis_pic, VMAF_PIX_FMT_YUV420P, 8, W, H);
    mu_assert("dis picture alloc", err == 0);

    for (unsigned p = 0; p < 3; p++) {
        unsigned pw = (p == 0) ? W : W / 2u;
        unsigned ph = (p == 0) ? H : H / 2u;
        uint8_t *rp = (uint8_t *)ref_pic.data[p];
        uint8_t *dp = (uint8_t *)dis_pic.data[p];
        for (unsigned r = 0; r < ph; r++) {
            memset(rp + r * (size_t)ref_pic.stride[p], 128u, pw);
            memset(dp + r * (size_t)dis_pic.stride[p], 128u, pw);
        }
    }

    err = vmaf_read_pictures(vmaf, &ref_pic, &dis_pic, 0);
    mu_assert("vmaf_read_pictures should succeed", err == 0);

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("flush vmaf_read_pictures should succeed", err == 0);

    double score = -1e300;
    err = vmaf_feature_score_at_index(vmaf, "ssimulacra2", &score, 0);
    mu_assert("ssimulacra2 score retrieval should succeed", err == 0);
    /* SSIMULACRA2 scores may be negative for badly distorted content;
     * identical frames should produce ~100. Only finiteness is required. */
    mu_assert("ssimulacra2 score should be finite", isfinite(score));

    (void)vmaf_close(vmaf);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Teardown.                                                            */
/* ------------------------------------------------------------------ */
static char *test_ssimulacra2_cuda_teardown(void)
{
    if (cuda_state != NULL) {
        (void)vmaf_cuda_state_free(cuda_state);
        cuda_state = NULL;
    }
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_ssimulacra2_cuda_setup);
    mu_run_test(test_ssimulacra2_cuda_registration);
    mu_run_test(test_ssimulacra2_cuda_smoke);
    mu_run_test(test_ssimulacra2_cuda_teardown);
    return NULL;
}

#else /* !HAVE_CUDA */

char *run_tests(void)
{
    return NULL;
}

#endif /* HAVE_CUDA */
