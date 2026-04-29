/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  MobileSal saliency feature extractor — structural + stub-path tests
 *  (T6-2a). Mirrors the test_lpips.c shape; verifies registration
 *  without depending on a real ORT session. Full end-to-end inference
 *  (YUV → ORT run → score append) requires a real ORT build and the
 *  in-tree ``model/tiny/mobilesal.onnx`` placeholder; that path is
 *  exercised by the CLI smoke gate. These unit tests run in every
 *  build and verify:
 *
 *    - the "mobilesal" feature extractor is discoverable by name and
 *      by provided feature name (registration wired correctly);
 *    - init() cleanly declines when no model_path and no env var are
 *      provided (input validation);
 *    - options metadata is well-formed (non-NULL name, non-NULL help).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "feature/feature_extractor.h"
#include "opt.h"

/* MinGW's mingw.org / MSYS2 headers do not expose POSIX setenv/unsetenv under
 * -std=c11 -pedantic. Shim via _putenv_s (MSVCRT) on _WIN32; use the POSIX
 * functions elsewhere. Both shims silently swallow failures — this is a test
 * helper, not a production code path. */
#if defined(_WIN32)
#include <stdlib.h>
static int test_setenv(const char *name, const char *value)
{
    return _putenv_s(name, value);
}
static int test_unsetenv(const char *name)
{
    return _putenv_s(name, "");
}
#else
static int test_setenv(const char *name, const char *value)
{
    return setenv(name, value, 1);
}
static int test_unsetenv(const char *name)
{
    return unsetenv(name);
}
#endif

static char *test_mobilesal_is_registered(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("mobilesal");
    mu_assert("mobilesal extractor must be registered by name", fex != NULL);
    mu_assert("registered extractor has wrong name", !strcmp(fex->name, "mobilesal"));
    mu_assert("mobilesal.init must be set", fex->init != NULL);
    mu_assert("mobilesal.extract must be set", fex->extract != NULL);
    mu_assert("mobilesal.close must be set", fex->close != NULL);
    mu_assert("mobilesal.priv_size must be non-zero", fex->priv_size > 0u);
    return NULL;
}

static char *test_mobilesal_provides_saliency_mean_feature(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_feature_name("saliency_mean", 0);
    mu_assert("'saliency_mean' feature name must resolve to an extractor", fex != NULL);
    mu_assert("'saliency_mean' must map to the mobilesal extractor",
              !strcmp(fex->name, "mobilesal"));
    return NULL;
}

static char *test_mobilesal_options_table_well_formed(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("mobilesal");
    mu_assert("mobilesal extractor missing", fex != NULL);
    mu_assert("mobilesal must expose at least one option", fex->options != NULL);

    int saw_model_path = 0;
    for (const VmafOption *opt = fex->options; opt && opt->name; ++opt) {
        mu_assert("every option must have non-NULL help text", opt->help != NULL);
        if (!strcmp(opt->name, "model_path")) {
            saw_model_path = 1;
            mu_assert("model_path must be a string option", opt->type == VMAF_OPT_TYPE_STRING);
        }
    }
    mu_assert("mobilesal must expose a 'model_path' option", saw_model_path);
    return NULL;
}

static char *test_mobilesal_init_rejects_missing_model(void)
{
    /* With neither model_path option set nor VMAF_MOBILESAL_MODEL_PATH in
     * the environment, init() must cleanly decline rather than crash. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("mobilesal");
    mu_assert("mobilesal extractor missing", fex != NULL);

    void *priv = calloc(1, fex->priv_size);
    mu_assert("priv alloc failed", priv != NULL);
    fex->priv = priv;

    /* Save + clear the env var for the duration of this test. */
    const char *saved = getenv("VMAF_MOBILESAL_MODEL_PATH");
    char *saved_copy = saved ? strdup(saved) : NULL;
    (void)test_unsetenv("VMAF_MOBILESAL_MODEL_PATH");

    int rc = fex->init(fex, VMAF_PIX_FMT_YUV420P, 8u, 64u, 64u);
    mu_assert("init must fail when no model path is provided", rc < 0);

    /* close() must be safe to call after a failed init — the extractor
     * either did full cleanup before returning, or close() tolerates the
     * partial state. Either is acceptable as long as we don't crash. */
    (void)fex->close(fex);

    if (saved_copy) {
        (void)test_setenv("VMAF_MOBILESAL_MODEL_PATH", saved_copy);
        free(saved_copy);
    }
    free(priv);
    fex->priv = NULL;
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_mobilesal_is_registered);
    mu_run_test(test_mobilesal_provides_saliency_mean_feature);
    mu_run_test(test_mobilesal_options_table_well_formed);
    mu_run_test(test_mobilesal_init_rejects_missing_model);
    return NULL;
}
