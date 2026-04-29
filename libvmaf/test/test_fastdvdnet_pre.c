/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  FastDVDnet temporal pre-filter (T6-7) — structural + stub-path tests.
 *
 *  Mirrors the test_lpips.c shape: covers extractor registration, the
 *  options-table well-formedness, and the init() rejection contract when
 *  no model_path is supplied. Full end-to-end inference (5-frame window
 *  → ORT run → score append) requires a real ORT build linked against
 *  the placeholder ONNX shipped under model/tiny/fastdvdnet_pre.onnx;
 *  that path is exercised by the CLI smoke gate, not by this unit test.
 *
 *  These tests run in every build and verify:
 *
 *    - the "fastdvdnet_pre" feature extractor is discoverable by name
 *      and by provided feature name (registration wired correctly);
 *    - init() cleanly declines when no model_path option and no
 *      VMAF_FASTDVDNET_PRE_MODEL_PATH env var are provided;
 *    - the options table is well-formed (non-NULL name + help, model_path
 *      is a string).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "feature/feature_extractor.h"
#include "opt.h"

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

static char *test_fastdvdnet_pre_is_registered(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("fastdvdnet_pre");
    mu_assert("fastdvdnet_pre extractor must be registered by name", fex != NULL);
    mu_assert("registered extractor has wrong name", !strcmp(fex->name, "fastdvdnet_pre"));
    mu_assert("fastdvdnet_pre.init must be set", fex->init != NULL);
    mu_assert("fastdvdnet_pre.extract must be set", fex->extract != NULL);
    mu_assert("fastdvdnet_pre.close must be set", fex->close != NULL);
    mu_assert("fastdvdnet_pre.priv_size must be non-zero", fex->priv_size > 0u);
    return NULL;
}

static char *test_fastdvdnet_pre_provides_residual_feature(void)
{
    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_feature_name("fastdvdnet_pre_l1_residual", 0);
    mu_assert("'fastdvdnet_pre_l1_residual' feature name must resolve to an extractor",
              fex != NULL);
    mu_assert("'fastdvdnet_pre_l1_residual' must map to the fastdvdnet_pre extractor",
              !strcmp(fex->name, "fastdvdnet_pre"));
    return NULL;
}

static char *test_fastdvdnet_pre_options_table_well_formed(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("fastdvdnet_pre");
    mu_assert("fastdvdnet_pre extractor missing", fex != NULL);
    mu_assert("fastdvdnet_pre must expose at least one option", fex->options != NULL);

    int saw_model_path = 0;
    for (const VmafOption *opt = fex->options; opt && opt->name; ++opt) {
        mu_assert("every option must have non-NULL help text", opt->help != NULL);
        if (!strcmp(opt->name, "model_path")) {
            saw_model_path = 1;
            mu_assert("model_path must be a string option", opt->type == VMAF_OPT_TYPE_STRING);
        }
    }
    mu_assert("fastdvdnet_pre must expose a 'model_path' option", saw_model_path);
    return NULL;
}

static char *test_fastdvdnet_pre_init_rejects_missing_model(void)
{
    /* With neither model_path option set nor VMAF_FASTDVDNET_PRE_MODEL_PATH
     * in the environment, init() must cleanly decline rather than crash. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("fastdvdnet_pre");
    mu_assert("fastdvdnet_pre extractor missing", fex != NULL);

    void *priv = calloc(1, fex->priv_size);
    mu_assert("priv alloc failed", priv != NULL);
    fex->priv = priv;

    /* Save + clear the env var for the duration of this test. */
    const char *saved = getenv("VMAF_FASTDVDNET_PRE_MODEL_PATH");
    char *saved_copy = saved ? strdup(saved) : NULL;
    (void)test_unsetenv("VMAF_FASTDVDNET_PRE_MODEL_PATH");

    int rc = fex->init(fex, VMAF_PIX_FMT_YUV420P, 8u, 64u, 64u);
    mu_assert("init must fail when no model path is provided", rc < 0);

    /* close() must be safe to call after a failed init — exactly the
     * same contract as feature_lpips.c. */
    (void)fex->close(fex);

    if (saved_copy) {
        (void)test_setenv("VMAF_FASTDVDNET_PRE_MODEL_PATH", saved_copy);
        free(saved_copy);
    }
    free(priv);
    fex->priv = NULL;
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_fastdvdnet_pre_is_registered);
    mu_run_test(test_fastdvdnet_pre_provides_residual_feature);
    mu_run_test(test_fastdvdnet_pre_options_table_well_formed);
    mu_run_test(test_fastdvdnet_pre_init_rejects_missing_model);
    return NULL;
}
