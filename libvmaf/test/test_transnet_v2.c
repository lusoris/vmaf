/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  TransNet V2 shot-boundary detector (T6-3a) — structural + stub-path tests.
 *
 *  Mirrors the test_lpips.c / test_fastdvdnet_pre.c shape: covers extractor
 *  registration, options-table well-formedness, the init() rejection contract
 *  when no model_path is supplied, and the dual-feature contract
 *  (shot_boundary_probability + shot_boundary).
 *
 *  Full end-to-end inference (100-frame window -> ORT run -> two scores
 *  per frame) requires a real ORT build linked against the placeholder
 *  ONNX shipped under model/tiny/transnet_v2.onnx; that path is exercised
 *  by the CLI smoke gate, not by this unit test.
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

static char *test_transnet_v2_is_registered(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("transnet_v2");
    mu_assert("transnet_v2 extractor must be registered by name", fex != NULL);
    mu_assert("registered extractor has wrong name", !strcmp(fex->name, "transnet_v2"));
    mu_assert("transnet_v2.init must be set", fex->init != NULL);
    mu_assert("transnet_v2.extract must be set", fex->extract != NULL);
    mu_assert("transnet_v2.close must be set", fex->close != NULL);
    mu_assert("transnet_v2.priv_size must be non-zero", fex->priv_size > 0u);
    return NULL;
}

static char *test_transnet_v2_provides_probability_feature(void)
{
    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_feature_name("shot_boundary_probability", 0);
    mu_assert("'shot_boundary_probability' feature name must resolve to an extractor", fex != NULL);
    mu_assert("'shot_boundary_probability' must map to the transnet_v2 extractor",
              !strcmp(fex->name, "transnet_v2"));
    return NULL;
}

static char *test_transnet_v2_provides_binary_flag_feature(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_feature_name("shot_boundary", 0);
    mu_assert("'shot_boundary' feature name must resolve to an extractor", fex != NULL);
    mu_assert("'shot_boundary' must map to the transnet_v2 extractor",
              !strcmp(fex->name, "transnet_v2"));
    return NULL;
}

static char *test_transnet_v2_options_table_well_formed(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("transnet_v2");
    mu_assert("transnet_v2 extractor missing", fex != NULL);
    mu_assert("transnet_v2 must expose at least one option", fex->options != NULL);

    int saw_model_path = 0;
    for (const VmafOption *opt = fex->options; opt && opt->name; ++opt) {
        mu_assert("every option must have non-NULL help text", opt->help != NULL);
        if (!strcmp(opt->name, "model_path")) {
            saw_model_path = 1;
            mu_assert("model_path must be a string option", opt->type == VMAF_OPT_TYPE_STRING);
        }
    }
    mu_assert("transnet_v2 must expose a 'model_path' option", saw_model_path);
    return NULL;
}

static char *test_transnet_v2_init_rejects_missing_model(void)
{
    /* With neither model_path option set nor VMAF_TRANSNET_V2_MODEL_PATH
     * in the environment, init() must cleanly decline rather than crash. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("transnet_v2");
    mu_assert("transnet_v2 extractor missing", fex != NULL);

    void *priv = calloc(1, fex->priv_size);
    mu_assert("priv alloc failed", priv != NULL);
    fex->priv = priv;

    /* Save + clear the env var for the duration of this test. */
    const char *saved = getenv("VMAF_TRANSNET_V2_MODEL_PATH");
    char *saved_copy = saved ? strdup(saved) : NULL;
    (void)test_unsetenv("VMAF_TRANSNET_V2_MODEL_PATH");

    int rc = fex->init(fex, VMAF_PIX_FMT_YUV420P, 8u, 64u, 64u);
    mu_assert("init must fail when no model path is provided", rc < 0);

    /* close() must be safe to call after a failed init — exactly the
     * same contract as feature_lpips.c. */
    (void)fex->close(fex);

    if (saved_copy) {
        (void)test_setenv("VMAF_TRANSNET_V2_MODEL_PATH", saved_copy);
        free(saved_copy);
    }
    free(priv);
    fex->priv = NULL;
    return NULL;
}

static char *test_transnet_v2_provided_features_list_terminated(void)
{
    /* The provided_features array must be NULL-terminated and contain
     * both feature names so the per-shot CRF predictor (T6-3b) can
     * discover the surface by name. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("transnet_v2");
    mu_assert("transnet_v2 extractor missing", fex != NULL);
    mu_assert("provided_features must be set", fex->provided_features != NULL);

    int saw_prob = 0;
    int saw_flag = 0;
    unsigned count = 0u;
    for (const char *const *p = fex->provided_features; *p; ++p) {
        if (!strcmp(*p, "shot_boundary_probability"))
            saw_prob = 1;
        if (!strcmp(*p, "shot_boundary"))
            saw_flag = 1;
        count += 1u;
        mu_assert("runaway provided_features (missing NULL terminator?)", count < 16u);
    }
    mu_assert("provided_features must contain shot_boundary_probability", saw_prob);
    mu_assert("provided_features must contain shot_boundary", saw_flag);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_transnet_v2_is_registered);
    mu_run_test(test_transnet_v2_provides_probability_feature);
    mu_run_test(test_transnet_v2_provides_binary_flag_feature);
    mu_run_test(test_transnet_v2_options_table_well_formed);
    mu_run_test(test_transnet_v2_init_rejects_missing_model);
    mu_run_test(test_transnet_v2_provided_features_list_terminated);
    return NULL;
}
