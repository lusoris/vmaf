/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  LPIPS feature extractor — structural + stub-path tests.
 *
 *  Mirrors the test_ciede.c pattern but targets the high-level extractor
 *  registration contract, not the metric internals. Full end-to-end
 *  inference (YUV → ORT run → score append) requires a real ORT build;
 *  that path is exercised by the CLI smoke gate against
 *  model/tiny/lpips_sq.onnx. These unit tests run in every build and
 *  verify:
 *
 *    - the "lpips" feature extractor is discoverable by name and by
 *      provided feature name (registration wired correctly);
 *    - init() cleanly declines when no model_path and no env var are
 *      provided (input validation);
 *    - options metadata is well-formed (non-NULL name, non-NULL help).
 *
 *  These do NOT open an ORT session — the intent is to catch
 *  registration / option-table regressions regardless of whether ORT is
 *  linked into the test binary.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "feature/feature_extractor.h"
#include "opt.h"

static char *test_lpips_is_registered(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("lpips");
    mu_assert("lpips extractor must be registered by name", fex != NULL);
    mu_assert("registered extractor has wrong name", !strcmp(fex->name, "lpips"));
    mu_assert("lpips.init must be set", fex->init != NULL);
    mu_assert("lpips.extract must be set", fex->extract != NULL);
    mu_assert("lpips.close must be set", fex->close != NULL);
    mu_assert("lpips.priv_size must be non-zero", fex->priv_size > 0u);
    return NULL;
}

static char *test_lpips_provides_lpips_feature(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_feature_name("lpips", 0);
    mu_assert("'lpips' feature name must resolve to an extractor", fex != NULL);
    mu_assert("'lpips' must map to the lpips extractor", !strcmp(fex->name, "lpips"));
    return NULL;
}

static char *test_lpips_options_table_well_formed(void)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("lpips");
    mu_assert("lpips extractor missing", fex != NULL);
    mu_assert("lpips must expose at least one option", fex->options != NULL);

    int saw_model_path = 0;
    for (const VmafOption *opt = fex->options; opt && opt->name; ++opt) {
        mu_assert("every option must have non-NULL help text", opt->help != NULL);
        if (!strcmp(opt->name, "model_path")) {
            saw_model_path = 1;
            mu_assert("model_path must be a string option", opt->type == VMAF_OPT_TYPE_STRING);
        }
    }
    mu_assert("lpips must expose a 'model_path' option", saw_model_path);
    return NULL;
}

static char *test_lpips_init_rejects_missing_model(void)
{
    /* With neither model_path option set nor VMAF_LPIPS_MODEL_PATH in the
     * environment, init() must cleanly decline rather than crash. */
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("lpips");
    mu_assert("lpips extractor missing", fex != NULL);

    void *priv = calloc(1, fex->priv_size);
    mu_assert("priv alloc failed", priv != NULL);
    fex->priv = priv;

    /* Save + clear the env var for the duration of this test. */
    const char *saved = getenv("VMAF_LPIPS_MODEL_PATH");
    char *saved_copy = saved ? strdup(saved) : NULL;
    (void)unsetenv("VMAF_LPIPS_MODEL_PATH");

    int rc = fex->init(fex, VMAF_PIX_FMT_YUV420P, 8u, 64u, 64u);
    mu_assert("init must fail when no model path is provided", rc < 0);

    /* close() must be safe to call after a failed init — the extractor
     * either did full cleanup before returning, or close() tolerates the
     * partial state. Either is acceptable as long as we don't crash. */
    (void)fex->close(fex);

    if (saved_copy) {
        (void)setenv("VMAF_LPIPS_MODEL_PATH", saved_copy, 1);
        free(saved_copy);
    }
    free(priv);
    fex->priv = NULL;
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_lpips_is_registered);
    mu_run_test(test_lpips_provides_lpips_feature);
    mu_run_test(test_lpips_options_table_well_formed);
    mu_run_test(test_lpips_init_rejects_missing_model);
    return NULL;
}
