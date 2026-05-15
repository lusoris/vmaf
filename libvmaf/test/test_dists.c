/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  DISTS-Sq feature extractor — structural + missing-model tests.
 *
 *  Full inference is exercised by the DNN smoke gate when
 *  model/tiny/dists_sq.onnx is available; this file verifies the
 *  registration / option-table / init-rejection contract.
 */

#include "tiny_ai_test_template.h"

VMAF_TINY_AI_DEFINE_REGISTRATION_TESTS("dists_sq", "dists_sq", "VMAF_DISTS_SQ_MODEL_PATH", dists_sq)

char *run_tests(void)
{
    VMAF_TINY_AI_RUN_REGISTRATION_TESTS(dists_sq);
    return NULL;
}
