/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Determinism test for svm_set_rand_seed() (ADR-0455).
 *
 *  Verifies that two calls to svm_cross_validation() with identical seed
 *  and identical training data produce identical fold-target vectors, and
 *  that two calls with different seeds produce different vectors with high
 *  probability (statistically guaranteed for any non-trivial dataset size).
 *
 *  Smoke command (see PR description):
 *      ./build/test/test_svm_rand_seed   # expect: 2 tests run, 2 passed
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "svm.h"
#include "test.h"

/* Minimal synthetic regression dataset: 30 points, 2 features.
 * Labels are a simple linear function so the SVM can fit them. */
#define N_POINTS 30
#define N_FEATURES 2

static void build_problem(struct svm_problem *prob, struct svm_node storage[][N_FEATURES + 1],
                          double labels[N_POINTS])
{
    prob->l = N_POINTS;
    prob->y = labels;
    prob->x = malloc((size_t)N_POINTS * sizeof(struct svm_node *));

    for (int i = 0; i < N_POINTS; i++) {
        labels[i] = (double)(i % 3); /* 3-class labels for stratified split */
        storage[i][0].index = 1;
        storage[i][0].value = (double)i * 0.1;
        storage[i][1].index = 2;
        storage[i][1].value = (double)(N_POINTS - i) * 0.1;
        storage[i][2].index = -1;
        storage[i][2].value = 0.0;
        prob->x[i] = storage[i];
    }
}

static struct svm_parameter make_param(void)
{
    struct svm_parameter param;
    memset(&param, 0, sizeof(param));
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.C = 1.0;
    param.eps = 1e-3;
    param.cache_size = 16.0; /* MB */
    param.shrinking = 0;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    return param;
}

/* Run cross-validation and return 1 if all targets are finite. */
static int run_cv(double *target, unsigned seed)
{
    struct svm_node storage[N_POINTS][N_FEATURES + 1];
    double labels[N_POINTS];
    struct svm_problem prob;
    struct svm_parameter param = make_param();

    build_problem(&prob, storage, labels);
    svm_set_rand_seed(seed);
    svm_cross_validation(&prob, &param, 3, target);
    free(prob.x);

    for (int i = 0; i < N_POINTS; i++) {
        if (!isfinite(target[i]))
            return 0;
    }
    return 1;
}

/* Same seed must produce identical fold-target vectors. */
static char *test_same_seed_deterministic(void)
{
    double targets_a[N_POINTS];
    double targets_b[N_POINTS];

    mu_assert("run A (seed 42) produced non-finite targets", run_cv(targets_a, 42u));
    mu_assert("run B (seed 42) produced non-finite targets", run_cv(targets_b, 42u));

    for (int i = 0; i < N_POINTS; i++) {
        mu_assert("same seed must yield identical cross-validation targets",
                  targets_a[i] == targets_b[i]);
    }
    return NULL;
}

/* Different seeds must produce different fold assignments (at least one
 * target differs).  This could theoretically fail by chance but the
 * probability is negligible for a 30-point, 3-fold dataset. */
static char *test_different_seeds_diverge(void)
{
    double targets_a[N_POINTS];
    double targets_b[N_POINTS];

    mu_assert("run A (seed 1) produced non-finite targets", run_cv(targets_a, 1u));
    mu_assert("run B (seed 9999) produced non-finite targets", run_cv(targets_b, 9999u));

    int any_differ = 0;
    for (int i = 0; i < N_POINTS; i++) {
        if (targets_a[i] != targets_b[i]) {
            any_differ = 1;
            break;
        }
    }
    mu_assert("different seeds must yield at least one different target", any_differ);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_same_seed_deterministic);
    mu_run_test(test_different_seeds_diverge);
    return NULL;
}
