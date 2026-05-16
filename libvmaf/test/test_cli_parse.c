/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#ifdef _WIN32
#include "compat/win32/getopt.h"
#else
#include <getopt.h>
#endif

#include "test.h"

#include "cli_parse.h"

static int cli_free_dicts(CLISettings *settings)
{
    for (unsigned i = 0; i < settings->feature_cnt; i++) {
        int err = vmaf_feature_dictionary_free(&(settings->feature_cfg[i].opts_dict));
        if (err)
            return err;
    }
    return 0;
}

static char *test_aom_ctc_v1_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v1.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but number of features is not 5",
              settings.feature_cnt == 5);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v2_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v2.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but number of features is not 5",
              settings.feature_cnt == 5);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v3_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v3.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v4_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v4.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v5_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v5.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v6_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v6.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but common_bitdepth not enabled",
              settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_nflx_ctc_v1_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--nflx_ctc", "v1.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but number of features is not 3",
              settings.feature_cnt == 3);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

/* `--backend cuda` must end up with gpumask == 0 (NOT 1), because
 * VmafConfiguration::gpumask is a CUDA-*disable* bitmask — any nonzero
 * value disables CUDA in compute_fex_flags. The CLI's job is only to
 * trip use_gpumask so vmaf_cuda_state_init runs; the runtime then
 * picks the CUDA extractors because gpumask is 0. Earlier revisions
 * set gpumask = 1 here, which silently routed every "CUDA" run
 * through the CPU path. */
static char *test_backend_cuda_engages_cuda()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "cuda"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend cuda must set use_gpumask = true (so CUDA inits)",
              settings.use_gpumask);
    mu_assert("cli_parse: --backend cuda must set gpumask = 0 (any nonzero DISABLES CUDA)",
              settings.gpumask == 0);
    mu_assert("cli_parse: --backend cuda must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend cuda must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend cuda must NOT set no_cuda", !settings.no_cuda);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_cpu()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "cpu"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend cpu must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend cpu must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend cpu must set no_vulkan = true", settings.no_vulkan);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_sycl()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "sycl"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend sycl must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend sycl must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend sycl must default sycl_device to 0", settings.sycl_device == 0);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_vulkan()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "vulkan"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend vulkan must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend vulkan must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend vulkan must default vulkan_device to 0",
              settings.vulkan_device == 0);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

/* Explicit `--gpumask=N --backend cuda` must preserve the user's gpumask,
 * NOT clobber it. Multi-GPU rigs need fine-grained disable bits. */
static char *test_backend_cuda_preserves_explicit_gpumask()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--gpumask=2", "--backend", "cuda"};
    int argc = 8;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --gpumask=2 --backend cuda must preserve gpumask = 2",
              settings.gpumask == 2);
    mu_assert("cli_parse: --gpumask=2 --backend cuda must keep use_gpumask = true",
              settings.use_gpumask);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *run_aom_ctc_tests(void)
{
    mu_run_test(test_aom_ctc_v1_0);
    mu_run_test(test_aom_ctc_v2_0);
    mu_run_test(test_aom_ctc_v3_0);
    mu_run_test(test_aom_ctc_v4_0);
    mu_run_test(test_aom_ctc_v5_0);
    mu_run_test(test_aom_ctc_v6_0);
    mu_run_test(test_nflx_ctc_v1_0);
    return NULL;
}

static char *run_backend_tests(void)
{
    mu_run_test(test_backend_cpu);
    mu_run_test(test_backend_cuda_engages_cuda);
    mu_run_test(test_backend_cuda_preserves_explicit_gpumask);
    mu_run_test(test_backend_sycl);
    mu_run_test(test_backend_vulkan);
    return NULL;
}

char *run_tests()
{
    char *result = run_aom_ctc_tests();
    if (result)
        return result;
    result = run_backend_tests();
    if (result)
        return result;
    return NULL;
}
