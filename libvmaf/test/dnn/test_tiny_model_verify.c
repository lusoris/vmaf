/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  T6-9 / ADR-0209 — exercise the failure modes of
 *  vmaf_dnn_verify_signature() without requiring `cosign` on the test
 *  host. The function fails closed, so every "missing X" path returns a
 *  negative errno and short-circuits before posix_spawnp is reached.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "test.h"

#include "dnn/model_loader.h"

static char *test_verify_null_path(void)
{
    const int err = vmaf_dnn_verify_signature(NULL, NULL);
    mu_assert("NULL onnx path -> -EINVAL", err == -EINVAL);
    return NULL;
}

#ifndef _WIN32
/* Helper: write a tiny scratch file + return its path inside `out_path`. */
static int write_tmp(const char *suffix, const char *body, char *out_path, size_t out_sz)
{
    const int n = snprintf(out_path, out_sz, "/tmp/vmaf_verify_test_%d_%s", (int)getpid(), suffix);
    if (n < 0 || (size_t)n >= out_sz)
        return -1;
    FILE *f = fopen(out_path, "wb");
    if (!f)
        return -1;
    if (body && fputs(body, f) < 0) {
        (void)fclose(f);
        return -1;
    }
    if (fclose(f) != 0)
        return -1;
    return 0;
}

static char *test_verify_missing_registry(void)
{
    /* No registry beside /tmp/foo.onnx — expect -ENOENT. */
    char onnx[256];
    const int wrc = write_tmp("missing_reg.onnx", "fake", onnx, sizeof(onnx));
    mu_assert("write_tmp succeeded", wrc == 0);
    const int err = vmaf_dnn_verify_signature(onnx, "/nonexistent/registry.json");
    mu_assert("missing registry -> -ENOENT", err == -ENOENT);
    (void)unlink(onnx);
    return NULL;
}

static char *test_verify_no_matching_entry(void)
{
    /* Registry exists but has no entry for `unknown.onnx`. */
    char reg_path[256];
    const int wrc = write_tmp(
        "reg.json", "{\"models\":[{\"id\":\"x\",\"onnx\":\"other.onnx\",\"sha256\":\"00\"}]}",
        reg_path, sizeof(reg_path));
    mu_assert("write_tmp reg succeeded", wrc == 0);
    const int err = vmaf_dnn_verify_signature("/tmp/unknown.onnx", reg_path);
    mu_assert("no entry -> -ENOENT", err == -ENOENT);
    (void)unlink(reg_path);
    return NULL;
}

static char *test_verify_missing_bundle(void)
{
    /* Registry has the entry, but the sigstore_bundle file does not exist. */
    char reg_path[256];
    const char *body = "{\"models\":[{\"id\":\"y\",\"onnx\":\"y.onnx\",\"sha256\":\"00\","
                       "\"sigstore_bundle\":\"y.onnx.sigstore.json\"}]}";
    const int wrc = write_tmp("reg2.json", body, reg_path, sizeof(reg_path));
    mu_assert("write_tmp reg2 succeeded", wrc == 0);
    const int err = vmaf_dnn_verify_signature("/tmp/y.onnx", reg_path);
    mu_assert("missing bundle -> -ENOENT", err == -ENOENT);
    (void)unlink(reg_path);
    return NULL;
}

static char *test_verify_entry_without_bundle(void)
{
    /* Entry exists but has no `sigstore_bundle` key — refuse to load
     * (unsigned model). */
    char reg_path[256];
    const char *body = "{\"models\":[{\"id\":\"z\",\"onnx\":\"z.onnx\",\"sha256\":\"00\"}]}";
    const int wrc = write_tmp("reg3.json", body, reg_path, sizeof(reg_path));
    mu_assert("write_tmp reg3 succeeded", wrc == 0);
    const int err = vmaf_dnn_verify_signature("/tmp/z.onnx", reg_path);
    mu_assert("no bundle key -> -ENOENT", err == -ENOENT);
    (void)unlink(reg_path);
    return NULL;
}
#endif /* !_WIN32 */

#ifdef _WIN32
static char *test_verify_windows_returns_enosys(void)
{
    const int err = vmaf_dnn_verify_signature("C:\\fake.onnx", NULL);
    mu_assert("Windows path -> -ENOSYS", err == -ENOSYS);
    return NULL;
}
#endif

char *run_tests(void)
{
    mu_run_test(test_verify_null_path);
#ifndef _WIN32
    mu_run_test(test_verify_missing_registry);
    mu_run_test(test_verify_no_matching_entry);
    mu_run_test(test_verify_missing_bundle);
    mu_run_test(test_verify_entry_without_bundle);
#else
    mu_run_test(test_verify_windows_returns_enosys);
#endif
    return NULL;
}
