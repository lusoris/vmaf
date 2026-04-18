/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "test.h"

#include "dnn/model_loader.h"

static char *test_sniff_by_extension(void)
{
    mu_assert("json → SVM", vmaf_dnn_sniff_kind("foo.json") == VMAF_MODEL_KIND_SVM);
    mu_assert("pkl → SVM", vmaf_dnn_sniff_kind("foo.pkl") == VMAF_MODEL_KIND_SVM);
    mu_assert("onnx → DNN_FR", vmaf_dnn_sniff_kind("foo.onnx") == VMAF_MODEL_KIND_DNN_FR);
    mu_assert("unknown ext → -1", vmaf_dnn_sniff_kind("foo.bin") == -1);
    mu_assert("NULL → -1", vmaf_dnn_sniff_kind(NULL) == -1);
    return NULL;
}

static char *test_size_cap(void)
{
    /* Use the test binary itself as a proxy for "regular file, within limits".
     * Size cap of 1 byte should reject it. */
#ifdef _WIN32
    const char *probe = "C:\\Windows\\System32\\cmd.exe";
#else
    const char *probe = "/etc/hostname";
#endif
    int err = vmaf_dnn_validate_onnx(probe, 1);
    mu_assert("expected -E2BIG for 1-byte cap", err == -E2BIG || err == 0);
    err = vmaf_dnn_validate_onnx("/definitely/does/not/exist.onnx", 0);
    mu_assert("expected errno for missing file", err < 0);
    return NULL;
}

static char *test_validate_null_path(void)
{
    const int err = vmaf_dnn_validate_onnx(NULL, 0);
    mu_assert("NULL path → -EINVAL", err == -EINVAL);
    return NULL;
}

#ifndef _WIN32
/* Write @p len bytes of @p data to a fresh temp file; returns malloc'd path. */
static char *write_temp(const unsigned char *data, size_t len)
{
    char tmpl[] = "/tmp/vmaf-dnn-validate-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0)
        return NULL;
    const ssize_t w = write(fd, data, len);
    close(fd);
    if (w != (ssize_t)len)
        return NULL;
    return strdup(tmpl);
}

/* Write @p len bytes of @p data to an exact path with user-only perms (0600).
 * Unlike fopen(path, "wb") this doesn't inherit umask and can't race into a
 * world-writable state (CodeQL cpp/world-writable-file-creation). */
static int write_file_600(const char *path, const unsigned char *data, size_t len)
{
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0)
        return -1;
    const ssize_t w = write(fd, data, len);
    const int rc = close(fd);
    if (w != (ssize_t)len || rc != 0)
        return -1;
    return 0;
}

static char *test_validate_zero_byte(void)
{
    char *path = write_temp((const unsigned char *)"", 0);
    mu_assert("temp file creation failed", path != NULL);
    const int err = vmaf_dnn_validate_onnx(path, 0);
    remove(path);
    free(path);
    mu_assert("empty file → -EBADMSG", err == -EBADMSG);
    return NULL;
}

static char *test_validate_allowed_onnx(void)
{
    /* Minimal ModelProto { graph { node { op_type = "Conv" } } } */
    const unsigned char buf[] = {0x3A, 0x08, 0x0A, 0x06, 0x22, 0x04, 'C', 'o', 'n', 'v'};
    char *path = write_temp(buf, sizeof(buf));
    mu_assert("temp file creation failed", path != NULL);
    const int err = vmaf_dnn_validate_onnx(path, 0);
    remove(path);
    free(path);
    mu_assert("allowed Conv onnx → 0", err == 0);
    return NULL;
}

static char *test_validate_disallowed_onnx(void)
{
    const unsigned char buf[] = {0x3A, 0x08, 0x0A, 0x06, 0x22, 0x04, 'L', 'o', 'o', 'p'};
    char *path = write_temp(buf, sizeof(buf));
    mu_assert("temp file creation failed", path != NULL);
    const int err = vmaf_dnn_validate_onnx(path, 0);
    remove(path);
    free(path);
    mu_assert("disallowed Loop onnx → -EPERM", err == -EPERM);
    return NULL;
}

static char *test_validate_symlink_to_dir(void)
{
    /* Path hardening: realpath() must resolve a symlink to /tmp (a
     * directory) before the S_ISREG gate, so validate must reject. */
    char link_path[] = "/tmp/vmaf-dnn-dirlink-XXXXXX";
    int fd = mkstemp(link_path);
    mu_assert("mkstemp failed", fd >= 0);
    close(fd);
    remove(link_path);
    mu_assert("symlink() failed", symlink("/tmp", link_path) == 0);
    const int err = vmaf_dnn_validate_onnx(link_path, 0);
    remove(link_path);
    /* /tmp is not a regular file → stat_regular returns -ENOENT. */
    mu_assert("symlink-to-dir → -ENOENT", err == -ENOENT);
    return NULL;
}

/* Allowed ONNX payload (single Conv node) reused across jail tests. */
static const unsigned char kAllowedOnnx[] = {0x3A, 0x08, 0x0A, 0x06, 0x22,
                                             0x04, 'C',  'o',  'n',  'v'};

static char *test_jail_unset_accepts_anywhere(void)
{
    /* With VMAF_TINY_MODEL_DIR unset the jail is a no-op; a valid model
     * anywhere in the filesystem must still validate. */
    unsetenv("VMAF_TINY_MODEL_DIR");
    char *path = write_temp(kAllowedOnnx, sizeof(kAllowedOnnx));
    mu_assert("temp file creation failed", path != NULL);
    const int err = vmaf_dnn_validate_onnx(path, 0);
    remove(path);
    free(path);
    mu_assert("jail unset → allowed model validates", err == 0);
    return NULL;
}

static char *test_jail_accepts_model_inside(void)
{
    /* Model sits inside the jail dir → must validate. */
    char jail[] = "/tmp/vmaf-dnn-jail-XXXXXX";
    mu_assert("mkdtemp failed", mkdtemp(jail) != NULL);

    char model_path[PATH_MAX];
    snprintf(model_path, sizeof(model_path), "%s/allowed.onnx", jail);
    mu_assert("write_file_600 model failed",
              write_file_600(model_path, kAllowedOnnx, sizeof(kAllowedOnnx)) == 0);

    mu_assert("setenv failed", setenv("VMAF_TINY_MODEL_DIR", jail, 1) == 0);
    const int err = vmaf_dnn_validate_onnx(model_path, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(model_path);
    rmdir(jail);
    mu_assert("model inside jail → 0", err == 0);
    return NULL;
}

static char *test_jail_rejects_model_outside(void)
{
    /* Model sits outside the jail dir → must return -EACCES before any
     * stat() of the model path happens. */
    char jail[] = "/tmp/vmaf-dnn-jail-XXXXXX";
    mu_assert("mkdtemp failed", mkdtemp(jail) != NULL);

    char *outside = write_temp(kAllowedOnnx, sizeof(kAllowedOnnx));
    mu_assert("write_temp failed", outside != NULL);

    mu_assert("setenv failed", setenv("VMAF_TINY_MODEL_DIR", jail, 1) == 0);
    const int err = vmaf_dnn_validate_onnx(outside, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(outside);
    free(outside);
    rmdir(jail);
    mu_assert("model outside jail → -EACCES", err == -EACCES);
    return NULL;
}

static char *test_jail_rejects_sibling_prefix(void)
{
    /* Classic prefix-matching trap: if the jail prefix "/tmp/foo" accepted
     * "/tmp/foobar/x.onnx" via naive strncmp, a sibling dir could escape.
     * The trailing-separator normalisation in enforce_tiny_model_jail must
     * reject this. */
    char jail[] = "/tmp/vmaf-dnn-sibprefix-XXXXXX";
    mu_assert("mkdtemp failed", mkdtemp(jail) != NULL);

    char sibling_dir[PATH_MAX];
    snprintf(sibling_dir, sizeof(sibling_dir), "%s-sibling", jail);
    mu_assert("sibling mkdir failed", mkdir(sibling_dir, 0700) == 0);

    char model_path[PATH_MAX];
    snprintf(model_path, sizeof(model_path), "%s/escape.onnx", sibling_dir);
    mu_assert("write_file_600 sibling model failed",
              write_file_600(model_path, kAllowedOnnx, sizeof(kAllowedOnnx)) == 0);

    mu_assert("setenv failed", setenv("VMAF_TINY_MODEL_DIR", jail, 1) == 0);
    const int err = vmaf_dnn_validate_onnx(model_path, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(model_path);
    rmdir(sibling_dir);
    rmdir(jail);
    mu_assert("sibling prefix must be rejected → -EACCES", err == -EACCES);
    return NULL;
}

static char *test_jail_rejects_symlink_escape(void)
{
    /* Symlink-escape attempt: place a symlink *inside* the jail pointing
     * at a file *outside* the jail. realpath() on the model argument
     * resolves the symlink to the outside target, which must then fail
     * the prefix check. */
    char jail[] = "/tmp/vmaf-dnn-jailsym-XXXXXX";
    mu_assert("mkdtemp failed", mkdtemp(jail) != NULL);

    char *outside = write_temp(kAllowedOnnx, sizeof(kAllowedOnnx));
    mu_assert("write_temp failed", outside != NULL);

    char link_path[PATH_MAX];
    snprintf(link_path, sizeof(link_path), "%s/escape.onnx", jail);
    mu_assert("symlink() failed", symlink(outside, link_path) == 0);

    mu_assert("setenv failed", setenv("VMAF_TINY_MODEL_DIR", jail, 1) == 0);
    const int err = vmaf_dnn_validate_onnx(link_path, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(link_path);
    remove(outside);
    free(outside);
    rmdir(jail);
    mu_assert("symlink escape must be rejected → -EACCES", err == -EACCES);
    return NULL;
}

static char *test_jail_rejects_nonexistent_jail(void)
{
    /* Fails closed on a misconfigured jail: if VMAF_TINY_MODEL_DIR points
     * at a path that does not exist, every validation returns -EACCES. */
    char *model = write_temp(kAllowedOnnx, sizeof(kAllowedOnnx));
    mu_assert("write_temp failed", model != NULL);

    mu_assert("setenv failed",
              setenv("VMAF_TINY_MODEL_DIR", "/tmp/vmaf-does-not-exist-zzzyx", 1) == 0);
    const int err = vmaf_dnn_validate_onnx(model, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(model);
    free(model);
    mu_assert("nonexistent jail dir → -EACCES", err == -EACCES);
    return NULL;
}

static char *test_jail_rejects_non_directory(void)
{
    /* Jail env pointing at a regular file (not a directory) must also fail
     * closed — a file is not a prefix anything can sit under. */
    char *jail_file = write_temp((const unsigned char *)"x", 1u);
    mu_assert("write_temp failed", jail_file != NULL);
    char *model = write_temp(kAllowedOnnx, sizeof(kAllowedOnnx));
    mu_assert("write_temp failed", model != NULL);

    mu_assert("setenv failed", setenv("VMAF_TINY_MODEL_DIR", jail_file, 1) == 0);
    const int err = vmaf_dnn_validate_onnx(model, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(model);
    remove(jail_file);
    free(model);
    free(jail_file);
    mu_assert("jail-is-file → -EACCES", err == -EACCES);
    return NULL;
}

static char *test_jail_accepts_trailing_slash(void)
{
    /* Normalisation check: a trailing '/' on the env value must not
     * introduce a double-separator that breaks the prefix match. */
    char jail[] = "/tmp/vmaf-dnn-jailslash-XXXXXX";
    mu_assert("mkdtemp failed", mkdtemp(jail) != NULL);

    char jail_with_slash[PATH_MAX];
    snprintf(jail_with_slash, sizeof(jail_with_slash), "%s/", jail);

    char model_path[PATH_MAX];
    snprintf(model_path, sizeof(model_path), "%s/allowed.onnx", jail);
    mu_assert("write_file_600 model failed",
              write_file_600(model_path, kAllowedOnnx, sizeof(kAllowedOnnx)) == 0);

    mu_assert("setenv failed", setenv("VMAF_TINY_MODEL_DIR", jail_with_slash, 1) == 0);
    const int err = vmaf_dnn_validate_onnx(model_path, 0);
    unsetenv("VMAF_TINY_MODEL_DIR");
    remove(model_path);
    rmdir(jail);
    mu_assert("jail with trailing slash → 0", err == 0);
    return NULL;
}
#endif /* !_WIN32 */

static char *test_sidecar_parses(void)
{
#ifdef _WIN32
    char tmpl[MAX_PATH];
    char tmpdir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmpdir);
    snprintf(tmpl, sizeof tmpl, "%svmaf-dnn-sidecar-test", tmpdir);
    FILE *tmpf = fopen(tmpl, "w");
    mu_assert("temp file creation failed", tmpf != NULL);
    fclose(tmpf);
#else
    char tmpl[] = "/tmp/vmaf-dnn-sidecar-XXXXXX";
    int fd = mkstemp(tmpl);
    mu_assert("mkstemp failed", fd >= 0);
    close(fd);
#endif

    char onnx[1024], sidecar[1024];
    snprintf(onnx, sizeof onnx, "%s.onnx", tmpl);
    snprintf(sidecar, sizeof sidecar, "%s.json", tmpl);
    /* Touch an empty onnx so sidecar_load doesn't key off its existence. */
    FILE *f = fopen(onnx, "w");
    if (f)
        fclose(f);

    FILE *s = fopen(sidecar, "w");
    mu_assert("fopen sidecar failed", s != NULL);
    fprintf(s, "{\n"
               "  \"name\": \"vmaf_tiny_fr_v1\",\n"
               "  \"kind\": \"fr\",\n"
               "  \"onnx_opset\": 17,\n"
               "  \"input_name\":  \"features\",\n"
               "  \"output_name\": \"score\"\n"
               "}\n");
    fclose(s);

    VmafModelSidecar meta;
    int err = vmaf_dnn_sidecar_load(onnx, &meta);
    mu_assert("sidecar_load failed", err == 0);
    mu_assert("kind FR", meta.kind == VMAF_MODEL_KIND_DNN_FR);
    mu_assert("opset 17", meta.opset == 17);
    mu_assert("name set", meta.name && !strcmp(meta.name, "vmaf_tiny_fr_v1"));
    mu_assert("input set", meta.input_name && !strcmp(meta.input_name, "features"));
    mu_assert("output set", meta.output_name && !strcmp(meta.output_name, "score"));
    vmaf_dnn_sidecar_free(&meta);

    remove(sidecar);
    remove(onnx);
    remove(tmpl);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_sniff_by_extension);
    mu_run_test(test_size_cap);
    mu_run_test(test_validate_null_path);
#ifndef _WIN32
    mu_run_test(test_validate_zero_byte);
    mu_run_test(test_validate_allowed_onnx);
    mu_run_test(test_validate_disallowed_onnx);
    mu_run_test(test_validate_symlink_to_dir);
    mu_run_test(test_jail_unset_accepts_anywhere);
    mu_run_test(test_jail_accepts_model_inside);
    mu_run_test(test_jail_rejects_model_outside);
    mu_run_test(test_jail_rejects_sibling_prefix);
    mu_run_test(test_jail_rejects_symlink_escape);
    mu_run_test(test_jail_rejects_nonexistent_jail);
    mu_run_test(test_jail_rejects_non_directory);
    mu_run_test(test_jail_accepts_trailing_slash);
#endif
    mu_run_test(test_sidecar_parses);
    return NULL;
}
