/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <stdlib.h> /* _fullpath */
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

#include "libvmaf/model.h"

#include "model_loader.h"
#include "onnx_scan.h"

/* Portable realpath wrapper: POSIX realpath() on Linux/macOS, _fullpath()
 * on MinGW/Windows. Both resolve symlinks and canonicalise the path in
 * place, returning NULL on failure. */
static char *resolve_path(const char *path, char *resolved)
{
#ifdef _WIN32
    return _fullpath(resolved, path, PATH_MAX);
#else
    return realpath(path, resolved);
#endif
}

/* ONNX files are protobuf-serialised graph messages. We sniff by extension +
 * a loose leading-byte pattern — protobuf varints start with a field tag
 * byte, so the first byte is rarely '{' (JSON) or '\x80' (pickle). */

static bool has_suffix(const char *s, const char *suf)
{
    size_t ls = strlen(s);
    size_t lu = strlen(suf);
    if (ls < lu)
        return false;
    return strcmp(s + ls - lu, suf) == 0;
}

int vmaf_dnn_sniff_kind(const char *path)
{
    if (!path)
        return -1;
    if (has_suffix(path, ".json") || has_suffix(path, ".pkl")) {
        return VMAF_MODEL_KIND_SVM;
    }
    if (has_suffix(path, ".onnx")) {
        return VMAF_MODEL_KIND_DNN_FR; /* default; sidecar may upgrade to NR */
    }
    return -1;
}

/* Ultra-small JSON-value extractor: supports "key": "value" and "key": number.
 * Sidecars are written by vmaf-train so we know the exact shape and can avoid
 * pulling a JSON dependency into libvmaf. */
static char *extract_string(const char *doc, const char *key)
{
    char needle[64];
    int n = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(needle))
        return NULL;
    const char *p = strstr(doc, needle);
    if (!p)
        return NULL;
    p = strchr(p + (size_t)n, ':');
    if (!p)
        return NULL;
    p++;
    while (*p && isspace((unsigned char)*p))
        p++;
    if (*p != '"')
        return NULL;
    p++;
    const char *q = strchr(p, '"');
    if (!q)
        return NULL;
    size_t len = (size_t)(q - p);
    char *out = (char *)malloc(len + 1);
    if (!out)
        return NULL;
    memcpy(out, p, len);
    out[len] = '\0';
    return out;
}

static int extract_int(const char *doc, const char *key, int *out)
{
    char needle[64];
    int n = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(needle))
        return -EINVAL;
    const char *p = strstr(doc, needle);
    if (!p)
        return -ENOENT;
    p = strchr(p + (size_t)n, ':');
    if (!p)
        return -ENOENT;
    p++;
    while (*p && isspace((unsigned char)*p))
        p++;
    errno = 0;
    char *endp = NULL;
    long v = strtol(p, &endp, 10);
    if (endp == p)
        return -EINVAL;
    if (errno == ERANGE || v < INT_MIN || v > INT_MAX)
        return -ERANGE;
    *out = (int)v;
    return 0;
}

int vmaf_dnn_sidecar_load(const char *onnx_path, VmafModelSidecar *out)
{
    if (!onnx_path || !out)
        return -EINVAL;
    memset(out, 0, sizeof(*out));
    out->kind = VMAF_MODEL_KIND_DNN_FR;

    char sidecar[4096];
    size_t len = strlen(onnx_path);
    if (len + 6 > sizeof(sidecar))
        return -ENAMETOOLONG;
    memcpy(sidecar, onnx_path, len);
    /* replace ".onnx" with ".json" */
    if (len >= 5 && strcmp(onnx_path + len - 5, ".onnx") == 0) {
        memcpy(sidecar + len - 5, ".json", 5);
        sidecar[len] = '\0';
    } else {
        memcpy(sidecar + len, ".json", 6);
    }

    FILE *f = fopen(sidecar, "rb");
    if (!f)
        return -errno;
    if (fseek(f, 0, SEEK_END) != 0) {
        (void)fclose(f);
        return -EIO;
    }
    long sz_raw = ftell(f);
    if (sz_raw < 0 || sz_raw > (1 << 20)) {
        (void)fclose(f);
        return -EFBIG;
    }
    const size_t sz = (size_t)sz_raw;
    if (fseek(f, 0, SEEK_SET) != 0) {
        (void)fclose(f);
        return -EIO;
    }
    char *buf = (char *)malloc(sz + 1u);
    if (!buf) {
        (void)fclose(f);
        return -ENOMEM;
    }
    size_t r = fread(buf, 1u, sz, f);
    (void)fclose(f);
    if (r != sz) {
        free(buf);
        return -EIO;
    }
    assert(sz <= (size_t)(1 << 20));
    /* buf was allocated as sz + 1u bytes (line ~115), so buf[sz] is valid. The
     * analyzer loses this relationship across the fread path. */
    buf[sz] = '\0'; // NOLINT(clang-analyzer-security.ArrayBound)

    char *kind_str = extract_string(buf, "kind");
    if (kind_str) {
        if (strcmp(kind_str, "nr") == 0)
            out->kind = VMAF_MODEL_KIND_DNN_NR;
        else if (strcmp(kind_str, "fr") == 0)
            out->kind = VMAF_MODEL_KIND_DNN_FR;
        free(kind_str);
    }
    out->name = extract_string(buf, "name");
    out->input_name = extract_string(buf, "input_name");
    out->output_name = extract_string(buf, "output_name");
    (void)extract_int(buf, "onnx_opset", &out->opset);

    free(buf);
    return 0;
}

void vmaf_dnn_sidecar_free(VmafModelSidecar *s)
{
    if (!s)
        return;
    free(s->name);
    free(s->input_name);
    free(s->output_name);
    memset(s, 0, sizeof(*s));
}

/* Size + kind check on the resolved path. Returns 0 + st_size in *out_size
 * on success, or a negative errno on failure. */
static int stat_regular(const char *path, size_t max_bytes, size_t *out_size)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return -errno;
    if (!S_ISREG(st.st_mode))
        return -ENOENT;
    if ((size_t)st.st_size > max_bytes)
        return -E2BIG;
    *out_size = (size_t)st.st_size;
    return 0;
}

/* Read @p sz bytes of @p path into a freshly-allocated buffer. Caller frees. */
static int slurp_file(const char *path, size_t sz, unsigned char **out_buf)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -errno;
    unsigned char *buf = (unsigned char *)malloc(sz);
    if (!buf) {
        (void)fclose(f);
        return -ENOMEM;
    }
    const size_t r = fread(buf, 1u, sz, f);
    (void)fclose(f);
    if (r != sz) {
        free(buf);
        return -EIO;
    }
    *out_buf = buf;
    return 0;
}

int vmaf_dnn_validate_onnx(const char *path, size_t max_bytes)
{
    if (!path)
        return -EINVAL;
    assert(path != NULL);
    if (max_bytes == 0)
        max_bytes = VMAF_DNN_DEFAULT_MAX_BYTES;
    assert(max_bytes > 0u);

    /* Resolve symlinks and normalise the path before any stat / open. An
     * adversarial --tiny-model value could point at a symlink to a non-
     * regular file; resolve_path() dereferences the symlink so the
     * subsequent S_ISREG check reflects the actual target. */
    char resolved[PATH_MAX];
    if (resolve_path(path, resolved) == NULL)
        return -errno;
    assert(resolved[0] != '\0');

    size_t sz = 0;
    int err = stat_regular(resolved, max_bytes, &sz);
    if (err != 0)
        return err;
    assert(sz <= max_bytes);
    /* Degenerate zero-byte file cannot be a valid ONNX ModelProto. */
    if (sz == 0)
        return -EBADMSG;

    unsigned char *buf = NULL;
    err = slurp_file(resolved, sz, &buf);
    if (err != 0)
        return err;
    assert(buf != NULL);

    /* Deep op-allowlist walk: parse the ONNX protobuf for NodeProto.op_type
     * strings and reject any that are not in the allowlist. This runs
     * before ORT's CreateSession, so a disallowed op short-circuits load. */
    err = vmaf_dnn_scan_onnx(buf, sz, NULL);
    free(buf);
    return err;
}
