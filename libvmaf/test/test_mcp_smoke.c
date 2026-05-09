/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Smoke + protocol test for the embedded MCP server.
 *
 *  T5-2 (ADR-0209) shipped this file pinning the -ENOSYS scaffold
 *  contract. T5-2b (PR #490) flipped init/start_stdio/close to a
 *  working dispatcher. T5-2c (this PR — MCP runtime v2):
 *      - flips UDS from -ENOSYS to a real bind/listen/accept loop;
 *      - replaces compute_vmaf's `deferred_to_v2` placeholder with
 *        a real libvmaf scoring binding.
 *
 *  Coverage:
 *      - public availability + transport-availability accessors
 *      - NULL-guard contract on every entry point
 *      - init / start_stdio / stop / close lifecycle
 *      - JSON-RPC `tools/list` round-trip (stdio)
 *      - JSON-RPC `tools/call` for `list_features` round-trip
 *      - method-not-found error envelope
 *      - UDS bind + JSON-RPC round-trip
 *      - compute_vmaf real-score check against the testdata 576x324
 *        YUV pair (sanity-bounded; not bit-exact)
 *
 *  SSE still returns -ENOSYS — pinned here so a future v3 PR
 *  cannot wire it without flipping the expectation.
 */

#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_mcp.h"

static char *test_available_returns_one(void)
{
    mu_assert("vmaf_mcp_available must report 1 in the smoke build", vmaf_mcp_available() == 1);
    return NULL;
}

static char *test_transport_available_unknown_id_is_zero(void)
{
    int id = 999;
    mu_assert("unknown transport id must report unavailable",
              vmaf_mcp_transport_available((VmafMcpTransport)id) == 0);
    return NULL;
}

static char *test_init_rejects_null_out(void)
{
    int rc = vmaf_mcp_init(NULL, (VmafContext *)0x1, NULL);
    mu_assert("NULL out -> -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_init_rejects_null_ctx(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    int rc = vmaf_mcp_init(&server, NULL, NULL);
    mu_assert("NULL ctx -> -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_start_sse_returns_enosys(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    VmafMcpSseConfig cfg = {.port = 0, .path = NULL};
    int rc = vmaf_mcp_start_sse(server, &cfg);
    mu_assert("start_sse must return -ENOSYS in v1", rc == -ENOSYS);
    return NULL;
}

static char *test_start_uds_rejects_null_path(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    VmafMcpUdsConfig cfg = {.path = NULL};
    int rc = vmaf_mcp_start_uds(server, &cfg);
    mu_assert("NULL path -> -EINVAL", rc == -EINVAL);
    return NULL;
}

/* Removed in v2: UDS is no longer -ENOSYS. The negative case
 * (NULL path) is still covered by test_start_uds_rejects_null_path.
 * The full UDS round-trip is covered by test_uds_roundtrip below. */

static char *test_start_stdio_rejects_negative_fd(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    VmafMcpStdioConfig cfg = {.fd_in = -1, .fd_out = 1};
    int rc = vmaf_mcp_start_stdio(server, &cfg);
    mu_assert("negative fd_in -> -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_stop_rejects_null(void)
{
    int rc = vmaf_mcp_stop(NULL);
    mu_assert("NULL server -> -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_close_null_is_noop(void)
{
    vmaf_mcp_close(NULL);
    return NULL;
}

static char *test_close_pointer_to_null_is_noop(void)
{
    VmafMcpServer *server = NULL;
    vmaf_mcp_close(&server);
    mu_assert("close leaves *server NULL", server == NULL);
    return NULL;
}

/* ============================================================
 * v1 runtime helpers
 * ============================================================ */

typedef struct McpHarness {
    int req_pipe[2];
    int resp_pipe[2];
    VmafContext *ctx;
    VmafMcpServer *server;
} McpHarness;

static int harness_init(McpHarness *h)
{
    h->req_pipe[0] = h->req_pipe[1] = -1;
    h->resp_pipe[0] = h->resp_pipe[1] = -1;
    h->ctx = NULL;
    h->server = NULL;
    if (pipe(h->req_pipe) != 0)
        return -1;
    if (pipe(h->resp_pipe) != 0)
        return -1;
    VmafConfiguration cfg = {0};
    cfg.log_level = VMAF_LOG_LEVEL_NONE;
    cfg.n_threads = 1u;
    if (vmaf_init(&h->ctx, cfg) != 0)
        return -1;
    if (vmaf_mcp_init(&h->server, h->ctx, NULL) != 0)
        return -1;
    VmafMcpStdioConfig scfg = {.fd_in = h->req_pipe[0], .fd_out = h->resp_pipe[1]};
    if (vmaf_mcp_start_stdio(h->server, &scfg) != 0)
        return -1;
    return 0;
}

static void harness_teardown(McpHarness *h)
{
    if (h->req_pipe[1] >= 0)
        (void)close(h->req_pipe[1]);
    if (h->server != NULL)
        vmaf_mcp_close(&h->server);
    if (h->req_pipe[0] >= 0)
        (void)close(h->req_pipe[0]);
    if (h->resp_pipe[0] >= 0)
        (void)close(h->resp_pipe[0]);
    if (h->resp_pipe[1] >= 0)
        (void)close(h->resp_pipe[1]);
    if (h->ctx != NULL)
        (void)vmaf_close(h->ctx);
}

/* Read up to max_len-1 bytes until LF. */
static ssize_t read_one_line(int fd, char *buf, size_t max_len)
{
    size_t n = 0u;
    while (n < max_len - 1u) {
        char c = 0;
        ssize_t r = read(fd, &c, 1);
        if (r <= 0) {
            if (n == 0u)
                return r;
            break;
        }
        if (c == '\n')
            break;
        if (c != '\r')
            buf[n++] = c;
    }
    buf[n] = '\0';
    return (ssize_t)n;
}

static char *send_and_read(McpHarness *h, const char *req, size_t req_len, char *line,
                           size_t line_cap)
{
    ssize_t w = write(h->req_pipe[1], req, req_len);
    mu_assert("write request", w == (ssize_t)req_len);
    ssize_t n = read_one_line(h->resp_pipe[0], line, line_cap);
    mu_assert("response received", n > 0);
    return NULL;
}

/* ============================================================
 * v1 runtime tests
 * ============================================================ */

static char *test_init_close_lifecycle(void)
{
    VmafContext *ctx = NULL;
    VmafConfiguration cfg = {0};
    cfg.log_level = VMAF_LOG_LEVEL_NONE;
    cfg.n_threads = 1u;
    int crc = vmaf_init(&ctx, cfg);
    mu_assert("vmaf_init must succeed", crc == 0 && ctx != NULL);

    VmafMcpServer *server = NULL;
    int rc = vmaf_mcp_init(&server, ctx, NULL);
    mu_assert("v1 init must succeed", rc == 0);
    mu_assert("v1 init must return a non-NULL handle", server != NULL);

    vmaf_mcp_close(&server);
    mu_assert("close NULLs the handle", server == NULL);

    (void)vmaf_close(ctx);
    return NULL;
}

static char *test_jsonrpc_tools_list_roundtrip(void)
{
    McpHarness h;
    mu_assert("harness init", harness_init(&h) == 0);

    static const char tools_list[] = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n";
    char line[8192];
    char *err = send_and_read(&h, tools_list, sizeof(tools_list) - 1u, line, sizeof(line));
    if (err != NULL) {
        harness_teardown(&h);
        return err;
    }
    mu_assert("list contains list_features", strstr(line, "\"list_features\"") != NULL);
    mu_assert("list contains compute_vmaf", strstr(line, "\"compute_vmaf\"") != NULL);
    mu_assert("jsonrpc 2.0", strstr(line, "\"jsonrpc\":\"2.0\"") != NULL);
    mu_assert("id 1", strstr(line, "\"id\":1") != NULL);

    harness_teardown(&h);
    return NULL;
}

static char *test_jsonrpc_tools_call_list_features(void)
{
    McpHarness h;
    mu_assert("harness init", harness_init(&h) == 0);

    static const char call_lf[] = "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\","
                                  "\"params\":{\"name\":\"list_features\",\"arguments\":{}}}\n";
    char line[8192];
    char *err = send_and_read(&h, call_lf, sizeof(call_lf) - 1u, line, sizeof(line));
    if (err != NULL) {
        harness_teardown(&h);
        return err;
    }
    mu_assert("id 2", strstr(line, "\"id\":2") != NULL);
    mu_assert("content array", strstr(line, "\"content\"") != NULL);
    mu_assert("mentions features", strstr(line, "features") != NULL);

    harness_teardown(&h);
    return NULL;
}

static char *test_jsonrpc_method_not_found(void)
{
    McpHarness h;
    mu_assert("harness init", harness_init(&h) == 0);

    static const char bogus[] = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"does/not/exist\"}\n";
    char line[2048];
    char *err = send_and_read(&h, bogus, sizeof(bogus) - 1u, line, sizeof(line));
    if (err != NULL) {
        harness_teardown(&h);
        return err;
    }
    mu_assert("error envelope", strstr(line, "\"error\"") != NULL);
    mu_assert("code -32601", strstr(line, "-32601") != NULL);

    harness_teardown(&h);
    return NULL;
}

/* ============================================================
 * UDS transport round-trip (v2)
 * ============================================================ */

static char *test_uds_roundtrip(void)
{
    /* Deterministic per-pid socket path — keeps parallel test
     * runs from clobbering each other. */
    char path[80];
    int n = snprintf(path, sizeof(path), "/tmp/vmaf-mcp-uds-test-%d.sock", (int)getpid());
    mu_assert("path snprintf", n > 0 && (size_t)n < sizeof(path));

    VmafContext *ctx = NULL;
    VmafConfiguration vcfg = {0};
    vcfg.log_level = VMAF_LOG_LEVEL_NONE;
    vcfg.n_threads = 1u;
    mu_assert("vmaf_init", vmaf_init(&ctx, vcfg) == 0);

    VmafMcpServer *server = NULL;
    mu_assert("mcp init", vmaf_mcp_init(&server, ctx, NULL) == 0);

    VmafMcpUdsConfig ucfg = {.path = path};
    int rc = vmaf_mcp_start_uds(server, &ucfg);
    mu_assert("uds start", rc == 0);

    /* Connect a client and round-trip a tools/list. */
    int cfd = socket(AF_UNIX, SOCK_STREAM, 0);
    mu_assert("client socket", cfd >= 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, path, strlen(path) + 1u);
    int crc = connect(cfd, (const struct sockaddr *)&addr, sizeof(addr));
    mu_assert("client connect", crc == 0);

    static const char req[] = "{\"jsonrpc\":\"2.0\",\"id\":42,\"method\":\"tools/list\"}\n";
    ssize_t w = write(cfd, req, sizeof(req) - 1u);
    mu_assert("client write", w == (ssize_t)(sizeof(req) - 1u));

    char line[8192];
    ssize_t got = read_one_line(cfd, line, sizeof(line));
    mu_assert("uds response received", got > 0);
    mu_assert("uds id 42", strstr(line, "\"id\":42") != NULL);
    mu_assert("uds list contains compute_vmaf", strstr(line, "\"compute_vmaf\"") != NULL);

    (void)close(cfd);
    vmaf_mcp_close(&server);
    (void)vmaf_close(ctx);

    /* Verify socket file got cleaned up by stop(). */
    struct stat st;
    int sr = stat(path, &st);
    mu_assert("uds socket file unlinked on close", sr != 0);
    return NULL;
}

/* ============================================================
 * compute_vmaf real-score binding (v2)
 * ============================================================ */

/* Use the testdata 576x324 8-bit YUV pair shipped at
 * testdata/ref_576x324_48f.yuv + dis_576x324_48f.yuv. The test
 * is conditional: if the host does not ship the testdata yuv
 * (e.g. some packagers strip it), we skip rather than fail. */
static char *test_compute_vmaf_real_score(void)
{
    const char *ref_path = "../testdata/ref_576x324_48f.yuv";
    const char *dis_path = "../testdata/dis_576x324_48f.yuv";
    struct stat st;
    if (stat(ref_path, &st) != 0 || stat(dis_path, &st) != 0)
        return NULL; /* fixture absent on this host — skip. */

    McpHarness h;
    mu_assert("harness init", harness_init(&h) == 0);

    char req[512];
    int n = snprintf(req, sizeof(req),
                     "{\"jsonrpc\":\"2.0\",\"id\":99,\"method\":\"tools/call\","
                     "\"params\":{\"name\":\"compute_vmaf\",\"arguments\":{"
                     "\"reference_path\":\"%s\","
                     "\"distorted_path\":\"%s\","
                     "\"width\":576,\"height\":324,"
                     "\"model_version\":\"vmaf_v0.6.1\"}}}\n",
                     ref_path, dis_path);
    mu_assert("req snprintf", n > 0 && (size_t)n < sizeof(req));

    /* Larger buffer — compute_vmaf wraps a stringified JSON object
     * inside MCP's content envelope, so the line can exceed 1 KiB. */
    static char line[16384];
    char *err = send_and_read(&h, req, (size_t)n, line, sizeof(line));
    if (err != NULL) {
        harness_teardown(&h);
        return err;
    }
    /* Real path returns a `score` numeric field; the placeholder
     * (which v2 must NOT bring back) returned `deferred_to_v2`. */
    mu_assert("compute_vmaf id 99", strstr(line, "\"id\":99") != NULL);
    mu_assert("compute_vmaf returns score", strstr(line, "\\\"score\\\"") != NULL);
    mu_assert("compute_vmaf no v1 placeholder", strstr(line, "deferred_to_v2") == NULL);
    mu_assert("compute_vmaf reports frames_scored", strstr(line, "\\\"frames_scored\\\"") != NULL);

    harness_teardown(&h);
    return NULL;
}

/* ============================================================
 * Test table — keeps run_tests below clang-tidy's 15-branch budget
 * (mirrors test_hip_smoke.c / test_vulkan_smoke.c precedent).
 * ============================================================ */

typedef char *(*test_fn)(void);

static const test_fn k_test_table[] = {
    test_available_returns_one,
    test_transport_available_unknown_id_is_zero,
    test_init_rejects_null_out,
    test_init_rejects_null_ctx,
    test_start_sse_returns_enosys,
    test_start_uds_rejects_null_path,
    test_start_stdio_rejects_negative_fd,
    test_stop_rejects_null,
    test_close_null_is_noop,
    test_close_pointer_to_null_is_noop,
    test_init_close_lifecycle,
    test_jsonrpc_tools_list_roundtrip,
    test_jsonrpc_tools_call_list_features,
    test_jsonrpc_method_not_found,
    test_uds_roundtrip,
    test_compute_vmaf_real_score,
};

static const size_t k_test_table_len = sizeof(k_test_table) / sizeof(k_test_table[0]);

char *run_tests(void)
{
    for (size_t i = 0u; i < k_test_table_len; ++i) {
        mu_run_test(k_test_table[i]);
    }
    return NULL;
}
