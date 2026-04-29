/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Build + ABI smoke test for the embedded MCP scaffold (ADR-0209 /
 *  T5-2). Pins the -ENOSYS contract so a future runtime PR (T5-2b)
 *  can't accidentally wire a transport without flipping the smoke
 *  expectations. Mirrors the pattern of test_vulkan_smoke.c
 *  (ADR-0175 scaffold).
 */

#include <errno.h>
#include <stddef.h>

#include "test.h"

#include "libvmaf/libvmaf_mcp.h"

static char *test_available_returns_one(void)
{
    /* Compiled with `-DHAVE_MCP=1` via the smoke target's c_args
     * regardless of the umbrella build flag — the test binary itself
     * is gated by `enable_mcp` in meson.build, so reaching this code
     * path means the umbrella flag was on at configure time. */
    mu_assert("vmaf_mcp_available must report 1 in the smoke build", vmaf_mcp_available() == 1);
    return NULL;
}

static char *test_transport_available_unknown_id_is_zero(void)
{
    /* Cast through int to feed an out-of-enum value; the API
     * contract for unknown ids is "returns 0". */
    int id = 999;
    mu_assert("unknown transport id must report unavailable",
              vmaf_mcp_transport_available((VmafMcpTransport)id) == 0);
    return NULL;
}

static char *test_init_rejects_null_out(void)
{
    int rc = vmaf_mcp_init(NULL, (VmafContext *)0x1, NULL);
    mu_assert("NULL out → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_init_rejects_null_ctx(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    int rc = vmaf_mcp_init(&server, NULL, NULL);
    mu_assert("NULL ctx → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_init_returns_enosys_until_runtime(void)
{
    /* T5-2b will flip this to 0 / -ENOMEM; until then the scaffold
     * must report -ENOSYS so callers can fall back cleanly. */
    VmafMcpServer *server = NULL;
    VmafContext *ctx = (VmafContext *)0x1; /* opaque; not deref'd in scaffold. */
    int rc = vmaf_mcp_init(&server, ctx, NULL);
    mu_assert("init must return -ENOSYS in scaffold", rc == -ENOSYS);
    mu_assert("init must NULL out on -ENOSYS", server == NULL);
    return NULL;
}

static char *test_start_sse_rejects_null_server(void)
{
    VmafMcpSseConfig cfg = {.port = 0, .path = NULL};
    int rc = vmaf_mcp_start_sse(NULL, &cfg);
    mu_assert("NULL server → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_start_uds_rejects_null_path(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    VmafMcpUdsConfig cfg = {.path = NULL};
    int rc = vmaf_mcp_start_uds(server, &cfg);
    mu_assert("NULL path → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_start_uds_rejects_null_cfg(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    int rc = vmaf_mcp_start_uds(server, NULL);
    mu_assert("NULL cfg → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_start_stdio_rejects_negative_fd(void)
{
    VmafMcpServer *server = (VmafMcpServer *)0x1;
    VmafMcpStdioConfig cfg = {.fd_in = -1, .fd_out = 1};
    int rc = vmaf_mcp_start_stdio(server, &cfg);
    mu_assert("negative fd_in → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_stop_rejects_null(void)
{
    int rc = vmaf_mcp_stop(NULL);
    mu_assert("NULL server → -EINVAL", rc == -EINVAL);
    return NULL;
}

static char *test_close_null_is_noop(void)
{
    /* Must not crash; nothing to assert beyond reaching the next
     * line. */
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

char *run_tests(void)
{
    mu_run_test(test_available_returns_one);
    mu_run_test(test_transport_available_unknown_id_is_zero);
    mu_run_test(test_init_rejects_null_out);
    mu_run_test(test_init_rejects_null_ctx);
    mu_run_test(test_init_returns_enosys_until_runtime);
    mu_run_test(test_start_sse_rejects_null_server);
    mu_run_test(test_start_uds_rejects_null_path);
    mu_run_test(test_start_uds_rejects_null_cfg);
    mu_run_test(test_start_stdio_rejects_negative_fd);
    mu_run_test(test_stop_rejects_null);
    mu_run_test(test_close_null_is_noop);
    mu_run_test(test_close_pointer_to_null_is_noop);
    return NULL;
}
