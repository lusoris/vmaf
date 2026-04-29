/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Embedded MCP server — scaffold-only stub TU.
 *
 *  Every entry point in this translation unit returns -ENOSYS or a
 *  trivially-safe constant. The runtime (cJSON + mongoose vendoring,
 *  dedicated MCP pthread, SPSC ring buffer drained at frame
 *  boundaries, SSE / UDS / stdio transport bodies) lands via T5-2b
 *  per ADR-0209 § "What lands next". Smoke test
 *  `libvmaf/test/test_mcp_smoke.c` pins the -ENOSYS contract so a
 *  future runtime PR cannot accidentally enable a transport without
 *  flipping the smoke expectations.
 *
 *  Power-of-10 invariants reserved for the runtime PR (rule 3 — no
 *  alloc on the measurement-thread hot path; rule 2 — bounded drain
 *  loop) are documented in libvmaf/include/libvmaf/libvmaf_mcp.h
 *  and ADR-0209.
 */

#include <errno.h>
#include <stddef.h>

#include "libvmaf/libvmaf_mcp.h"

/* Per-build-flag availability. The umbrella `enable_mcp` flag flips
 * `HAVE_MCP`; per-transport sub-flags flip the matching `HAVE_MCP_*`
 * macros. The header surface is identical either way; only the
 * runtime PR distinguishes built-without (returns -ENOSYS forever)
 * vs built-with (returns -ENOSYS only until the runtime is wired).
 */

int vmaf_mcp_available(void)
{
#ifdef HAVE_MCP
    return 1;
#else
    return 0;
#endif
}

int vmaf_mcp_transport_available(VmafMcpTransport transport)
{
    /* Build a per-call bitmask from the per-transport defines, then
     * test the bit matching the queried transport. The arithmetic
     * is preprocessor-fed, so the compiler folds the function body
     * to a constant load + bittest at every call site. Avoiding a
     * per-arm `#ifdef` keeps the body structurally distinct (no
     * `bugprone-branch-clone`) regardless of which sub-flags are
     * on. */
    unsigned mask = 0u;
#ifdef HAVE_MCP_SSE
    mask |= 1u << (unsigned)VMAF_MCP_TRANSPORT_SSE;
#endif
#ifdef HAVE_MCP_UDS
    mask |= 1u << (unsigned)VMAF_MCP_TRANSPORT_UDS;
#endif
#ifdef HAVE_MCP_STDIO
    mask |= 1u << (unsigned)VMAF_MCP_TRANSPORT_STDIO;
#endif
    if ((unsigned)transport > 31u)
        return 0;
    return (mask & (1u << (unsigned)transport)) != 0u ? 1 : 0;
}

int vmaf_mcp_init(VmafMcpServer **out, VmafContext *ctx, const VmafMcpConfig *cfg)
{
    /* Argument validation precedes the -ENOSYS so the caller-side
     * contract is enforceable from the scaffold day one — a runtime
     * PR that forgets a NULL guard would regress the smoke test. */
    (void)cfg;
    if (out == NULL)
        return -EINVAL;
    if (ctx == NULL)
        return -EINVAL;
    *out = NULL;
    return -ENOSYS;
}

int vmaf_mcp_start_sse(VmafMcpServer *server, VmafMcpSseConfig *cfg)
{
    (void)cfg;
    if (server == NULL)
        return -EINVAL;
    return -ENOSYS;
}

int vmaf_mcp_start_uds(VmafMcpServer *server, const VmafMcpUdsConfig *cfg)
{
    if (server == NULL)
        return -EINVAL;
    if (cfg == NULL)
        return -EINVAL;
    if (cfg->path == NULL)
        return -EINVAL;
    return -ENOSYS;
}

int vmaf_mcp_start_stdio(VmafMcpServer *server, const VmafMcpStdioConfig *cfg)
{
    if (server == NULL)
        return -EINVAL;
    if (cfg == NULL)
        return -EINVAL;
    if (cfg->fd_in < 0)
        return -EINVAL;
    if (cfg->fd_out < 0)
        return -EINVAL;
    return -ENOSYS;
}

int vmaf_mcp_stop(VmafMcpServer *server)
{
    if (server == NULL)
        return -EINVAL;
    /* No transports can be running in the scaffold — `_start_*` all
     * return -ENOSYS. Once the runtime PR lands, this will join the
     * MCP threads and drain pending replies. */
    return 0;
}

void vmaf_mcp_close(VmafMcpServer **server)
{
    if (server == NULL)
        return;
    /* No-op in the scaffold: `vmaf_mcp_init` never produces a
     * non-NULL handle. Defensively NULL the caller's pointer in
     * case some future caller stashed one for compatibility. */
    *server = NULL;
}
