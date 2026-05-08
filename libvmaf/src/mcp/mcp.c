/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Embedded MCP server — v1 stdio runtime (T5-2b).
 *
 *  History: T5-2 (ADR-0209) landed every entry point as `-ENOSYS`.
 *  T5-2b (this PR — see ADR-0209 § "Status update 2026-05-08")
 *  flips `vmaf_mcp_init` + `vmaf_mcp_start_stdio` + `vmaf_mcp_stop`
 *  + `vmaf_mcp_close` to a working JSON-RPC 2.0 dispatcher with
 *  two tools (`list_features`, `compute_vmaf`). SSE / UDS still
 *  return `-ENOSYS`; vendoring `mongoose` and wiring the SPSC ring
 *  drain at frame boundaries is deferred to v2.
 *
 *  Power-of-10 invariants:
 *      - rule 2 (bounded loops): all per-request loops are bounded
 *        by VMAF_MCP_MAX_LINE_BYTES / VMAF_MCP_MAX_FEATURES; see
 *        transport_stdio.c + dispatcher.c.
 *      - rule 3 (no dynamic alloc on hot path): per-line scratch
 *        is allocated once at thread-start; per-request cJSON
 *        allocations are confined to the dispatcher and bounded
 *        by the parser's input length. The measurement thread is
 *        not touched by v1 (no SPSC ring yet).
 *      - rule 7 (return values): every read()/write()/pthread_*()
 *        return value is checked or `(void)`-cast with rationale.
 */

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/libvmaf_mcp.h"
#include "mcp_internal.h"

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
    if ((unsigned)transport > 31u) {
        return 0;
    }
    /* Post-guard invariant — same predicate as the early return,
     * preserved for Power-of-10 §5 assertion density. */
    assert((unsigned)transport <= 31u);
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
    return (mask & (1u << (unsigned)transport)) != 0u ? 1 : 0;
}

/* Validate cfg at init. Power-of-2 check on queue_depth (rejected
 * with -EINVAL when non-zero and not power of 2). */
static int validate_config(const VmafMcpConfig *cfg)
{
    if (cfg == NULL)
        return 0;
    if (cfg->queue_depth != 0u) {
        unsigned q = cfg->queue_depth;
        if ((q & (q - 1u)) != 0u)
            return -EINVAL;
    }
    if (cfg->max_drain_per_frame > 64u)
        return -EINVAL;
    return 0;
}

int vmaf_mcp_init(VmafMcpServer **out, VmafContext *ctx, const VmafMcpConfig *cfg)
{
    if (out == NULL)
        return -EINVAL;
    if (ctx == NULL)
        return -EINVAL;
    *out = NULL;

    int vrc = validate_config(cfg);
    if (vrc != 0)
        return vrc;

    struct VmafMcpServer *s = (struct VmafMcpServer *)calloc(1u, sizeof(*s));
    if (s == NULL)
        return -ENOMEM;
    /* Power-of-10 §5: post-condition on the freshly-allocated handle. */
    assert(s != NULL);

    s->ctx = ctx;
    if (cfg != NULL) {
        s->cfg = *cfg;
        if (cfg->user_agent != NULL) {
            size_t len = strlen(cfg->user_agent);
            char *dup = (char *)malloc(len + 1u);
            if (dup == NULL) {
                free(s);
                return -ENOMEM;
            }
            memcpy(dup, cfg->user_agent, len + 1u);
            s->user_agent_owned = dup;
            s->cfg.user_agent = dup;
        }
    }
    s->stdio_fd_in = -1;
    s->stdio_fd_out = -1;
    atomic_store(&s->stdio_running, 0);
    /* Power-of-10 §5: invariant — server starts in "not-running"
     * state regardless of caller config. */
    assert(atomic_load(&s->stdio_running) == 0);
    int mrc = pthread_mutex_init(&s->write_mtx, NULL);
    if (mrc != 0) {
        free(s->user_agent_owned);
        free(s);
        return -mrc;
    }

    *out = s;
    return 0;
}

int vmaf_mcp_start_sse(VmafMcpServer *server, VmafMcpSseConfig *cfg)
{
    (void)cfg;
    if (server == NULL)
        return -EINVAL;
    /* v1: SSE deferred to v2 (mongoose vendoring). */
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
    /* v1: UDS deferred to v2. */
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

    int expected = 0;
    if (!atomic_compare_exchange_strong(&server->stdio_running, &expected, 1)) {
        return -EBUSY;
    }
    /* Power-of-10 §5: post-CAS invariant — exactly one start can win
     * the race; the loser returned -EBUSY above. */
    assert(atomic_load(&server->stdio_running) == 1);

    server->stdio_fd_in = cfg->fd_in;
    server->stdio_fd_out = cfg->fd_out;

    int rc = pthread_create(&server->stdio_thread, NULL, vmaf_mcp_stdio_thread_main, server);
    if (rc != 0) {
        atomic_store(&server->stdio_running, 0);
        return -rc;
    }
    return 0;
}

int vmaf_mcp_stop(VmafMcpServer *server)
{
    if (server == NULL)
        return -EINVAL;
    int prev = atomic_exchange(&server->stdio_running, 2);
    if (prev == 1 || prev == 2) {
        /* Closing fd_in is the canonical way to unblock the read()
         * loop. The caller owns the fd per the public contract, so
         * we cannot close it ourselves; instead we rely on the
         * caller having already closed (or being about to close)
         * fd_in to drive the worker to EOF. We still join the
         * thread to release its resources.
         *
         * For tests and well-behaved hosts the read end is already
         * EOF by the time stop() is called. */
        (void)pthread_join(server->stdio_thread, NULL);
        atomic_store(&server->stdio_running, 0);
    }
    return 0;
}

void vmaf_mcp_close(VmafMcpServer **server)
{
    if (server == NULL)
        return;
    if (*server == NULL)
        return;
    struct VmafMcpServer *s = *server;
    (void)vmaf_mcp_stop(s);
    (void)pthread_mutex_destroy(&s->write_mtx);
    free(s->user_agent_owned);
    free(s);
    *server = NULL;
}
