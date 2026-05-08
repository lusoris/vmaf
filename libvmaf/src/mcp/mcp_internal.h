/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Internal MCP runtime types — not part of the public ABI. The
 *  public header (`libvmaf/include/libvmaf/libvmaf_mcp.h`) only
 *  forward-declares `VmafMcpServer`; this header carries the
 *  storage class for the v1 stdio runtime.
 *
 *  v1 scope (T5-2b, ADR-0209 § "Status update 2026-05-08"):
 *      - stdio transport only; SSE/UDS still return -ENOSYS.
 *      - line-delimited JSON-RPC framing (one request per line);
 *        LSP `Content-Length:` framing deferred to v2.
 *      - dispatcher routes `tools/list`, `tools/call`,
 *        `resources/list`. Tools: `list_features`, `compute_vmaf`
 *        (placeholder — real scoring binding lands with v2).
 */

#ifndef LIBVMAF_SRC_MCP_INTERNAL_H_
#define LIBVMAF_SRC_MCP_INTERNAL_H_

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_mcp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hard caps — Power-of-10 rule 2 (bounded loops) + rule 3
 * (no dynamic alloc on the hot path). */
#define VMAF_MCP_MAX_LINE_BYTES ((size_t)64u * 1024u) /* 64 KiB request line. */
#define VMAF_MCP_MAX_FEATURES 256u                    /* `list_features` cap. */

/* Forward decl for the stdio worker — defined in transport_stdio.c,
 * dispatched from mcp.c. Declared here (rather than file-local in
 * mcp.c) to satisfy clang-tidy's misc-use-internal-linkage check. */
void *vmaf_mcp_stdio_thread_main(void *arg);

struct VmafMcpServer {
    VmafContext *ctx;       /* Borrowed; not owned. */
    VmafMcpConfig cfg;      /* Copied at init; user_agent dup'd. */
    char *user_agent_owned; /* NULL when defaulted. */

    /* stdio transport state. */
    int stdio_fd_in;
    int stdio_fd_out;
    pthread_t stdio_thread;
    atomic_int stdio_running;  /* 0 = not started, 1 = running, 2 = stop-requested. */
    pthread_mutex_t write_mtx; /* Serialises responses across future transports. */
};

/**
 * Dispatch a single JSON-RPC request string and produce a response
 * string. Caller frees `*response_out` with free().
 *
 * @param server       Server instance (borrowed).
 * @param request_buf  NUL-terminated JSON-RPC request.
 * @param response_out Receives heap-allocated NUL-terminated response,
 *                     or NULL for a notification (id absent). Caller frees.
 * @return 0 on success (response produced or notification swallowed).
 *         Negative errno on internal failure (parser OOM etc.) — the
 *         caller should still attempt to surface a JSON-RPC error
 *         response when the input was non-empty.
 */
int vmaf_mcp_dispatch(struct VmafMcpServer *server, const char *request_buf, char **response_out);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_SRC_MCP_INTERNAL_H_ */
