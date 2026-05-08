# Embedded MCP server (in-process, inside libvmaf)

> **Status: T5-2b v1 stdio runtime landed (2026-05-08).**
> `vmaf_mcp_init` / `vmaf_mcp_start_stdio` / `vmaf_mcp_stop` /
> `vmaf_mcp_close` are wired and respond to JSON-RPC 2.0 requests
> (`tools/list`, `tools/call`, `resources/list`, `initialize`).
> Tools shipped: `list_features`, `compute_vmaf` (placeholder —
> validates inputs and returns a deferred-to-v2 marker; the real
> scoring binding lands in v2). SSE and UDS transports remain
> `-ENOSYS`; mongoose vendoring deferred to v2. The standalone
> Python MCP server under [`mcp-server/vmaf-mcp/`](../../mcp-server/vmaf-mcp/)
> remains the recommended default for "score a video" workflows.

The embedded MCP server runs **inside the host process** that
loaded `libvmaf.so` — typically `vmaf` (the CLI), `ffmpeg`'s
`libvmaf` filter, or a Python harness. Unlike the standalone
Python server (which wraps the `vmaf` CLI), the embedded server
exposes JSON-RPC tools that can introspect and steer a *running*
measurement: query per-feature scores mid-stream, request a model
hot-swap at the next frame boundary, observe queue pressure, etc.

See [ADR-0128](../adr/0128-embedded-mcp-in-libvmaf.md) for the
governance decision and [Research-0005](../research/0005-embedded-mcp-transport.md)
for the design rationale.

## Two MCP surfaces in this fork — which one to use

| Workflow | Surface | Notes |
|---|---|---|
| "Score a video, hand the result to my agent." | **External Python MCP server** (`mcp-server/vmaf-mcp/`) | Recommended default. Spawns `vmaf` as a child process. See [`docs/mcp/index.md`](index.md). |
| "Steer a running measurement: hot-swap models, query mid-stream state." | **Embedded MCP server** (this doc) | Runs in-process. Currently scaffold-only — the runtime arrives in T5-2b. |

The two are additive — running both at the same time is fine.

## Build

```bash
# Default fork build does NOT include the embedded MCP surface.
meson setup build -Denable_cuda=false -Denable_sycl=false
ninja -C build

# Opt in to the scaffold (returns -ENOSYS everywhere until T5-2b):
meson setup build -Denable_mcp=true \
                  -Denable_mcp_sse=true \
                  -Denable_mcp_uds=true \
                  -Denable_mcp_stdio=true
ninja -C build
meson test -C build  # includes test_mcp_smoke
```

| Flag | Default | Purpose |
|---|---|---|
| `-Denable_mcp` | `false` | Umbrella — compiles `libvmaf/src/mcp/` + installs `libvmaf_mcp.h` + builds `test_mcp_smoke`. |
| `-Denable_mcp_sse` | `false` | Compile in the Server-Sent-Events / loopback HTTP transport. Requires `enable_mcp=true`. |
| `-Denable_mcp_uds` | `false` | Compile in the Unix domain socket transport. POSIX-only; non-POSIX hosts return `-ENODEV` at runtime. Requires `enable_mcp=true`. |
| `-Denable_mcp_stdio` | `false` | Compile in the stdio (LSP-framed JSON-RPC on a caller-supplied fd pair) transport. Requires `enable_mcp=true`. |

The umbrella flag is independent of the per-transport sub-flags.
A library built with `-Denable_mcp=true -Denable_mcp_sse=true`
exposes the public `vmaf_mcp_*` symbols, advertises SSE
availability via `vmaf_mcp_transport_available(VMAF_MCP_TRANSPORT_SSE)`,
and reports unavailability for UDS / stdio.

## API at a glance

```c
#include <libvmaf/libvmaf.h>
#include <libvmaf/libvmaf_mcp.h>

VmafContext *ctx = NULL;
/* ... vmaf_init(&ctx, ...) ... */

VmafMcpServer *server = NULL;
VmafMcpConfig cfg = { .queue_depth = 64, .max_drain_per_frame = 4 };
int rc = vmaf_mcp_init(&server, ctx, &cfg);
if (rc < 0) {
    /* Currently always -ENOSYS — runtime arrives in T5-2b. */
    fprintf(stderr, "MCP unavailable: %d\n", rc);
    /* Fall back to the external Python server, or proceed without. */
}

/* Each transport spawns a dedicated MCP pthread on the server. */
VmafMcpSseConfig sse = { .port = 8723, .path = "/mcp/sse" };
(void)vmaf_mcp_start_sse(server, &sse);

/* ... vmaf_read_pictures / vmaf_score_pooled ... */

(void)vmaf_mcp_stop(server);
vmaf_mcp_close(&server);
/* vmaf_close(ctx); */
```

The full API is documented in
[`libvmaf/include/libvmaf/libvmaf_mcp.h`](../../libvmaf/include/libvmaf/libvmaf_mcp.h).

## Transport summary

### SSE (Server-Sent Events)

- **Spec status:** canonical MCP transport (introduced 2024-11).
- **Bind:** `127.0.0.1` only — `vmaf_mcp_start_sse` refuses
  non-loopback in v1.
- **Wire:** SSE event stream on `GET /mcp/sse`; client posts
  requests to `POST /mcp/request` with an `X-MCP-Session` header.
- **Use case:** Claude Desktop, Cursor, and other agents speaking
  the canonical MCP remote transport.

### UDS (Unix domain socket — fork extension)

- **Spec status:** *not* in the MCP spec. A libvmaf-specific
  extension matching how systemd-managed pipelines already work.
- **Bind:** filesystem path you supply; server creates the socket
  file mode 0700.
- **Wire:** newline-delimited JSON-RPC (one JSON object per line,
  reply on the same connection).
- **Use case:** headless Linux boxes where filesystem permissions
  give cleaner auth than loopback TCP.

### stdio (LSP-framed)

- **Spec status:** canonical MCP transport.
- **Bind:** caller hands over an fd pair (typically fd 3 / fd 4
  from a parent-spawned wrapper). The host's own stdin / stdout
  are *not* claimed.
- **Wire:** LSP-style `Content-Length:` framed JSON-RPC, identical
  to the spec's reference framing.
- **Use case:** child-process spawned by an agent that already
  sets up an inheritable fd pair.

## Threading + Power-of-10 invariants (per ADR-0128 + Research-0005)

- One **MCP pthread** per active transport. The thread owns the
  socket, owns JSON parsing, owns all per-request allocation. It
  does **not** touch measurement state directly.
- One **SPSC ring buffer** per server, sized at `vmaf_mcp_init`
  from `VmafMcpConfig.queue_depth`. Pre-allocated; the
  measurement-thread hot path performs **no allocation** after
  init (NASA Power-of-10 rule 3).
- The measurement thread drains at most
  `VmafMcpConfig.max_drain_per_frame` envelopes per frame
  boundary (NASA Power-of-10 rule 2 — bounded loop).
- Queue overflow returns `-EAGAIN` style errors from the MCP
  thread to the agent, never blocks the measurement thread.

These invariants are documented in the public header
([`libvmaf_mcp.h`](../../libvmaf/include/libvmaf/libvmaf_mcp.h))
and locked in via the T5-2b runtime PR's TSan tests.

## Status table

| Component | Status | PR / ADR |
|---|---|---|
| Public header `libvmaf_mcp.h` | Landed (scaffold) | T5-2 / [ADR-0209](../adr/0209-mcp-embedded-scaffold.md) |
| TU `libvmaf/src/mcp/mcp.c` | Landed (v1 runtime; init / start_stdio / stop / close wired) | T5-2b / ADR-0209 § Status update 2026-05-08 |
| Vendored cJSON v1.7.18 (MIT) under `libvmaf/src/mcp/3rdparty/cJSON/` | Landed | T5-2b |
| JSON-RPC dispatcher (`tools/list`, `tools/call`, `resources/list`, `initialize`) | Landed | T5-2b |
| Build flags + per-transport sub-flags | Landed (default off) | T5-2 |
| Smoke + protocol test (15 sub-tests, real round-trip) | Landed | T5-2b |
| stdio transport body | Landed (line-delimited JSON-RPC; LSP `Content-Length:` framing deferred to v2) | T5-2b |
| SSE transport body | Pending — returns `-ENOSYS` | v2 (mongoose vendoring) |
| UDS transport body | Pending — returns `-ENOSYS` | v2 |
| Tool: `list_features` (read-only) | Landed | T5-2b |
| Tool: `compute_vmaf` (validates inputs; deferred-to-v2 marker) | Landed | T5-2b (binding to scoring engine in v2) |
| Tool: `vmaf.request_model_swap` (mutating, separate ADR) | Future | post-v2 |
| `enable_mcp` default flip from `false` → `auto` | Future | post all transports stable |

## What lands next (v2 roadmap)

T5-2b v1 (this PR) shipped the stdio transport + dispatcher +
two tools. v2 covers:

- **mongoose vendoring** + SSE transport body (`vmaf_mcp_start_sse`).
- **UDS transport body** (`vmaf_mcp_start_uds`).
- **LSP-framed stdio** (`Content-Length:` headers) — v1 ships
  line-delimited JSON-RPC only.
- **`compute_vmaf` binding to the real scoring engine** —
  currently validates inputs and returns a deferred-to-v2 marker.
- **SPSC ring drain at frame boundaries** — v1 dispatcher runs
  to completion on the transport thread; the measurement-thread
  hot path is not yet bridged. Tools that mutate measurement
  state (`request_model_swap`, etc.) require this bridge first.
- Tool-surface expansion (`vmaf.status`, `vmaf.score`,
  `vmaf.request_model_swap`) follows in its own PR per
  [ADR-0128](../adr/0128-embedded-mcp-in-libvmaf.md) § "Tool surface (v1)".

## Sample request / response

Request (newline-delimited JSON-RPC over stdio):

```json
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

Response:

```json
{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"list_features","description":"…","inputSchema":{…}},{"name":"compute_vmaf","description":"…","inputSchema":{…}}]}}
```

Tool call:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_features","arguments":{}}}
```

Tool response (MCP `content` envelope wrapping the tool's JSON):

```json
{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\"features\":[\"float_adm\",\"float_vif\",…],\"count\":15}"}],"isError":false}}
```

## See also

- [`docs/mcp/index.md`](index.md) — overview of both MCP surfaces.
- [`docs/mcp/tools.md`](tools.md) — tool-surface reference for
  the existing standalone Python MCP server.
- [ADR-0128](../adr/0128-embedded-mcp-in-libvmaf.md) — governance.
- [Research-0005](../research/0005-embedded-mcp-transport.md) — design.
- [`libvmaf/include/libvmaf/libvmaf_mcp.h`](../../libvmaf/include/libvmaf/libvmaf_mcp.h) — API reference.
