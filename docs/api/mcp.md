# Embedded MCP server — `libvmaf_mcp.h`

The fork ships an in-process MCP (Model Context Protocol) server that an
embedding host (e.g. an editor plugin or a measurement-orchestration
daemon) can drive over loopback HTTP, a Unix-domain socket, or a
caller-owned stdio fd pair. The C surface is defined in
[`libvmaf/include/libvmaf/libvmaf_mcp.h`](../../libvmaf/include/libvmaf/libvmaf_mcp.h).

This page covers the **embedded C API**. The standalone Python MCP
server under [`mcp-server/vmaf-mcp/`](../../mcp-server/vmaf-mcp/) is a
separate surface — see [`docs/mcp/`](../mcp/) for that one.

## Status

Audit-first scaffold per
[ADR-0209](../adr/0209-mcp-embedded-scaffold.md). The header surface is
stable for downstream consumers, but every entry point currently returns
`-ENOSYS` until the runtime PR (T5-2b — vendored cJSON + mongoose,
SPSC ring buffer, transport bodies) lands. Linking against the header
today gives a caller a predictable error path so they can fall back to
the external Python server.

When libvmaf is built without `-Denable_mcp=true`, every entry point
also returns `-ENOSYS` unconditionally; the symbols are resolved against
the same stub TU at `libvmaf/src/mcp/mcp.c`.

## Build flag

```bash
meson setup build -Denable_mcp=true \
                  -Denable_mcp_sse=true   `# loopback HTTP / SSE`     \
                  -Denable_mcp_uds=true   `# AF_UNIX, JSON-RPC`       \
                  -Denable_mcp_stdio=true `# LSP framing on host fds`
```

The umbrella flag (`enable_mcp`) compiles in the API surface; each
transport sub-flag adds the corresponding wire driver. Sub-flags
default to off so a minimal MCP build can ship with a single transport.

## Public surface (one-liner reference)

| Symbol                            | Returns         | Purpose                                                         |
|-----------------------------------|-----------------|-----------------------------------------------------------------|
| `vmaf_mcp_available()`            | `int` (0/1)     | Built with `-Denable_mcp=true`?                                 |
| `vmaf_mcp_transport_available(t)` | `int` (0/1)     | Built with the named transport sub-flag?                        |
| `vmaf_mcp_init(out, ctx, cfg)`    | `0 / -errno`    | Allocate a server handle bound to a `VmafContext`.              |
| `vmaf_mcp_start_sse(s, cfg)`      | `0 / -errno`    | Bind a loopback HTTP listener; spawn the SSE pthread.           |
| `vmaf_mcp_start_uds(s, cfg)`      | `0 / -errno`    | Bind an AF_UNIX listener at the configured path (mode 0700).    |
| `vmaf_mcp_start_stdio(s, cfg)`    | `0 / -errno`    | Spawn the stdio pthread on a caller-supplied fd pair.           |
| `vmaf_mcp_stop(s)`                | `0 / -errno`    | Join every running transport thread (idempotent).               |
| `vmaf_mcp_close(out)`             | `void`          | Release the handle; sets `*out` to `NULL`.                      |

## Threading and allocation

Per [ADR-0209](../adr/0209-mcp-embedded-scaffold.md) and
Research-0005:

- `vmaf_mcp_init()` pre-allocates the SPSC ring buffer at construction
  time. **No further allocation crosses the measurement-thread boundary**
  (NASA Power-of-10 rule 3).
- The measurement thread drains at most `max_drain_per_frame`
  envelopes per frame (default 4, cap 64). Bounded loop, NASA Power-of-10
  rule 2.
- Each `_start_*` call spawns one dedicated MCP pthread. Multiple
  transports can co-exist on one server handle.

## Auth

- **SSE** binds 127.0.0.1 only — the listener refuses non-loopback
  addresses.
- **UDS** creates the socket mode 0700.
- **stdio** is trusted by construction (the host owns the fds and
  decides who else sees them).

## Example

```c
#include <libvmaf/libvmaf.h>
#include <libvmaf/libvmaf_mcp.h>

VmafContext *ctx = NULL;
VmafConfiguration cfg = { /* … */ };
vmaf_init(&ctx, cfg);

if (vmaf_mcp_available()) {
    VmafMcpServer *mcp = NULL;
    VmafMcpConfig mc = { .queue_depth = 64,
                         .max_drain_per_frame = 4,
                         .user_agent = "my-host/1.0" };
    int rc = vmaf_mcp_init(&mcp, ctx, &mc);
    if (rc == 0 && vmaf_mcp_transport_available(VMAF_MCP_TRANSPORT_UDS)) {
        VmafMcpUdsConfig uds = { .path = "/run/vmaf/mcp.sock" };
        rc = vmaf_mcp_start_uds(mcp, &uds);
    }
    /* … vmaf_read_pictures + vmaf_score_pooled as usual … */
    vmaf_mcp_close(&mcp);
}

vmaf_close(ctx);
```

`-ENOSYS` from any of the calls means "MCP not built in / runtime not
yet wired" — fall back to the external Python server for the time
being.

## Error contract

All entry points return a negative `errno` on failure (`0` on success).
The most common codes:

- `-ENOSYS` — feature not built (or scaffold not yet wired).
- `-ENODEV` — transport-specific runtime unavailable (e.g. UDS on a
  non-POSIX host).
- `-EINVAL` — bad argument (NULL where required, malformed config,
  non-power-of-two `queue_depth`).
- `-ENOMEM` — ring/buffer allocation failed at init.
- `-EBUSY` — measurement already in flight, or the named transport is
  already running on this handle.
- `-EADDRINUSE` — SSE port / UDS path already bound.

## Related

- [ADR-0209](../adr/0209-mcp-embedded-scaffold.md) — embedded-MCP
  scaffold decision.
- [`docs/mcp/embedded.md`](../mcp/embedded.md) — user-side overview of
  the embedded server.
- [`docs/mcp/`](../mcp/) — standalone Python MCP server surface.
