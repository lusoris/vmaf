# MCP Server Release Channel

The fork ships **two** MCP server flavours:

1. The standalone Python server under `mcp-server/vmaf-mcp/`
   ([`docs/mcp/tools.md`](tools.md)).
2. The embedded server inside `libvmaf` itself, exposed via the
   `libvmaf_mcp.h` C surface
   ([`docs/mcp/embedded.md`](embedded.md)).

[ADR-0166](../adr/0166-mcp-server-release-channel.md) governs the
standalone Python server release channel. The `vmaf-mcp`
distribution is published to PyPI from the same release flow as the
libvmaf fork, uses the fork's `v3.x.y-lusoris.N` version line, and
is signed through the same keyless Sigstore/OIDC pipeline. The
embedded server ships inside libvmaf, so its compatibility version
is the libvmaf release and SOVERSION rather than a separate Python
package version.

Operators should install the standalone server from PyPI when an
agent needs a child-process tool surface:

```bash
pip install vmaf-mcp
```

For local development from a checkout:

```bash
cd mcp-server/vmaf-mcp
pip install -e .
```

Set `VMAF_BIN=/abs/path/to/vmaf` when the built CLI is not at the
repo-default `build/tools/vmaf`, and set `VMAF_MCP_ALLOW` to any
additional corpus roots the server is allowed to read.

Embedded-MCP users do not install `vmaf-mcp`; they build libvmaf
with `-Denable_mcp=true` and the needed transport flags, then call
the `libvmaf_mcp.h` C API from the host process.

## See also

- [`docs/mcp/tools.md`](tools.md) — the MCP tool surface served by
  both flavours.
- [`docs/mcp/embedded.md`](embedded.md) — the embedded-server build
  flag and transport matrix.
- [`docs/development/release.md`](../development/release.md) — the
  shared release / signing pipeline.
- [ADR-0166](../adr/0166-mcp-server-release-channel.md) — design
  decision.
