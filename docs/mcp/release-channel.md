# MCP server release channel (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

The fork ships **two** MCP server flavours:

1. The standalone Python server under `mcp-server/vmaf-mcp/`
   ([`docs/mcp/tools.md`](tools.md)).
2. The embedded server inside `libvmaf` itself, exposed via the
   `libvmaf_mcp.h` C surface
   ([`docs/mcp/embedded.md`](embedded.md)).

[ADR-0166](../adr/0166-mcp-server-release-channel.md) governs the
release channel for the standalone server: the package is
published to PyPI under the `vmaf-mcp` distribution name, follows
the same `v3.x.y-lusoris.N` versioning convention as the libvmaf
release line, and is signed via the same Sigstore-keyless flow
configured in `.github/workflows/release.yml`. Embedded-server
versioning rides the libvmaf SOVERSION; the two stay in lock-step
to avoid a "PyPI says 1.4.0, embedded says 1.3.9" client confusion.

Status: Accepted.

## See also

- [`docs/mcp/tools.md`](tools.md) — the MCP tool surface served by
  both flavours.
- [`docs/mcp/embedded.md`](embedded.md) — the embedded-server build
  flag and transport matrix.
- [`docs/development/release.md`](../development/release.md) — the
  shared release / signing pipeline.
- [ADR-0166](../adr/0166-mcp-server-release-channel.md) — design
  decision.
