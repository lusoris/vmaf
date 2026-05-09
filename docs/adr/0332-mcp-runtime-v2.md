# ADR-0332: MCP runtime v2 — UDS transport + real `compute_vmaf` binding

- **Status**: Accepted
- **Date**: 2026-05-09
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: mcp, agents, api, transport, fork-local

## Context

[ADR-0209](0209-mcp-embedded-scaffold.md) shipped the embedded
MCP server as audit-first scaffolding (`-ENOSYS` on every entry
point). PR #490 (T5-2b) flipped `vmaf_mcp_init`,
`vmaf_mcp_start_stdio`, `vmaf_mcp_stop`, `vmaf_mcp_close` plus a
JSON-RPC 2.0 dispatcher with two tools: `list_features` (real)
and `compute_vmaf` (placeholder returning
`{"status":"deferred_to_v2"}`). Two pieces remained deferred:

- **UDS transport** — needed for embedded scenarios where a host
  process drives libvmaf out-of-band over a local filesystem
  socket (mode 0700, owner-only).
- **`compute_vmaf` real implementation** — the placeholder was
  visibly half-finished; AI hosts that called it through the MCP
  surface got no useful output.

A third piece (SSE / loopback HTTP) was originally scoped here as
well but deferred to v3 — see *Alternatives considered*.

## Decision

We will land MCP runtime v2 with two additions on top of PR #490:

1. **UDS transport** in `libvmaf/src/mcp/transport_uds.c`. Standard
   POSIX `socket(AF_UNIX, SOCK_STREAM, 0)` + `bind` + `listen` +
   `accept` loop; per-connection serial dispatch through the
   existing `dispatcher.c`; line-delimited JSON-RPC framing
   identical to the stdio transport. The socket file is created
   with `chmod 0700` after `bind` per ADR-0128 § "Operational
   guardrails".
2. **`compute_vmaf` real binding** in `libvmaf/src/mcp/compute_vmaf.c`.
   Per-call ephemeral `VmafContext` (so the host's main scoring
   run is not perturbed), `vmaf_model_load` of the requested
   `model_version` (default `vmaf_v0.6.1`), POSIX YUV reader for
   YUV420p 8-bit pairs, `vmaf_read_pictures` per frame, then
   `vmaf_score_pooled` with `VMAF_POOL_METHOD_MEAN`. Returns
   `{score, frames_scored, model_version, pool_method}`.

The smoke test (`libvmaf/test/test_mcp_smoke.c`) is extended with
a UDS round-trip and a `compute_vmaf` real-score check against the
testdata 576x324 48-frame YUV pair. The pinned-`-ENOSYS`
expectation for UDS is dropped; SSE remains pinned at `-ENOSYS`
until v3.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Land all three (UDS + SSE + compute_vmaf) in one PR | Single review pass | Mongoose vendoring (50 KB+) needs its own due-diligence pass; would balloon the diff and risk no-guessing rule on SSE-spec wording | Deferred SSE to v3 — focused PR ships clean |
| Reuse the borrowed `server->ctx` for `compute_vmaf` | No per-call allocation | `vmaf_score_pooled` commits the model to the context; out-of-band MCP scoring would corrupt the host's main run | Per-call ephemeral `VmafContext` preserves the contract |
| Drop the file-size sanity check (size%bytes_per_frame == 0) | Simpler reader | Silently truncates frames or scores garbage when the host passes wrong width/height | Reject up front with `-EINVAL` and a structured error message |
| Per-client UDS thread (concurrent clients) | Higher throughput under load | Embedded use case is single-host-driver; thread-pool adds bug surface | Single-client serial accept; v3 may revisit |

## Consequences

- **Positive**: AI hosts driving libvmaf via MCP get a working
  `compute_vmaf` tool; embedded hosts get a UDS surface that
  doesn't claim the host's stdin/stdout.
- **Negative**: `compute_vmaf` allocates a per-call `VmafContext`
  (small, but non-zero); hosts that need batched scoring should
  use the libvmaf CLI, not the MCP tool.
- **Neutral / follow-ups**: SSE / mongoose vendoring still owed
  (v3); 10/12-bit YUV support still owed (v3 widens
  `compute_vmaf`); per-client UDS threading is on the v3 wish-list.

## References

- [ADR-0128](0128-embedded-mcp-in-libvmaf.md) — original embedded-MCP design.
- [ADR-0209](0209-mcp-embedded-scaffold.md) — audit-first scaffold.
- PR #490 — T5-2b stdio runtime + `compute_vmaf` placeholder.
- [Research-0005](../research/0005-embedded-mcp-transport.md) — transport sequencing.
- AF_UNIX(7) Linux man page (accessed 2026-05-09):
  https://man7.org/linux/man-pages/man7/unix.7.html
- Source: req — task brief for "MCP runtime v2 — three additions
  to PR #490's stdio-only v1" (paraphrased: ship UDS + SSE +
  real `compute_vmaf`; trim aggressively if needed in priority
  order compute_vmaf > UDS > SSE).
