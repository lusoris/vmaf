# ADR-0452: cJSON vendored copy — banned-function remediation

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `security`, `mcp`, `vendored`, `lint`, `fork-local`

## Context

A memory-safety audit of vendored third-party code under `libvmaf/src/mcp/3rdparty/`
identified that `cJSON.c` (pinned at v1.7.18) contained 11 call sites using
functions banned by [docs/principles.md](../principles.md) §1.2 rule 30:

- `sprintf` at lines 125, 515, 517, 520, 526, 904
- `strcpy` at lines 383, 827, 1254, 1262, 1270

The "vendored is in scope" rule applies: the fork's lint profile covers all files
in the tree, including vendored code. NOLINT annotations are not a valid resolution
(CLAUDE.md §12 rule 12 — a NOLINT without a load-bearing-invariant justification
is itself a lint violation).

## Decision

**Path B — per-call replacement** was chosen over Path A (sync to upstream cJSON head).

Each call site was replaced with the narrowest safe equivalent:

| Old call | Replacement | Rationale |
|---|---|---|
| `sprintf(buf, fmt, …)` into stack buffer | `snprintf(buf, sizeof(buf), fmt, …)` | Bounds are statically known; `sizeof` captures the array size exactly. |
| `sprintf(out_ptr, "u%04x", ch)` into pre-allocated printer buffer | `(void)snprintf(out_ptr, 6, fmt, ch)` | Buffer space is guaranteed by the printer's `ensure()` contract (5 payload bytes + 1 NUL). |
| `strcpy(dst, src)` where `strlen(src) <= strlen(dst)` | `memcpy(dst, src, strlen(src) + 1)` | The prior length check makes the copy unconditionally safe; `memcpy` avoids a second strlen scan and sidesteps the banned-function check. |
| `strcpy(dst, literal)` into `ensure()`-backed buffer | `memcpy(dst, literal, sizeof(literal))` | `sizeof` on a string literal includes the NUL; `ensure()` allocates exactly that many bytes. |

Return values of `snprintf` used as length sources are kept (they feed the
`length < 0 || length > buf - 1` guard that already existed). The `(void)` cast
is applied only where the return value is intentionally discarded (version string
formatting and the unicode escape write where the bound cannot overflow).

## Alternatives considered

**Path A — sync to upstream cJSON HEAD** was evaluated. The upstream repository
(`github.com/DaveGamble/cJSON`) is at v1.7.18 (same version), and the upstream
`cJSON.c` as of 2026-05-16 still contains `sprintf`/`strcpy` at the same call
sites — they have not been cleaned up upstream. A sync would produce zero diff on
the banned-function front and would introduce other upstream changes (unreviewed
by this fork). Rejected.

**Path C — NOLINT suppression** was not considered, per CLAUDE.md §12 rule 12 and
the `AGENTS.md` invariant added in this PR.

**Path D — replace cJSON with a different JSON library** (e.g. `jsmn`, `yajl`, or
the cJSON fork maintained by ibireme) was considered but judged disproportionate.
The MCP server's JSON surface is small and the per-call fix is mechanical. Deferred
unless a second audit round finds deeper structural issues.

## Consequences

- No banned functions remain in `libvmaf/src/mcp/3rdparty/cJSON/cJSON.c`.
- The file now diverges from upstream cJSON v1.7.18 in 11 call sites.
- `libvmaf/src/mcp/3rdparty/cJSON/AGENTS.md` documents the "fix or sync, never NOLINT" policy.
- Future upstream syncs must re-verify the banned-function gate after overwriting `cJSON.c`.
- No public C API change; no FFmpeg patch impact.

## References

- req: "vendored is in scope — banned functions are NOT covered by NOLINT — fix or sync to a clean upstream version"
- [docs/principles.md](../principles.md) §1.2 rule 30 — banned function list
- [CLAUDE.md §12 rule 12](../../CLAUDE.md) — touched-file cleanup rule
- [ADR-0141](0141-touched-file-cleanup-rule.md) — touched-file cleanup rule
- [ADR-0278](0278-nolint-citation-closeout.md) — NOLINT citation requirement
