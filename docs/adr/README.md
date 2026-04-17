# Architectural Decision Records (ADR)

This is the **canonical, tracked** decision log for the fork. Every non-trivial
architectural / policy / scope decision lands here before the corresponding
commit merges.

## Files

- [decisions-log.md](decisions-log.md) — the table of decisions (D1, D2, …) + rationale notes
- [questions-answered.md](questions-answered.md) — verbatim Q&A from popup rounds that produced those decisions

## Why it exists

Per [CLAUDE.md §12 rule 8–9](../../CLAUDE.md) — an ADR-row-before-commit
discipline. The risk this mitigates: a session makes a decision (directory
move, CI gate change, dependency swap), commits it, the session ends, and the
rationale is only recoverable from the commit message — which is fine until
the message omits the alternatives considered.

## What counts as non-trivial?

Another engineer could reasonably have chosen differently. Examples:

- Directory moves (e.g., D26: `workspace/` → `python/vmaf/workspace/`)
- Base-image / dependency policy (e.g., D27: non-conservative CUDA pins)
- CI-gate semantics (e.g., D24: Netflix golden tests as required status)
- Test selection / regeneration rules
- Coding-standards changes
- New user-visible flags or surfaces (e.g., D23: four tiny-AI surfaces)

**Not** ADR-worthy: bug fixes, implementation details, one-off refactors
that don't change any interface or policy.

## Format

Each row: `| ID | Phase | Decision | Source |` where `Source` cites either
`req` (direct user quote) or `Q<round>.<q>` (a popup answer in
[questions-answered.md](questions-answered.md)).

Non-trivial decisions also get a rationale paragraph below the table
explaining why this option, not its alternatives.

## Relation to `.workingdir2/`

The same file also exists under `.workingdir2/decisions/` as part of the
gitignored local planning dossier. `docs/adr/` is the source of truth that
ships with the repo; the `.workingdir2/` copy is kept in sync for session
continuity but **not authoritative**.
