# ADR-0165: Tracked `docs/state.md` for bug-status hygiene (T7-1)

- **Status**: Accepted
- **Date**: 2026-04-25
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: process, state-hygiene, claude-rule, fork-local

## Context

[Issue #20](https://github.com/lusoris/vmaf/issues/20) flagged a recurring
failure mode: Claude Code agents re-investigate already-closed bugs at the
start of each session, because no in-flight bug-status record gets updated
during the session that closed them. The closing comment on Issue #20
made the fix concrete:

> What's NOT yet done and matches the exact ask in this issue: a
> `STATE.md` file + commit-as-you-close discipline for *bug status*
> (vs. architectural decisions, which D28 covers). Leaving open until
> STATE.md exists and has a rule wired in CLAUDE.md §12.

The fork already has two state-shaped surfaces:

- `docs/adr/` — **decisions** (one file per non-trivial architectural
  / policy choice). [ADR-0028](0028-adr-maintenance-rule.md) makes
  these immutable once Accepted.
- `.workingdir2/{OPEN,BACKLOG,PLAN}.md` — **planning dossier**.
  Gitignored; intentionally local-only so that mid-flight task notes
  don't leak into the repo. Not a candidate for tracked bug status.

What's missing is a **tracked, in-tree** surface that says "these bugs
are open, these are closed, here is the PR / ADR that closed each one,
and here are the bugs we ruled out as not-affecting-the-fork." Without
it, future sessions reach for `git log` (slow, lossy, not query-able)
or guess from `.workingdir2/` (gitignored, can be stale on a fresh
clone).

## Decision

Ship a single tracked file [`docs/state.md`](../state.md) with three
sections:

1. **Open bugs** — known issues actively under investigation or queued
   for fix. Each row carries: bug ID (Netflix#N or fork-local), summary,
   reproducer status, owning ADR or PR if any, target.
2. **Recently closed** — bugs closed in the last ~3 months. Each row
   carries: bug ID, summary, closing PR + ADR, verification method.
   Older entries roll off into the git log naturally.
3. **Confirmed not-affected / explicitly deferred** — the negative
   results that protect future sessions from re-investigating dead
   ends. Includes Netflix issues that don't apply to fork's code paths
   and deliberate defers (e.g. Netflix#955 ADR-0155 deferred-pending-
   upstream).

Wire the update discipline as **CLAUDE.md §12 rule 13** (next free
slot in the hard-rules block):

> 13. **Every** PR that closes a bug, opens a bug, or rules a Netflix
>     upstream report not-affecting-the-fork updates `docs/state.md`
>     in the **same PR**. The update lands a row in the appropriate
>     section (Open / Recently closed / Confirmed not-affected) and
>     cross-links the ADR, PR, and Netflix issue (if any). State drift
>     compounds across sessions; the rule trades a 30-second edit for
>     hours of re-investigation cost when context resets.

## Alternatives considered

1. **Promote `.workingdir2/OPEN.md` to tracked**. Rejected: the
   planning-dossier convention is load-bearing — `.workingdir2/`
   accumulates work-in-progress notes that are deliberately not
   shipped. Inverting that convention to track one file would either
   blur the boundary (some `.workingdir2/` files tracked, others
   not — confusing) or require migrating multiple files all at once.
   The narrow ask is bug-status, not "all planning notes."

2. **Just add the rule to CLAUDE.md without a tracked file**. Rejected:
   without a target file, the rule has no anchor — every PR description
   is a different shape and bugs scroll off page-1 of `git log`
   quickly. The whole point of Issue #20 was to give future sessions
   a single place to look.

3. **Use GitHub Issues as the bug-status surface**. Rejected for
   fork-local provenance: GitHub Issues are great for inbound triage
   but lossy when a Netflix-side PR resolves a fork-local issue
   indirectly, or when the fork *deliberately defers* (e.g. ADR-0155).
   In-tree Markdown gives the audit trail traceable cross-links to
   ADRs / PRs / commits without depending on an external service that
   could rate-limit or change UI.

## Consequences

**Positive:**
- One canonical place for "what bugs are still real?"
- Cross-linkable: ADRs and PR descriptions can link to specific
  rows in `docs/state.md` for grounding.
- Low write cost: most rows are 1–2 sentences plus links.
- Survives session resets and fresh clones.

**Negative:**
- Yet another rule to remember. Mitigated by binding it to PR-merge
  events (the same moment Conventional Commit + ADR rules already
  fire). The PR template ([.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md))
  carries a checkbox so reviewers verify it.
- Risk of drift between `docs/state.md`, `.workingdir2/BACKLOG.md`,
  and the closed-issues view on GitHub. Convention: `docs/state.md`
  is authoritative for **bug status** only; backlog items
  (T-numbers) and ADR decisions stay in their respective surfaces.
- Format will need to evolve. Section headings stay stable; row
  shape can be refactored without an ADR amendment.

## References

- Issue #20 — original ask (closed COMPLETED 2026-04-19; this ADR
  is the concrete artifact that re-opens the audit trail).
- [ADR-0028](0028-adr-maintenance-rule.md) — ADR-row-before-commit
  rule (the decision-log half of state hygiene).
- [BACKLOG T7-1](../../.workingdir2/BACKLOG.md) — backlog row.
- `req` — user direction 2026-04-25: "well then update the state files
  thats bullshit as well" → "well then lets go" (popup choice:
  tracked `docs/state.md`).
