# ADR-0334: state.md-touch-check CI gate (ADR-0165 enforcement)

- **Status**: Accepted
- **Date**: 2026-05-08
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: ci, process, state-hygiene, claude-rule, fork-local

## Context

[ADR-0165](0165-state-md-bug-tracking.md) introduced
[`docs/state.md`](../state.md) and bound the update discipline to
**CLAUDE.md §12 rule 13** — every PR that closes, opens, or rules-out
a bug must update `docs/state.md` in the same PR (or carry the explicit
opt-out `no state delta: REASON`). Until now the rule has been
**reviewer-enforced**: the
[PR template](../../.github/PULL_REQUEST_TEMPLATE.md) carries the
"Bug-status hygiene" checkbox, and reviewers were expected to verify it.

The state.md audit-backfill PR #455 (the manual sweep that retroactively
filled in months of missed rows) surfaced this as a backlog row: reviewer
attention is the wrong substrate for a mechanically-decidable predicate.
"Did the diff touch `docs/state.md`?" is a one-line `grep` against
`git diff --name-only`. "Does the PR body carry the opt-out?" is one more
`grep` against the body. The same check runs on every PR; humans miss it
intermittently. Mechanical enforcement frees reviewer attention for the
substantive parts of a PR.

The fork already has a precedent for this exact shape:
[ADR-0124](0124-automated-rule-enforcement.md) /
[`.github/workflows/rule-enforcement.yml`](../../.github/workflows/rule-enforcement.yml)
runs three jobs (deliverables, doc-substance, ADR-backfill) parsing the
PR body + diff with plain bash. ADR-0167 promoted doc-substance from
advisory to blocking once the predicate became precise enough. This ADR
follows the same trajectory for ADR-0165.

## Decision

Add a fourth job, `state-md-touch-check`, to
`.github/workflows/rule-enforcement.yml`, backed by a single-purpose
script `scripts/ci/state-md-touch-check.sh`. The script is **blocking**
(no `continue-on-error: true`), draft-PR-gated (matches the existing
jobs), and runs the same predicate locally and in CI.

**Trigger predicate** (any one suffices):

1. PR title carries a Conventional-Commit `fix:` / `fix(scope):`
   prefix.
2. PR title contains the bare token `bug` (word-boundary, so `debug`
   does not fire).
3. PR title or body contains a `closes #N` / `fixes #N` /
   `resolves #N` GitHub-issue close keyword.
4. PR body carries the `## Bug-status hygiene` template section with
   the `docs/state.md` checkbox left unchecked.

**Pass conditions** (either is enough):

1. The diff against `BASE_SHA..HEAD_SHA` includes `docs/state.md`.
2. The PR body contains `no state delta: <REASON>` where REASON is a
   non-empty token that is not the literal placeholder `REASON`. HTML
   comments are stripped before the match so the template's
   instructional `<!-- ... no state delta: REASON ... -->` doesn't
   accidentally satisfy the gate.

**Failure mode**: print `::error title=ADR-0165 docs/state.md drift::`
naming the four section choices (Open / Recently closed / Confirmed
not-affected / Deferred) and the two ways to clear the gate.

A companion `scripts/ci/test-state-md-touch-check.sh` exercises the
script against eight fixture cases (5 primary + 3 regression). Tests
are bash-only and run in a throw-away `mktemp -d` git repo so the
diff input is real, not mocked.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| **Inline the bash in `rule-enforcement.yml` (current pattern for the doc-substance job)** | Zero indirection; one file to read | Cannot dry-run locally before `gh pr create`; doubles bash-in-YAML maintenance cost | Rejected — the existing `deliverables-check.sh` precedent shows the script-with-thin-wrapper shape is worth the extra file. Local dry-run saves one CI round-trip per "I forgot the opt-out" mistake. |
| **Keep the rule reviewer-enforced** | No new CI surface | Repeated drift across months (the audit-backfill PR exists because of this) | Rejected — the predicate is mechanically decidable; relying on reviewer attention is the wrong substrate. |
| **Promote to a non-bypassable required check via branch-protection** | Strongest enforcement | Hard to revert if the heuristic over-fires; legitimate `feat:` PRs would need to learn the opt-out before merging | Deferred — start as a regular blocking workflow. If false positives are rare after one month, promote to the `required-aggregator.yml` set in a follow-up PR. |
| **Use `danger.js` / a GitHub App for body parsing** | Richer DSL | Adds a Node runtime to a repo whose CI is bash + meson + Python; widens supply-chain surface for no functional gain | Rejected — same reasoning as ADR-0124 §"Why this design". |

## Consequences

**Positive:**

- Mechanical enforcement of CLAUDE.md §12 rule 13 — no more drift
  across sessions.
- Local dry-run via `PR_TITLE=… PR_BODY=… scripts/ci/state-md-touch-check.sh`
  saves a CI round-trip.
- The eight fixture tests pin the trigger heuristic so regressions
  surface early (e.g. the `debug`-vs-`bug` distinction).
- Symmetry with `deliverables-check.sh` — a contributor who has
  internalised one gate already understands the other.

**Negative:**

- One more CI step on every PR (~15 s on `ubuntu-latest`).
- The trigger heuristic is necessarily a heuristic; a `feat:` PR
  that incidentally fixes a bug without a `fix:` prefix will not
  fire the gate (false negative). Mitigated by the
  Bug-status-hygiene checkbox in the template — copy-pasting the
  template carries the trigger-on-unchecked path.
- A `fix:` PR that legitimately has no bug-status delta (e.g.
  `fix: typo in log message`) needs the `no state delta: REASON`
  opt-out. The error message names this explicitly.

**Neutral / follow-ups:**

- After ~30 days, audit false-positive rate. If <5%, promote to
  `required-aggregator.yml`; if ≥5%, tighten the trigger predicate.
- Monitor for cases where the literal `REASON` placeholder slips
  through despite the all-caps guard — escalate to a stricter
  "lowercase prose required" regex if needed.

## References

- [ADR-0165](0165-state-md-bug-tracking.md) — original
  `docs/state.md` + rule-13 decision.
- [ADR-0124](0124-automated-rule-enforcement.md) — the workflow
  this gate joins.
- [ADR-0167](0167-doc-drift-enforcement.md) — same
  reviewer-to-CI promotion pattern.
- [`docs/development/automated-rule-enforcement.md`](../development/automated-rule-enforcement.md)
  — contributor-facing documentation, updated in this PR.
- PR #455 — state.md audit-backfill that surfaced this as a
  backlog row.
- `req` — user direction 2026-05-08: "convert CLAUDE.md §12 r13
  from reviewer-enforced to CI-enforced".
