# Agent notes — `.github/` (workflows + templates)

This directory holds GitHub-facing config: Actions workflows, issue /
PR templates, the CODEOWNERS file. Everything here is fork-local —
Netflix/vmaf upstream has its own `.github/` that rarely overlaps
path-wise, so conflicts on merge tend to be rare but high-impact
when they happen (a silently-broken workflow is less visible than a
broken `.c` file).

## Invariants a reviewer or sync must preserve

### Rule-enforcement split (ADR-0124)

[`rule-enforcement.yml`](workflows/rule-enforcement.yml) has three
jobs. Only **one** is allowed to be required-status-check-blocking:

- `deep-dive-checklist` — **blocking**. Predicate is mechanically
  decidable (ticked checkboxes + referenced files in diff).
- `doc-substance-check` — **advisory** (`continue-on-error: true`).
  Predicate needs "is this a pure refactor?" judgement.
- `adr-backfill-check` — **advisory** (`continue-on-error: true`).
  Predicate needs "is this decision non-trivial?" judgement.

The advisory/blocking split is load-bearing — see
[ADR-0124](../docs/adr/0124-automated-rule-enforcement.md) §Consequences
and the VIF-fix false-positive in
[Research-0002](../docs/research/0002-automated-rule-enforcement.md)
§"Dead ends". Moving either advisory job into `required_status_checks`
(or flipping its `continue-on-error` flag) is a policy change and
needs a superseding ADR.

### Opt-out syntax parser

The `deep-dive-checklist` job parses PR bodies for ADR-0108's
opt-out lines:

```text
no digest needed: <reason>
no alternatives: <reason>
no rebase-sensitive invariants
no reproducer needed: <reason>
no changelog needed: <reason>
no rebase impact: <reason>
```

Regex is intentionally loose on wording. If
[`PULL_REQUEST_TEMPLATE.md`](PULL_REQUEST_TEMPLATE.md) ever renames
the six deliverables, the parser's `key` mapping in
`rule-enforcement.yml` (step "Parse six-deliverable checklist")
must move in lockstep. Search for the `case "${item}" in` block.

### Upstream-port exemption

`deep-dive-checklist` skips when the PR title starts with `port:` /
`port(scope):` or the branch name starts with `port/`. Those are
the only two knobs; a port PR that uses neither form WILL be
blocked. If the sync skill
([`.claude/skills/port-upstream-commit/`](../.claude/skills/port-upstream-commit/))
ever changes its branch-naming or title convention, update the
workflow's `Skip upstream-port PRs` step.

### Advisory surface-path lists

Both advisory jobs grep the diff for specific path prefixes
(`libvmaf/include/`, `meson_options.*`, `mcp-server/`, etc.).
These mirror
[ADR-0100](../docs/adr/0100-project-wide-doc-substance-rule.md) §Per-surface
and the ADR-policy-surface list from
[ADR-0106](../docs/adr/0106-adr-maintenance-rule.md). When either
ADR adds a new user-discoverable or policy-surface path, update the
grep patterns in `rule-enforcement.yml` in the same PR — otherwise
the advisory goes silent on the new surface.

## Upstream-merge guidance

Netflix/vmaf ships its own workflows under `.github/workflows/`
(CI, release, etc.). The fork's workflows live alongside them; file
collisions are rare because the fork-added workflow names
(`rule-enforcement.yml`, `nightly-bisect.yml`, `supply-chain.yml`,
etc.) don't clash with upstream's names. On sync:

1. Preserve every fork-added workflow verbatim unless the ADR that
   introduced it is superseded.
2. For workflows that exist in both trees (e.g. `codeql.yml`),
   prefer the fork version — it usually has stricter pins and
   broader matrix legs.
3. `PULL_REQUEST_TEMPLATE.md` is fork-authored; upstream has none.
   Never overwrite it on sync.

## OSSF Scorecard pin invariant

`.github/workflows/scorecard.yml` references
`github/codeql-action/upload-sarif@<sha>`. The Scorecard webapp at
`api.scorecard.dev` validates the pinned SHA against the action's
upstream repository on every publish; a SHA that no longer exists
under the declared tag (e.g. because upstream rewrote a release
branch or moved a tag) is rejected as an "imposter commit", returning
HTTP 400 and turning the workflow red. Whenever this pin is updated
(Dependabot or manual), spot-check that the new SHA still resolves:

```bash
pin=$(grep -oE 'codeql-action/upload-sarif@[a-f0-9]{40}' \
      .github/workflows/scorecard.yml | head -1 | cut -d@ -f2)
gh api "/repos/github/codeql-action/commits/$pin" --jq '.sha'
```

A 422 response here is the canary that the workflow is about to start
failing on the next push. See [ADR-0263](../docs/adr/0263-ossf-scorecard-policy.md)
and [Research-0053](../docs/research/0053-ossf-scorecard-investigation.md).

## Related

- [ADR-0124](../docs/adr/0124-automated-rule-enforcement.md) — this tooling
- [ADR-0263](../docs/adr/0263-ossf-scorecard-policy.md) — OSSF Scorecard
  policy + accepted blockers
- [Research-0002](../docs/research/0002-automated-rule-enforcement.md) — investigation
- [Research-0053](../docs/research/0053-ossf-scorecard-investigation.md) —
  OSSF Scorecard per-check breakdown
- [`docs/development/automated-rule-enforcement.md`](../docs/development/automated-rule-enforcement.md)
  — user-facing explainer
- [`docs/rebase-notes.md` entry 0026](../docs/rebase-notes.md) — sync ledger
