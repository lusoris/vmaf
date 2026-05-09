# Automated rule enforcement

How the fork mechanically enforces four of its process ADRs so that
reviewers can focus on substance instead of checklist policing.
Tracked by [ADR-0124](../adr/0124-automated-rule-enforcement.md);
supporting research in
[`docs/research/0002-automated-rule-enforcement.md`](../research/0002-automated-rule-enforcement.md).

Per [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md), CI
surfaces that contributors interact with ship human-readable
documentation in the same PR as the code — this page is that doc.

## What is enforced

| ADR | Rule | Enforcement | Where it runs |
| --- | --- | --- | --- |
| [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md) | Six deep-dive deliverables per fork-local PR | **Blocking** CI check | `.github/workflows/rule-enforcement.yml` job `deep-dive-checklist` |
| [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) | Docs ship with every user-discoverable surface change | Advisory CI comment | Same workflow, job `doc-substance-check` |
| [ADR-0106](../adr/0106-adr-maintenance-rule.md) | One ADR per non-trivial decision, written first | Advisory CI comment | Same workflow, job `adr-backfill-check` |
| [ADR-0105](../adr/0105-copyright-handling-dual-notice.md) | Every C / C++ / CUDA source ships a copyright header | Pre-commit hook | `scripts/ci/check-copyright.sh` via `.pre-commit-config.yaml` |

Blocking vs advisory is deliberate. ADR-0108 is the only rule whose
full predicate is mechanically decidable (a checkbox is either ticked
or it isn't, referenced files either appear in the diff or they
don't). The other rules involve human judgement — "is this a pure
refactor?", "is this decision non-trivial enough to warrant an ADR?"
— so their checks post comments instead of blocking the merge queue.

## ADR-0108: deep-dive deliverables (blocking)

The PR template
([`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md))
carries a six-item checklist under the `## Deep-dive deliverables`
heading. The workflow parses the PR body and, for each item, expects
either a ticked box or an opt-out line.

### What the checker accepts

A **ticked box** mentioning the item:

```markdown
- [x] Research digest under docs/research/ (or "no digest needed: trivial")
- [x] CHANGELOG.md "lusoris fork" entry
```

An **opt-out line** using the ADR-0108 opt-out syntax:

```markdown
- no digest needed: trivial
- no alternatives: only-one-way fix
- no rebase impact: workflow-only change
```

The parser is intentionally loose on surrounding punctuation because
reviewers sometimes reword the labels. What it checks:

- `- [x]` or `- [ ]` for the six labels: Research digest, Decision
  matrix, AGENTS.md invariant note, Reproducer / smoke-test command,
  CHANGELOG.md, Rebase note.
- `no <keyword> needed` / `no <keyword> impact` / `no
  rebase-sensitive` where `<keyword>` matches a shorthand for the
  item (`digest`, `alternatives`, `rebase`, `reproducer`, `smoke`,
  `changelog`, `AGENTS`).

### What triggers a hard fail

- A checkbox neither ticked nor opted-out — the job prints
  `::error title=ADR-0108 missing deliverable::<item> is neither
  ticked nor opted-out in the PR description.` and exits non-zero.
- A ticked "Research digest" box without a matching
  `docs/research/NNNN-*.md` in the PR diff.
- A ticked "CHANGELOG" box without `CHANGELOG.md` in the PR diff.
- A ticked "Rebase note" box without `docs/rebase-notes.md` in the
  PR diff.

### Upstream-port exemption

PRs that verbatim-port a Netflix commit are exempt: pure syncs have
no fork-local design choices to record. The workflow skips the check
when:

- The PR title starts with a Conventional Commit `port:` /
  `port(scope):` prefix, **or**
- The branch name starts with `port/`.

If neither applies, the gate runs.

### Fixing a failing check

1. Open the PR description in the GitHub UI.
2. Tick the missing checkbox, or replace the line with `no <item>
   needed: <reason>` explaining why the deliverable is absent.
3. If a referenced file is missing (research digest, CHANGELOG, or
   rebase note), add it to the branch and push — GitHub re-runs the
   workflow automatically.

The workflow re-runs on every `edited` / `synchronize` event, so
editing the description alone is enough when the referenced files
are already in the diff.

## ADR-0100: doc-substance (advisory)

Runs on every PR, never fails the check (`continue-on-error: true`).
Scans the diff for paths matching the user-discoverable surface list
from ADR-0100:

- `libvmaf/include/` — public C API
- `libvmaf/src/feature/feature_*.c` and `integer_*.c` — extractors
- `libvmaf/tools/` — CLI binaries
- `meson_options.txt` / `meson_options.toml` — build flags
- `mcp-server/` — MCP JSON-RPC surface
- `ai/src/vmaf_train/cli` — training CLI
- `ffmpeg-patches/` — ffmpeg filter surface

If any of these changed **and** nothing under `docs/` was touched in
the same PR, the job logs a ADR-0100 advisory note with the offending
paths. The rule has a first-class exemption for pure internal
refactors and bug fixes with no user-visible delta — those don't need
docs and the advisory can be ignored with a one-line reviewer ack.

## ADR-0106: ADR backfill (advisory)

Also advisory. Flags PRs that touch policy or public-surface paths
without adding a new `docs/adr/NNNN-*.md`:

- `libvmaf/include/`
- `meson_options.{txt,toml}`
- `.github/` (any workflow change)
- `docs/principles.md`
- `CLAUDE.md` / `AGENTS.md`
- `.pre-commit-config.yaml`

Bug fixes and refactors in these paths are legitimately ADR-free, so
this stays advisory — the reviewer decides whether a new ADR should
have been written.

## ADR-0105: copyright header (pre-commit)

Runs as a pre-commit hook, not CI. Every `*.c` / `*.h` / `*.cpp` /
`*.cxx` / `*.cc` / `*.hpp` / `*.hxx` / `*.cu` / `*.cuh` staged for
commit must have a `Copyright` line in its first 40 lines.

### What the hook checks

Pure presence. Template correctness — which of ADR-0105's three
templates (Netflix-only, Lusoris+Claude-only, dual notice) is the
right fit for a given file — remains a reviewer judgement. The year
range and the fork-authored-vs-upstream-modified split cannot be
derived from a diff alone, so the hook checks the cheapest
mechanically-decidable property.

### Exclusions

- `subprojects/` — vendored upstream trees
- `libvmaf/test/data/` — binary fixtures
- `python/vmaf/resource/` and `python/test/resource/` — upstream
  Netflix training-harness assets that predate the fork
- `*config.h.in` — Meson-templated headers
- `*generated*` — code-generator output

### Bypassing the hook

Don't. The global rule (`/home/kilian/.claude/CLAUDE.md`) forbids
`--no-verify`. If you hit a legitimate case that the hook misclassifies,
add an explicit exclude to [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)
in the same PR and cite the reason.

## Running the checks locally

The CI workflow mirrors scripts you can run by hand:

```bash
# Copyright hook on staged files
pre-commit run check-copyright --files path/to/file1.c path/to/file2.h

# All pre-commit hooks on staged files
pre-commit run --files $(git diff --cached --name-only)

# All pre-commit hooks against a PR's changed files
pre-commit run --from-ref origin/master --to-ref HEAD
```

### CI-parity hooks (pre-push)

The `.pre-commit-config.yaml` `local` block carries four hooks that
mirror CI lint gates so contributors catch cheap mistakes before a
CI round-trip:

| Hook | Stage | What it checks |
| --- | --- | --- |
| `assertion-density` | pre-push | NASA Power-of-10 §5 — every fork-added C function ≥20 lines has ≥1 `assert()`. Backed by `scripts/ci/assertion-density.sh`. |
| `mypy-local` | pre-push | `mypy ai/ scripts/` — same invocation as the `Python Lint` CI job. Requires `pip install mypy` (system tool, not in `pyproject.toml`). |
| `semgrep-local` | pre-commit | Project-local rules from `.semgrep.yml` (`--error` exit code on match). Standard rule packs (`p/cert-c-strict`, `p/cwe-top-25`) still run in CI only. |
| `ffmpeg-patches-apply-check` | pre-push | `git apply --check` every patch in `ffmpeg-patches/series.txt` against a cached FFmpeg `release/8.1` checkout under `/tmp/ffmpeg-n81`. Backed by `scripts/ci/ffmpeg-patches-check.sh`. |

Install the pre-push hook (one-time, fresh clones):

```bash
pre-commit install --install-hooks \
  --hook-type pre-commit \
  --hook-type pre-push \
  --hook-type commit-msg
```

The ffmpeg-patches gate degrades gracefully when offline: if it
cannot clone or fetch FFmpeg, it prints a stderr warning and exits
0 rather than blocking a local push on connectivity.

The deep-dive-checklist, doc-substance, and adr-backfill jobs run
purely against `git diff --name-only <base>..<head>` and the PR
body, so you can simulate them with `gh pr view --json body` +
`git diff --name-only` if you're curious whether a WIP PR would
pass.

### Stale code-scanning configuration: `security.yml:semgrep`

GitHub's **Settings → Code security → Code scanning → Tools → Semgrep
OSS** page shows a stale configuration pinned to
`.github/workflows/security.yml:semgrep` with a "workflow file no
longer exists" warning. The workflow was renamed `security.yml →
security-scans.yml` in PR #53 (ADR-0116, 2026-04-21 Title-Case
sweep). The current workflow uploads SARIFs under
`.github/workflows/security-scans.yml:semgrep` with categories
`semgrep-local` + `semgrep-registry`, so security scanning works
end-to-end — only the orphan tool registration lingers.

There is **no public REST endpoint** to delete a code-scanning tool
configuration (only individual analyses via
`DELETE /repos/{owner}/{repo}/code-scanning/analyses/{id}`), and the
original 2026-04-21 analyses have already rolled off the API window.
Cleanup is **manual**: open the Semgrep OSS Tools page and click the
`…` menu in the upper-right → **Delete configuration**. After that
the warning is gone permanently. Do not re-add a `security.yml`
shim — it would introduce a duplicate workflow registration.

## Why this design

- **Single workflow file, three jobs.** All four gates share the
  same trigger (`on: pull_request`), runner image
  (`ubuntu-latest`), and toolchain (`grep`, `git`). Four separate
  workflow files would duplicate boilerplate for no mental-model
  gain. See the research digest
  ([`docs/research/0002-automated-rule-enforcement.md`](../research/0002-automated-rule-enforcement.md))
  for the alternatives considered.
- **Plain bash, not `danger.js`.** CI is C / Python / meson / bash
  today. Adding a Node runtime purely for PR-body parsing would
  widen the supply-chain surface for no functional win.
- **Advisory-by-default when the rule has human judgement.** An
  earlier draft tried to block `doc-substance-check`; it would have
  blocked the VIF init leak fix on PR #47, which was a legitimate
  no-docs bug fix. Blocking rules need decidable predicates.

## Related

- [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) — doc-substance
- [ADR-0105](../adr/0105-copyright-handling-dual-notice.md) — copyright
- [ADR-0106](../adr/0106-adr-maintenance-rule.md) — ADR backfill
- [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md) — six deliverables
- [ADR-0124](../adr/0124-automated-rule-enforcement.md) — this tooling
- [Research-0002](../research/0002-automated-rule-enforcement.md) — supporting investigation
