# Dependency-update bot — operator playbook

This page describes how the fork's automated dependency-update bot (Mend Renovate)
is configured, how to provision it, and how to switch back to Dependabot if needed.
See [ADR-0363](../adr/0363-renovate-replaces-dependabot.md) for the decision record.

## Overview

The fork uses **Mend Renovate self-hosted** via a GitHub Actions workflow
(`.github/workflows/renovate.yml`). Renovate runs every Monday before 06:00
Vienna time (Europe/Vienna, cron `17 4 * * 1`) and opens grouped pull requests
for:

| Manager | What it tracks | Group name |
|---------|---------------|------------|
| `github-actions` (minor + patch) | SHA-pinned actions across `.github/workflows/*.yml` | GitHub Actions (minor + patch) |
| `github-actions` (major) | Major-version bumps to pinned actions | GitHub Actions (major) |
| `pre-commit` | Hook `rev:` values in `.pre-commit-config.yaml` | pre-commit hooks |
| `pep621` / `pip_requirements` (patch) | Python patch releases in `pyproject.toml` + `requirements*.txt` | Python (patch) |
| Custom regex | `FFMPEG_SHA:=n…` in `ffmpeg-patches/test/build-and-run.sh` and `FFMPEG_PATCHES_BRANCH` in `scripts/ci/ffmpeg-patches-check.sh` against `github-tags/FFmpeg/FFmpeg` | (individual PRs) |

Renovate supersedes Dependabot. The old Dependabot configuration is retained as
`.github/dependabot.yml.disabled` in case a rollback is ever needed.

## Prerequisites

### `RENOVATE_TOKEN` secret

Renovate needs a GitHub token to open PRs and update issues. The recommended
approach is a **fine-grained personal access token** scoped to `lusoris/vmaf`:

| Permission | Access level |
|-----------|-------------|
| Contents | Read and write |
| Pull requests | Read and write |
| Issues | Write |

To provision:

```bash
# Generate a fine-grained PAT at https://github.com/settings/personal-access-tokens
# then set it as a repository secret:
gh secret set RENOVATE_TOKEN --repo lusoris/vmaf -b "<token>"
```

Alternatively, if the `renovatebot/github-action` version in use supports it,
`GITHUB_TOKEN` with the permissions declared in the workflow (`contents: read`,
`pull-requests: write`, `issues: write`) may be sufficient for repositories
where the token's default permissions allow branch creation. Check the
[Renovate self-hosted authentication docs](https://docs.renovatebot.com/self-hosting/#githubapp-authentication)
before switching to `GITHUB_TOKEN` — the behaviour differs between GitHub App
tokens and PATs for cross-repo lookups.

## Running Renovate manually

Use `workflow_dispatch` to trigger a run outside the weekly schedule:

```bash
gh workflow run renovate.yml --repo lusoris/vmaf
```

Or navigate to **Actions → Renovate → Run workflow** in the GitHub UI.

## Configuration reference

The configuration lives in [`renovate.json`](../../renovate.json) at the
repository root. Key choices:

- **`config:recommended`** — Renovate's curated base preset; enables major
  version grouping, changelog lookups, and semantic-release compatibility.
- **`:dependencyDashboard`** — creates a `Dependency Dashboard` issue in the
  repository that tracks open and pending updates at a glance.
- **`:semanticCommits`** — commit messages follow Conventional Commits
  (`chore(deps): …`), matching the fork's commit-msg hook.
- **`helpers:disableTypesNodeMajor`** — suppresses `@types/node` major-version
  noise (not applicable to this repo's Node usage, but harmless).
- **`prHourlyLimit: 2` / `prConcurrentLimit: 6`** — rate limits to avoid
  flooding the merge queue.
- **`rangeStrategy: bump`** — for version-range deps, bumps the lower bound
  rather than widening the range.

## SHA-pin invariant

Every `uses:` directive produced by Renovate's `github-actions` manager arrives
as a 40-character commit SHA with the semver tag in a trailing comment, e.g.:

```yaml
uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
```

This is the format required by the OSSF Scorecard `Pinned-Dependencies` check
(ADR-0263) and enforced by the sync gate in `.github/AGENTS.md`. Renovate PRs
will automatically satisfy this invariant — do not manually replace SHAs with
floating tags.

## Rollback to Dependabot

If Renovate needs to be disabled:

1. Delete or disable `.github/workflows/renovate.yml`.
2. Rename `.github/dependabot.yml.disabled` back to `.github/dependabot.yml`.
3. Remove the `RENOVATE_TOKEN` secret (optional cleanup).
4. Close the Renovate Dependency Dashboard issue if it exists.

Do not run both bots simultaneously — they will open duplicate PRs for the same
packages.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Workflow fails with `token not found` | `RENOVATE_TOKEN` secret missing or expired | Re-provision the secret (see above) |
| No PRs opened after first run | `renovate.json` parse error or no eligible updates | Check the workflow logs for `WARN` / `ERROR` lines from Renovate |
| Renovate opens PRs outside the schedule | `workflow_dispatch` was triggered manually | Expected; cancel if unintended |
| Dependency Dashboard issue not created | First run still in progress, or `:dependencyDashboard` preset overridden | Wait for run to complete; verify `renovate.json` extends `:dependencyDashboard` |
| FFmpeg SHA not updated | `customManagers` regex did not match | Verify the shell script still uses the `FFMPEG_SHA:=n…` pattern; update the regex in `renovate.json` if the format changed |

## Related

- [ADR-0363](../adr/0363-renovate-replaces-dependabot.md) — decision record
- [ADR-0263](../adr/0263-ossf-scorecard-policy.md) — OSSF Scorecard policy (SHA-pin requirement)
- [`.github/AGENTS.md`](../../.github/AGENTS.md) — SHA-pin invariant + resolution recipe
- [`renovate.json`](../../renovate.json) — full configuration
- [Mend Renovate self-hosted docs](https://docs.renovatebot.com/self-hosting/)
