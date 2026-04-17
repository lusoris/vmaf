# CI secrets

Secrets consumed by workflows in `.github/workflows/`. Set at the
**organization** level (preferred — one place, reusable across fork repos)
or the **repository** level.

| Secret              | Scope today                 | Consumer                              | Required | Notes                                                        |
| ------------------- | --------------------------- | ------------------------------------- | -------- | ------------------------------------------------------------ |
| `SCORECARD_TOKEN`   | org (public-repos)          | `scorecard.yml`                       | No\*\*   | Classic PAT read-only on branch-protection + workflow APIs.  |
| `SEMGREP_APP_TOKEN` | org (public-repos) — to add | `security.yml` (semgrep registry job) | No\*     | See [Semgrep Registry auth](#semgrep-registry-auth) (D39).   |
| `GITHUB_TOKEN`      | per-run, auto               | All workflows                         | Auto     | GitHub-provided, short-lived. Never set manually.            |

\* Not required for merge gating. Without it the registry job's SARIF upload
is skipped (see `Check registry SARIF is non-empty` step). CodeQL
security-and-quality still runs independently and covers CWE Top 25.

\*\* Scorecard runs with `GITHUB_TOKEN` too; the PAT just unlocks a few
extra repo-metadata checks.

## Org vs. repo secrets

The `lusoris` org already holds `SCORECARD_TOKEN` at the org level with
visibility set to **public repositories**. Because `lusoris/vmaf` is public,
it inherits org secrets without a plan upgrade (the "org secrets cannot be
used by private repositories" banner is a private-repo restriction only).

Prefer org-level for anything reusable across fork repos; prefer repo-level
for anything truly repo-scoped (e.g. a deploy key).

## Semgrep Registry auth

Unauthenticated Semgrep Registry fetches are rate-limited, producing empty
SARIFs and a false "Semgrep OSS is reporting errors" banner on the repo
Security tab. Fix: authenticate with a free Semgrep Cloud token.

1. Sign up at <https://semgrep.dev> (GitHub OAuth — no billing info required).
2. `Settings → Tokens → Create new token` with scope "Agent (CI)".
3. Copy the token value.
4. Add as an org secret (sibling to `SCORECARD_TOKEN`):

   ```bash
   gh secret set SEMGREP_APP_TOKEN --org lusoris --visibility all \
       --body "<paste-token>"
   ```

   Or, if you want it repo-scoped instead:

   ```bash
   gh secret set SEMGREP_APP_TOKEN --repo lusoris/vmaf --body "<paste-token>"
   ```

The token is read by the `semgrep` job in `security.yml` via
`env: SEMGREP_APP_TOKEN`. It is read-only against the Semgrep Registry —
no repo access is granted.

## Rotation

Rotate annually; track in a calendar reminder. `GITHUB_TOKEN` rotates
automatically per job and does not need manual rotation.
