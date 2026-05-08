# Release process

The Lusoris fork of libvmaf releases via automation — not manual tag-and-draft.
Pushes to `master` drive a
[release-please](https://github.com/googleapis/release-please-action)
workflow that maintains a release PR, and merging that PR triggers the full
release pipeline (build, sign, publish).

## Version scheme

Releases follow `vX.Y.Z-lusoris.N`:

- `X.Y.Z` tracks the upstream Netflix VMAF version the fork is aligned to.
- `-lusoris.N` is the fork-specific revision. It bumps independently of
  upstream and resets to `.1` when `X.Y.Z` changes.

Example progression:

```text
v3.0.0-lusoris.1  # initial fork release against upstream 3.0.0
v3.0.0-lusoris.2  # second fork revision, still on upstream 3.0.0
v3.1.0-lusoris.1  # reset after upstream 3.1.0 sync
```

Upstream sync PRs (see `/sync-upstream`) update the upstream component; regular
fork PRs advance `-lusoris.N`.

## Automation flow

1. **release-please watches master.** On each push it inspects Conventional
   Commit headers (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, …) to determine
   whether a release is warranted. If so, it opens or updates a release PR that
   bumps `VERSION`, updates `CHANGELOG.md`, and collects user-visible change
   summaries.
2. **Merging the release PR** tags the commit and triggers the release
   workflow.
3. **The release workflow** builds artefacts (libvmaf binaries, Python wheels),
   runs the Netflix golden-data gate (CPU only — the GPU/SIMD backends
   are covered by per-backend snapshot tests at ULP tolerance, not by
   the goldens), and publishes signed artefacts to GitHub Releases.

## ADR index regeneration policy

`docs/adr/README.md` is the rendered index of every ADR in the fork. Its
"Index" table is generated from per-ADR fragments under
`docs/adr/_index_fragments/<slug>.md` plus an order manifest at
`docs/adr/_index_fragments/_order.txt`. The renderer is
[`scripts/docs/concat-adr-index.sh`](../../scripts/docs/concat-adr-index.sh)
(see [ADR-0221](../adr/0221-changelog-adr-fragment-pattern.md) for why the
pattern exists).

**When adding a new ADR (the common case)** — write the fragment as part of
the same PR and append its slug to `_order.txt`. The PR template's
ADR-index checklist row covers this. Manual append is preferred over
`--write` because it produces a one-line diff that reviewers can verify by
eye and avoids touching unrelated rows.

**When fixing drift between fragments and `README.md` (this sweep's case)**
— run `scripts/docs/concat-adr-index.sh --check` to capture the full diff,
then audit each row against the four drift classes:

- **Silent loss** — fragment exists, README is missing the row.
  Regenerating with `--write` keeps the fragment's row.
- **Orphan content** — README has a row, fragment does not exist.
  Backfill the fragment from the README row (the row content already
  reflects the ADR's accepted state). Do **not** delete the row without
  evidence the ADR is genuinely stale (`Status: Withdrawn` or
  `Superseded` in the ADR file body, plus the underlying decision being
  moot).
- **Reformatted** — same content, different shape (column order, status
  spelling, slug case). Regenerate; the fragment is canonical.
- **Duplicate** — the same row appears more than once in `README.md`,
  usually from a stale append-only edit. Regenerate; the fragment is
  emitted exactly once.

After every fragment-side fix, run `--write` once and verify the README
diff matches the audit's expected shape (rows preserved, duplicates
collapsed, missing rows restored). Reviewers can re-run `--check` against
the rebuilt branch and expect a clean exit.

**Renumbered slugs.** When the dedup sweep referenced in the script's
header comment renumbers an ADR (e.g. `0270-saliency-…` → `0286-saliency-…`),
the fragment must be **renamed** to match the new slug — not duplicated. The
`_order.txt` entry follows the same rename. The fragment body's
`[ADR-NNNN](NNNN-slug.md)` link must match the renumbered slug; mismatches
silently render rows that point at non-existent ADR files. The
fragment-vs-ADR-file slug audit is one line:

```bash
for f in docs/adr/_index_fragments/[0-9]*.md; do
    base=$(basename "$f" .md)
    [[ -f "docs/adr/$base.md" ]] || echo "STALE FRAGMENT: $f"
done
```

## Signing

All release artefacts are signed via
[Sigstore keyless](https://docs.sigstore.dev/cosign/overview/) using the
repository's GitHub OIDC identity. No long-lived signing keys live in the
repo or in CI secrets.

Consumers can verify signatures with
`cosign verify-blob --certificate-identity-regexp …`.

## Dry-running a release

Before merging a release PR, invoke the `/prep-release` skill locally. It
validates:

- All commits since the last release parse as Conventional Commits.
- The Netflix golden-data gate (CPU scalar + fixed-point) passes.
  GPU / SIMD backends are validated separately via per-backend
  snapshot tests.
- `CHANGELOG.md` renders correctly and references no removed files.
- Signing credentials (OIDC) resolve in the current CI environment.

See the [session orientation](../../CLAUDE.md#11-release) for the one-line
summary and the `/prep-release` skill definition for the full checklist.

## `master` branch protection

`master` is protected at the GitHub API layer — the policy in
[CLAUDE.md §12](../../CLAUDE.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md)
is enforced at the host, not just honored by convention.

- **Required status checks (19):** pre-commit, ruff+mypy+black, semgrep,
  Netflix CPU golden (D24), ASan/UBSan/MSan ×3, Assertion density,
  CodeQL ×4, clang-tidy, cppcheck, Tiny AI, MINGW build, dependency-review,
  gitleaks, shellcheck+shfmt.
- **Linear history required** — merges are squash-or-ff-only.
- **Force-push and deletion disabled.**
- **Admin bypass kept on** (owner can land emergency fixes that skip required
  checks — use sparingly; see the emergency-release section below).
- **Not required (non-blocking signals):** Coverage gate (~40 min — built
  with `-fprofile-update=atomic` since 2026-04-18 to survive parallel-meson
  SIMD-counter races, see [ADR-0110](../adr/0110-coverage-gate-fprofile-update-atomic.md)),
  GPU-advisory jobs, Semgrep OSS.

Management: `gh api --method PUT repos/lusoris/vmaf/branches/master/protection`
with a JSON payload. The current rule set is documented in
[ADR-0037](../adr/0037-master-branch-protection.md).
When adding or renaming a required CI job, update the `contexts` list.

## Emergency release (out-of-band)

If a CVE requires an out-of-band release that bypasses the release-please PR:

1. Branch off `master` into `hotfix/CVE-YYYY-NNNN`.
2. Land the fix with a `fix:` commit and a signed-off-by line.
3. Manually tag `vX.Y.Z-lusoris.N+1` — release-please will reconcile on the
   next regular push.
4. Backport the CVE fix to any active stacked release branches.

## Upstream parallel

The upstream Netflix release process (manual version bump, manual CHANGELOG
editing, draft-a-release on GitHub) is documented at
[Netflix/vmaf — release.md](https://github.com/Netflix/vmaf/blob/master/resource/doc/release.md).
It does not apply to this fork.
