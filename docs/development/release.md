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
   runs the Netflix golden-data gate on all backends, and publishes signed
   artefacts to GitHub Releases.

## Signing

All release artefacts are signed via
[Sigstore keyless](https://docs.sigstore.dev/cosign/overview/) using the
repository's GitHub OIDC identity. No long-lived signing keys live in the
repo or in CI secrets.

Consumers can verify signatures with `cosign verify-blob --certificate-identity-regexp …`.

## Dry-running a release

Before merging a release PR, invoke the `/prep-release` skill locally. It
validates:

- All commits since the last release parse as Conventional Commits.
- The Netflix golden-data gate passes on every available backend.
- `CHANGELOG.md` renders correctly and references no removed files.
- Signing credentials (OIDC) resolve in the current CI environment.

See the [session orientation](../../CLAUDE.md#11-release) for the one-line
summary and the `/prep-release` skill definition for the full checklist.

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
