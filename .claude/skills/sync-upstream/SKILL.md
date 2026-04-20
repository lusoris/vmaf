---
name: sync-upstream
description: Reconcile fork master with Netflix/vmaf master. Detects the fork's port-only topology (no shared merge-base) and emits a coverage report; falls back to merge-based sync when the histories are actually connected.
---

# /sync-upstream

## Invocation

```
/sync-upstream [--open-pr]
```

## Background — why pre-flight matters

The fork has been maintained with a **port-only** strategy: each upstream
commit is cherry-picked, rebranded (`feat: port upstream X`, subject preserved),
and squash-merged. That produces commit SHAs on fork master that do **not**
descend from the upstream-master SHAs they originated from. As a result,
`git merge-base master upstream/master` returns empty — the histories are
formally unrelated.

Running a bare `git merge upstream/master --no-ff` in that state requires
`--allow-unrelated-histories` and produces thousands of spurious conflicts
(every fork-local file collides with its upstream counterpart by path,
regardless of content). The skill must detect this topology before attempting
a merge.

## Steps

1. **Pre-flight: topology detection.**
   ```bash
   git fetch upstream
   mb=$(git merge-base master upstream/master 2>/dev/null) || true
   ```
   - **If `mb` is empty** → port-only topology. Go to step 2a (coverage check).
   - **If `mb` is non-empty** → merge-based topology. Go to step 2b (classic merge).

2a. **Port-only coverage check.** For each upstream commit reachable from
    `upstream/master` since the last fork-side port anchor:
    ```bash
    # Derive a reasonable upper bound: the last 50 upstream commits.
    # Expand as needed; subjects are the match key.
    git log upstream/master --pretty=format:'%H%x09%s' -50 > /tmp/sync-upstream-candidates.tsv
    ```
    For each `<sha>\t<subject>` line, search fork master for the subject:
    ```bash
    while IFS=$'\t' read -r sha subj; do
      if git log master --pretty=format:'%s' \
           | grep -Fxq "$subj"; then
        echo "PORTED    $sha  $subj"
      else
        echo "UNPORTED  $sha  $subj"
      fi
    done < /tmp/sync-upstream-candidates.tsv
    ```
    - If every listed commit is `PORTED` → exit with
      `sync-upstream: no action — fork at parity with upstream/master @ <tip-sha>`.
    - If any are `UNPORTED` → list them and recommend
      `/port-upstream-commit <sha>` for each. Do NOT attempt a merge. This is
      the expected outcome in port-only mode.

2b. **Merge-based sync.** Only reached when `mb` is non-empty:
    ```bash
    git switch -c sync/upstream-$(date +%Y%m%d) master
    git merge upstream/master --no-ff
    ```
    Conflict policy (matches D16):
    - Fork wins for: `.github/`, `README.md`, `CLAUDE.md`, `AGENTS.md`,
      `.claude/`, `Dockerfile*`, `libvmaf/meson_options.txt`, anything under
      `libvmaf/src/cuda/`, `libvmaf/src/sycl/`, `libvmaf/src/feature/{cuda,sycl}/`,
      `libvmaf/src/feature/x86/`, `libvmaf/src/feature/arm64/`.
    - Upstream wins for: feature metric code not touched by the fork —
      identify by checking `git log --follow origin/master -- <path>` for
      any fork commits.
    - Manual resolution required: `libvmaf/include/libvmaf/libvmaf.h`,
      `libvmaf/src/libvmaf.c`, `libvmaf/meson.build`,
      `libvmaf/tools/cli_parse.c`, `libvmaf/tools/vmaf.c`.
    For manual conflicts, STOP and surface them with `file:line` context. Do
    NOT resolve.

3. **On clean merge (step 2b only):** `/build-vmaf --backend=cpu`,
   `meson test -C build`, `/cross-backend-diff` on the normal Netflix pair.

4. **If `--open-pr` (step 2b only):** `gh pr create` with title
   `chore(upstream): sync to upstream/master @ <sha>` and body including
   upstream commit count, conflict summary, and test results.
   - In port-only mode (step 2a), `--open-pr` is a no-op when coverage is
     complete; when gaps exist, the recommendation is one PR per
     `/port-upstream-commit <sha>` invocation, not a single sync PR.

## Guardrails

- Refuses to run if working tree is dirty.
- Refuses to open a PR if the Netflix CPU golden tests fail after merge.
- Never `git push --force`.
- Pre-flight short-circuit is mandatory — the classic merge step MUST NOT
  be reached when no merge-base exists, even under operator override, because
  the `--allow-unrelated-histories` fallback has been known to corrupt
  hand-curated port history (see ADR-0028 and the post-mortem thread in
  [docs/rebase-notes.md](../../../docs/rebase-notes.md)).
