# CI overview

This page documents the fork's CI surface for contributors. The
authoritative trigger / gate behaviour lives in the workflow files
under [`.github/workflows/`](../../.github/workflows/); this doc
explains the rules a contributor needs to know without reading every
file.

## Workflows

The fork ships eight `pull_request`-triggered workflows:

| File | Purpose |
| --- | --- |
| [`docker-image.yml`](../../.github/workflows/docker-image.yml) | Docker image build (advisory). |
| [`security-scans.yml`](../../.github/workflows/security-scans.yml) | Semgrep / CodeQL / Gitleaks / Dependency Review. |
| [`lint-and-format.yml`](../../.github/workflows/lint-and-format.yml) | Pre-commit, clang-tidy, cppcheck, mypy, registry validate. |
| [`required-aggregator.yml`](../../.github/workflows/required-aggregator.yml) | Single required-check aggregator (ADR-0313). |
| [`ffmpeg-integration.yml`](../../.github/workflows/ffmpeg-integration.yml) | FFmpeg + libvmaf build (gcc / clang / SYCL / Vulkan). |
| [`libvmaf-build-matrix.yml`](../../.github/workflows/libvmaf-build-matrix.yml) | Cross-platform / cross-backend libvmaf build matrix. |
| [`rule-enforcement.yml`](../../.github/workflows/rule-enforcement.yml) | ADR-0100 / 0106 / 0108 process gates. |
| [`tests-and-quality-gates.yml`](../../.github/workflows/tests-and-quality-gates.yml) | Netflix golden, sanitizers, tiny-AI, MCP, coverage, assertion-density. |

## Draft pull requests do not trigger CI

Per [ADR-0331](../adr/0331-skip-ci-on-draft-prs.md), every
`pull_request`-triggered workflow above is gated to skip when the PR
is in `draft` state. Concretely:

- Each workflow's `pull_request:` block lists
  `types: [opened, synchronize, reopened, ready_for_review]`.
- Each top-level job carries
  `if: github.event_name != 'pull_request' || github.event.pull_request.draft == false`.

What this means for contributors:

1. **A draft PR shows no green checks.** The required-checks
   aggregator skips on drafts and branch protection treats the
   missing aggregator as "required check absent". This is benign —
   GitHub blocks merging a draft PR by definition, so the gate cannot
   be bypassed.
2. **Promoting the draft to ready-for-review fires CI exactly once.**
   GitHub's `ready_for_review` event is what re-triggers the
   workflows; subsequent `synchronize` events on the now-ready PR
   fire CI as before.
3. **Pushing to `master` is unaffected.** The job-level `if:` clause
   short-circuits to `true` when there is no PR object (for example
   on `push:` events).

To preview CI status before merging, mark the PR ready-for-review.
You can flip back to draft afterwards if more work is needed; the next
`ready_for_review` will fire a fresh matrix.

## Required-checks aggregator

The single required check on `master` branch protection is the
**Required Checks Aggregator** (see
[ADR-0313](../adr/0313-ci-required-checks-aggregator.md)). It runs on
every non-draft PR, polls for the named sibling check_runs to reach a
terminal state, and accepts `success`, `skipped`, or `neutral` per
check. Because the aggregator itself skips on drafts, draft PRs
display "missing required check" — same situation as item 1 above
and unmergeable for the same reason.

## Local pre-flight gate

Before pushing, run the local subset of CI to catch the common
formatter / lint / fast-test failures:

```bash
make format-check   # clang-format + black + isort, no writes
make lint           # clang-tidy + cppcheck + iwyu + ruff + semgrep
meson test -C build --suite=fast
pre-commit run --all-files  # if .pre-commit-config.yaml hooks are installed
```

The format-check + pre-commit pair catches roughly the same surface as
`lint-and-format.yml`'s `pre-commit` job in seconds, vs. a 10-minute
CI round-trip.
