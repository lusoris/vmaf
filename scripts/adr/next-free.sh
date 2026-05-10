#!/usr/bin/env bash
# scripts/adr/next-free.sh — print the next available ADR number.
#
# Accounts for both:
#   - ADR files already present in the local working tree, AND
#   - ADR files already merged into origin/master (cross-branch awareness).
#
# Usage:
#   scripts/adr/next-free.sh
#   # → e.g. "0387"
#
# When authoring a new ADR, run this script first and use its output as
# the NNNN prefix.  The pre-commit hook (scripts/ci/check-adr-numbering.sh)
# and the CI gate (rule-enforcement.yml adr-collision-check job) will reject
# the commit if the chosen number collides with any already-merged file.
#
# See ADR-0386 (docs/adr/0386-adr-numbering-collision-prevention.md).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Fetch the latest master tip so we see numbers claimed by PRs that merged
# since the branch was cut.  Shallow depth 50 is enough: origin/master
# accumulates ~1–3 ADRs per session; even on a very busy night 50 commits
# covers far more history than any single session produces.
# Failures are soft (network outage, offline dev) — the pre-commit hook and
# CI gate provide the hard backstop.
git fetch origin master --depth=50 --quiet 2>/dev/null || true

# Collect every 4-digit ADR prefix from:
#   1. Local working tree  (catches files added in the current branch)
#   2. origin/master tree  (catches files merged after the branch was cut)
{
  ls docs/adr/[0-9][0-9][0-9][0-9]-*.md 2>/dev/null || true
  git ls-tree -r --name-only origin/master docs/adr/ 2>/dev/null |
    grep -E '^docs/adr/[0-9]{4}-' || true
} |
  sed 's|.*/||' |
  grep -oE '^[0-9]{4}' |
  sort -u |
  tail -1 |
  awk '{ printf "%04d\n", $1 + 1 }'
