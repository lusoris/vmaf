#!/usr/bin/env bash
# Enforce coverage thresholds from docs/principles.md §3.
# Usage: coverage-check.sh <lcov.info> <overall_min%> <critical_min%>
#
# Overall = unweighted average line coverage across every file in the lcov
# report (which has already been filtered by the caller to exclude test code,
# subprojects, and /usr/*). Applies to upstream *and* fork-added code — we
# will be modifying upstream paths over time (SIMD fixes, refactors, bug
# fixes) and need a safety net.
#
# Security-critical = files under libvmaf/src/dnn/, plus opt.c and
# read_json_model.c (JSON + option parsing: user-supplied input paths).
# Files with zero lcov data are skipped (no tests → nothing to assert).

set -euo pipefail

INFO="${1:?usage: coverage-check.sh <lcov.info> <overall_min%> <critical_min%>}"
OVERALL_MIN="${2:-70}"
CRITICAL_MIN="${3:-85}"

if ! command -v lcov >/dev/null; then
  echo "lcov not installed — cannot enforce coverage" >&2
  exit 1
fi

# Format from `lcov --summary`:
#   lines......: 73.4% (1234 of 1681 lines)
SUMMARY="$(lcov --summary "$INFO" 2>&1)"
OVERALL="$(grep -oP 'lines\.+: \K[0-9.]+' <<<"$SUMMARY" | head -1 || echo 0)"
echo "Overall line coverage: ${OVERALL}% (min ${OVERALL_MIN}%)"

if awk -v c="$OVERALL" -v m="$OVERALL_MIN" 'BEGIN{exit !(c+0 < m+0)}'; then
  echo "FAIL: overall coverage ${OVERALL}% below minimum ${OVERALL_MIN}%" >&2
  exit 1
fi

# Per-file extraction via `lcov --list --list-full-path`:
#   /abs/path/file.c | lines_pct | fncov_pct | brcov_pct | ...
LIST="$(lcov --list "$INFO" --list-full-path 2>/dev/null | awk '
    /^Overall|^=====|^Lines|^\[/ {next}
    /\|/ && !/Filename/ {
        split($0, parts, "|");
        fname=parts[1]; gsub(/^ +| +$/, "", fname);
        lp=parts[2];    gsub(/[%[:space:]]/, "", lp);
        if (fname != "" && lp != "") print fname, lp;
    }
')"

fail=0
while IFS=' ' read -r path pct; do
  [ -z "$path" ] && continue
  case "$path" in
    */libvmaf/src/dnn/* | */libvmaf/src/opt.c | */libvmaf/src/read_json_model.c)
      if awk -v c="$pct" 'BEGIN{exit !(c+0 == 0)}'; then
        echo "  critical (no tests yet — not enforced): $path — ${pct}%"
        continue
      fi
      echo "  critical: $path — ${pct}%"
      if awk -v c="$pct" -v m="$CRITICAL_MIN" 'BEGIN{exit !(c+0 < m+0)}'; then
        echo "    FAIL: security-critical file below ${CRITICAL_MIN}%" >&2
        fail=1
      fi
      ;;
  esac
done <<<"$LIST"

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "PASS: coverage gate met (overall ≥${OVERALL_MIN}%, critical ≥${CRITICAL_MIN}% where tested)"
