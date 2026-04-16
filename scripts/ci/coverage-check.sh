#!/usr/bin/env bash
# Enforce coverage thresholds from docs/principles.md §3.
# Usage: coverage-check.sh <lcov.info> <overall_min%> <critical_min%>
#
# Parses an lcov --list output. Overall = average over all listed files.
# Security-critical = files under libvmaf/src/dnn/, plus opt.c and
# read_json_model.c (JSON + option parsing: user-supplied input paths).

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

# awk to compare floats
if awk -v c="$OVERALL" -v m="$OVERALL_MIN" 'BEGIN{exit !(c+0 < m+0)}'; then
  echo "FAIL: overall coverage ${OVERALL}% below minimum ${OVERALL_MIN}%" >&2
  exit 1
fi

# Per-file extraction via `lcov --list` (machine-parseable).
# Format:  filename | lines_pct | fncov_pct | brcov_pct | ...
LIST="$(lcov --list "$INFO" --list-full-path 2>/dev/null | awk '
    /^Overall|^=====|^Lines|^\[/ {next}
    /\|/ && !/Filename/ {
        # strip leading/trailing space from filename
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

echo "PASS: coverage gate met (overall ≥${OVERALL_MIN}%, critical ≥${CRITICAL_MIN}%)"
