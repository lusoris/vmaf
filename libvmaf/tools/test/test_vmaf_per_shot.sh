#!/bin/sh
# Smoke test for the vmaf-perShot per-shot CRF predictor sidecar
# (T6-3b / ADR-0222). Invoked from `meson test`; the binary lives
# under <build>/tools/vmaf-perShot.
set -eu

BIN=./tools/vmaf-perShot
WORK="${MESON_BUILD_ROOT:-.}/test_vmaf_per_shot.scratch"
mkdir -p "${WORK}"

# Locate the small test fixture shipped under <repo>/testdata/. The
# `vmaf` repo nests `libvmaf/` one directory below the testdata root.
ROOT="${MESON_SOURCE_ROOT:-${PWD}/..}"
SRC="${ROOT}/testdata/ref_576x324_48f.yuv"
if [ ! -f "${SRC}" ]; then
  SRC="${ROOT}/../testdata/ref_576x324_48f.yuv"
fi
if [ ! -f "${SRC}" ]; then
  echo "test_vmaf_per_shot: missing fixture (looked in ${ROOT}/testdata and ${ROOT}/../testdata)" >&2
  exit 77 # meson "skip"
fi

# 1. --help returns 0.
"${BIN}" --help >/dev/null

# 2. Basic invocation produces a CSV plan with at least one shot row.
PLAN_CSV="${WORK}/plan.csv"
"${BIN}" \
  --reference "${SRC}" \
  --width 576 --height 324 \
  --pixel_format 420 --bitdepth 8 \
  --output "${PLAN_CSV}" \
  --target-vmaf 90 \
  --crf-min 18 --crf-max 35

if ! head -n 1 "${PLAN_CSV}" | grep -q "shot_id,start_frame"; then
  echo "test_vmaf_per_shot: missing header in CSV plan" >&2
  cat "${PLAN_CSV}" >&2
  exit 1
fi

ROWS=$(($(wc -l <"${PLAN_CSV}") - 1))
if [ "${ROWS}" -lt 1 ]; then
  echo "test_vmaf_per_shot: expected ≥1 shot row, got ${ROWS}" >&2
  exit 1
fi

# Verify every predicted_crf is inside [18, 35].
awk -F, 'NR>1 { if ($7 < 18 || $7 > 35) { print "CRF out of range:", $0; exit 1 } }' \
  "${PLAN_CSV}"

# 3. JSON format also works.
PLAN_JSON="${WORK}/plan.json"
"${BIN}" \
  --reference "${SRC}" \
  --width 576 --height 324 \
  --pixel_format 420 --bitdepth 8 \
  --output "${PLAN_JSON}" \
  --format json

if ! grep -q '"shots"' "${PLAN_JSON}"; then
  echo "test_vmaf_per_shot: JSON plan missing 'shots' key" >&2
  cat "${PLAN_JSON}" >&2
  exit 1
fi

# 4. Invalid args fail with non-zero.
if "${BIN}" --reference /tmp/nope --width 0 --height 0 \
  --pixel_format 420 --bitdepth 8 \
  --output /tmp/out 2>/dev/null; then
  echo "test_vmaf_per_shot: expected failure on invalid width" >&2
  exit 1
fi

echo "test_vmaf_per_shot: PASS (${ROWS} shot rows)"
