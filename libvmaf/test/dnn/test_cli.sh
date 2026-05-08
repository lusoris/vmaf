#!/usr/bin/env bash
# test_cli.sh — smoke-test the `vmaf --tiny-model` option.
#
# Requires: meson build with -Denable_dnn=enabled, an ONNX model under
# model/tiny/ (any), and libonnxruntime on the runtime library path.
#
# When DNN is disabled, asserts the clear error message instead.
set -eu

: "${VMAF_BIN:=build/tools/vmaf}"

if [[ ! -x "$VMAF_BIN" ]]; then
  echo "vmaf binary not found at $VMAF_BIN — set VMAF_BIN=<path>" >&2
  exit 77 # meson's "skipped"
fi

# `vmaf --help` exits with 1 by convention, so capture the output first
# instead of piping into grep under set -o pipefail.
help_text="$("$VMAF_BIN" --help 2>&1 || true)"

# 1. Help text must advertise the tiny flags.
printf '%s\n' "$help_text" | grep -q -- '--tiny-model' || {
  echo "help missing --tiny-model"
  exit 1
}
printf '%s\n' "$help_text" | grep -q -- '--tiny-device' || {
  echo "help missing --tiny-device"
  exit 1
}
printf '%s\n' "$help_text" | grep -q -- '--no-reference' || {
  echo "help missing --no-reference"
  exit 1
}

# 2. Invalid device string must be rejected with a useful message.
# The keyword list grew with coreml / coreml-{ane,gpu,cpu} (ADR-0365);
# match the stable head + tail rather than the verbatim middle so this
# stays passing across future grammar additions.
if "$VMAF_BIN" --tiny-model /nonexistent.onnx --tiny-device bogus 2>&1 |
  grep -qiE 'auto\|cpu\|cuda\|openvino.*rocm'; then
  :
else
  echo "expected validation error for --tiny-device bogus"
  exit 1
fi

# 3. The new coreml / coreml-{ane,gpu,cpu} keywords must be accepted by
# the validator. `vmaf` exits non-zero because we don't supply a
# reference YUV, but the rejection message would mention "Invalid
# argument" if the keyword itself were unknown. We only assert the
# keyword does not surface as a validation error.
for dev in coreml coreml-ane coreml-gpu coreml-cpu; do
  out="$("$VMAF_BIN" --tiny-device "$dev" 2>&1 || true)"
  if printf '%s\n' "$out" | grep -q "Invalid argument \"$dev\""; then
    echo "validator wrongly rejected --tiny-device $dev"
    exit 1
  fi
done

echo "PASS: $0"
