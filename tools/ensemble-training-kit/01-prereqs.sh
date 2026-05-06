#!/usr/bin/env bash
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
#
# Step 01: Prerequisite check for the ensemble training kit (ADR-0324).
#
# Verifies that the host has the toolchain pieces required to run the
# Phase-A corpus generation + LOSO retrain end-to-end. Each check is
# fatal — if anything is missing, the script prints a remediation hint
# and exits non-zero so the operator sees one problem at a time.
#
# Environment overrides:
#   LIBVMAF_BIN  Path to libvmaf-CUDA binary (default: libvmaf/build-cuda/tools/vmaf)
#   GPU_MIN_MIB  Minimum free GPU memory in MiB (default: 6144)

set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$KIT_DIR/../.." && pwd)}"

LIBVMAF_BIN="${LIBVMAF_BIN:-$REPO_ROOT/libvmaf/build-cuda/tools/vmaf}"
GPU_MIN_MIB="${GPU_MIN_MIB:-6144}"

echo "[prereqs] repo_root=$REPO_ROOT"
echo "[prereqs] libvmaf_bin=$LIBVMAF_BIN"
echo

fail() {
  echo "[prereqs] FAIL: $1" >&2
  [[ -n "${2:-}" ]] && echo "[prereqs] hint: $2" >&2
  exit 2
}

ok() { echo "[prereqs] ok: $1"; }

# 1. ffmpeg + NVENC -----------------------------------------------------------
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg not in PATH" \
  "install ffmpeg with --enable-libnvenc (Linux distros usually ship it; verify via 'ffmpeg -encoders | grep nvenc')"
ffmpeg_ver=$(ffmpeg -version 2>/dev/null | head -1)
if ! ffmpeg -hide_banner -encoders 2>&1 | grep -q h264_nvenc; then
  fail "ffmpeg present but lacks h264_nvenc encoder" \
    "rebuild ffmpeg with --enable-libnvenc (NVIDIA Video Codec SDK headers must be on the include path)"
fi
ok "ffmpeg + h264_nvenc ($ffmpeg_ver)"

# 2. libvmaf with CUDA --------------------------------------------------------
if [[ ! -x "$LIBVMAF_BIN" ]]; then
  fail "libvmaf-CUDA binary not found at $LIBVMAF_BIN" \
    "build with: meson setup build-cuda -Denable_cuda=true && ninja -C build-cuda"
fi
# Probe its --help for cuda flags so we know it was actually compiled with CUDA.
if ! "$LIBVMAF_BIN" --help 2>&1 | grep -qi cuda; then
  fail "libvmaf binary at $LIBVMAF_BIN does not advertise CUDA in --help" \
    "rebuild with -Denable_cuda=true; the CPU-only binary will not exercise the corpus generator's --gpumask path"
fi
ok "libvmaf-CUDA binary ($LIBVMAF_BIN)"

# 3. Python + torch.cuda ------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not in PATH" "install Python >= 3.12"
fi
py_ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
py_major=$(python3 -c 'import sys; print(sys.version_info[0])')
py_minor=$(python3 -c 'import sys; print(sys.version_info[1])')
if [[ "$py_major" -lt 3 || ("$py_major" -eq 3 && "$py_minor" -lt 12) ]]; then
  fail "python $py_ver is too old (need >= 3.12)" \
    "install python3.12+; pyenv or your distro's python3.12 package both work"
fi
ok "python $py_ver"

if ! python3 -c "import torch" 2>/dev/null; then
  fail "torch not importable in python3" \
    "pip install -r tools/ensemble-training-kit/requirements-frozen.txt"
fi
torch_ver=$(python3 -c 'import torch; print(torch.__version__)')
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
  fail "torch.cuda.is_available() is False" \
    "ensure NVIDIA driver + CUDA runtime are installed and torch was built with CUDA support"
fi
ok "torch $torch_ver with CUDA"

# 4. GPU free memory ----------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not in PATH" "install the NVIDIA driver"
fi
gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
  head -1 | tr -d '[:space:]')
if [[ -z "$gpu_free_mib" ]]; then
  fail "nvidia-smi did not report memory.free" "verify the NVIDIA driver is loaded"
fi
if [[ "$gpu_free_mib" -lt "$GPU_MIN_MIB" ]]; then
  fail "GPU has only ${gpu_free_mib} MiB free (need >= ${GPU_MIN_MIB})" \
    "free up VRAM; close other CUDA workloads"
fi
ok "GPU free memory: ${gpu_free_mib} MiB (>= ${GPU_MIN_MIB})"

# 5. Optional: vainfo for QSV detection (informational) ----------------------
if command -v vainfo >/dev/null 2>&1 && vainfo --display drm 2>/dev/null | grep -q iHD; then
  echo "[prereqs] info: Intel iHD VA-API driver detected (QSV lanes available)"
else
  echo "[prereqs] info: Intel iHD VA-API driver not detected (QSV lanes will be skipped — NVENC-only run)"
fi

echo
echo "[prereqs] all checks passed"
