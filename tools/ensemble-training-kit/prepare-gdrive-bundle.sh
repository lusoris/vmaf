#!/usr/bin/env bash
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
#
# prepare-gdrive-bundle.sh — kilian-side bundler for the gdrive
# contributor share. Compresses the local BVI-DVC + Netflix raw YUVs
# to lossless HEVC (.mkv) for transit, generates a sha256 manifest of
# the *original* YUVs (so the contributor's extract step can verify
# decompression bit-exactness), and tars the kit + compressed corpus
# into one Google-Drive-friendly bundle.
#
# Compression: libx265 with ``-x265-params lossless=1``. Gets ~55%
# smaller than the raw YUV on natural content; decodes fast on any
# modern ffmpeg. The contributor uses ``extract-corpus.sh`` to
# unpack back to bit-identical YUVs.
#
# Default sources:
#   $REPO_ROOT/.workingdir2/bvi-dvc-raw   (BVI-DVC, 192 GiB, 772 .yuv)
#   $REPO_ROOT/.workingdir2/netflix       (Netflix drop, 37 GiB,
#                                          9 ref + 70 dis .yuv)
#
# Output:
#   $OUT_DIR/lossless/<corpus>/<source>.mkv      (HEVC lossless)
#   $OUT_DIR/lossless/manifest.sha256            (original YUV hashes)
#   $OUT_DIR/lossless/manifest.json              (size + geometry per file)
#   $OUT_DIR/ensemble-bundle-<ts>.tar.zst        (the gdrive upload)
#
# Usage:
#   bash prepare-gdrive-bundle.sh [--out-dir DIR] [--corpus DIR]...
#                                 [--codec hevc|ffv1|av1] [--threads N]
#                                 [--dry-run]
#
# License note: BVI-DVC is Bristol VI-Lab research-only; ADR-0310 keeps
# the raw archive local. Sharing the compressed corpus with one named
# collaborator (lawrence) for the ensemble retrain is an *informal*
# extension of that policy — see ADR-0310 §Consequences. Do not post
# the bundle URL publicly.

set -Eeuo pipefail

# ---- defaults --------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.workingdir2/gdrive-bundle}"
CODEC="${CODEC:-hevc}"
THREADS="${THREADS:-0}" # 0 = ffmpeg picks
DRY_RUN=0
CORPUS_DIRS=()

# ---- arg parse -------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --corpus)
      CORPUS_DIRS+=("$2")
      shift 2
      ;;
    --codec)
      CODEC="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ${#CORPUS_DIRS[@]} -eq 0 ]]; then
  CORPUS_DIRS=(
    "$REPO_ROOT/.workingdir2/bvi-dvc-raw"
    "$REPO_ROOT/.workingdir2/netflix"
  )
fi

# ---- prereqs ---------------------------------------------------------

for bin in ffmpeg ffprobe sha256sum tar zstd jq; do
  command -v "$bin" >/dev/null 2>&1 || {
    echo "missing dependency: $bin (install before running)" >&2
    exit 3
  }
done

# x265 / FFV1 / aom availability — gate the codec choice.
case "$CODEC" in
  hevc) ENC_LIBS=(libx265) ;;
  ffv1) ENC_LIBS=(ffv1) ;;
  av1) ENC_LIBS=(libaom-av1 libsvtav1) ;;
  *)
    echo "unknown --codec $CODEC; expected hevc, ffv1, or av1" >&2
    exit 2
    ;;
esac
ENC_AVAILABLE=""
for lib in "${ENC_LIBS[@]}"; do
  if ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -qx "$lib"; then
    ENC_AVAILABLE="$lib"
    break
  fi
done
if [[ -z "$ENC_AVAILABLE" ]]; then
  echo "ffmpeg lacks encoder for --codec $CODEC (need one of: ${ENC_LIBS[*]})" >&2
  exit 4
fi

# ---- helpers ---------------------------------------------------------

# Parse "Name_3840x2176_25fps_10bit_420.yuv" into geometry. The
# BVI-DVC + Netflix filename conventions both use this shape; if a
# corpus has a different naming scheme, plumb a sidecar JSON instead.
parse_yuv_geometry() {
  local fname="$1"
  local base
  base="$(basename "$fname")"
  # Strip extension.
  local stem="${base%.yuv}"
  # Pull width x height.
  if [[ "$stem" =~ _([0-9]+)x([0-9]+)_ ]]; then
    WIDTH="${BASH_REMATCH[1]}"
    HEIGHT="${BASH_REMATCH[2]}"
  else
    echo "could not parse W x H from $fname" >&2
    return 1
  fi
  # Pull fps (default 25 if absent).
  if [[ "$stem" =~ _([0-9]+)fps ]]; then
    FPS="${BASH_REMATCH[1]}"
  else
    FPS=25
  fi
  # Pull bit depth (8 / 10 / 12; default 8).
  if [[ "$stem" =~ _([0-9]+)bit ]]; then
    BITS="${BASH_REMATCH[1]}"
  else
    BITS=8
  fi
  # Chroma subsampling — BVI-DVC encodes "_420" / "_422" / "_444"
  # in the stem; Netflix omits and is implicitly 4:2:0.
  if [[ "$stem" =~ _(420|422|444) ]]; then
    CHROMA="${BASH_REMATCH[1]}"
  else
    CHROMA="420"
  fi
  case "$BITS:$CHROMA" in
    8:420) PIX_FMT="yuv420p" ;;
    8:422) PIX_FMT="yuv422p" ;;
    8:444) PIX_FMT="yuv444p" ;;
    10:420) PIX_FMT="yuv420p10le" ;;
    10:422) PIX_FMT="yuv422p10le" ;;
    10:444) PIX_FMT="yuv444p10le" ;;
    12:420) PIX_FMT="yuv420p12le" ;;
    *)
      echo "unsupported bit/chroma ${BITS}/${CHROMA}" >&2
      return 1
      ;;
  esac
}

encode_lossless() {
  local in_yuv="$1"
  local out_mkv="$2"
  parse_yuv_geometry "$in_yuv"

  local thread_arg=()
  if [[ "$THREADS" != "0" ]]; then
    thread_arg=(-threads "$THREADS")
  fi

  case "$ENC_AVAILABLE" in
    libx265)
      ffmpeg -y -hide_banner -loglevel warning \
        "${thread_arg[@]}" \
        -f rawvideo -pix_fmt "$PIX_FMT" -s "${WIDTH}x${HEIGHT}" \
        -framerate "$FPS" -i "$in_yuv" \
        -c:v libx265 -x265-params lossless=1 -preset medium \
        -an "$out_mkv"
      ;;
    ffv1)
      ffmpeg -y -hide_banner -loglevel warning \
        "${thread_arg[@]}" \
        -f rawvideo -pix_fmt "$PIX_FMT" -s "${WIDTH}x${HEIGHT}" \
        -framerate "$FPS" -i "$in_yuv" \
        -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 \
        -slices 24 -slicecrc 1 \
        -an "$out_mkv"
      ;;
    libaom-av1 | libsvtav1)
      # AV1 lossless: best compression, slowest encode. Pass
      # through libaom's `usage=allintra` so we don't
      # accidentally introduce inter-frame coding (libsvtav1
      # honours --tune lossless directly).
      if [[ "$ENC_AVAILABLE" == "libaom-av1" ]]; then
        ffmpeg -y -hide_banner -loglevel warning \
          "${thread_arg[@]}" \
          -f rawvideo -pix_fmt "$PIX_FMT" -s "${WIDTH}x${HEIGHT}" \
          -framerate "$FPS" -i "$in_yuv" \
          -c:v libaom-av1 -aom-params lossless=1 -cpu-used 4 \
          -an "$out_mkv"
      else
        ffmpeg -y -hide_banner -loglevel warning \
          "${thread_arg[@]}" \
          -f rawvideo -pix_fmt "$PIX_FMT" -s "${WIDTH}x${HEIGHT}" \
          -framerate "$FPS" -i "$in_yuv" \
          -c:v libsvtav1 -svtav1-params "tune=lossless" -preset 4 \
          -an "$out_mkv"
      fi
      ;;
  esac
}

# ---- main ------------------------------------------------------------

mkdir -p "$OUT_DIR/lossless"
manifest_jsonl="$OUT_DIR/lossless/manifest.jsonl"
manifest_sha="$OUT_DIR/lossless/manifest.sha256"
: >"$manifest_jsonl"
: >"$manifest_sha"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
total_yuvs=0
total_in_bytes=0
total_out_bytes=0

for src in "${CORPUS_DIRS[@]}"; do
  if [[ ! -d "$src" ]]; then
    echo "skip (no such dir): $src" >&2
    continue
  fi
  corpus_name="$(basename "$src")"
  out_corpus="$OUT_DIR/lossless/$corpus_name"
  mkdir -p "$out_corpus"
  echo "==> compressing $src -> $out_corpus ($CODEC, $ENC_AVAILABLE)"
  while IFS= read -r -d '' yuv; do
    rel="${yuv#"$src"/}"
    out_mkv="$out_corpus/${rel%.yuv}.mkv"
    mkdir -p "$(dirname "$out_mkv")"
    in_size="$(stat -c %s "$yuv")"
    total_in_bytes=$((total_in_bytes + in_size))

    if [[ -f "$out_mkv" ]] && [[ "$out_mkv" -nt "$yuv" ]]; then
      echo "  skip (already encoded): $rel"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
      echo "  DRY-RUN encode: $rel"
    else
      echo "  encoding: $rel ($((in_size / 1024 / 1024)) MiB)"
      encode_lossless "$yuv" "$out_mkv"
    fi

    if [[ "$DRY_RUN" -eq 0 ]] && [[ -f "$out_mkv" ]]; then
      out_size="$(stat -c %s "$out_mkv")"
      total_out_bytes=$((total_out_bytes + out_size))
      yuv_sha="$(sha256sum "$yuv" | awk '{print $1}')"
      echo "${yuv_sha}  ${corpus_name}/${rel}" >>"$manifest_sha"
      jq -nc \
        --arg corpus "$corpus_name" \
        --arg yuv "$rel" \
        --arg mkv "${rel%.yuv}.mkv" \
        --arg pix_fmt "$PIX_FMT" \
        --arg fps "$FPS" \
        --argjson width "$WIDTH" \
        --argjson height "$HEIGHT" \
        --argjson in_bytes "$in_size" \
        --argjson out_bytes "$out_size" \
        --arg yuv_sha256 "$yuv_sha" \
        '{corpus:$corpus, yuv:$yuv, mkv:$mkv, pix_fmt:$pix_fmt,
                  width:$width, height:$height, fps:($fps|tonumber),
                  in_bytes:$in_bytes, out_bytes:$out_bytes,
                  yuv_sha256:$yuv_sha256}' \
        >>"$manifest_jsonl"
    fi

    total_yuvs=$((total_yuvs + 1))
  done < <(find "$src" -maxdepth 4 -type f -name "*.yuv" -print0)
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "DRY-RUN: would encode $total_yuvs YUV(s) totalling $((total_in_bytes / 1024 / 1024 / 1024)) GiB"
  exit 0
fi

# ---- bundle ----------------------------------------------------------

# JSON summary alongside the per-file JSONL so the extract script can
# slurp it once and decide what to do.
jq -nc \
  --arg ts "$ts" \
  --arg codec "$CODEC" \
  --arg encoder "$ENC_AVAILABLE" \
  --argjson total_yuvs "$total_yuvs" \
  --argjson in_bytes "$total_in_bytes" \
  --argjson out_bytes "$total_out_bytes" \
  '{ts:$ts, codec:$codec, encoder:$encoder, total_yuvs:$total_yuvs,
      raw_yuv_bytes:$in_bytes, lossless_bytes:$out_bytes,
      reduction_pct: ((1 - ($out_bytes / $in_bytes)) * 100 | floor)}' \
  >"$OUT_DIR/lossless/summary.json"

echo
echo "==> compression done"
echo "    raw YUV total : $((total_in_bytes / 1024 / 1024 / 1024)) GiB"
echo "    lossless total: $((total_out_bytes / 1024 / 1024 / 1024)) GiB"
echo "    reduction     : $(jq -r .reduction_pct "$OUT_DIR/lossless/summary.json")%"

# Tar the kit + the lossless tree + the extract script + the runbook
# into one zstd-compressed archive. zstd --long=27 buys a couple
# percent more on top of HEVC-lossless without melting the operator's
# CPU on extraction.
bundle_tar="$OUT_DIR/ensemble-bundle-${ts}.tar.zst"
echo "==> bundling: $bundle_tar"

# Stage a kit directory next to the lossless tree so tar emits a
# self-contained layout.
stage="$OUT_DIR/_stage_${ts}"
mkdir -p "$stage"
cp -r "$REPO_ROOT/tools/ensemble-training-kit" "$stage/ensemble-training-kit"
ln -s "../lossless" "$stage/ensemble-training-kit/corpus"
# The README + extract-corpus.sh + the runbook live inside the kit
# already; the symlinked ./corpus/ makes the run-full-pipeline.sh
# default --ref-dir resolution land on the right files.

tar --use-compress-program="zstd --long=27 -19 -T0" \
  -C "$OUT_DIR" \
  -cf "$bundle_tar" \
  "lossless" "_stage_${ts}/ensemble-training-kit"

bundle_size="$(stat -c %s "$bundle_tar")"
sha256sum "$bundle_tar" >"${bundle_tar}.sha256"

echo
echo "==> bundle ready"
echo "    file  : $bundle_tar"
echo "    size  : $((bundle_size / 1024 / 1024 / 1024)) GiB"
echo "    sha256: $(awk '{print $1}' "${bundle_tar}.sha256")"
echo
echo "Upload ${bundle_tar} + ${bundle_tar}.sha256 to the contributor's"
echo "Google Drive folder. The contributor runs ./extract-corpus.sh"
echo "after download to restore bit-exact YUVs."
