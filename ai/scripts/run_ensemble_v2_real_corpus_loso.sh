#!/usr/bin/env bash
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
#
# Real-corpus LOSO retrain wrapper for the ``fr_regressor_v2`` deep
# ensemble (ADR-0309 — companion to ADR-0303).
#
# Invokes ``ai/scripts/train_fr_regressor_v2_ensemble_loso.py`` once
# per seed in {0,1,2,3,4} against the locally available Netflix Public
# Dataset (``.workingdir2/netflix/`` — 9 reference + 70 distorted YUVs
# provided by lawrence on 2026-04-27, ~37 GB, gitignored).
#
# Usage:
#   bash ai/scripts/run_ensemble_v2_real_corpus_loso.sh
#   CORPUS_ROOT=/path/to/netflix bash ai/scripts/run_ensemble_v2_real_corpus_loso.sh
#
# Wall-time estimate: 6-12 h on an 8 GB GPU (5 seeds x 9 LOSO folds).
#
# After the wrapper finishes, run:
#   python ai/scripts/validate_ensemble_seeds.py runs/ensemble_v2_real/
# to apply the two-part production-flip gate (ADR-0303) and emit
# either runs/ensemble_v2_real/PROMOTE.json or HOLD.json.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

corpus_root="${CORPUS_ROOT:-$repo_root/.workingdir2/netflix}"
out_dir="${OUT_DIR:-$repo_root/runs/ensemble_v2_real}"
log_dir="$out_dir/logs"
seeds=(0 1 2 3 4)

if [[ ! -d "$corpus_root" ]]; then
  echo "error: corpus root not found: $corpus_root" >&2
  echo "Either populate \$CORPUS_ROOT or copy the Netflix Public Dataset" \
    "to .workingdir2/netflix/ (see ADR-0309 prerequisites)." >&2
  exit 2
fi

# Sanity-check that at least one ref + one dis YUV is reachable. The
# Netflix corpus layout is .workingdir2/netflix/{ref,dis}/*.yuv per
# the existing run_training.sh wrapper (ADR-0203). Tolerate either
# {ref,dis} subdirs or a flat layout with *_ref / *_dis filenames.
ref_count=0
dis_count=0
if [[ -d "$corpus_root/ref" ]] && [[ -d "$corpus_root/dis" ]]; then
  ref_count=$(find "$corpus_root/ref" -maxdepth 2 -name '*.yuv' | wc -l)
  dis_count=$(find "$corpus_root/dis" -maxdepth 2 -name '*.yuv' | wc -l)
else
  ref_count=$(find "$corpus_root" -maxdepth 2 -name '*ref*.yuv' | wc -l)
  dis_count=$(find "$corpus_root" -maxdepth 2 -name '*dis*.yuv' | wc -l)
fi

if [[ "$ref_count" -lt 1 ]] || [[ "$dis_count" -lt 1 ]]; then
  echo "error: corpus at $corpus_root has insufficient YUVs" \
    "(ref=$ref_count dis=$dis_count; need >=1 of each)" >&2
  exit 2
fi

mkdir -p "$out_dir" "$log_dir"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
echo "[ensemble-v2-real] corpus_root=$corpus_root ref=$ref_count dis=$dis_count"
echo "[ensemble-v2-real] out_dir=$out_dir log_dir=$log_dir start=$ts"

start_secs=$(date -u +%s)

for seed in "${seeds[@]}"; do
  log_file="$log_dir/seed${seed}_${ts}.log"
  echo "[ensemble-v2-real] seed=$seed -> $log_file"
  python "$repo_root/ai/scripts/train_fr_regressor_v2_ensemble_loso.py" \
    --seeds "$seed" \
    --corpus-root "$corpus_root" \
    --output "$out_dir/loso_seed${seed}.json" \
    --out-dir "$out_dir" \
    2>&1 | tee "$log_file"
done

end_secs=$(date -u +%s)
elapsed=$((end_secs - start_secs))

echo "[ensemble-v2-real] done seeds=${seeds[*]} elapsed=${elapsed}s out_dir=$out_dir"
echo "[ensemble-v2-real] next: python ai/scripts/validate_ensemble_seeds.py $out_dir"
