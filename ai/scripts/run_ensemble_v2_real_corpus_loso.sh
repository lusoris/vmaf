#!/usr/bin/env bash
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
#
# Real-corpus LOSO retrain wrapper for the ``fr_regressor_v2`` deep
# ensemble (ADR-0309 — companion to ADR-0303; real loader / per-fold
# training landed in ADR-0319).
#
# Invokes ``ai/scripts/train_fr_regressor_v2_ensemble_loso.py`` once
# per seed in {0,1,2,3,4} against the Phase A canonical-6 JSONL corpus
# (``runs/phase_a/full_grid/per_frame_canonical6.jsonl``). The corpus
# is generated locally via ``scripts/dev/hw_encoder_corpus.py`` over
# the 9 Netflix ref YUVs in ``.workingdir2/netflix/`` (see
# ``docs/ai/ensemble-v2-real-corpus-retrain-runbook.md`` §Step 0).
#
# Usage:
#   bash ai/scripts/run_ensemble_v2_real_corpus_loso.sh
#   CORPUS_JSONL=/path/to/canonical6.jsonl \
#     bash ai/scripts/run_ensemble_v2_real_corpus_loso.sh
#
# Wall-time estimate: ~5–30 min on RTX 4090 / 8 GB-class GPUs (5 seeds
# × 9 LOSO folds × 200 epochs against ~5,640 NVENC rows). Slower CPUs
# scale linearly.
#
# After the wrapper finishes, run:
#   python ai/scripts/validate_ensemble_seeds.py runs/ensemble_v2_real/
# to apply the two-part production-flip gate (ADR-0303) and emit
# either runs/ensemble_v2_real/PROMOTE.json or HOLD.json.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

corpus_jsonl="${CORPUS_JSONL:-$repo_root/runs/phase_a/full_grid/per_frame_canonical6.jsonl}"
out_dir="${OUT_DIR:-$repo_root/runs/ensemble_v2_real}"
log_dir="$out_dir/logs"
seeds=(0 1 2 3 4)

if [[ ! -f "$corpus_jsonl" ]]; then
  echo "error: corpus JSONL not found: $corpus_jsonl" >&2
  echo "Generate it via scripts/dev/hw_encoder_corpus.py over the 9" \
    "Netflix ref YUVs (see docs/ai/ensemble-v2-real-corpus-retrain-runbook.md" \
    "§Step 0). Override the path via \$CORPUS_JSONL if it lives elsewhere." >&2
  exit 2
fi

# Sanity-check the corpus has at least 100 per-frame rows (a single
# (src, encoder, cq) triple is ~150 frames; the canonical-6 corpus is
# ~5,640 rows for NVENC-only 9 sources × 4 cqs).
row_count=$(wc -l <"$corpus_jsonl")
if [[ "$row_count" -lt 100 ]]; then
  echo "error: corpus $corpus_jsonl has only $row_count rows; need >=100." >&2
  echo "Regenerate via scripts/dev/hw_encoder_corpus.py — see runbook." >&2
  exit 2
fi

mkdir -p "$out_dir" "$log_dir"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
echo "[ensemble-v2-real] corpus=$corpus_jsonl rows=$row_count"
echo "[ensemble-v2-real] out_dir=$out_dir log_dir=$log_dir start=$ts"

start_secs=$(date -u +%s)

for seed in "${seeds[@]}"; do
  log_file="$log_dir/seed${seed}_${ts}.log"
  echo "[ensemble-v2-real] seed=$seed -> $log_file"
  python "$repo_root/ai/scripts/train_fr_regressor_v2_ensemble_loso.py" \
    --seeds "$seed" \
    --corpus "$corpus_jsonl" \
    --out-dir "$out_dir" \
    2>&1 | tee "$log_file"
done

end_secs=$(date -u +%s)
elapsed=$((end_secs - start_secs))

echo "[ensemble-v2-real] done seeds=${seeds[*]} elapsed=${elapsed}s out_dir=$out_dir"
echo "[ensemble-v2-real] next: python ai/scripts/validate_ensemble_seeds.py $out_dir"
