- Tiny-AI corpus-expansion research (`vmaf_tiny_v5` candidate, deferred per
  ADR-0270): added `ai/scripts/fetch_youtube_ugc_subset.py`,
  `ai/scripts/extract_ugc_features.py`,
  `ai/scripts/train_vmaf_tiny_v5.py`,
  `ai/scripts/eval_loso_vmaf_tiny_v5.py`,
  `ai/scripts/export_vmaf_tiny_v5.py` to ingest the YouTube-UGC vp9
  subset, build a 5-corpus parquet, and run a same-axes Netflix-LOSO
  comparison vs the shipped 4-corpus v2 baseline. Companion ADR-0270 +
  Research-0057 record the +0.000054 mean-PLCC delta as below the
  parent-task 1-σ ship gate; **no v5 ONNX is shipped**, but the
  variance-shrink finding (PLCC σ -58 %, mean RMSE -23 %) is
  documented for follow-up. The shipped tiny FR fusion models
  (v1 / v2 / v3 / v4) are unchanged.
