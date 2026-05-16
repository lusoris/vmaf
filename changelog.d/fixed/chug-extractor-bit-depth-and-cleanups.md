### CHUG extractor audit fixes — bit-depth preservation, dead disk write, frame-mismatch warning

Three independent bugs in `ai/scripts/extract_k150k_features.py` found during a
post-merge audit:

- **[H] Data corruption on HDR clips** — `_load_jsonl_metadata` omitted
  `"chug_bit_depth"` from its `keep` tuple, causing `_geometry_from_sidecar` to
  receive `None` and fall back to `pix_fmt="yuv420p"` (8-bit) for every 10-bit
  HDR CHUG clip. Feature vectors for all HDR rows were numerically incorrect.
  Fixed by adding `"chug_bit_depth"` to the `keep` tuple.

- **[M] Dead disk write per CUDA clip** — `_run_feature_passes` wrote the merged
  in-memory frame list back to `out_json` after every CUDA extraction, but no
  caller ever read that file (the caller receives the returned Python list, and
  `_process_clip` unconditionally unlinks it in `finally`). Removed the dead
  `write_text` call.

- **[L] Silent frame-count truncation** — `_merge_frame_metrics` truncated to
  `min(len(primary), len(residual))` without any audit signal when the CUDA and
  CPU residual passes returned different frame counts. Now emits a `warnings.warn`
  before truncating, preserving the soft-fail behaviour while surfacing the
  discrepancy.

Research digest: [Research-0137](../../docs/research/0137-chug-extractor-audit-fix-bundle-2026-05-16.md)
