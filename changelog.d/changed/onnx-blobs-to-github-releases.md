- The three model/tiny/*.onnx blobs ≥1MB (`transnet_v2.onnx`,
  `fastdvdnet_pre.onnx`, `lpips_sq.onnx`, totalling 44 MB) are no
  longer inlined in git. They live as attachments on the
  `tiny-blobs-v1` GitHub Release and are fetched on demand by
  `scripts/ai/fetch-tiny-blobs.sh`, which sha256-verifies each
  download against the recorded hash in
  `model/tiny/registry.json`. New checkouts run the fetcher once
  (~3.5 s) instead of inlining 44 MB into every clone. Per
  ADR-0457. Smaller ONNX files (<1 MB each, 25 files) stay inline;
  the per-file fetch overhead dominates the per-byte storage
  savings below the cutoff.
