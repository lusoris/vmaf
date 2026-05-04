- **`vmaf_tiny_v3` promoted into `model/tiny/registry.json`
  (ADR-0275).** The trained ONNX, sidecar, and model card landed
  with [ADR-0241](../docs/adr/0241-vmaf-tiny-v3-mlp-medium.md) but
  the registry row was missing — the runtime tiny-model loader
  could not select the model and PR #383 (PTQ workstream) had to
  scope-shift back to `vmaf_tiny_v2`. This entry adds a single
  registry row mirroring the v2 entry shape (kind `fr`, opset 17,
  BSD-3-Clause-Plus-Patent, sha256
  `57b2b7e0c62e84e3238b266e423e2008da76ec12ce957c173b2bbed11c65eb78`,
  `smoke: false`). Production default stays `vmaf_tiny_v2` per
  ADR-0241; v3 is opt-in via
  `--tiny-model=vmaf_tiny_v3` (lowest-variance LOSO PLCC of the
  tiny-MLP ladder). The ONNX file, sidecar JSON, and model card
  are unchanged. Unblocks PR #383 and future v3 follow-ups
  (multi-seed LOSO, KoNViD 5-fold, PTQ).
