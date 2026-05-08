# Tiny AI

The Tiny AI surface ships small ONNX perceptual-quality models
alongside classic VMAF SVM models.

- [Overview](overview.md) — architecture, capabilities, model
  lifecycle
- [Training](training.md) — train custom models with `vmaf-train`
- [Inference](inference.md) — run models via the C API or CLI
- [Benchmarks](benchmarks.md) — latency and accuracy numbers
- [Security](security.md) — op allowlists, model validation, supply
  chain
- [Bisect model quality](bisect-model-quality.md) — binary-search a
  checkpoint timeline for the first quality regression (also wired
  as a nightly CI gate)
- [Training data](training-data.md) — Netflix corpus path
  convention, `--data-root` loader API, and evaluation harness for
  fork-local training runs
- [PTQ across EPs](quant-eps.md) — investigate int8 PLCC drop on
  CPU vs CUDA vs OpenVINO (Arc / CPU) Execution Providers
- [Conformal VQA](conformal-vqa.md) — distribution-free prediction
  intervals on top of any vmaf-tune predictor (split-conformal + CV+,
  no new dependencies, ADR-0279 implementation surface)
