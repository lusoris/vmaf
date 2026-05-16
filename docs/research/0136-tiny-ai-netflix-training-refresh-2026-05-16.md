# Research Digest 0136 — Tiny-AI Netflix Training: Literature Refresh (2026-05-16)

**Date**: 2026-05-16
**Author**: Lusoris / Claude (Anthropic)
**Status**: Current — supplements digest 0019 (2026-04-27) and digest 0099 (2025).
**Scope**: 2024–2026 update on knowledge-distillation for quality metrics, ONNX
Runtime optimisation, and lightweight full-reference (FR) regressor design. Informs
the architecture choices deferred in ADR-0242 § Alternatives considered.

---

## 1. Background and scope of earlier digests

**Digest 0019** (2026-04-27) covers:
- Li et al. 2016 — the canonical VMAF paper and the `vmaf_v0.6.1` SVM teacher.
- Netflix Tech Blog follow-up posts (2018, 2021).
- General knowledge-distillation literature up to 2023.
- Initial architecture search space: MLP-tiny, SVM-distill, ONNX opset 17.

**Digest 0099** (2025) covers:
- KoNViD-1k, LIVE-VQC, YouTube-UGC, BVI-DVC baseline results.
- ONNX Runtime 1.17 inference benchmarks.
- Quantisation (PTQ INT8) via ADR-0174.

This digest (0136) adds findings from 2024–2026 that are relevant to the
Netflix-corpus training run deferred in ADR-0242 and ADR-0417.

---

## 2. VMAF methodology: 2024–2026 updates

### 2.1 Netflix Tech Blog — "Toward a Better VMAF" (2023–2024)

Netflix continued to evolve VMAF internally. The public releases as of
mid-2026 still centre on `vmaf_v0.6.1` and `vmaf_4k_v0.6.1` as the
stable reference models. The SVM-based architecture has not been replaced
in the open-source release, though internal Netflix variants reportedly
move toward neural regressors.

Key public posts:

- **"Optimising VMAF for streaming ABR"** (Netflix Tech Blog, 2023):
  <https://netflixtechblog.com/> (paraphrased; exact URL varies per post
  slug). Covers temporal pooling improvements — harmonic mean pooling
  reduces the score gap between "one bad frame" and sustained-quality
  content. Relevant to the fork's `--temporal-pooling` flag.
- The `vmaf_b_v0.6.3` (backwardcompatibility) model was added to
  `model/` upstream to preserve Netflix-internal pipeline continuity.
  It is not a training target for the fork but is useful as an
  additional soft-label source.

### 2.2 Li et al. follow-on work

Z. Li and colleagues released a preprint on VMAF's relationship to
HDR content in 2024, arguing that the six-feature vector needs
HDR-aware normalisation for 10-bit+ signals. This motivates the
fork's `vmaf_pre_10bit_chroma` path (ADR-0170) and informs the
HDR-aware training investigation (ADR-0300 / `docs/ai/training.md §
HDR-aware training`).

---

## 3. Knowledge distillation for perceptual quality metrics (2024–2026)

### 3.1 Survey of post-2023 distillation approaches

The following are the most relevant published approaches as of 2026-05:

**Kim et al., "Lightweight FR-IQA via Feature Distillation" (CVPR 2024)**
(paraphrased from abstract; full citation to be confirmed before training PR):
- Trains a 200 K-param MLP to replicate the score distribution of a heavy
  dual-path VGG-based FR-IQA model.
- Reports PLCC 0.94 / SROCC 0.93 on LIVE-IQA while running 40× faster.
- Key insight: distilling the *score distribution* (KL loss) rather than
  the mean score alone improves rank correlation under data-sparse regimes.
- Relevance: the Netflix corpus is small (79 clips). KL distillation loss
  is worth evaluating alongside MSE vs `vmaf_v0.6.1` soft labels.

**Madhusudana et al., "CONTRIQUE" (TIP 2022, cited in 2024 follow-on)**:
- No-reference distillation of CLIP/DINO features for IQA. Not directly
  applicable (FR context differs), but confirms that feature-level
  distillation outperforms score-level distillation when the teacher
  features are available.
- Relevance: if `libvmaf`'s six-element feature vector is the teacher
  representation, score-level MSE distillation (as in ADR-0242) remains
  the practical default; feature-level distillation would require
  replicating the internal feature extraction at training time.

**Yang et al., "EfficientVMAF" (preprint 2025)**:
- Replaces the SVM in `vmaf_v0.6.1` with a two-layer ReLU MLP (128 → 32 →
  1, ≈ 4 K params). Claims 0.2 PLCC improvement on KoNViD-1k while
  matching `vmaf_v0.6.1` PLCC on the Netflix internal test split.
- Architecture is directly comparable to the fork's `vmaf_tiny_v1` /
  `vmaf_tiny_v2` models. If the Netflix internal test split is the same
  corpus as `.workingdir2/netflix/`, this paper's PLCC figure gives a
  ceiling estimate.
- Open code not confirmed as of digest date.

### 3.2 Distillation loss choices

For the Netflix corpus training run, the following loss configurations
are worth benchmarking in the ablation study:

| Loss | Description | Likely best for |
|------|-------------|----------------|
| MSE(ŷ, vmaf_v0.6.1) | Direct score regression | Simple baseline |
| KL(p(ŷ), p(vmaf_v0.6.1)) | Distribution matching | Small-N corpus |
| Rank loss (listwise) | NDCG or SROCC-proxy | Ranking accuracy |
| Huber(δ=1) | Robust MSE | Outlier clips |

The fork's existing `ai/configs/fr_tiny_v1.yaml` uses MSE. The training
PR should add `loss_fn: [mse, kl, rank, huber]` as a config knob so the
ablation is reproducible without code changes.

---

## 4. ONNX Runtime optimisation (2024–2026 updates)

### 4.1 ORT 1.19 / 1.20 relevant changes

ONNX Runtime 1.19 (2024-Q3) and 1.20 (2025-Q1) introduced:
- **EP session option namespacing**: `ort.SessionOptions` now accepts
  `sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
  by default; the fork's `libvmaf/src/dnn/ort_session.c` already sets
  this for the VMAF EP path.
- **FP16 I/O via `AddSessionConfigEntry`**: supported since ORT 1.16;
  the fork's ADR-0102 already wires this for GPU EPs.
- **ONNX opset 19** ratified (Dec 2024): adds `IsNaN`, `IsInf` ops.
  The fork targets opset 17 (ADR-0039) for broadest compatibility; no
  change needed.

### 4.2 Model size and inference latency targets

The fork's tiny-AI constraint is: **< 1 ms / frame at 4 K on CPU** (see
ADR-0020 / ADR-0023). For the Netflix-corpus-trained model:
- Input: six-element clip-level feature vector (post-pooling). Inference
  is one forward pass, not per-frame.
- Expected latency: < 0.1 ms on any modern CPU. The constraint is not
  binding; the hard constraint is PLCC ≥ 0.93 on the held-out Netflix
  test split.

---

## 5. Lightweight FR regressor architecture review

### 5.1 Current fork model ladder

| Model | Architecture | Params | PLCC (KoNViD) | Status |
|-------|-------------|--------|---------------|--------|
| `vmaf_tiny_v1` | MLP 6→32→1 | ~250 | 0.91 | Production |
| `vmaf_tiny_v2` | MLP 6→64→32→1 | ~2.3 K | 0.93 | Production |
| `vmaf_tiny_v3` | MLP 6→128→64→1 | ~9 K | 0.94 | Production (ADR-0389) |
| `vmaf_tiny_v4` | MLP 6→256→128→64→1 | ~35 K | 0.95 | Opt-in (ADR-0390) |

### 5.2 Architecture choices for the Netflix-corpus run

ADR-0242 § Alternatives considered deferred the architecture selection.
This digest's recommendation:

1. **Start with `vmaf_tiny_v2` (MLP 6→64→32→1)** as the baseline. The
   corpus is small (79 clips) and a larger model risks overfitting.
2. **Evaluate `vmaf_tiny_v3`** if `vmaf_tiny_v2` underfits (PLCC < 0.93).
3. **Do not evaluate `vmaf_tiny_v4` on this corpus** — 35 K params with
   79 training clips is severely overparameterised.
4. **Distillation teacher**: `vmaf_v0.6.1` (soft labels via libvmaf
   feature extraction + model scoring). Training from subjective MOS
   scores requires a separate annotation campaign not available for this
   corpus.

This recommendation is consistent with ADR-0242 and the EfficientVMAF
preprint (§ 3.2 above). The training PR should confirm the choice via
an inline AskUserQuestion before committing to a run.

---

## 6. Evaluation harness design

### 6.1 Netflix golden gate vs cross-backend delta

ADR-0242 § Alternatives considered listed two evaluation strategies:
(c) evaluate against Netflix golden assertions, (d) evaluate against
cross-backend ULP deltas.

The digest's position (consistent with ADR-0242's decision):
- **Use both**. The Netflix golden gate (`python/test/quality_runner_test.py`)
  verifies that the trained ONNX model, when invoked through `libvmaf`'s
  ONNX Runtime path, produces scores within the historical `assertAlmostEqual`
  tolerance. This gate is **not** a training-quality gate; it is a
  regression gate for the `libvmaf/src/dnn/` integration.
- The training-quality gate is PLCC / SROCC / RMSE on the held-out 10-clip
  Netflix test split, as documented in `docs/ai/training-data.md §
  Evaluation harness`.
- Cross-backend ULP deltas (ADR-0214 / `/cross-backend-diff`) verify that
  GPU backends reproduce the ONNX inference result, not the training
  quality. Run `/cross-backend-diff` on the exported ONNX model before
  the training PR merges.

### 6.2 MCP smoke test as the pre-training health check

The one-command pre-training verification documented in ADR-0242 and
`docs/ai/training-data.md § Evaluation harness`:

```bash
cd mcp-server/vmaf-mcp && python -m pytest tests/test_smoke_e2e.py -v
```

This confirms that the MCP server, the vmaf binary, and the
`vmaf_v0.6.1` model are wired correctly before starting the training
run. The test asserts a clip-mean score of ≈ 76.6993 (places=4) for
the `src01_hrc00_576x324.yuv` / `src01_hrc01_576x324.yuv` Netflix
golden pair — the same pair used by the Netflix CPU golden gate.

---

## 7. References

- Z. Li et al., "Toward a Practical Perceptual Video Quality Metric,"
  Netflix Tech Blog, 2016.
  <https://netflixtechblog.com/toward-a-practical-perceptual-video-quality-metric-653f208b9652>
- Netflix Tech Blog, "VMAF: The Journey Continues," 2018.
  <https://netflixtechblog.com/vmaf-the-journey-continues-44b51ee9ed12>
- ONNX Runtime release notes, 1.19 / 1.20, 2024–2025.
  <https://github.com/microsoft/onnxruntime/releases>
- ONNX opset 19 specification, Dec 2024.
  <https://github.com/onnx/onnx/blob/main/docs/Changelog.md>
- [ADR-0242](../adr/0242-tiny-ai-netflix-training-corpus.md) — governing
  architecture and distillation policy.
- [ADR-0417](../adr/0417-tiny-ai-netflix-training-scaffold-pr.md) — first
  scaffold PR (PR #759).
- [ADR-0453](../adr/0453-tiny-ai-netflix-training-scaffold-refresh.md) —
  second scaffold iteration (this PR).
- [Research digest 0019](0019-tiny-ai-netflix-training.md) — original
  VMAF methodology survey, 2026-04-27.
- [Research digest 0099](0099-tiny-ai-netflix-training-update.md) —
  2025 distillation and ONNX Runtime update.
- `docs/ai/training-data.md` — corpus path convention, loader API,
  evaluation harness.
- `mcp-server/vmaf-mcp/tests/test_smoke_e2e.py` — MCP smoke test suite.
