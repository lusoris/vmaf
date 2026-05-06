# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Unit tests for the fr_regressor_v3 trainer (ADR-0323).

Covers the canonical-6 + ENCODER_VOCAB_V3 schema validation in
``_load_corpus`` and a 1-epoch end-to-end smoke training run that
exports a real ONNX file. CPU-only, sub-second runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai" / "scripts"))

# pandas / numpy are training-only deps; skip if unavailable.
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from train_fr_regressor_v3 import (  # noqa: E402
    CANONICAL_6,
    CODEC_BLOCK_DIM,
    ENCODER_VOCAB_V3,
    N_ENCODERS_V3,
    SHIP_GATE_MEAN_PLCC,
    _load_corpus,
    build_argparser,
    export_onnx,
    fit_full_corpus,
    run_loso,
)


def _write_synthetic_corpus(path: Path) -> None:
    """16-row JSONL fixture exercising 4 of the 16 v3 vocab slots.

    Hits ``libx264`` (slot 0), ``h264_nvenc`` (slot 3), ``libsvtav1``
    (slot 13) and ``hevc_videotoolbox`` (slot 15) so the one-hot
    population covers both pre-v2 slots and v3-new slots.
    """
    rng = np.random.default_rng(42)
    encoders = ["libx264", "h264_nvenc", "libsvtav1", "hevc_videotoolbox"]
    cqs = [19, 25, 31, 37]
    rows = []
    for enc in encoders:
        for cq in cqs:
            rows.append(
                {
                    "src": f"src_{enc}",
                    "encoder": enc,
                    "cq": cq,
                    "frame_index": 0,
                    "vmaf": float(95.0 - (cq - 19) * 0.3 + rng.normal(0, 0.2)),
                    "adm2": float(0.95 - (cq - 19) * 0.005),
                    "vif_scale0": float(0.85 - (cq - 19) * 0.003),
                    "vif_scale1": float(0.92 - (cq - 19) * 0.002),
                    "vif_scale2": float(0.97 - (cq - 19) * 0.001),
                    "vif_scale3": float(0.99 - (cq - 19) * 0.0005),
                    "motion2": float(rng.uniform(0.0, 5.0)),
                }
            )
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_vocab_invariants() -> None:
    """ENCODER_VOCAB_V3 must be 16 slots and the codec block 18-D."""
    assert N_ENCODERS_V3 == 16
    assert len(ENCODER_VOCAB_V3) == 16
    assert CODEC_BLOCK_DIM == 18
    # Sanity: the v3-new slots from ADR-0302 are present.
    for new_slot in ("libsvtav1", "h264_videotoolbox", "hevc_videotoolbox"):
        assert new_slot in ENCODER_VOCAB_V3
    # Append-only invariant: indices 0..2 are the v2 head.
    assert ENCODER_VOCAB_V3[0] == "libx264"
    assert SHIP_GATE_MEAN_PLCC == 0.95


def test_load_corpus_returns_expected_shape(tmp_path: Path) -> None:
    """4-vocab-slot fixture loads with the right shape + one-hot indices."""
    corpus_path = tmp_path / "synth.jsonl"
    _write_synthetic_corpus(corpus_path)
    corpus = _load_corpus(corpus_path)

    assert corpus["target_col"] == "vmaf"
    assert corpus["source_col"] == "src"
    assert corpus["feature_cols"] == list(CANONICAL_6)
    assert len(corpus["codec_block_cols"]) == CODEC_BLOCK_DIM
    assert corpus["codec_block"].shape == (corpus["n_rows"], CODEC_BLOCK_DIM)
    assert corpus["n_rows"] == 4 * 4  # 4 encoders × 4 cqs

    onehot = corpus["codec_block"][:, :N_ENCODERS_V3]
    assert (onehot.sum(axis=1) == 1.0).all()
    # The four exercised slots all received non-zero column sums.
    for slot in ("libx264", "h264_nvenc", "libsvtav1", "hevc_videotoolbox"):
        idx = ENCODER_VOCAB_V3.index(slot)
        assert onehot[:, idx].sum() > 0


def test_load_corpus_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_corpus(tmp_path / "missing.jsonl")


def test_load_corpus_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"src": "s", "encoder": "libx264", "cq": 23}) + "\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _load_corpus(p)


def test_one_epoch_train_and_export(tmp_path: Path) -> None:
    """1-epoch fit_full_corpus + export_onnx round-trip on the synthetic fixture."""
    corpus_path = tmp_path / "synth.jsonl"
    _write_synthetic_corpus(corpus_path)
    corpus = _load_corpus(corpus_path)

    args = build_argparser().parse_args(
        [
            "--corpus",
            str(corpus_path),
            "--epochs",
            "1",
            "--batch-size",
            "8",
        ]
    )
    model, scaler = fit_full_corpus(corpus, args)
    assert "feature_mean" in scaler and len(scaler["feature_mean"]) == 6
    assert "feature_std" in scaler and len(scaler["feature_std"]) == 6

    onnx_path = tmp_path / "fr_regressor_v3.onnx"
    export_onnx(model, onnx_path)
    assert onnx_path.is_file()

    # ORT load + run round-trip with both inputs.
    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feats = np.zeros((2, 6), dtype=np.float32)
    codec = np.zeros((2, CODEC_BLOCK_DIM), dtype=np.float32)
    codec[:, 0] = 1.0  # libx264 one-hot
    out = sess.run(None, {"features": feats, "codec_block": codec})[0]
    assert out.shape == (2,)


def test_run_loso_smoke(tmp_path: Path) -> None:
    """LOSO summary on the synthetic fixture has the expected schema."""
    corpus_path = tmp_path / "synth.jsonl"
    _write_synthetic_corpus(corpus_path)
    corpus = _load_corpus(corpus_path)
    args = build_argparser().parse_args(
        [
            "--corpus",
            str(corpus_path),
            "--epochs",
            "1",
            "--batch-size",
            "8",
        ]
    )
    summary = run_loso(corpus, args)
    assert "mean_plcc" in summary
    assert "folds" in summary
    assert summary["n_folds"] == 4  # 4 distinct sources
    for f in summary["folds"]:
        assert {"held_out", "n_train", "n_val", "plcc", "srocc", "rmse"} <= set(f)
