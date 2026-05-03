# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase A.5 fast-path smoke tests (ADR-0276).

Validates the Optuna-driven search loop end-to-end without needing
ffmpeg, ONNX Runtime, or a GPU. Production wiring (real encode +
fr_regressor_v2 inference + GPU verify) is gated behind a
``predictor=`` injection in :func:`vmaftune.fast.fast_recommend` and
covered by follow-up integration tests once the corpus exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

optuna = pytest.importorskip("optuna")  # noqa: F841 — gate the whole module

from vmaftune.fast import (  # noqa: E402
    DEFAULT_CRF_HI,
    DEFAULT_CRF_LO,
    SMOKE_N_TRIALS,
    TrialSample,
    fast_recommend,
)


def test_smoke_recommendation_hits_target_within_tolerance() -> None:
    """Synthetic predictor + Optuna TPE should land within ≈1 VMAF of the target."""
    result = fast_recommend(src=None, target_vmaf=92.0, smoke=True)
    assert result["smoke"] is True
    assert result["encoder"] == "libx264"
    assert result["n_trials"] == SMOKE_N_TRIALS
    assert DEFAULT_CRF_LO <= result["recommended_crf"] <= DEFAULT_CRF_HI
    assert abs(result["predicted_vmaf"] - 92.0) < 1.5


def test_smoke_low_target_picks_higher_crf() -> None:
    """Lower VMAF target should map to higher CRF on the synthetic curve."""
    high_q = fast_recommend(src=None, target_vmaf=95.0, smoke=True)
    low_q = fast_recommend(src=None, target_vmaf=70.0, smoke=True)
    assert low_q["recommended_crf"] > high_q["recommended_crf"]
    assert low_q["predicted_kbps"] < high_q["predicted_kbps"]


def test_production_loop_raises_until_implemented() -> None:
    """Calling ``fast_recommend`` without ``smoke`` and without an injected
    predictor must raise NotImplementedError — production wiring lands in a
    follow-up PR (ADR-0276 'What is deferred')."""
    with pytest.raises(NotImplementedError):
        fast_recommend(src=Path("any.mp4"), target_vmaf=92.0, smoke=False)


def test_predictor_injection_drives_search() -> None:
    """A custom predictor closes the loop without needing smoke mode."""
    calls: list[int] = []

    def _flat_predictor(crf: int) -> TrialSample:
        calls.append(crf)
        # Constant 88 VMAF, bitrate scales linearly with CRF inversely.
        return TrialSample(crf=crf, predicted_vmaf=88.0, predicted_kbps=float(60 - crf))

    result = fast_recommend(
        src=Path("any.mp4"),
        target_vmaf=88.0,
        smoke=False,
        n_trials=10,
        predictor=_flat_predictor,
    )
    # Optuna must have called the predictor at least once.
    assert len(calls) > 0
    assert result["predicted_vmaf"] == 88.0
    assert DEFAULT_CRF_LO <= result["recommended_crf"] <= DEFAULT_CRF_HI


def test_crf_range_is_respected() -> None:
    result = fast_recommend(
        src=None,
        target_vmaf=85.0,
        smoke=True,
        crf_range=(20, 30),
        n_trials=20,
    )
    assert 20 <= result["recommended_crf"] <= 30
