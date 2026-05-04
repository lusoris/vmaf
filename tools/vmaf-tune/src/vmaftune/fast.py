# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase A.5 fast-path — proxy + Bayesian + GPU-verify recommend.

This module wires the *scaffold* of the ``vmaf-tune fast`` subcommand
documented in :doc:`/adr/0276-vmaf-tune-fast-path` and
:doc:`/research/0060-vmaf-tune-fast-path`. It deliberately does **not**
ship the production encoder + ONNX inference loop; it ships:

1. The :func:`fast_recommend` entry point with the production-shape
   signature so a follow-up PR can swap the proxy / verify
   implementations behind it without touching callers.
2. A ``smoke=True`` mode that synthesises 50 fake trials and runs
   Optuna over them. This validates the search-loop wiring end-to-end
   without needing a real source, real proxy weights, or real ffmpeg.
3. A clear separation between the four pluggable surfaces (encode
   sample, extract canonical-6 features, predict VMAF, verify with
   real VMAF) so each can be implemented independently.

The slow Phase A grid path
(:mod:`vmaftune.corpus`) stays canonical and untouched. ``fast`` is
opt-in.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Optuna is an optional dependency, gated behind the ``[fast]`` install
# extra. Importing it lazily lets the rest of vmaftune import cleanly on
# hosts that never run the fast path.
try:  # pragma: no cover - import-guarded
    import optuna  # type: ignore[import-not-found]

    _OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - import-guarded
    optuna = None  # type: ignore[assignment]
    _OPTUNA_AVAILABLE = False


# Default CRF search range for x264. Other codecs override via the
# adapter once the production loop wires the codec-adapter registry.
DEFAULT_CRF_LO: int = 10
DEFAULT_CRF_HI: int = 51

# Sample-chunk duration used for proxy-grade encodes in the production
# loop. Documented here so the follow-up PR can lift it from a single
# constant rather than scattering magic numbers.
SAMPLE_CHUNK_SECONDS: float = 5.0

# Smoke mode synthesises this many trials so the Optuna wiring is
# exercised end-to-end. Match the speedup-model entry in Research-0060.
SMOKE_N_TRIALS: int = 50


# ---------------------------------------------------------------------------
# Pluggable surfaces — production wiring lands in a follow-up PR.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrialSample:
    """One ``(crf, predicted_vmaf, predicted_kbps)`` proposal.

    Production: filled by encoding a 5-second chunk, extracting
    canonical-6 features, and running ``fr_regressor_v2`` over the
    feature vector + codec one-hot. Smoke mode: synthesised by a
    deterministic mock.
    """

    crf: int
    predicted_vmaf: float
    predicted_kbps: float


@dataclasses.dataclass(frozen=True)
class FastRecommendResult:
    """Outcome of one ``fast_recommend`` call."""

    encoder: str
    target_vmaf: float
    recommended_crf: int
    predicted_vmaf: float
    predicted_kbps: float
    n_trials: int
    smoke: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _smoke_predictor(crf: int) -> TrialSample:
    """Deterministic mock that mimics x264's monotone CRF→VMAF curve.

    Higher CRF → lower VMAF, lower bitrate. Shape is loosely calibrated
    against published x264 medium-preset behaviour on 1080p material:
    VMAF ≈ 100 at CRF 10, VMAF ≈ 50 at CRF 51, with a smooth taper.
    Used by the smoke-mode pipeline so the optimiser has a sensible
    objective without needing real weights.
    """
    crf_norm = (crf - DEFAULT_CRF_LO) / max(DEFAULT_CRF_HI - DEFAULT_CRF_LO, 1)
    # VMAF curve: smooth taper from ~99 at CRF 10 to ~52 at CRF 51.
    vmaf = 99.0 - 47.0 * (crf_norm**1.2)
    # Bitrate curve: exponential decay (typical for x264).
    kbps = 8000.0 * math.exp(-3.5 * crf_norm) + 80.0
    return TrialSample(crf=crf, predicted_vmaf=vmaf, predicted_kbps=kbps)


def _objective_factory(
    target_vmaf: float,
    predict: Callable[[int], TrialSample],
    crf_range: tuple[int, int],
) -> Callable[[Any], float]:
    """Build an Optuna objective that minimises ``|vmaf - target| + λ·kbps``.

    The bitrate term is weighted small relative to the quality term so
    the optimiser primarily hits the target; ties (multiple CRFs at the
    target) break toward the lower-bitrate option. This mirrors the
    typical "pick the lowest CRF that hits VMAF≥X" framing while
    staying differentiable enough for TPE.
    """
    crf_lo, crf_hi = crf_range
    bitrate_weight = 1.0e-4

    def _objective(trial: Any) -> float:
        crf = trial.suggest_int("crf", crf_lo, crf_hi)
        sample = predict(crf)
        trial.set_user_attr("predicted_vmaf", sample.predicted_vmaf)
        trial.set_user_attr("predicted_kbps", sample.predicted_kbps)
        quality_gap = abs(sample.predicted_vmaf - target_vmaf)
        return float(quality_gap + bitrate_weight * sample.predicted_kbps)

    return _objective


def _require_optuna() -> None:
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError(
            "vmaf-tune fast requires Optuna. Install with: "
            "pip install 'vmaf-tune[fast]'  (see docs/usage/vmaf-tune.md)."
        )


def fast_recommend(
    src: Path | None,
    target_vmaf: float,
    encoder: str = "libx264",
    time_budget_s: int = 300,  # noqa: ARG001 — production only
    crf_range: tuple[int, int] = (DEFAULT_CRF_LO, DEFAULT_CRF_HI),
    n_trials: int = SMOKE_N_TRIALS,
    smoke: bool = False,
    predictor: Callable[[int], TrialSample] | None = None,
) -> dict[str, Any]:
    """Return a fast-path CRF recommendation for ``src`` at ``target_vmaf``.

    Phase A.5 scaffold. Production wiring (encode sample → canonical-6
    extract → fr_regressor_v2 inference → GPU verify) lands in a
    follow-up PR; today this function:

    - In ``smoke=True`` mode, runs Optuna's TPE sampler over a synthetic
      x264-shaped CRF→VMAF curve (no encode, no ONNX, no FFmpeg).
    - In ``smoke=False`` mode, raises ``NotImplementedError`` with a
      pointer to the follow-up issue.

    Parameters
    ----------
    src
        Path to the source video. ``None`` only in smoke mode.
    target_vmaf
        Quality target on the standard VMAF [0, 100] scale.
    encoder
        Codec adapter name (currently only ``libx264`` in Phase A).
    time_budget_s
        Soft wall-clock budget. Currently advisory; production loop
        will enforce it via ``optuna.TrialPruned``.
    crf_range
        ``(lo, hi)`` inclusive CRF search range.
    n_trials
        Number of TPE trials to run. Defaults to
        :data:`SMOKE_N_TRIALS`.
    smoke
        Use the deterministic mock predictor (no ffmpeg / no ONNX).
    predictor
        Optional override for the ``crf -> TrialSample`` callable.
        Kept exposed so the production PR can inject the real
        encode + extract + ONNX-inference pipeline without touching
        this module's signature.

    Returns
    -------
    dict
        Serialisable result; see :class:`FastRecommendResult`.

    Raises
    ------
    RuntimeError
        Optuna missing (install ``vmaf-tune[fast]``).
    NotImplementedError
        ``smoke=False`` and no ``predictor`` injected (production
        wiring is a follow-up PR).
    """
    _require_optuna()

    if not smoke and predictor is None:
        raise NotImplementedError(
            "vmaf-tune fast production loop is scaffold-only in this PR. "
            "Pass smoke=True for the demonstration pipeline, or inject a "
            "predictor=Callable[[int], TrialSample] to drive a custom "
            "search. See ADR-0276 'What is deferred to follow-up PRs'."
        )
    if smoke and src is not None:
        # Not an error — we just want to make the contract explicit:
        # smoke mode does not touch the source.
        pass

    chosen = predictor or _smoke_predictor
    objective = _objective_factory(target_vmaf, chosen, crf_range)

    # Suppress Optuna's default INFO-level chatter; the CLI is the
    # right place to surface progress (follow-up PR).
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    recommended_crf = int(best.params["crf"])
    predicted_vmaf = float(best.user_attrs.get("predicted_vmaf", float("nan")))
    predicted_kbps = float(best.user_attrs.get("predicted_kbps", float("nan")))

    notes = (
        "smoke mode — synthetic predictor; no ffmpeg / ONNX / GPU was used. "
        "See ADR-0276 + Research-0060 for the production roadmap."
        if smoke
        else "custom predictor injected; production loop deferred to follow-up PR."
    )

    result = FastRecommendResult(
        encoder=encoder,
        target_vmaf=float(target_vmaf),
        recommended_crf=recommended_crf,
        predicted_vmaf=predicted_vmaf,
        predicted_kbps=predicted_kbps,
        n_trials=n_trials,
        smoke=smoke,
        notes=notes,
    )
    return result.to_dict()


__all__ = [
    "DEFAULT_CRF_HI",
    "DEFAULT_CRF_LO",
    "SAMPLE_CHUNK_SECONDS",
    "SMOKE_N_TRIALS",
    "FastRecommendResult",
    "TrialSample",
    "fast_recommend",
]
