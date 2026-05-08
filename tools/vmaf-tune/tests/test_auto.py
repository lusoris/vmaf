# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase F.1 sequential-scaffold tests for ``vmaf-tune auto`` (ADR-0325).

These tests exercise the **scaffold contract** only:

* Sequential composition — every per-phase seam is invoked, in the
  order ADR-0325 §Decision tree pins.
* Smoke mode — synthetic plan generation without ffmpeg / vmaf /
  ONNX / GPU.
* Error propagation — a failing seam surfaces as an exception (no
  silent swallows).
* Output schema — ``Plan.to_dict()`` round-trips through JSON.

Short-circuits, FALL_BACK escalation, and recipe overrides are
**out of scope for F.1** — those land in F.2 / F.3 / F.4 and ship
their own tests then.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install (mirrors test_fast.py).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.auto import (  # noqa: E402
    PHASE_D_DURATION_GATE_S,
    PHASE_D_SHOT_VARIANCE_GATE,
    VERDICT_FALL_BACK,
    VERDICT_PASS,
    Meta,
    Plan,
    PlanEntry,
    Rung,
    Verdict,
    auto,
)

# ---------------------------------------------------------------------------
# Smoke-mode contract.
# ---------------------------------------------------------------------------


def test_smoke_returns_plan_with_winner() -> None:
    """``--smoke`` returns a fully-populated Plan without external IO."""
    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=92.0,
        max_budget_kbps=10000.0,
        allow_codecs=("libx264",),
        smoke=True,
    )
    assert isinstance(plan, Plan)
    assert plan.smoke is True
    assert plan.is_hdr is False
    assert plan.target_vmaf == 92.0
    assert plan.allow_codecs == ("libx264",)
    assert plan.winner is not None
    assert plan.winner.codec == "libx264"
    assert plan.winner.verdict.predicted_vmaf >= 92.0
    assert plan.notes.startswith("smoke mode")


def test_smoke_serialises_to_json() -> None:
    """``Plan.to_dict()`` round-trips through ``json.dumps``."""
    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264", "libx265"),
        smoke=True,
    )
    rendered = json.dumps(plan.to_dict(), sort_keys=True)
    parsed = json.loads(rendered)
    assert parsed["smoke"] is True
    assert parsed["target_vmaf"] == 90.0
    assert parsed["allow_codecs"] == ["libx264", "libx265"]
    assert isinstance(parsed["candidates"], list)
    assert len(parsed["candidates"]) == 2  # one rung × two codecs
    assert parsed["winner"] is not None


def test_smoke_no_winner_when_budget_exceeded() -> None:
    """A budget below the synthetic floor produces no winner."""
    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=10.0,  # below every smoke codec's predicted_kbps
        allow_codecs=("libx264",),
        smoke=True,
    )
    assert plan.winner is None
    assert plan.candidates  # candidates were still enumerated


# ---------------------------------------------------------------------------
# Sequential composition — assert every seam fires in the right order.
# ---------------------------------------------------------------------------


def _make_meta(**overrides: object) -> Meta:
    base = dict(
        width=1920,
        height=1080,
        framerate=24.0,
        duration_s=120.0,
        shot_variance=0.05,
        content_class="live_action",
    )
    base.update(overrides)
    return Meta(**base)  # type: ignore[arg-type]


def test_sequential_phase_call_order() -> None:
    """Probe → HDR → rungs → shortlist → predict → pareto → realise.

    F.1 must invoke every seam in the ADR-0325 order. We thread a
    shared ``calls`` list through each seam and assert the prefix.
    """
    calls: list[str] = []

    def _probe(src: Path) -> Meta:
        calls.append("probe")
        return _make_meta()

    def _hdr(src: Path, meta: Meta) -> bool:
        calls.append("hdr")
        return False

    def _rungs(meta: Meta):
        calls.append("rungs")
        return [Rung(width=meta.width, height=meta.height, label="1080p")]

    def _shortlist(allow, meta):
        calls.append("shortlist")
        return list(allow)

    def _predict(rung, codec, target, meta):
        calls.append(f"predict:{codec}")
        return Verdict(crf=23, predicted_vmaf=target + 1.0, predicted_kbps=4000.0)

    def _pareto(plan, target, budget):
        calls.append("pareto")
        return plan[0] if plan else None

    def _realise(winner, is_hdr):
        calls.append("realise")
        return winner

    auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264", "libx265"),
        probe=_probe,
        hdr_detect=_hdr,
        rungs=_rungs,
        shortlist=_shortlist,
        predict=_predict,
        pareto_pick=_pareto,
        realise=_realise,
    )
    # Prefix order: probe, hdr, rungs, shortlist, then per-cell predict,
    # then pareto, then realise.
    assert calls[0] == "probe"
    assert calls[1] == "hdr"
    assert calls[2] == "rungs"
    assert calls[3] == "shortlist"
    assert calls[4:6] == ["predict:libx264", "predict:libx265"]
    assert calls[6] == "pareto"
    assert calls[7] == "realise"


def test_predict_invoked_per_rung_codec_cell() -> None:
    """``predict`` runs len(rungs) × len(codecs) times."""
    invocations: list[tuple[int, str]] = []

    def _rungs(meta):
        return [
            Rung(width=1920, height=1080, label="1080p"),
            Rung(width=1280, height=720, label="720p"),
        ]

    def _predict(rung, codec, target, meta):
        invocations.append((rung.height, codec))
        return Verdict(crf=23, predicted_vmaf=target + 0.5, predicted_kbps=3000.0)

    auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264", "libx265", "libsvtav1"),
        probe=lambda _src: _make_meta(),
        hdr_detect=lambda _src, _m: False,
        rungs=_rungs,
        predict=_predict,
    )
    assert len(invocations) == 6  # 2 rungs × 3 codecs
    assert (1080, "libx264") in invocations
    assert (720, "libsvtav1") in invocations


def test_phase_d_gate_skipped_for_short_source() -> None:
    """Per-shot refine does NOT run for sources below the duration gate."""
    refined: list[PlanEntry] = []

    def _refine(entry: PlanEntry) -> PlanEntry:
        refined.append(entry)
        return entry

    short_meta = _make_meta(duration_s=10.0, shot_variance=0.5)  # under gate
    auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264",),
        probe=lambda _src: short_meta,
        hdr_detect=lambda _src, _m: False,
        predict=lambda r, c, t, m: Verdict(crf=23, predicted_vmaf=t + 0.5, predicted_kbps=2000.0),
        per_shot_refine=_refine,
    )
    assert refined == []


def test_phase_d_gate_runs_for_long_high_variance_source() -> None:
    """Per-shot refine runs when both duration AND variance exceed the gate."""
    refined: list[PlanEntry] = []

    def _refine(entry: PlanEntry) -> PlanEntry:
        refined.append(entry)
        return entry

    long_meta = _make_meta(
        duration_s=PHASE_D_DURATION_GATE_S + 60.0,
        shot_variance=PHASE_D_SHOT_VARIANCE_GATE + 0.05,
    )
    auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264", "libx265"),
        probe=lambda _src: long_meta,
        hdr_detect=lambda _src, _m: False,
        predict=lambda r, c, t, m: Verdict(crf=23, predicted_vmaf=t + 0.5, predicted_kbps=2000.0),
        per_shot_refine=_refine,
    )
    assert len(refined) == 2  # one entry per (rung, codec) cell


def test_saliency_gate_only_for_animation_or_screen_content() -> None:
    """Saliency apply skips ``live_action``; runs on ``animation`` /
    ``screen_content``."""
    applied: list[PlanEntry] = []

    def _saliency(entry: PlanEntry) -> PlanEntry:
        applied.append(entry)
        return entry

    # live_action — no saliency.
    auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264",),
        probe=lambda _src: _make_meta(content_class="live_action"),
        hdr_detect=lambda _src, _m: False,
        predict=lambda r, c, t, m: Verdict(crf=23, predicted_vmaf=t + 0.5, predicted_kbps=2000.0),
        saliency_apply=_saliency,
    )
    assert applied == []

    # animation — saliency runs.
    auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264",),
        probe=lambda _src: _make_meta(content_class="animation"),
        hdr_detect=lambda _src, _m: False,
        predict=lambda r, c, t, m: Verdict(crf=23, predicted_vmaf=t + 0.5, predicted_kbps=2000.0),
        saliency_apply=_saliency,
    )
    assert len(applied) == 1


def test_fall_back_verdict_is_propagated_unchanged() -> None:
    """F.1 contract: FALL_BACK does NOT trigger escalation (that's F.3)."""

    def _predict(rung, codec, target, meta):
        return Verdict(
            crf=18,
            predicted_vmaf=target + 0.1,
            verdict=VERDICT_FALL_BACK,
            predicted_kbps=5000.0,
        )

    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264",),
        probe=lambda _src: _make_meta(),
        hdr_detect=lambda _src, _m: False,
        predict=_predict,
    )
    assert plan.candidates[0].verdict.verdict == VERDICT_FALL_BACK


def test_pareto_pick_respects_budget_constraint() -> None:
    """Default Pareto pick excludes entries above ``max_budget_kbps``."""

    def _predict(rung, codec, target, meta):
        # x264 hits target cheaply; x265 hits target but exceeds budget.
        kbps = 1500.0 if codec == "libx264" else 50000.0
        return Verdict(
            crf=23,
            predicted_vmaf=target + 0.5,
            verdict=VERDICT_PASS,
            predicted_kbps=kbps,
        )

    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=2000.0,
        allow_codecs=("libx264", "libx265"),
        probe=lambda _src: _make_meta(),
        hdr_detect=lambda _src, _m: False,
        predict=_predict,
    )
    assert plan.winner is not None
    assert plan.winner.codec == "libx264"


def test_pareto_pick_prefers_lowest_bitrate() -> None:
    """When multiple candidates pass, lowest predicted_kbps wins."""

    def _predict(rung, codec, target, meta):
        kbps = {"libx264": 4000.0, "libx265": 2500.0, "libsvtav1": 3200.0}[codec]
        return Verdict(crf=23, predicted_vmaf=target + 1.0, predicted_kbps=kbps)

    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264", "libx265", "libsvtav1"),
        probe=lambda _src: _make_meta(),
        hdr_detect=lambda _src, _m: False,
        predict=_predict,
    )
    assert plan.winner is not None
    assert plan.winner.codec == "libx265"


def test_error_in_seam_propagates() -> None:
    """A failing seam surfaces as an exception — no silent swallow."""

    class _Boom(RuntimeError):
        pass

    def _bad_predict(rung, codec, target, meta):
        raise _Boom("predictor exploded")

    with pytest.raises(_Boom):
        auto(
            src=Path("/dev/null"),
            target_vmaf=90.0,
            max_budget_kbps=0.0,
            allow_codecs=("libx264",),
            probe=lambda _src: _make_meta(),
            hdr_detect=lambda _src, _m: False,
            predict=_bad_predict,
        )


def test_default_probe_raises_without_seam() -> None:
    """Outside smoke mode, missing ``probe`` is a loud failure."""
    with pytest.raises(NotImplementedError):
        auto(
            src=Path("/dev/null"),
            target_vmaf=90.0,
            max_budget_kbps=0.0,
            allow_codecs=("libx264",),
            smoke=False,
        )


def test_default_predict_raises_without_seam() -> None:
    """Probe seam alone is not enough — missing ``predict`` also raises."""
    with pytest.raises(NotImplementedError):
        auto(
            src=Path("/dev/null"),
            target_vmaf=90.0,
            max_budget_kbps=0.0,
            allow_codecs=("libx264",),
            smoke=False,
            probe=lambda _src: _make_meta(),
            hdr_detect=lambda _src, _m: False,
        )


def test_hdr_flag_threaded_through_realise() -> None:
    """``is_hdr`` from the HDR seam reaches ``realise`` and the Plan."""
    seen_hdr: list[bool] = []

    def _realise(winner, is_hdr):
        seen_hdr.append(is_hdr)
        return winner

    plan = auto(
        src=Path("/dev/null"),
        target_vmaf=90.0,
        max_budget_kbps=0.0,
        allow_codecs=("libx264",),
        probe=lambda _src: _make_meta(),
        hdr_detect=lambda _src, _m: True,
        predict=lambda r, c, t, m: Verdict(crf=23, predicted_vmaf=t + 0.5, predicted_kbps=2000.0),
        realise=_realise,
    )
    assert plan.is_hdr is True
    assert seen_hdr == [True]
