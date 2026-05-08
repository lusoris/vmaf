# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase F.1 ``vmaf-tune auto`` — sequential decision-tree scaffold.

This module implements the **F.1 scaffold** of Phase F as decided in
ADR-0325 (and in flight via PR #451 at the time of writing).

F.1 implements the 22-line decision-tree pseudocode from ADR-0325
**verbatim**, with **no smart routing yet**:

* No short-circuits (those land in F.2 — single-rung ladder, codec
  pinned, GOSPEL predictor, photographic content, SDR, sample-clip
  propagation, low-variance / short-source Phase D skip).
* No confidence-aware fallbacks (those land in F.3 — per-cell
  escalation to ``recommend.coarse_to_fine`` on FALL_BACK; ROI /
  saliency-binary missing degrades to a warning).
* No per-content-type recipe overrides (those land in F.4 —
  animation / live-action / screen-content auto-detect via
  TransNet V2 shot-cut histogram + heuristic fallback).

The flow is a **sequential composition** of the per-phase entry
points the rest of ``vmaftune`` already exposes. Each step is
exposed via an injectable callable so unit tests can drive the
sequence end-to-end without spawning ffmpeg / vmaf / ONNX:

1. ``probe(src) -> Meta`` — content metadata (resolution, duration,
   shot variance).
2. ``detect_hdr(meta) -> HdrInfo | None`` — wraps :mod:`vmaftune.hdr`.
3. ``ladder.candidate_rungs(meta) -> [Rung]`` — wraps
   :mod:`vmaftune.ladder`.
4. ``compare.shortlist(allow_codecs, meta) -> [str]`` — wraps
   :mod:`vmaftune.compare`.
5. ``predict.crf_for_target(rung, codec, target_vmaf, meta)`` —
   wraps :mod:`vmaftune.predictor` + :mod:`vmaftune.predictor_validate`.
6. ``tune_per_shot.refine(plan_entry)`` — wraps
   :mod:`vmaftune.per_shot`.
7. ``recommend_saliency.maybe_apply(plan_entry)`` — wraps
   :mod:`vmaftune.saliency`.
8. ``pick_pareto(plan, target_vmaf, max_budget_kbps) -> entry`` —
   in-tree helper; will be re-homed in :mod:`vmaftune.ladder` once
   F.2 lands.
9. ``realise(winner, hdr) -> Plan`` — emits the final
   :class:`Plan` as a serialisable artifact.

Smoke mode (``--smoke`` on the CLI; ``smoke=True`` in the API)
short-circuits **the whole tree** to a deterministic dry-run that
returns a synthetic plan without invoking real ffmpeg / vmaf
scoring. The CI smoke gate matches the ADR-0276 / ADR-0276 fast-path
pattern: every seam is exercised, no external tooling is touched.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Plan / verdict dataclasses.
# ---------------------------------------------------------------------------


# Predictor verdicts mirror :class:`vmaftune.predictor_validate.Verdict`
# without importing the heavy module at import time. F.1 only ever
# observes them as opaque strings — F.3 escalates ``FALL_BACK`` to
# ``recommend.coarse_to_fine``.
VERDICT_PASS: str = "PASS"
VERDICT_BIAS: str = "BIAS"
VERDICT_FALL_BACK: str = "FALL_BACK"


# Phase D placeholders (also baked into ADR-0325 §Decision tree).
# The 5-min duration / 0.15 shot-variance gate is a placeholder
# until F.3 fits real thresholds against corpus data.
PHASE_D_DURATION_GATE_S: float = 300.0
PHASE_D_SHOT_VARIANCE_GATE: float = 0.15

# Saliency-gate content classes (Bucket #2 / ADR-0287). Photographic
# content is intentionally absent — F.2 will short-circuit it out.
SALIENCY_CONTENT_CLASSES: frozenset[str] = frozenset({"animation", "screen_content"})


@dataclasses.dataclass(frozen=True)
class Meta:
    """Probed source metadata.

    Field set is **deliberately minimal** for F.1 — the smart-routing
    short-circuits in F.2 / F.3 / F.4 will widen this when they need
    GOSPEL-derived shot statistics, scene-cut histograms, etc.
    """

    width: int
    height: int
    framerate: float
    duration_s: float
    shot_variance: float = 0.0
    content_class: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Rung:
    """One candidate ladder rung — resolution + label."""

    width: int
    height: int
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Predictor verdict for one (rung, codec) cell.

    F.1 propagates ``FALL_BACK`` through the plan unchanged; F.3 will
    intercept it and escalate to ``recommend.coarse_to_fine``.
    """

    crf: int
    predicted_vmaf: float
    verdict: str = VERDICT_PASS
    predicted_kbps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PlanEntry:
    """One ``(rung, codec, verdict)`` tuple in the ``auto`` plan."""

    rung: Rung
    codec: str
    verdict: Verdict
    per_shot_refined: bool = False
    saliency_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung.to_dict(),
            "codec": self.codec,
            "verdict": self.verdict.to_dict(),
            "per_shot_refined": self.per_shot_refined,
            "saliency_applied": self.saliency_applied,
        }


@dataclasses.dataclass(frozen=True)
class Plan:
    """Realised plan returned by :func:`auto`."""

    source: str
    target_vmaf: float
    max_budget_kbps: float
    allow_codecs: tuple[str, ...]
    is_hdr: bool
    meta: Meta
    candidates: tuple[PlanEntry, ...]
    winner: PlanEntry | None
    smoke: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target_vmaf": self.target_vmaf,
            "max_budget_kbps": self.max_budget_kbps,
            "allow_codecs": list(self.allow_codecs),
            "is_hdr": self.is_hdr,
            "meta": self.meta.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "winner": self.winner.to_dict() if self.winner is not None else None,
            "smoke": self.smoke,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Injectable seam typedefs.
# ---------------------------------------------------------------------------


ProbeFn = Callable[[Path], Meta]
HdrDetectFn = Callable[[Path, Meta], bool]
RungFn = Callable[[Meta], Sequence[Rung]]
ShortlistFn = Callable[[Sequence[str], Meta], Sequence[str]]
PredictFn = Callable[[Rung, str, float, Meta], Verdict]
PerShotRefineFn = Callable[[PlanEntry], PlanEntry]
SaliencyApplyFn = Callable[[PlanEntry], PlanEntry]
ParetoPickFn = Callable[[Sequence[PlanEntry], float, float], PlanEntry | None]
RealiseFn = Callable[[PlanEntry, bool], PlanEntry]


# ---------------------------------------------------------------------------
# Default seams — production-shape adapters around the existing modules.
# Each adapter is intentionally thin: the F.1 scaffold contract is "wire
# the sequence", not "implement the per-phase production logic". The
# heavyweight wiring (real ffmpeg, real ONNX, real per-shot detection)
# lands incrementally in F.2 / F.3 / F.4.
# ---------------------------------------------------------------------------


def _default_probe(src: Path) -> Meta:
    """Default ``probe`` adapter — F.1 contract: caller injects.

    Production probing reuses ffprobe via :mod:`vmaftune.predictor_features`
    and TransNet V2 via :mod:`vmaftune.per_shot`. Wiring lands in F.3
    when the shot-variance gate becomes load-bearing; for now F.1 raises
    so a missing seam is a loud failure rather than a silent default.
    """
    raise NotImplementedError(
        "auto() needs a probe seam in F.1. Pass `probe=` explicitly or "
        "use --smoke. The default ffprobe/TransNet wiring lands in F.3."
    )


def _default_hdr_detect(src: Path, meta: Meta) -> bool:
    """Default HDR detection — wraps :func:`vmaftune.hdr.detect_hdr`."""
    from vmaftune.hdr import detect_hdr  # noqa: PLC0415  (deliberately lazy)

    info = detect_hdr(src)
    return info is not None


def _default_rungs(meta: Meta) -> Sequence[Rung]:
    """Default candidate rungs — single rung below 4K, full ladder at 4K+.

    Mirrors the ADR-0325 pseudocode line ``rungs = [meta.resolution]
    if meta.height < 2160 else ladder.candidate_rungs(meta)``. F.1
    short-circuits to a single rung at the source resolution; F.2 wires
    :func:`vmaftune.ladder.build_ladder` for the 4K+ branch.
    """
    return [Rung(width=meta.width, height=meta.height, label=f"{meta.height}p")]


def _default_shortlist(allow_codecs: Sequence[str], meta: Meta) -> Sequence[str]:
    """Default codec shortlist — F.1 returns ``allow_codecs`` unchanged.

    The ADR-0325 tree narrows to ``[user_pin]`` when one codec is
    pinned, or ``compare.shortlist(allow_codecs, meta)`` otherwise.
    F.2 will wire :func:`vmaftune.compare.compare_codecs` into a
    proper shortlist; F.1 keeps it pass-through so the sequential
    scaffold has a stable contract for tests to mock.
    """
    return list(allow_codecs)


def _default_predict(rung: Rung, codec: str, target_vmaf: float, meta: Meta) -> Verdict:
    """Default predictor seam — F.1 contract: caller injects.

    Production wiring threads :class:`vmaftune.predictor.Predictor` +
    :func:`vmaftune.predictor_validate.validate_predictor` through. F.1
    raises if no seam is supplied so missing wiring is loud.
    """
    raise NotImplementedError(
        "auto() needs a predict seam in F.1. Pass `predict=` explicitly or "
        "use --smoke. Production wiring lives in F.2 (GOSPEL short-circuit) "
        "and F.3 (FALL_BACK escalation)."
    )


def _default_per_shot_refine(entry: PlanEntry) -> PlanEntry:
    """Default per-shot refine — F.1 marks the entry without re-encoding."""
    return dataclasses.replace(entry, per_shot_refined=True)


def _default_saliency_apply(entry: PlanEntry) -> PlanEntry:
    """Default saliency apply — F.1 marks the entry without QP map work."""
    return dataclasses.replace(entry, saliency_applied=True)


def _default_pareto_pick(
    plan: Sequence[PlanEntry], target_vmaf: float, max_budget_kbps: float
) -> PlanEntry | None:
    """Pareto pick — smallest predicted_kbps with vmaf >= target.

    F.1 keeps this purely declarative: filter entries whose
    ``predicted_vmaf >= target_vmaf`` and whose
    ``predicted_kbps <= max_budget_kbps``, then pick the smallest
    bitrate (ties break on highest vmaf, then on first occurrence).
    F.2 / F.3 will swap this for a hull-based pick once the ladder
    seams produce real bitrate envelopes.
    """
    eligible: list[PlanEntry] = []
    for entry in plan:
        v = entry.verdict
        if v.predicted_vmaf < target_vmaf:
            continue
        if max_budget_kbps > 0 and v.predicted_kbps > max_budget_kbps:
            continue
        eligible.append(entry)
    if not eligible:
        return None

    def _key(e: PlanEntry) -> tuple[float, float]:
        # Lower bitrate first; tie-break on higher vmaf (negate for ascending).
        return (e.verdict.predicted_kbps, -e.verdict.predicted_vmaf)

    eligible.sort(key=_key)
    return eligible[0]


def _default_realise(winner: PlanEntry, is_hdr: bool) -> PlanEntry:
    """Realise — F.1 returns the winner unchanged.

    The ADR-0325 pseudocode threads the HDR flag through to
    :mod:`vmaftune.hdr.hdr_codec_args`; F.2 / F.3 will wire that
    through the encode dispatcher. F.1's contract is simply "the
    sequence terminates with a well-formed PlanEntry".
    """
    return winner


# ---------------------------------------------------------------------------
# Smoke-mode synthetic seams.
# ---------------------------------------------------------------------------


SMOKE_META: Meta = Meta(
    width=1920,
    height=1080,
    framerate=24.0,
    duration_s=120.0,
    shot_variance=0.05,
    content_class="live_action",
)


def _smoke_probe(src: Path) -> Meta:  # noqa: ARG001 — smoke seam
    return SMOKE_META


def _smoke_hdr(src: Path, meta: Meta) -> bool:  # noqa: ARG001 — smoke seam
    return False


def _smoke_predict(
    rung: Rung, codec: str, target_vmaf: float, meta: Meta
) -> Verdict:  # noqa: ARG001 — smoke seam
    # Synthetic predictor: pretend we hit the target at CRF 23 with a
    # codec-dependent bitrate. The shape matches the ADR-0276 fast-path
    # smoke curve so the two scaffolds share an intuition.
    base_kbps = {"libx264": 4500.0, "libx265": 3200.0, "libsvtav1": 2700.0}.get(codec, 5000.0)
    return Verdict(
        crf=23,
        predicted_vmaf=target_vmaf + 0.25,
        verdict=VERDICT_PASS,
        predicted_kbps=base_kbps,
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def auto(  # noqa: PLR0913 — F.1 contract: explicit seams beat hidden globals.
    src: Path,
    target_vmaf: float,
    max_budget_kbps: float,
    allow_codecs: Sequence[str],
    *,
    smoke: bool = False,
    probe: ProbeFn | None = None,
    hdr_detect: HdrDetectFn | None = None,
    rungs: RungFn | None = None,
    shortlist: ShortlistFn | None = None,
    predict: PredictFn | None = None,
    per_shot_refine: PerShotRefineFn | None = None,
    saliency_apply: SaliencyApplyFn | None = None,
    pareto_pick: ParetoPickFn | None = None,
    realise: RealiseFn | None = None,
) -> Plan:
    """Run the F.1 sequential decision-tree composition.

    Mirrors the ADR-0325 22-line pseudocode **verbatim** (no
    short-circuits, no fallbacks, no recipe overrides):

    .. code-block:: text

        meta = probe(src)
        is_hdr = detect_hdr(meta)
        rungs = [meta.resolution] if meta.height < 2160
                else ladder.candidate_rungs(meta)
        codecs = (allow_codecs if len==1
                  else [user_pin] if user_pinned
                  else compare.shortlist(allow_codecs, meta))
        plan = []
        for rung, codec in (rungs × codecs):
            v = predict.crf_for_target(rung, codec, target_vmaf, meta)
            # F.1: no FALL_BACK escalation; F.3 adds it.
            plan.append((rung, codec, v))
        if duration > 5min and shot_variance > 0.15:
            plan = [tune_per_shot.refine(p) for p in plan]
        if meta.content_class in {animation, screen_content}:
            plan = [recommend_saliency.maybe_apply(p) for p in plan]
        winner = pick_pareto(plan, target_vmaf, max_budget_kbps)
        return realise(winner, hdr=is_hdr)

    Parameters
    ----------
    src
        Source video path. ``Path("/dev/null")`` is fine in smoke
        mode — no IO happens.
    target_vmaf
        VMAF target on the [0, 100] scale.
    max_budget_kbps
        Bitrate ceiling for the Pareto pick. ``0`` disables the
        budget constraint.
    allow_codecs
        Codec adapters the user permits (e.g.
        ``["libx264", "libx265", "libsvtav1"]``).
    smoke
        When ``True``, every seam is replaced by a synthetic
        deterministic version — no ffmpeg, no vmaf, no ONNX,
        no GPU.
    probe, hdr_detect, rungs, shortlist, predict, per_shot_refine,
    saliency_apply, pareto_pick, realise
        Injectable seams. Defaults forward to the per-phase modules
        already shipped under ``vmaftune.*``; smoke mode swaps
        ``probe`` / ``hdr_detect`` / ``predict`` for synthetic stubs.
    """
    if smoke:
        probe = probe or _smoke_probe
        hdr_detect = hdr_detect or _smoke_hdr
        predict = predict or _smoke_predict
    probe_fn = probe or _default_probe
    hdr_fn = hdr_detect or _default_hdr_detect
    rungs_fn = rungs or _default_rungs
    shortlist_fn = shortlist or _default_shortlist
    predict_fn = predict or _default_predict
    refine_fn = per_shot_refine or _default_per_shot_refine
    saliency_fn = saliency_apply or _default_saliency_apply
    pareto_fn = pareto_pick or _default_pareto_pick
    realise_fn = realise or _default_realise

    # 1. Probe + HDR detection. ADR-0325 line 1-2.
    meta = probe_fn(src)
    is_hdr = hdr_fn(src, meta)

    # 2. Candidate rungs. ADR-0325 line 3-4.
    rung_list: list[Rung] = list(rungs_fn(meta))

    # 3. Codec shortlist. ADR-0325 line 5-7. F.1 does not implement
    # the user-pin branch separately — a single-element ``allow_codecs``
    # already collapses the shortlist to that codec, which matches
    # the pseudocode's intent without F.2's explicit short-circuit.
    codec_list: list[str] = list(shortlist_fn(list(allow_codecs), meta))

    # 4. Per-cell predict. ADR-0325 line 8-12. F.1 propagates
    # FALL_BACK unchanged; F.3 will intercept it.
    candidates: list[PlanEntry] = []
    for rung in rung_list:
        for codec in codec_list:
            verdict = predict_fn(rung, codec, target_vmaf, meta)
            candidates.append(PlanEntry(rung=rung, codec=codec, verdict=verdict))

    # 5. Phase D refine gate. ADR-0325 line 13-14.
    if (
        meta.duration_s > PHASE_D_DURATION_GATE_S
        and meta.shot_variance > PHASE_D_SHOT_VARIANCE_GATE
    ):
        candidates = [refine_fn(c) for c in candidates]

    # 6. Saliency gate. ADR-0325 line 15-16.
    if meta.content_class in SALIENCY_CONTENT_CLASSES:
        candidates = [saliency_fn(c) for c in candidates]

    # 7. Pareto pick. ADR-0325 line 17.
    winner = pareto_fn(candidates, target_vmaf, max_budget_kbps)

    # 8. Realise. ADR-0325 line 18.
    realised = realise_fn(winner, is_hdr) if winner is not None else None

    notes = (
        "F.1 scaffold — sequential composition; no short-circuits, no "
        "FALL_BACK escalation, no recipe overrides (those land in F.2 / "
        "F.3 / F.4 per ADR-0325)."
    )
    if smoke:
        notes = "smoke mode — synthetic probe/hdr/predict; no ffmpeg / vmaf. " + notes

    return Plan(
        source=str(src),
        target_vmaf=float(target_vmaf),
        max_budget_kbps=float(max_budget_kbps),
        allow_codecs=tuple(allow_codecs),
        is_hdr=bool(is_hdr),
        meta=meta,
        candidates=tuple(candidates),
        winner=realised,
        smoke=bool(smoke),
        notes=notes,
    )


__all__ = [
    "PHASE_D_DURATION_GATE_S",
    "PHASE_D_SHOT_VARIANCE_GATE",
    "SALIENCY_CONTENT_CLASSES",
    "VERDICT_BIAS",
    "VERDICT_FALL_BACK",
    "VERDICT_PASS",
    "Meta",
    "Plan",
    "PlanEntry",
    "Rung",
    "Verdict",
    "auto",
]
