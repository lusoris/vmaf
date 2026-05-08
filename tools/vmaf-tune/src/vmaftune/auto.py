# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""``vmaf-tune auto`` — Phase F adaptive recipe-aware tuning entry point.

Phase F composes the per-phase subcommands (``corpus``, ``recommend``,
``fast``, ``predict``, ``tune-per-shot``, ``recommend-saliency``,
``ladder``, ``compare``) plus the orthogonal modes (HDR, sample-clip,
resolution-aware) into a single deterministic decision tree. The
sequential composition (F.1) walks the tree top-to-bottom and runs every
stage; the seven short-circuits (F.2, this module) skip stages whose
output is determined by metadata alone.

Decision tree (per :doc:`docs/adr/0325-vmaf-tune-phase-f-auto.md`):

.. code-block:: text

   auto(src, target_vmaf, max_budget_kbps, allow_codecs):
       meta = probe(src); is_hdr = detect_hdr(meta)            # ADR-0300
       rungs = [meta.resolution] if meta.height < 2160         # ADR-0289
               else ladder.candidate_rungs(meta)
       codecs = (allow_codecs if len==1
                 else [user_pin] if user_pinned_codec
                 else compare.shortlist(allow_codecs, meta))
       plan = []
       for rung, codec in (rungs x codecs):
           v = predict.crf_for_target(rung, codec, target_vmaf, meta)
           if v.verdict == FALL_BACK:
               v = recommend.coarse_to_fine(rung, codec, target_vmaf)
           plan.append((rung, codec, v))
       if duration > 5min and shot_variance(src) > 0.15:        # Phase D gate
           plan = [tune_per_shot.refine(p) for p in plan]
       if meta.content_class in {animation, screen_content}:    # saliency gate
           plan = [recommend_saliency.maybe_apply(p) for p in plan]
       winner = pick_pareto(plan, target_vmaf, max_budget_kbps)
       return realise(winner, hdr=is_hdr)

The seven short-circuit predicates live in this module as standalone
helpers (``_should_short_circuit_<N>``) so each one is unit-testable in
isolation. Each predicate returns ``True`` when the corresponding stage
can be skipped; the main driver records the firing predicate names in
``plan.metadata.short_circuits`` for post-hoc speedup analysis.

This module does **not** ship the F.3 confidence-aware fallbacks (per-cell
escalation to ``recommend.coarse_to_fine`` on ``FALL_BACK``) or the F.4
per-content-type recipe overrides — those are sibling PRs gated on F.2.
The ``--smoke`` mode exercises the composition end-to-end with mocked
sub-phases (no ffmpeg, no ONNX) so this scaffold can ship without the
production wiring.

See also:

* :mod:`vmaftune.fast` — the ``fast`` subcommand. Phase F's
  short-circuits do **not** shadow ``fast``'s behaviour; ``fast`` is a
  different operator surface (proxy + Bayesian over a single codec) and
  Phase F's auto driver never invokes it from inside the tree.
* :mod:`vmaftune.compare` — codec shortlist when ``--allow-codecs`` has
  more than one entry.
* :mod:`vmaftune.ladder` — multi-rung ABR ladder for >= 2160p sources.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
from collections.abc import Sequence
from pathlib import Path

_LOG = logging.getLogger(__name__)


# Phase D gate thresholds (per ADR-0325 short-circuit #7). The 5-min /
# 0.15-shot-variance pair is a placeholder; F.3 fits these from a real
# corpus once Phase F has emitted enough labelled compositions to make
# the fit statistically defensible. Until then, the placeholders keep
# the gate conservative — short low-variance content skips per-shot
# tuning, everything else gets it.
PHASE_D_DURATION_GATE_S: float = 300.0  # 5 minutes
PHASE_D_SHOT_VARIANCE_GATE: float = 0.15

# 2160p (4K) is the resolution gate above which the ladder is
# multi-rung (per ADR-0289 / ADR-0295). Below it, the auto driver
# encodes only the source rung — no need to evaluate 720p / 1080p
# rungs of a 1080p source.
LADDER_MULTI_RUNG_HEIGHT: int = 2160

# Content classes that benefit from saliency-aware ROI tuning per
# ADR-0293. Photographic / live-action content does not — saliency
# weights the centre of the frame and live-action subjects already
# get foveal weighting from VMAF's perceptual model.
SALIENCY_CONTENT_CLASSES: frozenset[str] = frozenset({"animation", "screen_content"})


class ShortCircuit(enum.Enum):
    """Names of the seven short-circuits (per ADR-0325 §F.2).

    The string values are the canonical identifiers recorded in
    ``plan.metadata.short_circuits`` and surfaced in the JSON output.
    """

    LADDER_SINGLE_RUNG = "ladder-single-rung"
    CODEC_PINNED = "codec-pinned"
    PREDICTOR_GOSPEL = "predictor-gospel"
    SKIP_SALIENCY = "skip-saliency"
    SDR_SKIP = "sdr-skip"
    SAMPLE_CLIP_PROPAGATE = "sample-clip-propagate"
    SKIP_PER_SHOT = "skip-per-shot"


@dataclasses.dataclass(frozen=True)
class SourceMeta:
    """Source metadata consumed by the decision tree.

    The fields mirror the per-phase ADR contracts: ``height`` from
    ``ffprobe`` (ADR-0289), ``is_hdr`` from
    :func:`vmaftune.hdr.detect_hdr` (ADR-0300), ``content_class`` from
    the F.4 classifier (placeholder until F.4 lands; defaults to
    ``"live_action"`` so the saliency gate stays a no-op on unknown
    content), ``duration_s`` and ``shot_variance`` from the per-shot
    detector (ADR-0276 phase-d).
    """

    height: int
    width: int
    is_hdr: bool = False
    content_class: str = "live_action"
    duration_s: float = 0.0
    shot_variance: float = 0.0
    sample_clip_seconds: float = 0.0


@dataclasses.dataclass
class PlanState:
    """Mutable state threaded through the decision tree.

    Each stage of the tree consults this state and may mutate the
    ``short_circuits`` list, the ``predictor_verdict`` field, the
    ``codecs`` list, etc. The driver returns the final state in the
    plan's ``metadata`` block so post-hoc analysis can measure which
    short-circuits fired.
    """

    target_vmaf: float
    max_budget_kbps: float
    allow_codecs: tuple[str, ...]
    user_pinned_codec: str | None = None
    predictor_verdict: str | None = None  # "GOSPEL" / "LIKELY" / "FALL_BACK"
    short_circuits: list[str] = dataclasses.field(default_factory=list)

    def fired(self, sc: ShortCircuit) -> None:
        """Record that ``sc`` fired (idempotent on repeats)."""
        if sc.value not in self.short_circuits:
            self.short_circuits.append(sc.value)


# ---------------------------------------------------------------------------
# The seven short-circuit predicates. Each returns True when the stage
# the predicate guards can be skipped. Predicates are pure functions of
# (meta, plan_state) so tests mock the inputs and assert the branch
# fires / doesn't fire without invoking the full driver.
# ---------------------------------------------------------------------------


def _should_short_circuit_1_single_rung_ladder(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #1 — single-rung ladder when ``meta.height < 2160``.

    Per ADR-0325 / ADR-0289: sub-4K sources don't need a multi-rung
    ABR ladder evaluation; the source rung is the only candidate. The
    driver still runs the per-rung pipeline, just on one rung.
    """
    del plan_state  # unused — predicate depends on meta only
    return int(meta.height) < LADDER_MULTI_RUNG_HEIGHT


def _should_short_circuit_2_codec_pinned(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #2 — codec known / pinned.

    When ``--allow-codecs`` resolves to exactly one codec, the
    ``compare.shortlist`` stage adds no information — there's nothing
    to compare against. Skip directly to the per-codec sweep.
    """
    del meta  # unused — predicate depends on plan_state only
    if plan_state.user_pinned_codec is not None:
        return True
    return len(plan_state.allow_codecs) == 1


def _should_short_circuit_3_predictor_gospel(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #3 — predictor returned GOSPEL.

    Per ADR-0325 escalation rule: when ``predict.crf_for_target``
    returns ``GOSPEL`` (residuals within threshold across the
    validation sample), trust the predictor's CRF pick and skip the
    ``recommend.coarse_to_fine`` fallback for that cell.
    """
    del meta  # unused — predicate depends on plan_state only
    return plan_state.predictor_verdict == "GOSPEL"


def _should_short_circuit_4_skip_saliency(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #4 — photographic / non-screen-content skips saliency.

    Per ADR-0293: saliency-aware ROI tuning is gated on content class.
    ``animation`` and ``screen_content`` benefit; ``live_action`` /
    ``photographic`` does not (VMAF's perceptual model already gives
    centre-frame foveal weighting). Skip ``recommend_saliency.maybe_apply``.
    """
    del plan_state  # unused — predicate depends on meta only
    return meta.content_class not in SALIENCY_CONTENT_CLASSES


def _should_short_circuit_5_sdr_skip(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #5 — SDR source skips the HDR pipeline.

    Per ADR-0300: HDR detection is permissive; an SDR source skips the
    HDR resolution + model-selection branch. ``meta.is_hdr`` is
    populated by :func:`vmaftune.hdr.detect_hdr`; ``False`` here is
    the canonical "treat as SDR" signal.
    """
    del plan_state  # unused — predicate depends on meta only
    return not bool(meta.is_hdr)


def _should_short_circuit_6_sample_clip_propagate(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #6 — propagate user-supplied sample-clip.

    When the user passed ``--sample-clip-seconds`` (ADR-0301), the
    auto driver propagates that value to the internal sweeps rather
    than re-deciding clip length per stage. This is a propagation
    short-circuit, not a stage-skip — the flag is recorded in the
    metadata so downstream stages know to honour it verbatim.
    """
    del plan_state  # unused — predicate depends on meta only
    return float(meta.sample_clip_seconds) > 0.0


def _should_short_circuit_7_skip_per_shot(meta: SourceMeta, plan_state: PlanState) -> bool:
    """Short-circuit #7 — duration / shot-variance gate.

    Per ADR-0325: skip ``tune_per_shot.refine`` when the source is
    both short (< 5 min) **and** low-variance (shot variance < 0.15).
    Either condition alone is not enough — a short high-variance
    trailer benefits from per-shot, and a long low-variance lecture
    capture also benefits. The thresholds are placeholders pending
    F.3 empirical fit.
    """
    del plan_state  # unused — predicate depends on meta only
    short = float(meta.duration_s) < PHASE_D_DURATION_GATE_S
    low_variance = float(meta.shot_variance) < PHASE_D_SHOT_VARIANCE_GATE
    return short and low_variance


# Ordered tuple of (ShortCircuit, predicate). The order is the
# evaluation order in the driver and is part of the public contract:
# tests assert that an earlier-firing predicate doesn't shadow a
# later one whose result would have been different. Adding a new
# short-circuit means appending here, never reordering.
SHORT_CIRCUIT_PREDICATES: tuple[tuple[ShortCircuit, "callable"], ...] = (  # type: ignore[type-arg]
    (ShortCircuit.LADDER_SINGLE_RUNG, _should_short_circuit_1_single_rung_ladder),
    (ShortCircuit.CODEC_PINNED, _should_short_circuit_2_codec_pinned),
    (ShortCircuit.PREDICTOR_GOSPEL, _should_short_circuit_3_predictor_gospel),
    (ShortCircuit.SKIP_SALIENCY, _should_short_circuit_4_skip_saliency),
    (ShortCircuit.SDR_SKIP, _should_short_circuit_5_sdr_skip),
    (ShortCircuit.SAMPLE_CLIP_PROPAGATE, _should_short_circuit_6_sample_clip_propagate),
    (ShortCircuit.SKIP_PER_SHOT, _should_short_circuit_7_skip_per_shot),
)


def evaluate_short_circuits(meta: SourceMeta, plan_state: PlanState) -> list[str]:
    """Run every predicate in declaration order, recording firers.

    Returns the list of fired short-circuit names (the same list
    available on ``plan_state.short_circuits``). Pure function over
    its inputs — calling it twice on the same state yields the same
    list, and predicate order is deterministic.
    """
    for sc, predicate in SHORT_CIRCUIT_PREDICATES:
        if predicate(meta, plan_state):
            plan_state.fired(sc)
    return list(plan_state.short_circuits)


# ---------------------------------------------------------------------------
# F.1 sequential scaffold + F.2 short-circuit-aware driver.
# The ``--smoke`` mode skips real ffmpeg / ONNX wiring; production
# wiring lands in subsequent F.x PRs that swap the smoke stubs for the
# real per-phase calls.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AutoPlan:
    """Result of an ``auto`` run.

    ``cells`` is one entry per ``(rung, codec)`` cell; for the smoke
    path the entries are placeholders (no real encode happened) but
    the schema is stable from F.1 onwards.
    """

    cells: list[dict]
    metadata: dict


def run_auto(
    *,
    src: Path,
    target_vmaf: float,
    max_budget_kbps: float,
    allow_codecs: Sequence[str],
    user_pinned_codec: str | None = None,
    sample_clip_seconds: float = 0.0,
    smoke: bool = False,
    meta_override: SourceMeta | None = None,
) -> AutoPlan:
    """Drive the F.1 + F.2 decision tree.

    The non-smoke path is intentionally unimplemented at this PR's
    scope — production wiring lands in follow-up PRs that fill in
    each per-phase call. ``smoke=True`` exercises the composition
    end-to-end with mocked sub-phases.

    ``meta_override`` lets callers (and tests) inject a pre-built
    :class:`SourceMeta`. When ``None`` and ``smoke=True``, a synthetic
    1080p SDR live-action meta is fabricated so the smoke run is
    deterministic without touching ffprobe.
    """
    if not smoke and meta_override is None:
        # Production probe wiring is a follow-up PR; until it lands
        # the auto driver only runs in smoke mode or with an explicit
        # caller-supplied meta.
        raise NotImplementedError(
            "auto: non-smoke path requires meta_override until production "
            "probe wiring lands (F.3 follow-up). Re-run with --smoke or "
            "pass an explicit SourceMeta."
        )

    meta = meta_override or SourceMeta(
        height=1080,
        width=1920,
        is_hdr=False,
        content_class="live_action",
        duration_s=120.0,
        shot_variance=0.05,
        sample_clip_seconds=sample_clip_seconds,
    )

    plan_state = PlanState(
        target_vmaf=target_vmaf,
        max_budget_kbps=max_budget_kbps,
        allow_codecs=tuple(allow_codecs),
        user_pinned_codec=user_pinned_codec,
    )

    # ------------------------------------------------------------------
    # Stage 1 — ladder rung selection (short-circuit #1).
    # ------------------------------------------------------------------
    if _should_short_circuit_1_single_rung_ladder(meta, plan_state):
        plan_state.fired(ShortCircuit.LADDER_SINGLE_RUNG)
        rungs: tuple[int, ...] = (int(meta.height),)
    else:
        # Multi-rung path — production wiring delegates to ladder.py.
        rungs = (2160, 1440, 1080, 720, 540)

    # ------------------------------------------------------------------
    # Stage 2 — codec shortlist (short-circuit #2).
    # ------------------------------------------------------------------
    if _should_short_circuit_2_codec_pinned(meta, plan_state):
        plan_state.fired(ShortCircuit.CODEC_PINNED)
        codecs: tuple[str, ...] = (user_pinned_codec,) if user_pinned_codec else tuple(allow_codecs)
    else:
        # Production wiring delegates to compare.shortlist; smoke
        # path keeps the full allow-list.
        codecs = tuple(allow_codecs)

    # ------------------------------------------------------------------
    # Stage 3 — HDR pipeline (short-circuit #5).
    # ------------------------------------------------------------------
    if _should_short_circuit_5_sdr_skip(meta, plan_state):
        plan_state.fired(ShortCircuit.SDR_SKIP)
        hdr_args: tuple[str, ...] = ()
    else:
        # Production wiring delegates to hdr.hdr_codec_args + the
        # HDR VMAF model selector.
        hdr_args = ("-color_primaries", "bt2020", "-color_trc", "smpte2084")

    # ------------------------------------------------------------------
    # Stage 4 — sample-clip propagation (short-circuit #6).
    # ------------------------------------------------------------------
    propagated_clip = 0.0
    if _should_short_circuit_6_sample_clip_propagate(meta, plan_state):
        plan_state.fired(ShortCircuit.SAMPLE_CLIP_PROPAGATE)
        propagated_clip = float(meta.sample_clip_seconds)

    # ------------------------------------------------------------------
    # Stage 5 — per-cell predictor + escalation (short-circuit #3).
    # In smoke mode we synthesise a GOSPEL verdict so the gate fires
    # in the unit smoke run; production wiring will set the verdict
    # from predictor_validate.ValidationReport.verdict.
    # ------------------------------------------------------------------
    if smoke:
        plan_state.predictor_verdict = "GOSPEL"
    cells: list[dict] = []
    for rung in rungs:
        for codec in codecs:
            cell_state = dataclasses.replace(plan_state)
            cell_state.short_circuits = list(plan_state.short_circuits)
            if _should_short_circuit_3_predictor_gospel(meta, cell_state):
                cell_state.fired(ShortCircuit.PREDICTOR_GOSPEL)
                # Carry the cell-level firing back up so the metadata
                # block records that GOSPEL fired at least once.
                if ShortCircuit.PREDICTOR_GOSPEL.value not in plan_state.short_circuits:
                    plan_state.fired(ShortCircuit.PREDICTOR_GOSPEL)
            cells.append(
                {
                    "rung": int(rung),
                    "codec": str(codec),
                    "verdict": plan_state.predictor_verdict or "UNKNOWN",
                    "crf": 23,  # placeholder; production wiring fills this in
                    "estimated_vmaf": float(target_vmaf),
                    "estimated_bitrate_kbps": float(max_budget_kbps),
                    "hdr_args": list(hdr_args),
                    "sample_clip_seconds": propagated_clip,
                }
            )

    # ------------------------------------------------------------------
    # Stage 6 — saliency gate (short-circuit #4).
    # ------------------------------------------------------------------
    if _should_short_circuit_4_skip_saliency(meta, plan_state):
        plan_state.fired(ShortCircuit.SKIP_SALIENCY)
    # else: production wiring would call recommend_saliency.maybe_apply
    # on every cell.

    # ------------------------------------------------------------------
    # Stage 7 — per-shot refinement gate (short-circuit #7).
    # ------------------------------------------------------------------
    if _should_short_circuit_7_skip_per_shot(meta, plan_state):
        plan_state.fired(ShortCircuit.SKIP_PER_SHOT)
    # else: production wiring would call tune_per_shot.refine on
    # every cell.

    metadata = {
        "src": str(src),
        "target_vmaf": float(target_vmaf),
        "max_budget_kbps": float(max_budget_kbps),
        "allow_codecs": list(allow_codecs),
        "user_pinned_codec": user_pinned_codec,
        "smoke": bool(smoke),
        "source_meta": dataclasses.asdict(meta),
        "short_circuits": list(plan_state.short_circuits),
    }
    return AutoPlan(cells=cells, metadata=metadata)


def emit_plan_json(plan: AutoPlan) -> str:
    """Serialise ``plan`` to a stable JSON string.

    Schema is the public contract for downstream consumers (the MCP
    server, the CI corpus collector, post-hoc speedup analysis). Keys
    are sorted so the output is reproducible across runs.
    """
    payload = {"cells": plan.cells, "metadata": plan.metadata}
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "AutoPlan",
    "LADDER_MULTI_RUNG_HEIGHT",
    "PHASE_D_DURATION_GATE_S",
    "PHASE_D_SHOT_VARIANCE_GATE",
    "PlanState",
    "SALIENCY_CONTENT_CLASSES",
    "SHORT_CIRCUIT_PREDICATES",
    "ShortCircuit",
    "SourceMeta",
    "_should_short_circuit_1_single_rung_ladder",
    "_should_short_circuit_2_codec_pinned",
    "_should_short_circuit_3_predictor_gospel",
    "_should_short_circuit_4_skip_saliency",
    "_should_short_circuit_5_sdr_skip",
    "_should_short_circuit_6_sample_clip_propagate",
    "_should_short_circuit_7_skip_per_shot",
    "emit_plan_json",
    "evaluate_short_circuits",
    "run_auto",
]
