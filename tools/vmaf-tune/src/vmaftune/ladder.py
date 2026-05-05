# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase E — per-title bitrate-ladder generator.

Given a single source clip, sample the (resolution, quality) plane,
compute the convex hull of the resulting (bitrate, vmaf) points (the
Pareto frontier), pick a small number of "knee" renditions along the
hull, and emit an ABR ladder descriptor (HLS master playlist, DASH MPD,
or JSON).

This mirrors the Netflix per-title encoding paper's central idea: the
optimal ladder for one title is *not* a fixed grid — it is the set of
(resolution, bitrate) points that maximise quality per byte for *that*
title.

Production sampling is wired by composing Phase A's
:func:`vmaftune.corpus.iter_rows` (encode + score) with the
:func:`vmaftune.recommend.pick_target_vmaf` predicate from Phase B's
recommend surface (ADR-0306 / Research-0079). The :data:`SamplerFn`
seam stays open so callers can substitute a finer grid, a Bayesian
bisect, or a precomputed corpus row stream.

See ``docs/adr/0295-vmaf-tune-phase-e-bitrate-ladder.md`` for the
design rationale and the alternatives considered (geometric ladder,
fixed Apple HLS authoring spec, JND-based, etc.) and
``docs/adr/0307-vmaf-tune-ladder-default-sampler.md`` for the
default-sampler wiring decision.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LadderPoint:
    """One sampled (resolution, bitrate, vmaf, crf) point.

    Produced by :func:`build_ladder` for every (resolution, target_vmaf)
    request. ``crf`` is the encoder quality knob the bisect converged
    on; downstream encodes can re-use it directly.
    """

    width: int
    height: int
    bitrate_kbps: float
    vmaf: float
    crf: int

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclasses.dataclass(frozen=True)
class Rendition:
    """One rung of the final ABR ladder."""

    width: int
    height: int
    bitrate_kbps: float
    vmaf: float
    crf: int


@dataclasses.dataclass(frozen=True)
class Ladder:
    """Result of :func:`build_ladder` — the raw sampled grid.

    The convex hull and rendition picks are derived later by
    :func:`convex_hull` and :func:`select_knees`.
    """

    src: Path
    encoder: str
    points: tuple[LadderPoint, ...]


# ---------------------------------------------------------------------------
# Sampling — produces the (resolution, bitrate, vmaf) cloud
# ---------------------------------------------------------------------------


# Type alias for the corpus-generator callback. In production this wraps
# Phase B's target-VMAF bisect; in tests it returns synthetic points.
SamplerFn = Callable[[Path, str, int, int, float], LadderPoint]


def build_ladder(
    src: Path,
    encoder: str,
    resolutions: Sequence[tuple[int, int]],
    target_vmafs: Sequence[float],
    *,
    sampler: SamplerFn | None = None,
) -> Ladder:
    """Sample the (resolution x target_vmaf) plane for one source.

    For each (resolution, target_vmaf) cell, ``sampler`` produces a
    :class:`LadderPoint`. Production callers leave ``sampler`` ``None``
    to dispatch to :func:`_default_sampler`, which composes the Phase A
    corpus encode+score loop with :func:`recommend.pick_target_vmaf` to
    pick the (preset_default, CRF) row whose VMAF is closest to
    ``target_vmaf`` over the canonical 5-point CRF sweep
    ``(18, 23, 28, 33, 38)`` (ADR-0307, Research-0079). Tests inject a
    stub via ``sampler=`` to avoid live encoder runs.
    """
    if sampler is None:
        sampler = _default_sampler

    points: list[LadderPoint] = []
    for w, h in resolutions:
        for tv in target_vmafs:
            pt = sampler(src, encoder, w, h, tv)
            points.append(pt)
    return Ladder(src=src, encoder=encoder, points=tuple(points))


# Canonical 5-point CRF sweep used by the default sampler (ADR-0307).
# Spans the perceptually-informative range for libx264; non-x264
# adapters validate the points against their own ``quality_range``
# inside ``corpus.iter_rows``. Callers needing a finer grid pass an
# explicit ``sampler=`` to ``build_ladder``.
DEFAULT_SAMPLER_CRF_SWEEP: tuple[int, ...] = (18, 23, 28, 33, 38)


def _default_sampler_preset(encoder: str) -> str:
    """Pick the codec adapter's mid-range preset for the default sweep.

    Most adapters expose ``"medium"`` in their ``presets`` tuple; the
    fallback walks the tuple and returns its midpoint name.
    """
    # Lazy import — keeps the corpus / codec_adapters dependency off
    # the import path for callers that only use ``convex_hull`` /
    # ``select_knees`` / ``emit_manifest``.
    from .codec_adapters import get_adapter

    adapter = get_adapter(encoder)
    presets = tuple(adapter.presets)
    if "medium" in presets:
        return "medium"
    if not presets:
        raise ValueError(f"adapter {encoder!r} declares no presets")
    return presets[len(presets) // 2]


def _default_sampler(
    src: Path, encoder: str, width: int, height: int, target_vmaf: float
) -> LadderPoint:
    """Production sampler — encode the canonical CRF sweep, pick by VMAF.

    Composes :func:`vmaftune.corpus.iter_rows` (Phase A encode+score)
    with :func:`vmaftune.recommend.pick_target_vmaf` (Phase B-equivalent
    smallest-CRF-meeting-target predicate). The JSONL corpus is
    written to a tempfile that's discarded after the call returns; the
    encode-side temp dir lives under the same prefix and is cleaned up
    on exit.

    The source is treated as a raw YUV at ``yuv420p`` / 24 fps with a
    1-second nominal duration — these are placeholder defaults for
    rows whose ``bitrate_kbps`` is computed against the encoded
    duration; callers needing a different framerate / pix_fmt /
    duration should pass an explicit ``sampler=`` (the seam is
    deliberately preserved per ADR-0307).
    """
    # Lazy imports — see ``_default_sampler_preset``.
    from .corpus import CorpusJob, CorpusOptions, iter_rows
    from .recommend import pick_target_vmaf

    preset = _default_sampler_preset(encoder)
    cells = tuple((preset, crf) for crf in DEFAULT_SAMPLER_CRF_SWEEP)

    with tempfile.TemporaryDirectory(prefix="vmaftune-ladder-") as tmp:
        tmp_path = Path(tmp)
        job = CorpusJob(
            source=src,
            width=width,
            height=height,
            pix_fmt="yuv420p",
            framerate=24.0,
            duration_s=1.0,
            cells=cells,
        )
        opts = CorpusOptions(
            encoder=encoder,
            output=tmp_path / "corpus.jsonl",
            encode_dir=tmp_path / "encodes",
            keep_encodes=False,
            src_sha256=False,
        )
        rows = [r for r in iter_rows(job, opts) if int(r.get("exit_status", 0)) == 0]

    if not rows:
        raise RuntimeError(
            f"default sampler produced no scorable encodes for "
            f"{src} at {width}x{height} (encoder={encoder}); pass an "
            f"explicit sampler= to build_ladder() to debug."
        )

    pick = pick_target_vmaf(rows, target_vmaf)
    row = pick.row
    return LadderPoint(
        width=width,
        height=height,
        bitrate_kbps=float(row["bitrate_kbps"]),
        vmaf=float(row["vmaf_score"]),
        crf=int(row["crf"]),
    )


# ---------------------------------------------------------------------------
# Convex hull — Pareto frontier on (bitrate, vmaf)
# ---------------------------------------------------------------------------


def convex_hull(points: Iterable[LadderPoint]) -> list[LadderPoint]:
    """Upper-convex hull of the (bitrate, vmaf) cloud (Pareto frontier).

    Two passes:

    1. **Pareto filter** — drop any point dominated by another (lower
       or equal bitrate AND higher or equal vmaf, with one inequality
       strict). This already gives the staircase frontier.
    2. **Upper-convex hull** — over the remaining monotonically
       rising staircase, drop interior points whose surrounding
       neighbours form a *concave* arc above them (keep only the
       points on the *convex* envelope so an ABR algorithm switching
       between them sees diminishing returns, never increasing).

    On the resulting hull, both bitrate and vmaf are strictly
    monotonic; no other rendition strictly dominates any hull point.
    """
    pts = list(points)
    if not pts:
        return []
    # Sort by bitrate ascending, vmaf descending so duplicate-bitrate
    # clusters keep the higher-quality first.
    pts.sort(key=lambda p: (p.bitrate_kbps, -p.vmaf))

    # Pareto filter: walk left-to-right tracking the running max vmaf;
    # only emit points that strictly raise it.
    pareto: list[LadderPoint] = []
    best_vmaf = -math.inf
    last_bitrate: float | None = None
    for p in pts:
        if last_bitrate is not None and p.bitrate_kbps == last_bitrate:
            continue  # already kept higher-vmaf duplicate
        if p.vmaf > best_vmaf:
            pareto.append(p)
            best_vmaf = p.vmaf
            last_bitrate = p.bitrate_kbps

    if len(pareto) <= 2:
        return pareto

    # Upper-convex hull on the staircase. We want a *concave*
    # envelope on the (bitrate, vmaf) plane (diminishing returns):
    # walking left-to-right, the slope must be non-increasing. Pop
    # the previous point while it would create a non-concave angle
    # (cross product >= 0 means the new point is collinear or above
    # the line from -2 to -1, so -1 is redundant).
    hull: list[LadderPoint] = []
    for p in pareto:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def _cross(o: LadderPoint, a: LadderPoint, b: LadderPoint) -> float:
    """2D cross product of vectors (o->a) x (o->b).

    Positive = counter-clockwise turn (point ``b`` is above the line
    o->a, which is what we want on the upper hull).
    """
    return (a.bitrate_kbps - o.bitrate_kbps) * (b.vmaf - o.vmaf) - (a.vmaf - o.vmaf) * (
        b.bitrate_kbps - o.bitrate_kbps
    )


# ---------------------------------------------------------------------------
# Knee selection — pick `n` renditions along the hull
# ---------------------------------------------------------------------------


def select_knees(
    hull: Sequence[LadderPoint],
    n: int = 5,
    *,
    spacing: str = "log_bitrate",
) -> list[Rendition]:
    """Pick ``n`` rungs along the Pareto hull.

    ``spacing`` controls the parameter the rungs are evenly spaced in:

    - ``"log_bitrate"`` — Apple HLS authoring spec convention; rungs
      double in bitrate as you go up the ladder. Default.
    - ``"vmaf"`` — perceptually-spaced; equal VMAF gap per rung.

    The first and last hull points are always included, and the
    interior rungs are picked by snapping the ideal coordinate to the
    nearest hull point. Result is sorted ascending by bitrate.
    """
    if not hull:
        return []
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        # Pick the highest-quality point — most useful single rung.
        top = max(hull, key=lambda p: p.vmaf)
        return [_rendition_of(top)]
    if len(hull) <= n:
        return [_rendition_of(p) for p in hull]

    targets = _ideal_targets(hull, n, spacing)
    chosen: list[LadderPoint] = []
    seen: set[int] = set()
    for t in targets:
        idx = _nearest_index(hull, t, spacing)
        # Avoid duplicates if two targets snap to the same point.
        while idx in seen and idx + 1 < len(hull):
            idx += 1
        seen.add(idx)
        chosen.append(hull[idx])
    chosen.sort(key=lambda p: p.bitrate_kbps)
    return [_rendition_of(p) for p in chosen]


def _ideal_targets(hull: Sequence[LadderPoint], n: int, spacing: str) -> list[float]:
    if spacing == "log_bitrate":
        lo = math.log(max(hull[0].bitrate_kbps, 1.0))
        hi = math.log(max(hull[-1].bitrate_kbps, 1.0))
    elif spacing == "vmaf":
        lo = min(p.vmaf for p in hull)
        hi = max(p.vmaf for p in hull)
    else:
        raise ValueError(f"unknown spacing: {spacing!r}")
    if hi <= lo:
        return [lo] * n
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _nearest_index(hull: Sequence[LadderPoint], target: float, spacing: str) -> int:
    def _key(p: LadderPoint) -> float:
        if spacing == "log_bitrate":
            return math.log(max(p.bitrate_kbps, 1.0))
        return p.vmaf

    best_i = 0
    best_d = abs(_key(hull[0]) - target)
    for i in range(1, len(hull)):
        d = abs(_key(hull[i]) - target)
        if d < best_d:
            best_i = i
            best_d = d
    return best_i


def _rendition_of(p: LadderPoint) -> Rendition:
    return Rendition(
        width=p.width,
        height=p.height,
        bitrate_kbps=p.bitrate_kbps,
        vmaf=p.vmaf,
        crf=p.crf,
    )


# ---------------------------------------------------------------------------
# Manifest emit — HLS / DASH / JSON
# ---------------------------------------------------------------------------


def emit_manifest(ladder: Sequence[Rendition], format: str = "hls") -> str:
    """Serialise a list of :class:`Rendition` rungs in the requested format.

    Supported formats:

    - ``"hls"`` — Apple HLS master playlist with one
      ``#EXT-X-STREAM-INF`` per rung. Bandwidth is reported in bps
      (HLS spec); resolution as ``WxH``. Variant URIs are placeholders
      (``rendition_<W>x<H>_<kbps>k.m3u8``); the consumer re-points
      them at real per-rendition playlists.
    - ``"dash"`` — DASH MPD with one ``Representation`` per rung
      under a single ``AdaptationSet``. Minimal but spec-conformant.
    - ``"json"`` — JSON descriptor (the canonical machine-readable
      form for downstream tooling).

    Output is a string; callers write to disk if needed. Renditions
    are emitted in ascending-bitrate order.
    """
    sorted_ladder = sorted(ladder, key=lambda r: r.bitrate_kbps)
    if format == "hls":
        return _emit_hls(sorted_ladder)
    if format == "dash":
        return _emit_dash(sorted_ladder)
    if format == "json":
        return _emit_json(sorted_ladder)
    raise ValueError(f"unknown manifest format: {format!r} (expected hls/dash/json)")


def _emit_hls(ladder: Sequence[Rendition]) -> str:
    lines: list[str] = ["#EXTM3U", "#EXT-X-VERSION:6"]
    for r in ladder:
        bps = int(round(r.bitrate_kbps * 1000.0))
        uri = f"rendition_{r.width}x{r.height}_{int(round(r.bitrate_kbps))}k.m3u8"
        lines.append(
            f"#EXT-X-STREAM-INF:BANDWIDTH={bps},RESOLUTION={r.width}x{r.height},"
            f'CODECS="avc1.640028"'
        )
        lines.append(uri)
    return "\n".join(lines) + "\n"


def _emit_dash(ladder: Sequence[Rendition]) -> str:
    reps: list[str] = []
    for i, r in enumerate(ladder):
        bps = int(round(r.bitrate_kbps * 1000.0))
        reps.append(
            f'    <Representation id="r{i}" bandwidth="{bps}" '
            f'width="{r.width}" height="{r.height}" '
            f'codecs="avc1.640028" mimeType="video/mp4">\n'
            f"      <BaseURL>rendition_{r.width}x{r.height}_"
            f"{int(round(r.bitrate_kbps))}k.mp4</BaseURL>\n"
            f"    </Representation>"
        )
    body = "\n".join(reps)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
        'type="static" minBufferTime="PT2S" '
        'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011">\n'
        "  <Period>\n"
        '    <AdaptationSet contentType="video" segmentAlignment="true">\n'
        f"{body}\n"
        "    </AdaptationSet>\n"
        "  </Period>\n"
        "</MPD>\n"
    )


def _emit_json(ladder: Sequence[Rendition]) -> str:
    payload = {
        "schema": "vmaf-tune-ladder/v1",
        "renditions": [
            {
                "width": r.width,
                "height": r.height,
                "bitrate_kbps": r.bitrate_kbps,
                "bandwidth_bps": int(round(r.bitrate_kbps * 1000.0)),
                "vmaf": r.vmaf,
                "crf": r.crf,
            }
            for r in ladder
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Top-level convenience — build + hull + select + emit in one call
# ---------------------------------------------------------------------------


def build_and_emit(
    src: Path,
    encoder: str,
    resolutions: Sequence[tuple[int, int]],
    target_vmafs: Sequence[float],
    *,
    quality_tiers: int = 5,
    format: str = "hls",
    spacing: str = "log_bitrate",
    sampler: SamplerFn | None = None,
) -> str:
    """Convenience: build → hull → select → emit, returns the manifest string."""
    ladder = build_ladder(src, encoder, resolutions, target_vmafs, sampler=sampler)
    hull = convex_hull(ladder.points)
    rungs = select_knees(hull, n=quality_tiers, spacing=spacing)
    return emit_manifest(rungs, format=format)
