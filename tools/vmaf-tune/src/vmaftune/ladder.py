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

Phase E is **scaffold-only**: real (resolution, quality) sampling
requires Phase B's target-VMAF bisect (PR #347 in flight) and Phase A's
encode harness. The smoke path here mocks the corpus generator — see
``tools/vmaf-tune/tests/test_ladder.py``. End-to-end wiring against the
real bisect lands in a follow-up PR once Phase B merges.

See ``docs/adr/0295-vmaf-tune-phase-e-bitrate-ladder.md`` for the
design rationale and the alternatives considered (geometric ladder,
fixed Apple HLS authoring spec, JND-based, etc.).
"""

from __future__ import annotations

import dataclasses
import json
import math
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
    to dispatch to Phase B's target-VMAF bisect; tests inject a stub.

    Phase E does not run encodes itself — it composes Phase A
    (encoding) and Phase B (bisect) and turns their output into a
    ladder. Until Phase B is merged, ``sampler`` must be supplied
    explicitly; the default raises :class:`NotImplementedError` rather
    than silently producing garbage.
    """
    if sampler is None:
        sampler = _default_sampler

    points: list[LadderPoint] = []
    for w, h in resolutions:
        for tv in target_vmafs:
            pt = sampler(src, encoder, w, h, tv)
            points.append(pt)
    return Ladder(src=src, encoder=encoder, points=tuple(points))


def _default_sampler(
    src: Path, encoder: str, width: int, height: int, target_vmaf: float
) -> LadderPoint:  # pragma: no cover - guard
    raise NotImplementedError(
        "Phase E build_ladder() requires Phase B's target-VMAF bisect "
        "(PR #347 in flight). Pass `sampler=` explicitly until that lands."
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
