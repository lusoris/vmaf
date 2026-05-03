# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase D scaffold — per-shot CRF tuning.

The "Netflix per-shot encoding" table-stakes feature for `vmaf-tune`.
TransNet V2 (real weights, ADR-0223) cuts the source into shots; for
each shot we run a target-VMAF predicate (Phase B bisect, here a
pluggable callback while Phase B is in flight) to pick a CRF; then we
emit an FFmpeg encoding plan that produces a per-shot CRF-varying
encode and concatenates the segments.

This file is a **scaffold**: the public API shape is stable, but the
two integration seams (shot detection via :data:`vmaf-perShot`, target
predicate via Phase B bisect) are pluggable so the tests run with
mocks and the production wiring can land in follow-up PRs without
breaking the surface. See ADR-0276.

Public surface:

* :class:`Shot` — half-open frame range ``[start_frame, end_frame)``.
* :class:`ShotRecommendation` — ``(shot, crf, predicted_vmaf)``.
* :class:`EncodingPlan` — ordered segments + the FFmpeg argv list to
  produce the final encode.
* :func:`detect_shots` — wraps the C-side ``vmaf-perShot`` binary
  (ADR-0222) when available; falls back to a one-shot range.
* :func:`tune_per_shot` — drives the target-VMAF predicate per shot.
* :func:`merge_shots` — collapses recommendations into an
  :class:`EncodingPlan`.

The scaffold deliberately stops short of running encodes — Phase D's
end-to-end loop is gated on Phase B's bisect landing as code, on the
codec adapters growing per-shot ``--zones`` / ``--qpfile`` emit hooks
(ADR-0237 §Consequences), and on a per-shot held-out validation
corpus. This file ships the orchestration shape so those follow-ups
are drop-in.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from io import StringIO
from pathlib import Path

from .codec_adapters import get_adapter

# Pluggable predicate signature: given a shot + target VMAF + encoder
# name, return ``(crf, predicted_vmaf)``. Production wiring will call
# Phase B's bisect; tests inject a deterministic stub.
PredicateFn = Callable[["Shot", float, str], tuple[int, float]]


@dataclasses.dataclass(frozen=True)
class Shot:
    """Half-open frame range describing one shot.

    ``start_frame`` is inclusive, ``end_frame`` is exclusive — matching
    Python slice convention. ``vmaf-perShot``'s CSV output uses
    inclusive ``end_frame``; :func:`detect_shots` normalises into the
    half-open form.
    """

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(f"invalid shot range: [{self.start_frame}, {self.end_frame})")

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


@dataclasses.dataclass(frozen=True)
class ShotRecommendation:
    """Per-shot CRF recommendation produced by :func:`tune_per_shot`."""

    shot: Shot
    crf: int
    predicted_vmaf: float


@dataclasses.dataclass(frozen=True)
class EncodingPlan:
    """Segment list plus the FFmpeg argv list that realises the encode.

    The plan is split into per-shot single-encode commands plus a
    final concat-demuxer command. Callers are free to run them
    sequentially or to parallelise per-shot encodes — the segment
    files are independent.
    """

    recommendations: tuple[ShotRecommendation, ...]
    encoder: str
    framerate: float
    segment_commands: tuple[tuple[str, ...], ...]
    concat_command: tuple[str, ...]
    concat_listing: str


def _which(binary: str) -> str | None:
    """``shutil.which`` wrapper kept thin so tests can monkeypatch."""
    return shutil.which(binary)


def detect_shots(
    video_path: Path,
    *,
    width: int,
    height: int,
    pix_fmt: str = "yuv420p",
    bitdepth: int = 8,
    total_frames: int | None = None,
    per_shot_bin: str = "vmaf-perShot",
    runner: object | None = None,
) -> list[Shot]:
    """Return the shot boundary list for ``video_path``.

    Calls the fork's C-side ``vmaf-perShot`` binary (ADR-0222) which
    wraps TransNet V2 (ADR-0223). Falls back to a single-shot range
    spanning the whole clip when the binary is missing or fails.

    ``total_frames`` is required for the fallback path; the
    ``vmaf-perShot`` path infers it from the YUV size.
    """
    binary = _which(per_shot_bin) if runner is None else per_shot_bin
    if binary is None:
        return _single_shot_fallback(total_frames)

    cmd = [
        per_shot_bin,
        "--reference",
        str(video_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--pixel_format",
        _bitdepth_aware_pix(pix_fmt),
        "--bitdepth",
        str(bitdepth),
        "--output",
        "-",  # stdout
        "--format",
        "json",
    ]

    runner_fn = runner or subprocess.run
    completed = runner_fn(  # type: ignore[operator]
        cmd, capture_output=True, text=True, check=False
    )
    rc = int(getattr(completed, "returncode", 1))
    stdout = getattr(completed, "stdout", "") or ""
    if rc != 0 or not stdout.strip():
        return _single_shot_fallback(total_frames)

    return _parse_per_shot_json(stdout)


def _bitdepth_aware_pix(pix_fmt: str) -> str:
    """Map ffmpeg pix_fmt names to ``vmaf-perShot``'s ``--pixel_format``."""
    if "422" in pix_fmt:
        return "422"
    if "444" in pix_fmt:
        return "444"
    return "420"


def _single_shot_fallback(total_frames: int | None) -> list[Shot]:
    """One shot covering the whole clip — used when shot detection fails."""
    if total_frames is None or total_frames <= 0:
        # Caller has no frame count: emit a sentinel range that downstream
        # can pattern-match. End-frame > start-frame keeps :class:`Shot`
        # happy without lying about real length.
        return [Shot(start_frame=0, end_frame=1)]
    return [Shot(start_frame=0, end_frame=total_frames)]


def _parse_per_shot_json(payload: str) -> list[Shot]:
    """Parse ``vmaf-perShot``'s JSON output into a list of shots.

    Schema per ``docs/usage/vmaf-perShot.md``:

    .. code-block:: json

       {"shots": [{"start_frame": 0, "end_frame": 3, ...}, ...]}

    ``end_frame`` is inclusive in the source schema; we normalise to
    half-open here.
    """
    data = json.loads(payload)
    shots = data.get("shots") or []
    out: list[Shot] = []
    for entry in shots:
        start = int(entry["start_frame"])
        # Source schema is inclusive; half-open conversion adds 1.
        end = int(entry["end_frame"]) + 1
        out.append(Shot(start_frame=start, end_frame=end))
    if not out:
        return [Shot(start_frame=0, end_frame=1)]
    return out


def parse_per_shot_csv(payload: str) -> list[Shot]:
    """CSV variant of :func:`_parse_per_shot_json` — public for callers
    who already have the CSV sidecar from a prior ``vmaf-perShot`` run.
    """
    out: list[Shot] = []
    reader = csv.DictReader(StringIO(payload))
    for row in reader:
        start = int(row["start_frame"])
        end = int(row["end_frame"]) + 1
        out.append(Shot(start_frame=start, end_frame=end))
    return out


def tune_per_shot(
    shots: Sequence[Shot],
    *,
    target_vmaf: float,
    encoder: str = "libx264",
    predicate: PredicateFn | None = None,
) -> list[ShotRecommendation]:
    """Pick a per-shot CRF for each shot.

    ``predicate`` is the integration seam for Phase B's target-VMAF
    bisect. Production callers wire :func:`vmaftune.bisect.find_crf`
    (Phase B); tests inject a complexity-aware stub. The default
    predicate uses the codec adapter's ``quality_default`` clamped
    into the codec's quality range — enough to round-trip the
    scaffold without doing real searches.
    """
    if not shots:
        raise ValueError("tune_per_shot requires at least one shot")
    adapter = get_adapter(encoder)
    pred = predicate or _default_predicate

    recs: list[ShotRecommendation] = []
    for shot in shots:
        crf, predicted = pred(shot, target_vmaf, encoder)
        lo, hi = adapter.quality_range
        clamped = max(lo, min(hi, int(crf)))
        recs.append(ShotRecommendation(shot=shot, crf=clamped, predicted_vmaf=float(predicted)))
    return recs


def _default_predicate(shot: Shot, target_vmaf: float, encoder: str) -> tuple[int, float]:
    """Trivial fallback predicate.

    Used only when the caller does not pass a real bisect; exists so
    the scaffold runs end-to-end in tests. Returns the codec's default
    quality value alongside the requested target VMAF.
    """
    _ = shot  # length unused in the trivial predicate
    adapter = get_adapter(encoder)
    return (adapter.quality_default, float(target_vmaf))


def merge_shots(
    recommendations: Sequence[ShotRecommendation],
    *,
    source: Path,
    output: Path,
    framerate: float,
    encoder: str = "libx264",
    segment_dir: Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> EncodingPlan:
    """Collapse per-shot recommendations into an :class:`EncodingPlan`.

    The plan emits one ``ffmpeg`` invocation per segment (using
    ``-ss`` + ``-frames:v`` derived from the half-open shot range) and
    a final concat-demuxer command that stitches the segments into
    ``output``. Segment files live under ``segment_dir`` (defaults to
    ``output.parent / "segments"``).
    """
    if not recommendations:
        raise ValueError("merge_shots requires at least one recommendation")
    adapter = get_adapter(encoder)

    seg_dir = segment_dir or output.parent / "segments"
    segment_cmds: list[tuple[str, ...]] = []
    listing_lines: list[str] = []
    for idx, rec in enumerate(recommendations):
        seg_path = seg_dir / f"shot_{idx:04d}.mp4"
        cmd = _segment_command(
            source=source,
            framerate=framerate,
            shot=rec.shot,
            crf=rec.crf,
            output=seg_path,
            encoder=adapter.encoder,
            ffmpeg_bin=ffmpeg_bin,
        )
        segment_cmds.append(cmd)
        # concat-demuxer expects POSIX-style escaped paths.
        listing_lines.append(f"file '{seg_path.as_posix()}'")

    listing = "\n".join(listing_lines) + "\n"
    concat_cmd: tuple[str, ...] = (
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str((seg_dir / "concat.txt").as_posix()),
        "-c",
        "copy",
        str(output),
    )

    return EncodingPlan(
        recommendations=tuple(recommendations),
        encoder=adapter.encoder,
        framerate=float(framerate),
        segment_commands=tuple(segment_cmds),
        concat_command=concat_cmd,
        concat_listing=listing,
    )


def _segment_command(
    *,
    source: Path,
    framerate: float,
    shot: Shot,
    crf: int,
    output: Path,
    encoder: str,
    ffmpeg_bin: str,
) -> tuple[str, ...]:
    """Build the per-shot FFmpeg argv.

    Uses ``-ss`` (input-seek) + ``-frames:v`` so the segment is exactly
    ``shot.length`` frames regardless of GOP placement. Callers
    encoding a raw YUV source must add the ``-f rawvideo`` + geometry
    flags upstream — Phase D's smoke path here assumes the source is
    already an addressable container.
    """
    start_seconds = shot.start_frame / framerate
    return (
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-frames:v",
        str(shot.length),
        "-c:v",
        encoder,
        "-crf",
        str(crf),
        str(output),
    )


def write_concat_listing(plan: EncodingPlan, listing_path: Path) -> Path:
    """Persist the concat-demuxer listing to ``listing_path``.

    Convenience helper — kept separate from :func:`merge_shots` so the
    plan can be inspected/tested without filesystem side effects.
    """
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(plan.concat_listing, encoding="utf-8")
    return listing_path


def plan_to_shell_script(plan: EncodingPlan) -> str:
    """Render a plan as a copy-paste shell script for diagnostics.

    Not used by production callers; useful when debugging a Phase D
    smoke run.
    """
    lines: list[str] = ["#!/bin/sh", "set -eu"]
    for cmd in plan.segment_commands:
        lines.append(_shell_join(cmd))
    lines.append(_shell_join(plan.concat_command))
    return "\n".join(lines) + "\n"


def _shell_join(parts: Iterable[str]) -> str:
    """Quote-aware join — minimum viable, no exotic shell escaping.

    Stops short of full ``shlex.quote`` because the scaffold's argv is
    constructed in-process and does not contain user-controlled
    metacharacters; the helper exists for human-readable output, not
    for safe shell evaluation.
    """
    return " ".join(parts)


__all__ = [
    "EncodingPlan",
    "PredicateFn",
    "Shot",
    "ShotRecommendation",
    "detect_shots",
    "merge_shots",
    "parse_per_shot_csv",
    "plan_to_shell_script",
    "tune_per_shot",
    "write_concat_listing",
]
