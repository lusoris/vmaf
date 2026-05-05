# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Shared helpers for Intel QSV codec adapters.

The three QSV encoders (`h264_qsv`, `hevc_qsv`, `av1_qsv`) share the
same preset vocabulary and the same ICQ-mode quality knob
(`global_quality`). This module factors that shared shape out so each
codec adapter is a thin dataclass that pins only what differs (encoder
name, hardware-availability matrix).

QSV preset names map identically to x264-style names (`veryslow`
through `veryfast`) — the FFmpeg QSV bridge accepts the seven names
verbatim. ``preset_to_qsv`` is therefore an identity check that raises
on unknown inputs rather than a translation table.

ICQ rate control (``-global_quality N``) accepts integers in
``[1, 51]`` per the libmfx / VPL spec; values outside that window are
clipped to the encoder's preferred default by the driver, but
``vmaf-tune`` rejects them up-front so corpus rows stay reproducible.
"""

from __future__ import annotations

import shutil
import subprocess

# QSV preset vocabulary — identical to x264's medium/fast/... subset
# but without the libx264-specific `ultrafast` / `superfast` levels.
QSV_PRESETS: tuple[str, ...] = (
    "veryslow",
    "slower",
    "slow",
    "medium",
    "fast",
    "faster",
    "veryfast",
)

# `global_quality` ICQ window — full libmfx / VPL accepted range.
QSV_QUALITY_RANGE: tuple[int, int] = (1, 51)
QSV_QUALITY_DEFAULT: int = 23


def preset_to_qsv(preset: str) -> str:
    """Identity-map a preset name; raise ``ValueError`` if unknown.

    Kept as a function (not a constant lookup) so callers get a single
    error path that matches the other codec adapters.
    """
    if preset not in QSV_PRESETS:
        raise ValueError(f"unknown QSV preset {preset!r}; expected one of {QSV_PRESETS}")
    return preset


def validate_global_quality(value: int) -> None:
    """Raise ``ValueError`` if ``value`` is outside the ICQ window."""
    lo, hi = QSV_QUALITY_RANGE
    if not lo <= value <= hi:
        raise ValueError(f"global_quality {value} outside ICQ range [{lo}, {hi}]")


def ffmpeg_supports_encoder(
    encoder: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
) -> bool:
    """Probe whether ``ffmpeg -encoders`` advertises ``encoder``.

    Returns ``False`` when ``ffmpeg_bin`` is missing on ``PATH`` or
    when the encoder line is not present in the listing. ``runner`` is
    parameterised so unit tests can inject a stub without spawning a
    real process — same pattern as ``encode.run_encode`` /
    ``score.run_score``.
    """
    if runner is None and shutil.which(ffmpeg_bin) is None:
        return False
    runner_fn = runner or subprocess.run
    try:
        completed = runner_fn(  # type: ignore[operator]
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    stdout = getattr(completed, "stdout", "") or ""
    # Encoder lines look like ` V..... h264_qsv             H.264 / ... `
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == encoder:
            return True
    return False


def require_qsv_encoder(
    encoder: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
) -> None:
    """Raise ``RuntimeError`` if FFmpeg cannot drive ``encoder``.

    Caller-side helper for the eventual encode wiring (out of Phase A
    scope but pinned here so adapters can self-validate when the
    corpus pipeline grows multi-codec support).
    """
    if not ffmpeg_supports_encoder(encoder, ffmpeg_bin=ffmpeg_bin, runner=runner):
        raise RuntimeError(
            f"ffmpeg does not advertise {encoder!r}; rebuild with libmfx / "
            "VPL enabled or use a vendor build that includes Intel QSV"
        )
