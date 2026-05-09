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

``BaseQsvAdapter`` is a frozen dataclass that supplies the full method
body shared by all three per-codec adapters. Each per-codec class
inherits from it and overrides only the ``name`` / ``encoder`` fields.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess

from . import _gop_common

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


@dataclasses.dataclass(frozen=True)
class BaseQsvAdapter:
    """Shared implementation for the QSV H.264 / HEVC / AV1 adapters.

    Subclasses override only the two codec-identity fields (``name`` and
    ``encoder``). All method bodies are identical across the three
    per-codec adapters — they live here so the per-codec files each
    collapse to ~15 LOC of field declarations.
    """

    name: str = "h264_qsv"
    encoder: str = "h264_qsv"
    quality_knob: str = "global_quality"
    quality_range: tuple[int, int] = QSV_QUALITY_RANGE
    quality_default: int = QSV_QUALITY_DEFAULT
    invert_quality: bool = True  # higher global_quality = lower quality

    # Predictor probe-encode knobs. QSV has no "ultrafast"; the QSV
    # preset vocabulary tops out at "veryfast".
    probe_preset: str = "veryfast"
    probe_quality: int = 23
    supports_qpfile: bool = False
    # ADR-0332: hardware encoders have no parseable first-pass stats file.
    supports_encoder_stats: bool = False

    presets: tuple[str, ...] = QSV_PRESETS

    def validate(self, preset: str, quality: int) -> None:
        """Raise ``ValueError`` if ``(preset, quality)`` is unsupported."""
        preset_to_qsv(preset)
        validate_global_quality(quality)

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg argv slice for ICQ-mode QSV encode.

        QSV's quality knob is ``-global_quality`` (not ``-crf``); the
        preset vocabulary maps identity-style onto FFmpeg's QSV bridge
        names per :func:`preset_to_qsv`.
        """
        return [
            "-c:v",
            self.encoder,
            "-preset",
            preset_to_qsv(preset),
            "-global_quality",
            str(quality),
        ]

    def extra_params(self) -> tuple[str, ...]:
        """No additional non-codec argv for QSV encoders."""
        return ()

    def gop_args(self, keyint: int, min_keyint: int | None = None) -> tuple[str, ...]:
        """FFmpeg ``-g`` / ``-keyint_min``, honoured by QSV."""
        return _gop_common.default_gop_args(keyint, min_keyint)

    def force_keyframes_args(self, timestamps: tuple[float, ...]) -> tuple[str, ...]:
        """FFmpeg ``-force_key_frames`` with comma-separated seconds."""
        return _gop_common.default_force_keyframes_args(timestamps)

    def probe_args(self) -> list[str]:
        """Predictor probe-encode argv: QSV ``veryfast`` preset, fixed ICQ."""
        return [
            "-c:v",
            self.encoder,
            "-preset",
            preset_to_qsv(self.probe_preset),
            "-global_quality",
            str(self.probe_quality),
        ]
