# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Intel QSV AV1 codec adapter (`av1_qsv`).

ICQ-mode single-pass encodes via ``-global_quality``; same preset
vocabulary as the H.264 / HEVC QSV adapters.

Hardware availability is narrower than the H.264 / HEVC encoders —
AV1 fixed-function encode requires either a 12th-gen+ Intel iGPU
(Alder Lake-N and newer with the AV1 block) or Arc / Battlemage
discrete GPUs. Older silicon will register the encoder but fail at
runtime with a libmfx ``MFX_ERR_UNSUPPORTED`` style diagnostic; the
``ffmpeg_supports_encoder`` probe in ``_qsv_common`` catches the
build-time variant of that mismatch.
"""

from __future__ import annotations

import dataclasses

from . import _gop_common
from ._qsv_common import (
    QSV_PRESETS,
    QSV_QUALITY_DEFAULT,
    QSV_QUALITY_RANGE,
    preset_to_qsv,
    validate_global_quality,
)


@dataclasses.dataclass(frozen=True)
class Av1QsvAdapter:
    """Intel QSV AV1 single-pass ICQ adapter."""

    name: str = "av1_qsv"
    encoder: str = "av1_qsv"
    quality_knob: str = "global_quality"
    quality_range: tuple[int, int] = QSV_QUALITY_RANGE
    quality_default: int = QSV_QUALITY_DEFAULT
    invert_quality: bool = True

    probe_preset: str = "veryfast"
    probe_quality: int = 23
    supports_qpfile: bool = False
    # ADR-0332: this encoder has no parseable first-pass stats file.
    supports_encoder_stats: bool = False

    presets: tuple[str, ...] = QSV_PRESETS

    def validate(self, preset: str, quality: int) -> None:
        """Raise ``ValueError`` if ``(preset, quality)`` is unsupported."""
        preset_to_qsv(preset)
        validate_global_quality(quality)

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg argv slice for ``av1_qsv`` ICQ-mode encode."""
        return [
            "-c:v",
            self.encoder,
            "-preset",
            preset_to_qsv(preset),
            "-global_quality",
            str(quality),
        ]

    def extra_params(self) -> tuple[str, ...]:
        """No additional non-codec argv for av1_qsv."""
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
