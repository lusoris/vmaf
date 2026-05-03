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

    presets: tuple[str, ...] = QSV_PRESETS

    def validate(self, preset: str, quality: int) -> None:
        """Raise ``ValueError`` if ``(preset, quality)`` is unsupported."""
        preset_to_qsv(preset)
        validate_global_quality(quality)
