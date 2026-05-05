# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Intel QSV H.264 codec adapter (`h264_qsv`).

ICQ-mode single-pass encodes via ``-global_quality``; the seven QSV
presets share the same names as x264's medium-and-down subset.

Hardware availability: 7th-gen+ Intel iGPU (Kaby Lake and newer) or
Arc / Battlemage discrete GPUs. ``ffmpeg`` must be built with libmfx
or VPL support — probe via ``_qsv_common.ffmpeg_supports_encoder``.
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
class H264QsvAdapter:
    """Intel QSV H.264 single-pass ICQ adapter."""

    name: str = "h264_qsv"
    encoder: str = "h264_qsv"
    quality_knob: str = "global_quality"
    quality_range: tuple[int, int] = QSV_QUALITY_RANGE
    quality_default: int = QSV_QUALITY_DEFAULT
    invert_quality: bool = True  # higher global_quality = lower quality

    presets: tuple[str, ...] = QSV_PRESETS

    def validate(self, preset: str, quality: int) -> None:
        """Raise ``ValueError`` if ``(preset, quality)`` is unsupported."""
        preset_to_qsv(preset)
        validate_global_quality(quality)
