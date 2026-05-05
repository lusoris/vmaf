# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Intel QSV HEVC codec adapter (`hevc_qsv`).

ICQ-mode single-pass encodes via ``-global_quality``; same preset
vocabulary as ``h264_qsv``.

Hardware availability: 7th-gen+ Intel iGPU (Kaby Lake and newer; HEVC
10-bit needs Tiger Lake / 11th-gen+) or Arc / Battlemage discrete GPUs.
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
class HevcQsvAdapter:
    """Intel QSV HEVC single-pass ICQ adapter."""

    name: str = "hevc_qsv"
    encoder: str = "hevc_qsv"
    quality_knob: str = "global_quality"
    quality_range: tuple[int, int] = QSV_QUALITY_RANGE
    quality_default: int = QSV_QUALITY_DEFAULT
    invert_quality: bool = True

    presets: tuple[str, ...] = QSV_PRESETS

    def validate(self, preset: str, quality: int) -> None:
        """Raise ``ValueError`` if ``(preset, quality)`` is unsupported."""
        preset_to_qsv(preset)
        validate_global_quality(quality)
