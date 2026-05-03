# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Apple VideoToolbox HEVC codec adapter.

FFmpeg encoder name: ``hevc_videotoolbox``. Hardware-accelerated HEVC
encoding on Apple Silicon (M-series) and Intel Macs with a T2 chip.
See ``_videotoolbox_common.py`` for the shared quality/preset mapping.
"""

from __future__ import annotations

import dataclasses

from ._videotoolbox_common import (
    VIDEOTOOLBOX_PRESETS,
    VIDEOTOOLBOX_QUALITY_DEFAULT,
    VIDEOTOOLBOX_QUALITY_RANGE,
    validate_videotoolbox,
)


@dataclasses.dataclass(frozen=True)
class HEVCVideoToolboxAdapter:
    """``hevc_videotoolbox`` adapter — hardware HEVC on Apple Silicon."""

    name: str = "hevc_videotoolbox"
    encoder: str = "hevc_videotoolbox"
    quality_knob: str = "q:v"
    quality_range: tuple[int, int] = VIDEOTOOLBOX_QUALITY_RANGE
    quality_default: int = VIDEOTOOLBOX_QUALITY_DEFAULT
    invert_quality: bool = False  # higher q:v = higher quality

    presets: tuple[str, ...] = VIDEOTOOLBOX_PRESETS

    def validate(self, preset: str, crf: int) -> None:
        """Raise ``ValueError`` if ``(preset, q:v)`` is unsupported.

        The ``crf`` parameter carries the harness's quality value; for
        VideoToolbox this is the ``-q:v`` integer on the [0, 100] axis.
        """
        validate_videotoolbox(preset, crf)
