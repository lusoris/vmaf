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

_PRESET_TO_REALTIME = {
    "ultrafast": 1,
    "superfast": 1,
    "veryfast": 1,
    "faster": 1,
    "fast": 1,
    "medium": 0,
    "slow": 0,
    "slower": 0,
    "veryslow": 0,
}


@dataclasses.dataclass(frozen=True)
class HEVCVideoToolboxAdapter:
    """``hevc_videotoolbox`` adapter — hardware HEVC on Apple Silicon."""

    name: str = "hevc_videotoolbox"
    encoder: str = "hevc_videotoolbox"
    quality_knob: str = "q:v"
    # Bumps when the adapter's argv shape / preset list / quality
    # window changes (ADR-0298 cache key).
    adapter_version: str = "1"
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

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg argv slice for ``hevc_videotoolbox``.

        Maps the nine-name preset onto VT's binary ``-realtime`` flag.
        """
        realtime = _PRESET_TO_REALTIME[preset]
        return [
            "-c:v",
            self.encoder,
            "-realtime",
            str(realtime),
            "-q:v",
            str(quality),
        ]

    def extra_params(self) -> tuple[str, ...]:
        """No additional non-codec argv for VideoToolbox."""
        return ()
