# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""libx265 codec adapter — ADR-0288.

Mirrors :mod:`vmaftune.codec_adapters.x264` shape; differs only in the
codec-specific defaults (ten preset levels including ``placebo``,
default profile derived from pixel format). Phase A still drives the
ffmpeg subprocess via :mod:`vmaftune.encode`; this adapter contributes
metadata + per-codec validation only.
"""

from __future__ import annotations

import dataclasses

# Pixel-format → x265 profile mapping. Keys cover the common YUV
# fixture shapes the harness ingests; unmapped formats fall back to
# ``main`` (8-bit) to keep the adapter forgiving.
_PROFILE_BY_PIX_FMT: dict[str, str] = {
    "yuv420p": "main",
    "yuv422p": "main422-8",
    "yuv444p": "main444-8",
    "yuv420p10le": "main10",
    "yuv422p10le": "main422-10",
    "yuv444p10le": "main444-10",
    "yuv420p12le": "main12",
}


@dataclasses.dataclass(frozen=True)
class X265Adapter:
    """libx265 single-pass CRF adapter.

    Defaults match the ADR-0237 Phase A grid window for x264; the
    perceptually-informative CRF range is identical (HEVC's CRF axis
    is linear-perceptual on the same 0..51 scale).
    """

    name: str = "libx265"
    encoder: str = "libx265"
    quality_knob: str = "crf"
    # x265 nominally accepts 0..51; surface the same Phase A informative
    # window as x264 so the search loop is uniform across codecs.
    quality_range: tuple[int, int] = (15, 40)
    quality_default: int = 28  # x265 default; ~visually-lossless on most content
    invert_quality: bool = True  # higher CRF = lower quality

    # x265 ships ten presets — one more than x264 (adds ``placebo``).
    presets: tuple[str, ...] = (
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    )

    def validate(self, preset: str, crf: int) -> None:
        """Raise ``ValueError`` if ``(preset, crf)`` is unsupported."""
        if preset not in self.presets:
            raise ValueError(f"unknown x265 preset {preset!r}; expected one of {self.presets}")
        lo, hi = self.quality_range
        if not lo <= crf <= hi:
            raise ValueError(f"crf {crf} outside Phase A range [{lo}, {hi}]")

    def profile_for(self, pix_fmt: str) -> str:
        """Return the canonical x265 profile string for ``pix_fmt``.

        Falls back to ``main`` (8-bit 4:2:0) for unknown pixel formats —
        callers that need a stricter behaviour should consult
        :data:`_PROFILE_BY_PIX_FMT` directly.
        """
        return _PROFILE_BY_PIX_FMT.get(pix_fmt, "main")
