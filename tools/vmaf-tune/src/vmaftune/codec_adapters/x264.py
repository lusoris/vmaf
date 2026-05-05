# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""libx264 codec adapter — Phase A.

Single-pass CRF-mode encodes; the eight standard x264 presets;
quality range pinned to the canonical CRF window for which VMAF is
informative. Out-of-range CRFs are accepted but logged.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class X264Adapter:
    """libx264 single-pass CRF adapter."""

    name: str = "libx264"
    encoder: str = "libx264"
    quality_knob: str = "crf"
    # Bumps when the adapter's argv shape / preset list / CRF window
    # changes. Folded into the cache key so an adapter upgrade
    # invalidates older entries (ADR-0298).
    adapter_version: str = "1"
    # x264 nominally accepts 0..51; we surface the perceptually
    # informative window — ADR-0237 Phase A grid generation lives here.
    quality_range: tuple[int, int] = (15, 40)
    quality_default: int = 23
    invert_quality: bool = True  # higher CRF = lower quality

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
    )

    def validate(self, preset: str, crf: int) -> None:
        """Raise ``ValueError`` if ``(preset, crf)`` is unsupported."""
        if preset not in self.presets:
            raise ValueError(f"unknown x264 preset {preset!r}; expected one of {self.presets}")
        lo, hi = self.quality_range
        if not lo <= crf <= hi:
            raise ValueError(f"crf {crf} outside Phase A range [{lo}, {hi}]")

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg argv slice for libx264 single-pass CRF.

        Adapter-contract entry point used by the codec-agnostic
        dispatcher (ADR-0294). Identical to the legacy hard-coded
        x264 path: ``-c:v libx264 -preset <p> -crf <q>``.
        """
        return ["-c:v", self.encoder, "-preset", preset, "-crf", str(quality)]

    def extra_params(self) -> tuple[str, ...]:
        """Additional ffmpeg argv slices (none for libx264 Phase A)."""
        return ()
