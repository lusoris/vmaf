# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""NVIDIA NVENC HEVC codec adapter.

Hardware-accelerated H.265 encoder exposed through FFmpeg's
``hevc_nvenc`` encoder. Mirrors the H.264 NVENC adapter shape: shared
preset table, shared CQ window, shared validation. The only
distinguishing field is the FFmpeg encoder name.
"""

from __future__ import annotations

import dataclasses

from . import _gop_common
from . import _nvenc_common as _nvc


@dataclasses.dataclass(frozen=True)
class HevcNvencAdapter:
    """NVIDIA NVENC HEVC single-pass CQ adapter."""

    name: str = "hevc_nvenc"
    encoder: str = "hevc_nvenc"
    quality_knob: str = "cq"
    quality_range: tuple[int, int] = _nvc.NVENC_CQ_RANGE
    quality_default: int = 23
    invert_quality: bool = True

    probe_preset: str = "ultrafast"
    probe_quality: int = 28
    supports_qpfile: bool = False
    qpfile_format: str = "none"

    presets: tuple[str, ...] = _nvc.NVENC_PRESETS

    def validate(self, preset: str, cq: int) -> None:
        """Raise ``ValueError`` if ``(preset, cq)`` is unsupported."""
        _nvc.validate_nvenc(preset, cq)

    def nvenc_preset(self, preset: str) -> str:
        """Translate a mnemonic preset to its NVENC ``pN`` name."""
        return _nvc.nvenc_preset(preset)

    def gop_args(self, keyint: int, min_keyint: int | None = None) -> tuple[str, ...]:
        """FFmpeg ``-g`` / ``-keyint_min``, honoured by NVENC."""
        return _gop_common.default_gop_args(keyint, min_keyint)

    def force_keyframes_args(self, timestamps: tuple[float, ...]) -> tuple[str, ...]:
        """FFmpeg ``-force_key_frames`` plus NVENC's ``-forced-idr 1``."""
        if not timestamps:
            return ()
        return _gop_common.default_force_keyframes_args(timestamps) + ("-forced-idr", "1")

    def probe_args(self) -> list[str]:
        """Predictor probe-encode argv: NVENC ``p1`` preset, fixed CQ."""
        return [
            "-c:v",
            self.encoder,
            "-preset",
            self.nvenc_preset(self.probe_preset),
            "-cq",
            str(self.probe_quality),
        ]
