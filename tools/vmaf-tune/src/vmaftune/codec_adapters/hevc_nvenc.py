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

    presets: tuple[str, ...] = _nvc.NVENC_PRESETS

    def validate(self, preset: str, cq: int) -> None:
        """Raise ``ValueError`` if ``(preset, cq)`` is unsupported."""
        _nvc.validate_nvenc(preset, cq)

    def nvenc_preset(self, preset: str) -> str:
        """Translate a mnemonic preset to its NVENC ``pN`` name."""
        return _nvc.nvenc_preset(preset)
