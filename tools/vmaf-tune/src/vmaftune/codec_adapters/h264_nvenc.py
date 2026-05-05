# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""NVIDIA NVENC H.264 codec adapter.

Hardware-accelerated AVC encoder exposed through FFmpeg's ``h264_nvenc``
encoder. Uses the constant-quantizer (``-cq``) knob as the closest
analogue to libx264's CRF; preset names map onto NVENC's ``p1``..``p7``
levels via the shared mnemonic table in ``_nvenc_common``.
"""

from __future__ import annotations

import dataclasses

from . import _nvenc_common as _nvc


@dataclasses.dataclass(frozen=True)
class H264NvencAdapter:
    """NVIDIA NVENC H.264 single-pass CQ adapter."""

    name: str = "h264_nvenc"
    encoder: str = "h264_nvenc"
    quality_knob: str = "cq"
    quality_range: tuple[int, int] = _nvc.NVENC_CQ_RANGE
    quality_default: int = 23
    invert_quality: bool = True  # higher CQ = lower quality

    presets: tuple[str, ...] = _nvc.NVENC_PRESETS

    def validate(self, preset: str, cq: int) -> None:
        """Raise ``ValueError`` if ``(preset, cq)`` is unsupported."""
        _nvc.validate_nvenc(preset, cq)

    def nvenc_preset(self, preset: str) -> str:
        """Translate a mnemonic preset to its NVENC ``pN`` name."""
        return _nvc.nvenc_preset(preset)
