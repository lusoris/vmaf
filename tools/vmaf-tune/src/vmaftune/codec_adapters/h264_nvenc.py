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

from . import _gop_common
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

    # Predictor probe-encode knobs. "ultrafast" maps to NVENC ``p1``.
    probe_preset: str = "ultrafast"
    probe_quality: int = 28
    supports_qpfile: bool = False

    presets: tuple[str, ...] = _nvc.NVENC_PRESETS

    def validate(self, preset: str, cq: int) -> None:
        """Raise ``ValueError`` if ``(preset, cq)`` is unsupported."""
        _nvc.validate_nvenc(preset, cq)

    def nvenc_preset(self, preset: str) -> str:
        """Translate a mnemonic preset to its NVENC ``pN`` name."""
        return _nvc.nvenc_preset(preset)

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg argv slice for ``h264_nvenc`` constant-quality.

        NVENC uses ``-cq`` (not ``-crf``) and its native ``pN`` preset
        token (not the libx264 mnemonic); :func:`_nvc.nvenc_preset`
        collapses the canonical mnemonic onto NVENC's seven levels.
        """
        return [
            "-c:v",
            self.encoder,
            "-preset",
            self.nvenc_preset(preset),
            "-cq",
            str(quality),
        ]

    def extra_params(self) -> tuple[str, ...]:
        """No additional non-codec argv for h264_nvenc."""
        return ()

    def gop_args(self, keyint: int, min_keyint: int | None = None) -> tuple[str, ...]:
        """FFmpeg ``-g`` / ``-keyint_min``, honoured by NVENC."""
        return _gop_common.default_gop_args(keyint, min_keyint)

    def force_keyframes_args(self, timestamps: tuple[float, ...]) -> tuple[str, ...]:
        """FFmpeg ``-force_key_frames`` plus NVENC's ``-forced-idr 1``.

        NVENC honours ``-force_key_frames`` only when ``-forced-idr 1`` is
        also set; otherwise the encoder may emit non-IDR keyframes that
        downstream players treat as non-seekable.
        """
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
