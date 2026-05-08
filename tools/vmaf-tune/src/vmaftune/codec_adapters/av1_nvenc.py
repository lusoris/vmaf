# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""NVIDIA NVENC AV1 codec adapter.

Hardware-accelerated AV1 encoder exposed through FFmpeg's
``av1_nvenc`` encoder. Available only on NVIDIA Ada Lovelace
generation hardware (RTX 40-series and the L40 / L4 server parts) and
later — older GPUs return "Encoder not found" at FFmpeg invocation
time. Adapter shape is identical to ``h264_nvenc`` / ``hevc_nvenc``;
the hardware-availability check belongs at run time, not in the
adapter contract.
"""

from __future__ import annotations

import dataclasses

from . import _gop_common
from . import _nvenc_common as _nvc


@dataclasses.dataclass(frozen=True)
class Av1NvencAdapter:
    """NVIDIA NVENC AV1 single-pass CQ adapter (Ada+ hardware only)."""

    name: str = "av1_nvenc"
    encoder: str = "av1_nvenc"
    quality_knob: str = "cq"
    quality_range: tuple[int, int] = _nvc.NVENC_CQ_RANGE
    quality_default: int = 23
    invert_quality: bool = True

    probe_preset: str = "ultrafast"
    probe_quality: int = 28
    supports_qpfile: bool = False
    # ADR-0332: this encoder has no parseable first-pass stats file.
    supports_encoder_stats: bool = False

    presets: tuple[str, ...] = _nvc.NVENC_PRESETS

    def validate(self, preset: str, cq: int) -> None:
        """Raise ``ValueError`` if ``(preset, cq)`` is unsupported."""
        _nvc.validate_nvenc(preset, cq)

    def nvenc_preset(self, preset: str) -> str:
        """Translate a mnemonic preset to its NVENC ``pN`` name."""
        return _nvc.nvenc_preset(preset)

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg argv slice for ``av1_nvenc`` constant-quality."""
        return [
            "-c:v",
            self.encoder,
            "-preset",
            self.nvenc_preset(preset),
            "-cq",
            str(quality),
        ]

    def extra_params(self) -> tuple[str, ...]:
        """No additional non-codec argv for av1_nvenc."""
        return ()

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
