# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Codec adapter registry.

Per ADR-0237, every codec exposes a different parameter shape; the
harness must not branch on codec identity in the search loop. Each
adapter declares its quality knob, range, defaults, and FFmpeg encoder
name. Phase A wires ``libx264`` plus the NVIDIA NVENC family
(``h264_nvenc``, ``hevc_nvenc``, ``av1_nvenc``), the AMD AMF family
(``h264_amf``, ``hevc_amf``, ``av1_amf``), the Intel QSV family
(``h264_qsv``, ``hevc_qsv``, ``av1_qsv``), the Apple VideoToolbox
family (``h264_videotoolbox``, ``hevc_videotoolbox``), the Fraunhofer
VVenC H.266 encoder (``libvvenc``), and the SVT-AV1 software encoder
(``libsvtav1``) — software and hardware encoders share the same
adapter contract; later phases add one file per codec without
touching the search loop.

Mnemonic preset names (``ultrafast``..``placebo``) are normalised
across software and hardware encoders. NVENC's seven hardware presets
(``p1``..``p7``) collapse the ten mnemonic names per the table in
``_nvenc_common``: ``ultrafast``/``superfast``/``veryfast`` → ``p1``,
``faster`` → ``p2``, ``fast`` → ``p3``, ``medium`` → ``p4``,
``slow`` → ``p5``, ``slower`` → ``p6``, ``slowest``/``placebo`` →
``p7``.
"""

from __future__ import annotations

from typing import Protocol

from .av1_amf import AV1AMFAdapter
from .av1_nvenc import Av1NvencAdapter
from .av1_qsv import Av1QsvAdapter
from .h264_amf import H264AMFAdapter
from .h264_nvenc import H264NvencAdapter
from .h264_qsv import H264QsvAdapter
from .h264_videotoolbox import H264VideoToolboxAdapter
from .hevc_amf import HEVCAMFAdapter
from .hevc_nvenc import HevcNvencAdapter
from .hevc_qsv import HevcQsvAdapter
from .hevc_videotoolbox import HEVCVideoToolboxAdapter
from .libaom import LibaomAdapter
from .svtav1 import SvtAv1Adapter
from .vvenc import VVenCAdapter
from .x264 import X264Adapter
from .x265 import X265Adapter


class CodecAdapter(Protocol):
    """Codec-adapter contract (ADR-0237 Phase A + ADR-0294 dispatcher).

    The encode dispatcher (``encode.run_encode``) consumes the
    runtime-shaped subset (``encoder``, ``ffmpeg_codec_args``,
    ``extra_params``) and never branches on the adapter's ``name`` —
    that's the invariant that lets new codec adapters drop in
    one-PR-at-a-time without touching the search loop.
    """

    name: str
    encoder: str
    quality_knob: str
    quality_range: tuple[int, int]
    quality_default: int
    invert_quality: bool
    # Bumps when the adapter's argv shape / preset list / quality
    # window changes — see ADR-0298 (vmaf-tune cache key).
    adapter_version: str

    # Predictor probe-encode knobs (see _gop_common docstring).
    # The per-shot predictor consumes these to run one fast probe encode
    # per shot and read its bitrate as the complexity barometer.
    probe_preset: str
    probe_quality: int
    supports_qpfile: bool

    def ffmpeg_codec_args(self, preset: str, quality: int) -> list[str]:
        """FFmpeg ``-c:v ...`` argv slice for one encode."""
        ...

    def extra_params(self) -> tuple[str, ...]:
        """Additional non-codec argv (e.g. tile flags). May be empty."""
        ...

    def validate(self, preset: str, quality: int) -> None:
        """Raise ``ValueError`` if ``(preset, quality)`` is unsupported."""
        ...

    def gop_args(self, keyint: int, min_keyint: int | None = None) -> tuple[str, ...]:
        """FFmpeg argv slice that pins the GOP / keyint. Empty tuple = leave default."""
        ...

    def force_keyframes_args(self, timestamps: tuple[float, ...]) -> tuple[str, ...]:
        """FFmpeg argv slice that pins keyframes at the given seconds. Empty tuple = no override."""
        ...

    def probe_args(self) -> list[str]:
        """FFmpeg argv slice for a fast probe encode (predictor complexity barometer)."""
        ...


_REGISTRY: dict[str, CodecAdapter] = {
    "libx264": X264Adapter(),
    "libaom-av1": LibaomAdapter(),
    "libx265": X265Adapter(),
    "h264_nvenc": H264NvencAdapter(),
    "hevc_nvenc": HevcNvencAdapter(),
    "av1_nvenc": Av1NvencAdapter(),
    "h264_amf": H264AMFAdapter(),
    "hevc_amf": HEVCAMFAdapter(),
    "av1_amf": AV1AMFAdapter(),
    "h264_qsv": H264QsvAdapter(),
    "hevc_qsv": HevcQsvAdapter(),
    "av1_qsv": Av1QsvAdapter(),
    "h264_videotoolbox": H264VideoToolboxAdapter(),
    "hevc_videotoolbox": HEVCVideoToolboxAdapter(),
    "libvvenc": VVenCAdapter(),
    "libsvtav1": SvtAv1Adapter(),
}


def get_adapter(name: str) -> CodecAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown codec {name!r}; known codecs: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known_codecs() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


__all__ = [
    "AV1AMFAdapter",
    "Av1NvencAdapter",
    "Av1QsvAdapter",
    "CodecAdapter",
    "H264AMFAdapter",
    "H264NvencAdapter",
    "H264QsvAdapter",
    "H264VideoToolboxAdapter",
    "HEVCAMFAdapter",
    "HEVCVideoToolboxAdapter",
    "HevcNvencAdapter",
    "HevcQsvAdapter",
    "LibaomAdapter",
    "SvtAv1Adapter",
    "VVenCAdapter",
    "X264Adapter",
    "X265Adapter",
    "get_adapter",
    "known_codecs",
]
