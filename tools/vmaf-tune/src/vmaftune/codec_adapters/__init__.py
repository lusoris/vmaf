# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Codec adapter registry.

Per ADR-0237, every codec exposes a different parameter shape; the
harness must not branch on codec identity in the search loop. Each
adapter declares its quality knob, range, defaults, and FFmpeg encoder
name. Phase A wires ``libx264`` plus the NVIDIA NVENC family
(``h264_nvenc``, ``hevc_nvenc``, ``av1_nvenc``) — software and
hardware encoders share the same adapter contract; later phases add
one file per codec without touching the search loop.

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

from .av1_nvenc import Av1NvencAdapter
from .h264_nvenc import H264NvencAdapter
from .hevc_nvenc import HevcNvencAdapter
from .libaom import LibaomAdapter
from .x264 import X264Adapter
from .x265 import X265Adapter


class CodecAdapter(Protocol):
    """Phase A codec-adapter contract (subset of the ADR-0237 sketch).

    Phase B+ extends with two-pass / log-parsing / per-shot emit hooks.
    """

    name: str
    encoder: str
    quality_knob: str
    quality_range: tuple[int, int]
    quality_default: int
    invert_quality: bool


_REGISTRY: dict[str, CodecAdapter] = {
    "libx264": X264Adapter(),
    "libaom-av1": LibaomAdapter(),
    "libx265": X265Adapter(),
    "h264_nvenc": H264NvencAdapter(),
    "hevc_nvenc": HevcNvencAdapter(),
    "av1_nvenc": Av1NvencAdapter(),
}


def get_adapter(name: str) -> CodecAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown codec {name!r}; known codecs: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known_codecs() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


__all__ = [
    "Av1NvencAdapter",
    "CodecAdapter",
    "H264NvencAdapter",
    "HevcNvencAdapter",
    "LibaomAdapter",
    "X264Adapter",
    "X265Adapter",
    "get_adapter",
    "known_codecs",
]
