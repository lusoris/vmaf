# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Codec adapter registry.

Per ADR-0237, every codec exposes a different parameter shape; the
harness must not branch on codec identity in the search loop. Each
adapter declares its quality knob, range, defaults, and FFmpeg encoder
name. Phase A wires only ``libx264``; later phases add one file per
codec without touching the search loop.
"""

from __future__ import annotations

from typing import Protocol

from .x264 import X264Adapter


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
}


def get_adapter(name: str) -> CodecAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown codec {name!r}; phase A wires {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known_codecs() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


__all__ = ["CodecAdapter", "X264Adapter", "get_adapter", "known_codecs"]
