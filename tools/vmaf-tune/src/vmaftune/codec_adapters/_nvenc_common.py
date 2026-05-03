# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Shared helpers for the NVIDIA NVENC family of codec adapters.

Hardware encoders share a common preset / quality vocabulary that
differs from libx264's. Centralising it here keeps the per-codec files
(``h264_nvenc``, ``hevc_nvenc``, ``av1_nvenc``) thin and forces a
single source of truth for the preset map and the constant-quality
knob range.

NVENC presets are named ``p1`` (fastest) through ``p7`` (slowest, best
quality). The fork's codec-adapter contract surfaces the libx264-style
mnemonic preset names (``ultrafast``...``placebo``); this module owns
the canonical mnemonic-to-NVENC mapping so all three NVENC codecs
agree.
"""

from __future__ import annotations

from typing import Final

# Canonical NVENC quality knob is constant-quantizer (``-cq``); the
# valid integer range is 0..51, mirroring x264 CRF semantics. We
# surface the same perceptually-informative window as the libx264
# adapter for cross-codec corpus consistency.
NVENC_CQ_RANGE: Final[tuple[int, int]] = (15, 40)

# Full hardware-allowed CQ window — used for the validate path; the
# Phase A grid stays inside ``NVENC_CQ_RANGE`` but adapters reject
# anything outside the hardware window outright.
NVENC_CQ_HARD_LIMITS: Final[tuple[int, int]] = (0, 51)

# Mnemonic preset names accepted by every NVENC adapter, ordered fast
# to slow. ``placebo`` is included as an alias of ``p7`` so callers
# can reuse libx264-style sweeps unchanged; NVENC has no slower mode.
NVENC_PRESETS: Final[tuple[str, ...]] = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "slowest",
    "placebo",
)

# Mnemonic → NVENC ``-preset pN`` mapping. NVENC has 7 levels; the
# 10 mnemonic names collapse onto them per the comment in
# ``codec_adapters/__init__.py``. The convention is to clamp the fast
# end at p1 and the slow end at p7.
_NVENC_PRESET_MAP: Final[dict[str, str]] = {
    "ultrafast": "p1",
    "superfast": "p1",
    "veryfast": "p1",
    "faster": "p2",
    "fast": "p3",
    "medium": "p4",
    "slow": "p5",
    "slower": "p6",
    "slowest": "p7",
    "placebo": "p7",
}


def nvenc_preset(name: str) -> str:
    """Translate a mnemonic preset name to an NVENC ``pN`` string.

    Raises ``ValueError`` if the name is not recognised. Pure
    function — no I/O — so tests can pin the mapping.
    """
    if name not in _NVENC_PRESET_MAP:
        raise ValueError(f"unknown NVENC preset {name!r}; expected one of {NVENC_PRESETS}")
    return _NVENC_PRESET_MAP[name]


def validate_nvenc(preset: str, cq: int) -> None:
    """Common validation for any NVENC adapter.

    Rejects an unknown preset name or a ``-cq`` value outside the
    hardware window ``[0, 51]``. Out-of-Phase-A-window values
    (``< 15`` or ``> 40``) are accepted here — the adapter callers
    decide how strict they want to be — but kept available via
    ``NVENC_CQ_RANGE`` for grid generation.
    """
    if preset not in NVENC_PRESETS:
        raise ValueError(f"unknown NVENC preset {preset!r}; expected one of {NVENC_PRESETS}")
    lo, hi = NVENC_CQ_HARD_LIMITS
    if not lo <= cq <= hi:
        raise ValueError(f"cq {cq} outside NVENC range [{lo}, {hi}]")


__all__ = [
    "NVENC_CQ_HARD_LIMITS",
    "NVENC_CQ_RANGE",
    "NVENC_PRESETS",
    "nvenc_preset",
    "validate_nvenc",
]
