# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Shared helpers for Apple VideoToolbox codec adapters.

VideoToolbox is Apple's hardware encoder API on macOS. FFmpeg exposes
two encoders backed by it: ``h264_videotoolbox`` and
``hevc_videotoolbox``. AV1 is not available on Apple Silicon hardware
as of 2026 and is intentionally not surfaced here.

Quality knob
============

Both VT encoders accept ``-q:v`` on the ``[0, 100]`` integer scale
(higher = better quality). This is **different** from x264 / x265's
CRF axis (``[0, 51]``, lower = better quality):

| Codec                  | Knob   | Range     | Direction         |
|------------------------|--------|-----------|-------------------|
| libx264 / libx265      | crf    | 0..51     | lower = better    |
| h264_videotoolbox      | q:v    | 0..100    | higher = better   |
| hevc_videotoolbox      | q:v    | 0..100    | higher = better   |

The corpus row's ``crf`` column carries whatever the harness emitted —
adapters with ``invert_quality=False`` and a different range simply
fill the same slot with their native scale. Downstream consumers read
``encoder`` + ``crf`` together and interpret the knob via the adapter
registry, so the schema stays single-shape.

Preset mapping
==============

VideoToolbox has no x264-style preset axis. The closest knob is the
``-realtime`` flag (boolean): ``-realtime 1`` trades quality for
latency, ``-realtime 0`` (default) targets non-real-time quality.
We map the standard nine-name preset taxonomy onto these two buckets
so callers can drive every codec from one preset list:

  ultrafast / superfast / veryfast / faster / fast → ``-realtime 1``
  medium / slow / slower / veryslow                 → ``-realtime 0``

The mapping is intentionally lossy — VT cannot expose a finer dial.
For per-title encoding the ``q:v`` axis carries the bulk of the
search-space signal; preset mostly affects throughput, not quality.
"""

from __future__ import annotations

# Preset names shared with libx264 / libx265 so callers can use a single
# preset list across codecs. Maps to ``-realtime`` boolean.
_FAST_PRESETS: frozenset[str] = frozenset({"ultrafast", "superfast", "veryfast", "faster", "fast"})
_QUALITY_PRESETS: frozenset[str] = frozenset({"medium", "slow", "slower", "veryslow"})

VIDEOTOOLBOX_PRESETS: tuple[str, ...] = (
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

# VideoToolbox's ``-q:v`` axis. Higher is better; the perceptually
# informative window for both encoders sits in roughly [40, 80] —
# below 40 the output is mostly artefact, above 80 the bitrate
# explodes. The harness still accepts the full hardware range; the
# Phase A grid generator should default to a narrower band.
VIDEOTOOLBOX_QUALITY_RANGE: tuple[int, int] = (0, 100)
VIDEOTOOLBOX_QUALITY_DEFAULT: int = 50


def preset_to_realtime(preset: str) -> str:
    """Translate a preset name to the ``-realtime`` argv value.

    Raises ``ValueError`` for an unknown preset; the adapter's
    ``validate()`` is the upstream gate so this is a defensive check.
    """
    if preset in _FAST_PRESETS:
        return "1"
    if preset in _QUALITY_PRESETS:
        return "0"
    raise ValueError(
        f"unknown VideoToolbox preset {preset!r}; expected one of {VIDEOTOOLBOX_PRESETS}"
    )


def validate_videotoolbox(preset: str, q: int) -> None:
    """Shared validator for both VT adapters.

    The ``q`` argument is the harness's ``crf`` slot; for VT it carries
    the ``-q:v`` value on the ``[0, 100]`` scale.
    """
    if preset not in VIDEOTOOLBOX_PRESETS:
        raise ValueError(
            f"unknown VideoToolbox preset {preset!r}; expected one of " f"{VIDEOTOOLBOX_PRESETS}"
        )
    lo, hi = VIDEOTOOLBOX_QUALITY_RANGE
    if not lo <= q <= hi:
        raise ValueError(f"q:v {q} outside VideoToolbox range [{lo}, {hi}]")
