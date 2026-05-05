# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Smoke tests for the libaom-av1 codec adapter.

Mocks ``subprocess`` style — no actual ffmpeg / libaom binary required.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.codec_adapters import LibaomAdapter, get_adapter, known_codecs  # noqa: E402


def test_libaom_is_registered():
    assert "libaom-av1" in known_codecs()
    a = get_adapter("libaom-av1")
    assert isinstance(a, LibaomAdapter)
    assert a.encoder == "libaom-av1"
    assert a.quality_knob == "crf"
    assert a.invert_quality is True


def test_libaom_quality_range_is_full_av1_window():
    a = LibaomAdapter()
    assert a.quality_range == (0, 63)
    # Default sits in the perceptually informative middle.
    lo, hi = a.quality_range
    assert lo <= a.quality_default <= hi


def test_libaom_preset_to_cpu_used_mapping_is_complete():
    a = LibaomAdapter()
    expected = {
        "placebo": 0,
        "slowest": 1,
        "slower": 2,
        "slow": 3,
        "medium": 4,
        "fast": 5,
        "faster": 6,
        "veryfast": 7,
        "superfast": 8,
        "ultrafast": 9,
    }
    # Adapter's preset tuple covers every documented mapping key.
    assert set(a.presets) == set(expected.keys())
    for preset, cpu_used in expected.items():
        assert a.cpu_used(preset) == cpu_used


def test_libaom_cpu_used_rejects_unknown_preset():
    a = LibaomAdapter()
    with pytest.raises(ValueError, match="unknown libaom preset"):
        a.cpu_used("nope")


def test_libaom_validate_accepts_in_range():
    a = LibaomAdapter()
    a.validate("medium", 0)
    a.validate("medium", 63)
    a.validate("placebo", 35)
    a.validate("ultrafast", 1)


def test_libaom_validate_rejects_unknown_preset():
    a = LibaomAdapter()
    with pytest.raises(ValueError, match="unknown libaom preset"):
        a.validate("turbo", 35)


def test_libaom_validate_rejects_out_of_range_crf():
    a = LibaomAdapter()
    with pytest.raises(ValueError, match=r"crf -1 outside libaom range"):
        a.validate("medium", -1)
    with pytest.raises(ValueError, match=r"crf 64 outside libaom range"):
        a.validate("medium", 64)
    with pytest.raises(ValueError, match=r"crf 200 outside libaom range"):
        a.validate("medium", 200)


def test_libaom_ffmpeg_codec_args_shape():
    a = LibaomAdapter()
    # cpu-used 4 (= medium) at crf 35.
    args = a.ffmpeg_codec_args("medium", 35)
    assert args == ("-crf", "35", "-cpu-used", "4", "-an")


def test_libaom_ffmpeg_codec_args_slowest_preset():
    a = LibaomAdapter()
    # placebo collapses to cpu-used 0 (slowest / highest quality).
    args = a.ffmpeg_codec_args("placebo", 20)
    assert args == ("-crf", "20", "-cpu-used", "0", "-an")


def test_libaom_ffmpeg_codec_args_fastest_preset():
    a = LibaomAdapter()
    args = a.ffmpeg_codec_args("ultrafast", 50)
    assert args == ("-crf", "50", "-cpu-used", "9", "-an")


def test_libaom_ffmpeg_codec_args_validates_inputs():
    a = LibaomAdapter()
    with pytest.raises(ValueError):
        a.ffmpeg_codec_args("medium", 99)
    with pytest.raises(ValueError):
        a.ffmpeg_codec_args("turbo", 35)


def test_libaom_adapter_is_frozen_dataclass():
    # Adapters are passed around the search loop as keys; mutating them
    # mid-sweep would be a bug. Phase A x264 adapter is frozen; libaom
    # follows the same contract.
    a = LibaomAdapter()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.name = "other"  # type: ignore[misc]
