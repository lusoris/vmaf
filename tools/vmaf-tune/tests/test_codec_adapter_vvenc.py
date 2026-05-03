# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Smoke tests for the libvvenc (VVC / H.266 + NN-VC) codec adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.codec_adapters import VVenCAdapter, get_adapter, known_codecs  # noqa: E402
from vmaftune.codec_adapters.vvenc import native_presets  # noqa: E402
from vmaftune.encode import EncodeRequest, build_ffmpeg_command  # noqa: E402


def test_vvenc_registered_in_registry():
    assert "libvvenc" in known_codecs()
    a = get_adapter("libvvenc")
    assert a.encoder == "libvvenc"
    assert a.quality_knob == "qp"
    assert a.invert_quality is True
    lo, hi = a.quality_range
    assert 0 <= lo < hi <= 63


def test_vvenc_native_preset_compresses_seven_to_five():
    a = VVenCAdapter()
    # The fork's canonical superset compresses onto VVenC's 5 levels.
    assert a.native_preset("placebo") == "slower"
    assert a.native_preset("slowest") == "slower"
    assert a.native_preset("slower") == "slower"
    assert a.native_preset("slow") == "slow"
    assert a.native_preset("medium") == "medium"
    assert a.native_preset("fast") == "fast"
    assert a.native_preset("faster") == "faster"
    assert a.native_preset("veryfast") == "faster"
    assert a.native_preset("superfast") == "faster"
    assert a.native_preset("ultrafast") == "faster"


def test_vvenc_native_preset_vocabulary_matches_encoder():
    # The five preset names VVenC's CLI / FFmpeg wrapper accept.
    assert native_presets() == ("faster", "fast", "medium", "slow", "slower")


def test_vvenc_native_preset_rejects_unknown():
    a = VVenCAdapter()
    with pytest.raises(ValueError):
        a.native_preset("warpspeed")


def test_vvenc_validate_rejects_bad_inputs():
    a = VVenCAdapter()
    a.validate("medium", 32)
    with pytest.raises(ValueError):
        a.validate("warpspeed", 32)
    with pytest.raises(ValueError):
        a.validate("medium", -1)
    with pytest.raises(ValueError):
        a.validate("medium", 64)


def test_vvenc_quality_range_is_h266_canonical_window():
    a = VVenCAdapter()
    lo, hi = a.quality_range
    # Default sits inside the surfaced window.
    assert lo <= a.quality_default <= hi


def test_vvenc_extra_params_default_off():
    a = VVenCAdapter()
    assert a.extra_params() == ()


def test_vvenc_extra_params_nnvc_intra_emits_vvenc_params():
    a = VVenCAdapter(nnvc_intra=True)
    extras = a.extra_params()
    # FFmpeg-libvvenc surfaces VVenC config keys via -vvenc-params key=value.
    assert "-vvenc-params" in extras
    idx = extras.index("-vvenc-params")
    payload = extras[idx + 1]
    assert "IntraNN=1" in payload


def test_vvenc_ffmpeg_command_carries_native_preset_and_qp():
    # The harness validates / projects via the adapter, then composes the
    # ffmpeg argv via build_ffmpeg_command. We check the wired surface.
    a = VVenCAdapter(nnvc_intra=True)
    a.validate("slower", 27)
    req = EncodeRequest(
        source=Path("/tmp/ref.yuv"),
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        framerate=24.0,
        encoder=a.encoder,
        preset=a.native_preset("slower"),
        crf=27,  # encoder-agnostic name; carries QP for VVenC
        output=Path("/tmp/out.mkv"),
        extra_params=a.extra_params(),
    )
    cmd = build_ffmpeg_command(req)
    assert "-c:v" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libvvenc"
    assert "-preset" in cmd
    assert cmd[cmd.index("-preset") + 1] == "slower"
    # build_ffmpeg_command emits -crf for the shared scalar quality knob;
    # VVenC's wrapper consumes the integer regardless of label, and the
    # NNVC tool toggle rides through extra_params as -vvenc-params.
    assert "-vvenc-params" in cmd
    nnvc_idx = cmd.index("-vvenc-params")
    assert "IntraNN=1" in cmd[nnvc_idx + 1]
