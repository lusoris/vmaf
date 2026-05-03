# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""16-slot codec one-hot schema gate (ADR-0284).

Locks in the v2 vocabulary order and guarantees every codec name
registered under ``vmaftune.codec_adapters`` maps to a stable column
index in ``ai/src/vmaf_train/codec.py``. Reordering the vocabulary
silently invalidates every shipped ``fr_regressor_v2_*.onnx``; this
gate fails the moment the ordering drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "ai" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ai" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("numpy")

from vmaftune.codec_adapters import known_codecs  # noqa: E402

from vmaf_train.codec import (  # noqa: E402
    CODEC_VOCAB,
    CODEC_VOCAB_VERSION,
    NUM_CODECS,
    codec_index,
    codec_one_hot,
)

# Canonical v2 ordering — load-bearing. Every shipped
# ``fr_regressor_v2_*.onnx`` bakes this column index into its first
# Linear layer's weight tensor; reordering = silent model corruption.
EXPECTED_V2_ORDER = (
    "x264",
    "x265",
    "libsvtav1",
    "libaom",
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "av1_qsv",
    "h264_amf",
    "hevc_amf",
    "av1_amf",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "reserved",
)


def test_vocab_is_v2_with_16_slots():
    assert CODEC_VOCAB_VERSION == 2
    assert NUM_CODECS == 16
    assert CODEC_VOCAB == EXPECTED_V2_ORDER


def test_vocab_indices_are_stable_per_label():
    # Pin every (label, index) pair so any future reorder fails loudly.
    expected = {label: i for i, label in enumerate(EXPECTED_V2_ORDER)}
    for label, idx in expected.items():
        assert codec_index(label) == idx, label


def test_each_registered_adapter_resolves_to_a_dedicated_slot():
    # Every codec in the harness's registry must resolve to a non-
    # ``reserved`` slot — otherwise the model can't tell it apart.
    reserved_idx = CODEC_VOCAB.index("reserved")
    for name in known_codecs():
        # x264 lives under encoder name "libx264" in the harness;
        # the codec.py alias table maps it back to "x264".
        idx = codec_index(name)
        assert 0 <= idx < NUM_CODECS, f"{name} → {idx} out of range"
        assert idx != reserved_idx, (
            f"adapter {name!r} resolves to the reserved slot — "
            "add it to CODEC_VOCAB or extend the alias table"
        )


def test_one_hot_for_every_v2_label():
    for i, label in enumerate(EXPECTED_V2_ORDER):
        v = codec_one_hot(label)
        assert v.shape == (NUM_CODECS,)
        assert v.sum() == 1.0
        assert v[i] == 1.0


def test_unknown_label_buckets_to_reserved():
    assert codec_index("totally-not-a-codec") == CODEC_VOCAB.index("reserved")
    assert codec_index(None) == CODEC_VOCAB.index("reserved")


def test_hardware_codec_labels_pass_through_canonically():
    # NVENC / QSV / AMF / VideoToolbox labels are already canonical and
    # must NOT pass through the alias table to a software bucket.
    for name in (
        "h264_nvenc",
        "hevc_nvenc",
        "av1_nvenc",
        "h264_qsv",
        "hevc_qsv",
        "av1_qsv",
        "h264_amf",
        "hevc_amf",
        "av1_amf",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    ):
        assert codec_index(name) == CODEC_VOCAB.index(name)
