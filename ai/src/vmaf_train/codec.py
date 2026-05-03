# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Codec-id vocabulary + one-hot encoding for the codec-aware FR regressor.

Distortion signatures from software (x264, x265, libsvtav1, libaom)
and hardware (NVENC, QSV, AMF, VideoToolbox) encoders differ
systematically. Conditioning the FR MLP on a codec id lifts
cross-codec PLCC/SROCC by 1–3 points on multi-codec corpora — see the
2026 Bristol VI-Lab review §5.3 cited in
[ADR-0235](../../docs/adr/0235-codec-aware-fr-regressor.md) and
[Research-0040](../../docs/research/0040-codec-aware-fr-conditioning.md).

Schema versions:

  * v1 (6 slots) — software-only: x264, x265, libsvtav1, libvvenc,
    libvpx-vp9, unknown. Used by ``fr_regressor_v1.onnx`` (today the
    v1 ONNX is single-input and does not consume a codec one-hot, but
    the vocabulary slot count was specified in the v1 sidecar). The
    tuple ``CODEC_VOCAB_V1`` preserves the v1 ordering for back-compat.
  * v2 (16 slots, ADR-0284) — software + hardware-aware: x264, x265,
    libsvtav1, libaom, h264_nvenc, hevc_nvenc, av1_nvenc, h264_qsv,
    hevc_qsv, av1_qsv, h264_amf, hevc_amf, av1_amf, h264_videotoolbox,
    hevc_videotoolbox, reserved. Used by ``fr_regressor_v2_hw.onnx``.

The active vocabulary (``CODEC_VOCAB``) is v2. Adding a codec is a
schema bump; existing trained models pin the version they were trained
against via the ``codec_vocab_version`` field in the model-card
sidecar, so a re-training is required when this list grows.

The fallback bucket is the trailing ``"reserved"`` index — corpora
that ship raw distorted YUVs without codec metadata (e.g. the Netflix
Public corpus under ``ai/scripts/extract_full_features.py``) bucket to
``reserved`` rather than guessing.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Order is load-bearing — index 0..N-1 matches the one-hot column index
# baked into trained ONNX. Do not reorder; append new entries at the
# end and bump ``CODEC_VOCAB_VERSION``.
#
# Schema v2 (16 slots, ADR-0284) widens the v1 software-only vocabulary
# (6 slots, indices 0..5) with a contiguous block of hardware encoder
# buckets so an ``fr_regressor_v2_hw`` checkpoint can distinguish
# software vs. NVENC vs. QSV vs. AMF vs. VideoToolbox distortion
# signatures. Index 15 is reserved for a future codec without
# requiring a third schema bump.
#
# v2 layout (indices):
#   0: x264             (was v1 idx 0)
#   1: x265             (was v1 idx 1)
#   2: libsvtav1        (was v1 idx 2)
#   3: libaom           (NEW — software AV1, formerly aliased to libsvtav1)
#   4: h264_nvenc       (NEW — NVIDIA NVENC H.264)
#   5: hevc_nvenc       (NEW — NVIDIA NVENC HEVC)
#   6: av1_nvenc        (NEW — NVIDIA NVENC AV1, Ada+ only)
#   7: h264_qsv         (NEW — Intel Quick Sync H.264)
#   8: hevc_qsv         (NEW — Intel Quick Sync HEVC)
#   9: av1_qsv          (NEW — Intel Quick Sync AV1, Arc+ only)
#  10: h264_amf         (NEW — AMD AMF H.264)
#  11: hevc_amf         (NEW — AMD AMF HEVC)
#  12: av1_amf          (NEW — AMD AMF AV1, RDNA3+ only)
#  13: h264_videotoolbox (NEW — Apple VideoToolbox H.264)
#  14: hevc_videotoolbox (NEW — Apple VideoToolbox HEVC)
#  15: reserved          (placeholder for the next codec; aliases to
#                         "unknown" until a real label is assigned)
#
# v1 indices for ``libvvenc`` (was 3), ``libvpx-vp9`` (was 4), and
# ``unknown`` (was 5) are NOT preserved positionally — v2 is a fresh
# schema, not an extension. Existing ``fr_regressor_v1.onnx`` callers
# keep using the v1 vocabulary via the ``CODEC_VOCAB_V1`` tuple below;
# v2-trained checkpoints pin ``CODEC_VOCAB_VERSION = 2`` in their
# sidecar JSON.
CODEC_VOCAB_V1: tuple[str, ...] = (
    "x264",
    "x265",
    "libsvtav1",
    "libvvenc",
    "libvpx-vp9",
    "unknown",
)

CODEC_VOCAB: tuple[str, ...] = (
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
CODEC_VOCAB_VERSION = 2
NUM_CODECS = len(CODEC_VOCAB)
# ``unknown`` is no longer a vocabulary slot in v2 — unrecognised
# labels bucket to the trailing ``reserved`` index so the schema stays
# closed-set without a special "unknown" dimension. The constant is
# kept for backward-compat with code that imports it.
UNKNOWN_INDEX = CODEC_VOCAB.index("reserved")


def codec_index(name: str | None) -> int:
    """Return the canonical index for ``name`` or ``UNKNOWN_INDEX`` on miss.

    Empty / None / unrecognised codec names all bucket to the
    ``reserved`` slot so feature dumps never fail on novel labels —
    the model still receives a deterministic one-hot.
    """
    if name is None:
        return UNKNOWN_INDEX
    name = name.strip().lower()
    if not name:
        return UNKNOWN_INDEX
    # Common alias cleanup so feature-dump scripts that tag from
    # ``ffprobe -show_entries stream=codec_name`` (which emits e.g.
    # "h264", "hevc", "av1", "vp9") map to a software encoder bucket
    # by default. Hardware-encoder labels (``h264_nvenc`` etc.) are
    # already canonical and pass through unchanged.
    aliases = {
        "h264": "x264",
        "avc": "x264",
        "libx264": "x264",
        "hevc": "x265",
        "h265": "x265",
        "libx265": "x265",
        "av1": "libsvtav1",
        # v1-only labels collapse to the closest v2 bucket:
        # libvvenc (VVC) and libvpx-vp9 are not first-class in v2;
        # they fall through to ``reserved``.
    }
    name = aliases.get(name, name)
    if name not in CODEC_VOCAB:
        return UNKNOWN_INDEX
    return CODEC_VOCAB.index(name)


def codec_one_hot(name: str | None, dtype: np.dtype = np.float32) -> np.ndarray:
    """One-hot ``(NUM_CODECS,)`` vector for a single codec label."""
    out = np.zeros(NUM_CODECS, dtype=dtype)
    out[codec_index(name)] = 1.0
    return out


def codec_one_hot_batch(names: Iterable[str | None], dtype: np.dtype = np.float32) -> np.ndarray:
    """One-hot ``(N, NUM_CODECS)`` matrix for a sequence of codec labels."""
    names = list(names)
    out = np.zeros((len(names), NUM_CODECS), dtype=dtype)
    for i, n in enumerate(names):
        out[i, codec_index(n)] = 1.0
    return out
