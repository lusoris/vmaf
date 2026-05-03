# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""vmaf-tune — quality-aware encode automation harness (Phase A).

Phase A ships a grid-sweep corpus generator over libx264. The JSONL
schema emitted by `corpus.py` is the API contract Phase B (target-VMAF
bisect) and Phase C (per-title CRF predictor) will consume; bump
``SCHEMA_VERSION`` if a row's keys / value semantics change.

See ``docs/adr/0237-quality-aware-encode-automation.md`` for the
roadmap and ``docs/research/0044-quality-aware-encode-automation.md``
for the option-space digest.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Bump on any backward-incompatible row-schema change.
SCHEMA_VERSION = 1

# Canonical row-key tuple — exposed so tests, downstream loaders, and
# the Phase B bisect can verify the contract programmatically.
CORPUS_ROW_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "timestamp",
    "src",
    "src_sha256",
    "width",
    "height",
    "pix_fmt",
    "framerate",
    "duration_s",
    "encoder",
    "encoder_version",
    "preset",
    "crf",
    "extra_params",
    "encode_path",
    "encode_size_bytes",
    "bitrate_kbps",
    "encode_time_ms",
    "vmaf_score",
    "vmaf_model",
    "score_time_ms",
    "ffmpeg_version",
    "vmaf_binary_version",
    "exit_status",
)

__all__ = ["CORPUS_ROW_KEYS", "SCHEMA_VERSION", "__version__"]
