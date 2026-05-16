# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Regression test: chug_bit_depth must survive the _load_jsonl_metadata keep filter.

Fix 1 from the 2026-05-16 CHUG-extractor audit — data corruption on HDR clips.
The `keep` tuple in `_load_jsonl_metadata` previously omitted `chug_bit_depth`,
causing the dict comprehension to strip it. `_geometry_from_sidecar` then saw
`None` and fell back to `pix_fmt="yuv420p"` (8-bit), silently truncating every
10-bit HDR CHUG clip.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "ai" / "scripts" / "extract_k150k_features.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_k150k_features", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


K150K = _load_module()


def test_chug_bit_depth_survives_keep_filter(tmp_path: Path) -> None:
    """chug_bit_depth=10 in a sidecar JSONL row must appear in the loaded metadata."""
    sidecar = tmp_path / "meta.jsonl"
    rows = [
        {
            "src": "clip_a.mp4",
            "chug_bit_depth": 10,
            "chug_content_name": "scene1",
            "chug_width_manifest": 3840,
            "chug_height_manifest": 2160,
            "chug_framerate_manifest": 24.0,
            "mos_raw_0_100": 72.5,
        },
        {
            "src": "clip_b.mp4",
            "chug_bit_depth": 10,
            "chug_content_name": "scene2",
            "chug_width_manifest": 1920,
            "chug_height_manifest": 1080,
            "chug_framerate_manifest": 30.0,
            "mos_raw_0_100": 55.0,
        },
    ]
    sidecar.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    meta = K150K._load_jsonl_metadata(sidecar, split_seed="test-seed")

    assert "clip_a.mp4" in meta, "clip_a.mp4 not found in metadata"
    assert "clip_b.mp4" in meta, "clip_b.mp4 not found in metadata"

    assert meta["clip_a.mp4"]["chug_bit_depth"] == 10, (
        "chug_bit_depth was stripped from clip_a.mp4 by the keep filter — "
        "HDR clips would be silently treated as 8-bit"
    )
    assert meta["clip_b.mp4"]["chug_bit_depth"] == 10, (
        "chug_bit_depth was stripped from clip_b.mp4 by the keep filter — "
        "HDR clips would be silently treated as 8-bit"
    )
