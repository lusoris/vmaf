# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Unit tests for ``ai/scripts/chug_to_corpus_jsonl.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "ai" / "scripts" / "chug_to_corpus_jsonl.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("chug_to_corpus_jsonl", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHUG = _load_module()


def _make_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Video,mos_j,sos_j,ref,name,bitladder,resolution,bitrate,"
                "orientation,framerate,content_name,height,width",
                "abc123,50.0,2.5,0,360p_0.2M_src.mp4,360p_0.2M_,360p,0.2M,"
                "Landscape,30.0,src.mp4,360,640",
                "def456,75.0,1.5,1,1080p_5M_ref.mp4,1080p_5M_,1080p,5M,"
                "Portrait,29.97,ref.mp4,1080,1920",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_manifest_maps_chug_mos_to_one_to_five(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _make_manifest(manifest)

    rows = CHUG.parse_manifest_csv(manifest, min_rows=1)

    assert len(rows) == 2
    assert rows[0]["filename"] == "abc123.mp4"
    assert rows[0]["url"].endswith("/abc123.mp4")
    assert rows[0]["mos_raw_0_100"] == 50.0
    assert rows[0]["mos"] == 3.0
    assert rows[1]["chug_orientation"] == "Portrait"


class _FakeRunner:
    def __call__(self, cmd, **_kwargs):
        argv0 = Path(cmd[0]).name
        if argv0 == "curl":
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake mp4")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if argv0 == "ffprobe":
            payload = {
                "streams": [
                    {
                        "width": 640,
                        "height": 360,
                        "r_frame_rate": "30/1",
                        "avg_frame_rate": "30/1",
                        "duration": "10.0",
                        "pix_fmt": "yuv420p10le",
                        "codec_name": "hevc",
                    }
                ],
                "format": {"duration": "10.0"},
            }
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")


def test_run_writes_chug_jsonl_with_hdr_metadata(tmp_path: Path) -> None:
    chug_dir = tmp_path / "chug"
    manifest = chug_dir / "manifest.csv"
    output = chug_dir / "chug.jsonl"
    _make_manifest(manifest)

    stats = CHUG.run(
        chug_dir=chug_dir,
        output=output,
        min_csv_rows=1,
        max_rows=1,
        runner=_FakeRunner(),
        now_fn=lambda: "2026-05-14T00:00:00+00:00",
    )

    assert stats.written == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["corpus"] == "chug"
    assert row["pix_fmt"] == "yuv420p10le"
    assert row["mos"] == 3.0
    assert row["mos_raw_0_100"] == 50.0
    assert row["chug_bitladder"] == "360p_0.2M_"
