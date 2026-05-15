# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Unit tests for ``ai/scripts/extract_k150k_features.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
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


def test_cuda_feature_passes_split_gpu_and_cpu_residual(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append([str(part) for part in cmd])
        out = Path(cmd[cmd.index("--output") + 1])
        names = [cmd[idx + 1] for idx, part in enumerate(cmd) if part == "--feature"]
        metrics = {}
        if "adm_cuda" in names:
            metrics["integer_adm2"] = 1.0
            metrics["integer_vif_scale0"] = 2.0
            metrics["integer_motion2"] = 3.0
            metrics["psnr_y"] = 72.0
            metrics["ciede2000"] = 0.0
            metrics["float_ms_ssim"] = 1.0
            metrics["psnr_hvs"] = 99.0
            metrics["ssimulacra2"] = 100.0
            assert "--backend" in cmd
            assert cmd[cmd.index("--backend") + 1] == "cuda"
            assert "float_ssim_cuda" not in names
            assert "cambi_cuda" not in names
        else:
            metrics["float_ssim"] = 0.9
            metrics["cambi"] = 0.1
            assert "--no_cuda" in cmd
        out.write_text(json.dumps({"frames": [{"metrics": metrics}]}), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(K150K.subprocess, "run", fake_run)

    frames = K150K._run_feature_passes(
        vmaf_bin=Path("libvmaf/build-cuda/tools/vmaf"),
        cpu_vmaf_bin=Path("build-cpu/tools/vmaf"),
        yuv_path=tmp_path / "clip.yuv",
        width=1280,
        height=720,
        pix_fmt="yuv420p10le",
        out_json=tmp_path / "clip.json",
        threads=2,
        use_cuda=True,
    )

    assert len(calls) == 2
    assert frames == [
        {
            "integer_adm2": 1.0,
            "integer_vif_scale0": 2.0,
            "integer_motion2": 3.0,
            "psnr_y": 72.0,
            "ciede2000": 0.0,
            "float_ms_ssim": 1.0,
            "psnr_hvs": 99.0,
            "ssimulacra2": 100.0,
            "float_ssim": 0.9,
            "cambi": 0.1,
        }
    ]


def test_cpu_feature_pass_uses_generic_extractors(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append([str(part) for part in cmd])
        out = Path(cmd[cmd.index("--output") + 1])
        names = [cmd[idx + 1] for idx, part in enumerate(cmd) if part == "--feature"]
        assert names == list(K150K.EXTRACTOR_NAMES)
        assert "--no_cuda" in cmd
        out.write_text(json.dumps({"frames": [{"metrics": {"vmaf": 100.0}}]}), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(K150K.subprocess, "run", fake_run)

    frames = K150K._run_feature_passes(
        vmaf_bin=Path("build-cpu/tools/vmaf"),
        cpu_vmaf_bin=Path("build-cpu/tools/vmaf"),
        yuv_path=tmp_path / "clip.yuv",
        width=640,
        height=360,
        pix_fmt="yuv420p",
        out_json=tmp_path / "clip.json",
        threads=1,
        use_cuda=False,
    )

    assert len(calls) == 1
    assert frames == [{"vmaf": 100.0}]


def test_jsonl_metadata_preserves_chug_content_split(tmp_path: Path) -> None:
    meta_jsonl = tmp_path / "chug.jsonl"
    meta_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "src": "clip-a.mp4",
                        "mos_raw_0_100": 72.5,
                        "chug_video_id": "clip-a",
                        "chug_content_name": "source-a.mp4",
                        "chug_bitladder": "720p_1M_",
                    }
                ),
                json.dumps(
                    {
                        "src": "clip-b.mp4",
                        "chug_content_name": "source-b.mp4",
                        "split": "test",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = K150K._load_jsonl_metadata(meta_jsonl, split_seed="stable")

    assert metadata["clip-a.mp4"]["mos_raw_0_100"] == 72.5
    assert metadata["clip-a.mp4"]["chug_video_id"] == "clip-a"
    assert metadata["clip-a.mp4"]["chug_split_key"] == "source-a.mp4"
    assert metadata["clip-a.mp4"]["split"] in {"train", "val", "test"}
    assert metadata["clip-b.mp4"]["split"] == "test"
