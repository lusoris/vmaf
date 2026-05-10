# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Smoke tests for vmaf-roi-score Option C scaffold.

Mocks the ``vmaf`` subprocess so no binaries are required. Pins:

- the combine-math is a pure linear blend on Python ``float``;
- the JSON output schema (key order, schema_version);
- the synthetic-mask code path produces a score *between* the two
  underlying VMAF runs (the contract the user-facing docstring
  promises).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmafroiscore import ROI_RESULT_KEYS, SCHEMA_VERSION, blend_scores  # noqa: E402
from vmafroiscore.cli import _build_parser, main  # noqa: E402
from vmafroiscore.mask import synthesise_uniform_mask  # noqa: E402
from vmafroiscore.score import ScoreRequest, build_vmaf_command, parse_vmaf_json  # noqa: E402

# ---------------------------------------------------------------------------
# combine math


def test_blend_endpoints_are_pure_passthrough():
    assert blend_scores(80.0, 60.0, 0.0) == 80.0
    assert blend_scores(80.0, 60.0, 1.0) == 60.0


def test_blend_midpoint_is_arithmetic_mean():
    assert blend_scores(80.0, 60.0, 0.5) == 70.0


def test_blend_is_between_inputs_for_any_weight():
    """Contract: roi-vmaf is always between the two underlying scores."""
    a, b = 92.5, 41.25
    lo, hi = min(a, b), max(a, b)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = blend_scores(a, b, w)
        assert lo <= out <= hi, f"w={w} produced {out} outside [{lo}, {hi}]"


def test_blend_rejects_out_of_range_weight():
    with pytest.raises(ValueError):
        blend_scores(80.0, 60.0, 1.5)
    with pytest.raises(ValueError):
        blend_scores(80.0, 60.0, -0.1)


def test_blend_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        blend_scores(float("nan"), 60.0, 0.5)
    with pytest.raises(ValueError):
        blend_scores(80.0, float("inf"), 0.5)


# ---------------------------------------------------------------------------
# vmaf-cli wiring (pure-function pieces)


def test_build_vmaf_command_shape():
    req = ScoreRequest(
        reference=Path("/tmp/ref.yuv"),
        distorted=Path("/tmp/dis.yuv"),
        width=576,
        height=324,
        pix_fmt="yuv420p",
    )
    cmd = build_vmaf_command(req, Path("/tmp/out.json"))
    assert cmd[0] == "vmaf"
    assert "--reference" in cmd
    assert "/tmp/ref.yuv" in cmd
    assert "--distorted" in cmd
    assert "/tmp/dis.yuv" in cmd
    assert "--pixel_format" in cmd
    pf_idx = cmd.index("--pixel_format")
    assert cmd[pf_idx + 1] == "420"


def test_parse_vmaf_json_modern_shape():
    payload = {"pooled_metrics": {"vmaf": {"mean": 87.5, "min": 80.0}}}
    assert parse_vmaf_json(payload) == 87.5


def test_parse_vmaf_json_legacy_shape():
    payload = {"VMAF score": 72.5}
    assert parse_vmaf_json(payload) == 72.5


def test_parse_vmaf_json_missing_raises():
    with pytest.raises(ValueError):
        parse_vmaf_json({})


# ---------------------------------------------------------------------------
# argparse


def test_parser_accepts_minimum_args(tmp_path: Path):
    ref = tmp_path / "ref.yuv"
    dis = tmp_path / "dis.yuv"
    ref.write_bytes(b"\x80" * 16)
    dis.write_bytes(b"\x80" * 16)
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "--reference",
            str(ref),
            "--distorted",
            str(dis),
            "--width",
            "576",
            "--height",
            "324",
        ]
    )
    assert ns.saliency_model is None
    assert ns.synthetic_mask is None


# ---------------------------------------------------------------------------
# synthetic-mask helper


def test_synthesise_uniform_mask_shape_and_value():
    m = synthesise_uniform_mask(4, 3, fill=0.7)
    assert len(m) == 3
    assert len(m[0]) == 4
    assert all(v == 0.7 for row in m for v in row)


def test_synthesise_uniform_mask_rejects_bad_args():
    with pytest.raises(ValueError):
        synthesise_uniform_mask(4, 3, fill=1.5)
    with pytest.raises(ValueError):
        synthesise_uniform_mask(0, 3, fill=0.5)


# ---------------------------------------------------------------------------
# end-to-end smoke (synthetic-mask path) — mocks subprocess


class _FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_cli_synthetic_smoke(monkeypatch, tmp_path: Path):
    """Drive ``main`` end-to-end with a mocked vmaf binary.

    Verifies:
    - exit code is 0;
    - JSON output has the canonical key order;
    - the blended score equals the (identical) underlying scores;
    - the schema_version matches the package constant.
    """
    ref = tmp_path / "ref.yuv"
    dis = tmp_path / "dis.yuv"
    ref.write_bytes(b"\x80" * 16)
    dis.write_bytes(b"\x80" * 16)
    out = tmp_path / "result.json"

    def _fake_run(cmd, capture_output=False, text=False, check=False):
        # Find the --output path argparse handed us and drop a JSON
        # there in the modern pooled_metrics shape.
        out_idx = cmd.index("--output")
        Path(cmd[out_idx + 1]).write_text(
            json.dumps({"pooled_metrics": {"vmaf": {"mean": 87.5}}}),
            encoding="utf-8",
        )
        return _FakeCompleted(0, stderr="VMAF version: smoke-mock\n")

    monkeypatch.setattr("vmafroiscore.score.subprocess.run", _fake_run)

    rc = main(
        [
            "--reference",
            str(ref),
            "--distorted",
            str(dis),
            "--width",
            "576",
            "--height",
            "324",
            "--synthetic-mask",
            "0.5",
            "--weight",
            "0.7",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert tuple(payload.keys()) == ROI_RESULT_KEYS
    assert payload["schema_version"] == SCHEMA_VERSION
    # Synthetic-mask path scores the same YUV twice; both legs are 87.5
    # so the blend is also 87.5 regardless of weight.
    assert payload["vmaf_full"] == 87.5
    assert payload["vmaf_masked"] == 87.5
    assert payload["vmaf_roi"] == 87.5
    assert payload["weight"] == 0.7
    assert payload["saliency_model"] == "synthetic"


def test_cli_saliency_model_is_scaffolded_only(monkeypatch, tmp_path: Path):
    """The --saliency-model path is wired to a clear error today."""
    ref = tmp_path / "ref.yuv"
    dis = tmp_path / "dis.yuv"
    fake_model = tmp_path / "saliency.onnx"
    ref.write_bytes(b"\x80" * 16)
    dis.write_bytes(b"\x80" * 16)
    fake_model.write_bytes(b"")  # presence is enough; cli only checks exists()

    def _fake_run(cmd, capture_output=False, text=False, check=False):
        out_idx = cmd.index("--output")
        Path(cmd[out_idx + 1]).write_text(
            json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}),
            encoding="utf-8",
        )
        return _FakeCompleted(0, stderr="VMAF version: smoke-mock\n")

    monkeypatch.setattr("vmafroiscore.score.subprocess.run", _fake_run)

    rc = main(
        [
            "--reference",
            str(ref),
            "--distorted",
            str(dis),
            "--width",
            "576",
            "--height",
            "324",
            "--saliency-model",
            str(fake_model),
        ]
    )
    # Exit code 64 = data-not-ready (mask materialiser deferred).
    assert rc == 64
