# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""CLI wiring tests for ``vmaf-tune auto`` (Phase F.1, ADR-0325).

These tests cover the argparse surface only — the per-phase
sequential composition is covered by ``test_auto.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune.cli import main  # noqa: E402


def test_auto_help_lists_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """``vmaf-tune auto --help`` documents every documented F.1 flag."""
    with pytest.raises(SystemExit) as excinfo:
        main(["auto", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr().out
    for flag in (
        "--source",
        "--target-vmaf",
        "--max-budget-kbps",
        "--allow-codecs",
        "--smoke",
        "--output",
    ):
        assert flag in captured, f"{flag} missing from --help"


def test_auto_smoke_emits_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``--smoke`` emits a parseable JSON document on stdout."""
    rc = main(
        [
            "auto",
            "--source",
            "/dev/null",
            "--target-vmaf",
            "90",
            "--max-budget-kbps",
            "10000",
            "--allow-codecs",
            "libx264",
            "--smoke",
        ]
    )
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["smoke"] is True
    assert payload["target_vmaf"] == 90.0
    assert payload["max_budget_kbps"] == 10000.0
    assert payload["allow_codecs"] == ["libx264"]
    assert payload["winner"] is not None
    # Exit 0 when a winner exists.
    assert rc == 0


def test_auto_smoke_multi_codec(capsys: pytest.CaptureFixture[str]) -> None:
    """CSV ``--allow-codecs`` parses into a list."""
    main(
        [
            "auto",
            "--source",
            "/dev/null",
            "--allow-codecs",
            "libx264,libx265,libsvtav1",
            "--smoke",
        ]
    )
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["allow_codecs"] == ["libx264", "libx265", "libsvtav1"]
    # F.1 default rungs returns one rung; one cell per codec → 3 candidates.
    assert len(payload["candidates"]) == 3


def test_auto_output_file_writes_plan(tmp_path: Path) -> None:
    """``--output PATH`` writes the JSON plan to disk."""
    out = tmp_path / "plan.json"
    rc = main(
        [
            "auto",
            "--source",
            "/dev/null",
            "--allow-codecs",
            "libx264",
            "--smoke",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["smoke"] is True


def test_auto_empty_allow_codecs_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty ``--allow-codecs`` after parsing returns exit code 2."""
    rc = main(
        [
            "auto",
            "--source",
            "/dev/null",
            "--allow-codecs",
            "  ,  ",
            "--smoke",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "--allow-codecs is empty" in err


def test_auto_missing_source_argparse_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Argparse rejects calls without ``--source``."""
    with pytest.raises(SystemExit) as excinfo:
        main(["auto", "--smoke"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--source" in err


def test_auto_unreachable_target_returns_one(capsys: pytest.CaptureFixture[str]) -> None:
    """No-winner runs return exit code 1 (mirrors ``recommend``).

    Smoke mode adds 0.25 to the target VMAF, so an unreachable target
    is one above the ceiling (100). We instead force a no-winner case
    by setting a budget below every smoke codec's bitrate.
    """
    rc = main(
        [
            "auto",
            "--source",
            "/dev/null",
            "--target-vmaf",
            "90",
            "--max-budget-kbps",
            "10",
            "--allow-codecs",
            "libx264",
            "--smoke",
        ]
    )
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["winner"] is None
    assert rc == 1


def test_auto_appears_in_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``vmaf-tune --help`` lists the ``auto`` subcommand."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr().out
    assert "auto" in captured
