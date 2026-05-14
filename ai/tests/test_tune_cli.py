# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Tests for the ``vmaf-train tune`` CLI wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from ai.src.vmaf_train import cli


def test_parse_tune_param_specs_accepts_supported_kinds() -> None:
    specs = cli._parse_tune_param_specs(
        ["hidden=float:8:64:log", "layers=int:1:4", "activation=choice:relu,tanh"]
    )
    assert specs == [
        ("hidden", "float", ("8", "64", "log")),
        ("layers", "int", ("1", "4")),
        ("activation", "choice", ("relu,tanh",)),
    ]


def test_make_tune_suggest_dispatches_to_trial_methods() -> None:
    specs = cli._parse_tune_param_specs(
        ["lr=float:0.001:0.1:log", "layers=int:1:3", "bias=choice:true,false"]
    )
    calls: list[tuple[str, str, object]] = []

    class Trial:
        def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
            calls.append(("float", name, (low, high, log)))
            return 0.01

        def suggest_int(self, name: str, low: int, high: int) -> int:
            calls.append(("int", name, (low, high)))
            return 2

        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            calls.append(("choice", name, choices))
            return choices[0]

    assert cli._make_tune_suggest(specs)(Trial()) == {"lr": 0.01, "layers": 2, "bias": True}
    assert calls == [
        ("float", "lr", (0.001, 0.1, True)),
        ("int", "layers", (1, 3)),
        ("choice", "bias", [True, False]),
    ]


def test_tune_cli_invokes_sweep(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "model: fr_regressor",
                "model_args:",
                "  in_features: 6",
                "cache: features.parquet",
                "output: runs/base",
                "epochs: 1",
            ]
        )
    )
    calls: list[dict[str, object]] = []

    def fake_sweep(base_cfg, suggest, *, n_trials: int, study_name: str, storage):
        class Trial:
            def suggest_int(self, name: str, low: int, high: int) -> int:
                return high

        calls.append(
            {
                "model": base_cfg.model,
                "output": str(base_cfg.output),
                "n_trials": n_trials,
                "study_name": study_name,
                "storage": storage,
                "suggested": suggest(Trial()),
            }
        )
        return SimpleNamespace(best_value=0.25, best_params={"hidden": 32})

    monkeypatch.setitem(
        sys.modules,
        "ai.src.vmaf_train.tune",
        SimpleNamespace(sweep=fake_sweep),
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "tune",
            "--config",
            str(config),
            "--param",
            "hidden=int:8:32",
            "--trials",
            "3",
            "--study-name",
            "smoke",
            "--output",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "model": "fr_regressor",
            "output": str(tmp_path / "runs"),
            "n_trials": 3,
            "study_name": "smoke",
            "storage": None,
            "suggested": {"hidden": 32},
        }
    ]
    assert "Sweep done" in result.output
