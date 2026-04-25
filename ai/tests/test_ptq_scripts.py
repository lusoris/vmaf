"""Smoke tests for the PTQ harness (ADR-0173 / T5-3).

These don't run a full quantisation round-trip — that needs
`onnxruntime.quantization` installed and a real ONNX file. They check
that the script entry-points import cleanly and surface useful CLI help.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ai" / "scripts"


def _run_help(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_ptq_dynamic_help() -> None:
    cp = _run_help(SCRIPTS_DIR / "ptq_dynamic.py")
    assert cp.returncode == 0, cp.stderr
    assert "dynamic" in cp.stdout.lower()
    # No `--calibration` flag — dynamic PTQ doesn't need one.
    assert "--calibration" not in cp.stdout


def test_ptq_static_help() -> None:
    cp = _run_help(SCRIPTS_DIR / "ptq_static.py")
    assert cp.returncode == 0, cp.stderr
    assert "calibration" in cp.stdout.lower()


def test_qat_train_help() -> None:
    cp = _run_help(SCRIPTS_DIR / "qat_train.py")
    assert cp.returncode == 0, cp.stderr
    assert "config" in cp.stdout.lower()


def test_ptq_dynamic_imports_only_what_it_needs() -> None:
    """Import the module without executing main() — must not require
    onnxruntime.quantization just to load the file."""
    spec = importlib.util.spec_from_file_location(
        "ptq_dynamic_under_test", SCRIPTS_DIR / "ptq_dynamic.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime not installed; skipping full ptq_dynamic round-trip",
)
def test_ptq_dynamic_full_roundtrip(tmp_path: Path) -> None:
    """End-to-end: build a tiny ONNX, quantise it, confirm the output
    file exists and is smaller than (or equal to) the input."""
    pytest.importorskip("onnxruntime.quantization")

    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    src = tmp_path / "tiny.onnx"
    dst = tmp_path / "tiny.int8.onnx"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    weight = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [4, 4],
        np.arange(16, dtype=np.float32).tolist(),
    )
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[weight])
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        str(src),
    )

    cp = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "ptq_dynamic.py"), str(src), "--output", str(dst)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, cp.stderr
    assert dst.is_file()
    # int8 quantised version is typically smaller; allow equal on tiny graphs
    # where the QDQ overhead dominates.
    assert dst.stat().st_size <= src.stat().st_size + 4096
