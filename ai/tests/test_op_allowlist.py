"""Op-allowlist parser keeps Python and libvmaf's C source in lock-step."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper
from vmaf_train.op_allowlist import check_graph, check_model, load_allowlist


def test_allowlist_loads_nonempty() -> None:
    allowed = load_allowlist()
    # A few ops any realistic tiny model needs.
    for must_have in ("Conv", "Gemm", "Relu", "Add", "Reshape"):
        assert must_have in allowed, f"{must_have} missing — parser likely broken"


def test_control_flow_ops_are_not_allowed() -> None:
    allowed = load_allowlist()
    # Security policy forbids these explicitly (docs/tiny-ai/security.md).
    for banned in ("If", "Loop", "Scan"):
        assert banned not in allowed, f"{banned} slipped onto allowlist"


def _build_model(op_type: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], f"g_{op_type}", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_allowed_op_passes() -> None:
    report = check_graph(_build_model("Relu"))
    assert report.ok
    assert "Relu" in report.used
    assert not report.forbidden


def test_forbidden_op_rejected() -> None:
    report = check_graph(_build_model("Loop"))
    assert not report.ok
    assert "Loop" in report.forbidden
    assert "forbidden" in report.pretty()


def test_check_model_from_file(tmp_path: Path) -> None:
    model = _build_model("Sigmoid")
    p = tmp_path / "m.onnx"
    onnx.save(model, str(p))
    assert check_model(p).ok


def test_missing_source_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c"
    with pytest.raises(FileNotFoundError):
        load_allowlist(missing)
