"""Smoke tests for the vmaf-mcp server — no network, no GPU required."""
from __future__ import annotations

from pathlib import Path

import pytest

from vmaf_mcp import server as srv

REPO = Path(__file__).resolve().parents[3]


def test_repo_root_detects_testdata():
    root = srv._repo_root()
    assert (root / "testdata").is_dir()


def test_validate_path_accepts_golden_yuv():
    yuv = REPO / "python/test/resource/yuv/src01_hrc00_576x324.yuv"
    if not yuv.exists():
        pytest.skip("Netflix golden YUV not present")
    assert srv._validate_path(str(yuv)) == yuv.resolve()


def test_validate_path_rejects_outside_roots(tmp_path):
    bad = tmp_path / "evil.yuv"
    bad.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="not under an allowlisted root"):
        srv._validate_path(str(bad))


def test_validate_path_accepts_custom_allow(tmp_path, monkeypatch):
    f = tmp_path / "ok.yuv"
    f.write_bytes(b"\x00" * 16)
    monkeypatch.setenv("VMAF_MCP_ALLOW", str(tmp_path))
    assert srv._validate_path(str(f)) == f.resolve()


def test_list_models_returns_list():
    models = srv._list_models()
    assert isinstance(models, list)
    for m in models:
        assert "name" in m and "path" in m and "format" in m


def test_list_backends_always_includes_cpu():
    backends = srv._list_backends()
    assert backends["cpu"] is True


# ---------------------------------------------------------------------------
# eval_model_on_split / compare_models — require the 'eval' extra
# ---------------------------------------------------------------------------


def _has_eval_deps() -> bool:
    try:
        import numpy  # noqa: F401
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark_eval = pytest.mark.skipif(
    not _has_eval_deps(), reason="vmaf-mcp[eval] extras not installed"
)


def _make_tiny_mlp(path: Path, in_features: int = 6) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", in_features])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor(
        "W", TensorProto.FLOAT, [in_features, 1], [0.5] * in_features
    )
    b = helper.make_tensor("B", TensorProto.FLOAT, [1], [10.0])
    node = helper.make_node("Gemm", ["features", "W", "B"], ["score"])
    graph = helper.make_graph([node], "mlp", [x], [y], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def _make_feature_parquet(path: Path, n: int = 64) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data = {c: rng.standard_normal(n).astype(np.float32) for c in srv._FEATURE_COLUMNS}
    # MOS is a linear transform of the features so correlations are non-trivial.
    x = np.stack(list(data.values()), axis=1)
    data["mos"] = (x.sum(axis=1) * 0.5 + 10.0 + rng.normal(0, 0.01, n)).astype(np.float32)
    data["key"] = [f"sample_{i:04d}" for i in range(n)]
    pd.DataFrame(data).to_parquet(path)


@pytestmark_eval
def test_eval_model_on_split_reports_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("VMAF_MCP_ALLOW", str(tmp_path))
    model = tmp_path / "m.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(model)
    _make_feature_parquet(feats)
    result = srv._eval_model_on_split(model, feats, split="all", input_name="features")
    assert result["n"] == 64
    # The MOS column is a linear function of the features with tiny noise, so
    # the linear model's correlation should be very close to 1.
    assert result["plcc"] > 0.99
    assert result["rmse"] >= 0.0
    assert result["split"] == "all"


@pytestmark_eval
def test_eval_model_on_split_rejects_unknown_split(tmp_path):
    model = tmp_path / "m.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(model)
    _make_feature_parquet(feats)
    with pytest.raises(ValueError, match="split must be one of"):
        srv._eval_model_on_split(model, feats, split="nope", input_name="features")


@pytestmark_eval
def test_eval_split_filters_by_key(tmp_path):
    """train+val+test should partition the parquet exactly."""
    model = tmp_path / "m.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(model)
    _make_feature_parquet(feats, n=200)
    ns = {
        s: srv._eval_model_on_split(model, feats, split=s, input_name="features")["n"]
        for s in ("train", "val", "test")
    }
    assert sum(ns.values()) == 200
    # Small-sample fractions drift from 10% / 10% / 80% but should stay in-band.
    assert ns["train"] > ns["val"] and ns["train"] > ns["test"]


@pytestmark_eval
def test_compare_models_ranks_by_plcc(tmp_path):
    good = tmp_path / "good.onnx"
    bad = tmp_path / "bad.onnx"
    feats = tmp_path / "f.parquet"
    _make_tiny_mlp(good)
    # A model with opposite-sign weights will have strongly negative PLCC.
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", 6])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["N", 1])
    w = helper.make_tensor("W", TensorProto.FLOAT, [6, 1], [-0.5] * 6)
    b = helper.make_tensor("B", TensorProto.FLOAT, [1], [10.0])
    node = helper.make_node("Gemm", ["features", "W", "B"], ["score"])
    graph = helper.make_graph([node], "mlp", [x], [y], [w, b])
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(bad)
    )
    _make_feature_parquet(feats)
    report = srv._compare_models(
        [bad, good], feats, split="all", input_name="features"
    )
    assert report["errors"] == []
    assert report["ranked"][0]["model"] == str(good)
    assert report["ranked"][0]["plcc"] > report["ranked"][1]["plcc"]


@pytestmark_eval
def test_compare_models_captures_errors_without_aborting(tmp_path):
    ok = tmp_path / "ok.onnx"
    _make_tiny_mlp(ok)
    missing = tmp_path / "missing.onnx"
    feats = tmp_path / "f.parquet"
    _make_feature_parquet(feats)
    report = srv._compare_models(
        [ok, missing], feats, split="all", input_name="features"
    )
    # One report, one error — the missing model doesn't take the good one down.
    assert len(report["ranked"]) == 1
    assert len(report["errors"]) == 1
    assert report["errors"][0]["model"] == str(missing)


def test_new_tools_registered_in_list_tools():
    """Schema-level check that doesn't need eval extras installed."""
    import anyio

    async def get_tools():
        return await srv._list_tools()

    tools = anyio.run(get_tools)
    names = {t.name for t in tools}
    assert "eval_model_on_split" in names
    assert "compare_models" in names
