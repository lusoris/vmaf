"""MCP server for the Lusoris VMAF fork.

Exposes six tools over the Model Context Protocol (stdio transport):

- ``vmaf_score``          — score a (reference, distorted) pair.
- ``list_models``         — enumerate the VMAF models registered with the build.
- ``list_backends``       — report which backends (cpu/cuda/sycl) are available.
- ``run_benchmark``       — run the Netflix benchmark harness on a pair.
- ``eval_model_on_split`` — run an ONNX tiny-AI model against a parquet feature
  cache on a deterministic split and report PLCC/SROCC/RMSE.
- ``compare_models``      — rank several ONNX models on the same split.

The server assumes ``build/tools/vmaf`` exists (build first with
``meson compile -C build``). Paths are validated to live under either the
repository's ``testdata/`` / ``python/test/resource/`` / ``model/`` trees
or an explicitly-allowlisted prefix passed via ``VMAF_MCP_ALLOW``. This
prevents callers from coercing the server into reading arbitrary host
paths.
"""

from __future__ import annotations

# NOTE (risk-accept): the `subprocess` import below exec's our own signed
# vmaf binary with an argv list (no shell=True, no user-controlled
# strings in argv[0]); broad exception handlers on the call paths
# convert failures into JSON-RPC errors for the client. If ruff `select`
# is ever widened to include the bandit (`S`) or blind-except (`BLE`)
# rules, re-evaluate these sites deliberately rather than silencing
# with line-level suppression markers.
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Configuration & path validation
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _vmaf_binary() -> Path:
    env = os.environ.get("VMAF_BIN")
    if env:
        return Path(env)
    return _repo_root() / "build" / "tools" / "vmaf"


def _allowed_roots() -> list[Path]:
    roots = [
        _repo_root() / "testdata",
        _repo_root() / "python" / "test" / "resource",
        _repo_root() / "model",
    ]
    extra = os.environ.get("VMAF_MCP_ALLOW")
    if extra:
        roots.extend(Path(p).resolve() for p in extra.split(":") if p)
    return [r.resolve() for r in roots]


def _validate_path(p: str) -> Path:
    path = Path(p).resolve()
    allowed = _allowed_roots()
    if not any(path.is_relative_to(root) for root in allowed):
        raise ValueError(
            f"path {path} not under an allowlisted root; set VMAF_MCP_ALLOW to extend."
        )
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreRequest:
    ref: Path
    dis: Path
    width: int
    height: int
    pixfmt: str  # "420" | "422" | "444"
    bitdepth: int
    model: str = "version=vmaf_v0.6.1"
    backend: str = "auto"  # "cpu" | "cuda" | "sycl" | "auto"
    precision: str = "17"


async def _run_vmaf_score(req: ScoreRequest) -> dict[str, Any]:
    vmaf = _vmaf_binary()
    if not vmaf.exists():
        raise RuntimeError(
            f"vmaf binary not found at {vmaf}. "
            "Build first: meson compile -C build."
        )

    output = Path("/tmp") / f"vmaf-mcp-{os.getpid()}-{asyncio.current_task().get_name()}.json"
    try:
        argv = [
            str(vmaf),
            "-r", str(req.ref),
            "-d", str(req.dis),
            "--width", str(req.width),
            "--height", str(req.height),
            "-p", req.pixfmt,
            "-b", str(req.bitdepth),
            "-m", req.model,
            "--precision", req.precision,
            "-q",
            "-o", str(output),
            "--json",
        ]
        if req.backend == "cpu":
            argv.extend(["--no_cuda", "--no_sycl"])
        elif req.backend == "cuda":
            argv.extend(["--no_sycl"])
        elif req.backend == "sycl":
            argv.extend(["--no_cuda"])

        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"vmaf exited {proc.returncode}: {stderr.decode(errors='replace')}"
            )
        return json.loads(output.read_text())
    finally:
        output.unlink(missing_ok=True)


def _list_models() -> list[dict[str, Any]]:
    models_dir = _repo_root() / "model"
    out: list[dict[str, Any]] = []
    for p in sorted(models_dir.rglob("*")):
        if p.suffix in {".json", ".pkl", ".onnx"} and p.is_file():
            out.append({
                "name": p.stem,
                "path": str(p.relative_to(_repo_root())),
                "format": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size,
            })
    return out


def _list_backends() -> dict[str, bool]:
    vmaf = _vmaf_binary()
    if not vmaf.exists():
        return {"cpu": False, "cuda": False, "sycl": False, "hip": False}
    try:
        result = subprocess.run(
            [str(vmaf), "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        blob = (result.stdout + result.stderr).lower()
    except (subprocess.TimeoutExpired, OSError):
        blob = ""
    return {
        "cpu": True,
        "cuda": "cuda" in blob,
        "sycl": "sycl" in blob or "oneapi" in blob,
        "hip": "hip" in blob,
    }


_FEATURE_COLUMNS = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)
_VALID_SPLITS = ("train", "val", "test", "all")


def _eval_model_on_split(
    model: Path, features: Path, split: str, input_name: str
) -> dict[str, Any]:
    """Run @p model on @p split of @p features and return PLCC/SROCC/RMSE.

    Imports are lazy so the base mcp-server install (no pandas / onnxruntime
    / scipy) isn't forced to pull in ML deps just to score video.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}; got {split!r}")
    try:
        import numpy as np
        import onnxruntime as ort
        import pandas as pd
        from scipy.stats import pearsonr, spearmanr
    except ImportError as exc:  # pragma: no cover — exercised only without extras
        raise RuntimeError(
            "eval_model_on_split requires the 'eval' extra: "
            "pip install 'vmaf-mcp[eval]'"
        ) from exc

    df = pd.read_parquet(features)
    if "mos" not in df.columns:
        raise ValueError(f"{features} has no 'mos' column — can't score correlations")
    if split != "all" and "key" in df.columns:
        # Inline the split_keys hashing so we don't depend on vmaf_train.
        import hashlib

        def bucket(key: str) -> float:
            h = hashlib.sha256(f"vmaf-train-splits-v1:{key}".encode()).digest()
            return int.from_bytes(h[:8], "big") / (1 << 64)

        val_frac, test_frac = 0.1, 0.1

        def which(key: str) -> str:
            b = bucket(str(key))
            if b < test_frac:
                return "test"
            if b < test_frac + val_frac:
                return "val"
            return "train"

        keep = df["key"].astype(str).map(which) == split
        df = df[keep]

    cols = [c for c in _FEATURE_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(
            f"{features} has none of the expected feature columns "
            f"{_FEATURE_COLUMNS}; got {list(df.columns)}"
        )
    x = df[cols].to_numpy(dtype=np.float32)
    y = df["mos"].to_numpy(dtype=np.float32)
    if len(x) < 2:
        raise ValueError(
            f"split {split!r} has {len(x)} samples — need ≥2 to compute correlations"
        )

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    pred = np.asarray(sess.run(None, {input_name: x})[0]).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(
            f"model output shape {pred.shape} does not match target shape {y.shape}"
        )
    plcc = float(pearsonr(pred, y).statistic)
    srocc = float(spearmanr(pred, y).statistic)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    return {
        "model": str(model),
        "features": str(features),
        "split": split,
        "n": len(x),
        "plcc": plcc,
        "srocc": srocc,
        "rmse": rmse,
        "columns": cols,
    }


def _compare_models(
    models: list[Path], features: Path, split: str, input_name: str
) -> dict[str, Any]:
    """Rank @p models on the same feature split by descending PLCC."""
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for m in models:
        try:
            reports.append(_eval_model_on_split(m, features, split, input_name))
        except Exception as exc:
            errors.append({"model": str(m), "error": str(exc)})
    reports.sort(key=lambda r: r["plcc"], reverse=True)
    return {"ranked": reports, "errors": errors}


async def _run_benchmark(ref: Path, dis: Path, width: int, height: int) -> dict[str, Any]:
    script = _repo_root() / "testdata" / "bench_all.sh"
    if not script.exists():
        raise FileNotFoundError(f"benchmark harness not found: {script}")
    proc = await asyncio.create_subprocess_exec(
        str(script), "-r", str(ref), "-d", str(dis),
        "--width", str(width), "--height", str(height),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "exit_code": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------


server: Server = Server("vmaf-mcp")


@server.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(
            name="vmaf_score",
            description="Compute a VMAF score for a (reference, distorted) YUV pair.",
            inputSchema={
                "type": "object",
                "required": ["ref", "dis", "width", "height", "pixfmt", "bitdepth"],
                "properties": {
                    "ref":       {"type": "string", "description": "Reference YUV path."},
                    "dis":       {"type": "string", "description": "Distorted YUV path."},
                    "width":     {"type": "integer", "minimum": 1},
                    "height":    {"type": "integer", "minimum": 1},
                    "pixfmt":    {"type": "string", "enum": ["420", "422", "444"]},
                    "bitdepth":  {"type": "integer", "enum": [8, 10, 12, 16]},
                    "model":     {"type": "string", "default": "version=vmaf_v0.6.1"},
                    "backend":   {"type": "string", "enum": ["auto", "cpu", "cuda", "sycl"], "default": "auto"},
                    "precision": {"type": "string", "default": "17"},
                },
            },
        ),
        Tool(
            name="list_models",
            description="Enumerate VMAF models (JSON / pickle / ONNX) shipped with the repo.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_backends",
            description="Report which runtime backends (cpu / cuda / sycl / hip) the local vmaf binary was built with.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="run_benchmark",
            description="Run the Netflix benchmark harness on a pair and return stdout/stderr.",
            inputSchema={
                "type": "object",
                "required": ["ref", "dis", "width", "height"],
                "properties": {
                    "ref":    {"type": "string"},
                    "dis":    {"type": "string"},
                    "width":  {"type": "integer"},
                    "height": {"type": "integer"},
                },
            },
        ),
        Tool(
            name="eval_model_on_split",
            description=(
                "Run an ONNX tiny-AI regressor on a parquet feature cache, "
                "filter to a deterministic train/val/test split (keyed by the "
                "'key' column), and report PLCC / SROCC / RMSE."
            ),
            inputSchema={
                "type": "object",
                "required": ["model", "features"],
                "properties": {
                    "model":      {"type": "string", "description": "ONNX model path."},
                    "features":   {"type": "string", "description": "Parquet feature cache path."},
                    "split":      {"type": "string", "enum": list(_VALID_SPLITS), "default": "test"},
                    "input_name": {"type": "string", "default": "features"},
                },
            },
        ),
        Tool(
            name="compare_models",
            description=(
                "Rank several ONNX models on the same parquet feature split by "
                "descending PLCC. Models that fail to load or score are listed "
                "under 'errors' instead of aborting the whole call."
            ),
            inputSchema={
                "type": "object",
                "required": ["models", "features"],
                "properties": {
                    "models":     {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "features":   {"type": "string"},
                    "split":      {"type": "string", "enum": list(_VALID_SPLITS), "default": "test"},
                    "input_name": {"type": "string", "default": "features"},
                },
            },
        ),
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "vmaf_score":
            req = ScoreRequest(
                ref=_validate_path(arguments["ref"]),
                dis=_validate_path(arguments["dis"]),
                width=int(arguments["width"]),
                height=int(arguments["height"]),
                pixfmt=str(arguments["pixfmt"]),
                bitdepth=int(arguments["bitdepth"]),
                model=str(arguments.get("model", "version=vmaf_v0.6.1")),
                backend=str(arguments.get("backend", "auto")),
                precision=str(arguments.get("precision", "17")),
            )
            result = await _run_vmaf_score(req)
        elif name == "list_models":
            result = {"models": _list_models()}
        elif name == "list_backends":
            result = _list_backends()
        elif name == "run_benchmark":
            result = await _run_benchmark(
                ref=_validate_path(arguments["ref"]),
                dis=_validate_path(arguments["dis"]),
                width=int(arguments["width"]),
                height=int(arguments["height"]),
            )
        elif name == "eval_model_on_split":
            result = _eval_model_on_split(
                model=_validate_path(arguments["model"]),
                features=_validate_path(arguments["features"]),
                split=str(arguments.get("split", "test")),
                input_name=str(arguments.get("input_name", "features")),
            )
        elif name == "compare_models":
            models_in = arguments["models"]
            if not isinstance(models_in, list) or not models_in:
                raise ValueError("'models' must be a non-empty list of paths")
            result = _compare_models(
                models=[_validate_path(m) for m in models_in],
                features=_validate_path(arguments["features"]),
                split=str(arguments.get("split", "test")),
                input_name=str(arguments.get("input_name", "features")),
            )
        else:
            raise ValueError(f"unknown tool: {name}")
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _run() -> None:
    if not shutil.which("meson"):
        print("warning: meson not on PATH — benchmark tool may fail.", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio_impl = os.environ.get("VMAF_MCP_ASYNC", "asyncio")
    if anyio_impl == "asyncio":
        asyncio.run(_run())
    else:
        import anyio
        anyio.run(_run, backend=anyio_impl)


if __name__ == "__main__":
    main()
