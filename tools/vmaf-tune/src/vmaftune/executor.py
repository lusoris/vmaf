# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase F execute mode — drive real encodes + scores for a ``vmaf-tune auto`` plan.

``run_plan`` iterates the ``selected`` cell(s) from an :class:`~vmaftune.auto.AutoPlan`,
runs FFmpeg (via :func:`~vmaftune.encode.run_encode`) per rung, scores each output with
the libvmaf CLI (via :func:`~vmaftune.score.run_score`), and appends rows to a JSONL
results file under ``out_dir/tune_results.jsonl``.

Design notes (ADR-0454):

* Zero new mandatory dependencies — results are written as JSONL, matching the corpus
  path (``corpus.py``). A future polars/pyarrow layer can convert on demand.
* The subprocess boundary is the seam: ``encode_runner`` and ``score_runner`` kwargs
  accept the same mock-runner pattern used throughout the harness so the executor is
  fully testable without FFmpeg or ``vmaf`` binaries.
* Only the ``selected`` cell is executed by default; pass ``execute_all=True`` to run
  every cell in the plan (useful for a post-hoc A/B comparison).
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .encode import EncodeRequest, EncodeResult, run_encode
from .score import ScoreRequest, ScoreResult, run_score


@dataclasses.dataclass(frozen=True)
class ExecuteResult:
    """Outcome of one (encode + score) pair for a single plan cell.

    ``cell`` is the original plan dict (a reference to the ``cells`` entry).
    ``encode`` and ``score`` are ``None`` when the respective step was skipped
    (e.g. encode failed and ``score`` was not attempted).
    ``row`` is the flat dict written to the JSONL results file — it is a
    merged view of the cell metadata and the encode/score outcomes.
    """

    cell: dict[str, Any]
    encode: EncodeResult | None
    score: ScoreResult | None
    row: dict[str, Any]


def _cell_to_encode_request(
    cell: dict[str, Any],
    src: Path,
    out_dir: Path,
    *,
    pix_fmt: str,
    width: int,
    height: int,
    framerate: float,
    source_is_container: bool,
) -> EncodeRequest:
    """Build an :class:`EncodeRequest` from an ``AutoPlan`` cell dict."""
    codec = str(cell.get("codec", "libx264"))
    preset = str(cell.get("preset", "medium"))
    crf = int(cell.get("crf", 23))
    cell_index = int(cell.get("cell_index", 0))
    output = out_dir / f"encode_{cell_index:03d}_{codec}_{preset}_crf{crf}.mkv"
    return EncodeRequest(
        source=src,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        framerate=framerate,
        encoder=codec,
        preset=preset,
        crf=crf,
        output=output,
        source_is_container=source_is_container,
    )


def _make_row(
    cell: dict[str, Any],
    enc: EncodeResult | None,
    sc: ScoreResult | None,
) -> dict[str, Any]:
    """Flatten cell + encode + score outcome into a single results dict."""
    row: dict[str, Any] = {
        "cell_index": cell.get("cell_index"),
        "codec": cell.get("codec"),
        "preset": cell.get("preset"),
        "crf": cell.get("crf"),
        "selected": bool(cell.get("selected", False)),
        "estimated_vmaf": cell.get("estimated_vmaf"),
        "estimated_bitrate_kbps": cell.get("estimated_bitrate_kbps"),
        "prediction_source": cell.get("prediction_source"),
        # Encode outcomes
        "encode_size_bytes": enc.encode_size_bytes if enc else None,
        "encode_time_ms": enc.encode_time_ms if enc else None,
        "encode_exit_status": enc.exit_status if enc else None,
        "ffmpeg_version": enc.ffmpeg_version if enc else None,
        "encoder_version": enc.encoder_version if enc else None,
        "encode_path": str(enc.request.output) if enc else None,
        # Score outcomes
        "vmaf_score": sc.vmaf_score if sc else None,
        "score_time_ms": sc.score_time_ms if sc else None,
        "score_exit_status": sc.exit_status if sc else None,
        "vmaf_binary_version": sc.vmaf_binary_version if sc else None,
    }
    if sc is not None:
        for feat, val in sc.feature_means.items():
            row[f"feature_{feat}_mean"] = val
        for feat, val in sc.feature_stds.items():
            row[f"feature_{feat}_std"] = val
    return row


def run_plan(
    plan: "AutoPlan",  # type: ignore[name-defined]  # noqa: F821
    src: Path,
    out_dir: Path,
    *,
    pix_fmt: str = "yuv420p",
    width: int = 1920,
    height: int = 1080,
    framerate: float = 25.0,
    source_is_container: bool = True,
    execute_all: bool = False,
    vmaf_model: str = "vmaf_v0.6.1",
    vmaf_bin: str = "vmaf",
    ffmpeg_bin: str = "ffmpeg",
    encode_runner: Callable[..., Any] | None = None,
    score_runner: Callable[..., Any] | None = None,
) -> list[ExecuteResult]:
    """Realise an ``AutoPlan`` by running real encodes and scores.

    Parameters
    ----------
    plan:
        The :class:`~vmaftune.auto.AutoPlan` returned by :func:`~vmaftune.auto.run_auto`.
    src:
        Reference source path (forwarded to the encoder as input).
    out_dir:
        Directory for encoded files and the ``tune_results.jsonl`` log.
        Created if absent.
    pix_fmt:
        FFmpeg pixel format string (default ``yuv420p``).  When
        ``source_is_container=True`` the encoder driver reads format from the
        container; ``pix_fmt`` is still stored in :class:`EncodeRequest` for
        the score driver.
    width, height:
        Frame geometry; taken from plan ``metadata.source_meta`` when not
        overridden (the CLI wrapper does this automatically).
    framerate:
        Frame rate; same override semantics as ``width``/``height``.
    source_is_container:
        When ``True`` (default) the encoder driver omits raw-YUV input flags
        and lets FFmpeg detect the format from the container.
    execute_all:
        When ``True`` run every cell; otherwise only cells with
        ``selected=True`` are executed (default).
    vmaf_model:
        libvmaf model identifier forwarded to :class:`~vmaftune.score.ScoreRequest`.
    vmaf_bin, ffmpeg_bin:
        Binary names / paths for the ``vmaf`` and ``ffmpeg`` executables.
    encode_runner, score_runner:
        Optional ``subprocess.run``-compatible callables used as test seams.
        Pass ``None`` in production (the drivers call ``subprocess.run``
        directly).

    Returns
    -------
    list[ExecuteResult]
        One entry per executed cell, in plan order.  Always written to
        ``out_dir/tune_results.jsonl`` even on partial failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "tune_results.jsonl"

    # Pull geometry from plan metadata when caller did not override.
    source_meta = plan.metadata.get("source_meta", {})
    eff_width = int(source_meta.get("width", width)) if width == 1920 else width
    eff_height = int(source_meta.get("height", height)) if height == 1080 else height

    cells_to_run = [cell for cell in plan.cells if execute_all or bool(cell.get("selected", False))]

    results: list[ExecuteResult] = []

    with results_path.open("a", encoding="utf-8") as fh:
        for cell in cells_to_run:
            enc_req = _cell_to_encode_request(
                cell,
                src,
                out_dir,
                pix_fmt=pix_fmt,
                width=eff_width,
                height=eff_height,
                framerate=framerate,
                source_is_container=source_is_container,
            )

            enc: EncodeResult | None = None
            sc: ScoreResult | None = None

            try:
                enc = run_encode(enc_req, ffmpeg_bin=ffmpeg_bin, runner=encode_runner)
            except Exception as exc:  # noqa: BLE001
                # Encode failure is recorded in the row; scoring is skipped.
                _log(f"executor: encode failed for cell {cell.get('cell_index')}: {exc}")

            if enc is not None and enc.exit_status == 0:
                with tempfile.TemporaryDirectory() as td:
                    score_req = ScoreRequest(
                        reference=src,
                        distorted=enc_req.output,
                        width=eff_width,
                        height=eff_height,
                        pix_fmt=pix_fmt,
                        model=vmaf_model,
                    )
                    try:
                        sc = run_score(
                            score_req,
                            vmaf_bin=vmaf_bin,
                            runner=score_runner,
                            workdir=Path(td),
                        )
                    except Exception as exc:  # noqa: BLE001
                        _log(f"executor: score failed for cell " f"{cell.get('cell_index')}: {exc}")

            row = _make_row(cell, enc, sc)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            results.append(ExecuteResult(cell=cell, encode=enc, score=sc, row=row))

    return results


def _log(msg: str) -> None:
    """Write a timestamped line to stderr (no logging dep)."""
    import sys

    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[{ts}] {msg}\n")


__all__ = [
    "ExecuteResult",
    "run_plan",
]
