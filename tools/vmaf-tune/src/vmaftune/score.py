# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""vmaf binary driver — Phase A.

Spawns the libvmaf CLI (`vmaf`) against a (reference YUV, distorted
encode) pair and parses the pooled VMAF score from the JSON output.

Subprocess boundary is the integration seam — tests mock subprocess.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class ScoreRequest:
    """Pair to score: reference YUV vs distorted encode.

    ``frame_skip_ref`` / ``frame_cnt`` mirror the libvmaf CLI flags
    (``--frame_skip_ref`` / ``--frame_cnt``). Sample-clip mode (ADR-0297)
    sets these so VMAF compares the same time window of the reference
    that was fed to the encoder, instead of slicing the reference YUV
    on disk. Both ``0`` (default) keeps the legacy full-source scoring.
    """

    reference: Path
    distorted: Path
    width: int
    height: int
    pix_fmt: str
    model: str = "vmaf_v0.6.1"
    frame_skip_ref: int = 0
    frame_cnt: int = 0


@dataclasses.dataclass(frozen=True)
class ScoreResult:
    """Outcome of one scoring call."""

    request: ScoreRequest
    vmaf_score: float
    score_time_ms: float
    vmaf_binary_version: str
    exit_status: int
    stderr_tail: str


_VMAF_VERSION_RE = re.compile(r"VMAF version[: ]+(\S+)")


def build_vmaf_command(
    req: ScoreRequest,
    json_output: Path,
    *,
    vmaf_bin: str = "vmaf",
) -> list[str]:
    """Compose the libvmaf CLI argv. Pure function for test pinning."""
    cmd = [
        vmaf_bin,
        "--reference",
        str(req.reference),
        "--distorted",
        str(req.distorted),
        "--width",
        str(req.width),
        "--height",
        str(req.height),
        "--pixel_format",
        _pixfmt_to_vmaf(req.pix_fmt),
        "--bitdepth",
        str(_bitdepth_for(req.pix_fmt)),
        "--model",
        f"version={req.model}",
        "--json",
        "--output",
        str(json_output),
    ]
    # Sample-clip mode (ADR-0297): align reference window with the
    # encoded slice so VMAF compares matching frames. The distorted is
    # already a clip-length encode, so no --frame_skip_dist is needed.
    if req.frame_skip_ref > 0:
        cmd.extend(["--frame_skip_ref", str(req.frame_skip_ref)])
    if req.frame_cnt > 0:
        cmd.extend(["--frame_cnt", str(req.frame_cnt)])
    return cmd


def _pixfmt_to_vmaf(pix_fmt: str) -> str:
    """Map ffmpeg pix_fmt to libvmaf's --pixel_format vocabulary.

    Only the subset Phase A actually drives. Falls back to ``420``.
    """
    if pix_fmt.startswith("yuv422"):
        return "422"
    if pix_fmt.startswith("yuv444"):
        return "444"
    return "420"


def _bitdepth_for(pix_fmt: str) -> int:
    if "10le" in pix_fmt or "p10" in pix_fmt:
        return 10
    if "12le" in pix_fmt or "p12" in pix_fmt:
        return 12
    return 8


def parse_vmaf_json(payload: dict) -> float:
    """Pull the pooled VMAF score from libvmaf's JSON output.

    Tries the modern ``pooled_metrics.vmaf.mean`` shape first, falls
    back to the older top-level ``VMAF score``. Raises ``ValueError``
    if neither is present.
    """
    pooled = payload.get("pooled_metrics") or {}
    vmaf = pooled.get("vmaf") or {}
    if "mean" in vmaf:
        return float(vmaf["mean"])
    if "VMAF score" in payload:
        return float(payload["VMAF score"])
    raise ValueError("vmaf JSON missing pooled_metrics.vmaf.mean")


def run_score(
    req: ScoreRequest,
    *,
    vmaf_bin: str = "vmaf",
    runner: object | None = None,
    workdir: Path | None = None,
) -> ScoreResult:
    """Drive the vmaf CLI for a single (ref, dist) pair."""
    runner_fn = runner or subprocess.run

    if workdir is None:
        workdir_ctx = tempfile.TemporaryDirectory()
        workdir_path = Path(workdir_ctx.name)
    else:
        workdir_ctx = None
        workdir_path = workdir
        workdir_path.mkdir(parents=True, exist_ok=True)

    json_path = workdir_path / "vmaf.json"
    cmd = build_vmaf_command(req, json_path, vmaf_bin=vmaf_bin)

    try:
        started = time.monotonic()
        completed = runner_fn(  # type: ignore[operator]
            cmd, capture_output=True, text=True, check=False
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0

        stderr = getattr(completed, "stderr", "") or ""
        rc = int(getattr(completed, "returncode", 1))

        score = float("nan")
        if rc == 0 and json_path.exists():
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            try:
                score = parse_vmaf_json(payload)
            except ValueError:
                rc = rc or 65

        match = _VMAF_VERSION_RE.search(stderr)
        version = match.group(1) if match else "unknown"

        return ScoreResult(
            request=req,
            vmaf_score=score,
            score_time_ms=elapsed_ms,
            vmaf_binary_version=version,
            exit_status=rc,
            stderr_tail=stderr[-2048:],
        )
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()
