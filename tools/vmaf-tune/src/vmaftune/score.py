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
    """Pair to score: reference YUV vs distorted encode."""

    reference: Path
    distorted: Path
    width: int
    height: int
    pix_fmt: str
    model: str = "vmaf_v0.6.1"


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
    backend: str | None = None,
) -> list[str]:
    """Compose the libvmaf CLI argv. Pure function for test pinning.

    When ``backend`` is set (one of ``cpu|cuda|sycl|vulkan``), append
    ``--backend NAME`` so the libvmaf CLI engages the GPU dispatch.
    ``None`` (the default) leaves the binary in its built-in ``auto``
    mode for full backwards compatibility.
    """
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
        _model_arg(req.model),
        "--json",
        "--output",
        str(json_output),
    ]
    if backend:
        cmd.extend(["--backend", backend])
    return cmd


def _model_arg(model: str) -> str:
    """Format the ``--model`` argument for the libvmaf CLI.

    Accepts either a bare version identifier (``"vmaf_v0.6.1"``) or a
    pre-formatted ``key=value`` string (``"path=/abs/model.json"``,
    ``"version=vmaf_v0.6.1"``). Bare identifiers are wrapped as
    ``version=...``; pre-formatted strings pass through. Used by
    ``corpus.py`` to inject HDR-model paths (see ADR-0295).
    """
    if "=" in model:
        return model
    return f"version={model}"


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


_RAW_DIST_EXTS = {".yuv", ".y4m"}


def _decode_to_raw_yuv(
    src: Path,
    dst: Path,
    pix_fmt: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner_fn=subprocess.run,
) -> tuple[int, str]:
    """Decode an encoded container (mp4/webm/etc.) back to raw YUV.

    libvmaf's CLI only accepts raw YUV/Y4M on `--distorted`; the
    encoder adapter produces mp4. Without this decode-back step every
    score call returns NaN. See ADR-0237 §"score-path requires raw
    distorted" (Phase A bug-fix).
    """
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        str(dst),
    ]
    completed = runner_fn(cmd, capture_output=True, text=True, check=False)
    rc = int(getattr(completed, "returncode", 1))
    stderr = getattr(completed, "stderr", "") or ""
    return rc, stderr


def _needs_decode(distorted: Path) -> bool:
    return distorted.suffix.lower() not in _RAW_DIST_EXTS


def run_score(
    req: ScoreRequest,
    *,
    vmaf_bin: str = "vmaf",
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
    workdir: Path | None = None,
    backend: str | None = None,
) -> ScoreResult:
    """Drive the vmaf CLI for a single (ref, dist) pair.

    If the distorted path is an encoded container (mp4/webm/...), it is
    transparently decoded to a raw YUV in the scratch workdir first;
    libvmaf's CLI only consumes raw YUV/Y4M.
    ``backend`` selects the libvmaf dispatch path
    (``cpu|cuda|sycl|vulkan``). ``None`` preserves legacy behaviour
    (binary picks its own default).
    """
    runner_fn = runner or subprocess.run

    if workdir is None:
        workdir_ctx = tempfile.TemporaryDirectory()
        workdir_path = Path(workdir_ctx.name)
    else:
        workdir_ctx = None
        workdir_path = workdir
        workdir_path.mkdir(parents=True, exist_ok=True)

    json_path = workdir_path / "vmaf.json"

    score_req = req
    if _needs_decode(req.distorted):
        dist_yuv = workdir_path / "dist.yuv"
        dec_rc, dec_stderr = _decode_to_raw_yuv(
            req.distorted,
            dist_yuv,
            req.pix_fmt,
            ffmpeg_bin=ffmpeg_bin,
            runner_fn=runner_fn,
        )
        if dec_rc != 0 or not dist_yuv.exists():
            if workdir_ctx is not None:
                workdir_ctx.cleanup()
            return ScoreResult(
                request=req,
                vmaf_score=float("nan"),
                score_time_ms=0.0,
                vmaf_binary_version="ffmpeg-decode-failed",
                exit_status=dec_rc or 65,
                stderr_tail=("ffmpeg decode failed:\n" + dec_stderr)[-2048:],
            )
        score_req = dataclasses.replace(req, distorted=dist_yuv)

    cmd = build_vmaf_command(score_req, json_path, vmaf_bin=vmaf_bin)
    cmd = build_vmaf_command(req, json_path, vmaf_bin=vmaf_bin, backend=backend)

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
