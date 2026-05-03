# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""ffmpeg/libx264 driver — Phase A.

Wraps a single ffmpeg invocation that re-encodes a raw YUV source with
``libx264`` at a given (preset, crf). Captures wall time, output size,
and the encoder's reported version string.

Subprocess boundary is the integration seam — tests mock
``subprocess.run`` rather than running ffmpeg.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class EncodeRequest:
    """Single (preset, crf) request against one raw YUV source.

    ``sample_clip_seconds`` opts the request into sample-clip mode
    (ADR-0297): FFmpeg input is sliced to the centre N-second window of
    the reference, cutting encode time roughly linearly with the slice
    length. ``0.0`` (default) keeps the legacy full-source encode.
    ``sample_clip_start_s`` is the start offset (set by the caller from
    ``duration_s`` and ``sample_clip_seconds``); the encode driver does
    not recompute it so that the score driver can mirror the same
    window via ``--frame_skip_ref`` / ``--frame_cnt``.
    """

    source: Path
    width: int
    height: int
    pix_fmt: str
    framerate: float
    encoder: str
    preset: str
    crf: int
    output: Path
    extra_params: tuple[str, ...] = ()
    sample_clip_seconds: float = 0.0
    sample_clip_start_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class EncodeResult:
    """Outcome of one encode call."""

    request: EncodeRequest
    encode_size_bytes: int
    encode_time_ms: float
    encoder_version: str
    ffmpeg_version: str
    exit_status: int
    stderr_tail: str


def build_ffmpeg_command(req: EncodeRequest, ffmpeg_bin: str = "ffmpeg") -> list[str]:
    """Compose the ffmpeg argv for a single encode.

    Pure function — no I/O — so tests can pin the exact command line.

    When ``req.sample_clip_seconds > 0``, ``-ss <start> -t <N>`` are
    inserted as **input-side** options (before ``-i``) so FFmpeg fast-
    seeks the raw YUV by skipping ``start * framerate`` frame-sized
    byte chunks. Output-side seeking would still decode (and the
    rawvideo demuxer would still read) the full source, defeating the
    speedup.
    """
    cmd: list[str] = [
        ffmpeg_bin,
        "-y",  # overwrite
        "-hide_banner",
        "-loglevel",
        "info",
        "-f",
        "rawvideo",
        "-pix_fmt",
        req.pix_fmt,
        "-s",
        f"{req.width}x{req.height}",
        "-r",
        f"{req.framerate}",
    ]
    if req.sample_clip_seconds > 0.0:
        # Input-side -ss / -t — fast-seek for raw YUV.
        cmd.extend(["-ss", f"{req.sample_clip_start_s}"])
        cmd.extend(["-t", f"{req.sample_clip_seconds}"])
    cmd.extend(["-i", str(req.source)])
    cmd.extend(
        [
            "-c:v",
            req.encoder,
            "-preset",
            req.preset,
            "-crf",
            str(req.crf),
        ]
    )
    cmd.extend(req.extra_params)
    cmd.append(str(req.output))
    return cmd


_FFMPEG_VERSION_RE = re.compile(r"ffmpeg version (\S+)")
_X264_VERSION_RE = re.compile(r"x264 - core (\d+)")


def parse_versions(stderr: str) -> tuple[str, str]:
    """Return (ffmpeg_version, x264_version) extracted from stderr.

    Returns ``("unknown", "unknown")`` for missing matches rather than
    raising — corpus rows record what we can detect and move on.
    """
    ffm = _FFMPEG_VERSION_RE.search(stderr)
    enc = _X264_VERSION_RE.search(stderr)
    return (
        ffm.group(1) if ffm else "unknown",
        f"libx264-{enc.group(1)}" if enc else "unknown",
    )


def run_encode(
    req: EncodeRequest,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
) -> EncodeResult:
    """Drive ffmpeg to produce ``req.output``.

    ``runner`` defaults to ``subprocess.run`` and is parameterised so
    tests inject a stub.
    """
    cmd = build_ffmpeg_command(req, ffmpeg_bin=ffmpeg_bin)
    runner_fn = runner or subprocess.run
    started = time.monotonic()
    completed = runner_fn(  # type: ignore[operator]
        cmd, capture_output=True, text=True, check=False
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0

    stderr = getattr(completed, "stderr", "") or ""
    rc = int(getattr(completed, "returncode", 1))

    size = 0
    if rc == 0 and req.output.exists():
        size = os.path.getsize(req.output)

    ffmpeg_v, encoder_v = parse_versions(stderr)
    return EncodeResult(
        request=req,
        encode_size_bytes=size,
        encode_time_ms=elapsed_ms,
        encoder_version=encoder_v,
        ffmpeg_version=ffmpeg_v,
        exit_status=rc,
        stderr_tail=_tail(stderr, n=2048),
    )


def _tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[-n:]


def bitrate_kbps(size_bytes: int, duration_s: float) -> float:
    """File-size-derived bitrate. 0 if duration is non-positive."""
    if duration_s <= 0:
        return 0.0
    return (size_bytes * 8.0 / 1000.0) / duration_s


def iter_grid(presets: Sequence[str], crfs: Sequence[int]) -> list[tuple[str, int]]:
    """Cartesian product of presets x crfs as a deterministic list."""
    return [(p, c) for p in presets for c in crfs]
