# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""ffmpeg/libx264 driver — Phase A.

Wraps a single ffmpeg invocation that re-encodes a raw YUV source with
``libx264`` at a given (preset, crf). Captures wall time, output size,
and the encoder's reported version string.

Subprocess boundary is the integration seam — tests mock
``subprocess.run`` rather than running ffmpeg.

Phase F (ADR-0333) adds 2-pass encoding via :func:`run_two_pass_encode`
and the ``pass_number`` / ``stats_path`` fields on :class:`EncodeRequest`.
The single-pass path (``pass_number == 0``, the default) is unchanged.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
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
    # Phase F (ADR-0333): 2-pass encoding control. ``pass_number`` is
    # 0 (single-pass / default), 1 (first pass — analyse, write
    # stats), or 2 (second pass — read stats, encode). ``stats_path``
    # is the per-encode unique stats file path; required when
    # ``pass_number != 0``. The driver materialises the stats file
    # itself in :func:`run_two_pass_encode`; callers building one
    # ``EncodeRequest`` at a time can leave ``stats_path = None`` for
    # single-pass.
    pass_number: int = 0
    stats_path: Path | None = None


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


def _legacy_codec_args(encoder: str, preset: str, quality: int) -> list[str]:
    """Fallback ``-c:v ... -preset <p> -crf <q>`` shape (HP-1).

    Used when an ``EncodeRequest`` names an encoder not in the
    adapter registry, or when a registered adapter doesn't yet
    expose ``ffmpeg_codec_args``. Mirrors the historic libx264-only
    argv shape so callers that bypass the registry still produce
    something invocable.
    """
    return ["-c:v", encoder, "-preset", preset, "-crf", str(quality)]


def _resolve_codec_args(req: "EncodeRequest") -> list[str]:
    """Resolve the codec-specific argv slice for ``req``.

    Routes through the codec-adapter registry per ADR-0237 / ADR-0326
    (HP-1): every adapter exposes ``ffmpeg_codec_args(preset, quality)``
    that returns the ``-c:v ...`` slice in the codec-correct shape (e.g.
    ``-cpu-used`` for libaom-av1, ``-cq`` for NVENC, ``-global_quality``
    for QSV). When the encoder isn't in the registry (or the adapter
    is a legacy stub without ``ffmpeg_codec_args``) we fall back to
    the historic libx264 shape so existing callers keep working.
    """
    # Local import: ``codec_adapters`` imports ``encode`` indirectly
    # through downstream modules in some test paths; the late binding
    # keeps the dependency one-way at module load. The import statement
    # also re-resolves ``get_adapter`` from the package on every call,
    # so tests that ``mock.patch.object(codec_adapters, "get_adapter", ...)``
    # see the patched callable.
    try:
        from . import codec_adapters as _ca
    except ImportError:  # pragma: no cover - defensive
        return _legacy_codec_args(req.encoder, req.preset, req.crf)

    try:
        adapter = _ca.get_adapter(req.encoder)
    except KeyError:
        return _legacy_codec_args(req.encoder, req.preset, req.crf)

    fn = getattr(adapter, "ffmpeg_codec_args", None)
    if fn is None:
        return _legacy_codec_args(req.encoder, req.preset, req.crf)

    args = fn(req.preset, req.crf)
    # Adapters historically returned tuples; normalise to list so the
    # composed argv is mutation-safe and uniformly typed.
    return list(args)


def build_ffmpeg_command(req: EncodeRequest, ffmpeg_bin: str = "ffmpeg") -> list[str]:
    """Compose the ffmpeg argv for a single encode.

    Pure function — no I/O — so tests can pin the exact command line.

    When ``req.sample_clip_seconds > 0``, ``-ss <start> -t <N>`` are
    inserted as **input-side** options (before ``-i``) so FFmpeg fast-
    seeks the raw YUV by skipping ``start * framerate`` frame-sized
    byte chunks. Output-side seeking would still decode (and the
    rawvideo demuxer would still read) the full source, defeating the
    speedup.

    Phase F (ADR-0333): when ``req.pass_number != 0`` the adapter's
    ``two_pass_args`` argv is spliced in before ``extra_params``; pass
    1 redirects the encoded output to ``-f null -`` (avoiding writing
    a useless pass-1 mp4) while pass 2 keeps the requested
    ``req.output`` destination.

    The codec-specific argv slice (``-c:v ...``) is delegated to the
    codec adapter's ``ffmpeg_codec_args`` per HP-1 / ADR-0326 so
    non-x264 codecs get their correct flags (e.g. ``-cpu-used`` for
    libaom-av1, ``-cq`` for NVENC, ``-global_quality`` for QSV). The
    legacy ``-c:v <enc> -preset <p> -crf <q>`` shape stays available
    as a fallback for unregistered encoders.
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
    cmd.extend(_resolve_codec_args(req))

    # Phase F: 2-pass argv from the codec adapter, when requested.
    if req.pass_number != 0:
        if req.stats_path is None:
            raise ValueError("build_ffmpeg_command: pass_number != 0 requires stats_path")
        # Lazy import to avoid the codec_adapters import cost on
        # plain single-pass paths and to keep the module import
        # graph identical for the legacy fast path.
        from .codec_adapters import get_adapter

        adapter = get_adapter(req.encoder)
        if not getattr(adapter, "supports_two_pass", False):
            raise ValueError(
                f"build_ffmpeg_command: encoder {req.encoder!r} does not "
                "support 2-pass encoding (supports_two_pass = False)"
            )
        cmd.extend(adapter.two_pass_args(req.pass_number, req.stats_path))

    cmd.extend(req.extra_params)

    if req.pass_number == 1:
        # Pass 1 only writes the stats file; the encoded bitstream is
        # discarded via the null muxer. Saves I/O + disk space (some
        # codecs emit hundreds of MB on long sources).
        cmd.extend(["-f", "null", "-"])
    else:
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
    # Pass 1 of a 2-pass encode writes only the stats file; the
    # encoded bitstream is discarded via -f null - and req.output is
    # not produced. Skip the size probe for pass 1 to avoid spurious
    # zeros tripping callers that interpret a zero-size on a non-pass-1
    # encode as failure.
    if rc == 0 and req.pass_number != 1 and req.output.exists():
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


def _stats_path_for(req: EncodeRequest, scratch_dir: Path) -> Path:
    """Build a per-encode unique stats-file path under ``scratch_dir``.

    The stats file name embeds the source stem, encoder, preset, and
    CRF so a debug session can correlate it back to the encode that
    produced it. A short uuid suffix prevents collisions when the
    same (src, codec, preset, crf) is run more than once in parallel.
    """
    stem = f"{req.source.stem}__{req.encoder}__{req.preset}__crf{req.crf}__{uuid.uuid4().hex[:8]}"
    return scratch_dir / f"{stem}.stats"


def run_two_pass_encode(
    req: EncodeRequest,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
    scratch_dir: Path | None = None,
    on_unsupported: str = "fallback",
) -> EncodeResult:
    """Drive a 2-pass ffmpeg encode (Phase F / ADR-0333).

    Runs the encoder twice with the codec adapter's ``two_pass_args``
    spliced into each invocation. Pass 1 redirects to ``-f null -``
    (no output file); pass 2 writes ``req.output``. The stats file
    lives in ``scratch_dir`` (default: a fresh ``tempfile.mkdtemp``)
    and is removed after pass 2 completes regardless of exit status.

    Returns one :class:`EncodeResult` representing the combined op:

    - ``encode_size_bytes`` — pass-2 output size.
    - ``encode_time_ms`` — sum of pass-1 + pass-2 wall time.
    - ``encoder_version`` / ``ffmpeg_version`` — pass-2 stderr (which
      carries the actual encode banner; pass 1 emits the same lines
      but pass 2 is the canonical source).
    - ``exit_status`` — first non-zero of {pass 1, pass 2}, else 0.
    - ``stderr_tail`` — pass-2 stderr tail (pass-1 failures get
      surfaced via ``exit_status``).

    ``on_unsupported`` controls behaviour when the request's encoder
    has ``supports_two_pass = False``:

    - ``"fallback"`` (default) — log a one-line stderr warning and
      run a single-pass encode (returning that result). Mirrors the
      saliency.py "x264-only, fall back to plain encode" precedent.
    - ``"raise"`` — raise :class:`ValueError`. For callers that want
      to fail loudly rather than silently degrade.
    """
    from .codec_adapters import get_adapter

    adapter = get_adapter(req.encoder)
    if not getattr(adapter, "supports_two_pass", False):
        msg = (
            f"vmaf-tune: encoder {req.encoder!r} does not support 2-pass "
            "encoding; falling back to single-pass."
        )
        if on_unsupported == "raise":
            raise ValueError(msg)
        if on_unsupported != "fallback":
            raise ValueError(
                f"run_two_pass_encode: unknown on_unsupported={on_unsupported!r}; "
                "expected 'fallback' or 'raise'"
            )
        sys.stderr.write(msg + "\n")
        return run_encode(req, ffmpeg_bin=ffmpeg_bin, runner=runner)

    own_scratch = scratch_dir is None
    if scratch_dir is None:
        scratch_dir = Path(tempfile.mkdtemp(prefix="vmaftune-2pass-"))

    stats_path = _stats_path_for(req, scratch_dir)
    pass1_req = dataclasses.replace(req, pass_number=1, stats_path=stats_path)
    pass2_req = dataclasses.replace(req, pass_number=2, stats_path=stats_path)

    try:
        pass1 = run_encode(pass1_req, ffmpeg_bin=ffmpeg_bin, runner=runner)
        if pass1.exit_status != 0:
            # Don't bother with pass 2 if pass 1 failed; surface the
            # pass-1 failure in the EncodeResult (with a clarifying
            # tail) so the caller can disambiguate from a pass-2 fault.
            return dataclasses.replace(
                pass1,
                request=req,  # report against the user-supplied request
                stderr_tail=f"[pass 1 failed]\n{pass1.stderr_tail}",
            )
        pass2 = run_encode(pass2_req, ffmpeg_bin=ffmpeg_bin, runner=runner)
        combined_status = pass2.exit_status  # pass1 was 0 by branch above
        return EncodeResult(
            request=req,
            encode_size_bytes=pass2.encode_size_bytes,
            encode_time_ms=pass1.encode_time_ms + pass2.encode_time_ms,
            encoder_version=pass2.encoder_version,
            ffmpeg_version=pass2.ffmpeg_version,
            exit_status=combined_status,
            stderr_tail=pass2.stderr_tail,
        )
    finally:
        # Remove the stats file (libx265 also writes a sidecar
        # ``<stats>.cutree`` — clean both). A user-provided
        # ``scratch_dir`` is left in place; only the auto-tempdir
        # is rmtree'd.
        for candidate in (stats_path, stats_path.with_suffix(stats_path.suffix + ".cutree")):
            try:
                candidate.unlink()
            except (OSError, FileNotFoundError):
                pass
        if own_scratch:
            try:
                # Best-effort cleanup; if anything remains the OS will
                # garbage-collect /tmp eventually.
                for child in scratch_dir.iterdir():
                    try:
                        child.unlink()
                    except OSError:
                        pass
                scratch_dir.rmdir()
            except OSError:
                pass
