# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""ffmpeg encode driver — codec-agnostic dispatcher.

Phase A originally hard-wired ``libx264`` (single-pass CRF). The
multi-codec follow-up (ADR-0294) replaces that with a thin dispatcher
that asks the registered codec adapter for its FFmpeg argv slice and
plugs it into the encode invocation. The harness itself never branches
on codec identity — that invariant is what unblocks the parallel
adapter PRs (libx265, libsvtav1, NVENC, QSV, AMF, libaom, VVenC,
VideoToolbox, ...).

Adapter contract (duck-typed; see
``codec_adapters/README.md`` once the per-codec PRs land):

- ``adapter.ffmpeg_codec_args(preset: str, quality: int) -> list[str]``
  — codec-specific encoder argv slice (``-c:v <enc> -preset <p>
  <quality-knob> <q>`` or whatever shape that codec needs). Optional;
  if absent, the dispatcher falls back to the legacy x264-CRF shape so
  any pre-contract adapter still works.
- ``adapter.extra_params() -> Sequence[str]`` — additional argv (e.g.
  ``-pix_fmt yuv420p`` overrides). Optional; defaults to empty.
- ``adapter.encoder: str`` — the FFmpeg ``-c:v`` value (used by both
  the fallback path and the version parser).

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

from . import codec_adapters as _codec_adapters


@dataclasses.dataclass(frozen=True)
class EncodeRequest:
    """Single (preset, quality) request against one raw YUV source.

    ``crf`` is the historical name of the quality knob (kept verbatim
    for x264 compatibility and the Phase A row schema). The
    codec-agnostic dispatcher reads ``quality`` instead — it returns
    ``crf`` unchanged so x264 callers, tests, and the JSONL row stay
    bit-identical.
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

    @property
    def quality(self) -> int:
        """Codec-agnostic name for the quality knob value.

        Mirrors ``crf`` for x264; for codecs whose knob is ``-cq`` /
        ``-qp`` / ``--rc cqp -q`` / etc., the adapter still receives a
        plain integer here and maps it to its own flag in
        ``ffmpeg_codec_args``.
        """
        return self.crf


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


def _legacy_x264_codec_args(encoder: str, preset: str, quality: int) -> list[str]:
    """Pre-dispatcher x264 argv slice. Used only as a fallback for
    adapters that haven't yet exposed ``ffmpeg_codec_args``."""
    return ["-c:v", encoder, "-preset", preset, "-crf", str(quality)]


def _adapter_codec_args(adapter: object, encoder: str, preset: str, quality: int) -> list[str]:
    fn = getattr(adapter, "ffmpeg_codec_args", None)
    if fn is None:
        return _legacy_x264_codec_args(encoder, preset, quality)
    args = fn(preset, quality)
    return list(args)


def _adapter_extra_params(adapter: object) -> list[str]:
    fn = getattr(adapter, "extra_params", None)
    if fn is None:
        return []
    if callable(fn):
        return list(fn())
    # Some adapters expose extra_params as a tuple/list attribute (legacy).
    return list(fn)  # type: ignore[arg-type]


def build_ffmpeg_command(req: EncodeRequest, ffmpeg_bin: str = "ffmpeg") -> list[str]:
    """Compose the ffmpeg argv for a single encode.

    Pure function — no I/O — so tests can pin the exact command line.
    Looks up the codec adapter via ``codec_adapters.get_adapter`` and
    asks it for its argv slice. Falls back to the legacy x264-CRF shape
    when the adapter is missing or doesn't expose
    ``ffmpeg_codec_args``.
    """
    try:
        adapter: object | None = _codec_adapters.get_adapter(req.encoder)
    except KeyError:
        adapter = None

    codec_args = (
        _adapter_codec_args(adapter, req.encoder, req.preset, req.quality)
        if adapter is not None
        else _legacy_x264_codec_args(req.encoder, req.preset, req.quality)
    )
    extra_from_adapter = _adapter_extra_params(adapter) if adapter is not None else []

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
        "-i",
        str(req.source),
    ]
    cmd.extend(codec_args)
    cmd.extend(extra_from_adapter)
    cmd.extend(req.extra_params)
    cmd.append(str(req.output))
    return cmd


_FFMPEG_VERSION_RE = re.compile(r"ffmpeg version (\S+)")
_X264_VERSION_RE = re.compile(r"x264 - core (\d+)")
_X265_VERSION_RE = re.compile(r"x265 \[info\]:.*?version (\S+)")
_SVTAV1_VERSION_RE = re.compile(r"SVT-AV1.*?\sv?(\d+\.\d+(?:\.\d+)?)")
_LIBVPX_VERSION_RE = re.compile(r"libvpx-vp9.*?(\d+\.\d+\.\d+)|vpxenc\s+v(\d+\.\d+\.\d+)")
_LIBAOM_VERSION_RE = re.compile(r"libaom.*?(\d+\.\d+\.\d+)|aomenc.*?(\d+\.\d+\.\d+)")
_VVENC_VERSION_RE = re.compile(r"VVenC.*?Version\s+(\S+)|libvvenc\s+(\S+)")
_NVENC_VERSION_RE = re.compile(r"\b(h264_nvenc|hevc_nvenc|av1_nvenc)\b")
_QSV_VERSION_RE = re.compile(r"\b(h264_qsv|hevc_qsv|av1_qsv|vp9_qsv)\b")
_AMF_VERSION_RE = re.compile(r"\b(h264_amf|hevc_amf|av1_amf)\b")
_VT_VERSION_RE = re.compile(r"\b(h264_videotoolbox|hevc_videotoolbox|prores_videotoolbox)\b")


# Per-encoder version probe table. Each entry maps an FFmpeg ``-c:v``
# encoder name (or a stem prefix) to (regex, formatter). Adapters ship
# their own entries when registered; this table is the
# stdlib-bundled fallback so the harness still extracts versions for
# the codecs we already wire.
_ENCODER_VERSION_PROBES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("libx264", _X264_VERSION_RE, "libx264-{0}"),
    ("libx265", _X265_VERSION_RE, "libx265-{0}"),
    ("libsvtav1", _SVTAV1_VERSION_RE, "libsvtav1-{0}"),
    ("libvpx-vp9", _LIBVPX_VERSION_RE, "libvpx-vp9-{0}"),
    ("libaom-av1", _LIBAOM_VERSION_RE, "libaom-av1-{0}"),
    ("libvvenc", _VVENC_VERSION_RE, "libvvenc-{0}"),
    ("nvenc", _NVENC_VERSION_RE, "{0}"),
    ("qsv", _QSV_VERSION_RE, "{0}"),
    ("amf", _AMF_VERSION_RE, "{0}"),
    ("videotoolbox", _VT_VERSION_RE, "{0}"),
)


def _probe_for(encoder: str) -> tuple[re.Pattern[str], str] | None:
    """Return the (regex, formatter) probe matching ``encoder``.

    Match is by substring on either side so ``h264_nvenc`` resolves to
    the ``nvenc`` probe and ``libx264`` resolves to its exact entry.
    """
    if not encoder:
        return None
    for stem, pat, fmt in _ENCODER_VERSION_PROBES:
        if stem == encoder or stem in encoder or encoder in stem:
            return (pat, fmt)
    return None


def parse_versions(stderr: str, encoder: str = "libx264") -> tuple[str, str]:
    """Return ``(ffmpeg_version, encoder_version)`` from ffmpeg stderr.

    The ``encoder`` keyword selects which encoder version probe to run.
    Defaults to ``libx264`` so the legacy x264 callers stay
    bit-compatible. Returns ``("unknown", "unknown")`` for missing
    matches rather than raising — the corpus row records what we
    detect and moves on.
    """
    ffm = _FFMPEG_VERSION_RE.search(stderr)
    ffmpeg_v = ffm.group(1) if ffm else "unknown"

    probe = _probe_for(encoder)
    if probe is None:
        return (ffmpeg_v, "unknown")

    pat, fmt = probe
    m = pat.search(stderr)
    if m is None:
        return (ffmpeg_v, "unknown")

    # Pick the first non-empty group so alternation patterns
    # (``A|B``) collapse to the matched side.
    groups = [g for g in m.groups() if g]
    token = groups[0] if groups else m.group(0)
    return (ffmpeg_v, fmt.format(token))


def run_encode(
    req: EncodeRequest,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: object | None = None,
    encoder_runner: object | None = None,
) -> EncodeResult:
    """Drive ffmpeg to produce ``req.output``.

    ``runner`` (or its alias ``encoder_runner``, kept for parity with
    the dispatcher's documented kwarg) defaults to ``subprocess.run``
    and is parameterised so tests inject a stub.
    """
    cmd = build_ffmpeg_command(req, ffmpeg_bin=ffmpeg_bin)
    runner_fn = encoder_runner or runner or subprocess.run
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

    ffmpeg_v, encoder_v = parse_versions(stderr, encoder=req.encoder)
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
