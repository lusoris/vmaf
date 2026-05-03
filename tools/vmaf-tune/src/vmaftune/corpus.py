# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase A corpus orchestrator.

Sweeps a (preset, crf) grid against one or more raw YUV references,
runs the encoder, scores each encode against the reference with the
libvmaf CLI, and writes one JSONL row per (source, preset, crf)
combination.

Schema lives in :mod:`vmaftune` (``CORPUS_ROW_KEYS``,
``SCHEMA_VERSION``). Phase B/C are downstream consumers — bumping the
schema is a coordinated change.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import CORPUS_ROW_KEYS, SCHEMA_VERSION
from .codec_adapters import get_adapter
from .encode import EncodeRequest, bitrate_kbps, run_encode
from .score import ScoreRequest, ScoreResult, run_score


@dataclasses.dataclass(frozen=True)
class CorpusJob:
    """One source + a list of (preset, crf) cells to evaluate."""

    source: Path
    width: int
    height: int
    pix_fmt: str
    framerate: float
    duration_s: float
    cells: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True)
class CorpusOptions:
    """Knobs that govern a corpus run.

    ``sample_clip_seconds`` opts the run into sample-clip mode
    (ADR-0297): each grid point encodes the centre N-second window of
    the reference YUV instead of the full source, scoring the matching
    reference window via the libvmaf CLI's ``--frame_skip_ref`` /
    ``--frame_cnt``. ``0.0`` (default) keeps the legacy full-source
    behaviour. The encoded clip's bitrate and timing are reported as
    measured on the slice — Phase B/C should weight or filter rows on
    ``clip_mode`` rather than mixing sample and full rows blindly.
    """

    encoder: str = "libx264"
    output: Path = Path("corpus.jsonl")
    encode_dir: Path = Path(".workingdir2/encodes")
    vmaf_model: str = "vmaf_v0.6.1"
    ffmpeg_bin: str = "ffmpeg"
    vmaf_bin: str = "vmaf"
    keep_encodes: bool = False
    src_sha256: bool = True
    sample_clip_seconds: float = 0.0


def _sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _encode_path(opts: CorpusOptions, source: Path, preset: str, crf: int) -> Path:
    stem = f"{source.stem}__{opts.encoder}__{preset}__crf{crf}.mp4"
    return opts.encode_dir / stem


def _resolve_sample_clip(
    job: "CorpusJob", opts: CorpusOptions
) -> tuple[float, float, int, int, str]:
    """Return ``(clip_seconds, start_s, frame_skip_ref, frame_cnt, clip_mode)``.

    Caps the requested slice at ``job.duration_s`` (so a 10-second
    request against an 8-second source falls back to full-clip rather
    than encoding a short tail). Centre-anchored: ``start_s = (D - N) / 2``.
    Returns the no-op tuple ``(0.0, 0.0, 0, 0, "full")`` when sample-clip
    mode is off or the source is too short.
    """
    requested = float(opts.sample_clip_seconds)
    duration = float(job.duration_s)
    if requested <= 0.0 or duration <= 0.0 or requested >= duration:
        return (0.0, 0.0, 0, 0, "full")
    start_s = max(0.0, (duration - requested) / 2.0)
    # libvmaf CLI takes integer frame counts; framerate may be
    # fractional (e.g. 23.976). Round to nearest to keep the window
    # symmetric around the centre.
    frame_skip_ref = int(round(start_s * job.framerate))
    frame_cnt = int(round(requested * job.framerate))
    label = f"sample_{int(round(requested))}s"
    return (requested, start_s, frame_skip_ref, frame_cnt, label)


def iter_rows(
    job: CorpusJob,
    opts: CorpusOptions,
    *,
    encode_runner: object | None = None,
    score_runner: object | None = None,
) -> Iterator[dict]:
    """Yield one JSONL row per (preset, crf) cell.

    ``encode_runner`` / ``score_runner`` are subprocess-runner stubs
    parameterised for tests. Production callers leave them ``None``.
    """
    adapter = get_adapter(opts.encoder)
    src_hash = _sha256_of(job.source) if (opts.src_sha256 and job.source.exists()) else ""

    opts.encode_dir.mkdir(parents=True, exist_ok=True)

    clip_seconds, start_s, frame_skip_ref, frame_cnt, clip_mode = _resolve_sample_clip(job, opts)

    for preset, crf in job.cells:
        adapter.validate(preset, crf)

        out = _encode_path(opts, job.source, preset, crf)
        enc_req = EncodeRequest(
            source=job.source,
            width=job.width,
            height=job.height,
            pix_fmt=job.pix_fmt,
            framerate=job.framerate,
            encoder=adapter.encoder,
            preset=preset,
            crf=crf,
            output=out,
            sample_clip_seconds=clip_seconds,
            sample_clip_start_s=start_s,
        )
        enc_res = run_encode(enc_req, ffmpeg_bin=opts.ffmpeg_bin, runner=encode_runner)

        score_req = ScoreRequest(
            reference=job.source,
            distorted=out,
            width=job.width,
            height=job.height,
            pix_fmt=job.pix_fmt,
            model=opts.vmaf_model,
            frame_skip_ref=frame_skip_ref,
            frame_cnt=frame_cnt,
        )
        if enc_res.exit_status == 0:
            score_res = run_score(score_req, vmaf_bin=opts.vmaf_bin, runner=score_runner)
        else:
            # Skip scoring on encode failure; row records the failure.
            score_res = ScoreResult(
                request=score_req,
                vmaf_score=float("nan"),
                score_time_ms=0.0,
                vmaf_binary_version="skipped",
                exit_status=enc_res.exit_status,
                stderr_tail="encode failed; score skipped",
            )

        row = _row_for(
            job=job,
            opts=opts,
            preset=preset,
            crf=crf,
            src_sha=src_hash,
            enc_res=enc_res,
            score_res=score_res,
            clip_mode=clip_mode,
        )
        if not opts.keep_encodes and out.exists() and enc_res.exit_status == 0:
            # best-effort cleanup; corpus row stays valid either way
            with contextlib.suppress(OSError):
                out.unlink()
        yield row


def _row_for(
    *,
    job: CorpusJob,
    opts: CorpusOptions,
    preset: str,
    crf: int,
    src_sha: str,
    enc_res,
    score_res,
    clip_mode: str = "full",
) -> dict:
    # Bitrate is computed against the *encoded* duration so sample-clip
    # rows aren't biased low by dividing slice-bytes by full-source
    # seconds. ``duration_s`` keeps the source provenance.
    encoded_duration_s = (
        enc_res.request.sample_clip_seconds
        if enc_res.request.sample_clip_seconds > 0.0
        else job.duration_s
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "timestamp": _utc_now_iso(),
        "src": str(job.source),
        "src_sha256": src_sha,
        "width": job.width,
        "height": job.height,
        "pix_fmt": job.pix_fmt,
        "framerate": job.framerate,
        "duration_s": job.duration_s,
        "encoder": opts.encoder,
        "encoder_version": enc_res.encoder_version,
        "preset": preset,
        "crf": crf,
        "extra_params": list(enc_res.request.extra_params),
        "encode_path": (str(enc_res.request.output) if opts.keep_encodes else ""),
        "encode_size_bytes": enc_res.encode_size_bytes,
        "bitrate_kbps": bitrate_kbps(enc_res.encode_size_bytes, encoded_duration_s),
        "encode_time_ms": enc_res.encode_time_ms,
        "vmaf_score": score_res.vmaf_score,
        "vmaf_model": opts.vmaf_model,
        "score_time_ms": score_res.score_time_ms,
        "ffmpeg_version": enc_res.ffmpeg_version,
        "vmaf_binary_version": score_res.vmaf_binary_version,
        "exit_status": enc_res.exit_status or score_res.exit_status,
        "clip_mode": clip_mode,
    }
    # Schema-shape assertion — catches drift in development; cheap.
    missing = set(CORPUS_ROW_KEYS) - row.keys()
    if missing:
        raise AssertionError(f"corpus row missing keys: {sorted(missing)}")
    return row


def write_jsonl(rows: Sequence[dict] | Iterator[dict], path: Path) -> int:
    """Write ``rows`` to ``path`` (one JSON object per line). Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write(os.linesep)
            n += 1
    return n
