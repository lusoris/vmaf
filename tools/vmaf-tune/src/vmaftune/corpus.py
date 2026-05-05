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
from .resolution import select_vmaf_model_version
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
    """Knobs that govern a corpus run."""

    encoder: str = "libx264"
    output: Path = Path("corpus.jsonl")
    encode_dir: Path = Path(".workingdir2/encodes")
    vmaf_model: str = "vmaf_v0.6.1"
    ffmpeg_bin: str = "ffmpeg"
    vmaf_bin: str = "vmaf"
    keep_encodes: bool = False
    src_sha256: bool = True
    # When True (default), the scorer auto-picks the resolution-appropriate
    # VMAF model (1080p vs 4K) per ``resolution.select_vmaf_model_version``.
    # When False, every row scores against ``vmaf_model`` regardless of
    # encode dimensions — useful for legacy corpora that need a single-model
    # baseline. See ADR-0289 + docs/usage/vmaf-tune.md "Resolution-aware mode".
    resolution_aware: bool = True


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

    # Resolve the VMAF model once per job — encode dimensions are fixed
    # across all (preset, crf) cells, so the model never changes inside
    # the loop. When --resolution-aware is off, fall back to the user's
    # explicit `opts.vmaf_model`.
    if opts.resolution_aware:
        effective_model = select_vmaf_model_version(job.width, job.height)
    else:
        effective_model = opts.vmaf_model

    for preset, crf in job.cells:
        adapter.validate(preset, crf)

        # Some codecs (SVT-AV1) want an integer preset on the argv even
        # though the corpus row records the human-readable name. The
        # adapter exposes a translator when it needs one; absent the
        # hook we forward the name verbatim (libx264 path).
        ffmpeg_preset = preset
        translator = getattr(adapter, "ffmpeg_preset_token", None)
        if callable(translator):
            ffmpeg_preset = translator(preset)

        out = _encode_path(opts, job.source, preset, crf)
        enc_req = EncodeRequest(
            source=job.source,
            width=job.width,
            height=job.height,
            pix_fmt=job.pix_fmt,
            framerate=job.framerate,
            encoder=adapter.encoder,
            preset=ffmpeg_preset,
            crf=crf,
            output=out,
        )
        enc_res = run_encode(enc_req, ffmpeg_bin=opts.ffmpeg_bin, runner=encode_runner)

        score_req = ScoreRequest(
            reference=job.source,
            distorted=out,
            width=job.width,
            height=job.height,
            pix_fmt=job.pix_fmt,
            model=effective_model,
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
) -> dict:
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
        "bitrate_kbps": bitrate_kbps(enc_res.encode_size_bytes, job.duration_s),
        "encode_time_ms": enc_res.encode_time_ms,
        "vmaf_score": score_res.vmaf_score,
        # Record the *effective* model used for this row — when
        # `resolution_aware` is on, this can differ from `opts.vmaf_model`
        # (e.g. opts says vmaf_v0.6.1 but a 4K row scored against
        # vmaf_4k_v0.6.1). The score request is the source of truth.
        "vmaf_model": score_res.request.model,
        "score_time_ms": score_res.score_time_ms,
        "ffmpeg_version": enc_res.ffmpeg_version,
        "vmaf_binary_version": score_res.vmaf_binary_version,
        "exit_status": enc_res.exit_status or score_res.exit_status,
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
