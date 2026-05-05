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
import logging
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import CORPUS_ROW_KEYS, SCHEMA_VERSION
from .cache import CachedResult, TuneCache, cache_key
from .codec_adapters import get_adapter
from .encode import EncodeRequest, EncodeResult, bitrate_kbps, probe_ffmpeg_version, run_encode
from .hdr import HdrInfo, detect_hdr, hdr_codec_args, select_hdr_vmaf_model
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
    ffprobe_bin: str = "ffprobe"
    vmaf_bin: str = "vmaf"
    keep_encodes: bool = False
    src_sha256: bool = True
    # When True (default), the scorer auto-picks the resolution-appropriate
    # VMAF model (1080p vs 4K) per ``resolution.select_vmaf_model_version``.
    # When False, every row scores against ``vmaf_model`` regardless of
    # encode dimensions — useful for legacy corpora that need a single-model
    # baseline. See ADR-0289 + docs/usage/vmaf-tune.md "Resolution-aware mode".
    resolution_aware: bool = True
    score_backend: str | None = None  # libvmaf --backend value; None = binary default
    # HDR mode (Bucket #9, ADR-0295).
    # - "auto": probe each source via ffprobe, inject HDR flags + HDR
    #   model if signaling found. Default — safe for SDR sources.
    # - "force-sdr": skip detection; treat every source as SDR.
    # - "force-hdr-pq": treat every source as HDR PQ (overrides probe).
    # - "force-hdr-hlg": treat every source as HDR HLG.
    hdr_mode: str = "auto"
    # ADR-0298: content-addressed cache. Default ON; honours
    # XDG_CACHE_HOME via cache.default_cache_dir().
    cache_enabled: bool = True
    cache_dir: Path | None = None
    cache_size_bytes: int = 10 * 1024 * 1024 * 1024


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
    probe_runner: object | None = None,
) -> Iterator[dict]:
    """Yield one JSONL row per (preset, crf) cell.

    ``encode_runner`` / ``score_runner`` are subprocess-runner stubs
    parameterised for tests. Production callers leave them ``None``.
    ``probe_runner`` is the matching seam for the one-shot
    ``ffmpeg -version`` probe used by the cache layer.
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
    hdr_info, hdr_forced = _resolve_hdr(opts, job.source)
    hdr_extra = hdr_codec_args(adapter.encoder, hdr_info) if hdr_info is not None else ()
    effective_model = _resolve_vmaf_model(opts, hdr_info)

    cache, ffmpeg_v = _maybe_build_cache(opts, src_hash, probe_runner=probe_runner)

    for preset, crf in job.cells:
        adapter.validate(preset, crf)
        yield _row_for_cell(
            job=job,
            opts=opts,
            adapter=adapter,
            preset=preset,
            crf=crf,
            src_sha=src_hash,
            hdr_info=hdr_info,
            hdr_forced=hdr_forced,
            hdr_extra=hdr_extra,
            effective_model=effective_model,
            cache=cache,
            ffmpeg_v=ffmpeg_v,
            encode_runner=encode_runner,
            score_runner=score_runner,
        )


def _maybe_build_cache(
    opts: CorpusOptions,
    src_hash: str,
    *,
    probe_runner: object | None,
) -> tuple[TuneCache | None, str]:
    """Construct a ``TuneCache`` and probe ffmpeg version, or return
    ``(None, "")`` if caching is off / unusable.

    Caching is disabled when:
    - ``opts.cache_enabled`` is False, or
    - the source hash is empty (no stable content key), or
    - ffmpeg version cannot be probed (the encode would fail anyway).
    """
    if not opts.cache_enabled or not src_hash:
        return None, ""
    ffmpeg_v = probe_ffmpeg_version(opts.ffmpeg_bin, runner=probe_runner)
    if ffmpeg_v == "unknown":
        return None, ""
    cache = TuneCache(opts.cache_dir, size_bytes=opts.cache_size_bytes)
    return cache, ffmpeg_v


def _row_for_cell(
    *,
    job: CorpusJob,
    opts: CorpusOptions,
    adapter,
    preset: str,
    crf: int,
    src_hash: str,
    cache: TuneCache | None,
    ffmpeg_v: str,
    encode_runner: object | None,
    score_runner: object | None,
) -> dict:
    """Encode + score one cell, with cache lookup on the front end.

    On cache hit: synthesise ``EncodeResult`` / ``ScoreResult`` from
    the cached tuple and skip both subprocess calls. On miss: run
    encode + score normally and write back to the cache before
    returning.
    """
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
    )

    key = ""
    if cache is not None:
        key = cache_key(
            src_sha256=src_hash,
            encoder=opts.encoder,
            preset=preset,
            crf=crf,
            adapter_version=getattr(adapter, "adapter_version", "1"),
            ffmpeg_version=ffmpeg_v,
        )
        hit = cache.get(key)
        if hit is not None:
            enc_res, score_res = _results_from_cache(enc_req, opts, hit)
            return _row_for(
                job=job,
                opts=opts,
                preset=preset,
                crf=crf,
                src_sha=src_hash,
                enc_res=enc_res,
                score_res=score_res,
            )

    enc_res = run_encode(enc_req, ffmpeg_bin=opts.ffmpeg_bin, runner=encode_runner)
    score_req = ScoreRequest(
        reference=job.source,
        distorted=out,
        width=job.width,
        height=job.height,
        pix_fmt=job.pix_fmt,
        model=opts.vmaf_model,
    )
    if enc_res.exit_status == 0:
        score_res = run_score(score_req, vmaf_bin=opts.vmaf_bin, runner=score_runner)
    else:
        score_res = ScoreResult(
            request=score_req,
            vmaf_score=float("nan"),
            score_time_ms=0.0,
            vmaf_binary_version="skipped",
            exit_status=enc_res.exit_status,
            stderr_tail="encode failed; score skipped",
        )

    if (
        cache is not None
        and key
        and enc_res.exit_status == 0
        and score_res.exit_status == 0
        and out.exists()
    ):
        cache.put(
            key,
            CachedResult(
                encode_size_bytes=enc_res.encode_size_bytes,
                encode_time_ms=enc_res.encode_time_ms,
                encoder_version=enc_res.encoder_version,
                ffmpeg_version=enc_res.ffmpeg_version,
                vmaf_score=score_res.vmaf_score,
                vmaf_model=opts.vmaf_model,
                score_time_ms=score_res.score_time_ms,
                vmaf_binary_version=score_res.vmaf_binary_version,
                artifact_path=out,
            ),
            artifact_path=out,
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
        with contextlib.suppress(OSError):
            out.unlink()
    return row


def _results_from_cache(
    enc_req: EncodeRequest,
    opts: CorpusOptions,
    hit: CachedResult,
) -> tuple[EncodeResult, ScoreResult]:
    """Reconstruct ``EncodeResult`` / ``ScoreResult`` from a cache hit
    so the row-builder downstream doesn't need to know the cache
    exists.
    """
    enc_res = EncodeResult(
        request=enc_req,
        encode_size_bytes=hit.encode_size_bytes,
        encode_time_ms=hit.encode_time_ms,
        encoder_version=hit.encoder_version,
        ffmpeg_version=hit.ffmpeg_version,
        exit_status=0,
        stderr_tail="cache hit; encode skipped",
    )
    score_req = ScoreRequest(
        reference=enc_req.source,
        distorted=enc_req.output,
        width=enc_req.width,
        height=enc_req.height,
        pix_fmt=enc_req.pix_fmt,
        model=opts.vmaf_model,
    )
    score_res = ScoreResult(
        request=score_req,
        vmaf_score=hit.vmaf_score,
        score_time_ms=hit.score_time_ms,
        vmaf_binary_version=hit.vmaf_binary_version,
        exit_status=0,
        stderr_tail="cache hit; score skipped",
    )
    return enc_res, score_res


def _row_for(
    *,
    job: CorpusJob,
    opts: CorpusOptions,
    preset: str,
    crf: int,
    src_sha: str,
    enc_res,
    score_res,
    hdr_info: HdrInfo | None,
    hdr_forced: bool,
    effective_model: str,
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
        "vmaf_model": effective_model,
        "score_time_ms": score_res.score_time_ms,
        "ffmpeg_version": enc_res.ffmpeg_version,
        "vmaf_binary_version": score_res.vmaf_binary_version,
        "exit_status": enc_res.exit_status or score_res.exit_status,
        "hdr_transfer": hdr_info.transfer if hdr_info else "",
        "hdr_primaries": hdr_info.primaries if hdr_info else "",
        "hdr_forced": hdr_forced,
    }
    # Schema-shape assertion — catches drift in development; cheap.
    missing = set(CORPUS_ROW_KEYS) - row.keys()
    if missing:
        raise AssertionError(f"corpus row missing keys: {sorted(missing)}")
    return row


def _resolve_hdr(opts: CorpusOptions, source: Path) -> tuple[HdrInfo | None, bool]:
    """Apply ``opts.hdr_mode`` over the source's detected HDR signaling.

    Returns ``(hdr_info, forced)``. ``forced`` is true iff the user
    overrode the probe via ``--force-hdr-*`` / ``--force-sdr``.
    """
    mode = opts.hdr_mode
    if mode == "force-sdr":
        return None, True
    if mode == "force-hdr-pq":
        return _synthetic_hdr_info("pq"), True
    if mode == "force-hdr-hlg":
        return _synthetic_hdr_info("hlg"), True
    # auto: probe.
    return detect_hdr(source, ffprobe_bin=opts.ffprobe_bin), False


def _synthetic_hdr_info(transfer: str) -> HdrInfo:
    """Build a minimal :class:`HdrInfo` for ``--force-hdr-*`` overrides."""
    return HdrInfo(
        transfer=transfer,
        primaries="bt2020",
        matrix="bt2020nc",
        color_range="tv",
        pix_fmt="yuv420p10le",
    )


def _resolve_vmaf_model(opts: CorpusOptions, hdr_info: HdrInfo | None) -> str:
    """Pick the VMAF model: HDR-trained if shipped + source is HDR, else SDR.

    Returns the model identifier string for the ``vmaf --model`` arg.
    Falls back to ``opts.vmaf_model`` and logs a one-shot warning if
    HDR was detected but no HDR model is shipped.
    """
    if hdr_info is None:
        return opts.vmaf_model
    hdr_path = select_hdr_vmaf_model()
    if hdr_path is None:
        # Fork hasn't ported Netflix's HDR model yet — see ADR-0295
        # follow-up backlog item.
        logging.getLogger(__name__).warning(
            "hdr-model: source is %s HDR but no vmaf_hdr_*.json found; "
            "falling back to SDR model %r (scores will trend low)",
            hdr_info.transfer,
            opts.vmaf_model,
        )
        return opts.vmaf_model
    # libvmaf accepts a path= form for explicit model JSON paths.
    return f"path={hdr_path}"


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
