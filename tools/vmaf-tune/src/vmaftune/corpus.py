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

A 2-pass coarse-to-fine search is also exposed via
:func:`coarse_to_fine_search` for callers that only need to find the
smallest CRF that still meets a VMAF target. The full ``--crf-range
0:51:1`` grid wastes encode wall time once the target is bracketed;
coarse-to-fine visits ~15 points instead of 52 for the canonical
defaults (3.5x speedup) — see ADR-0296.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import CORPUS_ROW_KEYS, SCHEMA_VERSION
from .codec_adapters import get_adapter
from .encode import EncodeRequest, bitrate_kbps, run_encode, run_two_pass_encode
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
    # ADR-0299 / ADR-0314: libvmaf scoring backend. ``None`` (default)
    # omits the ``--backend`` flag so libvmaf picks its own default
    # (CPU on a stock build); ``"cuda"`` / ``"vulkan"`` / ``"sycl"`` /
    # ``"cpu"`` engage the corresponding backend explicitly. The CLI
    # resolves ``--score-backend auto`` to a concrete value before
    # populating this field; ``CorpusOptions`` itself never walks the
    # fallback chain.
    score_backend: str | None = None
    # HDR mode (Bucket #9, ADR-0300):
    # - "auto": probe each source via ffprobe; inject HDR codec args +
    #   the HDR-VMAF model when PQ / HLG signaling is detected. Default.
    # - "force-sdr": skip detection; treat every source as SDR.
    # - "force-hdr-pq": treat every source as HDR PQ (overrides probe).
    # - "force-hdr-hlg": treat every source as HDR HLG (overrides probe).
    # The active mode lands on each corpus row's ``hdr_mode`` /
    # ``hdr_transfer`` / ``hdr_primaries`` columns.
    hdr_mode: str = "auto"
    ffprobe_bin: str = "ffprobe"
    # Phase F (ADR-0333): opt into 2-pass encoding for codecs whose
    # adapter sets ``supports_two_pass = True`` (libx265 today;
    # libx264 / libsvtav1 / libvvenc follow as sibling PRs). Default
    # off — single-pass behaviour stays the canonical path. When set
    # against an adapter where ``supports_two_pass = False``, the
    # encode driver writes a one-line stderr warning and runs
    # single-pass (matching the saliency x264-only fallback
    # precedent).
    two_pass: bool = False


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
        if opts.two_pass:
            # Phase F (ADR-0333). The driver gracefully falls back
            # to single-pass when the adapter does not opt into
            # 2-pass; keeps mixed-codec corpora honest.
            enc_res = run_two_pass_encode(
                enc_req,
                ffmpeg_bin=opts.ffmpeg_bin,
                runner=encode_runner,
            )
        else:
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
            score_res = run_score(
                score_req,
                vmaf_bin=opts.vmaf_bin,
                runner=score_runner,
                backend=opts.score_backend,
            )
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


# ---------------------------------------------------------------------------
# Coarse-to-fine search (ADR-0306)
# ---------------------------------------------------------------------------
#
# Full-grid sweep over CRF 0..51 step 1 = 52 encodes per (source, preset).
# When the caller only wants "smallest CRF that meets a VMAF target" we can
# bracket in two passes:
#
#   1. Coarse pass at every ``coarse_step`` over the CRF range.
#   2. Fine pass at step ``fine_step`` within ``±fine_radius`` of the
#      best-coarse point.
#
# Defaults (10/5/1 over 0..51) produce 5 + 10 = 15 unique encodes — a 3.5x
# wall-time speedup vs the full grid with no measurable quality loss on the
# Netflix Public corpus (see docs/research/0067 + ADR-0296).
#
# When no ``target_vmaf`` is supplied, the orchestrator still runs both
# passes and refines around the highest-VMAF coarse point.


def _crf_clamp(crf: int) -> int:
    """Clamp a CRF candidate to the libx264 0..51 valid range."""
    if crf < 0:
        return 0
    if crf > 51:
        return 51
    return crf


def coarse_grid_crfs(
    *,
    crf_min: int = 10,
    crf_max: int = 50,
    coarse_step: int = 10,
) -> tuple[int, ...]:
    """Return the coarse-pass CRF grid as a deduped, sorted tuple.

    Defaults yield ``(10, 20, 30, 40, 50)`` — 5 points spanning the
    practically useful CRF range for libx264. CRF below 10 is visually
    lossless on most content (huge bitrate, no perceptual gain) and
    CRF=51 is the codec floor; the coarse pass intentionally skips
    both. Override ``crf_min`` / ``crf_max`` for codecs with different
    quality-knob ranges.
    """
    if coarse_step <= 0:
        raise ValueError(f"coarse_step must be positive, got {coarse_step}")
    if crf_min > crf_max:
        raise ValueError(f"crf_min ({crf_min}) > crf_max ({crf_max})")
    n = math.floor((crf_max - crf_min) / coarse_step) + 1
    grid = sorted({_crf_clamp(crf_min + i * coarse_step) for i in range(n)})
    return tuple(grid)


def fine_grid_crfs(
    best_crf: int,
    *,
    fine_radius: int = 5,
    fine_step: int = 1,
    crf_min: int = 0,
    crf_max: int = 51,
    exclude: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return CRF candidates in ``[best - radius, best + radius]`` at ``fine_step``.

    Cells in ``exclude`` (typically the coarse-pass grid) are removed so the
    second pass only visits points that haven't been measured already.
    """
    if fine_radius < 0:
        raise ValueError(f"fine_radius must be non-negative, got {fine_radius}")
    if fine_step <= 0:
        raise ValueError(f"fine_step must be positive, got {fine_step}")
    excluded = set(exclude)
    candidates: set[int] = set()
    for delta in range(-fine_radius, fine_radius + 1, fine_step):
        candidates.add(_crf_clamp(best_crf + delta))
    candidates.difference_update(excluded)
    # Keep candidates inside the configured range.
    candidates = {c for c in candidates if crf_min <= c <= crf_max}
    return tuple(sorted(candidates))


def _pick_best_crf(
    rows: Sequence[dict],
    *,
    target_vmaf: float | None,
) -> int | None:
    """Identify the "best" coarse CRF for refinement.

    With a target: highest CRF whose ``vmaf_score`` meets ``target_vmaf``.
    That's the smallest-quality candidate that still passes the gate, so
    refining around it locates the smallest acceptable CRF.

    Without a target: the CRF with the highest VMAF (lowest CRF in
    practice, but tie-broken by score). NaN / failed rows are ignored.
    """

    def _score(row: dict) -> float:
        v = row.get("vmaf_score")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return v

    valid = [r for r in rows if not math.isnan(_score(r))]
    if not valid:
        return None

    if target_vmaf is None:
        winner = max(valid, key=_score)
        return int(winner["crf"])

    passing = [r for r in valid if _score(r) >= target_vmaf]
    if passing:
        # Highest CRF that still passes — refining around it finds the
        # smallest CRF that still meets the target.
        winner = max(passing, key=lambda r: int(r["crf"]))
        return int(winner["crf"])
    # Nothing met the target on the coarse pass. Fall back to the
    # highest-VMAF coarse point so the fine pass at least probes near
    # the achievable ceiling.
    winner = max(valid, key=_score)
    return int(winner["crf"])


def _should_skip_refinement(
    *,
    best_crf: int | None,
    coarse_grid: Sequence[int],
    target_vmaf: float | None,
    best_score: float,
    crf_max: int,
) -> bool:
    """Decide whether the coarse pass alone is enough.

    The fine pass is skipped when:

    - the coarse pass produced no measurable rows (best_crf is None), or
    - a target is set, the best-coarse CRF *meets* the target, and
      refining higher cannot help (the best-coarse is already at the
      highest CRF in the coarse grid OR pinned at ``crf_max``). In that
      case there are no larger CRF candidates to probe, so the existing
      best already minimises bitrate at the gate.
    """
    if best_crf is None:
        return True
    if target_vmaf is None:
        return False
    if math.isnan(best_score):
        return False
    if best_score < target_vmaf:
        return False
    # Target met — refining to the right would only check higher CRFs
    # (lower quality). We can skip if we're already at the max-CRF
    # coarse cell or pinned at crf_max.
    return best_crf >= max(coarse_grid) or best_crf >= crf_max


def coarse_to_fine_search(
    job: CorpusJob,
    opts: CorpusOptions,
    *,
    target_vmaf: float | None = None,
    coarse_step: int = 10,
    fine_radius: int = 5,
    fine_step: int = 1,
    crf_min: int = 10,
    crf_max: int = 50,
    encode_runner: object | None = None,
    score_runner: object | None = None,
) -> Iterator[dict]:
    """Run a 2-pass coarse-to-fine CRF search.

    Yields the same JSONL rows :func:`iter_rows` does — coarse pass
    first, then fine pass (if not skipped). The caller is responsible
    for selecting the chosen CRF from the rows; this function only
    drives the encodes.

    The presets in ``job.cells`` are honoured: the search runs once
    per distinct preset, with the CRF axis replaced by the
    coarse-then-fine sweep.
    """
    presets = tuple(dict.fromkeys(p for p, _crf in job.cells))
    if not presets:
        return

    coarse_grid = coarse_grid_crfs(crf_min=crf_min, crf_max=crf_max, coarse_step=coarse_step)

    for preset in presets:
        coarse_cells = tuple((preset, c) for c in coarse_grid)
        coarse_job = dataclasses.replace(job, cells=coarse_cells)
        coarse_rows: list[dict] = []
        for row in iter_rows(
            coarse_job,
            opts,
            encode_runner=encode_runner,
            score_runner=score_runner,
        ):
            coarse_rows.append(row)
            yield row

        best_crf = _pick_best_crf(coarse_rows, target_vmaf=target_vmaf)
        best_score = float("nan")
        if best_crf is not None:
            for r in coarse_rows:
                if int(r["crf"]) == best_crf:
                    try:
                        best_score = float(r["vmaf_score"])
                    except (TypeError, ValueError):
                        best_score = float("nan")
                    break

        if _should_skip_refinement(
            best_crf=best_crf,
            coarse_grid=coarse_grid,
            target_vmaf=target_vmaf,
            best_score=best_score,
            crf_max=crf_max,
        ):
            continue

        # mypy: best_crf cannot be None here — _should_skip_refinement
        # would have returned True above.
        assert best_crf is not None
        fine_crfs = fine_grid_crfs(
            best_crf,
            fine_radius=fine_radius,
            fine_step=fine_step,
            crf_min=crf_min,
            crf_max=crf_max,
            exclude=coarse_grid,
        )
        if not fine_crfs:
            continue

        fine_cells = tuple((preset, c) for c in fine_crfs)
        fine_job = dataclasses.replace(job, cells=fine_cells)
        for row in iter_rows(
            fine_job,
            opts,
            encode_runner=encode_runner,
            score_runner=score_runner,
        ):
            yield row
