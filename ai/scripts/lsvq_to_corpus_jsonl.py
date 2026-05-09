#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""LSVQ → MOS-corpus JSONL adapter (ADR-0367).

LSVQ — LIVE Large-Scale Social Video Quality (Ying, Mandal,
Ghadiyaram, Bovik; ICCV 2021) — is the canonical large-scale
no-reference VQA training corpus: ~39 000 user-generated videos
with ~5.5 M individual subjective ratings collapsed into per-clip
MOS values on a 1.0–5.0 scale. Train split is ~28 056 clips
(``LSVQ_whole_train``); the held-out leaderboard splits are
``LSVQ_test`` (~7 400 clips, mixed-resolution) and
``LSVQ_test_1080p`` (~3 600 1080p clips). DOVER, FAST-VQA,
Q-Align, MaxVQA, and every other open-weight NR-VQA model trained
in the 2021–2025 window benchmarks PLCC against
``LSVQ_test`` / ``LSVQ_test_1080p``.

The fork's ``nr_metric_v1`` (~19 K params, KoNViD-1k-only) needs
LSVQ in addition to KonViD-150k (ADR-0325 Phase 2) to compete with
DOVER-Mobile (PLCC 0.853 KoNViD / 0.867 LSVQ_test, ~9.86 M
params). KonViD alone is too narrow a content distribution.
LSVQ is hosted on Hugging Face under ``teowu/LSVQ-videos`` —
mirror of the LIVE/Patch-VQ author drop. License is CC-BY-4.0
per the dataset card.

Pipeline shape::

    .workingdir2/lsvq/
      ├── .download-progress.json           # resumable state (this script)
      ├── manifest.csv                      # split-CSV (operator drops)
      ├── clips/                            # downloaded *.mp4 (this script)
      │     ├── 0001.mp4
      │     └── ...
      └── lsvq.jsonl                        # output (this script)

                  │
                  ▼  ai/scripts/lsvq_to_corpus_jsonl.py
                  │
                  ▼
    .workingdir2/lsvq/lsvq.jsonl   # one row per surviving clip

Schema (one JSON object per line — same shape as the KonViD-150k
adapter, only the ``corpus`` literal differs)::

    {
      "src":               "<basename>.mp4",
      "src_sha256":        "<hex>",
      "src_size_bytes":    <int>,
      "width":             <int>,
      "height":            <int>,
      "framerate":         <float>,
      "duration_s":        <float>,
      "pix_fmt":           "<yuv420p|...>",
      "encoder_upstream":  "<ffprobe codec_name; e.g. h264, vp9>",
      "mos":               <float, 1..5>,
      "mos_std_dev":       <float>,
      "n_ratings":         <int>,
      "corpus":            "lsvq",
      "corpus_version":    "<dataset version string>",
      "ingested_at_utc":   "<ISO 8601>"
    }

Manifest CSV format (per the LSVQ release):

* The canonical LSVQ split CSVs distributed by the authors and
  re-hosted on Hugging Face (``teowu/LSVQ-videos`` [#hf]_) carry
  one row per clip with columns roughly of the form::

    name,height,width,mos,frame_number,fps[,is_test,is_valid]

  ``name`` is the stem (no extension); the ``.mp4`` extension and
  optional ``--clips-subdir`` path are joined at runtime.
* Header columns vary slightly between the original Patch-VQ
  drop, the Hugging Face mirror, and the DOVER / FAST-VQA
  redistributions. Recognised spellings:

  - filename: ``name`` / ``video_name`` / ``filename`` / ``file_name``
  - URL:      ``url`` / ``download_url`` / ``video_url`` (rare —
              most LSVQ shards distribute the clips alongside the
              CSV, so a missing URL is normal)
  - MOS:      ``mos`` / ``MOS`` / ``mos_score``
  - SD:       ``sd`` / ``SD`` / ``mos_std`` / ``mos_std_dev``
  - n:        ``n`` / ``ratings`` / ``num_ratings`` / ``n_ratings``

* Add new aliases via the ``_CSV_*_KEYS`` tuples at the top of this
  module.

Refusal: if the CSV has fewer than :data:`_LSVQ_MIN_ROWS` rows
(currently 1000) the script refuses with a hint that the operator
should pass ``--full`` for whole-corpus runs and otherwise rely
on ``--max-rows`` for laptop-class subsets. The lower floor (1000
vs KonViD-150k's 5000) reflects LSVQ's split structure — the
1080p test split alone is ~3 600 rows, so refusing at 5 000 would
reject canonical leaderboard manifests.

Partial-corpus runs: LSVQ raw is ~500 GB end-to-end. The script
defaults to ``--max-rows=500`` for laptop-class development; pass
``--full`` to ingest the entire manifest.

License: LSVQ is CC-BY-4.0 [#hf]_. The script does **not** ship
any clip, MOS value, or derived feature in tree (per ADR-0367).
Only the adapter, the schema, and this docstring land in the
repo; CC-BY attribution travels with any redistributed derived
artifact.

.. [#hf] Hugging Face dataset card: ``teowu/LSVQ-videos`` —
   https://huggingface.co/datasets/teowu/LSVQ-videos. Mirrored
   from Patch-VQ author drop at
   https://github.com/baidut/PatchVQ. License field: CC-BY-4.0.
   Verified 2026-05-08.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("lsvq_to_corpus_jsonl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lower-bound on row count below which the script assumes the
#: operator handed it a tiny-fragment CSV by mistake. The LSVQ
#: 1080p test split alone is ~3 600 rows; whole-train is ~28 056.
#: 1 000 is the smallest plausible useful subset and well below
#: every published canonical split.
_LSVQ_MIN_ROWS: int = 1000

#: Default ``--max-rows`` cap for laptop-class development. LSVQ
#: raw is ~500 GB end-to-end — full ingestion is opt-in via
#: ``--full``. 500 rows ≈ 5–10 GB working set.
_LSVQ_DEFAULT_MAX_ROWS: int = 500

#: SHA-256 chunk size — matches the existing manifest_scan reader.
_SHA_CHUNK_BYTES: int = 1 << 20  # 1 MiB

#: Default location of an extracted LSVQ corpus.
_DEFAULT_LSVQ_DIR: Path = Path(__file__).resolve().parents[2] / ".workingdir2" / "lsvq"

#: Default JSONL output path; lives under ``.workingdir2/lsvq/``
#: alongside the source clips so it never accidentally lands in tree.
_DEFAULT_OUTPUT: Path = _DEFAULT_LSVQ_DIR / "lsvq.jsonl"

#: Default download-state JSON. ``Ctrl-C`` + re-run resumes from here.
_DEFAULT_PROGRESS: Path = _DEFAULT_LSVQ_DIR / ".download-progress.json"

#: Default subdirectory for downloaded clip MP4s (under ``lsvq_dir``).
_DEFAULT_CLIPS_SUBDIR: str = "clips"

#: Conventional manifest-CSV filename. Operators may override via
#: ``--manifest-csv``.
_DEFAULT_MANIFEST_NAME: str = "manifest.csv"

#: Default suffix appended to the ``name`` column when the CSV
#: stems lack an extension. LSVQ ships clips as ``.mp4`` per the
#: Patch-VQ author drop.
_DEFAULT_CLIP_SUFFIX: str = ".mp4"

#: CSV column-name aliases — LSVQ ships variants across mirrors.
_CSV_FILENAME_KEYS: tuple[str, ...] = ("name", "video_name", "filename", "file_name")
_CSV_URL_KEYS: tuple[str, ...] = ("url", "download_url", "video_url")
_CSV_MOS_KEYS: tuple[str, ...] = ("mos", "MOS", "mos_score")
_CSV_MOS_STD_KEYS: tuple[str, ...] = ("sd", "SD", "mos_std", "mos_std_dev", "SD_MOS")
_CSV_NRATINGS_KEYS: tuple[str, ...] = ("n", "ratings", "num_ratings", "n_ratings")

#: Default dataset-version string baked into rows when the
#: operator has not pinned one with ``--corpus-version``.
#:
#: > Ying, Z., Mandal, M., Ghadiyaram, D., Bovik, A. C., "Patch-VQ:
#: > 'Patching Up' the Video Quality Problem," CVPR/ICCV 2021.
_DEFAULT_CORPUS_VERSION: str = "lsvq-2021"

#: ``corpus`` field literal — ADR-0367.
_CORPUS_LABEL: str = "lsvq"

#: Per-URL download progress states. ``done`` means the file is on
#: disk and was probed at least once; ``failed`` means the
#: download attempt terminated with a non-retriable failure (HTTP
#: 404 / 410 etc.); ``pending`` is the implicit state for URLs not
#: yet attempted.
_STATE_DONE: str = "done"
_STATE_FAILED: str = "failed"


# ---------------------------------------------------------------------------
# Small helpers (mirrored from KonViD-150k Phase 2)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return current time as ISO-8601 UTC, second-precision."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    """Stream a chunked SHA-256 of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_SHA_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pick(row: dict[str, str], keys: Iterable[str]) -> str | None:
    """Return the first non-empty value at any of ``keys`` (case-insensitive)."""
    lower = {k.lower(): k for k in row}
    for key in keys:
        actual = lower.get(key.lower())
        if actual is None:
            continue
        val = row[actual]
        if val is None:
            continue
        s = str(val).strip()
        if s:
            return s
    return None


def _parse_framerate(rate: str) -> float:
    """Parse ffprobe's ``a/b`` rational rate string."""
    if not rate:
        return 0.0
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            return 0.0
        if den == 0.0:
            return 0.0
        return num / den
    try:
        return float(rate)
    except ValueError:
        return 0.0


def _normalise_clip_name(stem: str, *, suffix: str = _DEFAULT_CLIP_SUFFIX) -> str:
    """Append the default suffix if ``stem`` has none.

    LSVQ split CSVs frequently store bare numeric stems (``"0001"``)
    rather than full ``"0001.mp4"`` filenames; this helper makes
    both shapes accepted.
    """
    if "." in stem:
        return stem
    return stem + suffix


# ---------------------------------------------------------------------------
# Manifest CSV parsing
# ---------------------------------------------------------------------------


class _ManifestRow:
    """One row of the LSVQ manifest CSV."""

    __slots__ = ("filename", "mos", "mos_std_dev", "n_ratings", "url")

    def __init__(
        self,
        filename: str,
        url: str,
        mos: float,
        mos_std_dev: float,
        n_ratings: int,
    ) -> None:
        self.filename = filename
        self.url = url
        self.mos = mos
        self.mos_std_dev = mos_std_dev
        self.n_ratings = n_ratings


def parse_manifest_csv(
    csv_path: Path,
    *,
    min_rows: int = _LSVQ_MIN_ROWS,
    clip_suffix: str = _DEFAULT_CLIP_SUFFIX,
) -> list[_ManifestRow]:
    """Parse an LSVQ split CSV.

    Returns one :class:`_ManifestRow` per data line. Refuses if the
    row count is below :data:`_LSVQ_MIN_ROWS`. The ``min_rows``
    override is a test seam; production callers leave it at the
    default.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"LSVQ manifest CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty / headerless CSV")
        all_rows = list(reader)

    if len(all_rows) < min_rows:
        raise ValueError(
            f"{csv_path}: {len(all_rows)} rows is below the LSVQ "
            f"sanity floor ({min_rows}). Canonical LSVQ splits "
            f"start at ~3 600 rows (LSVQ_test_1080p) and run up "
            f"to ~28 056 (LSVQ_whole_train). For laptop-class "
            f"ingestion runs use --max-rows on a full split "
            f"rather than passing a hand-trimmed CSV."
        )

    parsed: list[_ManifestRow] = []
    for line_no, row in enumerate(all_rows, start=2):  # 1 = header
        stem = _pick(row, _CSV_FILENAME_KEYS)
        if not stem:
            _LOG.warning(
                "%s:%d: no filename column found (looked for %s); skipping",
                csv_path,
                line_no,
                ", ".join(_CSV_FILENAME_KEYS),
            )
            continue
        filename = _normalise_clip_name(stem, suffix=clip_suffix)

        url = _pick(row, _CSV_URL_KEYS) or ""
        # URL is optional for already-downloaded LSVQ shards. The
        # canonical dataset distribution is one big tarball, so
        # most operators never exercise the download path.
        if not url:
            _LOG.debug(
                "%s:%d: no URL column for %s; clip must already exist on disk",
                csv_path,
                line_no,
                filename,
            )

        mos_str = _pick(row, _CSV_MOS_KEYS)
        if mos_str is None:
            _LOG.warning(
                "%s:%d: no MOS column for %s; skipping",
                csv_path,
                line_no,
                filename,
            )
            continue
        try:
            mos = float(mos_str)
        except ValueError:
            _LOG.warning("%s:%d: bad MOS value %r; skipping", csv_path, line_no, mos_str)
            continue

        std_str = _pick(row, _CSV_MOS_STD_KEYS)
        try:
            mos_std_dev = float(std_str) if std_str is not None else 0.0
        except ValueError:
            mos_std_dev = 0.0

        n_str = _pick(row, _CSV_NRATINGS_KEYS)
        try:
            n_ratings = int(float(n_str)) if n_str is not None else 0
        except ValueError:
            n_ratings = 0

        parsed.append(
            _ManifestRow(
                filename=filename,
                url=url,
                mos=mos,
                mos_std_dev=mos_std_dev,
                n_ratings=n_ratings,
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# Resumable-download progress state
# ---------------------------------------------------------------------------


def load_progress(progress_path: Path) -> dict[str, dict[str, Any]]:
    """Load the resumable-download progress JSON.

    Returns ``{filename: {"state": "done"|"failed", "reason": str, ...}}``.
    Missing or unreadable files yield an empty dict — re-runs treat
    every URL as ``pending``.
    """
    if not progress_path.is_file():
        return {}
    try:
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("progress file %s unreadable (%s); starting fresh", progress_path, exc)
        return {}
    if not isinstance(raw, dict):
        _LOG.warning("progress file %s has wrong shape; starting fresh", progress_path)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, val in raw.items():
        if isinstance(key, str) and isinstance(val, dict):
            out[key] = val
    return out


def save_progress(progress_path: Path, state: dict[str, dict[str, Any]]) -> None:
    """Atomically write the progress JSON.

    Atomic via tempfile + rename so an interrupted write never
    leaves a half-truncated file the next run would discard.
    """
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=".download-progress.", suffix=".tmp", dir=str(progress_path.parent)
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, sort_keys=True, indent=2)
            fh.write("\n")
        os.replace(tmp_path, progress_path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def mark_done(state: dict[str, dict[str, Any]], filename: str) -> None:
    """Mark ``filename`` as successfully downloaded."""
    state[filename] = {"state": _STATE_DONE}


def mark_failed(state: dict[str, dict[str, Any]], filename: str, reason: str) -> None:
    """Mark ``filename`` as a non-retriable download failure."""
    state[filename] = {"state": _STATE_FAILED, "reason": reason}


def should_attempt(state: dict[str, dict[str, Any]], filename: str, clip_path: Path) -> bool:
    """Return True if we should (re-)attempt this clip.

    - ``pending`` (no entry): yes.
    - ``done`` and clip is on disk: no.
    - ``done`` but clip is missing on disk: yes (re-fetch).
    - ``failed``: no — non-retriable per the resume contract;
      rerun with the progress file deleted to retry.
    """
    entry = state.get(filename)
    if entry is None:
        return True
    s = entry.get("state")
    if s == _STATE_FAILED:
        return False
    if s == _STATE_DONE:
        return not clip_path.is_file()
    return True


# ---------------------------------------------------------------------------
# Downloader (subprocess seam)
# ---------------------------------------------------------------------------


def download_clip(
    *,
    url: str,
    dest: Path,
    curl_bin: str = "curl",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout_s: int = 120,
) -> tuple[bool, str]:
    """Download ``url`` to ``dest`` via curl.

    Returns ``(ok, reason)`` where ``reason`` is a short diagnostic
    string on failure (HTTP code, curl exit code, etc.) or empty
    on success. The ``runner`` argument is the test seam.
    """
    if not url:
        return False, "no-url-in-manifest"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        curl_bin,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_s),
        "--output",
        str(part),
        url,
    ]
    try:
        proc = runner(cmd, check=False, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as exc:
        return False, f"curl-spawn-failed: {exc}"

    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        with contextlib.suppress(OSError):
            part.unlink()
        stderr = (getattr(proc, "stderr", "") or "").strip()
        return False, f"curl-rc={rc}: {stderr[:200]}"

    if not part.is_file() or part.stat().st_size == 0:
        with contextlib.suppress(OSError):
            part.unlink()
        return False, "curl-empty-output"

    try:
        os.replace(part, dest)
    except OSError as exc:
        return False, f"rename-failed: {exc}"
    return True, ""


# ---------------------------------------------------------------------------
# ffprobe (mirrors KonViD-150k Phase 2)
# ---------------------------------------------------------------------------


def probe_geometry(
    clip_path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any] | None:
    """Probe ``(width, height, fps, duration_s, pix_fmt, codec_name)``.

    Returns ``None`` on any failure. The caller logs and skips the
    clip — the run continues. The ``runner`` argument is the test
    seam.
    """
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,pix_fmt,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(clip_path),
    ]
    try:
        proc = runner(cmd, check=False, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as exc:
        _LOG.warning("ffprobe spawn failed for %s: %s", clip_path.name, exc)
        return None

    rc = getattr(proc, "returncode", 1)
    stdout = getattr(proc, "stdout", "") or ""
    if rc != 0:
        _LOG.warning(
            "ffprobe rc=%d for %s; stderr=%s",
            rc,
            clip_path.name,
            (getattr(proc, "stderr", "") or "").strip()[:200],
        )
        return None

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        _LOG.warning("ffprobe non-JSON output for %s: %s", clip_path.name, exc)
        return None

    streams = payload.get("streams") or []
    if not streams:
        _LOG.warning("ffprobe: no video streams in %s", clip_path.name)
        return None
    s = streams[0]

    width = int(s.get("width", 0) or 0)
    height = int(s.get("height", 0) or 0)
    fps_raw = s.get("avg_frame_rate") or s.get("r_frame_rate") or ""
    framerate = _parse_framerate(fps_raw)

    duration_s = 0.0
    for src in (s, payload.get("format") or {}):
        d = src.get("duration")
        if d is None:
            continue
        try:
            duration_s = float(d)
        except (TypeError, ValueError):
            continue
        if duration_s > 0:
            break

    return {
        "width": width,
        "height": height,
        "framerate": framerate,
        "duration_s": duration_s,
        "pix_fmt": str(s.get("pix_fmt") or ""),
        "encoder_upstream": str(s.get("codec_name") or ""),
    }


# ---------------------------------------------------------------------------
# JSONL row build + write
# ---------------------------------------------------------------------------


def build_row(
    *,
    clip_path: Path,
    csv_row: _ManifestRow,
    geometry: dict[str, Any],
    corpus_version: str,
    ingested_at_utc: str,
    src_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one JSONL row from probed geometry + parsed manifest row.

    The MOS scaling convention matches the KonViD-150k Phase 2
    adapter: the manifest's per-clip ``mos`` value is recorded
    verbatim on a 1.0–5.0 scale. LSVQ MOS values are already
    distributed on the same five-point Likert scale (Ying et al.
    2021 §3.2), so no rescaling is applied at ingestion time. The
    ``mos_std_dev`` and ``n_ratings`` columns are likewise
    pass-through; the trainer-side data loader is responsible for
    any further per-corpus normalisation.
    """
    if src_sha256 is None:
        src_sha256 = _sha256_file(clip_path)
    return {
        "src": clip_path.name,
        "src_sha256": src_sha256,
        "src_size_bytes": int(clip_path.stat().st_size),
        "width": int(geometry["width"]),
        "height": int(geometry["height"]),
        "framerate": float(geometry["framerate"]),
        "duration_s": float(geometry["duration_s"]),
        "pix_fmt": geometry["pix_fmt"],
        "encoder_upstream": geometry["encoder_upstream"],
        "mos": float(csv_row.mos),
        "mos_std_dev": float(csv_row.mos_std_dev),
        "n_ratings": int(csv_row.n_ratings),
        "corpus": _CORPUS_LABEL,
        "corpus_version": corpus_version,
        "ingested_at_utc": ingested_at_utc,
    }


def _read_existing_sha_index(jsonl_path: Path) -> set[str]:
    """Return ``src_sha256`` values already present in an existing JSONL."""
    if not jsonl_path.is_file():
        return set()
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = obj.get("src_sha256")
            if isinstance(sha, str) and sha:
                seen.add(sha)
    return seen


# ---------------------------------------------------------------------------
# Counter (small dataclass-ish for the run() return)
# ---------------------------------------------------------------------------


class RunStats:
    """Aggregate counters returned from :func:`run`."""

    __slots__ = ("attrition_pct", "dedups", "skipped_broken", "skipped_download", "written")

    def __init__(self) -> None:
        self.written = 0
        self.skipped_download = 0
        self.skipped_broken = 0
        self.dedups = 0
        self.attrition_pct = 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Compact representation, written / dl-failed / broken / dedup."""
        return (self.written, self.skipped_download, self.skipped_broken, self.dedups)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    lsvq_dir: Path,
    output: Path,
    manifest_csv: Path | None = None,
    progress_path: Path | None = None,
    clips_subdir: str = _DEFAULT_CLIPS_SUBDIR,
    ffprobe_bin: str = "ffprobe",
    curl_bin: str = "curl",
    corpus_version: str = _DEFAULT_CORPUS_VERSION,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    now_fn: Callable[[], str] = _utc_now_iso,
    attrition_warn_threshold: float = 0.10,
    download_timeout_s: int = 120,
    min_csv_rows: int = _LSVQ_MIN_ROWS,
    max_rows: int | None = _LSVQ_DEFAULT_MAX_ROWS,
    clip_suffix: str = _DEFAULT_CLIP_SUFFIX,
) -> RunStats:
    """Build the JSONL.

    Returns a :class:`RunStats` with ``written / skipped_download /
    skipped_broken / dedups`` counters and an ``attrition_pct``
    fraction. Idempotent: re-runs against an existing ``output``
    append only new rows (keyed by ``src_sha256``), never rewrite
    existing ones. Re-runs against an existing progress file
    resume download attempts (skipping clips already marked
    ``done`` / ``failed``).

    A WARNING is logged when the download-failure rate exceeds
    ``attrition_warn_threshold`` (default 10 %). The run still
    completes; the threshold is advisory.

    ``max_rows`` caps the number of manifest rows ingested in one
    run. ``None`` means "ingest the whole CSV". The default is
    :data:`_LSVQ_DEFAULT_MAX_ROWS` (500) for laptop-class runs;
    pass ``--full`` on the CLI (or ``max_rows=None`` here) to opt
    into whole-corpus ingestion.
    """
    if not lsvq_dir.is_dir():
        raise FileNotFoundError(
            f"LSVQ directory not found: {lsvq_dir}\n"
            f"  Create it and drop the manifest CSV from\n"
            f"  https://huggingface.co/datasets/teowu/LSVQ-videos\n"
            f"  inside (or use --lsvq-dir to point at an existing\n"
            f"  extraction). Default layout:\n"
            f"    .workingdir2/lsvq/manifest.csv\n"
            f"    .workingdir2/lsvq/clips/\n"
        )

    if manifest_csv is None:
        manifest_csv = lsvq_dir / _DEFAULT_MANIFEST_NAME
    if progress_path is None:
        progress_path = lsvq_dir / ".download-progress.json"

    manifest_rows = parse_manifest_csv(manifest_csv, min_rows=min_csv_rows, clip_suffix=clip_suffix)
    _LOG.info("parsed %d manifest rows from %s", len(manifest_rows), manifest_csv.name)

    if max_rows is not None and len(manifest_rows) > max_rows:
        _LOG.info(
            "capping manifest at --max-rows=%d (full CSV had %d); pass --full to ingest all",
            max_rows,
            len(manifest_rows),
        )
        manifest_rows = manifest_rows[:max_rows]

    clips_dir = lsvq_dir / clips_subdir
    clips_dir.mkdir(parents=True, exist_ok=True)

    state = load_progress(progress_path)
    if state:
        already_done = sum(1 for v in state.values() if v.get("state") == _STATE_DONE)
        already_failed = sum(1 for v in state.values() if v.get("state") == _STATE_FAILED)
        _LOG.info(
            "resume: %d clips done, %d clips failed (from %s)",
            already_done,
            already_failed,
            progress_path,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    seen_sha = _read_existing_sha_index(output)
    if seen_sha:
        _LOG.info("resume: %d existing rows already in %s", len(seen_sha), output)

    ingested_at_utc = now_fn()
    stats = RunStats()
    total = len(manifest_rows)
    saves_since_flush = 0

    with output.open("a", encoding="utf-8") as fp:
        for idx, csv_row in enumerate(manifest_rows, start=1):
            clip_path = clips_dir / csv_row.filename

            # Step 1: ensure the clip is on disk (download if needed).
            if not clip_path.is_file():
                if not should_attempt(state, csv_row.filename, clip_path):
                    stats.skipped_download += 1
                    continue
                ok, reason = download_clip(
                    url=csv_row.url,
                    dest=clip_path,
                    curl_bin=curl_bin,
                    runner=runner,
                    timeout_s=download_timeout_s,
                )
                if not ok:
                    _LOG.warning(
                        "download failed for %s: %s",
                        csv_row.filename,
                        reason,
                    )
                    mark_failed(state, csv_row.filename, reason)
                    stats.skipped_download += 1
                    saves_since_flush += 1
                    if saves_since_flush >= 50:
                        save_progress(progress_path, state)
                        saves_since_flush = 0
                    continue
                mark_done(state, csv_row.filename)
                saves_since_flush += 1
            else:
                if state.get(csv_row.filename, {}).get("state") != _STATE_DONE:
                    mark_done(state, csv_row.filename)
                    saves_since_flush += 1

            # Step 2: probe + JSONL row.
            geometry = probe_geometry(clip_path, ffprobe_bin=ffprobe_bin, runner=runner)
            if geometry is None:
                stats.skipped_broken += 1
                continue
            if geometry["width"] <= 0 or geometry["height"] <= 0:
                _LOG.warning(
                    "ffprobe returned zero geometry for %s; skipping",
                    clip_path.name,
                )
                stats.skipped_broken += 1
                continue

            sha = _sha256_file(clip_path)
            if sha in seen_sha:
                stats.dedups += 1
                continue

            row = build_row(
                clip_path=clip_path,
                csv_row=csv_row,
                geometry=geometry,
                corpus_version=corpus_version,
                ingested_at_utc=ingested_at_utc,
                src_sha256=sha,
            )
            fp.write(json.dumps(row, sort_keys=True) + "\n")
            seen_sha.add(sha)
            stats.written += 1

            if saves_since_flush >= 50:
                save_progress(progress_path, state)
                saves_since_flush = 0

            if idx % 1000 == 0:
                _LOG.info(
                    "progress: %d/%d (wrote=%d, dl-failed=%d, broken=%d, dedups=%d)",
                    idx,
                    total,
                    stats.written,
                    stats.skipped_download,
                    stats.skipped_broken,
                    stats.dedups,
                )

    save_progress(progress_path, state)

    if total > 0:
        stats.attrition_pct = stats.skipped_download / total

    summary = (
        f"[lsvq-jsonl] wrote {stats.written} rows, "
        f"skipped {stats.skipped_download} (download-failed), "
        f"{stats.skipped_broken} (broken-clip), "
        f"{stats.dedups} dedups -> {output}"
    )
    print(summary, file=sys.stderr)
    _LOG.info(
        "wrote %d rows, skipped %d (download-failed), %d (broken-clip), %d dedups",
        stats.written,
        stats.skipped_download,
        stats.skipped_broken,
        stats.dedups,
    )

    if stats.attrition_pct > attrition_warn_threshold:
        _LOG.warning(
            "download attrition %.1f%% exceeds advisory threshold %.1f%% "
            "(check %s for failure reasons)",
            stats.attrition_pct * 100.0,
            attrition_warn_threshold * 100.0,
            progress_path,
        )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lsvq_to_corpus_jsonl.py",
        description=(
            "ADR-0367: walk a local LSVQ extraction (or build one "
            "via resumable downloads), probe each clip via "
            "ffprobe, join with the manifest CSV's MOS scores, "
            "and emit one JSONL row per clip. Defaults to a "
            "500-row subset for laptop-class runs; pass --full "
            "for whole-corpus ingestion (~500 GB working set). "
            "Failed-clip state persists across runs in "
            ".download-progress.json so Ctrl-C is safe."
        ),
    )
    ap.add_argument(
        "--lsvq-dir",
        type=Path,
        default=_DEFAULT_LSVQ_DIR,
        help=(
            "Local LSVQ working directory (default: "
            ".workingdir2/lsvq/). Must contain the manifest CSV; "
            "downloads land in <lsvq-dir>/clips/. Obtain the "
            "manifest from "
            "https://huggingface.co/datasets/teowu/LSVQ-videos."
        ),
    )
    ap.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help=("Path to the split CSV (default: <lsvq-dir>/manifest.csv)."),
    )
    ap.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help=(
            "Resumable-download state file (default: "
            "<lsvq-dir>/.download-progress.json). Delete to retry "
            "previously-failed downloads."
        ),
    )
    ap.add_argument(
        "--clips-subdir",
        default=_DEFAULT_CLIPS_SUBDIR,
        help=(
            f"Subdirectory under --lsvq-dir where downloaded "
            f"clips live (default: {_DEFAULT_CLIPS_SUBDIR!r})."
        ),
    )
    ap.add_argument(
        "--clip-suffix",
        default=_DEFAULT_CLIP_SUFFIX,
        help=(
            f"Default file extension appended to bare-stem CSV "
            f"``name`` columns (default: {_DEFAULT_CLIP_SUFFIX!r})."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output JSONL path (default: "
            ".workingdir2/lsvq/lsvq.jsonl). Existing files are "
            "appended to with src_sha256-based dedup."
        ),
    )
    ap.add_argument(
        "--ffprobe-bin",
        default=os.environ.get("FFPROBE_BIN", "ffprobe"),
        help="ffprobe binary (default: $FFPROBE_BIN or 'ffprobe').",
    )
    ap.add_argument(
        "--curl-bin",
        default=os.environ.get("CURL_BIN", "curl"),
        help="curl binary used for downloads (default: $CURL_BIN or 'curl').",
    )
    ap.add_argument(
        "--corpus-version",
        default=_DEFAULT_CORPUS_VERSION,
        help=(
            f"Dataset version string baked into each row (default: {_DEFAULT_CORPUS_VERSION!r})."
        ),
    )
    ap.add_argument(
        "--attrition-warn-threshold",
        type=float,
        default=0.10,
        help=("Fraction of download-failed clips above which a WARNING is logged (default: 0.10)."),
    )
    ap.add_argument(
        "--download-timeout-s",
        type=int,
        default=120,
        help="Per-clip curl --max-time seconds (default: 120).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=_LSVQ_DEFAULT_MAX_ROWS,
        help=(
            f"Cap manifest at this many rows; default is the "
            f"laptop-class subset ({_LSVQ_DEFAULT_MAX_ROWS}). "
            f"Mutually exclusive with --full."
        ),
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help=(
            "Ingest the entire manifest. Overrides --max-rows. "
            "Working set is ~500 GB end-to-end on the canonical "
            "LSVQ_whole_train split."
        ),
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if shutil.which(args.curl_bin) is None:
        _LOG.warning(
            "curl binary %r not on PATH; downloads will fail (this is fine "
            "for already-extracted corpora)",
            args.curl_bin,
        )
    max_rows: int | None = None if args.full else args.max_rows
    try:
        run(
            lsvq_dir=args.lsvq_dir,
            output=args.output,
            manifest_csv=args.manifest_csv,
            progress_path=args.progress_path,
            clips_subdir=args.clips_subdir,
            clip_suffix=args.clip_suffix,
            ffprobe_bin=args.ffprobe_bin,
            curl_bin=args.curl_bin,
            corpus_version=args.corpus_version,
            attrition_warn_threshold=args.attrition_warn_threshold,
            download_timeout_s=args.download_timeout_s,
            max_rows=max_rows,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
