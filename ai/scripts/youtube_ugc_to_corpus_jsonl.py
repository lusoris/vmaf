#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""YouTube UGC -> MOS-corpus JSONL adapter (ADR-0368).

The Google YouTube UGC dataset (Wang, Inguva, Adsumilli; MMSP
2019) is the field's canonical large-scale UGC video corpus
underpinning user-generated-content quality work. The official
release is hosted by Google in the public-readable Google Cloud
Storage bucket ``gs://ugc-dataset/`` [#bucket]_; bucket-rooted
``ATTRIBUTION`` lists the Creative-Commons-Attribution licence
travelling with every clip. There is **no request form, no
sign-up, no auth** — the bucket serves over plain anonymous HTTPS
[#public]_.

Why an adapter on top of LSVQ + KonViD: the contributor-pack
research digest #465 flags YouTube UGC as the under-represented
*content-distribution* axis of the fork's `nr_metric_v1` training
mix. KonViD-1k / 150k draws its content from Flickr; LSVQ from
Internet Archive social-video; YouTube UGC adds the genuine
YouTube content distribution (gaming, vlogs, lyric-videos,
HDR clips, animation, ...) the production scoring path cares
about most. The corpus is also unique in that the quality
annotations are paired with **transcoded outputs at multiple
bitrates** rather than only the original — see the
"Per-clip scoring methodology" section below for the pass-through
contract this adapter adopts.

Pipeline shape (mirrors ADR-0333 / LSVQ verbatim modulo paths)::

    .workingdir2/youtube-ugc/
      |- .download-progress.json    # resumable state (this script)
      |- manifest.csv               # split-CSV (operator drops)
      |- clips/                     # downloaded *.mp4 / *.webm
      |       |- Gaming_720P-25aa_orig.mp4
      |       \\- ...
      \\- youtube-ugc.jsonl         # output (this script)

                  |
                  v  ai/scripts/youtube_ugc_to_corpus_jsonl.py
                  |
                  v
    .workingdir2/youtube-ugc/youtube-ugc.jsonl  # one row per clip

Schema (one JSON object per line — byte-identical to LSVQ /
KonViD-150k modulo the ``corpus`` and ``corpus_version``
literals)::

    {
      "src":               "Gaming_720P-25aa_orig.mp4",
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
      "corpus":            "youtube-ugc",
      "corpus_version":    "<dataset version string>",
      "ingested_at_utc":   "<ISO 8601>"
    }

Manifest CSV format (per the YouTube UGC release):

* The canonical Google release distributes a per-clip listing
  CSV at the bucket root [#csv]_ with one row per ``orig`` clip.
  Header columns vary across the original 2019 MMSP release, the
  2020 transcoded-quality follow-up, and the various academic
  redistributions. Recognised spellings (mirrors LSVQ alias set):

  - filename: ``vid`` / ``name`` / ``video_name`` / ``filename`` /
              ``file_name``
  - URL:      ``url`` / ``download_url`` / ``video_url`` —
              optional; when missing, the script reconstructs
              the canonical bucket URL
              ``https://storage.googleapis.com/ugc-dataset/original_videos/<filename>``
              from the row's filename.
  - MOS:      ``mos`` / ``MOS`` / ``mos_score`` / ``dmos``
  - SD:       ``sd`` / ``SD`` / ``mos_std`` / ``mos_std_dev`` /
              ``sd_mos`` / ``SD_MOS``
  - n:        ``n`` / ``ratings`` / ``num_ratings`` / ``n_ratings``

* Add new aliases via the ``_CSV_*_KEYS`` tuples at the top of
  this module.

Per-clip scoring methodology (read this carefully — YouTube UGC
is **not** identical to KonViD / LSVQ):

* The 2019 release [#wang2019]_ provides per-original-clip
  *category-labelled MOS values* derived from a small
  crowdsourced study on the original-video set (1380 of the
  ~1500 originals). These values are on the same 1.0-5.0 Likert
  scale as LSVQ / KonViD.
* The 2020 transcoded follow-up [#wang2020]_ provides additional
  per-bitrate crowd ratings on transcoded outputs at four
  rate points (``cbr`` / ``vod`` / ``vodlb`` and the original
  ``orig``). These are stored separately and are **not** what
  this adapter ingests. Operators who need the transcoded-output
  ratings should pre-aggregate them into a CSV with one row per
  ``orig`` clip whose MOS column is the per-clip mean across the
  four transcoding levels (the field calls this "averaged DMOS",
  pass-through methodology); the adapter then records that mean
  verbatim.
* This adapter records whatever the manifest's MOS column
  contains, without rescaling, on the same 1.0-5.0 contract as
  LSVQ. Documenting the source-of-the-MOS is the operator's
  responsibility (it propagates through the row's
  ``corpus_version`` literal — pin to ``ugc-2019-orig`` for the
  Wang-2019 set or ``ugc-2020-transcoded-mean`` for the
  transcoded-mean variant).

Refusal: if the CSV has fewer than :data:`_UGC_MIN_ROWS` rows
(currently 200) the script refuses with a hint to pass ``--full``
for whole-corpus runs and otherwise rely on ``--max-rows`` for
laptop-class subsets. The lower floor (200 vs LSVQ's 1000)
reflects YouTube UGC's smaller-than-LSVQ catalogue: the public
bucket has ~1500 originals, and the smallest plausible useful
subset is the per-resolution slice (~150-300 rows for 360P /
480P / 720P / 1080P / 4K).

Partial-corpus runs: YouTube UGC raw is ~2 TB end-to-end (~1500
originals at 4K + transcoded ladder). The script defaults to
``--max-rows=300`` for laptop-class development; pass ``--full``
to ingest the entire manifest.

License: YouTube UGC is Creative-Commons-Attribution per the
bucket-root ``ATTRIBUTION`` file [#bucket]_. The script does
**not** ship any clip, MOS value, or derived feature in tree
(per ADR-0368). Only the adapter, the schema, and this docstring
land in the repo; CC-BY attribution travels with any
redistributed derived artifact.

.. [#bucket] Public-readable GCS bucket: ``gs://ugc-dataset/``
   serves over anonymous HTTPS at
   https://storage.googleapis.com/ugc-dataset/<object>. License
   block lives at
   https://storage.googleapis.com/ugc-dataset/ATTRIBUTION.
   Verified 2026-05-08.
.. [#public] No sign-up, no request form, no API key. The bucket
   has the ``allUsers:objectViewer`` IAM role; verifiable with
   ``curl -I https://storage.googleapis.com/ugc-dataset/ATTRIBUTION``.
   Verified 2026-05-08.
.. [#csv] Original-video listing CSV is published alongside the
   bucket as ``original_videos.csv`` /
   ``original_videos_<resolution>.csv``. Operator drops the file
   the run consumes at ``.workingdir2/youtube-ugc/manifest.csv``.
.. [#wang2019] Wang, Y., Inguva, S., Adsumilli, B.,
   "YouTube UGC Dataset for Video Compression Research,"
   IEEE Workshop on Multimedia Signal Processing (MMSP) 2019.
.. [#wang2020] Wang, Y. et al., "Rich features for perceptual
   quality assessment of UGC videos," CVPR 2021 (the YT-UGC
   transcoded-quality follow-up release).
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

_LOG = logging.getLogger("youtube_ugc_to_corpus_jsonl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lower-bound on row count below which the script assumes the
#: operator handed it a tiny-fragment CSV by mistake. The smallest
#: plausible useful subset is the per-resolution slice
#: (~150-300 rows for 360P / 480P / 720P / 1080P / 4K). 200 is the
#: smallest plausible useful subset and well below every published
#: per-resolution split.
_UGC_MIN_ROWS: int = 200

#: Default ``--max-rows`` cap for laptop-class development.
#: YouTube UGC raw is ~2 TB end-to-end across the originals + the
#: transcoded ladder; full ingestion is opt-in via ``--full``.
#: 300 rows is roughly the per-resolution slice size.
_UGC_DEFAULT_MAX_ROWS: int = 300

#: SHA-256 chunk size — matches the existing manifest_scan reader.
_SHA_CHUNK_BYTES: int = 1 << 20  # 1 MiB

#: Default location of an extracted YouTube UGC corpus.
_DEFAULT_UGC_DIR: Path = Path(__file__).resolve().parents[2] / ".workingdir2" / "youtube-ugc"

#: Default JSONL output path; lives under
#: ``.workingdir2/youtube-ugc/`` alongside the source clips so it
#: never accidentally lands in tree.
_DEFAULT_OUTPUT: Path = _DEFAULT_UGC_DIR / "youtube-ugc.jsonl"

#: Default download-state JSON. ``Ctrl-C`` + re-run resumes from here.
_DEFAULT_PROGRESS: Path = _DEFAULT_UGC_DIR / ".download-progress.json"

#: Default subdirectory for downloaded clip files (under ``ugc_dir``).
_DEFAULT_CLIPS_SUBDIR: str = "clips"

#: Conventional manifest-CSV filename. Operators may override via
#: ``--manifest-csv``.
_DEFAULT_MANIFEST_NAME: str = "manifest.csv"

#: Default suffix appended to the ``vid`` / ``name`` column when
#: the CSV stems lack an extension. YouTube UGC originals ship as
#: ``.mp4`` (the ``orig`` track) per the canonical 2019 release;
#: transcoded variants ship as ``.webm``.
_DEFAULT_CLIP_SUFFIX: str = ".mp4"

#: Canonical bucket URL prefix for original clips; used to
#: synthesise download URLs when the manifest CSV does not carry
#: a ``url`` column. The 2019 release stores originals under
#: ``original_videos/`` inside the public bucket.
_DEFAULT_BUCKET_PREFIX: str = "https://storage.googleapis.com/ugc-dataset/original_videos/"

#: CSV column-name aliases — YouTube UGC ships variants across
#: the 2019 / 2020 / academic-redist mirrors.
_CSV_FILENAME_KEYS: tuple[str, ...] = (
    "vid",
    "name",
    "video_name",
    "filename",
    "file_name",
)
_CSV_URL_KEYS: tuple[str, ...] = ("url", "download_url", "video_url")
_CSV_MOS_KEYS: tuple[str, ...] = ("mos", "MOS", "mos_score", "dmos", "DMOS")
_CSV_MOS_STD_KEYS: tuple[str, ...] = (
    "sd",
    "SD",
    "mos_std",
    "mos_std_dev",
    "sd_mos",
    "SD_MOS",
)
_CSV_NRATINGS_KEYS: tuple[str, ...] = ("n", "ratings", "num_ratings", "n_ratings")

#: Default dataset-version string baked into rows when the
#: operator has not pinned one with ``--corpus-version``.
#:
#: > Wang, Y., Inguva, S., Adsumilli, B., "YouTube UGC Dataset
#: > for Video Compression Research," MMSP 2019.
_DEFAULT_CORPUS_VERSION: str = "ugc-2019-orig"

#: ``corpus`` field literal — ADR-0368.
_CORPUS_LABEL: str = "youtube-ugc"

#: Per-URL download progress states. ``done`` means the file is on
#: disk and was probed at least once; ``failed`` means the
#: download attempt terminated with a non-retriable failure (HTTP
#: 404 / 410 etc.); ``pending`` is the implicit state for URLs not
#: yet attempted.
_STATE_DONE: str = "done"
_STATE_FAILED: str = "failed"


# ---------------------------------------------------------------------------
# Small helpers (mirrored from KonViD-150k Phase 2 / LSVQ ADR-0333)
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

    YouTube UGC manifest CSVs commonly store bare stems
    (``"Gaming_720P-25aa_orig"``) rather than full
    ``"Gaming_720P-25aa_orig.mp4"`` filenames; this helper makes
    both shapes accepted.
    """
    if "." in stem:
        return stem
    return stem + suffix


def _synth_bucket_url(filename: str, *, prefix: str = _DEFAULT_BUCKET_PREFIX) -> str:
    """Synthesise a canonical bucket download URL for ``filename``."""
    return f"{prefix}{filename}"


# ---------------------------------------------------------------------------
# Manifest CSV parsing
# ---------------------------------------------------------------------------


class _ManifestRow:
    """One row of the YouTube UGC manifest CSV."""

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
    min_rows: int = _UGC_MIN_ROWS,
    clip_suffix: str = _DEFAULT_CLIP_SUFFIX,
    bucket_prefix: str = _DEFAULT_BUCKET_PREFIX,
) -> list[_ManifestRow]:
    """Parse a YouTube UGC split CSV.

    Returns one :class:`_ManifestRow` per data line. Refuses if
    the row count is below :data:`_UGC_MIN_ROWS`. The ``min_rows``
    override is a test seam; production callers leave it at the
    default.

    When the manifest does not carry an explicit ``url`` column,
    each row's URL is synthesised from the filename + the public
    bucket prefix (default ``original_videos/``).
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"YouTube UGC manifest CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty / headerless CSV")
        all_rows = list(reader)

    if len(all_rows) < min_rows:
        raise ValueError(
            f"{csv_path}: {len(all_rows)} rows is below the YouTube "
            f"UGC sanity floor ({min_rows}). The canonical "
            f"original_videos.csv has ~1500 rows; per-resolution "
            f"slices range from ~150-300 rows. For laptop-class "
            f"ingestion runs use --max-rows on a full split rather "
            f"than passing a hand-trimmed CSV."
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
        if not url:
            # YouTube UGC's original_videos.csv typically lacks a
            # URL column — every clip lives at the canonical
            # bucket path. Synthesise it.
            url = _synth_bucket_url(filename, prefix=bucket_prefix)
            _LOG.debug(
                "%s:%d: synthesised bucket URL for %s",
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
    timeout_s: int = 300,
) -> tuple[bool, str]:
    """Download ``url`` to ``dest`` via curl.

    Returns ``(ok, reason)`` where ``reason`` is a short diagnostic
    string on failure (HTTP code, curl exit code, etc.) or empty
    on success. The ``runner`` argument is the test seam.

    YouTube UGC originals are large (median ~1 GB at 4K), hence
    the higher default timeout vs LSVQ.
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
# ffprobe (mirrors KonViD-150k Phase 2 / LSVQ)
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

    The MOS scaling convention matches the LSVQ / KonViD-150k
    adapters: the manifest's per-clip ``mos`` value is recorded
    verbatim on a 1.0-5.0 scale. YouTube UGC's 2019 original-set
    MOS values are already on the same five-point Likert scale
    (Wang et al. 2019 §III.B), so no rescaling is applied at
    ingestion time. The ``mos_std_dev`` and ``n_ratings`` columns
    are likewise pass-through; the trainer-side data loader is
    responsible for any further per-corpus normalisation.
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
    ugc_dir: Path,
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
    download_timeout_s: int = 300,
    min_csv_rows: int = _UGC_MIN_ROWS,
    max_rows: int | None = _UGC_DEFAULT_MAX_ROWS,
    clip_suffix: str = _DEFAULT_CLIP_SUFFIX,
    bucket_prefix: str = _DEFAULT_BUCKET_PREFIX,
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
    :data:`_UGC_DEFAULT_MAX_ROWS` (300) for laptop-class runs;
    pass ``--full`` on the CLI (or ``max_rows=None`` here) to opt
    into whole-corpus ingestion.
    """
    if not ugc_dir.is_dir():
        raise FileNotFoundError(
            f"YouTube UGC directory not found: {ugc_dir}\n"
            f"  Create it and drop the manifest CSV from\n"
            f"  https://storage.googleapis.com/ugc-dataset/original_videos.csv\n"
            f"  inside (or use --ugc-dir to point at an existing\n"
            f"  extraction). Default layout:\n"
            f"    .workingdir2/youtube-ugc/manifest.csv\n"
            f"    .workingdir2/youtube-ugc/clips/\n"
        )

    if manifest_csv is None:
        manifest_csv = ugc_dir / _DEFAULT_MANIFEST_NAME
    if progress_path is None:
        progress_path = ugc_dir / ".download-progress.json"

    manifest_rows = parse_manifest_csv(
        manifest_csv,
        min_rows=min_csv_rows,
        clip_suffix=clip_suffix,
        bucket_prefix=bucket_prefix,
    )
    _LOG.info("parsed %d manifest rows from %s", len(manifest_rows), manifest_csv.name)

    if max_rows is not None and len(manifest_rows) > max_rows:
        _LOG.info(
            "capping manifest at --max-rows=%d (full CSV had %d); pass --full to ingest all",
            max_rows,
            len(manifest_rows),
        )
        manifest_rows = manifest_rows[:max_rows]

    clips_dir = ugc_dir / clips_subdir
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
        f"[youtube-ugc-jsonl] wrote {stats.written} rows, "
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
        prog="youtube_ugc_to_corpus_jsonl.py",
        description=(
            "ADR-0368: walk a local YouTube UGC extraction (or "
            "build one via resumable downloads from the public "
            "Google Cloud Storage bucket gs://ugc-dataset/), "
            "probe each clip via ffprobe, join with the manifest "
            "CSV's MOS scores, and emit one JSONL row per clip. "
            "Defaults to a 300-row subset for laptop-class runs; "
            "pass --full for whole-corpus ingestion (~2 TB working "
            "set). Failed-clip state persists across runs in "
            ".download-progress.json so Ctrl-C is safe."
        ),
    )
    ap.add_argument(
        "--ugc-dir",
        type=Path,
        default=_DEFAULT_UGC_DIR,
        help=(
            "Local YouTube UGC working directory (default: "
            ".workingdir2/youtube-ugc/). Must contain the manifest "
            "CSV; downloads land in <ugc-dir>/clips/. Obtain the "
            "manifest from "
            "https://storage.googleapis.com/ugc-dataset/original_videos.csv."
        ),
    )
    ap.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help=("Path to the split CSV (default: <ugc-dir>/manifest.csv)."),
    )
    ap.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help=(
            "Resumable-download state file (default: "
            "<ugc-dir>/.download-progress.json). Delete to retry "
            "previously-failed downloads."
        ),
    )
    ap.add_argument(
        "--clips-subdir",
        default=_DEFAULT_CLIPS_SUBDIR,
        help=(
            f"Subdirectory under --ugc-dir where downloaded "
            f"clips live (default: {_DEFAULT_CLIPS_SUBDIR!r})."
        ),
    )
    ap.add_argument(
        "--clip-suffix",
        default=_DEFAULT_CLIP_SUFFIX,
        help=(
            f"Default file extension appended to bare-stem CSV "
            f"``vid``/``name`` columns (default: {_DEFAULT_CLIP_SUFFIX!r})."
        ),
    )
    ap.add_argument(
        "--bucket-prefix",
        default=_DEFAULT_BUCKET_PREFIX,
        help=(
            f"Public bucket URL prefix used to synthesise download "
            f"URLs for manifest rows that lack an explicit ``url`` "
            f"column (default: {_DEFAULT_BUCKET_PREFIX!r})."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output JSONL path (default: "
            ".workingdir2/youtube-ugc/youtube-ugc.jsonl). Existing "
            "files are appended to with src_sha256-based dedup."
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
            f"Dataset version string baked into each row (default: "
            f"{_DEFAULT_CORPUS_VERSION!r}). Pin to "
            f"'ugc-2020-transcoded-mean' when ingesting "
            f"transcoded-output mean MOS values."
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
        default=300,
        help="Per-clip curl --max-time seconds (default: 300; UGC clips are large).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=_UGC_DEFAULT_MAX_ROWS,
        help=(
            f"Cap manifest at this many rows; default is the "
            f"laptop-class subset ({_UGC_DEFAULT_MAX_ROWS}). "
            f"Mutually exclusive with --full."
        ),
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help=(
            "Ingest the entire manifest. Overrides --max-rows. "
            "Working set is ~2 TB end-to-end on the canonical "
            "original_videos.csv."
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
            ugc_dir=args.ugc_dir,
            output=args.output,
            manifest_csv=args.manifest_csv,
            progress_path=args.progress_path,
            clips_subdir=args.clips_subdir,
            clip_suffix=args.clip_suffix,
            bucket_prefix=args.bucket_prefix,
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
