#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""KoNViD-1k → MOS-corpus JSONL adapter (Phase 1 of ADR-0325).

ADR-0325 plans a two-phase ingestion of the Konstanz UGC datasets. This
script is **Phase 1**: the small KoNViD-1k predecessor (~1.2k clips,
~5 GB) that validates the conversion pipeline shape before scaling to
KonViD-150k (Phase 2, separate script).

The companion fetcher
:mod:`ai.scripts.fetch_konvid_1k` already exists and downloads the
videos zip + the metadata zip into ``$VMAF_DATA_ROOT/konvid-1k/``. This
adapter does **not** download; it expects an already-extracted local
copy under ``.workingdir2/konvid-1k/`` (the fork's standard
gitignored research-data drop, mirroring ADR-0310 / ADR-0303). If the
operator placed it elsewhere, ``--konvid-dir`` redirects.

Pipeline shape::

    .workingdir2/konvid-1k/
      ├── KoNViD_1k_videos/                 # *.mp4 clips (1200)
      └── KoNViD_1k_metadata/
          └── KoNViD_1k_attributes.csv      # MOS table

                  │
                  ▼  ai/scripts/konvid_1k_to_corpus_jsonl.py
                  │      (this script — ffprobe per clip + CSV join)
                  │
                  ▼
    .workingdir2/konvid-1k/konvid_1k.jsonl  # one row per clip

Schema (one JSON object per line)::

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
      "corpus":            "konvid-1k",
      "corpus_version":    "<dataset version string>",
      "ingested_at_utc":   "<ISO 8601>"
    }

Note: this schema does **not** collide with the existing vmaf-tune
Phase A :data:`CORPUS_ROW_KEYS` — it sits *next to* it. The vmaf-tune
corpus rows carry an algorithmic ``vmaf_score`` per ``(src, encoder,
preset, crf)`` triple; the KonViD shard carries a *human* ``mos`` per
source clip. The two corpora are merged at the trainer level (Phase 3
of ADR-0325), not at the JSONL level.

KoNViD-1k CSV format (per the dataset README, 2017 release):

* Canonical name: ``KoNViD_1k_attributes.csv``
* Header columns include (the exact names vary across the 2017 / 2019
  releases): ``flickr_id``, ``file_name`` *(or* ``video_name`` *)*,
  ``MOS`` *(or* ``mos`` *)*, ``SD`` *(or* ``mos_std`` */* ``SD_MOS`` *)*,
  and a per-clip rating count column (``ratings`` / ``n`` / ``num_ratings``).
* This script tolerates the column-name variations and documents the
  exact expected names in :data:`_CSV_FILENAME_KEYS`,
  :data:`_CSV_MOS_KEYS`, etc. If your CSV uses a column name not in the
  alias lists, edit them in-place and add a comment.

Refusal: if the CSV has more than :data:`_KONVID_1K_MAX_ROWS` rows
(currently 1500) the script refuses with a hint pointing at the
KonViD-150k script (Phase 2; not yet shipped). Silently ingesting a
~150k-row CSV through a 1k-shaped pipeline would mask the wrong
corpus being mounted.

License: KonViD-1k is research-only. The script does **not** ship any
clip, MOS value, or derived feature in tree (per ADR-0325). Only the
adapter, the schema, and this docstring land in the repo.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("konvid_1k_to_corpus_jsonl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard-cap above which the script assumes the operator handed it the
#: 150k attributes CSV by mistake. KonViD-1k is exactly 1200 clips; we
#: pad the threshold to 1500 to absorb minor index-row variations.
_KONVID_1K_MAX_ROWS: int = 1500

#: SHA-256 chunk size — same as the existing
#: :func:`ai.src.vmaf_train.data.manifest_scan._sha256` reader. Keeps the
#: working set small even for the larger Phase 2 clips.
_SHA_CHUNK_BYTES: int = 1 << 20  # 1 MiB

#: Default location of an extracted KonViD-1k corpus on the fork's dev
#: machines. ``.workingdir2/`` is gitignored (see CLAUDE.md §5).
_DEFAULT_KONVID_DIR: Path = Path(__file__).resolve().parents[2] / ".workingdir2" / "konvid-1k"

#: Default JSONL output path; lives under ``.workingdir2/konvid-1k/``
#: alongside the source clips so it never accidentally lands in tree.
_DEFAULT_OUTPUT: Path = _DEFAULT_KONVID_DIR / "konvid_1k.jsonl"

#: Subdirectories conventionally created by ``fetch_konvid_1k.py``.
_VIDEOS_SUBDIRS: tuple[str, ...] = ("KoNViD_1k_videos", "videos")
_METADATA_SUBDIRS: tuple[str, ...] = ("KoNViD_1k_metadata", "metadata")

#: CSV column-name aliases — KonViD ships variants across release years.
_CSV_FILENAME_KEYS: tuple[str, ...] = ("file_name", "video_name", "filename", "name")
_CSV_MOS_KEYS: tuple[str, ...] = ("MOS", "mos")
_CSV_MOS_STD_KEYS: tuple[str, ...] = ("SD", "sd", "mos_std", "mos_std_dev", "SD_MOS")
_CSV_NRATINGS_KEYS: tuple[str, ...] = ("n", "ratings", "num_ratings", "n_ratings")
_CSV_FLICKR_KEYS: tuple[str, ...] = ("flickr_id", "id")

#: Default dataset-version string baked into rows when the operator has
#: not pinned one with ``--corpus-version``. The 2017 QoMEX release is
#: the canonical citation:
#:
#: > Hosu et al., "The Konstanz natural video database (KoNViD-1k),"
#: > QoMEX 2017.
_DEFAULT_CORPUS_VERSION: str = "konvid-1k-2017"

#: ``corpus`` field literal — ADR-0325 Phase 1.
_CORPUS_LABEL: str = "konvid-1k"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return current time as ISO-8601 UTC, second-precision."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    """Stream a chunked SHA-256 of ``path`` (matches ``corpus.py`` shape)."""
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


def _resolve_subdir(root: Path, candidates: Iterable[str]) -> Path | None:
    """First existing ``root/<candidate>`` directory, else ``None``."""
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def _find_attributes_csv(metadata_dir: Path) -> Path | None:
    """Locate the MOS CSV inside ``metadata_dir``; tolerant of layout."""
    candidates: list[Path] = []
    candidates.extend(metadata_dir.glob("*attributes*.csv"))
    candidates.extend(metadata_dir.glob("*mos*.csv"))
    if not candidates:
        # Last-resort: any single CSV in the directory.
        csvs = list(metadata_dir.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        return None
    return sorted(candidates)[0]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


class _ParsedCsvRow:
    """One row of the MOS CSV reduced to the four fields we keep."""

    __slots__ = ("filename", "flickr_id", "mos", "mos_std_dev", "n_ratings")

    def __init__(
        self,
        filename: str,
        mos: float,
        mos_std_dev: float,
        n_ratings: int,
        flickr_id: str,
    ) -> None:
        self.filename = filename
        self.mos = mos
        self.mos_std_dev = mos_std_dev
        self.n_ratings = n_ratings
        self.flickr_id = flickr_id


def parse_mos_csv(csv_path: Path) -> list[_ParsedCsvRow]:
    """Parse a KonViD-1k MOS CSV.

    Returns one :class:`_ParsedCsvRow` per data line. Refuses if the row
    count exceeds :data:`_KONVID_1K_MAX_ROWS` — that's the Phase 2
    150k corpus, which has its own (unwritten) ingestion script.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"KonViD-1k MOS CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty / headerless CSV")
        all_rows = list(reader)

    if len(all_rows) > _KONVID_1K_MAX_ROWS:
        raise ValueError(
            f"{csv_path}: {len(all_rows)} rows exceeds the KonViD-1k "
            f"sanity ceiling ({_KONVID_1K_MAX_ROWS}). This looks like "
            f"the KonViD-150k attributes CSV. Run the Phase 2 ingestion "
            f"script (ai/scripts/konvid_150k_to_corpus_jsonl.py — not "
            f"yet shipped, see ADR-0325 §Phase 2)."
        )

    parsed: list[_ParsedCsvRow] = []
    for line_no, row in enumerate(all_rows, start=2):  # 1 = header
        filename = _pick(row, _CSV_FILENAME_KEYS)
        if not filename:
            _LOG.warning(
                "%s:%d: no filename column found (looked for %s); skipping",
                csv_path,
                line_no,
                ", ".join(_CSV_FILENAME_KEYS),
            )
            continue

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

        flickr_id = _pick(row, _CSV_FLICKR_KEYS) or ""

        parsed.append(
            _ParsedCsvRow(
                filename=filename,
                mos=mos,
                mos_std_dev=mos_std_dev,
                n_ratings=n_ratings,
                flickr_id=flickr_id,
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------


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


def probe_geometry(
    clip_path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any] | None:
    """Probe ``(width, height, fps, duration_s, pix_fmt, codec_name)``.

    Returns ``None`` on any failure (non-zero rc, no stream, parse
    error). The caller logs and skips the clip — the run continues.
    The ``runner`` argument is a test seam; production callers leave
    it as the default :func:`subprocess.run`.
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
        proc = runner(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
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

    # Duration may live on the stream or only on the format object.
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
    csv_row: _ParsedCsvRow,
    geometry: dict[str, Any],
    corpus_version: str,
    ingested_at_utc: str,
    src_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one JSONL row from probed geometry + parsed CSV row."""
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
    """Return ``src_sha256`` values already present in an existing JSONL.

    Used for resume / dedup on re-runs (criterion: spec test
    ``test_existing_jsonl_resumed``). Tolerates malformed lines by
    skipping them — re-run output is append-only, never destructive.
    """
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
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    konvid_dir: Path,
    output: Path,
    ffprobe_bin: str = "ffprobe",
    corpus_version: str = _DEFAULT_CORPUS_VERSION,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    now_fn: Callable[[], str] = _utc_now_iso,
) -> tuple[int, int, int]:
    """Build the JSONL.

    Returns ``(written, skipped_broken, dedups)``. Idempotent: re-runs
    against an existing ``output`` append only new rows (keyed by
    ``src_sha256``), never rewrite existing ones.
    """
    if not konvid_dir.is_dir():
        raise FileNotFoundError(
            f"KonViD-1k directory not found: {konvid_dir}\n"
            f"  Download the dataset from "
            f"https://database.mmsp-kn.de/konvid-1k-database.html and\n"
            f"  extract it under that path (or use --konvid-dir to point\n"
            f"  at an existing extraction). The companion "
            f"ai/scripts/fetch_konvid_1k.py downloader writes there too."
        )

    videos_dir = _resolve_subdir(konvid_dir, _VIDEOS_SUBDIRS) or konvid_dir
    metadata_dir = _resolve_subdir(konvid_dir, _METADATA_SUBDIRS) or konvid_dir

    csv_path = _find_attributes_csv(metadata_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"KonViD-1k MOS CSV not found under {metadata_dir}.\n"
            f"  Expected one of: KoNViD_1k_attributes.csv / *mos*.csv"
        )

    csv_rows = parse_mos_csv(csv_path)
    _LOG.info("parsed %d CSV rows from %s", len(csv_rows), csv_path.name)

    output.parent.mkdir(parents=True, exist_ok=True)
    seen_sha = _read_existing_sha_index(output)
    if seen_sha:
        _LOG.info("resume: %d existing rows already in %s", len(seen_sha), output)

    ingested_at_utc = now_fn()
    written = 0
    skipped_broken = 0
    dedups = 0

    with output.open("a", encoding="utf-8") as fp:
        for csv_row in csv_rows:
            clip_path = videos_dir / csv_row.filename
            if not clip_path.is_file():
                # Try one level deeper in case the videos dir was
                # extracted with an extra wrapper.
                hits = list(videos_dir.rglob(csv_row.filename))
                if not hits:
                    _LOG.warning("clip missing on disk: %s", csv_row.filename)
                    skipped_broken += 1
                    continue
                clip_path = hits[0]

            geometry = probe_geometry(clip_path, ffprobe_bin=ffprobe_bin, runner=runner)
            if geometry is None:
                skipped_broken += 1
                continue
            if geometry["width"] <= 0 or geometry["height"] <= 0:
                _LOG.warning(
                    "ffprobe returned zero geometry for %s; skipping",
                    clip_path.name,
                )
                skipped_broken += 1
                continue

            sha = _sha256_file(clip_path)
            if sha in seen_sha:
                dedups += 1
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
            written += 1

    _LOG.info(
        "wrote %d rows, skipped %d (broken), %d dedups",
        written,
        skipped_broken,
        dedups,
    )
    print(
        f"[konvid-1k-jsonl] wrote {written} rows, skipped "
        f"{skipped_broken} (broken), {dedups} dedups -> {output}",
        file=sys.stderr,
    )
    return written, skipped_broken, dedups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="konvid_1k_to_corpus_jsonl.py",
        description=(
            "Phase 1 of ADR-0325: walk a local KonViD-1k extraction, "
            "probe each clip via ffprobe, join with the attribute CSV's "
            "MOS scores, and emit one JSONL row per clip."
        ),
    )
    ap.add_argument(
        "--konvid-dir",
        type=Path,
        default=_DEFAULT_KONVID_DIR,
        help=(
            "Local KonViD-1k extraction (default: "
            ".workingdir2/konvid-1k/). Must contain the videos and a "
            "metadata CSV. Download from "
            "https://database.mmsp-kn.de/konvid-1k-database.html."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output JSONL path (default: "
            ".workingdir2/konvid-1k/konvid_1k.jsonl). Existing files "
            "are appended to with src_sha256-based dedup."
        ),
    )
    ap.add_argument(
        "--ffprobe-bin",
        default=os.environ.get("FFPROBE_BIN", "ffprobe"),
        help="ffprobe binary (default: $FFPROBE_BIN or 'ffprobe').",
    )
    ap.add_argument(
        "--corpus-version",
        default=_DEFAULT_CORPUS_VERSION,
        help=(
            f"Dataset version string baked into each row (default: {_DEFAULT_CORPUS_VERSION!r})."
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
    try:
        run(
            konvid_dir=args.konvid_dir,
            output=args.output,
            ffprobe_bin=args.ffprobe_bin,
            corpus_version=args.corpus_version,
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
