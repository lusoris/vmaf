#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""KonViD-150k -> MOS-corpus JSONL adapter (Phase 2 of ADR-0325).

ADR-0325 plans a two-phase ingestion of the Konstanz UGC datasets. Phase
1 (KonViD-1k, ~1.2k clips) is shipped by
:mod:`ai.scripts.konvid_1k_to_corpus_jsonl`. This script is **Phase 2**:
the full ~150k-clip KonViD-150k corpus (~120-200 GB working set).

Shared infrastructure: :mod:`ai.src.corpus.base` (ADR-0371).

Pipeline shape::

    .workingdir2/konvid-150k/
      +-- .download-progress.json
      +-- manifest.csv
      +-- clips/
      +-- konvid_150k.jsonl

Differences from Phase 1:

1. Resumable downloads (clips pulled per-URL from YouTube/Vimeo).
2. Attrition tolerance (92-98% hit rate; rest is takedowns).
3. Row-count lower-bound guard (refuses < 5 000 rows to catch wrong CSV).

License: KonViD-150k is research-only (per ADR-0325). This script does
not ship any clip, MOS value, or derived feature in tree; only the adapter
and schema land in the repo.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from corpus.base import CorpusIngestBase, RunStats, normalise_clip_name, pick, utc_now_iso

_LOG = logging.getLogger("konvid_150k_to_corpus_jsonl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KONVID_150K_MIN_ROWS: int = 5000
_DEFAULT_KONVID_DIR: Path = Path(__file__).resolve().parents[2] / ".workingdir2" / "konvid-150k"
_DEFAULT_OUTPUT: Path = _DEFAULT_KONVID_DIR / "konvid_150k.jsonl"
_DEFAULT_CLIPS_SUBDIR: str = "clips"
_DEFAULT_MANIFEST_NAME: str = "manifest.csv"

_CSV_FILENAME_KEYS: tuple[str, ...] = ("file_name", "video_name", "filename", "name")
_CSV_URL_KEYS: tuple[str, ...] = ("url", "download_url", "flickr_url", "video_url")
_CSV_MOS_KEYS: tuple[str, ...] = ("MOS", "mos")
_CSV_MOS_STD_KEYS: tuple[str, ...] = ("SD", "sd", "mos_std", "mos_std_dev", "SD_MOS")
_CSV_NRATINGS_KEYS: tuple[str, ...] = ("n", "ratings", "num_ratings", "n_ratings")

_DEFAULT_CORPUS_VERSION: str = "konvid-150k-2019"
_CORPUS_LABEL: str = "konvid-150k"


# ---------------------------------------------------------------------------
# Manifest CSV
# ---------------------------------------------------------------------------


def parse_manifest_csv(
    csv_path: Path, *, min_rows: int = _KONVID_150K_MIN_ROWS
) -> list[dict[str, Any]]:
    """Parse a KonViD-150k manifest CSV; refuse if row count is below floor."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"KonViD-150k manifest CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty / headerless CSV")
        all_rows = list(reader)

    if len(all_rows) < min_rows:
        raise ValueError(
            f"{csv_path}: {len(all_rows)} rows is below the KonViD-150k "
            f"sanity floor ({min_rows}). This looks like the KonViD-1k CSV. "
            f"Run ai/scripts/konvid_1k_to_corpus_jsonl.py instead."
        )

    parsed: list[dict[str, Any]] = []
    for line_no, row in enumerate(all_rows, start=2):
        filename_str = pick(row, _CSV_FILENAME_KEYS)
        if not filename_str:
            _LOG.warning(
                "%s:%d: no filename column (looked for %s); skipping",
                csv_path,
                line_no,
                ", ".join(_CSV_FILENAME_KEYS),
            )
            continue
        filename = normalise_clip_name(filename_str)
        url = pick(row, _CSV_URL_KEYS) or ""
        mos_str = pick(row, _CSV_MOS_KEYS)
        if mos_str is None:
            _LOG.warning("%s:%d: no MOS column for %s; skipping", csv_path, line_no, filename)
            continue
        try:
            mos = float(mos_str)
        except ValueError:
            _LOG.warning("%s:%d: bad MOS value %r; skipping", csv_path, line_no, mos_str)
            continue
        std_str = pick(row, _CSV_MOS_STD_KEYS)
        try:
            mos_std_dev = float(std_str) if std_str is not None else 0.0
        except ValueError:
            mos_std_dev = 0.0
        n_str = pick(row, _CSV_NRATINGS_KEYS)
        try:
            n_ratings = int(float(n_str)) if n_str is not None else 0
        except ValueError:
            n_ratings = 0
        parsed.append(
            {
                "filename": filename,
                "url": url,
                "mos": mos,
                "mos_std_dev": mos_std_dev,
                "n_ratings": n_ratings,
            }
        )
    return parsed


# ---------------------------------------------------------------------------
# Adapter subclass
# ---------------------------------------------------------------------------


class KonViD150kIngest(CorpusIngestBase):
    """MOS-corpus ingest adapter for KonViD-150k (ADR-0325 Phase 2)."""

    corpus_label = _CORPUS_LABEL

    def __init__(
        self,
        *,
        konvid_dir: Path,
        min_csv_rows: int = _KONVID_150K_MIN_ROWS,
        **kwargs: Any,
    ) -> None:
        super().__init__(corpus_dir=konvid_dir, **kwargs)
        self._min_csv_rows = min_csv_rows

    def iter_source_rows(self, clips_dir: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
        rows = parse_manifest_csv(self.manifest_csv, min_rows=self._min_csv_rows)
        _LOG.info("parsed %d manifest rows from %s", len(rows), self.manifest_csv.name)
        for row in rows:
            yield clips_dir / row["filename"], row


# ---------------------------------------------------------------------------
# Module-level run()
# ---------------------------------------------------------------------------


def run(
    *,
    konvid_dir: Path,
    output: Path,
    manifest_csv: Path | None = None,
    progress_path: Path | None = None,
    clips_subdir: str = _DEFAULT_CLIPS_SUBDIR,
    ffprobe_bin: str = "ffprobe",
    curl_bin: str = "curl",
    corpus_version: str = _DEFAULT_CORPUS_VERSION,
    runner=subprocess.run,
    now_fn=utc_now_iso,
    attrition_warn_threshold: float = 0.10,
    download_timeout_s: int = 120,
    min_csv_rows: int = _KONVID_150K_MIN_ROWS,
) -> RunStats:
    """Build the JSONL. Returns a :class:`RunStats`."""
    ingest = KonViD150kIngest(
        konvid_dir=konvid_dir,
        output=output,
        manifest_csv=manifest_csv,
        progress_path=progress_path,
        clips_subdir=clips_subdir,
        ffprobe_bin=ffprobe_bin,
        curl_bin=curl_bin,
        corpus_version=corpus_version,
        runner=runner,
        now_fn=now_fn,
        attrition_warn_threshold=attrition_warn_threshold,
        download_timeout_s=download_timeout_s,
        max_rows=None,
        min_csv_rows=min_csv_rows,
    )
    return ingest.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="konvid_150k_to_corpus_jsonl.py",
        description=(
            "Phase 2 of ADR-0325: walk a local KonViD-150k extraction "
            "(or build one via resumable downloads), probe each clip via "
            "ffprobe, join with the manifest CSV's MOS scores, and emit "
            "one JSONL row per clip."
        ),
    )
    ap.add_argument(
        "--konvid-dir",
        type=Path,
        default=_DEFAULT_KONVID_DIR,
        help="Local KonViD-150k working directory (default: .workingdir2/konvid-150k/).",
    )
    ap.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Path to the manifest CSV (default: <konvid-dir>/manifest.csv).",
    )
    ap.add_argument(
        "--progress-path", type=Path, default=None, help="Resumable-download state file."
    )
    ap.add_argument(
        "--clips-subdir",
        default=_DEFAULT_CLIPS_SUBDIR,
        help=f"Subdirectory for downloaded clips (default: {_DEFAULT_CLIPS_SUBDIR!r}).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output JSONL path (default: .workingdir2/konvid-150k/konvid_150k.jsonl).",
    )
    ap.add_argument(
        "--ffprobe-bin", default=os.environ.get("FFPROBE_BIN", "ffprobe"), help="ffprobe binary."
    )
    ap.add_argument("--curl-bin", default=os.environ.get("CURL_BIN", "curl"), help="curl binary.")
    ap.add_argument(
        "--corpus-version",
        default=_DEFAULT_CORPUS_VERSION,
        help=f"Dataset version string (default: {_DEFAULT_CORPUS_VERSION!r}).",
    )
    ap.add_argument(
        "--attrition-warn-threshold",
        type=float,
        default=0.10,
        help="Download-failure fraction above which a WARNING is logged.",
    )
    ap.add_argument(
        "--download-timeout-s",
        type=int,
        default=120,
        help="Per-clip curl --max-time seconds (default: 120).",
    )
    ap.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if shutil.which(args.curl_bin) is None:
        _LOG.warning("curl binary %r not on PATH; downloads will fail", args.curl_bin)
    try:
        run(
            konvid_dir=args.konvid_dir,
            output=args.output,
            manifest_csv=args.manifest_csv,
            progress_path=args.progress_path,
            clips_subdir=args.clips_subdir,
            ffprobe_bin=args.ffprobe_bin,
            curl_bin=args.curl_bin,
            corpus_version=args.corpus_version,
            attrition_warn_threshold=args.attrition_warn_threshold,
            download_timeout_s=args.download_timeout_s,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: download KonViD-150k from "
            "https://dl.acm.org/do/10.1145/3474085.3475608/full/ or "
            "https://database.mmsp-kn.de/konvid-150k-vqa-database.html",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
