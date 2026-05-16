#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Merge two or more vmaf-tune corpus JSONL files into one.

The vmaf-tune corpus schema is the row contract emitted by
``tools/vmaf-tune/src/vmaftune/corpus.py`` (Phase A, ADR-0237). It is
also the input contract consumed by
``ai/scripts/train_fr_regressor_v2.py``. When mixing the Netflix Public
drop (``.workingdir2/netflix/``) with BVI-DVC (ADR-0310), this utility
concatenates row streams, validates each row carries the canonical
:data:`vmaftune.CORPUS_ROW_KEYS`, and de-duplicates by ``src_sha256`` —
the corpus uses a content hash of the source YUV as the natural key
across mirrors and re-encodes.

Usage::

    python ai/scripts/merge_corpora.py \\
        --inputs runs/netflix_corpus.jsonl runs/bvi_dvc_corpus.jsonl \\
        --output runs/fr_v2_train_corpus.jsonl

Exit codes:
  0 — merged successfully (summary on stderr)
  1 — at least one row failed schema validation
  2 — input file missing or unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

# The merge utility lives outside the vmaf-tune package; resolve the
# tools/vmaf-tune source dir at import time so we depend only on the
# canonical key tuple, not on a pip-installed copy.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VMAFTUNE_SRC = _REPO_ROOT / "tools" / "vmaf-tune" / "src"
if str(_VMAFTUNE_SRC) not in sys.path:
    sys.path.insert(0, str(_VMAFTUNE_SRC))

# Resolve aiutils package.
_AI_SRC = _REPO_ROOT / "ai" / "src"
if str(_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_AI_SRC))

from vmaftune import CORPUS_ROW_KEYS  # noqa: E402  (sys.path adjusted above)

from aiutils.jsonl_utils import iter_jsonl  # noqa: E402  (sys.path adjusted above)

_REQUIRED_KEYS: frozenset[str] = frozenset(CORPUS_ROW_KEYS)


def _validate_row(path: Path, line_no: int, row: dict) -> None:
    """Assert ``row`` carries every key in :data:`CORPUS_ROW_KEYS`.

    Hard-fails the run on first violation. The training pipeline cannot
    silently drop schema-bad rows: a missing ``src_sha256`` or
    ``vmaf_score`` would corrupt the dedup pass or the regression
    target.
    """
    if not isinstance(row, dict):
        print(
            f"error: {path}:{line_no}: expected JSON object, got {type(row).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    missing = _REQUIRED_KEYS - row.keys()
    if missing:
        print(
            f"error: {path}:{line_no}: missing required keys: " f"{sorted(missing)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _dedup_key(row: dict) -> tuple[str, str, str | int, str | int]:
    """Compose a per-encode identity tuple.

    A ``src_sha256`` collision alone is not a duplicate — the same
    source can legitimately appear under multiple ``(encoder, preset,
    crf)`` triples. Treat the four-tuple as the natural key.
    """
    return (
        str(row.get("src_sha256", "")),
        str(row.get("encoder", "")),
        row.get("preset", ""),
        row.get("crf", ""),
    )


def merge(inputs: Iterable[Path], output: Path) -> tuple[int, int, int, int]:
    """Stream-merge ``inputs`` into ``output``; return summary counters.

    Returns ``(rows_in, rows_out, duplicates, unique_sources)``.
    """
    seen: set[tuple[str, str, str | int, str | int]] = set()
    unique_sources: set[str] = set()
    rows_in = 0
    rows_out = 0
    duplicates = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out_fp:
        for path in inputs:
            if not path.is_file():
                print(f"error: input not found: {path}", file=sys.stderr)
                raise SystemExit(2)
            for line_no, row in iter_jsonl(path):
                rows_in += 1
                _validate_row(path, line_no, row)
                key = _dedup_key(row)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                src = row.get("src_sha256", "")
                if isinstance(src, str) and src:
                    unique_sources.add(src)
                out_fp.write(json.dumps(row, sort_keys=True) + "\n")
                rows_out += 1
    return rows_in, rows_out, duplicates, len(unique_sources)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merge_corpora.py", description=__doc__)
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Two or more vmaf-tune corpus JSONL files.",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL.",
    )
    args = ap.parse_args(argv)

    if len(args.inputs) < 2:
        print(
            "error: --inputs requires at least 2 paths to be a meaningful merge",
            file=sys.stderr,
        )
        return 2

    rows_in, rows_out, dupes, sources = merge(args.inputs, args.output)
    print(
        f"[merge_corpora] rows_in={rows_in} rows_out={rows_out} "
        f"duplicates={dupes} unique_sources={sources} -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
