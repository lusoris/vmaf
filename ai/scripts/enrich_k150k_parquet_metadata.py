#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Add CHUG/K150K side metadata to an existing FULL_FEATURES parquet.

This salvages long-running feature-extraction jobs that were started
without ``extract_k150k_features.py --metadata-jsonl``. The feature rows
are matched by ``clip_name`` to the basename of each JSONL row's ``src``
or ``filename`` field, then CHUG content identity, raw MOS, ladder, and
content-level split columns are filled into the parquet.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from extract_k150k_features import DEFAULT_CHUG_SPLIT_SEED, _load_jsonl_metadata


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def enrich_frame(
    frame: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return ``frame`` with metadata columns filled by ``clip_name``."""
    if "clip_name" not in frame.columns:
        raise ValueError("input parquet must contain a clip_name column")

    out = frame.copy()
    metadata_keys = sorted({key for meta in metadata.values() for key in meta})
    for key in metadata_keys:
        if key not in out.columns:
            out[key] = pd.NA

    matched = 0
    updated = 0
    missing = 0
    for idx, clip_name in out["clip_name"].items():
        meta = metadata.get(str(clip_name))
        if not meta:
            missing += 1
            continue
        matched += 1
        for key, value in meta.items():
            old = out.at[idx, key]
            if overwrite or _is_missing(old):
                out.at[idx, key] = value
                updated += 1

    return out, {
        "rows": len(out),
        "metadata_rows": len(metadata),
        "matched_rows": matched,
        "missing_rows": missing,
        "updated_cells": updated,
    }


def write_parquet_atomic(frame: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=out_path.parent,
        delete=False,
    ) as fh:
        tmp = Path(fh.name)
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="enrich_k150k_parquet_metadata.py")
    ap.add_argument(
        "--features-parquet",
        type=Path,
        required=True,
        help="Existing FULL_FEATURES parquet to enrich.",
    )
    ap.add_argument(
        "--metadata-jsonl",
        type=Path,
        required=True,
        help="Corpus JSONL sidecar, e.g. .workingdir2/chug/chug.jsonl.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to in-place rewrite of --features-parquet.",
    )
    ap.add_argument(
        "--split-seed",
        default=DEFAULT_CHUG_SPLIT_SEED,
        help="Seed for deterministic CHUG content-level splits.",
    )
    ap.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Overwrite existing metadata cells instead of filling only missing values.",
    )
    args = ap.parse_args(argv)

    if not args.features_parquet.is_file():
        raise SystemExit(f"error: features parquet not found: {args.features_parquet}")
    if not args.metadata_jsonl.is_file():
        raise SystemExit(f"error: metadata JSONL not found: {args.metadata_jsonl}")

    metadata = _load_jsonl_metadata(args.metadata_jsonl, split_seed=args.split_seed)
    frame = pd.read_parquet(args.features_parquet)
    enriched, stats = enrich_frame(
        frame,
        metadata,
        overwrite=args.overwrite_metadata,
    )
    out_path = args.out or args.features_parquet
    write_parquet_atomic(enriched, out_path)
    print(json.dumps({"out": str(out_path), **stats}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
