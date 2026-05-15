#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Materialise full-reference feature rows from local CHUG clips.

CHUG ships one reference row plus six bitrate-ladder rows for each
``chug_content_name``.  ``chug_to_corpus_jsonl.py`` downloads and probes
those clips; this script turns the local JSONL into MOS-head training
rows with real libvmaf features by pairing each distorted clip with its
matching reference clip.

The distorted side is spatially scaled to the reference geometry before
feature extraction.  CHUG's ladder rows intentionally differ in
resolution, while libvmaf's raw-YUV CLI path expects equal geometry.
The output rows remain local-only under ``.workingdir2/chug/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.data.feature_extractor import (
    DEFAULT_FEATURES,
    FULL_FEATURES,
    FeatureExtractionResult,
    aggregate_clip_stats,
    extract_features,
)

DEFAULT_CHUG_DIR = Path(".workingdir2") / "chug"
DEFAULT_INPUT = DEFAULT_CHUG_DIR / "chug.jsonl"
DEFAULT_OUTPUT = DEFAULT_CHUG_DIR / "chug_features.jsonl"
DEFAULT_CLIPS_DIR = DEFAULT_CHUG_DIR / "clips"
DEFAULT_CACHE_DIR = DEFAULT_CHUG_DIR / "feature-cache"
DEFAULT_FEATURE_SET = "canonical"
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "canonical": DEFAULT_FEATURES,
    "full": FULL_FEATURES,
}


@dataclass(frozen=True)
class FeaturePair:
    """One CHUG distorted/reference pair ready for feature extraction."""

    row: dict[str, Any]
    ref_row: dict[str, Any]
    dis_path: Path
    ref_path: Path
    width: int
    height: int


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _truthy_ref_flag(row: dict[str, Any]) -> bool:
    raw = row.get("chug_ref")
    if isinstance(raw, bool):
        return raw
    try:
        return int(raw) == 1
    except (TypeError, ValueError):
        return False


def is_reference_row(row: dict[str, Any]) -> bool:
    """Return true for CHUG reference rows."""
    bitrate = str(row.get("chug_bitrate_label", "")).strip().lower()
    ladder = str(row.get("chug_bitladder", "")).strip().lower()
    return _truthy_ref_flag(row) or bitrate == "ref" or "_ref" in ladder


def _row_clip_path(row: dict[str, Any], clips_dir: Path) -> Path:
    return clips_dir / str(row["src"])


def build_feature_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    clips_dir: Path,
    include_reference_identity: bool = False,
) -> list[FeaturePair]:
    """Pair CHUG distorted rows with the matching reference row.

    Rows without a ``chug_content_name`` or without an available reference
    are skipped.  Reference identity pairs are opt-in because they add a
    large cluster of near-perfect samples that can dominate small smoke
    runs.
    """
    rows_list = list(rows)
    refs: dict[str, dict[str, Any]] = {}
    for row in rows_list:
        content = str(row.get("chug_content_name", ""))
        if content and is_reference_row(row):
            refs[content] = row

    pairs: list[FeaturePair] = []
    for row in rows_list:
        content = str(row.get("chug_content_name", ""))
        if not content:
            continue
        row_is_ref = is_reference_row(row)
        if row_is_ref and not include_reference_identity:
            continue
        ref = refs.get(content)
        if ref is None:
            continue
        width = int(ref.get("width", 0) or ref.get("chug_width_manifest", 0) or 0)
        height = int(ref.get("height", 0) or ref.get("chug_height_manifest", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        pairs.append(
            FeaturePair(
                row=row,
                ref_row=ref,
                dis_path=_row_clip_path(row, clips_dir),
                ref_path=_row_clip_path(ref, clips_dir),
                width=width,
                height=height,
            )
        )
    return pairs


def _decode_to_yuv10(
    *,
    src: Path,
    out: Path,
    width: int,
    height: int,
    ffmpeg_bin: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = f"scale={width}:{height}:flags=bicubic,format=yuv420p10le"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        scale_filter,
        "-pix_fmt",
        "yuv420p10le",
        "-f",
        "rawvideo",
        str(out),
    ]
    runner(cmd, check=True)


def _cache_key(pair: FeaturePair, feature_set: str) -> str:
    dis_sha = str(pair.row.get("src_sha256", pair.dis_path.stem))
    ref_sha = str(pair.ref_row.get("src_sha256", pair.ref_path.stem))
    return f"{feature_set}-{ref_sha[:16]}-{dis_sha[:16]}-{pair.width}x{pair.height}"


def _read_done_keys(output: Path) -> set[str]:
    if not output.is_file():
        return set()
    done: set[str] = set()
    for row in _load_jsonl(output):
        key = row.get("feature_pair_key")
        if isinstance(key, str) and key:
            done.add(key)
    return done


def _extract_pair_features(
    pair: FeaturePair,
    *,
    feature_names: tuple[str, ...],
    feature_set: str,
    cache_dir: Path,
    ffmpeg_bin: str,
    vmaf_bin: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    extractor: Callable[..., FeatureExtractionResult] = extract_features,
) -> FeatureExtractionResult:
    cache_path = cache_dir / f"{_cache_key(pair, feature_set)}.json"
    if cache_path.is_file():
        return FeatureExtractionResult.from_jsonable(json.loads(cache_path.read_text()))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chug-features-") as tmp_s:
        tmp = Path(tmp_s)
        ref_yuv = tmp / "ref.yuv"
        dis_yuv = tmp / "dis.yuv"
        _decode_to_yuv10(
            src=pair.ref_path,
            out=ref_yuv,
            width=pair.width,
            height=pair.height,
            ffmpeg_bin=ffmpeg_bin,
            runner=runner,
        )
        _decode_to_yuv10(
            src=pair.dis_path,
            out=dis_yuv,
            width=pair.width,
            height=pair.height,
            ffmpeg_bin=ffmpeg_bin,
            runner=runner,
        )
        result = extractor(
            ref_yuv,
            dis_yuv,
            pair.width,
            pair.height,
            features=feature_names,
            vmaf_binary=vmaf_bin,
            pix_fmt="420",
            bitdepth=10,
        )

    cache_path.write_text(json.dumps(result.to_jsonable(), sort_keys=True), encoding="utf-8")
    return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _build_output_row(
    pair: FeaturePair,
    result: FeatureExtractionResult,
    *,
    feature_set: str,
) -> dict[str, Any]:
    stats = aggregate_clip_stats(result.per_frame)
    n = len(result.feature_names)
    row = dict(pair.row)
    row.update(
        {
            "feature_pair_key": _cache_key(pair, feature_set),
            "feature_source": "chug-fr-ref-aligned",
            "feature_set": feature_set,
            "feature_names": list(result.feature_names),
            "feature_width": pair.width,
            "feature_height": pair.height,
            "feature_bitdepth": 10,
            "feature_alignment": "distorted_scaled_to_reference",
            "feature_ref_src": pair.ref_row.get("src", ""),
            "feature_ref_sha256": pair.ref_row.get("src_sha256", ""),
            "chug_reference_video_id": pair.ref_row.get("chug_video_id", ""),
            "n_feature_frames": int(result.n_frames),
        }
    )
    stat_names = ("mean", "p10", "p90", "std")
    for stat_idx, stat_name in enumerate(stat_names):
        offset = stat_idx * n
        for feat_idx, feat_name in enumerate(result.feature_names):
            value = _finite_or_none(float(stats[offset + feat_idx]))
            row[f"{feat_name}_{stat_name}"] = value
            if stat_name == "mean":
                row[feat_name] = value
    return row


def run(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    clips_dir: Path,
    cache_dir: Path,
    feature_set: str = DEFAULT_FEATURE_SET,
    max_rows: int | None = None,
    include_reference_identity: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    vmaf_bin: Path = Path("build/tools/vmaf"),
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    extractor: Callable[..., FeatureExtractionResult] = extract_features,
) -> int:
    """Materialise CHUG feature rows and return the number written."""
    try:
        feature_names = FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature set {feature_set!r}; valid: {sorted(FEATURE_SETS)}"
        ) from exc

    rows = _load_jsonl(input_jsonl)
    pairs = build_feature_pairs(
        rows,
        clips_dir=clips_dir,
        include_reference_identity=include_reference_identity,
    )
    if max_rows is not None:
        pairs = pairs[:max_rows]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_keys(output_jsonl)
    written = 0
    with output_jsonl.open("a", encoding="utf-8") as out:
        for idx, pair in enumerate(pairs, start=1):
            pair_key = _cache_key(pair, feature_set)
            if pair_key in done:
                continue
            if not pair.ref_path.is_file() or not pair.dis_path.is_file():
                continue
            result = _extract_pair_features(
                pair,
                feature_names=feature_names,
                feature_set=feature_set,
                cache_dir=cache_dir,
                ffmpeg_bin=ffmpeg_bin,
                vmaf_bin=vmaf_bin,
                runner=runner,
                extractor=extractor,
            )
            out.write(json.dumps(_build_output_row(pair, result, feature_set=feature_set)) + "\n")
            out.flush()
            done.add(pair_key)
            written += 1
            if idx % 100 == 0:
                print(f"[chug-features] {idx}/{len(pairs)} rows considered; wrote {written}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chug_extract_features.py")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET)
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--full", action="store_true", help="Process every available pair.")
    ap.add_argument(
        "--include-reference-identity",
        action="store_true",
        help="Also emit ref==distorted identity rows for CHUG reference clips.",
    )
    ap.add_argument("--ffmpeg-bin", default="ffmpeg")
    ap.add_argument("--vmaf-bin", type=Path, default=Path("build/tools/vmaf"))
    args = ap.parse_args(argv)

    with contextlib.suppress(KeyboardInterrupt):
        written = run(
            input_jsonl=args.input,
            output_jsonl=args.output,
            clips_dir=args.clips_dir,
            cache_dir=args.cache_dir,
            feature_set=args.feature_set,
            max_rows=None if args.full else args.max_rows,
            include_reference_identity=args.include_reference_identity,
            ffmpeg_bin=args.ffmpeg_bin,
            vmaf_bin=args.vmaf_bin,
        )
        print(f"[chug-features] wrote {written} rows to {args.output}")
        return 0
    return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
