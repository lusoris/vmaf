#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Extract middle-frame luma planes from KoNViD-1k MP4s for C2/C3 training.

Reads the populated `konvid-1k.yaml` manifest, drives ffmpeg per clip to
dump the middle frame as 224x224 grayscale `.npy` (uint8), and writes:

  * `<root>/_frames_c2/<key>.npy`        — clean middle-frame luma (C2 input)
  * `<root>/_frames_c3_pairs/<key>_clean.npy`  — C3 self-supervised pair clean
  * `<root>/_frames_c3_pairs/<key>_deg.npy`    — gaussian + JPEG-degraded
  * `ai/data/konvid_frames_nr.parquet`   — (key, frame_path, mos) for C2
  * `ai/data/konvid_pairs_filter.parquet` — (key, deg_path, clean_path) for C3

Idempotent: existing .npy files are skipped. Run once after
`ai/scripts/fetch_konvid_1k.py` + `vmaf-train manifest-scan`.

Usage::

    python ai/scripts/extract_konvid_frames.py
    python ai/scripts/extract_konvid_frames.py --root /custom/path \
                                                 --target-hw 224
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ai" / "src" / "vmaf_train" / "data" / "manifests" / "konvid-1k.yaml"
DATA_DIR = REPO_ROOT / "ai" / "data"

C2_PARQUET = DATA_DIR / "konvid_frames_nr.parquet"
C3_PARQUET = DATA_DIR / "konvid_pairs_filter.parquet"


def _default_root() -> Path:
    env = os.environ.get("VMAF_DATA_ROOT")
    base = Path(env) if env else Path.home() / "datasets"
    return base / "konvid-1k"


def _extract_middle_frame_y(mp4: Path, target_hw: int) -> np.ndarray:
    """Use ffmpeg to read 1 luma frame at the clip midpoint, resized to NxN."""
    # Probe duration once so we can seek to the middle.
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dur = float(proc.stdout.strip())
    seek_s = max(0.0, dur / 2.0)

    # Grab a single grayscale frame at target_hw x target_hw via the
    # `gray` pix_fmt (single-channel uint8). Output rawvideo to stdout.
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{seek_s:.3f}",
        "-i",
        str(mp4),
        "-frames:v",
        "1",
        "-vf",
        f"scale={target_hw}:{target_hw}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    rv = subprocess.run(cmd, check=True, capture_output=True)
    arr = np.frombuffer(rv.stdout, dtype=np.uint8)
    expected = target_hw * target_hw
    if arr.size != expected:
        raise RuntimeError(f"{mp4.name}: ffmpeg returned {arr.size} bytes, expected {expected}")
    return arr.reshape(target_hw, target_hw)


def _make_degraded(clean: np.ndarray) -> np.ndarray:
    """Self-supervised C3 pair: gaussian blur σ=1.2 + JPEG quality 35."""
    img = Image.fromarray(clean, mode="L")
    # Gaussian blur via PIL (radius is approx-σ).
    from PIL import ImageFilter

    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    # Round-trip through JPEG quality 35.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=35)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root (default: $VMAF_DATA_ROOT/konvid-1k or ~/datasets/konvid-1k)",
    )
    parser.add_argument(
        "--target-hw",
        type=int,
        default=224,
        help="Square frame size after resize (multiple of 32)",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}; run vmaf-train manifest-scan first")

    with MANIFEST.open() as fh:
        doc = yaml.safe_load(fh)
    entries = doc.get("entries") or []
    if not entries:
        sys.exit("manifest has no entries")

    root = args.root or _default_root()
    if not root.is_dir():
        sys.exit(f"dataset root not found: {root}")

    c2_dir = root / "_frames_c2"
    c3_dir = root / "_frames_c3_pairs"
    c2_dir.mkdir(parents=True, exist_ok=True)
    c3_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    c2_rows = []
    c3_rows = []
    n = len(entries)
    for i, e in enumerate(entries):
        key = e["key"]
        mp4 = root / e["path"]
        if not mp4.is_file():
            print(f"[skip] missing {mp4}")
            continue

        c2_path = c2_dir / f"{key}.npy"
        c3_clean_path = c3_dir / f"{key}_clean.npy"
        c3_deg_path = c3_dir / f"{key}_deg.npy"

        if not c2_path.exists():
            try:
                clean = _extract_middle_frame_y(mp4, args.target_hw)
            except Exception as exc:
                print(f"[error] {mp4.name}: {exc}")
                continue
            np.save(c2_path, clean)
        else:
            clean = np.load(c2_path)

        if not c3_clean_path.exists() or not c3_deg_path.exists():
            np.save(c3_clean_path, clean)
            np.save(c3_deg_path, _make_degraded(clean))

        c2_rows.append({"key": key, "frame_path": str(c2_path), "mos": float(e["mos"])})
        c3_rows.append(
            {
                "key": key,
                "deg_path": str(c3_deg_path),
                "clean_path": str(c3_clean_path),
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[extract] {i + 1}/{n} clips")

    pd.DataFrame(c2_rows).to_parquet(C2_PARQUET, index=False)
    pd.DataFrame(c3_rows).to_parquet(C3_PARQUET, index=False)
    print(f"[done] C2 parquet: {C2_PARQUET} ({len(c2_rows)} rows)")
    print(f"[done] C3 parquet: {C3_PARQUET} ({len(c3_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
