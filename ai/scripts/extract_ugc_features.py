#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Extract canonical-6 features (+ teacher VMAF) over a UGC manifest.

For each (orig, dis) pair in the YouTube UGC vp9 manifest written by
``fetch_youtube_ugc_subset.py``, decode both clips to a common YUV
geometry via ffmpeg, run ``vmaf`` with the canonical-6 feature set
(adm2, vif_scale0..3, motion2) plus the production vmaf_v0.6.1
predictor as the teacher, and append per-frame rows to a parquet
matching the existing 4-corpus schema.

Decode geometry: smallest of (orig, dis) original height, capped at
``--max-height`` (default 360). The cap keeps wall-time + intermediate
YUV size bounded; documented trade-off in the v5 ADR.

Output schema (matches runs/full_features_4corpus.parquet):
    corpus, source, frame_index,
    adm2, adm_scale0..3, cambi, ciede2000, float_ms_ssim, float_ssim,
    motion, motion2, motion3, psnr_cb, psnr_cr, psnr_hvs, psnr_y,
    ssimulacra2, vif_scale0..3, vmaf

Only the canonical-6 columns are populated for UGC (others NaN); v5
training reads only canonical-6 + vmaf so this is sufficient.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

CANONICAL_6 = ("adm2", "vif_scale0", "vif_scale1", "vif_scale2", "vif_scale3", "motion2")
SCHEMA_COLS = (
    "adm2",
    "adm_scale0",
    "adm_scale1",
    "adm_scale2",
    "adm_scale3",
    "cambi",
    "ciede2000",
    "float_ms_ssim",
    "float_ssim",
    "motion",
    "motion2",
    "motion3",
    "psnr_cb",
    "psnr_cr",
    "psnr_hvs",
    "psnr_y",
    "ssimulacra2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "vmaf",
)


def _ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    return json.loads(out)["streams"][0]


def _decode_to_yuv(src: Path, dest: Path, w: int, h: int, max_frames: int) -> int:
    """Decode src video to dest as raw yuv420p 8-bit, scaled to w*h.

    Returns the number of frames written.
    """
    if dest.exists() and dest.stat().st_size > 0:
        # frame count from size
        frame_bytes = w * h * 3 // 2
        return dest.stat().st_size // frame_bytes
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"scale={w}:{h}:flags=bicubic",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.rename(dest)
    frame_bytes = w * h * 3 // 2
    return dest.stat().st_size // frame_bytes


def _run_vmaf(vmaf_bin: Path, ref: Path, dis: Path, w: int, h: int, n_threads: int) -> list[dict]:
    """Run vmaf with canonical-6 features + the v0.6.1 model. Return frames list."""
    out = Path(f"/tmp/ugc_vmaf_{ref.stem}_{dis.stem}.json")
    cmd = [
        str(vmaf_bin),
        "-r",
        str(ref),
        "-d",
        str(dis),
        "-w",
        str(w),
        "-h",
        str(h),
        "-p",
        "420",
        "-b",
        "8",
        "--feature",
        "adm",
        "--feature",
        "vif",
        "--feature",
        "motion",
        "--threads",
        str(n_threads),
        "--output",
        str(out),
        "--json",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with out.open() as f:
        doc = json.load(f)
    out.unlink(missing_ok=True)
    return doc.get("frames", [])


def _frame_row(metrics: dict) -> dict:
    """Translate libvmaf JSON metric names to our parquet schema."""

    def m(key: str, alt: str | None = None) -> float:
        v = metrics.get(key)
        if v is None and alt is not None:
            v = metrics.get(alt)
        return float(v) if v is not None else float("nan")

    return {
        "adm2": m("integer_adm2", "adm2"),
        "adm_scale0": m("integer_adm_scale0", "adm_scale0"),
        "adm_scale1": m("integer_adm_scale1", "adm_scale1"),
        "adm_scale2": m("integer_adm_scale2", "adm_scale2"),
        "adm_scale3": m("integer_adm_scale3", "adm_scale3"),
        "cambi": float("nan"),
        "ciede2000": float("nan"),
        "float_ms_ssim": float("nan"),
        "float_ssim": float("nan"),
        "motion": m("integer_motion", "motion"),
        "motion2": m("integer_motion2", "motion2"),
        "motion3": m("integer_motion3", "motion3"),
        "psnr_cb": m("psnr_cb"),
        "psnr_cr": m("psnr_cr"),
        "psnr_hvs": float("nan"),
        "psnr_y": m("psnr_y"),
        "ssimulacra2": float("nan"),
        "vif_scale0": m("integer_vif_scale0", "vif_scale0"),
        "vif_scale1": m("integer_vif_scale1", "vif_scale1"),
        "vif_scale2": m("integer_vif_scale2", "vif_scale2"),
        "vif_scale3": m("integer_vif_scale3", "vif_scale3"),
        "vmaf": m("vmaf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--yuv-dir",
        type=Path,
        required=True,
        help="Working dir for decoded raw YUVs (deleted after extract).",
    )
    ap.add_argument("--vmaf-bin", type=Path, default=Path("build-cpu/tools/vmaf"))
    ap.add_argument("--out-parquet", type=Path, required=True)
    ap.add_argument(
        "--max-height",
        type=int,
        default=360,
        help="Cap decode height; smaller = faster, less memory.",
    )
    ap.add_argument(
        "--max-frames", type=int, default=300, help="Cap frames per pair; ~10s @ 30fps."
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--keep-yuv", action="store_true")
    args = ap.parse_args()

    if not args.vmaf_bin.is_file():
        print(f"error: vmaf binary not found: {args.vmaf_bin}", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("error: ffmpeg/ffprobe not on PATH", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text())
    print(f"[ugc-extract] manifest stems={len(manifest)}", flush=True)

    rows: list[dict] = []
    pair_count = 0
    fail_count = 0
    t0 = time.monotonic()
    for stem, files in sorted(manifest.items()):
        orig = Path(files["orig"])
        if not orig.is_file():
            print(f"  [{stem}] missing orig, skip", flush=True)
            continue
        try:
            probe = _ffprobe(orig)
        except Exception as exc:  # pragma: no cover
            print(f"  [{stem}] ffprobe failed: {exc}", flush=True)
            fail_count += 1
            continue
        ow = int(probe["width"])
        oh = int(probe["height"])
        # Down-scale to max_height keeping aspect
        target_h = min(oh, args.max_height)
        target_w = (ow * target_h) // oh
        # Make even
        target_w -= target_w & 1
        target_h -= target_h & 1
        ref_yuv = args.yuv_dir / f"{stem}_orig_{target_w}x{target_h}.yuv"
        try:
            _decode_to_yuv(orig, ref_yuv, target_w, target_h, args.max_frames)
        except subprocess.CalledProcessError as exc:
            print(f"  [{stem}] decode-orig failed: {exc}", flush=True)
            fail_count += 1
            continue

        for sfx in ("cbr", "vod", "vodlb"):
            if sfx not in files:
                continue
            dis_src = Path(files[sfx])
            if not dis_src.is_file():
                continue
            dis_yuv = args.yuv_dir / f"{stem}_{sfx}_{target_w}x{target_h}.yuv"
            try:
                _decode_to_yuv(dis_src, dis_yuv, target_w, target_h, args.max_frames)
            except subprocess.CalledProcessError as exc:
                print(f"  [{stem}/{sfx}] decode-dis failed: {exc}", flush=True)
                fail_count += 1
                continue
            try:
                frames = _run_vmaf(
                    args.vmaf_bin, ref_yuv, dis_yuv, target_w, target_h, args.threads
                )
            except subprocess.CalledProcessError as exc:
                print(f"  [{stem}/{sfx}] vmaf failed: {exc}", flush=True)
                fail_count += 1
                if not args.keep_yuv:
                    dis_yuv.unlink(missing_ok=True)
                continue
            source_name = f"ugc-{stem}-{sfx}"
            for frame in frames:
                m = frame.get("metrics", {})
                row = _frame_row(m)
                row["corpus"] = "ugc"
                row["source"] = source_name
                row["frame_index"] = int(frame.get("frameNum", len(rows)))
                rows.append(row)
            pair_count += 1
            print(
                f"  [{stem}/{sfx}] {target_w}x{target_h} frames={len(frames)} "
                f"vmaf~{frames[0].get('metrics',{}).get('vmaf','-') if frames else '-'} "
                f"({time.monotonic()-t0:.0f}s)",
                flush=True,
            )
            if not args.keep_yuv:
                dis_yuv.unlink(missing_ok=True)
        if not args.keep_yuv:
            ref_yuv.unlink(missing_ok=True)

    if not rows:
        print("error: no rows extracted", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    # Reorder to canonical schema
    full_cols = ("corpus", "source", "frame_index", *SCHEMA_COLS)
    for c in full_cols:
        if c not in df.columns:
            df[c] = float("nan")
    df = df[list(full_cols)]
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(
        f"[ugc-extract] wrote {args.out_parquet} pairs={pair_count} fails={fail_count} "
        f"rows={len(df)} sources={df['source'].nunique()} "
        f"wall={time.monotonic()-t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
