#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Extract FULL_FEATURES (Research-0026) from KoNViD-150k-A using FR-from-NR adapter.

KoNViD-150k-A (K150K-A) is a no-reference corpus: each clip carries a human MOS
label but no reference video.  To run full-reference libvmaf extractors we use the
FR-from-NR adapter pattern (ADR-0346): the same decoded YUV is fed as *both*
reference and distorted.  This makes all difference-based metrics (ciede2000,
psnr_hvs, ADM, VIF, SSIM) measure "identity" — they return null / floor at their
trivial value — while content-sensitive metrics (cambi, motion, vmaf teacher) remain
informative.  The NaN columns are expected and documented in ADR-0362.

Output: ``runs/full_features_k150k.parquet`` (one row per clip, gitignored).

Schema (48 columns):

    clip_name, mos,
    <22 features>_mean, <22 features>_std    (44 feature columns)

Feature columns follow the FEATURE_NAMES tuple order exactly (column-order-locked;
see ai/AGENTS.md §K150K-A corpus extraction invariants before reordering).

Restartability: a ``.done`` checkpoint file (one clip name per line, append-only)
lets interrupted runs resume without re-processing already-extracted clips.  The
parquet is flushed atomically (via a ``.tmp`` sibling) every ``--flush-every`` clips
and once more at the end.

Usage::

    python ai/scripts/extract_k150k_features.py \\
        --clips-dir .workingdir2/konvid-150k/k150ka_extracted \\
        --scores   .workingdir2/konvid-150k/k150ka_scores.csv  \\
        --out      runs/full_features_k150k.parquet

Smoke-test (10 clips)::

    python ai/scripts/extract_k150k_features.py --limit 10

Hardware: RTX 4090 via ``--backend cuda`` on ``build-cpu/tools/vmaf`` (fork build).
The system ``/usr/local/bin/vmaf`` v3.0.0 lacks ssimulacra2 and motion_v2 — it must
NOT be used for this pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Feature / extractor configuration (column-order-locked per ai/AGENTS.md)
# ---------------------------------------------------------------------------

# Extractor names passed via --feature to the vmaf CLI.
EXTRACTOR_NAMES: tuple[str, ...] = (
    "adm",
    "vif",
    "motion",
    "motion_v2",
    "psnr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede",
    "psnr_hvs",
    "ssimulacra2",
)

# Canonical 22-feature output columns (Research-0026).
# WARNING: column order is locked — do not reorder without incrementing the
# parquet schema version and updating ai/AGENTS.md.
FEATURE_NAMES: tuple[str, ...] = (
    "adm2",
    "adm_scale0",
    "adm_scale1",
    "adm_scale2",
    "adm_scale3",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion",
    "motion2",
    "motion3",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede2000",
    "psnr_hvs",
    "ssimulacra2",
    "vmaf",
)

# Map feature names to their JSON key(s) in libvmaf output.  libvmaf may emit
# ``integer_<name>`` for fixed-point kernels; try both in order.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "adm2": ("adm2", "integer_adm2"),
    "adm_scale0": ("adm_scale0", "integer_adm_scale0"),
    "adm_scale1": ("adm_scale1", "integer_adm_scale1"),
    "adm_scale2": ("adm_scale2", "integer_adm_scale2"),
    "adm_scale3": ("adm_scale3", "integer_adm_scale3"),
    "vif_scale0": ("vif_scale0", "integer_vif_scale0"),
    "vif_scale1": ("vif_scale1", "integer_vif_scale1"),
    "vif_scale2": ("vif_scale2", "integer_vif_scale2"),
    "vif_scale3": ("vif_scale3", "integer_vif_scale3"),
    "motion": ("motion", "integer_motion"),
    "motion2": ("motion2", "integer_motion2"),
    "motion3": ("motion3", "integer_motion3"),
    "psnr_y": ("psnr_y", "integer_psnr_y"),
    "psnr_cb": ("psnr_cb", "integer_psnr_cb"),
    "psnr_cr": ("psnr_cr", "integer_psnr_cr"),
    "float_ssim": ("float_ssim",),
    "float_ms_ssim": ("float_ms_ssim",),
    "cambi": ("cambi",),
    "ciede2000": ("ciede2000",),
    "psnr_hvs": ("psnr_hvs",),
    "ssimulacra2": ("ssimulacra2",),
    "vmaf": ("vmaf",),
}

# ---------------------------------------------------------------------------
# YUV decode / geometry helpers
# ---------------------------------------------------------------------------


def _probe_geometry(mp4: Path) -> tuple[int, int, str, str]:
    """Return (width, height, pix_fmt, fps) for the first video stream."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate",
            "-of",
            "json",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = json.loads(proc.stdout)["streams"][0]
    pix_fmt: str = s.get("pix_fmt", "yuv420p")
    # Normalise to libvmaf-safe pixel formats.
    pix_fmt = "yuv420p10le" if "10" in pix_fmt else "yuv420p"
    return int(s["width"]), int(s["height"]), pix_fmt, s.get("r_frame_rate", "25/1")


def _decode_to_yuv(mp4: Path, yuv_path: Path, pix_fmt: str) -> None:
    """Decode ``mp4`` to raw YUV.  Writes atomically via a ``.tmp`` sibling."""
    tmp = yuv_path.with_suffix(".tmp")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-pix_fmt",
                pix_fmt,
                "-f",
                "rawvideo",
                str(tmp),
            ],
            check=True,
        )
        tmp.rename(yuv_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# vmaf invocation
# ---------------------------------------------------------------------------


def _build_vmaf_cmd(
    vmaf_bin: Path,
    yuv_path: Path,
    width: int,
    height: int,
    pix_fmt: str,
    out_json: Path,
    threads: int,
    use_cuda: bool,
) -> list[str]:
    bitdepth = "10" if "10" in pix_fmt else "8"
    feat_args: list[str] = []
    for ex in EXTRACTOR_NAMES:
        feat_args += ["--feature", ex]

    backend_args: list[str] = []
    if use_cuda:
        backend_args += ["--backend", "cuda"]
    else:
        backend_args += ["--no_cuda", "--no_sycl", "--no_vulkan"]

    return [
        str(vmaf_bin),
        "--reference",
        str(yuv_path),
        "--distorted",
        str(yuv_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--pixel_format",
        "420",
        "--bitdepth",
        bitdepth,
        *feat_args,
        "--threads",
        str(threads),
        *backend_args,
        "--output",
        str(out_json),
        "--json",
        "-q",
    ]


def _run_vmaf_json(
    vmaf_bin: Path,
    yuv_path: Path,
    width: int,
    height: int,
    pix_fmt: str,
    out_json: Path,
    threads: int,
    use_cuda: bool,
) -> list[dict]:
    """Run vmaf and return a list of per-frame metric dicts."""
    cmd = _build_vmaf_cmd(vmaf_bin, yuv_path, width, height, pix_fmt, out_json, threads, use_cuda)
    subprocess.run(cmd, check=True, capture_output=True)
    with out_json.open() as f:
        data = json.load(f)
    return [fr["metrics"] for fr in data.get("frames", [])]


# ---------------------------------------------------------------------------
# Metric lookup and per-clip aggregation
# ---------------------------------------------------------------------------


def _lookup_metric(metrics: dict, feature: str) -> float:
    """Return the float value for ``feature`` from a libvmaf metrics dict.

    Tries each alias in order; returns NaN if none match or value is None.
    """
    for alias in _METRIC_ALIASES.get(feature, (feature,)):
        v = metrics.get(alias)
        if v is not None:
            return float(v)
    return float("nan")


def _aggregate_frames(frames: list[dict]) -> dict[str, float]:
    """Return nanmean and nanstd per feature across all frames."""
    if not frames:
        result: dict[str, float] = {}
        for feat in FEATURE_NAMES:
            result[f"{feat}_mean"] = float("nan")
            result[f"{feat}_std"] = float("nan")
        return result

    data: dict[str, list[float]] = {feat: [] for feat in FEATURE_NAMES}
    for m in frames:
        for feat in FEATURE_NAMES:
            data[feat].append(_lookup_metric(m, feat))

    result = {}
    for feat in FEATURE_NAMES:
        arr = np.array(data[feat], dtype=np.float64)
        # Suppress all-NaN warnings — ciede2000 and psnr_hvs are all-NaN
        # for identity pairs (ref == dis, ADR-0362 §Negative consequences).
        # numpy emits RuntimeWarning via warnings, not the FP error machinery,
        # so errstate alone does not suppress it — use warnings.catch_warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[f"{feat}_mean"] = float(np.nanmean(arr))
            result[f"{feat}_std"] = float(np.nanstd(arr))
    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_done_set(done_path: Path) -> set[str]:
    """Load the set of already-processed clip names from the checkpoint file."""
    if not done_path.is_file():
        return set()
    with done_path.open() as f:
        return {line.strip() for line in f if line.strip()}


def _append_done(done_path: Path, clip_name: str) -> None:
    """Append a clip name to the checkpoint file (append-only, one per line)."""
    with done_path.open("a") as f:
        f.write(clip_name + "\n")


# ---------------------------------------------------------------------------
# Parquet flush
# ---------------------------------------------------------------------------


def _flush_parquet(rows: list[dict], out_path: Path) -> None:
    """Merge ``rows`` with any existing parquet and write atomically."""
    new_df = pd.DataFrame(rows)
    if out_path.is_file():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["clip_name"], keep="last")
    else:
        combined = new_df
    tmp = out_path.with_suffix(".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(tmp, index=False)
    tmp.rename(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="extract_k150k_features.py",
        description="Extract FULL_FEATURES from KoNViD-150k-A via FR-from-NR adapter (ADR-0346).",
    )
    ap.add_argument(
        "--clips-dir",
        type=Path,
        default=Path(".workingdir2/konvid-150k/k150ka_extracted"),
        help="Directory containing K150K-A .mp4 clips.",
    )
    ap.add_argument(
        "--scores",
        type=Path,
        default=Path(".workingdir2/konvid-150k/k150ka_scores.csv"),
        help="CSV with columns video_name, video_score (MOS labels).",
    )
    ap.add_argument(
        "--vmaf-bin",
        type=Path,
        default=REPO_ROOT / "build-cpu" / "tools" / "vmaf",
        help="Path to the fork vmaf binary (must support ssimulacra2 and motion_v2).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "full_features_k150k.parquet",
        help="Output parquet path (gitignored).",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="vmaf --threads value.",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Flush parquet every N clips.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N clips (smoke-test mode).",
    )
    ap.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA backend (use CPU).",
    )
    ap.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "k150k_yuv_scratch",
        help="Scratch directory for temporary YUV files.",
    )
    args = ap.parse_args()

    use_cuda = not args.no_cuda

    # Pre-flight checks
    if not args.clips_dir.is_dir():
        print(f"error: clips-dir not found: {args.clips_dir}", file=sys.stderr)
        return 2
    if not args.scores.is_file():
        print(f"error: scores CSV not found: {args.scores}", file=sys.stderr)
        return 2
    if not args.vmaf_bin.is_file():
        print(
            f"error: vmaf binary not found: {args.vmaf_bin}\n"
            "Build with: meson setup build-cpu -Denable_cuda=true && ninja -C build-cpu",
            file=sys.stderr,
        )
        return 2

    # Load MOS labels
    scores_df = pd.read_csv(args.scores)
    scores_df = scores_df.rename(columns={"video_score": "mos"})
    mos_map: dict[str, float] = dict(zip(scores_df["video_name"], scores_df["mos"], strict=True))

    # Enumerate clips
    clips = sorted(args.clips_dir.glob("*.mp4"))
    if args.limit is not None:
        clips = clips[: args.limit]

    # Resume via checkpoint
    done_path = args.out.with_suffix(".done")
    done_set = _load_done_set(done_path)
    pending = [c for c in clips if c.name not in done_set]

    print(
        f"[k150k] total={len(clips)} done={len(done_set)} pending={len(pending)} "
        f"cuda={'yes' if use_cuda else 'no'} out={args.out}",
        flush=True,
    )

    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ok = 0
    fail = 0
    t0 = time.time()

    for i, mp4 in enumerate(pending):
        clip_name = mp4.name
        mos = mos_map.get(clip_name, float("nan"))

        yuv_path = args.scratch_dir / (mp4.stem + ".yuv")
        out_json = args.scratch_dir / (mp4.stem + ".json")

        try:
            width, height, pix_fmt, _fps = _probe_geometry(mp4)
            _decode_to_yuv(mp4, yuv_path, pix_fmt)
            frames = _run_vmaf_json(
                args.vmaf_bin,
                yuv_path,
                width,
                height,
                pix_fmt,
                out_json,
                args.threads,
                use_cuda,
            )
            agg = _aggregate_frames(frames)
            row: dict = {
                "clip_name": clip_name,
                "mos": mos,
                "width": width,
                "height": height,
                **agg,
            }
            rows.append(row)
            _append_done(done_path, clip_name)
            ok += 1
        except Exception as exc:
            print(f"[k150k] FAIL {clip_name}: {exc}", file=sys.stderr, flush=True)
            fail += 1
        finally:
            yuv_path.unlink(missing_ok=True)
            out_json.unlink(missing_ok=True)

        # Periodic progress + flush
        if (i + 1) % args.flush_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (len(pending) - i - 1) / rate / 3600.0 if rate > 0 else float("nan")
            print(
                f"[k150k] {i + 1}/{len(pending)} ok={ok} fail={fail} "
                f"{rate:.2f} clip/s eta={remaining:.1f}h",
                flush=True,
            )
            if rows:
                _flush_parquet(rows, args.out)
                rows = []

    # Final flush
    if rows:
        _flush_parquet(rows, args.out)

    elapsed = time.time() - t0
    print(
        f"[k150k] done. ok={ok} fail={fail} total_time={elapsed:.1f}s out={args.out}",
        flush=True,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
