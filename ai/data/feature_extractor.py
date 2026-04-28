# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Per-frame feature extraction via the libvmaf CLI.

The extractor mirrors what ``vmaf_v0.6.1`` consumes::

    DEFAULT_FEATURES = ("adm2", "vif_scale0", "vif_scale1",
                        "vif_scale2", "vif_scale3", "motion2")

It calls the local CPU build of ``vmaf`` (default ``build/tools/vmaf``)
in JSON mode, parses the per-frame metrics, and returns a NumPy
``(n_frames, n_features)`` matrix plus a 4-stat aggregate
``(mean, p10, p90, std)`` per feature for clip-level pooling.

The binary path can be overridden via ``VMAF_BIN`` or the explicit
``vmaf_binary`` argument. If the binary does not exist, the extractor
raises :class:`RuntimeError` with the canonical build instructions
rather than emitting a misleading subprocess error.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_FEATURES: tuple[str, ...] = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)

# Mapping from the metric names the JSON output uses to the extractor
# names the CLI's ``--feature`` flag wants. Mirrors ``feature_dump.py``
# but kept private to this module so tests can exercise it cheaply.
_METRIC_TO_EXTRACTOR: dict[str, str] = {
    "adm2": "adm",
    "adm_scale0": "adm",
    "adm_scale1": "adm",
    "adm_scale2": "adm",
    "adm_scale3": "adm",
    "vif_scale0": "vif",
    "vif_scale1": "vif",
    "vif_scale2": "vif",
    "vif_scale3": "vif",
    "motion": "motion",
    "motion2": "motion",
}


@dataclass
class FeatureExtractionResult:
    """Per-clip feature payload produced by :func:`extract_features`."""

    feature_names: tuple[str, ...]
    per_frame: np.ndarray  # shape (n_frames, n_features), float32
    n_frames: int

    def to_jsonable(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "per_frame": self.per_frame.tolist(),
            "n_frames": int(self.n_frames),
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> "FeatureExtractionResult":
        per_frame = np.asarray(payload["per_frame"], dtype=np.float32)
        return cls(
            feature_names=tuple(payload["feature_names"]),
            per_frame=per_frame,
            n_frames=int(payload["n_frames"]),
        )


def default_vmaf_binary() -> Path:
    env = os.environ.get("VMAF_BIN")
    if env:
        return Path(env)
    # Repo-relative default (matches ``meson setup build && ninja -C build``).
    return Path("build") / "tools" / "vmaf"


def _ensure_binary(binary: Path) -> None:
    if not binary.is_file():
        raise RuntimeError(
            "libvmaf CLI not found at "
            f"{binary}. Build it first with `meson setup build "
            "&& ninja -C build`, or set $VMAF_BIN to point at an "
            "existing binary."
        )


def _extractors_for(metrics: tuple[str, ...]) -> list[str]:
    seen: list[str] = []
    for m in metrics:
        ex = _METRIC_TO_EXTRACTOR.get(m, m)
        if ex not in seen:
            seen.append(ex)
    return seen


def _lookup(metrics: dict, name: str):
    """libvmaf may emit ``integer_<name>`` for fixed-point kernels."""
    if name in metrics:
        return metrics[name]
    return metrics.get(f"integer_{name}")


def _run_vmaf_json(
    binary: Path,
    ref: Path,
    dis: Path,
    width: int,
    height: int,
    *,
    pix_fmt: str = "420",
    bitdepth: int = 8,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    extra_args: tuple[str, ...] = (),
) -> dict:
    feat_args: list[str] = []
    for f in _extractors_for(features):
        feat_args += ["--feature", f]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        cmd = [
            str(binary),
            "-r",
            str(ref),
            "-d",
            str(dis),
            "-w",
            str(width),
            "-h",
            str(height),
            "-p",
            pix_fmt,
            "-b",
            str(bitdepth),
            "--json",
            "-o",
            str(out_path),
            *feat_args,
            *extra_args,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)


def extract_features(
    ref: Path,
    dis: Path,
    width: int,
    height: int,
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    vmaf_binary: Path | None = None,
    pix_fmt: str = "420",
    bitdepth: int = 8,
) -> FeatureExtractionResult:
    """Extract per-frame ``features`` from ``(ref, dis)``.

    Returns:
        A :class:`FeatureExtractionResult` whose ``per_frame`` matrix has
        one row per decoded frame and ``len(features)`` columns. Missing
        metrics are reported as NaN — callers should mask or drop those
        frames before training.
    """
    binary = Path(vmaf_binary) if vmaf_binary is not None else default_vmaf_binary()
    _ensure_binary(binary)

    doc = _run_vmaf_json(
        binary,
        ref,
        dis,
        width,
        height,
        pix_fmt=pix_fmt,
        bitdepth=bitdepth,
        features=features,
    )
    rows: list[list[float]] = []
    for frame in doc.get("frames", []):
        fmetrics = frame.get("metrics", {})
        row = []
        for f in features:
            v = _lookup(fmetrics, f)
            row.append(float("nan") if v is None else float(v))
        rows.append(row)
    arr = (
        np.asarray(rows, dtype=np.float32)
        if rows
        else np.zeros((0, len(features)), dtype=np.float32)
    )
    return FeatureExtractionResult(
        feature_names=tuple(features),
        per_frame=arr,
        n_frames=arr.shape[0],
    )


def aggregate_clip_stats(features: np.ndarray) -> np.ndarray:
    """Reduce ``(n_frames, n_features)`` to ``(4 * n_features,)``.

    Returns one row containing ``[mean..., p10..., p90..., std...]`` —
    matches what the existing SVR consumes after temporal pooling.
    NaN frames are ignored via ``np.nanmean`` etc.; if a feature is
    entirely NaN the corresponding stat is also NaN (caller's problem).
    """
    if features.ndim != 2:
        raise ValueError(f"expected 2-D features, got shape {features.shape}")
    if features.shape[0] == 0:
        return np.full((4 * features.shape[1],), np.nan, dtype=np.float32)
    mean = np.nanmean(features, axis=0)
    p10 = np.nanpercentile(features, 10, axis=0)
    p90 = np.nanpercentile(features, 90, axis=0)
    std = np.nanstd(features, axis=0)
    return np.concatenate([mean, p10, p90, std]).astype(np.float32)
