#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""BVI-DVC → vmaf-tune corpus JSONL adapter (ADR-0310).

Companion to :mod:`ai.scripts.bvi_dvc_to_full_features`. The parquet
script emits a per-frame feature pool consumed by the tiny-AI student
trainers (`vmaf_tiny_v*`, `fr_regressor_v1`). This script takes the
*same* libvmaf JSON cache the parquet stage produced and re-shapes
each `(source, preset, CRF)` triple into the vmaf-tune Phase A corpus
row schema (:data:`vmaftune.CORPUS_ROW_KEYS`) — the input contract
:mod:`ai.scripts.train_fr_regressor_v2` consumes.

Pipeline::

    bvi_dvc_to_full_features.py  (parquet + cached vmaf JSON)
                  │
                  ▼
    bvi_dvc_to_corpus_jsonl.py   (one JSONL row per encode)
                  │
                  ▼
    merge_corpora.py             (concatenate with Netflix shard)

The script is harness-only — it does not invoke ffmpeg or libvmaf.
Heavy GPU / CPU feature extraction stays in
``bvi_dvc_to_full_features.py``; this stage is a pure JSON → JSONL
transform that runs in seconds.

Output: ``runs/bvi_dvc_corpus.jsonl`` (gitignored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VMAFTUNE_SRC = _REPO_ROOT / "tools" / "vmaf-tune" / "src"
if str(_VMAFTUNE_SRC) not in sys.path:
    sys.path.insert(0, str(_VMAFTUNE_SRC))

from vmaftune import CORPUS_ROW_KEYS, SCHEMA_VERSION  # noqa: E402


def _stable_sha(key: str) -> str:
    """Deterministic 64-hex-char surrogate for ``src_sha256``.

    The cached libvmaf JSON does not preserve the source MP4's content
    hash. Hashing the BVI-DVC clip key (e.g.
    ``DBookcaseBVITexture_480x272_120fps_10bit_420``) yields a stable
    identifier for de-duplication across re-runs without forcing a
    re-decode of the source archive.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _row_from_cache(
    cache_path: Path,
    *,
    crf: int,
    preset: str,
    encoder: str,
    pix_fmt: str,
) -> dict:
    """Build one :data:`CORPUS_ROW_KEYS`-shaped row from a cached vmaf JSON."""
    payload = json.loads(cache_path.read_text())
    pooled = payload.get("pooled_metrics", {}).get("vmaf", {})
    vmaf_score = float(pooled.get("mean", payload["frames"][-1]["metrics"]["vmaf"]))
    frames = payload.get("frames", [])

    key = cache_path.stem
    # Filename pattern: e.g. "DBookcaseBVITexture_480x272_120fps_10bit_420".
    parts = key.split("_")
    if len(parts) >= 3 and "x" in parts[1]:
        width, height = (int(x) for x in parts[1].split("x"))
        fps_token = parts[2]  # e.g. "120fps"
        framerate = float(fps_token.replace("fps", "")) if fps_token.endswith("fps") else 0.0
    else:
        width, height, framerate = 0, 0, 0.0

    duration_s = (len(frames) / framerate) if framerate > 0 else 0.0

    row: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "src": f"bvi-dvc:{key}",
        "src_sha256": _stable_sha(key),
        "width": width,
        "height": height,
        "pix_fmt": pix_fmt,
        "framerate": framerate,
        "duration_s": duration_s,
        "encoder": encoder,
        "encoder_version": "",
        "preset": preset,
        "crf": crf,
        "extra_params": "",
        "encode_path": "",
        "encode_size_bytes": 0,
        "bitrate_kbps": 0.0,
        "encode_time_ms": 0,
        "vmaf_score": vmaf_score,
        "vmaf_model": "vmaf_v0.6.1",
        "score_time_ms": 0,
        "ffmpeg_version": "",
        "vmaf_binary_version": "",
        "exit_status": 0,
        # BVI-DVC clips are processed end-to-end; the parquet
        # pipeline does not slice them via the ADR-0297
        # sample-clip mode, so the corpus row is always "full".
        "clip_mode": "full",
    }
    missing = set(CORPUS_ROW_KEYS) - row.keys()
    assert not missing, f"row missing keys {missing}"
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bvi_dvc_to_corpus_jsonl.py")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "vmaf-tiny-ai-bvi-dvc-full",
        help="Directory of cached libvmaf JSON files (one per clip).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "runs" / "bvi_dvc_corpus.jsonl",
    )
    ap.add_argument(
        "--encoder",
        default="libx264",
        help="Encoder label baked into each row's `encoder` field.",
    )
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--crf", type=int, default=35)
    ap.add_argument("--pix-fmt", default="yuv420p10le")
    args = ap.parse_args(argv)

    if not args.cache_dir.is_dir():
        print(f"error: cache dir not found: {args.cache_dir}", file=sys.stderr)
        return 2
    caches = sorted(args.cache_dir.glob("*.json"))
    if not caches:
        print(f"error: no .json files under {args.cache_dir}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.output.open("w", encoding="utf-8") as fp:
        for cache_path in caches:
            row = _row_from_cache(
                cache_path,
                crf=args.crf,
                preset=args.preset,
                encoder=args.encoder,
                pix_fmt=args.pix_fmt,
            )
            fp.write(json.dumps(row, sort_keys=True) + "\n")
            rows += 1
    print(
        f"[bvi-dvc-jsonl] wrote {rows} rows to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
