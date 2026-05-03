#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Fetch a small subset of the YouTube UGC dataset for tiny-AI corpus
expansion (T6-x — vmaf_tiny_v5 candidate).

Downloads the N smallest 4-tuple ``(orig, cbr, vod, vodlb)`` stems from
``gs://ugc-dataset/vp9_compressed_videos/``. Each tuple yields three
``(orig, dis)`` ref/dis pairs (orig→cbr, orig→vod, orig→vodlb), so
N stems = 3*N training pairs. Picks the smallest by total compressed
size to keep wall-time + disk reasonable for a corpus-expansion probe.

YouTube UGC license: Creative Commons Attribution (per ATTRIBUTION
file at the bucket root). See docs/ai/training-data.md for the
combined-corpus license register.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

GCS_LIST_URL = "https://storage.googleapis.com/storage/v1/b/ugc-dataset/o"
GCS_OBJ_URL = "https://storage.googleapis.com/ugc-dataset/{name}"
PREFIX = "vp9_compressed_videos/"
SUFFIXES = ("orig", "cbr", "vod", "vodlb")


def _list_bucket(prefix: str) -> list[dict]:
    url = f"{GCS_LIST_URL}?prefix={prefix}&maxResults=5000&fields=items(name,size)"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    return data.get("items", [])


def _group_by_stem(items: list[dict]) -> dict[str, list[tuple[str, int]]]:
    groups: dict[str, list[tuple[str, int]]] = {}
    for it in items:
        n = it["name"].split("/")[-1]
        if not n:
            continue
        # Files look like Gaming_720P-25aa_orig.mp4 / _cbr.webm / _vod.webm / _vodlb.webm
        # Find which suffix prefixes the extension.
        stem = None
        for sfx in SUFFIXES:
            for ext in (".mp4", ".webm"):
                tag = f"_{sfx}{ext}"
                if n.endswith(tag):
                    stem = n[: -len(tag)]
                    break
            if stem is not None:
                break
        if stem is None:
            continue
        groups.setdefault(stem, []).append((n, int(it["size"])))
    # only complete 4-tuples
    return {s: f for s, f in groups.items() if len(f) == 4}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Directory to drop downloaded mp4/webm into."
    )
    ap.add_argument(
        "--n-stems", type=int, default=30, help="Pick the N smallest 4-tuple stems. Default 30."
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output JSON manifest (stem -> {orig,cbr,vod,vodlb}).",
    )
    args = ap.parse_args()

    print(f"[ugc-fetch] listing {PREFIX} ...", flush=True)
    items = _list_bucket(PREFIX)
    print(f"[ugc-fetch] {len(items)} items in bucket", flush=True)
    groups = _group_by_stem(items)
    print(f"[ugc-fetch] {len(groups)} complete 4-tuple stems", flush=True)
    ranked = sorted(groups.items(), key=lambda kv: sum(s for _, s in kv[1]))[: args.n_stems]
    total_mb = sum(sum(s for _, s in files) for _, files in ranked) / 1e6
    print(f"[ugc-fetch] selected {len(ranked)} stems totalling {total_mb:.1f} MB", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for i, (stem, files) in enumerate(ranked, 1):
        print(f"[ugc-fetch] [{i}/{len(ranked)}] stem={stem}", flush=True)
        entry: dict[str, str] = {}
        for fname, _sz in files:
            url = GCS_OBJ_URL.format(name=PREFIX + fname)
            dest = args.out_dir / fname
            _download(url, dest)
            for sfx in SUFFIXES:
                if f"_{sfx}." in fname:
                    entry[sfx] = str(dest)
                    break
        manifest[stem] = entry

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ugc-fetch] wrote {args.manifest} ({len(manifest)} stems)", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
