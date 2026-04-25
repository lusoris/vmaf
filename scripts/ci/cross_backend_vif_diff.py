#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Cross-backend VIF diff — gates the Vulkan VIF kernel against the CPU
scalar reference. Runs `vmaf` twice on the same (ref, dist) pair: once
with the CPU integer-VIF extractor, once with the Vulkan compute kernel
(`--vulkan_device 0`). Compares per-frame `integer_vif_scale0..3` scores
at places=4 and prints a per-scale verdict.

Default tolerance is places=4 (matches the fork's GPU-vs-CPU snapshot
contract — see `docs/principles.md` and the user's "GPU is NOT bit-exact"
invariant). Empirically the GLSL kernel in
`libvmaf/src/feature/vulkan/shaders/vif.comp` is bit-exact with the
scalar reference because both use deterministic int64 accumulators; the
slack is preserved for forward compatibility (e.g., if Mesa lavapipe
diverges from a real ICD on a future driver).

Exit code 0 on agreement, 1 on a places=4 mismatch, 2 on a binary or
fixture failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

VIF_METRICS = (
    "integer_vif_scale0",
    "integer_vif_scale1",
    "integer_vif_scale2",
    "integer_vif_scale3",
)


def run_vmaf(
    binary: Path,
    ref: Path,
    dist: Path,
    w: int,
    h: int,
    pix_fmt: str,
    bitdepth: int,
    output: Path,
    vulkan_device: int | None,
) -> None:
    cmd = [
        str(binary),
        "--reference",
        str(ref),
        "--distorted",
        str(dist),
        "--width",
        str(w),
        "--height",
        str(h),
        "--pixel_format",
        pix_fmt,
        "--bitdepth",
        str(bitdepth),
        "--feature",
        "vif",
        "--output",
        str(output),
        "--json",
    ]
    if vulkan_device is not None:
        cmd += ["--vulkan_device", str(vulkan_device)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(2)


def load_frames(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["frames"]


def diff(cpu: list[dict], vk: list[dict], places: int) -> int:
    if len(cpu) != len(vk):
        print(f"FAIL: frame count mismatch (cpu={len(cpu)}, vk={len(vk)})")
        return 1

    per_scale_max = dict.fromkeys(VIF_METRICS, 0.0)
    per_scale_mismatch = dict.fromkeys(VIF_METRICS, 0)

    for cf, vf in zip(cpu, vk, strict=True):
        for m in VIF_METRICS:
            c, v = cf["metrics"][m], vf["metrics"][m]
            d = abs(c - v)
            per_scale_max[m] = max(per_scale_max[m], d)
            if round(c, places) != round(v, places):
                per_scale_mismatch[m] += 1

    print(f"VIF cross-backend (CPU vs Vulkan), {len(cpu)} frames, " f"tolerance places={places}")
    print(f"{'metric':<25} {'max_abs_diff':<15} {'places=4 mismatches'}")
    fail = False
    for m in VIF_METRICS:
        verdict = "OK" if per_scale_mismatch[m] == 0 else "FAIL"
        print(
            f"  {m:<25} {per_scale_max[m]:<15.6e} " f"{per_scale_mismatch[m]}/{len(cpu)}  {verdict}"
        )
        if per_scale_mismatch[m] > 0:
            fail = True

    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vmaf-binary", type=Path, required=True, help="path to libvmaf/build/tools/vmaf"
    )
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--distorted", type=Path, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--pixel-format", default="420")
    ap.add_argument("--bitdepth", type=int, default=8)
    ap.add_argument("--vulkan-device", type=int, default=0)
    ap.add_argument("--places", type=int, default=4)
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "vmaf_cross_backend",
    )
    args = ap.parse_args()

    if not args.vmaf_binary.exists():
        print(f"vmaf binary not found: {args.vmaf_binary}")
        return 2
    for p in (args.reference, args.distorted):
        if not p.exists():
            print(f"fixture not found: {p}")
            return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    cpu_json = args.workdir / "cpu_vif.json"
    vk_json = args.workdir / "vk_vif.json"

    print(f"running CPU VIF → {cpu_json}")
    run_vmaf(
        args.vmaf_binary,
        args.reference,
        args.distorted,
        args.width,
        args.height,
        args.pixel_format,
        args.bitdepth,
        cpu_json,
        vulkan_device=None,
    )

    print(f"running Vulkan VIF (device {args.vulkan_device}) → {vk_json}")
    run_vmaf(
        args.vmaf_binary,
        args.reference,
        args.distorted,
        args.width,
        args.height,
        args.pixel_format,
        args.bitdepth,
        vk_json,
        vulkan_device=args.vulkan_device,
    )

    return diff(load_frames(cpu_json), load_frames(vk_json), args.places)


if __name__ == "__main__":
    sys.exit(main())
