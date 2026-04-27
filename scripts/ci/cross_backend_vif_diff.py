#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Cross-backend feature diff — gates GPU compute kernels (CUDA, SYCL,
Vulkan) against the CPU scalar reference. Runs `vmaf` twice on the
same (ref, dist) pair: once with the CPU integer extractor, once with
the chosen GPU backend's named twin (e.g. ``adm_cuda`` /
``adm_sycl`` / ``adm_vulkan``). Compares per-frame scores at
``places=4`` and prints a per-metric verdict.

The script's filename is historical (it started life as the
VIF-only Vulkan gate from PR #118 / ADR-0176); the broader scope is
controlled by ``--feature {vif,motion,adm}`` and
``--backend {cuda,sycl,vulkan}``.

Default tolerance is ``places=4`` (matches the fork's GPU-vs-CPU
snapshot contract — see ``docs/principles.md`` and the user's "GPU is
NOT bit-exact" invariant). Empirically the GLSL kernels under
``libvmaf/src/feature/vulkan/shaders/`` are essentially bit-exact
with the scalar reference because both sides use deterministic
``int64`` accumulators. CUDA / SYCL kernels have their own histories.

The gate uses an absolute-tolerance check
(``abs(cpu - gpu) <= 0.5e-places``) rather than Python's
``round(cpu, places) == round(gpu, places)`` because banker rounding
on the IEEE-754 representation flips at the rounding boundary even
when the underlying floats agree to better than the contract — see
PR #120's commit message for the worked example.

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

FEATURE_METRICS: dict[str, tuple[str, ...]] = {
    "vif": (
        "integer_vif_scale0",
        "integer_vif_scale1",
        "integer_vif_scale2",
        "integer_vif_scale3",
    ),
    "motion": (
        "integer_motion",
        "integer_motion2",
    ),
    "adm": (
        "integer_adm2",
        "integer_adm_scale0",
        "integer_adm_scale1",
        "integer_adm_scale2",
        "integer_adm_scale3",
    ),
    # GPU long-tail batch 1 (T7-23 / ADR-0182): luma-only PSNR.
    # The Vulkan extractor only emits psnr_y; chroma is a focused
    # follow-up (the picture_vulkan upload path is luma-only today).
    "psnr": ("psnr_y",),
    # GPU long-tail batch 1d (T7-23 / ADR-0182): float_moment.
    # The CPU extractor is registered as `float_moment`; its GPU twin
    # is `float_moment_vulkan` (etc.). The 4 emitted metrics — 1st and
    # 2nd raw moment of ref + dis luma — match byte-for-byte at JSON
    # precision (int64 sum is exact on integer YUV inputs).
    "float_moment": (
        "float_moment_ref1st",
        "float_moment_dis1st",
        "float_moment_ref2nd",
        "float_moment_dis2nd",
    ),
    # GPU long-tail batch 1c (T7-23 / ADR-0182 / ADR-0187):
    # ciede2000 ΔE. The CPU extractor is registered as `ciede`; the
    # GPU twin is `ciede_vulkan` (etc.). Per-pixel transcendentals
    # (pow / sqrt / sin / atan2) — places=2 contract, NOT bit-exact.
    "ciede": ("ciede2000",),
    # GPU long-tail batch 2 part 1 (T7-23 / ADR-0188 / ADR-0189):
    # float_ssim. Active CPU extractor is `float_ssim`; GPU twin
    # is `float_ssim_vulkan` (etc.). Single emitted metric;
    # places=4 contract per ADR-0189 (measure-then-set-the-contract,
    # relax to places=3 if the gate exceeds 5e-5 max_abs).
    "float_ssim": ("float_ssim",),
    # GPU long-tail batch 2 part 2 (T7-23 / ADR-0188 / ADR-0190):
    # float_ms_ssim. 5-level pyramid + Wang product combine. Single
    # emitted metric in v1 (enable_lcs deferred).
    "float_ms_ssim": ("float_ms_ssim",),
}

# Per-backend extractor-name suffix and the device-selection flag the
# CLI uses to actually route to it. CPU is the implicit baseline (no
# suffix, no device flag). If a future backend uses a different naming
# convention or a different device-selection flag, add it here.
BACKEND_SUFFIX: dict[str, str] = {
    "cuda": "_cuda",
    "sycl": "_sycl",
    "vulkan": "_vulkan",
}
BACKEND_DEVICE_FLAG: dict[str, str] = {
    "cuda": "--gpumask",
    "sycl": "--sycl_device",
    "vulkan": "--vulkan_device",
}


def run_vmaf(
    binary: Path,
    ref: Path,
    dist: Path,
    w: int,
    h: int,
    pix_fmt: str,
    bitdepth: int,
    feature: str,
    output: Path,
    backend: str | None,
    device: int | None,
) -> None:
    """Invoke `vmaf` with `--feature <feature>` (or its backend twin)
    and `--no_prediction` so the default model doesn't auto-load the
    CPU extractor alongside the GPU one and race them on the same
    feature names. See PR #120 commit message for the silent-CPU bug
    that motivated this contract.
    """
    extractor = feature if backend is None else f"{feature}{BACKEND_SUFFIX[backend]}"
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
        extractor,
        "--no_prediction",
        "--output",
        str(output),
        "--json",
    ]
    if backend is not None:
        # --backend forces backend exclusivity (no_cuda / no_sycl /
        # no_vulkan) so a build with multiple backends doesn't try
        # to init the unselected ones (which can hang on SYCL when
        # the device map differs between backends). The device flag
        # still pins the index for the chosen backend.
        cmd += ["--backend", backend]
        if device is not None:
            cmd += [BACKEND_DEVICE_FLAG[backend], str(device)]
    if backend is None:
        cmd += ["--backend", "cpu"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(2)


def load_frames(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["frames"]


def diff(cpu: list[dict], gpu: list[dict], metrics: tuple[str, ...], places: int) -> int:
    if len(cpu) != len(gpu):
        print(f"FAIL: frame count mismatch (cpu={len(cpu)}, gpu={len(gpu)})")
        return 1

    threshold = 0.5 * (10**-places)

    per_metric_max = dict.fromkeys(metrics, 0.0)
    per_metric_mismatch = dict.fromkeys(metrics, 0)

    for cf, gf in zip(cpu, gpu, strict=True):
        for m in metrics:
            c, v = cf["metrics"][m], gf["metrics"][m]
            d = abs(c - v)
            per_metric_max[m] = max(per_metric_max[m], d)
            if d > threshold:
                per_metric_mismatch[m] += 1

    print(f"cross-backend diff, {len(cpu)} frames, tolerance places={places}")
    print(f"{'metric':<25} {'max_abs_diff':<15} {'places={} mismatches'.format(places)}")
    fail = False
    for m in metrics:
        verdict = "OK" if per_metric_mismatch[m] == 0 else "FAIL"
        print(
            f"  {m:<25} {per_metric_max[m]:<15.6e} "
            f"{per_metric_mismatch[m]}/{len(cpu)}  {verdict}"
        )
        if per_metric_mismatch[m] > 0:
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
    ap.add_argument("--places", type=int, default=4)
    ap.add_argument(
        "--feature",
        choices=tuple(FEATURE_METRICS),
        default="vif",
        help="extractor to gate (vif | motion | adm)",
    )
    ap.add_argument(
        "--backend",
        choices=tuple(BACKEND_SUFFIX),
        default="vulkan",
        help="GPU backend to compare against CPU (cuda | sycl | vulkan)",
    )
    ap.add_argument(
        "--device",
        type=int,
        default=None,
        help=(
            "device index for the chosen backend. Vulkan/SYCL: 0+. "
            "CUDA: gpumask (e.g. 1 = first GPU). Defaults: vulkan=0, "
            "sycl=0, cuda=1."
        ),
    )
    # Back-compat alias for the existing CI lane that was wired before
    # --backend / --device existed. If --vulkan-device is passed, use
    # it as the Vulkan device index.
    ap.add_argument("--vulkan-device", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "vmaf_cross_backend",
    )
    args = ap.parse_args()

    if args.vulkan_device is not None:
        args.backend = "vulkan"
        args.device = args.vulkan_device
    if args.device is None:
        # Per-backend defaults: gpumask=1 picks the first GPU on CUDA;
        # device 0 is the first compute-capable on SYCL/Vulkan.
        args.device = 1 if args.backend == "cuda" else 0

    if not args.vmaf_binary.exists():
        print(f"vmaf binary not found: {args.vmaf_binary}")
        return 2
    for p in (args.reference, args.distorted):
        if not p.exists():
            print(f"fixture not found: {p}")
            return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    cpu_json = args.workdir / f"cpu_{args.feature}.json"
    gpu_json = args.workdir / f"{args.backend}_{args.feature}.json"

    print(f"running CPU {args.feature} → {cpu_json}")
    run_vmaf(
        args.vmaf_binary,
        args.reference,
        args.distorted,
        args.width,
        args.height,
        args.pixel_format,
        args.bitdepth,
        args.feature,
        cpu_json,
        backend=None,
        device=None,
    )

    print(f"running {args.backend} {args.feature} (device {args.device}) " f"→ {gpu_json}")
    run_vmaf(
        args.vmaf_binary,
        args.reference,
        args.distorted,
        args.width,
        args.height,
        args.pixel_format,
        args.bitdepth,
        args.feature,
        gpu_json,
        backend=args.backend,
        device=args.device,
    )

    return diff(
        load_frames(cpu_json),
        load_frames(gpu_json),
        FEATURE_METRICS[args.feature],
        args.places,
    )


if __name__ == "__main__":
    sys.exit(main())
