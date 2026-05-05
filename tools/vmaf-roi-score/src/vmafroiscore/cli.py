# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""argparse entry-point for ``vmaf-roi-score``.

Option C scaffold: drives the ``vmaf`` binary twice (full-frame +
saliency-masked) and emits a JSON record with both pooled scores plus
the saliency-weighted blend.

Per ADR-0288 phasing this PR ships the CLI surface, the combine-math,
and the test seam. The mask-materialisation path is behind
``--saliency-model`` and currently surfaces a clear "deferred" exit
status — synthetic-mode (``--synthetic-mask``) is used by the smoke
tests to verify the combine math end-to-end without ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ROI_RESULT_KEYS, SCHEMA_VERSION, __version__, blend_scores
from .score import ScoreRequest, run_score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmaf-roi-score",
        description=(
            "Region-of-interest VMAF score (Option C scaffold). Combines "
            "a full-frame VMAF run with a saliency-masked VMAF run via a "
            "user-supplied weight. Useful for content where bad "
            "background should not penalise a good salient region. "
            "Distinct from the libvmaf/tools/vmaf_roi binary (ADR-0247), "
            "which emits encoder QP-offset sidecars."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="raw reference YUV",
    )
    parser.add_argument(
        "--distorted",
        type=Path,
        required=True,
        help="raw distorted YUV",
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        help="ffmpeg pix_fmt (default yuv420p)",
    )
    parser.add_argument(
        "--saliency-model",
        type=Path,
        default=None,
        help=(
            "path to saliency ONNX (e.g. model/tiny/saliency_student_v1.onnx "
            "once PR #359 lands; falls back to mobilesal.onnx today). "
            "Mutually exclusive with --synthetic-mask."
        ),
    )
    parser.add_argument(
        "--synthetic-mask",
        type=float,
        default=None,
        metavar="FILL",
        help=(
            "skip ONNX inference; use a constant-value mask for testing. "
            "Value in [0, 1]. The combine math still runs; the masked "
            "VMAF run uses the same distorted YUV (no mask applied)."
        ),
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="saliency-masked component weight in [0, 1] (default 0.5)",
    )
    parser.add_argument(
        "--model",
        default="vmaf_v0.6.1",
        help="VMAF model version passed to the underlying vmaf CLI",
    )
    parser.add_argument(
        "--vmaf-bin",
        default="vmaf",
        help="path to the libvmaf CLI binary (default: vmaf on PATH)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write JSON result to this file; default writes to stdout",
    )
    return parser


def _validate(ns: argparse.Namespace) -> None:
    """Hand-rolled cross-arg validation; argparse-only is too coarse."""
    if ns.saliency_model is None and ns.synthetic_mask is None:
        raise SystemExit("vmaf-roi-score: one of --saliency-model or --synthetic-mask is required")
    if ns.saliency_model is not None and ns.synthetic_mask is not None:
        raise SystemExit(
            "vmaf-roi-score: --saliency-model and --synthetic-mask are mutually exclusive"
        )
    if ns.saliency_model is not None and not ns.saliency_model.exists():
        raise SystemExit(f"vmaf-roi-score: saliency model not found: {ns.saliency_model}")
    if not ns.reference.exists():
        raise SystemExit(f"vmaf-roi-score: reference not found: {ns.reference}")
    if not ns.distorted.exists():
        raise SystemExit(f"vmaf-roi-score: distorted not found: {ns.distorted}")
    if not (0.0 <= ns.weight <= 1.0):
        raise SystemExit(f"vmaf-roi-score: --weight must be in [0, 1], got {ns.weight}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    _validate(ns)

    score_req = ScoreRequest(
        reference=ns.reference,
        distorted=ns.distorted,
        width=ns.width,
        height=ns.height,
        pix_fmt=ns.pix_fmt,
        model=ns.model,
    )

    full = run_score(score_req, vmaf_bin=ns.vmaf_bin)
    if full.exit_status != 0:
        sys.stderr.write(
            f"vmaf-roi-score: full-frame vmaf run failed (exit={full.exit_status}); "
            f"stderr tail:\n{full.stderr_tail}\n"
        )
        return full.exit_status

    if ns.saliency_model is not None:
        # Real saliency-masked variant — Option C wiring deferred (T6-2c).
        # Surface a clear error rather than silently scoring the same YUV
        # twice.
        sys.stderr.write(
            "vmaf-roi-score: --saliency-model is scaffolded but mask "
            "materialisation is not wired yet (see ADR-0288). Use "
            "--synthetic-mask for the combine-math smoke today.\n"
        )
        return 64

    # --synthetic-mask path: re-score the same distorted YUV. The two
    # scalars are identical by construction, so the blend collapses to
    # vmaf_full. This is the smoke-test contract — it proves the
    # subprocess seam + JSON parse + combine math without depending on
    # the deferred mask materialiser.
    masked = run_score(score_req, vmaf_bin=ns.vmaf_bin)

    roi = blend_scores(full.vmaf_score, masked.vmaf_score, ns.weight)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "vmaf_full": full.vmaf_score,
        "vmaf_masked": masked.vmaf_score,
        "weight": ns.weight,
        "vmaf_roi": roi,
        "model": ns.model,
        "saliency_model": "synthetic" if ns.synthetic_mask is not None else None,
        "reference": str(ns.reference),
        "distorted": str(ns.distorted),
    }
    # Pin key order to the canonical schema; tests assert on this.
    payload = {k: payload[k] for k in ROI_RESULT_KEYS}

    text = json.dumps(payload, indent=2)
    if ns.output is None:
        sys.stdout.write(text + "\n")
    else:
        ns.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    raise SystemExit(main())
