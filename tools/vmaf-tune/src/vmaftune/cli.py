# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""argparse entry-point for ``vmaf-tune``.

Phase A exposes the ``corpus`` subcommand: it expands a (preset, crf)
grid against one or more reference YUVs and emits a JSONL row per
encode. Phase A.5 adds the opt-in ``fast`` subcommand (proxy +
Bayesian + GPU-verify recommend, ADR-0276 / Research-0060). Phase B
(``bisect``) and Phase C (``predict``) will register sibling
subcommands here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .codec_adapters import known_codecs
from .corpus import CorpusJob, CorpusOptions, iter_rows, write_jsonl
from .encode import iter_grid


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmaf-tune",
        description=(
            "Quality-aware encode automation harness. Phase A drives a "
            "(preset, crf) grid through libx264 + libvmaf and emits a JSONL "
            "corpus."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    corpus = sub.add_parser("corpus", help="run the Phase A grid sweep + emit JSONL")
    corpus.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="raw YUV reference (repeat for multiple sources)",
    )
    corpus.add_argument("--width", type=int, required=True)
    corpus.add_argument("--height", type=int, required=True)
    corpus.add_argument("--pix-fmt", default="yuv420p", help="ffmpeg pix_fmt (default yuv420p)")
    corpus.add_argument("--framerate", type=float, default=24.0, help="reference framerate")
    corpus.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="reference duration in seconds (used for bitrate calc)",
    )
    corpus.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (Phase A: libx264 only)",
    )
    corpus.add_argument(
        "--preset",
        action="append",
        required=True,
        help="x264 preset (repeatable)",
    )
    corpus.add_argument(
        "--crf",
        type=int,
        action="append",
        required=True,
        help="x264 CRF value (repeatable)",
    )
    corpus.add_argument(
        "--output",
        type=Path,
        default=Path("corpus.jsonl"),
        help="JSONL output path (default corpus.jsonl)",
    )
    corpus.add_argument(
        "--encode-dir",
        type=Path,
        default=Path(".workingdir2/encodes"),
        help="scratch dir for encodes (default .workingdir2/encodes, gitignored)",
    )
    corpus.add_argument(
        "--keep-encodes",
        action="store_true",
        help="retain encoded outputs after scoring (default: delete)",
    )
    corpus.add_argument(
        "--vmaf-model",
        default="vmaf_v0.6.1",
        help="vmaf model version string (default vmaf_v0.6.1)",
    )
    corpus.add_argument("--ffmpeg-bin", default="ffmpeg")
    corpus.add_argument("--vmaf-bin", default="vmaf")
    corpus.add_argument(
        "--no-source-hash",
        action="store_true",
        help="skip src_sha256 (faster on huge YUVs; loses provenance)",
    )

    fast = sub.add_parser(
        "fast",
        help="proxy-based recommend (ADR-0276, Phase A.5 scaffold)",
        description=(
            "Fast-path recommend: Bayesian search over CRF using a tiny-AI "
            "VMAF proxy in place of the encode-decode-score loop. Production "
            "wiring is scaffold-only in this PR; --smoke exercises the "
            "pipeline end-to-end without ffmpeg / ONNX. See ADR-0276."
        ),
    )
    fast.add_argument(
        "--src",
        type=Path,
        default=None,
        help="source video (only required outside --smoke mode)",
    )
    fast.add_argument(
        "--target-vmaf",
        type=float,
        default=92.0,
        help="quality target on VMAF [0, 100] scale (default 92.0)",
    )
    fast.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (Phase A.5: libx264 only; defaults to host's available)",
    )
    fast.add_argument(
        "--crf-lo",
        type=int,
        default=10,
        help="lower bound of the CRF search range (inclusive, default 10)",
    )
    fast.add_argument(
        "--crf-hi",
        type=int,
        default=51,
        help="upper bound of the CRF search range (inclusive, default 51)",
    )
    fast.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="number of Optuna TPE trials (default 50)",
    )
    fast.add_argument(
        "--time-budget-s",
        type=int,
        default=300,
        help="soft wall-clock budget in seconds (advisory in scaffold)",
    )
    fast.add_argument(
        "--smoke",
        action="store_true",
        help="run synthetic-predictor smoke pipeline (no ffmpeg / no ONNX)",
    )
    return parser


def _run_corpus(args: argparse.Namespace) -> int:
    cells = tuple(iter_grid(args.preset, args.crf))
    opts = CorpusOptions(
        encoder=args.encoder,
        output=args.output,
        encode_dir=args.encode_dir,
        vmaf_model=args.vmaf_model,
        ffmpeg_bin=args.ffmpeg_bin,
        vmaf_bin=args.vmaf_bin,
        keep_encodes=args.keep_encodes,
        src_sha256=not args.no_source_hash,
    )

    def _all_rows():
        for src in args.source:
            job = CorpusJob(
                source=src,
                width=args.width,
                height=args.height,
                pix_fmt=args.pix_fmt,
                framerate=args.framerate,
                duration_s=args.duration,
                cells=cells,
            )
            yield from iter_rows(job, opts)

    n = write_jsonl(_all_rows(), opts.output)
    sys.stderr.write(f"wrote {n} rows -> {opts.output}\n")
    return 0


def _run_fast(args: argparse.Namespace) -> int:
    # Lazy import: keeps Optuna optional on the core install path.
    from .fast import fast_recommend

    try:
        result = fast_recommend(
            src=args.src,
            target_vmaf=args.target_vmaf,
            encoder=args.encoder,
            time_budget_s=args.time_budget_s,
            crf_range=(args.crf_lo, args.crf_hi),
            n_trials=args.n_trials,
            smoke=args.smoke,
        )
    except RuntimeError as exc:
        sys.stderr.write(f"vmaf-tune fast: {exc}\n")
        return 2
    except NotImplementedError as exc:
        sys.stderr.write(f"vmaf-tune fast: {exc}\n")
        return 3
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "corpus":
        return _run_corpus(args)
    if args.cmd == "fast":
        return _run_fast(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
