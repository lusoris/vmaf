# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""argparse entry-point for ``vmaf-tune``.

Phase A exposes one subcommand: ``corpus``. It expands a (preset, crf)
grid against one or more reference YUVs and emits a JSONL row per
encode. Phase B (``bisect``) and Phase C (``predict``) will register
sibling subcommands here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .codec_adapters import known_codecs
from .corpus import CorpusJob, CorpusOptions, coarse_to_fine_search, iter_rows, write_jsonl
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
        default=None,
        help=(
            "x264 CRF value (repeatable). Required unless "
            "--coarse-to-fine selects the CRF axis automatically."
        ),
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
    corpus.add_argument(
        "--sample-clip-seconds",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "encode/score only the centre N-second slice of each source "
            "(default 0 = full source). Encode time scales linearly with "
            "the slice length, so e.g. 10s of a 60s source is a ~6x "
            "speedup; expect a 1-2 VMAF-point delta vs full-clip on "
            "diverse content. See ADR-0297."
        ),
    )
    _add_coarse_to_fine_flags(corpus)

    recommend = sub.add_parser(
        "recommend",
        help=(
            "find the smallest CRF whose VMAF >= --target-vmaf "
            "(coarse-to-fine, ~3.5x fewer encodes than the full grid)"
        ),
    )
    _add_recommend_args(recommend)
    return parser


def _add_coarse_to_fine_flags(p: argparse.ArgumentParser) -> None:
    """Wire ``--coarse-to-fine`` + tunables onto a subparser.

    Used by both ``corpus`` (opt-in) and ``recommend`` (always on).
    """
    p.add_argument(
        "--coarse-to-fine",
        action="store_true",
        help=(
            "run a 2-pass coarse-then-fine CRF search instead of the "
            "full grid (ADR-0296). With defaults: 5 coarse + up to 10 "
            "fine = 15 encodes vs 52 for a full 0..51 sweep."
        ),
    )
    p.add_argument(
        "--coarse-step",
        type=int,
        default=10,
        help="CRF step for the coarse pass (default 10 -> [10,20,30,40,50])",
    )
    p.add_argument(
        "--fine-radius",
        type=int,
        default=5,
        help="±radius around best-coarse CRF for the fine pass (default 5)",
    )
    p.add_argument(
        "--fine-step",
        type=int,
        default=1,
        help="CRF step for the fine pass (default 1)",
    )
    p.add_argument(
        "--target-vmaf",
        type=float,
        default=None,
        help=(
            "target VMAF score; the orchestrator picks the smallest "
            "CRF whose score >= target. Optional for `corpus`, "
            "required for `recommend`."
        ),
    )


def _add_recommend_args(p: argparse.ArgumentParser) -> None:
    """Mirror the corpus subparser's source/encode flags for ``recommend``.

    ``recommend`` always runs coarse-to-fine — keeping the flag surface
    aligned with ``corpus`` means downstream scripts can swap one for
    the other without re-learning the CLI.
    """
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--pix-fmt", default="yuv420p")
    p.add_argument("--framerate", type=float, default=24.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--encoder", default="libx264", choices=list(known_codecs()))
    p.add_argument("--preset", action="append", required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("corpus.jsonl"),
        help="JSONL destination for the visited points",
    )
    p.add_argument(
        "--encode-dir",
        type=Path,
        default=Path(".workingdir2/encodes"),
    )
    p.add_argument("--keep-encodes", action="store_true")
    p.add_argument("--vmaf-model", default="vmaf_v0.6.1")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--vmaf-bin", default="vmaf")
    p.add_argument("--no-source-hash", action="store_true")
    _add_coarse_to_fine_flags(p)


def _build_opts(args: argparse.Namespace) -> CorpusOptions:
    return CorpusOptions(
        encoder=args.encoder,
        output=args.output,
        encode_dir=args.encode_dir,
        vmaf_model=args.vmaf_model,
        ffmpeg_bin=args.ffmpeg_bin,
        vmaf_bin=args.vmaf_bin,
        keep_encodes=args.keep_encodes,
        src_sha256=not args.no_source_hash,
        sample_clip_seconds=args.sample_clip_seconds,
    )


def _build_job(args: argparse.Namespace, src: Path, cells: tuple) -> CorpusJob:
    return CorpusJob(
        source=src,
        width=args.width,
        height=args.height,
        pix_fmt=args.pix_fmt,
        framerate=args.framerate,
        duration_s=args.duration,
        cells=cells,
    )


def _run_corpus(args: argparse.Namespace) -> int:
    opts = _build_opts(args)

    if args.coarse_to_fine:
        # Coarse-to-fine ignores --crf and uses the configured grid.
        # Use a sentinel preset-only cell list so coarse_to_fine_search
        # can extract the preset axis.
        if not args.preset:
            sys.stderr.write("--preset is required\n")
            return 2
        sentinel_cells = tuple((p, 0) for p in args.preset)

        def _all_rows():
            for src in args.source:
                job = _build_job(args, src, sentinel_cells)
                yield from coarse_to_fine_search(
                    job,
                    opts,
                    target_vmaf=args.target_vmaf,
                    coarse_step=args.coarse_step,
                    fine_radius=args.fine_radius,
                    fine_step=args.fine_step,
                )

        n = write_jsonl(_all_rows(), opts.output)
        sys.stderr.write(f"coarse-to-fine: wrote {n} rows -> {opts.output}\n")
        return 0

    if not args.crf:
        sys.stderr.write("--crf is required (or use --coarse-to-fine)\n")
        return 2
    cells = tuple(iter_grid(args.preset, args.crf))

    def _all_rows():
        for src in args.source:
            job = _build_job(args, src, cells)
            yield from iter_rows(job, opts)

    n = write_jsonl(_all_rows(), opts.output)
    sys.stderr.write(f"wrote {n} rows -> {opts.output}\n")
    return 0


def _run_recommend(args: argparse.Namespace) -> int:
    if args.target_vmaf is None:
        sys.stderr.write("recommend requires --target-vmaf\n")
        return 2

    opts = _build_opts(args)
    sentinel_cells = tuple((p, 0) for p in args.preset)

    visited: list[dict] = []

    def _capture():
        for src in args.source:
            job = _build_job(args, src, sentinel_cells)
            for row in coarse_to_fine_search(
                job,
                opts,
                target_vmaf=args.target_vmaf,
                coarse_step=args.coarse_step,
                fine_radius=args.fine_radius,
                fine_step=args.fine_step,
            ):
                visited.append(row)
                yield row

    write_jsonl(_capture(), opts.output)
    pick = _smallest_passing_crf(visited, args.target_vmaf)
    if pick is None:
        sys.stderr.write(
            f"recommend: no CRF meets target VMAF >= {args.target_vmaf}; "
            f"visited {len(visited)} encodes -> {opts.output}\n"
        )
        return 1
    src, preset, crf, score = pick
    sys.stdout.write(
        f"src={src} preset={preset} crf={crf} vmaf={score:.3f} "
        f"(visited {len(visited)} encodes)\n"
    )
    return 0


def _smallest_passing_crf(
    rows: list[dict], target_vmaf: float
) -> tuple[str, str, int, float] | None:
    """Return (src, preset, crf, vmaf) for the cheapest passing encode.

    "Cheapest" here means the LARGEST CRF whose ``vmaf_score`` still
    meets ``target_vmaf`` — for libx264 a larger CRF means a smaller
    bitrate, so the largest passing CRF is the smallest bitrate that
    clears the quality gate. Grouped per (src, preset); we return the
    first such (src, preset) pair in the natural row order.
    """
    best: dict[tuple[str, str], tuple[int, float]] = {}
    for r in rows:
        try:
            score = float(r.get("vmaf_score"))
        except (TypeError, ValueError):
            continue
        if score < target_vmaf:
            continue
        key = (str(r["src"]), str(r["preset"]))
        crf = int(r["crf"])
        cur = best.get(key)
        # We want the LARGEST CRF that still meets the target — that's
        # the smallest bitrate at acceptable quality. Tie-break on the
        # higher VMAF score for determinism.
        if cur is None or crf > cur[0] or (crf == cur[0] and score > cur[1]):
            best[key] = (crf, score)
    if not best:
        return None
    # Return the first key in row order.
    for r in rows:
        key = (str(r["src"]), str(r["preset"]))
        if key in best:
            crf, score = best[key]
            return key[0], key[1], crf, score
    return None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "corpus":
        return _run_corpus(args)
    if args.cmd == "recommend":
        return _run_recommend(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
