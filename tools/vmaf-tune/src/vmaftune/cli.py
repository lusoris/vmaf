# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""argparse entry-point for ``vmaf-tune``.

Phase A exposes the ``corpus`` subcommand: it expands a (preset, crf)
grid against one or more reference YUVs and emits a JSONL row per
encode. Phase A.5 adds the opt-in ``fast`` subcommand (proxy +
Bayesian + GPU-verify recommend, ADR-0276 / Research-0060). Codecs
available today: ``libx264`` (default), ``libaom-av1`` (ADR-0279),
and ``libx265`` (ADR-0288). Phase B (``bisect``) and Phase C
(``predict``) will register sibling subcommands here.
Subcommands:

- ``corpus`` — Phase A grid sweep, emits JSONL rows.
- ``recommend`` — Phase B-lite. Apply ``--target-vmaf`` or
  ``--target-bitrate`` predicate over a corpus (built on the fly or
  loaded from a pre-existing JSONL). Implements Buckets #4 and #5 of
  Research-0061.

Phase C (``predict``) will register a sibling subcommand here.
Phase A: ``corpus`` (grid sweep -> JSONL). Phase E: ``ladder``
(per-title bitrate-ladder generator -> HLS / DASH / JSON manifest).
Phase B (``bisect``), Phase C (``predict``), and Phase D
(``per-shot``) will register sibling subcommands here.
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
from .ladder import build_and_emit
from .per_shot import (
    detect_shots,
    merge_shots,
    plan_to_shell_script,
    tune_per_shot,
    write_concat_listing,
)
from .recommend import RecommendRequest, format_result, load_corpus_jsonl, recommend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmaf-tune",
        description=(
            "Quality-aware encode automation harness. Phase A drives a "
            "(preset, crf) grid through the selected codec (libx264, "
            "libx265) + libvmaf and emits a JSONL corpus."
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
        help="codec adapter (one of libx264, libx265)",
    )
    corpus.add_argument(
        "--preset",
        action="append",
        required=True,
        help="encoder preset (repeatable; codec-specific name)",
    )
    corpus.add_argument(
        "--crf",
        type=int,
        action="append",
        required=True,
        help="encoder CRF value (repeatable; codec-specific range)",
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
        help=(
            "vmaf model version string (default vmaf_v0.6.1). Only used "
            "when --no-resolution-aware is set; otherwise the model is "
            "auto-picked per encode resolution."
        ),
    )
    corpus.add_argument(
        "--resolution-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "auto-pick the VMAF model per encode resolution "
            "(height>=2160 -> vmaf_4k_v0.6.1, else vmaf_v0.6.1). "
            "Default: on. Disable with --no-resolution-aware to force "
            "a single model via --vmaf-model."
        ),
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

    rec = sub.add_parser(
        "recommend",
        help="apply --target-vmaf / --target-bitrate predicate over a corpus",
        description=(
            "Find the (preset, crf) cell that best satisfies a target. "
            "Either point at a pre-built corpus JSONL (--from-corpus) or "
            "let the subcommand build one on the fly via the same Phase A "
            "pipeline (--source + grid flags)."
        ),
    )
    target = rec.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-vmaf",
        type=float,
        help="return the smallest CRF whose VMAF >= TARGET",
    )
    target.add_argument(
        "--target-bitrate",
        type=float,
        metavar="KBPS",
        help="return the row whose bitrate is closest to KBPS",
    )
    rec.add_argument(
        "--from-corpus",
        type=Path,
        default=None,
        help="read rows from this JSONL instead of building a corpus on the fly",
    )
    rec.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="raw YUV reference (repeatable); ignored when --from-corpus is set",
    )
    rec.add_argument("--width", type=int, default=None)
    rec.add_argument("--height", type=int, default=None)
    rec.add_argument("--pix-fmt", default="yuv420p")
    rec.add_argument("--framerate", type=float, default=24.0)
    rec.add_argument("--duration", type=float, default=0.0)
    rec.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter; also filters --from-corpus rows",
    )
    rec.add_argument(
        "--preset",
        action="append",
        default=None,
        help="preset(s) to sweep / filter; repeatable",
    )
    rec.add_argument(
        "--crf",
        type=int,
        action="append",
        default=None,
        help="CRF integer(s) to sweep; ignored when --from-corpus is set",
    )
    rec.add_argument(
        "--encode-dir",
        type=Path,
        default=Path(".workingdir2/encodes"),
    )
    rec.add_argument("--keep-encodes", action="store_true")
    rec.add_argument("--vmaf-model", default="vmaf_v0.6.1")
    rec.add_argument("--ffmpeg-bin", default="ffmpeg")
    rec.add_argument("--vmaf-bin", default="vmaf")
    rec.add_argument("--no-source-hash", action="store_true")
    rec.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="emit the winning row as JSON on stdout (default: human-readable)",
    )

    per_shot = sub.add_parser(
        "tune-per-shot",
        help=(
            "Phase D scaffold — detect shots via vmaf-perShot/TransNet V2, "
            "tune CRF per shot, and emit an FFmpeg encoding plan."
        ),
    )
    per_shot.add_argument(
        "--src",
        type=Path,
        required=True,
        help="reference video (raw YUV or any FFmpeg-readable container)",
    )
    per_shot.add_argument("--width", type=int, required=True)
    per_shot.add_argument("--height", type=int, required=True)
    per_shot.add_argument("--pix-fmt", default="yuv420p")
    per_shot.add_argument("--framerate", type=float, default=24.0)
    per_shot.add_argument(
        "--target-vmaf",
        type=float,
        default=92.0,
        help="target pooled-mean VMAF for the per-shot predicate (default 92)",
    )
    per_shot.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (Phase D scaffold: libx264 only)",
    )
    per_shot.add_argument(
        "--bitdepth",
        type=int,
        default=8,
        choices=(8, 10, 12),
        help="source YUV bit depth (forwarded to vmaf-perShot)",
    )
    per_shot.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help="frame count for the single-shot fallback (used when " "vmaf-perShot is unavailable)",
    )
    per_shot.add_argument(
        "--per-shot-bin",
        default="vmaf-perShot",
        help="path to the vmaf-perShot binary (default vmaf-perShot on PATH)",
    )
    per_shot.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="path to the ffmpeg binary (default ffmpeg on PATH)",
    )
    per_shot.add_argument(
        "--output",
        type=Path,
        default=Path("per_shot_encode.mp4"),
        help="final concatenated encode destination (default per_shot_encode.mp4)",
    )
    per_shot.add_argument(
        "--segment-dir",
        type=Path,
        default=None,
        help="directory for per-shot segment files (default <output>.parent/segments)",
    )
    per_shot.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="emit the JSON plan to this path; default: stdout",
    )
    per_shot.add_argument(
        "--script-out",
        type=Path,
        default=None,
        help="optional: write a copy-paste shell script of the plan",
    )

    ladder = sub.add_parser(
        "ladder",
        help=(
            "Phase E — generate a per-title ABR bitrate ladder from the "
            "Pareto hull of (resolution, vmaf) samples"
        ),
    )
    ladder.add_argument(
        "--src",
        type=Path,
        required=True,
        help="source video (used to label the manifest; sampling is currently "
        "Phase B / Phase A driven)",
    )
    ladder.add_argument(
        "--encoder",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (Phase E uses Phase A's adapters)",
    )
    ladder.add_argument(
        "--resolutions",
        default="1920x1080,1280x720,854x480,640x360,426x240",
        help="comma-separated WxH list; default is the canonical 5-rung "
        '"1080p/720p/480p/360p/240p" set',
    )
    ladder.add_argument(
        "--target-vmafs",
        default="95,90,85,75,65",
        help="comma-separated VMAF targets to bisect against per resolution; "
        "Phase B's bisect logic resolves each to a (bitrate, crf)",
    )
    ladder.add_argument(
        "--quality-tiers",
        type=int,
        default=5,
        help="number of rungs to pick from the Pareto hull (default 5)",
    )
    ladder.add_argument(
        "--spacing",
        choices=["log_bitrate", "vmaf"],
        default="log_bitrate",
        help="rung spacing on the hull (default: log-bitrate, Apple HLS spec)",
    )
    ladder.add_argument(
        "--format",
        choices=["hls", "dash", "json"],
        default="hls",
        help="manifest format to emit (default: hls)",
    )
    ladder.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write manifest to PATH (default: stdout)",
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
        resolution_aware=args.resolution_aware,
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


def _rows_for_recommend(args: argparse.Namespace):
    """Yield rows for the recommend subcommand.

    Either streams a pre-built corpus (``--from-corpus``) or runs the
    Phase A pipeline against ``--source`` and yields rows as they're
    produced.
    """
    if args.from_corpus is not None:
        yield from load_corpus_jsonl(args.from_corpus)
        return
    if not args.source or args.width is None or args.height is None:
        raise SystemExit(
            "vmaf-tune recommend: pass --from-corpus PATH, "
            "or supply --source/--width/--height to build a corpus on the fly"
        )
    presets = args.preset or ["medium"]
    crfs = args.crf or list(range(18, 36, 2))
    cells = tuple(iter_grid(presets, crfs))
    opts = CorpusOptions(
        encoder=args.encoder,
        output=Path("/dev/null"),  # not written here; caller iterates
        encode_dir=args.encode_dir,
        vmaf_model=args.vmaf_model,
        ffmpeg_bin=args.ffmpeg_bin,
        vmaf_bin=args.vmaf_bin,
        keep_encodes=args.keep_encodes,
        src_sha256=not args.no_source_hash,
    )
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


def _run_recommend(args: argparse.Namespace) -> int:
    import json as _json

    req = RecommendRequest(
        target_vmaf=args.target_vmaf,
        target_bitrate_kbps=args.target_bitrate,
        encoder=args.encoder,
        preset=(args.preset[0] if args.preset and len(args.preset) == 1 else None),
    )
    try:
        rows = list(_rows_for_recommend(args))
        result = recommend(rows, req)
    except ValueError as exc:
        sys.stderr.write(f"vmaf-tune recommend: {exc}\n")
        return 2
    if args.emit_json:
        sys.stdout.write(_json.dumps(result.row, sort_keys=True) + "\n")
    else:
        sys.stdout.write(format_result(result) + "\n")


def _run_tune_per_shot(args: argparse.Namespace) -> int:
    total_frames = args.total_frames if args.total_frames > 0 else None
    shots = detect_shots(
        args.src,
        width=args.width,
        height=args.height,
        pix_fmt=args.pix_fmt,
        bitdepth=args.bitdepth,
        total_frames=total_frames,
        per_shot_bin=args.per_shot_bin,
    )
    recs = tune_per_shot(
        shots,
        target_vmaf=args.target_vmaf,
        encoder=args.encoder,
    )
    plan = merge_shots(
        recs,
        source=args.src,
        output=args.output,
        framerate=args.framerate,
        encoder=args.encoder,
        segment_dir=args.segment_dir,
        ffmpeg_bin=args.ffmpeg_bin,
    )

    plan_doc = {
        "encoder": plan.encoder,
        "framerate": plan.framerate,
        "target_vmaf": args.target_vmaf,
        "shots": [
            {
                "start_frame": r.shot.start_frame,
                "end_frame": r.shot.end_frame,
                "crf": r.crf,
                "predicted_vmaf": r.predicted_vmaf,
            }
            for r in plan.recommendations
        ],
        "segment_commands": [list(c) for c in plan.segment_commands],
        "concat_command": list(plan.concat_command),
    }
    rendered = json.dumps(plan_doc, indent=2, sort_keys=True)
    if args.plan_out is None:
        sys.stdout.write(rendered)
        sys.stdout.write("\n")
    else:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(rendered + "\n", encoding="utf-8")
        sys.stderr.write(f"wrote plan -> {args.plan_out}\n")

    if args.script_out is not None:
        args.script_out.parent.mkdir(parents=True, exist_ok=True)
        args.script_out.write_text(plan_to_shell_script(plan), encoding="utf-8")
        sys.stderr.write(f"wrote shell script -> {args.script_out}\n")

    seg_dir = args.segment_dir or args.output.parent / "segments"
    write_concat_listing(plan, seg_dir / "concat.txt")


def _parse_resolutions(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"resolution {token!r} must be WxH (e.g. 1920x1080)")
        w_s, h_s = token.split("x", 1)
        out.append((int(w_s), int(h_s)))
    if not out:
        raise ValueError("empty --resolutions list")
    return out


def _parse_target_vmafs(raw: str) -> list[float]:
    out = [float(t) for t in raw.split(",") if t.strip()]
    if not out:
        raise ValueError("empty --target-vmafs list")
    return out


def _run_ladder(args: argparse.Namespace) -> int:
    resolutions = _parse_resolutions(args.resolutions)
    target_vmafs = _parse_target_vmafs(args.target_vmafs)
    # No production sampler is wired yet — Phase B's target-VMAF bisect
    # (PR #347) lands the integration. Until then, the CLI errors out
    # clearly rather than silently faking points.
    manifest = build_and_emit(
        src=args.src,
        encoder=args.encoder,
        resolutions=resolutions,
        target_vmafs=target_vmafs,
        quality_tiers=args.quality_tiers,
        format=args.format,
        spacing=args.spacing,
    )
    if args.output is None:
        sys.stdout.write(manifest)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest)
        sys.stderr.write(f"wrote ladder manifest -> {args.output}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "corpus":
        return _run_corpus(args)
    if args.cmd == "fast":
        return _run_fast(args)
    if args.cmd == "recommend":
        return _run_recommend(args)
    if args.cmd == "tune-per-shot":
        return _run_tune_per_shot(args)
    if args.cmd == "ladder":
        return _run_ladder(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
