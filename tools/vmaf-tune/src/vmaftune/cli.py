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
from .per_shot import (
    detect_shots,
    merge_shots,
    plan_to_shell_script,
    tune_per_shot,
    write_concat_listing,
)
from .score_backend import ALL_BACKENDS, BackendUnavailableError, select_backend


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
        "--score-backend",
        default="auto",
        choices=("auto", *ALL_BACKENDS),
        help=(
            "libvmaf scoring backend (default: auto). 'auto' picks the "
            "fastest available (cuda > vulkan > sycl > cpu); a specific "
            "name is honoured strictly and errors out if unavailable. "
            "Use 'vulkan' on AMD / Intel Arc / Apple-MoltenVK hosts "
            "(ADR-0314)."
        ),
    )
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

    predict = sub.add_parser(
        "predict",
        help=(
            "Phase C — predict per-shot VMAF without running it. Probes-encode "
            "each shot, runs a learned ONNX predictor (or analytical fallback), "
            "validates against real VMAF on K shots, then emits the verdict."
        ),
    )
    predict.add_argument(
        "--source",
        type=Path,
        required=True,
        help="reference video (any FFmpeg-readable container)",
    )
    predict.add_argument(
        "--codec",
        default="libx264",
        choices=list(known_codecs()),
        help="codec adapter (default libx264)",
    )
    predict.add_argument(
        "--target-vmaf",
        type=float,
        default=93.0,
        help="target pooled-mean VMAF (default 93)",
    )
    predict.add_argument(
        "--validate-k",
        type=int,
        default=8,
        help="number of shots to verify against real libvmaf (default 8)",
    )
    predict.add_argument(
        "--residual-threshold",
        type=float,
        default=1.5,
        help="max abs(predicted - measured) VMAF before falling back (default 1.5)",
    )
    predict.add_argument(
        "--use-saliency",
        action="store_true",
        help="layer the saliency QP-offset map on top of the picked CRF "
        "(libx264 only for now; other codecs warn and skip)",
    )
    predict.add_argument(
        "--model",
        type=Path,
        default=None,
        help="path to predictor_<codec>.onnx (default: analytical fallback)",
    )
    predict.add_argument(
        "--per-shot-bin",
        default="vmaf-perShot",
        help="path to the vmaf-perShot binary (default vmaf-perShot on PATH)",
    )
    predict.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="path to the ffmpeg binary (default ffmpeg on PATH)",
    )
    predict.add_argument(
        "--bitdepth",
        type=int,
        default=8,
        choices=(8, 10, 12),
        help="source bit depth (forwarded to vmaf-perShot)",
    )
    predict.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help="frame count for the single-shot fallback (when vmaf-perShot is unavailable)",
    )
    predict.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="emit the validation report (verdict + residuals) to this path; default: stdout",
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
        help=(
            "frame count for the single-shot fallback (used when " "vmaf-perShot is unavailable)"
        ),
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
    p.add_argument(
        "--score-backend",
        default="auto",
        choices=("auto", *ALL_BACKENDS),
        help=(
            "libvmaf scoring backend (default: auto; cuda > vulkan > "
            "sycl > cpu). See `vmaf-tune corpus --help`."
        ),
    )
    p.add_argument("--no-source-hash", action="store_true")
    _add_coarse_to_fine_flags(p)


def _build_opts(args: argparse.Namespace) -> CorpusOptions:
    # ADR-0299 / ADR-0314: resolve --score-backend up-front so an
    # unavailable backend errors out before we burn cycles on encodes.
    # `select_backend` raises `BackendUnavailableError` (caught by the
    # caller) when a non-auto backend is requested but the host can't
    # provide it.
    selected = select_backend(prefer=args.score_backend, vmaf_bin=args.vmaf_bin)
    sys.stderr.write(f"vmaf-tune: scoring backend = {selected}\n")
    return CorpusOptions(
        encoder=args.encoder,
        output=args.output,
        encode_dir=args.encode_dir,
        vmaf_model=args.vmaf_model,
        ffmpeg_bin=args.ffmpeg_bin,
        vmaf_bin=args.vmaf_bin,
        keep_encodes=args.keep_encodes,
        src_sha256=not args.no_source_hash,
        sample_clip_seconds=getattr(args, "sample_clip_seconds", 0.0),
        score_backend=selected,
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
    try:
        opts = _build_opts(args)
    except BackendUnavailableError as exc:
        sys.stderr.write(f"vmaf-tune: {exc}\n")
        return 2

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

    try:
        opts = _build_opts(args)
    except BackendUnavailableError as exc:
        sys.stderr.write(f"vmaf-tune: {exc}\n")
        return 2
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


def _run_predict(args: argparse.Namespace) -> int:
    """Phase C — per-shot VMAF prediction + validation harness.

    Pipeline:

    1.  Detect shots via :func:`per_shot.detect_shots` (TransNet V2
        binary if available; one-shot fallback otherwise).
    2.  Build a :class:`predictor.Predictor` (ONNX or analytical
        fallback).
    3.  Validate the predictor on K stratified shots — for each, run
        the real ffmpeg encode at the predictor-picked CRF + libvmaf
        score, compute residuals.
    4.  Emit the verdict + residuals + recommended per-shot CRFs as a
        JSON report.
    """
    import subprocess
    import tempfile

    from .encode import EncodeRequest, run_encode
    from .per_shot import Shot, detect_shots
    from .predictor import Predictor
    from .predictor_features import FeatureExtractorConfig, _probe_video_geometry, extract_features
    from .predictor_validate import Verdict, validate_predictor
    from .score import ScoreRequest, run_score

    shots = detect_shots(
        source=args.source,
        width=0,
        height=0,
        bitdepth=args.bitdepth,
        framerate=0.0,
        total_frames=args.total_frames or 0,
        per_shot_bin=args.per_shot_bin,
    )
    if not shots:
        print("predict: no shots detected; nothing to do", file=sys.stderr)
        return 1

    feat_cfg = FeatureExtractorConfig(
        ffmpeg_bin=args.ffmpeg_bin,
        use_saliency=args.use_saliency,
    )

    # Probe geometry once — every validation shot reuses the same
    # width/height/fps/pix_fmt for both reference extraction and the
    # encode dispatch.
    width, height, fps = _probe_video_geometry(args.source, feat_cfg, subprocess.run)
    if width <= 0 or height <= 0:
        print(
            "predict: ffprobe could not read source geometry "
            "(width/height); falling back is not safe — aborting.",
            file=sys.stderr,
        )
        return 1
    pix_fmt = "yuv420p"  # canonical reference format; matches saliency.py + the corpus loop

    predictor = Predictor(model_path=args.model)

    def _features(shot):
        return extract_features(
            shot=shot,
            source=args.source,
            codec=args.codec,
            config=feat_cfg,
        )

    # Validation work-area lives for the lifetime of _run_predict so
    # ``run_score``'s lazy decode of the distorted output finds the
    # encoded file still on disk. Cleaned at function exit.
    workdir = Path(tempfile.mkdtemp(prefix="vmaf-tune-predict-"))

    def _real_encode_and_score(shot: Shot, crf: int, codec: str) -> tuple[Path, float]:
        """Run the actual encode + libvmaf score for one validation shot.

        Workflow: extract the shot range from ``args.source`` to a raw
        YUV reference, encode that reference at the predictor-picked CRF
        via :func:`encode.run_encode`, score with
        :func:`score.run_score` (which handles the distorted-side
        decode internally), and return ``(encoded_path, vmaf_score)``.
        """
        ref_yuv = workdir / f"ref_{shot.start_frame}_{shot.end_frame}.yuv"
        dist_path = workdir / f"dist_{shot.start_frame}_{shot.end_frame}.mp4"

        if fps > 0.0:
            ss_arg = f"{shot.start_frame / fps:.6f}"
        else:
            ss_arg = str(shot.start_frame)
        extract_cmd = [
            args.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            ss_arg,
            "-i",
            str(args.source),
            "-frames:v",
            str(shot.length),
            "-pix_fmt",
            pix_fmt,
            "-f",
            "rawvideo",
            str(ref_yuv),
        ]
        completed = subprocess.run(extract_cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not ref_yuv.exists():
            return dist_path, float("nan")

        encode_req = EncodeRequest(
            source=ref_yuv,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            framerate=fps if fps > 0.0 else 24.0,
            encoder=codec,
            preset="medium",
            crf=crf,
            output=dist_path,
        )
        encode_result = run_encode(encode_req, ffmpeg_bin=args.ffmpeg_bin)
        if encode_result.exit_status != 0 or not dist_path.exists():
            return dist_path, float("nan")

        score_req = ScoreRequest(
            reference=ref_yuv,
            distorted=dist_path,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
        )
        score_result = run_score(score_req, ffmpeg_bin=args.ffmpeg_bin)
        return dist_path, float(score_result.vmaf_score)

    try:
        report = validate_predictor(
            predictor=predictor,
            shots=shots,
            target_vmaf=args.target_vmaf,
            codec=args.codec,
            feature_extractor=_features,
            real_encode_and_score=_real_encode_and_score,
            k=args.validate_k,
            residual_threshold_vmaf=args.residual_threshold,
        )
    finally:
        # Clean the per-run scratch dir even on interrupt — the encoded
        # distorted files can run to gigabytes for long shots.
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    payload = {
        "verdict": report.verdict.value,
        "target_vmaf": report.target_vmaf,
        "residual_threshold": report.threshold_vmaf,
        "max_abs_residual": report.max_abs_residual,
        "mean_residual": report.mean_residual,
        "bias_correction": report.bias_correction,
        "k_validated": len(report.residuals),
        "residuals": [
            {
                "shot_start": r.shot.start_frame,
                "shot_end": r.shot.end_frame,
                "crf": r.crf_picked,
                "predicted_vmaf": r.predicted_vmaf,
                "measured_vmaf": r.measured_vmaf,
                "residual": r.residual,
            }
            for r in report.residuals
        ],
    }
    rendered = json.dumps(payload, indent=2)
    if args.report_out is not None:
        args.report_out.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    return 0 if report.verdict != Verdict.FALL_BACK else 2


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
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "corpus":
        return _run_corpus(args)
    if args.cmd == "recommend":
        return _run_recommend(args)
    if args.cmd == "predict":
        return _run_predict(args)
    if args.cmd == "tune-per-shot":
        return _run_tune_per_shot(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
