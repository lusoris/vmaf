# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Phase A smoke tests — mocks ffmpeg + vmaf so no binaries required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from vmaftune import CORPUS_ROW_KEYS, SCHEMA_VERSION  # noqa: E402
from vmaftune.codec_adapters import get_adapter, known_codecs  # noqa: E402
from vmaftune.corpus import CorpusJob, CorpusOptions, iter_rows, write_jsonl  # noqa: E402
from vmaftune.encode import (  # noqa: E402
    EncodeRequest,
    build_ffmpeg_command,
    iter_grid,
    parse_versions,
)
from vmaftune.score import ScoreRequest, build_vmaf_command, parse_vmaf_json  # noqa: E402


class _FakeCompleted:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_yuv(path: Path, nbytes: int = 1024) -> Path:
    path.write_bytes(b"\x80" * nbytes)
    return path


def test_known_codecs_phase_a_is_x264_only():
    assert known_codecs() == ("libx264",)
    a = get_adapter("libx264")
    assert a.encoder == "libx264"
    assert a.invert_quality is True


def test_x264_validate_rejects_bad_inputs():
    a = get_adapter("libx264")
    a.validate("medium", 23)
    with pytest.raises(ValueError):
        a.validate("nope", 23)
    with pytest.raises(ValueError):
        a.validate("medium", 100)


def test_iter_grid_is_deterministic_and_complete():
    cells = iter_grid(["fast", "medium"], [22, 28, 34])
    assert cells == [
        ("fast", 22),
        ("fast", 28),
        ("fast", 34),
        ("medium", 22),
        ("medium", 28),
        ("medium", 34),
    ]


def test_build_ffmpeg_command_shape():
    req = EncodeRequest(
        source=Path("ref.yuv"),
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        framerate=24.0,
        encoder="libx264",
        preset="medium",
        crf=23,
        output=Path("out.mp4"),
    )
    cmd = build_ffmpeg_command(req, ffmpeg_bin="ffmpeg")
    assert cmd[0] == "ffmpeg"
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "medium"
    assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "23"
    assert cmd[-1] == "out.mp4"


def test_build_vmaf_command_shape():
    req = ScoreRequest(
        reference=Path("ref.yuv"),
        distorted=Path("dist.mp4"),
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
    )
    cmd = build_vmaf_command(req, json_output=Path("v.json"), vmaf_bin="vmaf")
    assert cmd[0] == "vmaf"
    assert "--reference" in cmd
    assert "--pixel_format" in cmd
    assert cmd[cmd.index("--pixel_format") + 1] == "420"


def test_parse_versions_extracts_known_lines():
    stderr = "ffmpeg version 6.1.1 built with gcc\nx264 - core 164 r3107\n"
    ffm, enc = parse_versions(stderr)
    assert ffm == "6.1.1"
    assert enc == "libx264-164"


def test_parse_versions_returns_unknown_on_miss():
    assert parse_versions("nothing matches") == ("unknown", "unknown")


def test_parse_vmaf_json_modern_shape():
    payload = {"pooled_metrics": {"vmaf": {"mean": 91.42}}}
    assert parse_vmaf_json(payload) == pytest.approx(91.42)


def test_parse_vmaf_json_legacy_shape():
    assert parse_vmaf_json({"VMAF score": 88.0}) == pytest.approx(88.0)


def test_parse_vmaf_json_raises_on_missing():
    with pytest.raises(ValueError):
        parse_vmaf_json({})


def test_corpus_row_keys_match_init_contract():
    # Schema-shape contract — Phase B / C will rely on this. v2 added
    # ``clip_mode`` for the sample-clip mode (ADR-0301).
    assert SCHEMA_VERSION == 2
    assert "vmaf_score" in CORPUS_ROW_KEYS
    assert "bitrate_kbps" in CORPUS_ROW_KEYS
    assert "encode_time_ms" in CORPUS_ROW_KEYS
    assert "run_id" in CORPUS_ROW_KEYS
    assert "clip_mode" in CORPUS_ROW_KEYS


def test_smoke_corpus_end_to_end_with_mocks(tmp_path: Path):
    src = _make_yuv(tmp_path / "ref.yuv")

    def fake_encode_run(cmd, capture_output, text, check):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\x00" * 4096)  # encode-size proxy
        return _FakeCompleted(
            returncode=0,
            stderr="ffmpeg version 6.1.1\nx264 - core 164 r3107\n",
        )

    def fake_score_run(cmd, capture_output, text, check):
        # write the JSON the parser expects
        out_idx = cmd.index("--output") + 1
        out_path = Path(cmd[out_idx])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.5}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0-lusoris\n")

    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=2.0,
        cells=(("medium", 23), ("slow", 28)),
    )
    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        keep_encodes=False,
        src_sha256=False,
    )
    rows = list(iter_rows(job, opts, encode_runner=fake_encode_run, score_runner=fake_score_run))

    assert len(rows) == 2
    for r in rows:
        assert set(CORPUS_ROW_KEYS) == set(r.keys())
        assert r["encoder"] == "libx264"
        assert r["vmaf_score"] == pytest.approx(92.5)
        assert r["encode_size_bytes"] == 4096
        assert r["bitrate_kbps"] == pytest.approx(4096 * 8 / 1000 / 2.0)
        assert r["exit_status"] == 0
        assert r["schema_version"] == SCHEMA_VERSION

    # JSONL writer round-trip
    out = tmp_path / "out.jsonl"
    n = write_jsonl(rows, out)
    assert n == 2
    parsed = [json.loads(line) for line in out.read_text().splitlines()]
    assert parsed[0]["preset"] == "medium"
    assert parsed[1]["preset"] == "slow"


def test_build_ffmpeg_command_inserts_sample_clip_flags():
    req = EncodeRequest(
        source=Path("ref.yuv"),
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        framerate=24.0,
        encoder="libx264",
        preset="medium",
        crf=23,
        output=Path("out.mp4"),
        sample_clip_seconds=10.0,
        sample_clip_start_s=25.0,
    )
    cmd = build_ffmpeg_command(req, ffmpeg_bin="ffmpeg")
    # -ss / -t must appear *before* -i so FFmpeg input-side seeks the
    # raw YUV instead of decoding the full source first.
    i_pos = cmd.index("-i")
    ss_pos = cmd.index("-ss")
    t_pos = cmd.index("-t")
    assert ss_pos < i_pos
    assert t_pos < i_pos
    assert cmd[ss_pos + 1] == "25.0"
    assert cmd[t_pos + 1] == "10.0"


def test_build_ffmpeg_command_no_sample_clip_flags_when_off():
    req = EncodeRequest(
        source=Path("ref.yuv"),
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        framerate=24.0,
        encoder="libx264",
        preset="medium",
        crf=23,
        output=Path("out.mp4"),
    )
    cmd = build_ffmpeg_command(req, ffmpeg_bin="ffmpeg")
    assert "-ss" not in cmd
    # `-t` is the encode-side flag we insert; FFmpeg has no other usage
    # in this command so its absence is the no-clip signal.
    assert "-t" not in cmd


def test_build_vmaf_command_appends_frame_skip_and_count():
    req = ScoreRequest(
        reference=Path("ref.yuv"),
        distorted=Path("dist.mp4"),
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        frame_skip_ref=600,
        frame_cnt=240,
    )
    cmd = build_vmaf_command(req, json_output=Path("v.json"), vmaf_bin="vmaf")
    assert "--frame_skip_ref" in cmd
    assert cmd[cmd.index("--frame_skip_ref") + 1] == "600"
    assert "--frame_cnt" in cmd
    assert cmd[cmd.index("--frame_cnt") + 1] == "240"


def test_sample_clip_mode_tags_rows_and_passes_argv(tmp_path: Path):
    src = _make_yuv(tmp_path / "ref.yuv")
    captured: dict[str, list[str]] = {}

    def fake_encode_run(cmd, capture_output, text, check):
        captured["encode"] = list(cmd)
        Path(cmd[-1]).write_bytes(b"\x00" * 4096)
        return _FakeCompleted(
            returncode=0,
            stderr="ffmpeg version 6.1.1\nx264 - core 164 r3107\n",
        )

    def fake_score_run(cmd, capture_output, text, check):
        captured["score"] = list(cmd)
        out_idx = cmd.index("--output") + 1
        out_path = Path(cmd[out_idx])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.1}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0-lusoris\n")

    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=60.0,
        cells=(("medium", 23),),
    )
    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        keep_encodes=False,
        src_sha256=False,
        sample_clip_seconds=10.0,
    )
    rows = list(iter_rows(job, opts, encode_runner=fake_encode_run, score_runner=fake_score_run))

    assert len(rows) == 1
    assert rows[0]["clip_mode"] == "sample_10s"
    # Centre window of a 60s source: start = (60 - 10) / 2 = 25.0
    enc_cmd = captured["encode"]
    assert enc_cmd[enc_cmd.index("-ss") + 1] == "25.0"
    assert enc_cmd[enc_cmd.index("-t") + 1] == "10.0"
    # Score-side: 25 * 24 = 600 skip frames, 10 * 24 = 240 frames.
    score_cmd = captured["score"]
    assert score_cmd[score_cmd.index("--frame_skip_ref") + 1] == "600"
    assert score_cmd[score_cmd.index("--frame_cnt") + 1] == "240"
    # Bitrate is computed against the slice duration, not the source.
    assert rows[0]["bitrate_kbps"] == pytest.approx(4096 * 8 / 1000 / 10.0)


def test_sample_clip_mode_falls_back_to_full_when_source_too_short(tmp_path: Path):
    src = _make_yuv(tmp_path / "ref.yuv")

    def fake_encode_run(cmd, capture_output, text, check):
        Path(cmd[-1]).write_bytes(b"\x00" * 1024)
        return _FakeCompleted(returncode=0, stderr="ffmpeg version 6.1.1\n")

    def fake_score_run(cmd, capture_output, text, check):
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 88.0}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0\n")

    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=8.0,  # shorter than the 10s sample-clip request
        cells=(("medium", 23),),
    )
    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        src_sha256=False,
        sample_clip_seconds=10.0,
    )
    rows = list(iter_rows(job, opts, encode_runner=fake_encode_run, score_runner=fake_score_run))

    assert rows[0]["clip_mode"] == "full"


def test_default_full_clip_mode_tag(tmp_path: Path):
    src = _make_yuv(tmp_path / "ref.yuv")

    def fake_encode_run(cmd, capture_output, text, check):
        Path(cmd[-1]).write_bytes(b"\x00" * 1024)
        return _FakeCompleted(returncode=0, stderr="ffmpeg version 6.1.1\n")

    def fake_score_run(cmd, capture_output, text, check):
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 95.0}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0\n")

    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=2.0,
        cells=(("medium", 23),),
    )
    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        src_sha256=False,
    )
    rows = list(iter_rows(job, opts, encode_runner=fake_encode_run, score_runner=fake_score_run))
    assert rows[0]["clip_mode"] == "full"


def test_corpus_row_keys_includes_clip_mode():
    assert "clip_mode" in CORPUS_ROW_KEYS
    assert SCHEMA_VERSION == 2


def test_encode_failure_emits_row_with_skipped_score(tmp_path: Path):
    src = _make_yuv(tmp_path / "ref.yuv")

    def failing_encode(cmd, capture_output, text, check):
        return _FakeCompleted(returncode=1, stderr="x264 boom")

    def never_score(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("score must not run when encode failed")

    job = CorpusJob(
        source=src,
        width=64,
        height=64,
        pix_fmt="yuv420p",
        framerate=24.0,
        duration_s=2.0,
        cells=(("medium", 23),),
    )
    opts = CorpusOptions(
        output=tmp_path / "corpus.jsonl",
        encode_dir=tmp_path / "encodes",
        src_sha256=False,
    )
    rows = list(iter_rows(job, opts, encode_runner=failing_encode, score_runner=never_score))
    assert len(rows) == 1
    assert rows[0]["exit_status"] == 1
    assert rows[0]["vmaf_binary_version"] == "skipped"
