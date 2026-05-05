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


def test_known_codecs_includes_x264():
    # Phase A wires libx264 first; later adapters (libaom-av1, ...) join
    # the registry without disturbing the search-loop contract. Assert
    # membership rather than exact tuple identity so adding a codec is
    # not a registry-test churn event.
    assert "libx264" in known_codecs()


def test_known_codecs_includes_x264_and_x265():
    # Phase A wired x264; ADR-0288 added x265. Further codecs append
    # to this tuple — keep the assertion membership-based so it
    # doesn't bit-rot on every new adapter.
    codecs = known_codecs()
    assert "libx264" in codecs
    assert "libx265" in codecs


def test_known_codecs_phase_a_includes_x264_and_nvenc():
    # Phase A wires libx264 plus the NVENC hardware family.
    codecs = known_codecs()
    assert "libx264" in codecs
    assert "h264_nvenc" in codecs
    assert "hevc_nvenc" in codecs
    assert "av1_nvenc" in codecs
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
    # Schema-shape contract — Phase B / C will rely on this.
    assert SCHEMA_VERSION == 1
    assert "vmaf_score" in CORPUS_ROW_KEYS
    assert "bitrate_kbps" in CORPUS_ROW_KEYS
    assert "encode_time_ms" in CORPUS_ROW_KEYS
    assert "run_id" in CORPUS_ROW_KEYS


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
        # The score path now calls ffmpeg first to decode mp4 -> raw
        # YUV (libvmaf CLI only reads raw), then vmaf. Distinguish by
        # the presence of `--output` (vmaf only).
        if "--output" not in cmd:
            # ffmpeg decode call: materialise an empty raw-yuv stub so
            # `dist_yuv.exists()` is true and the score branch proceeds.
            out_path = Path(cmd[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"\x00" * 4096)
            return _FakeCompleted(returncode=0, stderr="")
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


def test_score_decodes_mp4_distorted_to_raw_yuv(tmp_path: Path):
    """Score path must decode container -> raw YUV before vmaf.

    Regression test for the Phase-A bug where the corpus pipeline
    handed an .mp4 directly to libvmaf's CLI (which only reads raw
    YUV/Y4M), producing every row with vmaf_score=NaN.
    """
    from vmaftune.score import ScoreRequest, run_score

    ref = _make_yuv(tmp_path / "ref.yuv")
    dist_mp4 = tmp_path / "dist.mp4"
    dist_mp4.write_bytes(b"\x00" * 4096)

    seen_cmds: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        seen_cmds.append(list(cmd))
        if cmd[0] == "ffmpeg":
            # Simulate decode: write a non-empty raw-yuv at cmd[-1].
            Path(cmd[-1]).write_bytes(b"\x00" * 4096)
            return _FakeCompleted(returncode=0)
        # vmaf call: write the JSON the parser expects.
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 88.0}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0\n")

    res = run_score(
        ScoreRequest(
            reference=ref,
            distorted=dist_mp4,
            width=64,
            height=64,
            pix_fmt="yuv420p",
        ),
        runner=fake_run,
    )

    assert res.vmaf_score == 88.0
    assert res.exit_status == 0
    # Two subprocess calls: ffmpeg decode, then vmaf.
    assert len(seen_cmds) == 2
    assert seen_cmds[0][0] == "ffmpeg"
    assert seen_cmds[1][0] == "vmaf"
    # The vmaf invocation must point at the raw-yuv, not the mp4.
    dist_idx = seen_cmds[1].index("--distorted") + 1
    assert seen_cmds[1][dist_idx].endswith(".yuv")


def test_score_skips_decode_for_raw_yuv_distorted(tmp_path: Path):
    """Raw YUV/Y4M distorted must skip the decode step."""
    from vmaftune.score import ScoreRequest, run_score

    ref = _make_yuv(tmp_path / "ref.yuv")
    dist = _make_yuv(tmp_path / "dist.yuv")

    seen_cmds: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        seen_cmds.append(list(cmd))
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 99.5}}}))
        return _FakeCompleted(returncode=0, stderr="VMAF version: 3.0.0\n")

    res = run_score(
        ScoreRequest(
            reference=ref,
            distorted=dist,
            width=64,
            height=64,
            pix_fmt="yuv420p",
        ),
        runner=fake_run,
    )
    assert res.vmaf_score == 99.5
    # Single subprocess call: vmaf only, no ffmpeg decode.
    assert len(seen_cmds) == 1
    assert seen_cmds[0][0] == "vmaf"


def test_score_decode_failure_propagates_as_nan(tmp_path: Path):
    """ffmpeg decode failure must yield NaN row, not crash."""
    from vmaftune.score import ScoreRequest, run_score

    ref = _make_yuv(tmp_path / "ref.yuv")
    dist_mp4 = tmp_path / "broken.mp4"
    dist_mp4.write_bytes(b"\x00" * 4)  # too small / not a real mp4

    def fake_run(cmd, capture_output, text, check):
        # All ffmpeg attempts fail.
        return _FakeCompleted(returncode=1, stderr="ffmpeg: invalid input")

    res = run_score(
        ScoreRequest(
            reference=ref,
            distorted=dist_mp4,
            width=64,
            height=64,
            pix_fmt="yuv420p",
        ),
        runner=fake_run,
    )

    import math

    assert math.isnan(res.vmaf_score)
    assert res.exit_status != 0
    assert "ffmpeg-decode-failed" in res.vmaf_binary_version
