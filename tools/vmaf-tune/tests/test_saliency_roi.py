# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Codec-extension ROI tests for saliency.py.

Pins the on-disk format every emitter writes against the encoder's
documented surface (x265 ``zones=…``, SVT-AV1 binary signed-int8 grid,
VVenC ASCII per-CTU QP-delta), and asserts ``saliency_aware_encode``
dispatches to the right emitter per codec adapter.

The ONNX session is mocked via ``session_factory``; the encode runner
is mocked via ``encode_runner`` — no onnxruntime / ffmpeg install
required. Companion to
[ADR-0293](../../../docs/adr/0293-vmaf-tune-saliency-aware.md)
codec-extension amendment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Make src/ importable without an editable install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

np = pytest.importorskip("numpy")

from vmaftune import saliency  # noqa: E402
from vmaftune.codec_adapters import get_adapter  # noqa: E402
from vmaftune.encode import EncodeRequest  # noqa: E402
from vmaftune.saliency import (  # noqa: E402
    SVTAV1_ROI_SIDE,
    VVENC_CTU_SIDE,
    X265_CTU_SIDE,
    SaliencyConfig,
    saliency_aware_encode,
    write_svtav1_roi_map,
    write_vvenc_qp_delta,
    write_x265_zones,
)

# ---- helpers ---------------------------------------------------------------


def _write_yuv420p(path: Path, width: int, height: int, nframes: int) -> Path:
    """Write a tiny synthetic yuv420p clip with constant grey frames."""
    frame_bytes = width * height * 3 // 2
    path.write_bytes(b"\x80" * (frame_bytes * nframes))
    return path


class _FakeOnnxSession:
    """Stub ONNX session: emits a deterministic right-bright mask."""

    def __init__(self, h: int, w: int):
        self._h = h
        self._w = w

    def run(self, _outputs, feeds):
        del feeds
        mask = np.zeros((1, 1, self._h, self._w), dtype=np.float32)
        mask[..., self._w // 2 :] = 0.9
        mask[..., : self._w // 2] = 0.1
        return [mask]


def _session_factory_for(h: int, w: int):
    def _factory(_path):
        return _FakeOnnxSession(h, w)

    return _factory


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _half_split_mask(h: int, w: int) -> "np.ndarray":
    """Right-bright 4×4 saliency mask scaled to (h, w) for emitters."""
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, w // 2 :] = 1.0
    return mask


# ---- write_x265_zones -------------------------------------------------------


def test_write_x265_zones_roundtrip(tmp_path):
    """Feed a high-saliency mask, parse the emitted zones= line."""
    h, w = X265_CTU_SIDE * 2, X265_CTU_SIDE * 2
    mask = _half_split_mask(h, w)
    out = tmp_path / "x265_zones.txt"
    write_x265_zones(mask, out, qp_offset=-4, baseline_qp=28, duration_frames=120)

    text = out.read_text(encoding="ascii").strip()
    # Format: "zones=<start>,<end>,q=<qp>"
    assert text.startswith("zones=")
    payload = text[len("zones=") :]
    parts = payload.split(",")
    # One zone -> three comma-separated tokens.
    assert len(parts) == 3, f"expected one zone, got {parts!r}"
    start_frame = int(parts[0])
    end_frame = int(parts[1])
    assert start_frame == 0
    assert end_frame == 120
    assert parts[2].startswith("q=")
    abs_qp = int(parts[2][len("q=") :])
    # Mean-mask is 0.5 (half-bright), centred=0 -> mean offset 0,
    # so absolute QP ≈ baseline (clamped to [0, 51]).
    assert 0 <= abs_qp <= 51
    assert abs(abs_qp - 28) <= 4  # within the QP window


def test_write_x265_zones_clamps_to_legal_qp(tmp_path):
    """Even a fully-bright mask cannot push absolute QP outside [0, 51]."""
    h, w = X265_CTU_SIDE, X265_CTU_SIDE
    bright = np.ones((h, w), dtype=np.float32)
    out = tmp_path / "z.txt"
    # baseline near the upper edge + negative offset on bright -> still legal
    write_x265_zones(bright, out, qp_offset=-12, baseline_qp=50, duration_frames=10)
    text = out.read_text(encoding="ascii").strip()
    abs_qp = int(text.rsplit("=", 1)[1])
    assert 0 <= abs_qp <= 51


# ---- write_svtav1_roi_map ---------------------------------------------------


def test_write_svtav1_roi_map_binary_format(tmp_path):
    """Pin the byte layout: signed-int8 row-major, no header."""
    h, w = SVTAV1_ROI_SIDE * 2, SVTAV1_ROI_SIDE * 3  # 2×3 SB grid
    mask = _half_split_mask(h, w)
    out = tmp_path / "svtav1.bin"
    write_svtav1_roi_map(mask, out, qp_offset=-4, duration_frames=1)

    raw = out.read_bytes()
    # 2x3 grid = 6 bytes per frame, 1 frame = 6 bytes total.
    assert len(raw) == 2 * 3
    arr = np.frombuffer(raw, dtype=np.int8).reshape(2, 3)
    # Right SB column should have a more negative QP delta than left.
    assert arr[:, -1].mean() < arr[:, 0].mean()
    # All deltas inside the clamped window.
    assert arr.min() >= -12
    assert arr.max() <= 12


def test_write_svtav1_roi_map_repeats_per_frame(tmp_path):
    """`duration_frames=N` repeats the per-frame block N times."""
    h, w = SVTAV1_ROI_SIDE, SVTAV1_ROI_SIDE * 2  # 1×2 grid -> 2 bytes/frame
    mask = _half_split_mask(h, w)
    out = tmp_path / "svtav1.bin"
    write_svtav1_roi_map(mask, out, qp_offset=-4, duration_frames=5)
    raw = out.read_bytes()
    assert len(raw) == 1 * 2 * 5
    # Each 2-byte frame is identical (per-clip aggregate).
    frames = [raw[i : i + 2] for i in range(0, len(raw), 2)]
    assert all(f == frames[0] for f in frames)


# ---- write_vvenc_qp_delta ---------------------------------------------------


def test_write_vvenc_qp_delta_format(tmp_path):
    """Pin the ASCII format: ``rows`` lines of space-sep ints per frame."""
    h, w = VVENC_CTU_SIDE, VVENC_CTU_SIDE * 3  # 1×3 CTU grid
    mask = _half_split_mask(h, w)
    out = tmp_path / "vvenc.txt"
    write_vvenc_qp_delta(mask, out, qp_offset=-4, duration_frames=1)
    text = out.read_text(encoding="ascii")
    rows = [r for r in text.splitlines() if r != ""]
    assert len(rows) == 1
    cols = rows[0].split()
    assert len(cols) == 3
    deltas = [int(c) for c in cols]
    # Right CTU should be more negative than left (more saliency -> better quality).
    assert deltas[-1] < deltas[0]


def test_write_vvenc_qp_delta_multi_frame_uses_blank_separator(tmp_path):
    """Multi-frame: blank line separates per-frame blocks."""
    h, w = VVENC_CTU_SIDE, VVENC_CTU_SIDE * 2
    mask = _half_split_mask(h, w)
    out = tmp_path / "vvenc.txt"
    write_vvenc_qp_delta(mask, out, qp_offset=-4, duration_frames=3)
    text = out.read_text(encoding="ascii")
    # 3 frames × 1 row + 2 blank separators = 5 lines (+ trailing newline).
    lines = text.splitlines()
    blank_count = sum(1 for line in lines if line == "")
    assert blank_count == 2  # N-1 separators


def test_write_vvenc_qp_delta_rejects_undersized_mask(tmp_path):
    mask = np.zeros((4, 4), dtype=np.float32)
    out = tmp_path / "vvenc.txt"
    with pytest.raises(ValueError):
        write_vvenc_qp_delta(mask, out, qp_offset=-4)


# ---- adapter contract: every codec declares qpfile_format ------------------


def test_every_adapter_declares_qpfile_format():
    """No silent ``getattr`` fallback — every registered codec is explicit."""
    from vmaftune.codec_adapters import known_codecs

    for name in known_codecs():
        adapter = get_adapter(name)
        # Field is a non-empty string; if missing, getattr default would
        # pollute the dispatch table at runtime. Be strict in tests.
        assert hasattr(adapter, "qpfile_format"), f"{name}: missing qpfile_format"
        value = adapter.qpfile_format
        assert isinstance(value, str) and value, f"{name}: empty qpfile_format"
        assert value in {
            "x264-mb",
            "x265-zones",
            "svtav1-roi",
            "vvenc-qp-delta",
            "none",
        }, f"{name}: unknown qpfile_format {value!r}"


def test_supports_qpfile_consistent_with_format():
    """``supports_qpfile`` must agree with ``qpfile_format != 'none'``."""
    from vmaftune.codec_adapters import known_codecs

    for name in known_codecs():
        adapter = get_adapter(name)
        has_format = adapter.qpfile_format != "none"
        assert (
            adapter.supports_qpfile is has_format
        ), f"{name}: supports_qpfile={adapter.supports_qpfile} but format={adapter.qpfile_format!r}"


# ---- dispatcher -----------------------------------------------------------


def _make_request(tmp_path: Path, encoder: str, w: int, h: int, nframes: int) -> EncodeRequest:
    src = _write_yuv420p(tmp_path / "src.yuv", w, h, nframes)
    return EncodeRequest(
        source=src,
        width=w,
        height=h,
        pix_fmt="yuv420p",
        framerate=24.0,
        encoder=encoder,
        preset="medium" if encoder != "libsvtav1" else "medium",
        crf=28,
        output=tmp_path / "out.mp4",
    )


_DISPATCH_CASES = [
    ("libx264", "-x264-params", "qpfile="),
    ("libx265", "-x265-params", "zones="),
    ("libsvtav1", "-svtav1-params", "roi-map-file="),
    ("libvvenc", "-vvenc-params", "QpaperROIFile="),
]


@pytest.mark.parametrize(("encoder", "flag", "value_prefix"), _DISPATCH_CASES)
def test_dispatch_uses_correct_emitter_per_codec(tmp_path, encoder, flag, value_prefix):
    """Each codec's saliency-aware encode injects its own ``-CODEC-params``.

    Mock the encode runner to capture the argv; assert the right
    codec-specific flag + value-prefix are present.
    """
    # Pick a width/height comfortably above every emitter's block size
    # so the per-block reducer always has at least one block.
    w, h = 256, 256
    request = _make_request(tmp_path, encoder, w, h, nframes=4)
    fake_model = tmp_path / "saliency_student_v1.onnx"
    fake_model.write_bytes(b"\x00")

    captured: dict[str, list[str]] = {}

    def _fake_runner(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[-1]).write_bytes(b"\x00\x01")
        return _FakeCompleted(0, stderr="ffmpeg version 6.1.1")

    cfg = SaliencyConfig(foreground_offset=-4, frame_samples=2, persist_qpfile=True)
    result = saliency_aware_encode(
        request,
        duration_frames=4,
        model_path=fake_model,
        config=cfg,
        encode_runner=_fake_runner,
        session_factory=_session_factory_for(h, w),
    )
    assert result.exit_status == 0
    cmd = captured["cmd"]
    assert flag in cmd, f"{encoder}: {flag} missing from argv {cmd!r}"
    value_idx = cmd.index(flag) + 1
    assert cmd[value_idx].startswith(value_prefix), (
        f"{encoder}: expected value to start with {value_prefix!r}, got " f"{cmd[value_idx]!r}"
    )


def test_hw_codecs_fall_back_to_plain_encode_with_warning(tmp_path, caplog):
    """NVENC has no portable ROI surface — degrade and warn."""
    w, h = 64, 64
    request = _make_request(tmp_path, "h264_nvenc", w, h, nframes=2)
    fake_model = tmp_path / "saliency_student_v1.onnx"
    fake_model.write_bytes(b"\x00")

    captured: dict[str, list[str]] = {}

    def _fake_runner(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[-1]).write_bytes(b"\x00")
        return _FakeCompleted(0, stderr="ffmpeg version 6.1.1")

    with caplog.at_level(logging.WARNING, logger="vmaftune.saliency"):
        result = saliency_aware_encode(
            request,
            duration_frames=2,
            model_path=fake_model,
            encode_runner=_fake_runner,
            session_factory=_session_factory_for(h, w),
        )
    assert result.exit_status == 0
    cmd = captured["cmd"]
    # No ROI-flag injected for HW codecs.
    for flag in ("-x264-params", "-x265-params", "-svtav1-params", "-vvenc-params"):
        assert flag not in cmd, f"unexpected {flag} for h264_nvenc"
    # Warning emitted with a hint about the missing ROI surface.
    assert any(
        "does not expose" in rec.message or "ROI" in rec.message for rec in caplog.records
    ), [rec.message for rec in caplog.records]


# ---- public API surface -----------------------------------------------------


def test_public_api_exposes_codec_emitters():
    expected = {
        "write_x265_zones",
        "write_svtav1_roi_map",
        "write_vvenc_qp_delta",
        "augment_extra_params_with_x265_zones",
        "augment_extra_params_with_svtav1_roi",
        "augment_extra_params_with_vvenc_qp_delta",
        "X265_CTU_SIDE",
        "SVTAV1_ROI_SIDE",
        "VVENC_CTU_SIDE",
    }
    assert expected.issubset(set(saliency.__all__))
