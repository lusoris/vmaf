# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Saliency-aware ROI tuning — Bucket #2 of the PR #354 audit.

Wires the fork's `saliency_student_v1` ONNX model
(`model/tiny/saliency_student_v1.onnx`, ~113K params, BSD-3-Clause-
Plus-Patent, ADR-0286) into the vmaf-tune harness so callers can hand
encoders a per-MB QP-offset map that biases bits toward salient regions
(faces, focal subjects) and saves bits on background.

Pipeline:

    raw YUV ── sample N frames ── YUV->RGB ── ImageNet-normalise
            ── onnxruntime(saliency_student_v1) ── mean saliency mask [0, 1]
            ── per-MB reduce ── linear map to QP offsets [-12, +12]
            ── x264 ASCII qpfile ── ffmpeg --qpfile

Design notes:

- The saliency model is loaded lazily and is **optional**: callers
  can graceful-fallback to non-saliency encoding when onnxruntime or
  the model file is unavailable. This matches the `vmaf-roi` C
  sidecar's posture (ADR-0247).
- Per-MB granularity is chosen to match x264's ``--qpfile`` natural
  granularity (16x16 luma) and SVT-AV1's ROI-map granularity (the
  only two encoders Phase A / Bucket #2 need).
- All numeric kernels (RGB conversion, ImageNet normalisation,
  per-MB reduce, QP-offset clamp) are pure NumPy so the test suite
  can exercise them without onnxruntime installed.
- The subprocess boundary stays the integration seam — encoding
  itself is delegated to ``encode.run_encode`` with augmented
  ``extra_params``.

ADR-0293; companion to ADR-0237 (parent) and ADR-0286
(saliency_student_v1, fork-trained on DUTS-TR).
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from .encode import EncodeRequest, EncodeResult

_LOG = logging.getLogger(__name__)

# Default location of the fork-trained student model. The harness
# resolves this path relative to the repo root when the caller does
# not pass an explicit ``model_path``.
DEFAULT_SALIENCY_MODEL_RELPATH = Path("model/tiny/saliency_student_v1.onnx")

# QP offsets are clamped to a window both x264 and x265 accept
# comfortably (per ADR-0247 sidecar convention).
QP_OFFSET_MIN = -12
QP_OFFSET_MAX = 12

# x264 macroblock side in luma samples — fixed by the codec.
X264_MB_SIDE = 16

# ImageNet mean/std the saliency_student_v1 input layer expects
# (matches the C-side `vmaf_tensor_from_rgb_imagenet` helper).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Default frame-sample count when reducing a clip to one aggregate
# saliency mask. Eight frames is the same default the saliency
# scoring extractor uses for per-clip aggregates.
DEFAULT_FRAME_SAMPLES = 8


class SaliencyUnavailableError(RuntimeError):
    """Raised when saliency inference is requested but onnxruntime or
    the model file is missing. Callers catch this to drop back to
    non-saliency encoding."""


@dataclasses.dataclass(frozen=True)
class SaliencyConfig:
    """User-tunable knobs for the saliency-aware encode path."""

    # Per-pixel QP offset applied at peak saliency (1.0). Negative
    # values lower the QP (better quality) in salient regions.
    foreground_offset: int = -4
    # Number of frames sampled across the clip for the aggregate
    # saliency mask. Higher = more stable, slower.
    frame_samples: int = DEFAULT_FRAME_SAMPLES
    # If True, write the qpfile to ``encode_dir`` next to the
    # encoded artifact (gitignored). If False, use a
    # ``tempfile.NamedTemporaryFile`` cleaned up by the OS.
    persist_qpfile: bool = False


def _import_numpy() -> Any:
    """Import numpy lazily so the module imports on systems without
    it (the grid-sweep corpus path doesn't need numpy)."""
    try:
        import numpy as np  # noqa: PLC0415  (deliberately lazy)
    except ImportError as exc:  # pragma: no cover - defensive
        raise SaliencyUnavailableError("numpy is required for saliency-aware encoding") from exc
    return np


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:
        raise SaliencyUnavailableError(
            "onnxruntime is required for saliency-aware encoding; "
            "install with `pip install onnxruntime`"
        ) from exc
    return ort


def _yuv420p_frame_size(width: int, height: int) -> int:
    """Return raw 8-bit yuv420p frame size in bytes."""
    return width * height * 3 // 2


def _read_yuv420p_y_plane(path: Path, frame_index: int, width: int, height: int) -> "np.ndarray":
    """Read the luma plane of one yuv420p frame as ``uint8 [H, W]``.

    Saliency inference uses RGB derived from luma replicated across
    channels — sufficient quality for foreground-vs-background
    discrimination and avoids a chroma upsample on the hot path.
    """
    np = _import_numpy()
    frame_bytes = _yuv420p_frame_size(width, height)
    luma_bytes = width * height
    with path.open("rb") as fh:
        fh.seek(frame_index * frame_bytes)
        buf = fh.read(luma_bytes)
    if len(buf) != luma_bytes:
        raise ValueError(
            f"short read on {path}: wanted {luma_bytes} bytes at "
            f"frame {frame_index}, got {len(buf)}"
        )
    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width)


def _luma_to_rgb_imagenet(luma: "np.ndarray") -> "np.ndarray":
    """Replicate luma into 3 channels and ImageNet-normalise.

    Returns ``float32 [1, 3, H, W]`` — the NCHW tensor
    `saliency_student_v1` expects.
    """
    np = _import_numpy()
    # uint8 [H, W] -> float32 [H, W] in [0, 1]
    f = luma.astype(np.float32) / 255.0
    # Replicate to 3 channels.
    rgb = np.stack([f, f, f], axis=0)  # [3, H, W]
    mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(_IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std
    return rgb[None, ...]  # [1, 3, H, W]


def _sample_frame_indices(total: int, n: int) -> list[int]:
    """Pick ``n`` evenly-spaced frame indices in ``[0, total)``."""
    if total <= 0:
        return []
    if n <= 1 or total == 1:
        return [0]
    step = max(1, total // n)
    idx = list(range(0, total, step))[:n]
    if not idx:
        idx = [0]
    return idx


def _frame_count(path: Path, width: int, height: int) -> int:
    """Number of yuv420p frames in ``path``."""
    fsize = path.stat().st_size
    return fsize // _yuv420p_frame_size(width, height)


def compute_saliency_map(
    video_path: Path,
    width: int,
    height: int,
    *,
    model_path: Path | None = None,
    frame_samples: int = DEFAULT_FRAME_SAMPLES,
    session_factory: Any = None,
) -> "np.ndarray":
    """Run ``saliency_student_v1`` over a sampled subset of frames.

    Returns a ``float32 [H, W]`` aggregate saliency mask in ``[0, 1]``
    — the per-pixel mean of the per-frame outputs.

    ``session_factory`` is the test seam: tests pass a fake that
    returns a stub session object exposing ``.run(...)``.

    Raises :class:`SaliencyUnavailableError` if onnxruntime or the
    model file cannot be loaded.
    """
    np = _import_numpy()

    if model_path is None:
        model_path = DEFAULT_SALIENCY_MODEL_RELPATH
    if not Path(model_path).exists():
        raise SaliencyUnavailableError(
            f"saliency model not found: {model_path} (Bucket #2 needs "
            "`saliency_student_v1.onnx` from PR #359 / ADR-0286)"
        )

    if session_factory is None:
        ort = _import_onnxruntime()
        session_factory = lambda p: ort.InferenceSession(  # noqa: E731
            str(p), providers=["CPUExecutionProvider"]
        )
    session = session_factory(model_path)

    nframes = _frame_count(video_path, width, height)
    if nframes <= 0:
        raise ValueError(f"no frames in {video_path} for {width}x{height} yuv420p")
    indices = _sample_frame_indices(nframes, frame_samples)

    accum = np.zeros((height, width), dtype=np.float32)
    for fi in indices:
        luma = _read_yuv420p_y_plane(video_path, fi, width, height)
        tensor = _luma_to_rgb_imagenet(luma)
        outputs = session.run(None, {"input": tensor})
        # saliency_student_v1 returns NCHW [1, 1, H, W] in [0, 1].
        mask = np.asarray(outputs[0]).reshape(height, width)
        accum += mask.astype(np.float32)

    accum /= float(len(indices))
    # Numerically pin to [0, 1] in case of FP drift on the boundary.
    return np.clip(accum, 0.0, 1.0)


def saliency_to_qp_map(
    mask: "np.ndarray",
    baseline_qp: int,
    foreground_offset: int = -4,
) -> "np.ndarray":
    """Map a saliency mask to a per-pixel QP offset map.

    Convention (matches ``vmaf-roi`` ADR-0247):

    - High saliency (1.0) -> ``foreground_offset`` (typically negative)
    - Low saliency (0.0)  -> ``-foreground_offset`` (background gets
      *higher* QP, saving bits)
    - Neutral (0.5)        -> 0

    Returns ``int32 [H, W]`` of QP offsets clamped to
    ``[QP_OFFSET_MIN, QP_OFFSET_MAX]``. ``baseline_qp`` is *not*
    folded in — the offsets are deltas the encoder applies on top of
    its own rate-control decision.
    """
    np = _import_numpy()
    # Map [0, 1] saliency to [-1, +1] linearly.
    centred = (np.asarray(mask, dtype=np.float32) * 2.0) - 1.0
    # Apply the user-chosen gain. ``foreground_offset`` is signed:
    # -4 means "subtract 4 QP at peak saliency". With ``centred`` in
    # [-1, +1], scaling by ``foreground_offset`` directly produces the
    # desired sign convention:
    #   sal=1.0 -> centred=+1 -> offset=foreground_offset (negative)
    #   sal=0.0 -> centred=-1 -> offset=-foreground_offset (positive)
    #   sal=0.5 -> centred= 0 -> offset=0
    offsets = centred * float(foreground_offset)
    # Clamp to the encoder-accepted band.
    offsets = np.clip(np.round(offsets), QP_OFFSET_MIN, QP_OFFSET_MAX)
    # Bound the API to a known dtype.
    return offsets.astype(np.int32)


def reduce_qp_map_to_blocks(qp_map: "np.ndarray", block: int = X264_MB_SIDE) -> "np.ndarray":
    """Reduce a per-pixel QP-offset map to per-block (mean-rounded).

    x264's ``--qpfile`` is per-MB (16x16). Block-mean keeps offsets
    locally smooth and avoids the encoder rejecting a too-noisy file.
    """
    np = _import_numpy()
    h, w = qp_map.shape
    bh = h // block
    bw = w // block
    if bh == 0 or bw == 0:
        raise ValueError(f"qp_map {h}x{w} smaller than block size {block}x{block}")
    cropped = qp_map[: bh * block, : bw * block]
    reshaped = cropped.reshape(bh, block, bw, block).astype(np.float32)
    means = reshaped.mean(axis=(1, 3))
    return np.clip(np.round(means), QP_OFFSET_MIN, QP_OFFSET_MAX).astype(np.int32)


def write_x264_qpfile(
    block_offsets: "np.ndarray",
    out_path: Path,
    *,
    duration_frames: int = 1,
) -> Path:
    """Write a x264 ``--qpfile`` for an N-frame clip.

    x264 qpfile format: one line per frame, ``frame_index frame_type
    qp``. We emit one frame_index entry per encode-frame so the
    encoder applies the same per-MB pattern across the full clip
    (the saliency mask is a per-clip aggregate). Per-MB granularity
    is delivered via the ``--qpfile`` extension x264 honours since
    r2390 — the 0..bw lines after the frame header carry the MB
    deltas.

    NOTE: this is the conservative fallback that targets the widely
    available x264 ``--qpfile`` syntax. Encoders that prefer a true
    per-MB ROI map (SVT-AV1, x265 ``--qpfile``) consume the same
    block_offsets via a sibling formatter — see ``vmaf-roi``.
    """
    bh, bw = block_offsets.shape
    lines: list[str] = []
    for frame_idx in range(duration_frames):
        # 'I' for the first frame so x264 anchors the GOP; 'P' for the
        # rest. Preset/tune still owns the actual GOP structure.
        kind = "I" if frame_idx == 0 else "P"
        lines.append(f"{frame_idx} {kind} 0")
        for row in range(bh):
            row_offsets = " ".join(str(int(v)) for v in block_offsets[row])
            lines.append(row_offsets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out_path


def augment_extra_params_with_qpfile(base: Sequence[str], qpfile: Path) -> tuple[str, ...]:
    """Return ``base + ('-x264-params', f'qpfile={qpfile}')``.

    Keeping this a pure helper means the encode driver itself stays
    agnostic of saliency.
    """
    return tuple(base) + ("-x264-params", f"qpfile={qpfile}")


def saliency_aware_encode(
    request: "EncodeRequest",
    *,
    duration_frames: int,
    model_path: Path | None = None,
    config: SaliencyConfig | None = None,
    encode_runner: Any = None,
    session_factory: Any = None,
    ffmpeg_bin: str = "ffmpeg",
) -> "EncodeResult":
    """Drive a single saliency-aware encode end-to-end.

    Steps:

    1. Run :func:`compute_saliency_map` over the source.
    2. Map -> per-MB QP offsets.
    3. Materialise an x264 qpfile.
    4. Augment ``request.extra_params`` with the qpfile arg.
    5. Delegate to :func:`encode.run_encode` (or the injected runner).

    Falls back to a plain encode (no qpfile) if saliency is
    unavailable, so callers always get a result.
    """
    from .encode import run_encode  # local import to avoid cycles

    cfg = config or SaliencyConfig()
    runner = encode_runner

    try:
        mask = compute_saliency_map(
            request.source,
            request.width,
            request.height,
            model_path=model_path,
            frame_samples=cfg.frame_samples,
            session_factory=session_factory,
        )
    except SaliencyUnavailableError as exc:
        _LOG.warning("saliency unavailable, falling back to plain encode: %s", exc)
        return run_encode(request, ffmpeg_bin=ffmpeg_bin, runner=runner)

    qp_map = saliency_to_qp_map(
        mask, baseline_qp=request.crf, foreground_offset=cfg.foreground_offset
    )
    block_offsets = reduce_qp_map_to_blocks(qp_map, block=X264_MB_SIDE)

    if cfg.persist_qpfile:
        qpfile = request.output.with_suffix(".qpfile.txt")
        write_x264_qpfile(block_offsets, qpfile, duration_frames=duration_frames)
        augmented = dataclasses.replace(
            request,
            extra_params=augment_extra_params_with_qpfile(request.extra_params, qpfile),
        )
        return run_encode(augmented, ffmpeg_bin=ffmpeg_bin, runner=runner)

    # Ephemeral file path — kept alive across the encode.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".qpfile.txt", delete=False) as fh:
        qpfile = Path(fh.name)
    try:
        write_x264_qpfile(block_offsets, qpfile, duration_frames=duration_frames)
        augmented = dataclasses.replace(
            request,
            extra_params=augment_extra_params_with_qpfile(request.extra_params, qpfile),
        )
        return run_encode(augmented, ffmpeg_bin=ffmpeg_bin, runner=runner)
    finally:
        try:
            qpfile.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort cleanup
            pass


__all__ = [
    "DEFAULT_FRAME_SAMPLES",
    "DEFAULT_SALIENCY_MODEL_RELPATH",
    "QP_OFFSET_MAX",
    "QP_OFFSET_MIN",
    "X264_MB_SIDE",
    "SaliencyConfig",
    "SaliencyUnavailableError",
    "augment_extra_params_with_qpfile",
    "compute_saliency_map",
    "reduce_qp_map_to_blocks",
    "saliency_aware_encode",
    "saliency_to_qp_map",
    "write_x264_qpfile",
]
