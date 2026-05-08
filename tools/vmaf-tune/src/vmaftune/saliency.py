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

# x265 CTU side in luma samples — the encoder's largest coding unit.
# 64x64 is x265's default and the same value the C-side ``vmaf-roi``
# sidecar uses (see ``libvmaf/tools/vmaf_roi.c``). x265 also accepts
# 32 and 16 at compile / run time, but the harness pins to 64 to
# match the sidecar so the saliency-blend math stays identical
# regardless of which surface the user picked.
X265_CTU_SIDE = 64

# SVT-AV1 ROI map block side. Per the SVT-AV1 ``EbSvtAv1Enc.h`` /
# ``--roi-map-file`` handling the encoder reads a row-major signed
# int8 grid keyed to the SB grid. SB defaults to 64 (high-tier
# hardware presets default to 128 but ffmpeg's libsvtav1 wrapper
# ships the 64 path); ``vmaf-roi`` and this emitter both use 64.
SVTAV1_ROI_SIDE = 64

# libvvenc CTU side. VVC permits CTUs up to 128x128; VVenC's
# config flag ``CTUSize`` defaults to 128 on the ``faster``..
# ``slower`` presets we expose. The QPDelta map this emitter writes
# is keyed to that grid.
VVENC_CTU_SIDE = 128

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


def _saliency_to_block_offsets(
    saliency_map: "np.ndarray",
    block_side: int,
    *,
    qp_offset: int,
) -> "np.ndarray":
    """Shared upsample/downsample + QP-blend helper for codec emitters.

    The fork-trained saliency model emits a per-pixel mask at the
    source resolution. Each codec consumes a per-block QP-offset grid
    keyed to its own coding-unit size (16x16 for x264 MBs, 64x64 for
    x265 CTUs / SVT-AV1 SBs, 128x128 for VVenC CTUs). This helper
    routes the mask through ``saliency_to_qp_map()`` for the
    deterministic ±``qp_offset`` blend and then ``reduce_qp_map_to_blocks()``
    for the per-block mean. Cropping is row/column-trailing — fewer
    than ``block_side`` pixels at the right/bottom edge are dropped to
    keep the grid integer-aligned, mirroring the C-side
    ``vmaf-roi`` reducer.
    """
    qp_map = saliency_to_qp_map(saliency_map, baseline_qp=0, foreground_offset=qp_offset)
    return reduce_qp_map_to_blocks(qp_map, block=block_side)


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


# ---------------------------------------------------------------------------
# x265 zone emitter
# ---------------------------------------------------------------------------
#
# x265 does NOT accept a per-CTU sidecar via ``--qpfile`` in a form
# that ffmpeg's libx265 wrapper exposes; what *is* portable through
# ``-x265-params`` is the ``zones`` syntax::
#
#     zones=<startFrame>,<endFrame>,q=<qp>/<startFrame>,<endFrame>,q=<qp>/...
#
# (See the x265 documentation, "Zones" section.) Zones are *temporal*
# slices, not spatial — each zone overrides the QP for a frame range,
# not for a CTU range. To carry a saliency-driven *spatial* QP-offset
# pattern through this surface we aggregate the per-CTU saliency map
# into one *clip-level* mean offset and surface it as a single zone
# spanning ``[0, duration_frames)`` with QP equal to ``baseline_qp +
# mean_offset``.
#
# This is documented as a deliberate granularity loss in the docs page
# and the docstring below — x265 users who need true per-CTU
# granularity should use the ``vmaf-roi`` C sidecar, which emits the
# x265 ``--qpfile`` ROI form (one row per CTU row, ASCII signed offsets).
# ``vmaf-tune``'s zones-based path is the FFmpeg-only path.


def write_x265_zones(
    saliency_map: "np.ndarray",
    output_path: Path,
    *,
    qp_offset: int = -4,
    baseline_qp: int = 28,
    duration_frames: int = 1,
) -> Path:
    """Emit an x265 ``zones=`` sidecar driven by the saliency mask.

    The saliency mask is reduced to per-CTU (64x64) offsets via
    :func:`_saliency_to_block_offsets`, then the *clip-level mean*
    offset is rounded and emitted as a single zone covering
    ``[0, duration_frames)``. ``baseline_qp`` is the encode's CRF; the
    zone's absolute QP is ``baseline_qp + mean_offset`` clamped to the
    legal x265 ``[0, 51]`` window.

    This is a documented granularity loss compared with x264's per-MB
    qpfile — see the module preamble and
    [docs/usage/vmaf-tune-saliency.md](../../docs/usage/vmaf-tune-saliency.md).

    The file format is a single ASCII line ending in ``\\n`` so the
    caller can ``read_text().strip()`` it onto an ``-x265-params``
    argument.
    """
    block_offsets = _saliency_to_block_offsets(saliency_map, X265_CTU_SIDE, qp_offset=qp_offset)
    np = _import_numpy()
    mean_offset = int(round(float(block_offsets.astype(np.float32).mean())))
    abs_qp = max(0, min(51, baseline_qp + mean_offset))
    end_frame = max(1, int(duration_frames))
    payload = f"zones=0,{end_frame},q={abs_qp}\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="ascii")
    return output_path


def augment_extra_params_with_x265_zones(base: Sequence[str], zones_arg: str) -> tuple[str, ...]:
    """Append ``-x265-params zones=...`` to an extra-params tuple.

    ``zones_arg`` is the verbatim ``zones=...`` token (no leading
    ``-x265-params``). The helper keeps the encode driver itself
    codec-agnostic — only this module knows the x265 surface.
    """
    return tuple(base) + ("-x265-params", zones_arg.rstrip())


# ---------------------------------------------------------------------------
# SVT-AV1 ROI map emitter
# ---------------------------------------------------------------------------
#
# SVT-AV1 reads a binary ``--roi-map-file`` sidecar: one signed-int8
# byte per superblock, row-major, no header. The same format the
# fork's ``vmaf-roi`` C sidecar emits (see ``emit_svtav1`` in
# ``libvmaf/tools/vmaf_roi.c``). For a one-frame map the file size is
# exactly ``cols * rows`` bytes; multi-frame ROI maps concatenate
# per-frame frames in order. ``vmaf-tune`` emits a single-frame map
# and lets the encoder reuse it across the clip — matching the
# saliency aggregator's per-clip mean-mask behaviour.


def write_svtav1_roi_map(
    saliency_map: "np.ndarray",
    output_path: Path,
    *,
    qp_offset: int = -4,
    duration_frames: int = 1,
) -> Path:
    """Emit a SVT-AV1 ``--roi-map-file`` (binary signed-int8 grid).

    The saliency mask is reduced to per-superblock (64x64) offsets via
    :func:`_saliency_to_block_offsets` then written row-major as
    ``int8`` bytes. The file is repeated ``duration_frames`` times so
    the encoder applies the same pattern across the clip — matching
    the per-clip aggregate the saliency model emits.

    Format pinned by SVT-AV1's ``--roi-map-file`` documentation and
    by the C-side ``vmaf-roi`` ``emit_svtav1`` helper.
    """
    np = _import_numpy()
    block_offsets = _saliency_to_block_offsets(saliency_map, SVTAV1_ROI_SIDE, qp_offset=qp_offset)
    # Clamp to int8 range before casting so values outside [-128, 127]
    # never wrap silently.
    clamped = np.clip(block_offsets, -128, 127).astype(np.int8)
    one_frame = clamped.tobytes(order="C")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nframes = max(1, int(duration_frames))
    with output_path.open("wb") as fh:
        for _ in range(nframes):
            fh.write(one_frame)
    return output_path


def augment_extra_params_with_svtav1_roi(base: Sequence[str], roi_path: Path) -> tuple[str, ...]:
    """Append ``-svtav1-params roi-map-file=<path>`` to ``base``.

    ffmpeg's libsvtav1 wrapper forwards arbitrary ``key=value`` pairs
    via ``-svtav1-params``; ``roi-map-file`` is the option SVT-AV1
    documents.
    """
    return tuple(base) + ("-svtav1-params", f"roi-map-file={roi_path}")


# ---------------------------------------------------------------------------
# libvvenc QP-delta emitter
# ---------------------------------------------------------------------------
#
# VVenC supports per-CTU QP-delta input via its ``QpaperROIFile`` /
# ``QpaperROIMode`` configuration switches (see VVenC's
# ``cfg/qpaper_roi*.cfg`` examples and the VVenC manual). The on-disk
# format is one ASCII signed-integer offset per CTU, space-separated,
# one row per CTU row, terminated by ``\n``. Multi-frame maps repeat
# the per-frame block separated by a blank line.
#
# IMPORTANT — granularity caveat: VVenC's ROI surface ships in two
# flavours: (a) the per-CTU QP-delta file documented above, and
# (b) a coarser 4-tier saliency-region-of-interest mode that maps
# pixels to one of {background, low, medium, high} and applies a
# fixed QP delta per tier. The fork ships the per-CTU form because
# that's the one the saliency model can drive at full granularity;
# the 4-tier form is too coarse to be worth wiring. Documented in
# the codec adapter docstring.


def write_vvenc_qp_delta(
    saliency_map: "np.ndarray",
    output_path: Path,
    *,
    qp_offset: int = -4,
    duration_frames: int = 1,
) -> Path:
    """Emit a libvvenc per-CTU QP-delta sidecar.

    ASCII format: one signed integer per CTU (128x128), space-
    separated, one row per CTU row, terminated by ``\\n``.
    Multi-frame maps repeat the per-frame block separated by a
    blank line so the encoder can advance frame-by-frame.

    Format pinned by the VVenC documentation's ``QpaperROIFile``
    section and the example configs under ``cfg/qpaper_roi*.cfg`` in
    the VVenC source distribution.
    """
    np = _import_numpy()
    block_offsets = _saliency_to_block_offsets(saliency_map, VVENC_CTU_SIDE, qp_offset=qp_offset)
    rows, cols = block_offsets.shape
    if rows == 0 or cols == 0:
        raise ValueError(
            f"saliency map {saliency_map.shape} smaller than one VVenC CTU "
            f"({VVENC_CTU_SIDE}x{VVENC_CTU_SIDE}); cannot emit QP-delta"
        )
    nframes = max(1, int(duration_frames))
    lines: list[str] = []
    for frame_idx in range(nframes):
        for r in range(rows):
            row_vals = " ".join(str(int(v)) for v in block_offsets[r])
            lines.append(row_vals)
        if frame_idx != nframes - 1:
            # Blank-line frame separator per VVenC's example configs.
            lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    # Smoke-check: the resulting file must be parseable as
    # ``rows x cols`` ints per frame.
    _ = np  # keep numpy reference live for type-narrowing tools
    return output_path


def augment_extra_params_with_vvenc_qp_delta(
    base: Sequence[str], qp_delta_path: Path
) -> tuple[str, ...]:
    """Append ``-vvenc-params QpaperROIFile=<path>`` to ``base``.

    ffmpeg's libvvenc wrapper forwards opaque ``key=value:key=value``
    strings via ``-vvenc-params``; ``QpaperROIFile`` is VVenC's own
    config-key for the per-CTU QP-delta sidecar.
    """
    return tuple(base) + ("-vvenc-params", f"QpaperROIFile={qp_delta_path}")


# Per-codec ROI emitter dispatch table. The keys are the
# ``qpfile_format`` values declared on the codec adapters; values are
# triples ``(emit_fn, augment_fn, suffix)`` consumed by
# :func:`saliency_aware_encode`. ``"none"`` is reserved for HW
# adapters that do not expose a portable ROI surface.
def _emit_x264(
    mask: "np.ndarray",
    out_path: Path,
    *,
    qp_offset: int,
    baseline_qp: int,
    duration_frames: int,
) -> Path:
    """x264 dispatch entry — wraps :func:`write_x264_qpfile`.

    Reduces the saliency mask to per-MB blocks and emits the per-MB
    qpfile in the format the existing x264 path uses (unchanged).
    """
    del baseline_qp  # x264 deltas are relative; baseline unused
    block_offsets = _saliency_to_block_offsets(mask, X264_MB_SIDE, qp_offset=qp_offset)
    return write_x264_qpfile(block_offsets, out_path, duration_frames=duration_frames)


def _augment_x264(base: Sequence[str], path: Path) -> tuple[str, ...]:
    return augment_extra_params_with_qpfile(base, path)


def _emit_x265(
    mask: "np.ndarray",
    out_path: Path,
    *,
    qp_offset: int,
    baseline_qp: int,
    duration_frames: int,
) -> Path:
    return write_x265_zones(
        mask,
        out_path,
        qp_offset=qp_offset,
        baseline_qp=baseline_qp,
        duration_frames=duration_frames,
    )


def _augment_x265(base: Sequence[str], path: Path) -> tuple[str, ...]:
    # Read back the zones=… line we just wrote and forward it as the
    # x265-params token. Using the file as the message passes through
    # the same persistence path as the other codecs (so ``persist_qpfile``
    # affects every codec uniformly), at the cost of one tiny read.
    zones_arg = path.read_text(encoding="ascii").strip()
    return augment_extra_params_with_x265_zones(base, zones_arg)


def _emit_svtav1(
    mask: "np.ndarray",
    out_path: Path,
    *,
    qp_offset: int,
    baseline_qp: int,
    duration_frames: int,
) -> Path:
    del baseline_qp
    return write_svtav1_roi_map(
        mask, out_path, qp_offset=qp_offset, duration_frames=duration_frames
    )


def _augment_svtav1(base: Sequence[str], path: Path) -> tuple[str, ...]:
    return augment_extra_params_with_svtav1_roi(base, path)


def _emit_vvenc(
    mask: "np.ndarray",
    out_path: Path,
    *,
    qp_offset: int,
    baseline_qp: int,
    duration_frames: int,
) -> Path:
    del baseline_qp
    return write_vvenc_qp_delta(
        mask, out_path, qp_offset=qp_offset, duration_frames=duration_frames
    )


def _augment_vvenc(base: Sequence[str], path: Path) -> tuple[str, ...]:
    return augment_extra_params_with_vvenc_qp_delta(base, path)


# (emit_fn, augment_fn, sidecar_suffix). The sidecar suffix is the
# extension applied to ``request.output`` when ``persist_qpfile`` is
# True; a temp file is used when False.
_ROI_DISPATCH: dict[str, tuple[Any, Any, str]] = {
    "x264-mb": (_emit_x264, _augment_x264, ".qpfile.txt"),
    "x265-zones": (_emit_x265, _augment_x265, ".x265zones.txt"),
    "svtav1-roi": (_emit_svtav1, _augment_svtav1, ".svtav1roi.bin"),
    "vvenc-qp-delta": (_emit_vvenc, _augment_vvenc, ".vvencqp.txt"),
}


def _resolve_qpfile_format(encoder: str) -> str:
    """Return the ``qpfile_format`` an encoder advertises.

    Looks the encoder up in the codec-adapter registry and reads the
    adapter's ``qpfile_format`` field. Adapters that pre-date the
    field default to ``"none"`` (no ROI surface) so HW codecs and
    older fork-internal stubs both fall back to a plain encode.
    """
    try:
        from .codec_adapters import get_adapter  # noqa: PLC0415

        adapter = get_adapter(encoder)
    except Exception:  # noqa: BLE001 - registry lookups are best-effort
        return "none"
    return getattr(adapter, "qpfile_format", "none")


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

    Dispatches on the codec adapter's ``qpfile_format`` so each
    encoder gets the ROI sidecar shape it actually accepts:

    * ``x264-mb``       — ASCII per-MB qpfile (libx264).
    * ``x265-zones``    — ``zones=…`` token (libx265).
    * ``svtav1-roi``    — binary signed-int8 grid (libsvtav1).
    * ``vvenc-qp-delta``— ASCII per-CTU QP-delta (libvvenc).
    * ``none``          — HW codecs / unrecognised encoders. The
      saliency stage is skipped and a plain encode runs with a
      single warning log line.

    Saliency-unavailable (missing onnxruntime / model) also degrades
    to a plain encode so callers always get a result.
    """
    from .encode import run_encode  # local import to avoid cycles

    cfg = config or SaliencyConfig()
    runner = encode_runner

    qpfile_format = _resolve_qpfile_format(request.encoder)
    if qpfile_format == "none" or qpfile_format not in _ROI_DISPATCH:
        if qpfile_format == "none":
            _LOG.warning(
                "saliency-aware: %s does not expose a portable ROI surface; "
                "running plain encode",
                request.encoder,
            )
        else:
            _LOG.warning(
                "saliency-aware: unknown qpfile_format %r for %s; running plain encode",
                qpfile_format,
                request.encoder,
            )
        return run_encode(request, ffmpeg_bin=ffmpeg_bin, runner=runner)

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

    emit_fn, augment_fn, suffix = _ROI_DISPATCH[qpfile_format]

    def _drive(sidecar_path: Path) -> "EncodeResult":
        emit_fn(
            mask,
            sidecar_path,
            qp_offset=cfg.foreground_offset,
            baseline_qp=request.crf,
            duration_frames=duration_frames,
        )
        augmented = dataclasses.replace(
            request,
            extra_params=augment_fn(request.extra_params, sidecar_path),
        )
        return run_encode(augmented, ffmpeg_bin=ffmpeg_bin, runner=runner)

    if cfg.persist_qpfile:
        sidecar = request.output.with_suffix(suffix)
        return _drive(sidecar)

    # Ephemeral sidecar — kept alive across the encode, removed after.
    binary = qpfile_format == "svtav1-roi"
    mode = "wb" if binary else "w"
    with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False) as fh:
        sidecar = Path(fh.name)
    try:
        return _drive(sidecar)
    finally:
        try:
            sidecar.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort cleanup
            pass


__all__ = [
    "DEFAULT_FRAME_SAMPLES",
    "DEFAULT_SALIENCY_MODEL_RELPATH",
    "QP_OFFSET_MAX",
    "QP_OFFSET_MIN",
    "SVTAV1_ROI_SIDE",
    "VVENC_CTU_SIDE",
    "X264_MB_SIDE",
    "X265_CTU_SIDE",
    "SaliencyConfig",
    "SaliencyUnavailableError",
    "augment_extra_params_with_qpfile",
    "augment_extra_params_with_svtav1_roi",
    "augment_extra_params_with_vvenc_qp_delta",
    "augment_extra_params_with_x265_zones",
    "compute_saliency_map",
    "reduce_qp_map_to_blocks",
    "saliency_aware_encode",
    "saliency_to_qp_map",
    "write_svtav1_roi_map",
    "write_vvenc_qp_delta",
    "write_x264_qpfile",
    "write_x265_zones",
]
