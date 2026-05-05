# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Saliency mask materialiser — Option C scaffold.

Given a reference YUV and a saliency ONNX model, produce a distorted
YUV variant where low-saliency pixels are replaced by the reference's
pixel content. This is the cheap way to make the existing ``vmaf``
binary score "the salient regions only" — every non-salient pixel
scores as a perfect match, so the pooled VMAF ends up dominated by the
saliency-weighted region.

This file declares the interface and a numpy-free synthetic helper.
Real ONNX-Runtime inference is gated on the ``[runtime]`` optional dep
and lazy-imported inside :func:`apply_saliency_mask` so that core unit
tests can run without ``onnxruntime`` installed.

Compose-mode (per-frame mask) is intentionally simple — a hard
threshold with a fade band so the mask boundary is not a sharp step
that would itself contribute to VMAF's edge-sensitive features.
Documented in ``docs/usage/vmaf-roi-score.md`` and in research-0063.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable


@dataclasses.dataclass(frozen=True)
class MaskRequest:
    """Inputs for a saliency-mask materialisation pass."""

    reference: Path
    distorted: Path
    output: Path
    width: int
    height: int
    pix_fmt: str
    saliency_model: Path
    threshold: float = 0.3
    fade: float = 0.1


def apply_saliency_mask(
    req: MaskRequest,
    *,
    inference: Callable[[bytes, int, int], "object"] | None = None,
) -> Path:
    """Materialise the saliency-masked distorted YUV.

    The contract:

    - reads the reference + distorted YUVs frame-by-frame;
    - runs the saliency ONNX on the **reference** so the mask is
      content-derived, not artefact-derived;
    - blends the distorted Y plane towards the reference Y plane
      outside the salient region;
    - writes the output YUV.

    ``inference`` is the per-frame callable that takes (rgb_bytes,
    width, height) and returns a height x width float mask in
    ``[0, 1]``. Defaults to a lazy-loaded ONNX Runtime session against
    ``req.saliency_model``. The seam exists so unit tests can inject a
    synthetic mask without ORT.

    Returns ``req.output``. Raises ``RuntimeError`` if onnxruntime is
    required but not importable.

    **Status**: Option C scaffold — interface + deterministic synthetic
    helper ship in the first PR. The YUV reader/writer + ORT
    pre/post-proc loop is a follow-up under T6-2c (see
    ADR-0288 §Implementation phasing).
    """
    if inference is None:
        inference = _lazy_onnx_inference(req.saliency_model)

    raise NotImplementedError(
        "vmaf-roi-score mask materialisation is scaffolded but not yet "
        "wired to the YUV I/O loop. Use --synthetic-mask in tests; "
        "see ADR-0288 for the phased rollout."
    )


def synthesise_uniform_mask(
    width: int,
    height: int,
    fill: float = 0.5,
) -> "list[list[float]]":
    """Return a constant-value mask without numpy.

    Used by the smoke tests to verify the combine-math without pulling
    onnxruntime / numpy into the test path.
    """
    if not (0.0 <= fill <= 1.0):
        raise ValueError(f"fill must be in [0, 1], got {fill!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    return [[float(fill)] * int(width) for _ in range(int(height))]


def _lazy_onnx_inference(model_path: Path) -> Callable[[bytes, int, int], "object"]:
    """Return a per-frame inference callable backed by ORT.

    Imported lazily so the package loads when ``onnxruntime`` is not
    installed (e.g. in the unit-test environment that only pulls the
    pure-Python combine math).
    """
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised by integration tests
        raise RuntimeError(
            "vmaf-roi-score requires onnxruntime for saliency inference; "
            "install with `pip install vmaf-roi-score[runtime]`"
        ) from exc

    # Construct the session eagerly so model-load errors surface here
    # rather than on first frame; the callable closure keeps the session
    # alive for follow-up T6-2c when the real per-frame inference lands.
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def _run(rgb_bytes: bytes, width: int, height: int) -> object:
        # Real preproc/postproc deferred — see apply_saliency_mask docstring.
        # Reference `session` so the closure keeps it live; the bound
        # name itself is discarded by the explicit `del`.
        _ = session
        del rgb_bytes, width, height
        raise NotImplementedError(
            "ORT inference seam wired but YUV->RGB preproc deferred to T6-2c."
        )

    return _run
