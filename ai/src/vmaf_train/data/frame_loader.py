"""ffmpeg-decoded frame batches for NR / learned-filter training (C2, C3)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np


@dataclass(frozen=True)
class FrameSource:
    path: Path
    width: int
    height: int
    pix_fmt: str = "gray"


class _PopenLike(Protocol):
    stdout: BinaryIO

    def wait(self) -> int: ...


_PIX_FMT_CHANNELS: dict[str, int] = {
    "gray": 1,
    "rgb24": 3,
    "bgr24": 3,
    "rgba": 4,
    "bgra": 4,
}


def _frame_shape(source: FrameSource) -> tuple[int, ...]:
    channels = _PIX_FMT_CHANNELS.get(source.pix_fmt)
    if channels is None:
        supported = ", ".join(sorted(_PIX_FMT_CHANNELS))
        raise ValueError(f"unsupported pix_fmt={source.pix_fmt!r}; expected one of {supported}")
    if channels == 1:
        return (source.height, source.width)
    return (source.height, source.width, channels)


def iter_frames(
    source: FrameSource,
    ffmpeg: str = "ffmpeg",
    popen=subprocess.Popen,
) -> Iterator[np.ndarray]:
    """Yield ffmpeg-decoded frames as uint8 numpy arrays.

    ``gray`` yields ``HxW`` arrays. Packed colour formats
    (``rgb24``, ``bgr24``, ``rgba``, ``bgra``) yield ``HxWxC`` arrays.
    """
    shape = _frame_shape(source)
    frame_bytes = source.width * source.height * _PIX_FMT_CHANNELS[source.pix_fmt]
    proc: _PopenLike = popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            source.pix_fmt,
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                return
            yield np.frombuffer(buf, dtype=np.uint8).reshape(shape).copy()
    finally:
        proc.stdout.close()
        proc.wait()
