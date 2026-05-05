# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Backend selection for the libvmaf CLI used by `vmaf-tune`.

`vmaf` exposes a unified ``--backend NAME`` selector
(values: ``auto|cpu|cuda|sycl|vulkan``) per ADR-0127 / ADR-0175.
The selector engages the GPU dispatch in libvmaf and gives a
~10-30x speedup on the score axis at 1080p relative to the CPU path.

This module turns user intent (``--score-backend cuda|vulkan|sycl|cpu|auto``)
into a concrete, validated choice by intersecting:

1. What the **vmaf binary** advertises in its ``--help`` output (the
   `--backend` line lists which values are recognised);
2. What the **host hardware / runtime** actually offers, probed via
   cheap external tools (``nvidia-smi``, ``vulkaninfo``,
   ``sycl-ls``) with conservative fallbacks.

Hard rules (per task spec):

- ``--score-backend cuda`` on a host without CUDA must FAIL with a
  clear error. We never silently fall back when the user explicitly
  requested a backend.
- Only ``auto`` walks the fallback chain. The default chain is
  ``cuda -> vulkan -> sycl -> cpu``, picking the first that is both
  binary-supported and hardware-available.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from collections.abc import Sequence

#: Backends the vmaf CLI accepts via ``--backend NAME``.
ALL_BACKENDS: tuple[str, ...] = ("cpu", "cuda", "sycl", "vulkan")

#: Default fallback chain for ``auto``. CUDA first because it is the
#: most-tuned GPU backend on this fork; CPU last as the always-available
#: floor.
DEFAULT_FALLBACKS: tuple[str, ...] = ("cuda", "vulkan", "sycl", "cpu")


class BackendUnavailableError(RuntimeError):
    """User explicitly requested a backend the host cannot provide.

    Raised when ``select_backend(prefer=X)`` is called with a
    non-``auto`` ``X`` and either the local ``vmaf`` binary lacks
    ``X`` support or the host hardware does not advertise it.
    Never raised by ``auto``-mode selection — that path falls back.
    """


@dataclasses.dataclass(frozen=True)
class BackendProbe:
    """One probe outcome (per backend) used by `detect_available_backends`."""

    name: str
    binary_supports: bool
    hardware_available: bool

    @property
    def usable(self) -> bool:
        return self.binary_supports and self.hardware_available


def _vmaf_help(vmaf_bin: str, runner: object | None = None) -> str:
    """Return the vmaf ``--help`` output (stderr+stdout joined).

    Returns empty string on any error so probe logic degrades to
    "binary doesn't support GPU backends" rather than raising.
    """
    runner_fn = runner or subprocess.run
    if shutil.which(vmaf_bin) is None and "/" not in vmaf_bin:
        return ""
    try:
        completed = runner_fn(  # type: ignore[operator]
            [vmaf_bin, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    out = getattr(completed, "stdout", "") or ""
    err = getattr(completed, "stderr", "") or ""
    return f"{out}\n{err}"


def parse_supported_backends(help_text: str) -> frozenset[str]:
    """Extract the backends the vmaf binary advertises from `--help`.

    The fork's CLI prints a line like::

        --backend $name:              exclusive backend selector — auto|cpu|cuda|sycl|vulkan.

    We parse the alternation (``a|b|c``) and intersect it with
    `ALL_BACKENDS`. ``cpu`` is added unconditionally — every build
    has a CPU path, even if the help line is missing.

    Returns a frozenset for cheap membership tests.
    """
    found: set[str] = {"cpu"}
    for backend in ALL_BACKENDS:
        # Look for the exact token surrounded by | or whitespace as
        # a robust check; matches any of: auto|cpu|cuda|sycl|vulkan
        # without false-positives on substrings (e.g. "cuda" inside
        # a comment about CUDA).
        for needle in (f"|{backend}|", f"|{backend}.", f"|{backend}\n", f"|{backend} "):
            if needle in help_text:
                found.add(backend)
                break
    return frozenset(found)


def _probe_cuda(runner: object | None = None) -> bool:
    """True if a CUDA device is reachable. Tries `nvidia-smi -L`."""
    if shutil.which("nvidia-smi") is None:
        return False
    runner_fn = runner or subprocess.run
    try:
        completed = runner_fn(  # type: ignore[operator]
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    rc = int(getattr(completed, "returncode", 1))
    out = getattr(completed, "stdout", "") or ""
    return rc == 0 and "GPU" in out


def _probe_vulkan(runner: object | None = None) -> bool:
    """True if a Vulkan device is reachable. Tries `vulkaninfo --summary`."""
    if shutil.which("vulkaninfo") is None:
        return False
    runner_fn = runner or subprocess.run
    try:
        completed = runner_fn(  # type: ignore[operator]
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    rc = int(getattr(completed, "returncode", 1))
    out = getattr(completed, "stdout", "") or ""
    # vulkaninfo prints "deviceName" entries per detected GPU.
    return rc == 0 and "deviceName" in out


def _probe_sycl(runner: object | None = None) -> bool:
    """True if a SYCL device is reachable. Tries `sycl-ls`."""
    if shutil.which("sycl-ls") is None:
        return False
    runner_fn = runner or subprocess.run
    try:
        completed = runner_fn(  # type: ignore[operator]
            ["sycl-ls"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    rc = int(getattr(completed, "returncode", 1))
    out = getattr(completed, "stdout", "") or ""
    # sycl-ls prints one line per device, prefixed with bracketed
    # backend tokens: "[opencl:gpu]", "[ext_oneapi_level_zero:gpu]", ...
    return rc == 0 and "[" in out and ":gpu" in out.lower()


def detect_available_backends(
    *,
    vmaf_bin: str = "vmaf",
    runner: object | None = None,
) -> list[str]:
    """Return backends usable on this host, in `ALL_BACKENDS` order.

    "Usable" means both:
      - the local ``vmaf`` binary advertises ``--backend NAME`` support, and
      - the corresponding hardware/runtime probe succeeds.

    CPU is always present (every libvmaf build has a CPU path).
    """
    help_text = _vmaf_help(vmaf_bin, runner=runner)
    supported = parse_supported_backends(help_text)

    probes = {
        "cpu": True,
        "cuda": _probe_cuda(runner=runner) if "cuda" in supported else False,
        "vulkan": _probe_vulkan(runner=runner) if "vulkan" in supported else False,
        "sycl": _probe_sycl(runner=runner) if "sycl" in supported else False,
    }
    return [b for b in ALL_BACKENDS if b in supported and probes[b]]


def select_backend(
    prefer: str = "auto",
    *,
    fallbacks: Sequence[str] = DEFAULT_FALLBACKS,
    available: Sequence[str] | None = None,
    vmaf_bin: str = "vmaf",
    runner: object | None = None,
) -> str:
    """Pick a backend honouring user preference and host capability.

    - ``prefer="auto"`` walks ``fallbacks`` and returns the first
      entry present in ``available``. ``cpu`` must be in the chain
      (or in ``available``) to guarantee a result.
    - Any other ``prefer`` value (``cpu``, ``cuda``, ``sycl``,
      ``vulkan``) is honoured **strictly**: if it is not in
      ``available``, raise `BackendUnavailableError`. Never
      silently falls back — that would mask hardware/build mismatches
      and lie to the operator about wall-clock expectations.

    ``available`` defaults to ``detect_available_backends(...)``;
    tests inject a literal list to keep the unit boundary tight.
    """
    if prefer not in {"auto", *ALL_BACKENDS}:
        raise ValueError(
            f"unknown backend {prefer!r}; expected one of: " f"auto, {', '.join(ALL_BACKENDS)}"
        )

    if available is None:
        available = detect_available_backends(vmaf_bin=vmaf_bin, runner=runner)

    if prefer == "auto":
        for candidate in fallbacks:
            if candidate in available:
                return candidate
        # Last-ditch: cpu is universally available even if probes failed.
        return "cpu"

    if prefer in available:
        return prefer

    # Strict-mode failure — never silently downgrade.
    raise BackendUnavailableError(
        f"backend {prefer!r} requested but not available on this host "
        f"(available: {', '.join(available) or 'cpu'}). "
        f"Check that the local vmaf binary was built with the matching "
        f"backend support and the corresponding runtime/driver is installed."
    )
