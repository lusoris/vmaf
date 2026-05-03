# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""libvvenc codec adapter — VVC / H.266 with optional NN-VC tools.

VVenC is Fraunhofer HHI's open-source VVC (H.266) encoder. VVC is the
ITU-T / ISO standard that succeeds HEVC and delivers ~30-50% better
compression at equal quality. For the fork's quality-aware encode
automation harness this matters in two ways:

1.  It is the first standardised codec on the fork's adapter set whose
    rate-distortion curve materially shifts the corpus distribution —
    every Phase B / C predictor that conditions on ``encoder`` will
    have to learn a different curve for VVC than for HEVC / AV1.
2.  VVC is the first standard with first-class **neural-network video
    coding (NNVC)** tool-points exposed through the encoder CLI:

    *   **NN-based intra prediction** — replaces the handcrafted
        intra-prediction directional / planar / DC modes with a
        learned 5×5 / 7×7 / 9×9 convolutional layer that predicts the
        block's pixels from its causal neighbourhood. Quality gain
        ~1-3% bitrate at iso-VMAF for natural content; ~5-10× slower
        intra encode time.
    *   **NN-based loop filter** — a learned post-processing CNN that
        replaces (or augments) VVC's deblocking + SAO + ALF cascade.
        Quality gain ~2-4% bitrate at iso-VMAF; cost is decode-side as
        well as encode-side.
    *   **NN-based super-resolution** — encode at low resolution,
        decode and run a learned upsampler. Gain skews to low-bitrate
        regimes; cost is decoder-side.

    These tools are toggleable per-encode and are the closest thing
    the open-source video stack has to a "neural-augmented codec"
    today. The fork's existing tiny-AI surface (vmaf-tiny, NR
    metrics, learned filters) is end-to-end *measurement*; NNVC is
    end-to-end *generation*. Putting both behind the same
    ``vmaf-tune`` harness lets future Phase B / C predictors learn
    when the NNVC tools are worth their compute cost.

Phase A wires only the deterministic surface: a single-pass QP encode
with optional NNVC-intra toggle. Two-pass / Pareto / per-shot dynamic
QP land in later phases per ADR-0237 and the new ADR for this PR.

Subprocess boundary is the integration seam — tests mock
``subprocess.run`` rather than running ffmpeg / vvencapp.
"""

from __future__ import annotations

import dataclasses

# Compress the fork's canonical 7-name preset vocabulary onto VVenC's
# 5-level scale. The 7-name vocabulary is the union of x264's 10
# presets minus duplicates and is the one the search loop emits;
# every adapter decides locally how to project onto its native scale.
# Anything strictly slower than ``slow`` (placebo / slowest / slower)
# pins to VVenC's deepest preset; anything strictly faster than
# ``fast`` pins to ``faster``. ``medium`` is the default.
_PRESET_MAP: dict[str, str] = {
    "placebo": "slower",
    "slowest": "slower",
    "slower": "slower",
    "slow": "slow",
    "medium": "medium",
    "fast": "fast",
    "faster": "faster",
    "veryfast": "faster",
    "superfast": "faster",
    "ultrafast": "faster",
}

_NATIVE_PRESETS: tuple[str, ...] = (
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
)


@dataclasses.dataclass(frozen=True)
class VVenCAdapter:
    """libvvenc single-pass QP adapter (VVC / H.266 + optional NNVC)."""

    name: str = "libvvenc"
    encoder: str = "libvvenc"
    quality_knob: str = "qp"
    # VVenC accepts QP 0..63; we surface the perceptually informative
    # window, mirroring the AV1-style scale. The default lands in the
    # middle of the curve for typical 1080p/4K natural content.
    quality_range: tuple[int, int] = (17, 50)
    quality_default: int = 32
    invert_quality: bool = True  # higher QP = lower quality

    # Vocabulary the search loop sees — the canonical 7-name superset.
    # The adapter compresses to VVenC's 5-level native vocabulary at
    # ``encode()`` time via ``native_preset()``.
    presets: tuple[str, ...] = (
        "placebo",
        "slowest",
        "slower",
        "slow",
        "medium",
        "fast",
        "faster",
        "veryfast",
        "superfast",
        "ultrafast",
    )

    # NNVC tool toggles. Default off so Phase A grids stay deterministic
    # and reasonably fast; flipping any of these shifts the encoder's
    # rate-distortion curve and is recorded in the corpus row's
    # ``extra_params`` for downstream predictor conditioning.
    nnvc_intra: bool = False

    def native_preset(self, preset: str) -> str:
        """Return the native VVenC preset for a 7-name canonical preset.

        Raises ``ValueError`` for unknown names. Pure function — no
        I/O — so the search loop can pre-compute the projection.
        """
        if preset not in _PRESET_MAP:
            raise ValueError(
                f"unknown libvvenc preset {preset!r}; expected one of " f"{tuple(_PRESET_MAP)}"
            )
        return _PRESET_MAP[preset]

    def validate(self, preset: str, qp: int) -> None:
        """Raise ``ValueError`` if ``(preset, qp)`` is unsupported."""
        if preset not in _PRESET_MAP:
            raise ValueError(
                f"unknown libvvenc preset {preset!r}; expected one of " f"{tuple(_PRESET_MAP)}"
            )
        lo, hi = self.quality_range
        if not lo <= qp <= hi:
            raise ValueError(f"qp {qp} outside libvvenc range [{lo}, {hi}]")

    def extra_params(self) -> tuple[str, ...]:
        """FFmpeg ``-c:v libvvenc`` arg suffix for the NNVC toggles.

        Returns an immutable tuple so callers can safely concatenate
        into ``EncodeRequest.extra_params``. Empty when no NNVC tool
        is enabled.

        FFmpeg's ``libvvenc`` wrapper forwards opaque ``-vvenc-params
        key=value:key=value`` strings down to the underlying VVenC
        config object, which is the surface VVenC's CLI documents
        for NNVC toggles.
        """
        toggles: list[str] = []
        if self.nnvc_intra:
            # ``IntraNN`` is the VVenC config-key for the learned
            # intra-prediction tool. Value 1 enables the 5×5 / 7×7 /
            # 9×9 conv ladder; value 0 keeps the handcrafted modes.
            toggles.append("IntraNN=1")
        if not toggles:
            return ()
        return ("-vvenc-params", ":".join(toggles))


def native_presets() -> tuple[str, ...]:
    """Return VVenC's native 5-level preset vocabulary.

    Exposed for tests and for any consumer that needs to describe the
    encoder's actual scale rather than the harness's canonical
    superset.
    """
    return _NATIVE_PRESETS
