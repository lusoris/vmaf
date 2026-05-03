# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Predicate-driven CRF recommendation over a corpus.

The Phase A corpus already produces ``(preset, crf, bitrate_kbps,
vmaf_score)`` tuples. ``recommend`` re-uses those rows and applies a
user-supplied predicate:

- ``--target-vmaf T`` — return the row with the *smallest* CRF whose
  ``vmaf_score >= T``. Falls back to the row with the highest VMAF if
  no row clears the bar.
- ``--target-bitrate B`` — return the row whose ``bitrate_kbps`` is
  closest to ``B`` (absolute distance, ties broken by smaller CRF).

Mutually exclusive — exactly one target must be specified.

The orchestration is deliberately thin: the corpus loader either
consumes a pre-existing JSONL (``--from-corpus``) or builds one on the
fly with the same machinery as the ``corpus`` subcommand. The
predicate evaluation is independent of how the rows were obtained.

Implements Buckets #4 (target-bitrate) and #5 (target-vmaf) from the
capability audit (Research-0061).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class RecommendRequest:
    """Predicate description.

    Exactly one of ``target_vmaf`` / ``target_bitrate_kbps`` must be set.
    Validation lives in :func:`validate_request` so the CLI layer and
    library callers share the same exit-code semantics.
    """

    target_vmaf: float | None = None
    target_bitrate_kbps: float | None = None
    encoder: str | None = None
    preset: str | None = None


@dataclasses.dataclass(frozen=True)
class RecommendResult:
    """Single winning row + the predicate that picked it."""

    row: dict
    predicate: str
    margin: float
    """Predicate-specific distance from the target.

    For ``target-vmaf``: ``vmaf_score - target`` (positive = clears the bar).
    For ``target-bitrate``: ``bitrate_kbps - target`` (signed).
    """


def validate_request(req: RecommendRequest) -> None:
    """Enforce mutually-exclusive target. Raises :class:`ValueError`."""
    has_vmaf = req.target_vmaf is not None
    has_bitrate = req.target_bitrate_kbps is not None
    if has_vmaf and has_bitrate:
        raise ValueError(
            "--target-vmaf and --target-bitrate are mutually exclusive; " "specify exactly one"
        )
    if not (has_vmaf or has_bitrate):
        raise ValueError("missing target: pass --target-vmaf or --target-bitrate")


def _filter_rows(rows: Iterable[dict], req: RecommendRequest) -> list[dict]:
    """Drop rows that fail the encoder/preset filter or have NaN VMAF."""
    out: list[dict] = []
    for row in rows:
        if req.encoder is not None and row.get("encoder") != req.encoder:
            continue
        if req.preset is not None and row.get("preset") != req.preset:
            continue
        if int(row.get("exit_status", 0)) != 0:
            continue
        v = row.get("vmaf_score")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out.append(row)
    return out


def pick_target_vmaf(rows: Sequence[dict], target: float) -> RecommendResult:
    """Smallest CRF whose VMAF clears ``target``.

    Falls back to the row with the highest VMAF if none clears the bar
    — the user gets the closest miss rather than an empty result.
    """
    if not rows:
        raise ValueError("no eligible rows to evaluate (after filtering)")
    clearing = [r for r in rows if float(r["vmaf_score"]) >= target]
    if clearing:
        winner = min(clearing, key=lambda r: (int(r["crf"]), -float(r["vmaf_score"])))
        return RecommendResult(
            row=winner,
            predicate=f"target_vmaf>={target}",
            margin=float(winner["vmaf_score"]) - target,
        )
    # No row clears the bar — return the row that comes closest from below.
    winner = max(rows, key=lambda r: float(r["vmaf_score"]))
    return RecommendResult(
        row=winner,
        predicate=f"target_vmaf>={target} (UNMET)",
        margin=float(winner["vmaf_score"]) - target,
    )


def pick_target_bitrate(rows: Sequence[dict], target_kbps: float) -> RecommendResult:
    """Row whose bitrate is closest to ``target_kbps`` (absolute distance).

    Ties on distance go to the lower CRF (higher quality), which matches
    the producer intent "give me the best quality fitting under the
    bitrate cap" when multiple cells land on the same point.
    """
    if not rows:
        raise ValueError("no eligible rows to evaluate (after filtering)")
    winner = min(
        rows,
        key=lambda r: (
            abs(float(r["bitrate_kbps"]) - target_kbps),
            int(r["crf"]),
        ),
    )
    return RecommendResult(
        row=winner,
        predicate=f"|bitrate-{target_kbps}|->min",
        margin=float(winner["bitrate_kbps"]) - target_kbps,
    )


def recommend(rows: Iterable[dict], req: RecommendRequest) -> RecommendResult:
    """Top-level dispatcher: validate, filter, apply the predicate."""
    validate_request(req)
    eligible = _filter_rows(rows, req)
    if req.target_vmaf is not None:
        return pick_target_vmaf(eligible, req.target_vmaf)
    assert req.target_bitrate_kbps is not None  # proven by validate_request
    return pick_target_bitrate(eligible, req.target_bitrate_kbps)


def load_corpus_jsonl(path: Path) -> Iterator[dict]:
    """Stream rows from a JSONL file written by the ``corpus`` subcommand."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def format_result(result: RecommendResult) -> str:
    """Human-readable single-line summary for the CLI."""
    row = result.row
    return (
        f"encoder={row.get('encoder')} preset={row.get('preset')} "
        f"crf={row.get('crf')} vmaf={float(row['vmaf_score']):.3f} "
        f"bitrate_kbps={float(row['bitrate_kbps']):.2f} "
        f"predicate={result.predicate} margin={result.margin:+.3f}"
    )
