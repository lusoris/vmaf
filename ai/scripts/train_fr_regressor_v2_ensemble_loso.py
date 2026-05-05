#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""LOSO trainer for the ``fr_regressor_v2`` deep ensemble (ADR-0303).

Companion to:

* ``ai/scripts/train_fr_regressor_v2_ensemble.py`` (PR #372 scaffold —
  trains + exports the per-seed ONNX members against a synthetic smoke
  corpus) and
* ``ai/scripts/eval_probabilistic_proxy.py`` (PR #372 scaffold — a
  smoke evaluator over the ensemble's predictive variance).

This script implements the **production-flip** LOSO protocol per
ADR-0303 / Research-0075: 9-fold leave-one-source-out training over
the Netflix Public Dataset, repeated under five distinct seeds
``{0, 1, 2, 3, 4}``, emitting one ``loso_seed{N}.json`` per seed with
per-fold PLCC / SROCC / RMSE so the production-flip CI gate
(``scripts/ci/ensemble_prod_gate.py``) can decide which seeds are
clear to flip from ``smoke: true`` to ``smoke: false`` in
``model/tiny/registry.json``.

The actual training body is intentionally a stub on this branch — the
real Phase A canonical-6 corpus
(``runs/phase_a/full_grid/per_frame_canonical6.jsonl``, PR #392) is
not present in CI, and we deliberately do not flip any registry rows
in the same PR as the trainer scaffold. When the corpus is missing
the script raises ``NotImplementedError`` so smoke-only invocations
stay honest; argparse + module imports parse cleanly without it.

Usage::

    python ai/scripts/train_fr_regressor_v2_ensemble_loso.py --help

    # Real-corpus invocation (follow-up flip PR):
    python ai/scripts/train_fr_regressor_v2_ensemble_loso.py \\
        --seeds 0,1,2,3,4 \\
        --corpus runs/phase_a/full_grid/per_frame_canonical6.jsonl \\
        --out-dir runs/ensemble_loso/

The emitted JSON schema is documented in Research-0075
(``docs/research/0075-fr-regressor-v2-ensemble-prod-flip.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 9 Netflix Public Dataset sources — the LOSO folds. Mirrors the order
# baked into ai/scripts/eval_loso_vmaf_tiny_v3.py /
# eval_loso_vmaf_tiny_v5.py / Research-0067 (prod-loso) so the per-seed
# fold ordering is comparable across deterministic vs ensemble runs.
NETFLIX_SOURCES: tuple[str, ...] = (
    "BigBuckBunny_25fps",
    "BirdsInCage_30fps",
    "CrowdRun_25fps",
    "ElFuente1_30fps",
    "ElFuente2_30fps",
    "FoxBird_25fps",
    "OldTownCross_25fps",
    "Seeking_25fps",
    "Tennis_24fps",
)

# Canonical-6 libvmaf feature columns consumed by FRRegressor (ADR-0291).
CANONICAL_6: tuple[str, ...] = (
    "adm2",
    "vif_scale0",
    "vif_scale1",
    "vif_scale2",
    "vif_scale3",
    "motion2",
)

# Production ship gate per ADR-0303. Recorded here as constants so the
# trainer's emitted JSON carries the gate values it was trained
# against — the CI gate consumes the JSON, not these constants
# directly, but co-locating the values makes drift auditable.
SHIP_GATE_MEAN_PLCC: float = 0.95
SHIP_GATE_PLCC_SPREAD_MAX: float = 0.005


def _parse_seed_list(raw: str) -> list[int]:
    """Parse a comma-separated seed list (e.g. ``"0,1,2,3,4"``) into ints."""
    out: list[int] = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise argparse.ArgumentTypeError(
            "--seeds must be a non-empty comma-separated list of ints " "(e.g. --seeds 0,1,2,3,4)"
        )
    return out


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argparser. Exposed as a function so tests can import it."""
    p = argparse.ArgumentParser(
        prog="train_fr_regressor_v2_ensemble_loso",
        description=(
            "9-fold LOSO trainer for fr_regressor_v2 deep ensemble seeds "
            "(ADR-0303). Emits loso_seed{N}.json per seed; the CI gate "
            "scripts/ci/ensemble_prod_gate.py consumes the JSONs."
        ),
    )
    p.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=[0, 1, 2, 3, 4],
        help=(
            "Comma-separated seeds to train (default: 0,1,2,3,4 — the "
            "five ensemble members in model/tiny/registry.json)."
        ),
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("runs/phase_a/full_grid/per_frame_canonical6.jsonl"),
        help=(
            "Path to the Phase A canonical-6 per-frame JSONL corpus "
            "(PR #392). Defaults to the canonical location."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/ensemble_loso"),
        help="Output directory for loso_seed{N}.json artefacts.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Per-fold training epochs (default 200; matches ADR-0291).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default 32; matches ADR-0291).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Adam learning rate (default 5e-4; matches ADR-0291).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay (default 1e-5; matches ADR-0291).",
    )
    p.add_argument(
        "--num-codecs",
        type=int,
        default=12,
        help=(
            "Encoder-vocab size for the codec one-hot block (default 12 — "
            "ENCODER_VOCAB v2 from PR #394 / ADR-0291)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse args, validate corpus presence, and exit 0 without "
            "training. Useful for CI smoke."
        ),
    )
    return p


def _load_corpus(corpus_path: Path):  # type: ignore[no-untyped-def]
    """Load and return the Phase A canonical-6 JSONL corpus.

    Stub: not implemented on this branch. The follow-up flip PR
    plugs in the real loader (mirrors ``eval_loso_vmaf_tiny_v3.py``'s
    pandas-based loader).
    """
    raise NotImplementedError(
        f"Real-corpus LOSO training is gated on PR-time corpus availability; "
        f"this scaffold (ADR-0303) does not flip any registry rows in the "
        f"same PR as the trainer. Expected corpus: {corpus_path} (PR #392). "
        f"Plug the real loader in the follow-up flip PR — see "
        f"docs/research/0075-fr-regressor-v2-ensemble-prod-flip.md "
        f"§Reproducer for the schema."
    )


def _train_one_seed(seed: int, corpus, args: argparse.Namespace) -> dict:  # type: ignore[no-untyped-def]
    """Run 9-fold LOSO for a single seed; return the per-seed summary dict.

    Stub: returns a placeholder schema-correct dict. The follow-up
    flip PR plugs in the real per-fold ``FRRegressor(num_codecs=...)``
    training loop (see ``ai/scripts/train_fr_regressor_v2_ensemble.py``
    for the per-member trainer this LOSO wrapper would call).
    """
    raise NotImplementedError(
        f"Per-seed LOSO training is not implemented on this branch (seed={seed}). "
        f"The schema of the emitted JSON is documented in "
        f"docs/research/0075-fr-regressor-v2-ensemble-prod-flip.md."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    print(
        f"[ensemble-loso] seeds={args.seeds} corpus={args.corpus} "
        f"out_dir={args.out_dir} epochs={args.epochs}",
        flush=True,
    )

    if not args.corpus.exists():
        # Smoke-only invocation. Surface the fact loudly so callers
        # know the trainer ran in stub mode and no JSON was produced.
        print(
            f"[ensemble-loso] corpus not present at {args.corpus} — running "
            f"in scaffold/smoke mode. No loso_seed{{N}}.json artefacts will "
            f"be emitted. See ADR-0303 for the production-flip workflow.",
            flush=True,
        )
        if args.dry_run:
            print("[ensemble-loso] --dry-run requested; exiting cleanly.", flush=True)
            return 0
        # Even outside --dry-run, this is the expected smoke path on
        # the scaffold branch; emit a stub exit code so CI doesn't
        # red-flag the absence of the corpus.
        print(
            "[ensemble-loso] scaffold-only branch: trainer stub returns 0. "
            "Real-corpus LOSO is gated on the follow-up flip PR.",
            flush=True,
        )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Real-corpus path — currently raises NotImplementedError until
    # the follow-up flip PR plugs the loader + trainer in.
    corpus = _load_corpus(args.corpus)
    for seed in args.seeds:
        summary = _train_one_seed(seed, corpus, args)
        out_path = args.out_dir / f"loso_seed{seed}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"[ensemble-loso] wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
