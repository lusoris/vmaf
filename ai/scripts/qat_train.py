#!/usr/bin/env python3
"""ONNX quant-aware training scaffold (ADR-0173 / T5-3).

Wraps the existing tiny-AI Lightning trainer with a fake-quant
forward pass via ``torch.ao.quantization`` (the modern PyTorch
quantization namespace), then exports an int8 ONNX. Recovers most
of the accuracy lost by static PTQ; cost is one extra training
phase (~1.5× the fp32 training time).

This is a **scaffold** — wired so the rest of the harness (registry
field, CI gate, runtime selection) can land first per the
"audit-first" sequence in ADR-0129. A real call requires a model
config + dataset; the wrapper currently raises ``NotImplementedError``
unless ``--config`` is supplied AND the trainer is extended to honour
the QAT switch.

Usage::

    python ai/scripts/qat_train.py --config ai/configs/c2_qat.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Lightning training config YAML (e.g. ai/configs/c2_qat.yaml)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output `.int8.onnx` path")
    parser.add_argument(
        "--epochs-fp32",
        type=int,
        default=20,
        help="Number of epochs to pretrain in fp32 before fake-quant insertion",
    )
    parser.add_argument(
        "--epochs-qat",
        type=int,
        default=10,
        help="Number of QAT fine-tuning epochs after fake-quant insertion",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        sys.exit(f"torch not available: {exc}")

    cfg = args.config.resolve()
    if not cfg.is_file():
        sys.exit(f"config not found: {cfg}")

    # Wiring point for the trainer integration. The audit-first PR
    # ships this stub so the rest of the harness (registry, CI gate,
    # runtime selection) can land in isolation. The follow-up PR
    # that actually runs QAT on a model fills in the trainer hook
    # alongside its accuracy-vs-fp32 measurement.
    msg = (
        "QAT integration is scaffolded but not yet wired into the "
        "Lightning trainer. To complete:\n"
        "  1. Extend ai/src/vmaf_train/train.py with a QAT phase that\n"
        "     (a) trains fp32 for cfg.qat.epochs_fp32 epochs,\n"
        "     (b) inserts fake-quant via torch.ao.quantization.prepare_qat,\n"
        "     (c) fine-tunes for cfg.qat.epochs_qat epochs,\n"
        "     (d) exports via torch.ao.quantization.convert + torch.onnx.export.\n"
        "  2. Drop a config under ai/configs/<model>_qat.yaml.\n"
        "  3. Per ADR-0173 § 'Audit-first sequence', this lands as a\n"
        "     standalone PR with its own accuracy-drop measurement.\n"
    )
    print(msg, file=sys.stderr)
    print(f"[qat_train] config={cfg} epochs_fp32={args.epochs_fp32} epochs_qat={args.epochs_qat}")
    raise NotImplementedError("QAT trainer hook not yet wired (see message above)")


if __name__ == "__main__":
    sys.exit(main())
