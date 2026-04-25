#!/usr/bin/env python3
"""ONNX dynamic post-training quantisation (ADR-0173 / T5-3).

The cheapest of the three quantisation modes — quantises weights
offline; activations are quantised at runtime by ORT. Single CLI
call, no calibration data needed. Best fit for models where:

- We don't ship a calibration set,
- The deployment box differs from the training box,
- A small accuracy hit is acceptable for a 2× speedup.

Output lands at ``<input>.int8.onnx`` next to the fp32 source.
``model/tiny/registry.json`` should set ``quant_mode = "dynamic"``
on the matching entry; the libvmaf loader prefers the int8 file
when it sees that mode.

Usage::

    python ai/scripts/ptq_dynamic.py model/tiny/nr_metric_v1.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="Path to fp32 ONNX file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; default appends `.int8.onnx`",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Per-channel weight quantisation (better accuracy, marginal "
        "size cost). Default off; flip on for sensitive models.",
    )
    args = parser.parse_args()

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        sys.exit(f"onnxruntime.quantization not available: {exc}")

    src = args.onnx.resolve()
    if not src.is_file():
        sys.exit(f"input not found: {src}")
    if not src.name.endswith(".onnx"):
        sys.exit(f"input must end with .onnx: {src}")
    dst = args.output or src.with_name(src.stem + ".int8.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ptq_dynamic] {src}  ->  {dst}  per-channel={args.per_channel}")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )
    sz_in = src.stat().st_size
    sz_out = dst.stat().st_size
    print(f"[ptq_dynamic] done — {sz_in:,} -> {sz_out:,} bytes ({sz_out / sz_in:.2f}×)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
