#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Export ``vmaf_tiny_v5`` to a self-contained ONNX file.

Architecturally identical to ``export_vmaf_tiny_v2`` (same mlp_small
6 → 16 → 8 → 1 + bundled StandardScaler topology, opset 17). The only
delta is the model id and sidecar metadata strings, so the exporter
reuses the v2 wrapper unchanged. v5 differs from v2 only in its
training corpus (4-corpus + UGC vp9 subset).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OPSET = 17


def _build_mlp_small(in_dim: int):  # type: ignore[no-untyped-def]
    from torch import nn

    return nn.Sequential(
        nn.Linear(in_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def _bundled_wrapper(mlp, mean: np.ndarray, std: np.ndarray):  # type: ignore[no-untyped-def]
    import torch
    from torch import nn

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = mlp
            self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))

        def forward(self, features):  # type: ignore[no-untyped-def]
            normed = (features - self.mean) / self.std
            out = self.mlp(normed)
            return out.squeeze(-1)

    return _Wrap().eval()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(prog="export_vmaf_tiny_v5.py", description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-onnx", type=Path, required=True)
    ap.add_argument("--out-sidecar", type=Path, required=True)
    args = ap.parse_args()

    import torch

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    features = state["features"]
    in_dim = len(features)
    mean = np.asarray(state["input_mean"], dtype=np.float64)
    std = np.asarray(state["input_std"], dtype=np.float64)
    if in_dim != 6:
        print(f"[export-v5] expected 6 features, got {in_dim}", file=sys.stderr)
        return 2

    mlp = _build_mlp_small(in_dim)
    mlp.load_state_dict(state["state_dict"])
    mlp.eval()
    wrapper = _bundled_wrapper(mlp, mean, std)

    dummy = torch.zeros(1, in_dim, dtype=torch.float32)
    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export-v5] tracing wrapper -> {args.out_onnx} (opset={OPSET})")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.out_onnx),
        input_names=["features"],
        output_names=["vmaf"],
        dynamic_axes={"features": {0: "N"}, "vmaf": {0: "N"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    import onnx

    proto = onnx.load(str(args.out_onnx))
    onnx.save(proto, str(args.out_onnx), save_as_external_data=False)
    sidecar_data = args.out_onnx.with_suffix(".onnx.data")
    if sidecar_data.exists():
        sidecar_data.unlink()

    digest = _sha256(args.out_onnx)
    print(f"[export-v5] sha256={digest}")
    print(f"[export-v5] size  ={args.out_onnx.stat().st_size} bytes")

    sidecar = {
        "id": "vmaf_tiny_v5",
        "kind": "fr",
        "onnx": args.out_onnx.name,
        "opset": OPSET,
        "sha256": digest,
        "input_name": "features",
        "input_shape": [-1, in_dim],
        "output_name": "vmaf",
        "output_shape": [-1],
        "features": list(features),
        "input_mean": mean.tolist(),
        "input_std": std.tolist(),
        "notes": (
            "vmaf_tiny_v5 — same arch as vmaf_tiny_v2 (mlp_small + bundled "
            "StandardScaler, 257 params, opset 17) trained on 5-corpus "
            "(NF + KoNViD + BVI-DVC A+B+C+D + YouTube UGC vp9 subset). "
            "Opt-in only; v2 remains the production default."
        ),
    }
    args.out_sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.out_sidecar.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[export-v5] wrote {args.out_sidecar}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
