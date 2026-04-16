"""Parse libvmaf's ONNX op allowlist and check models against it.

Single source of truth is libvmaf/src/dnn/op_allowlist.c — parsing the C
string table keeps Python and C from drifting. Any model whose graph uses
an op not on the list will be rejected by vmaf_dnn_op_allowed() at
session-init time, so catching it at export time turns a runtime load
failure into a trainer-side error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import onnx

_ALLOWLIST_C = Path(__file__).resolve().parents[3] / "libvmaf/src/dnn/op_allowlist.c"
_STRING_RE = re.compile(r'"([A-Za-z][A-Za-z0-9_]*)"')


def load_allowlist(source: Path = _ALLOWLIST_C) -> frozenset[str]:
    """Return the set of op types libvmaf will accept at session init."""
    if not source.is_file():
        raise FileNotFoundError(
            f"op_allowlist.c not found at {source}; cannot validate ONNX ops"
        )
    text = source.read_text(encoding="utf-8")
    return frozenset(_STRING_RE.findall(text))


@dataclass(frozen=True)
class AllowlistReport:
    allowed: frozenset[str]
    used: frozenset[str]
    forbidden: frozenset[str]

    @property
    def ok(self) -> bool:
        return not self.forbidden

    def pretty(self) -> str:
        if self.ok:
            return f"allowlist OK ({len(self.used)} distinct ops, all allowed)"
        bad = ", ".join(sorted(self.forbidden))
        return f"allowlist FAIL: {len(self.forbidden)} forbidden op(s): {bad}"


def check_model(
    onnx_path: Path, allowed: frozenset[str] | None = None
) -> AllowlistReport:
    """Walk @p onnx_path and compare its ops against libvmaf's allowlist."""
    if allowed is None:
        allowed = load_allowlist()
    model = onnx.load(str(onnx_path))
    used = frozenset(node.op_type for node in model.graph.node)
    return AllowlistReport(
        allowed=allowed,
        used=used,
        forbidden=frozenset(op for op in used if op not in allowed),
    )


def check_graph(
    model: onnx.ModelProto, allowed: frozenset[str] | None = None
) -> AllowlistReport:
    """Same as check_model() but against an in-memory onnx.ModelProto."""
    if allowed is None:
        allowed = load_allowlist()
    used = frozenset(node.op_type for node in model.graph.node)
    return AllowlistReport(
        allowed=allowed,
        used=used,
        forbidden=frozenset(op for op in used if op not in allowed),
    )
