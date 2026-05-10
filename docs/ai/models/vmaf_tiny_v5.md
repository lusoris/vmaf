# `vmaf_tiny_v5` (deferred — stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Records the deferral; no shipped model yet.

Per [ADR-0287](../adr/0287-vmaf-tiny-v5-corpus-expansion.md) the
`vmaf_tiny_v5` corpus-expansion experiment was **deferred** — the
training corpus expansion produced a smaller PLCC delta than the
v3 → v4 step, and the architecture ladder is treated as saturated
at v4 (per
[Research-0048](../research/0048-vmaf-tiny-v4-mlp-large-evaluation.md)).
There is no `vmaf_tiny_v5.onnx` shipped in `model/tiny/` and no
`v5` runtime path. This stub exists so a reader looking for a v5
model card finds the deferral verdict instead of a 404.

## See also

- [`vmaf_tiny_v4.md`](vmaf_tiny_v4.md) — the current top-of-ladder
  tiny model (architecture ladder ends here per the v4 evaluation).
- [`vmaf_tiny_v3.md`](vmaf_tiny_v3.md) — the second-from-top tiny
  model (also shipped).
- [ADR-0287](../adr/0287-vmaf-tiny-v5-corpus-expansion.md) —
  defer-decision context.
