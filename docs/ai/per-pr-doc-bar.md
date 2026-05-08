# Tiny-AI per-PR doc bar (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> The substantive content lives in the ADR; this stub exists to give
> readers a topic-tree entry point.

[ADR-0042](../adr/0042-tinyai-docs-required-per-pr.md) sets a tighter
**5-point doc bar** for any PR touching the tiny-AI surface (`ai/`,
`libvmaf/src/dnn/`, anything model-card-bearing). It is the
specialisation of the project-wide
[ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) doc-
substance rule for tiny-AI.

The 5 points (per ADR-0042) — **paraphrased**, see the ADR for the
authoritative wording:

1. A model card under [`docs/ai/models/`](models/) or an updated
   training-data / inference / quantisation page.
2. Reproducer command in the PR description.
3. PLCC / SROCC / RMSE reading on a held-out fold.
4. ONNX-graph operator-allowlist conformance check.
5. CHANGELOG entry under `changelog.d/added/` or `changelog.d/changed/`.

This bar co-exists with — does not replace — the project-wide r10
rule from [`CLAUDE.md` §12](../../CLAUDE.md). PRs that touch *both*
a tiny-AI surface and a non-tiny-AI surface satisfy both bars
independently.

## See also

- [ADR-0042](../adr/0042-tinyai-docs-required-per-pr.md) — the
  policy itself.
- [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md) — the
  project-wide rule this specialises.
- [`docs/ai/overview.md`](overview.md) — tiny-AI surface overview.
