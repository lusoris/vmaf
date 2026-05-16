# `vmaf-tune --coarse-to-fine` (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

`vmaf-tune corpus` and the `coarse-to-fine` sub-mode (per
[ADR-0306](../adr/0306-vmaf-tune-coarse-to-fine.md)) walk the CRF
axis in two passes: a coarse sweep with a wide step, followed by a
fine sweep around the cell that came closest to the user's target.
The flag short-circuits manual `--crf X --crf Y --crf Z`
enumeration when the operator has a target VMAF in mind.

Status: Accepted. Wired in `tools/vmaf-tune/src/vmaftune/cli.py`
behind `--coarse-to-fine`.

## See also

- [`vmaf-tune.md`](vmaf-tune.md) — the base tool, including the
  `recommend` subcommand that consumes coarse-to-fine output.
- [ADR-0306](../adr/0306-vmaf-tune-coarse-to-fine.md) — design
  decision and search-strategy detail.
