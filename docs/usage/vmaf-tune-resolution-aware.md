# `vmaf-tune --resolution-aware` (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

The `--resolution-aware` mode (per
[ADR-0289](../adr/0289-vmaf-tune-resolution-aware.md)) tunes the CRF
sweep range and the encoder preset choice to the source resolution
instead of using one global grid. 1080p sources get a different
`(preset, CRF)` rectangle than 4K sources.

Status: Accepted. Wired in `tools/vmaf-tune/src/vmaftune/cli.py`.

## See also

- [`vmaf-tune.md`](vmaf-tune.md) — the base tool.
- [ADR-0289](../adr/0289-vmaf-tune-resolution-aware.md) — design
  decision; lookup table that maps resolution buckets to grid shapes.
