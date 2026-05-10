# `vmaf-tune` ladder default sampler (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

The `--sampler` flag selects the rate-axis sampler used by the Phase E
bitrate-ladder generator (per
[ADR-0307](../adr/0307-vmaf-tune-ladder-default-sampler.md)). The
default is the *log-space-uniform* sampler over a `(min_kbps,
max_kbps)` envelope; alternative samplers cover Pareto-frontier
geometric sequences (per
[ADR-0295](../adr/0295-vmaf-tune-phase-e-bitrate-ladder.md)).

Status: Accepted. Wired in `tools/vmaf-tune/src/vmaftune/cli.py`.

## See also

- [`vmaf-tune.md`](vmaf-tune.md) — the base tool, especially the
  Phase E ladder section once implemented.
- [ADR-0295](../adr/0295-vmaf-tune-phase-e-bitrate-ladder.md) /
  [ADR-0307](../adr/0307-vmaf-tune-ladder-default-sampler.md) —
  design decisions.
