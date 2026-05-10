# `vmaf-tune` Phase E — bitrate ladder (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

Phase E of the `vmaf-tune` six-phase roadmap (per
[ADR-0237](../adr/0237-quality-aware-encode-automation.md) §Phases) is
the **bitrate ladder generator** — given a (source, encoder) pair, emit
a Pareto-optimal sequence of `(width, height, bitrate, target-VMAF)`
rungs suitable for ABR streaming. The Phase E sampler / scoring
methodology is documented in
[ADR-0295](../adr/0295-vmaf-tune-phase-e-bitrate-ladder.md);
[ADR-0307](../adr/0307-vmaf-tune-ladder-default-sampler.md) chooses
the default rate-axis sampler.

Status: Phase E is *Proposed* — the scaffold is in tree but is gated
on Phase B (target-VMAF bisect) merging upstream of it. This stub
will be replaced with full prose once the ship-blocking dependency
clears.

## See also

- [`vmaf-tune.md`](vmaf-tune.md) — the base tool and roadmap.
- [`vmaf-tune-ladder-default-sampler.md`](vmaf-tune-ladder-default-sampler.md)
  — the rate-axis sampler used by the ladder generator.
- [ADR-0295](../adr/0295-vmaf-tune-phase-e-bitrate-ladder.md) — Phase
  E design decision.
