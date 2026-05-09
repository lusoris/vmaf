# OSSF Scorecard policy (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

[ADR-0263](../adr/0263-ossf-scorecard-policy.md) sets the fork's
policy on the OSSF Scorecard supply-chain audit: which checks are
gated CI failures vs informational, the documented exception list
(e.g. signed-releases requirements that interact with the fork's
Sigstore-keyless flow per
[ADR-0010](../adr/0010-sigstore-keyless-signing.md)), and the
quarterly cadence for re-evaluating exceptions.

The Scorecard workflow ships at
`.github/workflows/scorecards.yml`; the published score is
visible on the project's GitHub Insights page.

Status: Accepted.

## See also

- [`docs/development/release.md`](release.md) — release pipeline
  including signing prerequisites.
- [ADR-0010](../adr/0010-sigstore-keyless-signing.md) — Sigstore
  signing flow that the Scorecard "signed-releases" check
  validates.
- [ADR-0263](../adr/0263-ossf-scorecard-policy.md) — design
  decision.
