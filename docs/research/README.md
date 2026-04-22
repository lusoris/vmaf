# Research digests

Iteration-time research notes for the lusoris vmaf fork. Each digest
captures **what was investigated and why** for a fork-local
workstream — source links, alternatives weighed, prior art, dead ends.

These are *not* ADRs:

- An [ADR](../adr/) records a *decision* and its alternatives at the
  moment it was made. The body is frozen once Accepted.
- A research digest records the *learning* behind that decision (and
  the iterations that followed). It can be amended as new evidence
  arrives, the same way a lab notebook is.

A typical workstream has one ADR (the decision) and one research
digest (the supporting investigation). Some PRs reuse an existing
digest by linking; that is fine.

## When to write one

Required by [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md) on
every fork-local PR that makes a non-trivial design choice. PRs
without a design choice (e.g., a one-line bug fix in fork-added code)
state "no research digest needed: trivial" in the PR description and
skip the file. Reuse over duplication: if the workstream already has
a digest, link it from the new PR instead of starting a parallel one.

## Format

Each file is named `NNNN-kebab-case-topic.md` with a 4-digit
zero-padded ID assigned in commit order. The structure mirrors
[`0000-template.md`](0000-template.md):

```markdown
# Research-NNNN: <short, descriptive title>

- **Status**: Active | Superseded by Research-MMMM | Archived
- **Workstream**: <ADR-NNNN, ADR-MMMM, ...>
- **Last updated**: YYYY-MM-DD

## Question         — what was the unknown going in
## Sources          — papers, upstream docs, Netflix issues, prior PRs
## Findings         — what was learned, with citations
## Alternatives explored — what didn't work and why
## Open questions   — what is still unknown
## Related          — ADRs, PRs, issues
```

Conventions:

- IDs are assigned in commit order and never reused.
- Digests are *amendable* — update the `Last updated` date when you
  add findings. To replace one entirely, add `Status: Superseded by
  Research-MMMM` and write a new file.
- Cite sources inline with `[link text](URL)` so readers can verify.
- Keep one digest per workstream, not per PR. Cross-link from the PR
  description.

## Index

| ID | Title | Status | Workstream |
| --- | --- | --- | --- |
| [0001](0001-bisect-model-quality-cache.md) | Cache shape for `bisect-model-quality` nightly | Active | [ADR-0109](../adr/0109-nightly-bisect-model-quality.md) |
| [0002](0002-automated-rule-enforcement.md) | Automating process-ADR enforcement (0100 / 0105 / 0106 / 0108) | Active | [ADR-0124](../adr/0124-automated-rule-enforcement.md) |
| [0007](0007-ssimulacra2-scalar-port.md) | SSIMULACRA 2 scalar port — YUV handling, blur deviation, snapshot tooling | Active | [ADR-0126](../adr/0126-ssimulacra2-feature-extractor.md), [ADR-0130](../adr/0130-ssimulacra2-scalar-implementation.md) |
| [0008](0008-ms-ssim-decimate-simd.md) | MS-SSIM decimate SIMD — FLOP accounting, summation order, bit-exactness | Active | [ADR-0125](../adr/0125-ms-ssim-decimate-simd.md) |
| [0010](0010-speed-netflix-upstream-direction.md) | Is Netflix about to ship a SpEED-driven VMAF successor? (informational) | Active | — |
| [0011](0011-iqa-convolve-avx2.md) | `_iqa_convolve` AVX2 — bit-exactness via `__m256d`, kernel invariants, Amdahl | Active | [ADR-0138](../adr/0138-iqa-convolve-avx2-bitexact-double.md) |
| [0012](0012-ssim-simd-bitexact.md) | SSIM SIMD bit-exactness to scalar — where the ULP drifted | Active | [ADR-0139](../adr/0139-ssim-simd-bitexact-double.md) |

<!-- Backfill entries for older workstreams land here as their authors
     revisit the corresponding code. -->

*(Index seeded by [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md)'s
adoption PR; backfilled digests for the existing major workstreams
will be added as their authors revisit the corresponding code.)*
