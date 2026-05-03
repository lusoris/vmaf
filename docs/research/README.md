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
| [0003](0003-ssimulacra2-port-sourcing.md) | SSIMULACRA 2 port source selection + upstream-drift strategy | Active | [ADR-0126](../adr/0126-ssimulacra2-extractor.md) |
| [0004](0004-vulkan-backend-design.md) | Vulkan compute backend — loader, shader language, allocator, DMABUF import | Active | [ADR-0127](../adr/0127-vulkan-compute-backend.md) |
| [0005](0005-embedded-mcp-transport.md) | Embedded MCP in libvmaf — threading, JSON library, SSE server, Power-of-10 fit | Active | [ADR-0128](../adr/0128-embedded-mcp-in-libvmaf.md) |
| [0006](0006-tinyai-ptq-accuracy-targets.md) | Tiny-AI PTQ int8 — accuracy targets, ORT API comparison, calibration sourcing | Active | [ADR-0129](../adr/0129-tinyai-ptq-quantization.md) |
| [0007](0007-ssimulacra2-scalar-port.md) | SSIMULACRA 2 scalar port — YUV handling, blur deviation, snapshot tooling | Active | [ADR-0126](../adr/0126-ssimulacra2-extractor.md), [ADR-0130](../adr/0130-ssimulacra2-scalar-implementation.md) |
| [0008](0008-ms-ssim-decimate-simd.md) | MS-SSIM decimate SIMD — FLOP accounting, summation order, bit-exactness | Active | [ADR-0125](../adr/0125-ms-ssim-decimate-simd.md) |
| [0010](0010-speed-netflix-upstream-direction.md) | Is Netflix about to ship a SpEED-driven VMAF successor? (informational) | Active | — |
| [0011](0011-iqa-convolve-avx2.md) | `_iqa_convolve` AVX2 — bit-exactness via `__m256d`, kernel invariants, Amdahl | Active | [ADR-0138](../adr/0138-iqa-convolve-avx2-bitexact-double.md) |
| [0012](0012-ssim-simd-bitexact.md) | SSIM SIMD bit-exactness to scalar — where the ULP drifted | Active | [ADR-0139](../adr/0139-ssim-simd-bitexact-double.md) |
| [0013](0013-simd-dx-framework.md) | SIMD DX framework — audit + NEON bit-exactness port | Active | [ADR-0140](../adr/0140-simd-dx-framework.md) |
| [0014](0014-psnr-hvs-neon.md) | `psnr_hvs` NEON sister port — half-wide split strategy, aarch64 gotchas, QEMU verification limits | Active | [ADR-0160](../adr/0160-psnr-hvs-neon-bitexact.md) |
| [0015](0015-ssimulacra2-simd.md) | SSIMULACRA 2 AVX2 + AVX-512 + NEON — per-lane `cbrtf`, left-to-right summation, 2×2 downsample deinterleave | Active | [ADR-0161](../adr/0161-ssimulacra2-simd-bitexact.md) |
| [0016](0016-ssimulacra2-iir-blur-simd.md) | SSIMULACRA 2 IIR blur SIMD — row-batching with gather (horizontal), column-SIMD (vertical), bit-exact to scalar | Active | [ADR-0162](../adr/0162-ssimulacra2-iir-blur-simd.md) |
| [0017](0017-ssimulacra2-ptlr-simd.md) | SSIMULACRA 2 `picture_to_linear_rgb` SIMD — per-lane scalar reads, SIMD matmul, per-lane scalar `powf` | Active | [ADR-0163](../adr/0163-ssimulacra2-ptlr-simd.md) |
| [0018](0018-ssimulacra2-snapshot-gate.md) | SSIMULACRA 2 snapshot-JSON regression gate — why fork self-consistency beats libjxl/Pacidus cross-check at this scope | Active | [ADR-0164](../adr/0164-ssimulacra2-snapshot-gate.md) |
| [0031](0031-intel-ai-pc-applicability.md) | Intel AI-PC NPU + EP applicability to tiny-AI / `dnn/` — verdict: defer NPU; iGPU already covered by OpenVINO EP | Active | — (backlog T7-9) |
| [0046](0046-vmaf-tiny-v3-mlp-medium-evaluation.md) | `vmaf_tiny_v3` (mlp_medium 6→32→16→1, 769 params) vs v2 (mlp_small 257 params): 4-corpus parquet, identical recipe; Netflix LOSO mean PLCC 0.9986 ± 0.0015 vs v2's 0.9978 ± 0.0021 (+0.0008 mean, -29 % std). Decision matrix + per-fold table; ship-alongside-v2 recommendation. | Active | [ADR-0241](../adr/0241-vmaf-tiny-v3-mlp-medium.md) |
| [0048](0048-vmaf-tiny-v4-mlp-large-evaluation.md) | `vmaf_tiny_v4` (mlp_large, 3 073 params) — does the architecture ladder saturate? Verdict: yes, +0.0001 mean PLCC vs v3 (below 1 std). Ladder stops at v4. | Active | [ADR-0242](../adr/0242-vmaf-tiny-v4-mlp-large.md) |

| [0053](0053-post-merge-cpu-profile-2026-05-03.md) | Post-merge CPU profile 2026-05-03 — perf top-10 after PRs #310–#321; surfaces 3 new opt targets (convolve widen, SSIM double reduction, VIF gather elimination) | Active | — |
<!-- Backfill entries for older workstreams land here as their authors
     revisit the corresponding code. -->

*(Index seeded by [ADR-0108](../adr/0108-deep-dive-deliverables-rule.md)'s
adoption PR; backfilled digests for the existing major workstreams
will be added as their authors revisit the corresponding code.)*
