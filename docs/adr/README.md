# Architectural Decision Records (ADR)

This is the **canonical, tracked** decision log for the fork. Every non-trivial
architectural / policy / scope decision lands here as its own markdown file
before the corresponding commit merges.

## Format

We use [Michael Nygard's ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
(MADR-style), one markdown file per decision — not a mega-table. See
[joelparkerhenderson/architecture-decision-record](https://github.com/joelparkerhenderson/architecture-decision-record)
for background.

Each ADR file is named `NNNN-kebab-case-title.md` with a zero-padded 4-digit ID
and follows the structure in [0000-template.md](0000-template.md):

```markdown
# ADR-NNNN: <short, declarative title>

- **Status**: Proposed | Accepted | Deprecated | Superseded by [ADR-NNNN](NNNN-title.md)
- **Date**: YYYY-MM-DD
- **Deciders**: <names / handles>
- **Tags**: <comma-separated area tags>

## Context              — the problem, the forces at play
## Decision             — one paragraph in active voice
## Alternatives considered  — at minimum the runner-up, in a pros/cons table
## Consequences         — Positive / Negative / Neutral-follow-ups
## References           — upstream docs, prior ADRs, related PRs, popup-answer source
```

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`. IDs are assigned in commit order
  and never reused.
- **Immutable once Accepted**: the body is frozen. To change a decision, write
  a new ADR with `Status: Supersedes ADR-NNNN` and flip the old one to
  `Superseded by ADR-MMMM`.
- **One decision per ADR** — if you find yourself writing "and also…", split it.
- **Tagging**: use the flat tag palette below so `grep -l 'Tags:.*cuda'
  docs/adr/*.md` works. New tags are fine when justified.
- **Link from per-package AGENTS.md**: the relevant per-package `AGENTS.md`
  points to the ADRs that govern that subtree, so the rationale is one click
  away from the code.
- **Backfill policy**: ADRs ≤ 0099 are *backfills* — decisions made before
  the ADR practice was formalised on 2026-04-17, captured retroactively from
  commit history and planning dossiers. Their `Status` reflects the current
  code, not the original decision date. New decisions start at 0100.

### Tag palette

`ai`, `agents`, `build`, `ci`, `claude`, `cli`, `cuda`, `dnn`, `docs`, `framework`,
`git`, `github`, `license`, `lint`, `matlab`, `mcp`, `planning`, `python`,
`readme`, `release`, `security`, `simd`, `supply-chain`, `sycl`, `testing`,
`workspace`.

## Why it exists

A Claude session makes a decision (directory move, CI gate change, dependency
swap), commits it, the session ends, and the rationale is recoverable only from
the commit message — which typically summarises the *what* but omits the
*alternatives considered*. ADRs preserve "we chose X over Y because Z" in a
single auditable place. See [ADR-0028](0028-adr-maintenance-rule.md).

## What counts as non-trivial?

Another engineer could reasonably have chosen differently. Examples:

- Directory moves (e.g., [ADR-0026](0026-workspace-relocated-under-python.md):
  `workspace/` → `python/vmaf/workspace/`)
- Base-image / dependency policy (e.g.,
  [ADR-0027](0027-non-conservative-image-pins.md): non-conservative CUDA pins)
- CI-gate semantics (e.g., [ADR-0024](0024-netflix-golden-preserved.md):
  Netflix golden tests as required status)
- Test-selection / regeneration rules
- Coding-standards changes (e.g.,
  [ADR-0012](0012-coding-standards-jpl-cert-misra.md))
- New user-visible flags or surfaces (e.g.,
  [ADR-0023](0023-tinyai-user-surfaces.md))

**Not** ADR-worthy: bug fixes, implementation details, one-off refactors that
don't change any interface or policy.

## Relation to `.workingdir2/`

Planning dossiers live under `.workingdir2/` (gitignored). Mirrored copies of
ADRs may exist there for local session continuity, but the tracked
`docs/adr/` tree is authoritative.

## Index

| ID | Title | Status | Tags |
| --- | --- | --- | --- |
| [ADR-0001](0001-stash-benchmark-noise-file.md) | Treat uncommitted benchmark result JSON as noise | Accepted | workspace, git, testing |
| [ADR-0002](0002-merge-path-master-default.md) | Merge path gpu-opt → sycl → master, master is fork default | Accepted | git, release, workspace |
| [ADR-0003](0003-workingdir2-empty-planning-dir.md) | Introduce `.workingdir2` as new planning directory | Accepted | workspace, planning, claude |
| [ADR-0004](0004-auto-push-after-merges.md) | Auto-push sycl and master to origin after merges | Accepted | git, ci, release |
| [ADR-0005](0005-framework-adaptation-full-scope.md) | Adopt full framework adaptation scope (a–g) | Accepted | framework, ci, docs, build, mcp |
| [ADR-0006](0006-cli-precision-17g-default.md) | Set CLI precision default to `%.17g` with `--precision` flag | Accepted | cli, testing, python |
| [ADR-0007](0007-claude-settings-fresh-rewrite.md) | Rewrite `.claude/settings.json` from scratch | Accepted | claude, agents |
| [ADR-0008](0008-readme-fork-rebrand.md) | Rewrite README with fork branding, preserve Netflix attribution | Accepted | docs, readme, license |
| [ADR-0009](0009-mcp-server-tool-surface.md) | MCP server exposes four core tools | Accepted | mcp, python, framework |
| [ADR-0010](0010-sigstore-keyless-signing.md) | Sign release artifacts keyless via Sigstore | Accepted | security, release, supply-chain |
| [ADR-0011](0011-versioning-lusoris-suffix.md) | Version scheme `v3.x.y-lusoris.N` | Accepted | release, framework |
| [ADR-0012](0012-coding-standards-jpl-cert-misra.md) | Coding standards stack: JPL + CERT + MISRA | Accepted | lint, docs, license |
| [ADR-0013](0013-local-dev-distro-matrix.md) | Support full local dev distro matrix | Accepted | build, docs, framework |
| [ADR-0014](0014-vscode-clangd-disable-ms-cpp.md) | VSCode uses clangd; disable MS C/C++ IntelliSense | Accepted | build, framework, lint |
| [ADR-0015](0015-ci-matrix-asan-ubsan-tsan.md) | CI matrix: Linux / macOS / Windows with sanitizers | Accepted | ci, testing, security |
| [ADR-0016](0016-sycl-to-master-merge-conflict-policy.md) | `sycl → master` merge conflict resolution policy | Accepted | git, workspace |
| [ADR-0017](0017-claude-skills-scope.md) | Claude skills scope includes domain scaffolding | Accepted | claude, agents, framework |
| [ADR-0018](0018-claude-hooks-scope.md) | Claude hooks scope: safety + auto-format + git | Accepted | claude, agents, ci, git |
| [ADR-0019](0019-workingdir2-full-dossier.md) | `.workingdir2` is the full planning dossier | Accepted | workspace, planning, docs |
| [ADR-0020](0020-tinyai-four-capabilities.md) | Tiny-AI scope covers all four capabilities | Accepted | ai, dnn, framework, cli |
| [ADR-0021](0021-training-stack-pytorch-lightning.md) | Training stack: PyTorch + Lightning with ONNX export | Accepted | ai, python, framework |
| [ADR-0022](0022-inference-runtime-onnx.md) | Inference runtime: ONNX Runtime via execution providers | Accepted | ai, dnn, cuda, sycl, build |
| [ADR-0023](0023-tinyai-user-surfaces.md) | Tiny-AI user surfaces: CLI, C API, ffmpeg, training | Accepted | ai, dnn, cli, framework |
| [ADR-0024](0024-netflix-golden-preserved.md) | Preserve Netflix source-of-truth tests verbatim | Accepted | testing, ci, license |
| [ADR-0025](0025-copyright-handling-dual-notice.md) | Copyright handling preserves Netflix, adds Lusoris/Claude | Accepted | license, docs |
| [ADR-0026](0026-workspace-relocated-under-python.md) | Relocate Python harness workspace under `python/vmaf/` | Accepted | workspace, python, docs |
| [ADR-0027](0027-non-conservative-image-pins.md) | Non-conservative image pins + experimental toolchain flags | Accepted | ci, cuda, sycl, build, supply-chain |
| [ADR-0028](0028-adr-maintenance-rule.md) | Every non-trivial decision gets an ADR before the commit | Accepted | docs, planning, agents |
| [ADR-0029](0029-resource-tree-relocated.md) | Relocate resource tree under `python/vmaf/` | Accepted | workspace, python, docs |
| [ADR-0030](0030-matlab-sources-relocated.md) | Relocate MATLAB sources under `python/vmaf/` | Accepted | workspace, matlab, python |
| [ADR-0031](0031-fork-docs-moved-under-docs.md) | Fork-added docs live under `docs/` | Accepted | docs, workspace |
| [ADR-0032](0032-unittest-script-moved-to-scripts.md) | Relocate root `unittest` script to `scripts/` | Accepted | testing, workspace |
| [ADR-0033](0033-codeql-config-moved-to-github.md) | Relocate CodeQL config to `.github/` | Accepted | security, ci, github |
| [ADR-0034](0034-single-patches-directory.md) | Delete `patches/` leftover; keep only `ffmpeg-patches/` | Accepted | workspace, build |
| [ADR-0035](0035-claude-hooks-schema-fix.md) | Migrate `.claude/settings.json` hooks to current schema | Accepted | claude, agents |
| [ADR-0036](0036-tinyai-wave1-scope-expansion.md) | Tiny-AI Wave 1 scope expanded beyond D20–D23 | Accepted | ai, dnn, cli, framework, mcp |
| [ADR-0037](0037-master-branch-protection.md) | Protect `master` branch on GitHub with required checks | Accepted | github, ci, security, release |
| [ADR-0038](0038-purge-upstream-matlab-mex-binaries.md) | Purge upstream MATLAB MEX compiled binaries from tree | Accepted | security, matlab, supply-chain |
| [ADR-0039](0039-onnx-runtime-op-walk-registry.md) | Pull forward runtime op-allowlist walk + model registry | Accepted | ai, dnn, security, supply-chain |
| [ADR-0040](0040-dnn-session-multi-input-api.md) | Extend DNN session API to multi-input/output with named bindings | Accepted | ai, dnn, cli |
| [ADR-0041](0041-lpips-sq-extractor.md) | Ship LPIPS-SqueezeNet FR extractor with inverse-ImageNet in graph | Accepted | ai, dnn, cli |
| [ADR-0042](0042-tinyai-docs-required-per-pr.md) | Tiny-AI PRs must ship human-readable docs in the same PR | Accepted | ai, dnn, docs |
| [ADR-0100](0100-project-wide-doc-substance-rule.md) | Every user-discoverable change ships docs in the same PR (project-wide) | Accepted | docs, agents, framework |
| [ADR-0103](0103-sycl-d3d11-surface-import.md) | Implement `vmaf_sycl_import_d3d11_surface` as staging-texture H2D path | Accepted | sycl, windows, api |
