# AGENTS.md — python/vmaf

Orientation for agents working on the Python bindings and the **classic**
(SVM-based) VMAF training / eval harness. Parent: [../../AGENTS.md](../../AGENTS.md).

## Scope

- Python bindings around libvmaf (`vmafrc`, `quality_runner`, …)
- The upstream Netflix training / analysis harness (SVM, MOS analysis, plots)
- Fork-local scratch and resource trees relocated from the repo root

Not in scope: tiny-AI training — that lives in [../../ai/](../../ai/AGENTS.md).

```
python/vmaf/
  config.py              # WORKSPACE / RESOURCE constants + env overrides
  workspace/             # classic-harness scratch (gitignored subtrees except placeholders)
  resource/              # example datasets + param files
  matlab/                # MATLAB reference implementations (strred, SpEED, STMAD, cid_icid)
  …                      # bindings + harness modules
```

## Ground rules

- **Parent rules** apply (see [../../AGENTS.md](../../AGENTS.md)).
- **Never commit Netflix golden-score changes.** The Python-side golden
  assertions in [../test/](../test/) are the numerical-correctness gate
  for VMAF — they run in CI as a required status check and are never
  modified by any PR. See
  [ADR-0024](../../docs/adr/0024-netflix-golden-preserved.md).
- **Never commit MEX / compiled MATLAB binaries**: upstream shipped ~53
  `.mexa64` / `.dll` / `.o` / `.lib` artefacts in `matlab/`; these were
  purged on 2026-04-17 and are blocked by `.gitignore`. See
  [ADR-0038](../../docs/adr/0038-purge-upstream-matlab-mex-binaries.md).
  The `.c` and `.m` sources stay — anyone needing the MATLAB path rebuilds
  locally with `mex file.c`.
- **Workspace and resource paths go through `config.py` constants**
  (`WORKSPACE`, `RESOURCE`). Overridable via `VMAF_WORKSPACE` /
  `VMAF_RESOURCE` env vars. See
  [ADR-0026](../../docs/adr/0026-workspace-relocated-under-python.md),
  [ADR-0029](../../docs/adr/0029-resource-tree-relocated.md).
- **Precision**: `result.py` serialises floats at `%.17g` by default,
  matching the CLI — see
  [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md).

## Governing ADRs

- [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md) — precision default.
- [ADR-0024](../../docs/adr/0024-netflix-golden-preserved.md) — Netflix goldens (Python-side).
- [ADR-0026](../../docs/adr/0026-workspace-relocated-under-python.md) — workspace relocation.
- [ADR-0029](../../docs/adr/0029-resource-tree-relocated.md) — resource tree relocation.
- [ADR-0030](../../docs/adr/0030-matlab-sources-relocated.md) — MATLAB source relocation.
- [ADR-0038](../../docs/adr/0038-purge-upstream-matlab-mex-binaries.md) — MEX binary purge.
