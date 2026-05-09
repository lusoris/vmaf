# `scripts/ci/` — agent invariants

Fork-local CI utilities. Anything in this directory is invoked from
`.github/workflows/*.yml` (see "Rebase-sensitive surfaces" below);
upstream Netflix/vmaf has no equivalent tree, so the rebase risk is
"workflow drift" rather than "merge conflict".

## Rebase-sensitive surfaces

The following pairs are tightly coupled — a rename or signature
change in one **must** land alongside the matching update in the
other, in the **same PR**. Required-status-check names are derived
from the workflow file's `name:` fields, so a check that gets dropped
or renamed turns into a phantom-required gate that blocks every PR
until master is fixed.

| Script | Workflow lane(s) that invoke it | What couples them |
|---|---|---|
| `cross_backend_vif_diff.py` | `tests-and-quality-gates.yml` — every `*-cross-backend-diff` step | The `--feature`, `--backend`, `--places` flag names; the `FEATURE_METRICS` dict (workflow steps reference feature names verbatim). |
| `cross_backend_parity_gate.py` | `tests-and-quality-gates.yml` — `Run GPU-parity matrix gate` step | The `--gpu-id`, `--calibration-table`, `--backends`, `--features`, `--fp16-features`, `--json-out`, `--md-out` flag names. The matrix-gate step name (`gpu-parity-matrix-gate`) is itself a required-status check on PRs. |
| `cross_backend_calibration.py` | (loader, not invoked directly by workflow) | Imported by the two gate scripts via `sys.path.insert(0, …)`; lives next to them on purpose. Don't move it without updating the import sites. |
| `gpu_ulp_calibration.yaml` | (data, not invoked directly by workflow) | The default path is hard-coded as `Path(__file__).parent / "gpu_ulp_calibration.yaml"` in `cross_backend_calibration.DEFAULT_CALIBRATION_PATH`. Renaming this file is a breaking change for the gate scripts and any caller that didn't pass `--calibration-table` explicitly. |
| `test_calibration.py` | `tests-and-quality-gates.yml` — pytest collection (`pytest-tests` lane) | Discovered automatically by pytest; the test module name is part of the gate's contract. |

## Calibration table contract (ADR-0234)

`gpu_ulp_calibration.yaml` is the single source of truth for
per-GPU-generation tolerance overrides on the cross-backend parity
gate. The lookup contract:

1. Caller passes `--gpu-id <runtime_id>` to the gate. ID format
   follows Research-0041:
   - `vulkan:0xVVVV:0xDDDD`
   - `cuda:M.m`
   - `sycl:0xVVVV:DRIVER`
2. The loader picks the most-specific glob match (longest non-
   wildcard prefix wins; trailing `*` is supported).
3. If a row has a `features:` override for the cell, that wins.
   Otherwise the gate falls back to its built-in
   `FEATURE_TOLERANCE` default (preserving backward compatibility
   for every caller that pre-dates ADR-0234).
4. If `--gpu-id` is omitted, no calibration is consulted at all
   (legacy behaviour exact).

**Invariant**: `tolerance_for(feature, gpu_id, default)` returns
`default` whenever any of the resolution steps above falls through.
This is enforced by `test_calibration.py`. A future PR that
"optimises" the lookup must keep all four fallback paths intact, or
existing CI lanes that don't pass `--gpu-id` will silently change
behaviour.

## When adding a new lane

1. New `--feature` value → add to `FEATURE_METRICS` in *both*
   gate scripts (single source of truth lives in the parity gate;
   the per-feature script mirrors it). Add the workflow step to
   `tests-and-quality-gates.yml`.
2. New backend → extend `BACKEND_SUFFIX`, `BACKEND_DEVICE_FLAG`,
   `BACKEND_DEFAULT_DEVICE` in both scripts.
3. New GPU arch → add a row to `gpu_ulp_calibration.yaml`. Mark it
   `status: placeholder` until a real-hardware corpus exists; the
   placeholder row is operationally a no-op (empty `features:`
   block).

## When updating from upstream

`scripts/ci/` is fork-introduced; nothing in here merges from
upstream. The risk on `/sync-upstream` is the opposite: an upstream
change to a feature extractor's emitted-metric names would silently
invalidate `FEATURE_METRICS` rows. Re-run the matrix gate after any
upstream sync that touches `libvmaf/src/feature/`.

## PR-body deliverables validator (`validate-pr-body.sh`)

`scripts/ci/validate-pr-body.sh` and `scripts/git-hooks/pre-push`
are local mirrors of the `.github/workflows/rule-enforcement.yml`
deep-dive-checklist gate (ADR-0108). They re-use
`scripts/ci/deliverables-check.sh` verbatim as the parser; the
validator only injects the diff via a `PATH`-shim that intercepts
`git diff --name-only`.

**Invariant — single parser source of truth**: do not fork or
re-implement the deliverables-check parsing logic in any other
language. If the gate's regex shape ever changes, the change lands
in `deliverables-check.sh` and the validator picks it up
automatically. The test harness `test-validate-pr-body.sh` should
catch any drift between the validator's expectations and the
parser's actual behaviour.

**Invariant — shim scope**: the `git` shim built inside
`validate-pr-body.sh` intercepts only the `diff --name-only` call
shape. Every other `git` invocation falls through to the real
binary. A future change to `deliverables-check.sh` that uses a
different git subcommand to compute the diff must update the shim
accordingly, or `validate-pr-body.sh` will silently use the real
git's output (potentially fine, potentially wrong depending on
local repo state).
