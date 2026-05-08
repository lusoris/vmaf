# ADR-0325: Partial port of upstream `python/test/MyTestCase` migration batch (2026-05-08)

- **Status**: Accepted
- **Date**: 2026-05-08
- **Deciders**: lusoris, Claude (Opus 4.7)
- **Tags**: python, tests, upstream-port, golden-data

## Context

Netflix/vmaf upstream landed a 22-commit batch on `master` between
2026-05-05 and 2026-05-06 that migrates the `python/test/` test suite
from `unittest.TestCase` to a fork-internal `MyTestCase` helper. The
batch also reformats BD-rate fixture data to snake_case + one-per-line
style, ports new tests for the upstream `aim`/`adm3`/`motion3` feature
options, loosens several macOS-FP-precision tolerances, and updates a
small set of existing `assertAlmostEqual` values.

The fork's `python/test/` tree has diverged from upstream's:

- Fork applies `black` + `isort` formatting (multi-line `Asset(...)`,
  double-quoted strings, explicit multi-line imports) — upstream uses
  single-quoted, single-line, ad-hoc-aligned style.
- Fork tightened tolerances on several assertions (e.g. `places=4`)
  beyond upstream's `places=2` / `places=3`.
- Fork is on a different `VmafFeatureExtractor` version
  (`V0.2.7`) for some assertions while upstream's batch assumes
  `V0.2.21`.
- Several upstream new tests (`322ca041`, `005988ea`) reference YUV
  fixture files (`src01_hrc00_576x324_Nframes.yuv`, mixed-resampling
  dataset python files) that are **not** present in the fork's
  `python/test/resource/`.

CLAUDE.md §1 hard rule: **"NEVER modify Netflix golden-data
assertions"**. Any value-change hunk in upstream that would alter an
asserted number must be inspected against the fork's CPU output before
landing.

## Decision

Port the **tractable subset** of the 22-commit batch — four commits
(38e905d1, 7df50f3a, 3a041a97, e3827e4d) covering files where the
fork's content delta is purely stylistic and the upstream commit either
adds net-new tests, syncs values that already match the fork's CPU
output, or loosens tolerances. Defer the remaining 18 commits to a
follow-up port effort because:

- They touch heavily-diverged test files
  (`feature_extractor_test.py`, `vmafexec_feature_extractor_test.py`,
  `quality_runner_test.py`, `vmafexec_test.py`) where every cherry-pick
  conflicts on dozens of regions and resolution requires running the
  fork's CPU per assertion to verify whether each value-change hunk is
  a "sync" (OK) or an "upstream-only drift" (reject per §1). Doing 18
  such ports in one session compromises the §1 invariant.
- Several deferred commits add tests that depend on YUV fixtures the
  fork does not ship (`*_Nframes.yuv`,
  `test_read_dataset_dataset_mixed_resampling_types*.py`); landing
  them would introduce file-not-found ERRORs in CI.
- Upstream's `cf02b126` ("align whitespace and import order")
  *removes* PEP-E302 blank lines between top-level functions —
  anti-aligned with the fork's `black` formatting profile. That
  commit is rejected.

The four ports applied:

| Upstream SHA | Files | Classification |
|---|---|---|
| `38e905d1` | `python/test/bd_rate_calculator_test.py` | Structural — `MyTestCase` adoption + snake_case fixture rename. No assertion-value changes. |
| `7df50f3a` | `python/test/testutil.py` | Structural — adds new fixture functions for 1/2/3/4-/5-frame YUVs and 4K/1080p cambi tests. Existing functions reformatted via `black`. |
| `3a041a97` | `python/test/feature_assembler_test.py`, `command_line_test.py`, `noref_feature_extractor_test.py`, `perf_metric_test.py`, `train_test_model_test.py` | Mixed — `MyTestCase` adoption + value updates (`vif_score 0.4460930625 -> 0.44641922916`, `motion_score 4.0498253541 -> 4.0488208125`). **Verified the fork's CPU produces upstream's new values** (vif=`0.44641922916666665`, motion=`4.0488208125`). The old values were within `places=2` / `places=3` tolerance only; the new values match exactly with `places=4`. Hence this is a §1-compatible *sync*, not a value drift. |
| `e3827e4d` | `python/test/asset_test.py`, `bootstrap_train_test_model_test.py`, `local_explainer_test.py` | Structural + new tests. `asset_test.py`'s upstream `test_fps_cmd` collides with the fork's pre-existing `test_fps_cmd` (different API surface — fork tests `asset.fps_cmd`, upstream tests `asset.get_filter_cmd("fps")`). Renamed upstream's to `test_fps_cmd_via_filter_cmd` to keep both. |

The Netflix golden-data gate (`make test-netflix-golden`-equivalent
target) was run before and after the port:

- Baseline: 9 failed, 162 passed, 2 skipped, 1 error (pre-existing
  Python 3.14 / `numpy 2.x` repr issues + missing `skimage` /
  `libsvm.svmutil.RBF` / niqe model loader env issues — unrelated to
  this port).
- Post-port: **identical** — 9 failed, 162 passed, 2 skipped, 1 error.
  No new test failures. No assertion changed produced a regression.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Port all 22 commits with manual conflict resolution** | Full upstream parity; no follow-up debt | 18 of 22 commits conflict heavily on the four heavily-diverged Netflix-golden test files; resolving each requires per-assertion CPU verification under §1 — order-of-magnitude more work than a single PR can absorb safely | Risk of compromising §1 by mis-classifying a value-change as a "sync" outweighs the parity benefit; deferring to a focused follow-up PR keeps the §1 audit trail per-commit. |
| **Port only the structural commits (this ADR)** | Bounded scope, easy review, §1 invariant verifiably preserved (Netflix-golden gate identical pre/post) | Fork remains 18 commits behind on test-suite parity; future upstream syncs accumulate further drift | **Chosen.** The four ports cover the BD-rate, testutil, feature_assembler/command_line/perf_metric/train_test_model, and asset/bootstrap/local_explainer surfaces — net-new fixture functions land in the fork now and are reusable by future ports. |
| **Skip the entire batch** | Zero risk | Defers a guaranteed-tractable subset for no benefit; future syncs will face the same conflicts plus more | The four tractable commits have already been verified as §1-safe; banking them now reduces follow-up scope. |

## Consequences

- **Positive**: Fork's BD-rate, testutil, feature_assembler,
  command_line, perf_metric, train_test_model, asset, bootstrap, and
  local_explainer test files are now structurally aligned with
  upstream's `MyTestCase` migration. The `testutil` fixture surface is
  full-superset (1/2/3/4-/5-frame YUVs + 4K/1080p cambi). New
  `local_explainer` and `bootstrap_train_test_model` tests are
  available for use. Future ports of upstream test commits that touch
  these nine files will have far fewer conflicts.
- **Negative**: Eighteen of 22 upstream commits remain unported. They
  accumulate as port debt. The four heavily-diverged test files
  (`feature_extractor_test.py`, `vmafexec_feature_extractor_test.py`,
  `quality_runner_test.py`, `vmafexec_test.py`) still need a
  follow-up port that walks each assertion through a CPU verification
  pass.
- **Neutral / follow-ups**:
  - File a follow-up issue tracking the deferred 18 commits, with
    per-commit notes on conflict shape and the §1 verification cost.
  - The next time someone runs `/sync-upstream` against `python/test/`,
    pre-flight whether the missing YUV / dataset fixtures
    (`src01_hrc00_576x324_*frames.yuv`,
    `test_read_dataset_dataset_mixed_resampling_types*.py`) ship in
    `vmaf_resource` and pull them in if so.

## References

- req: "Port the upstream python/test MyTestCase migration onto the
  lusoris/vmaf fork. Read CLAUDE.md carefully — especially §1 (NEVER
  modify Netflix golden assertions) and §8 (Netflix golden-data gate).
  This task has a strict hard-edge: assertion values MUST stay
  byte-identical."
- req: "If the value changes match what fork's CPU runs already
  produce, it's a sync (OK). If the change is upstream-only (e.g.
  their CI's score), reject the value change and keep the fork's
  existing assertion. The fork's CPU golden run is the source of
  truth, NOT upstream's."
- Upstream batch (chronological):
  - `7d1ad54b`, `d93495f5`, `9fa593eb`, `0341f730`, `3cbf352d`,
    `eb3374d0`, `a3776335`, `74bdce1b`, `a333ba4c`, `403dafed`,
    `322ca041`, `7df50f3a`, `6c097fc4`, `ead2d12b`, `cf02b126`,
    `3a041a97`, `25ff9f18`, `e3827e4d`, `38e905d1`, `005988ea`,
    `4679db83`, `3e075107`.
- Related ADRs:
  - [ADR-0028](0028-adr-maintenance-rule.md) — ADR maintenance rule.
  - [CLAUDE.md §1, §8, §12 r1](../../CLAUDE.md) — Netflix golden-data
    gate.
