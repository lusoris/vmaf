# ADR-0418: macOS Python test recalibration post-VIF-sync (PR #758)

Status: Accepted
Date: 2026-05-11
Tags: testing, macos, vif, fork-local, upstream-deferred

## Context

PR #758 (`ADR-0416`) synced VIF to Netflix upstream's on-the-fly filter
(`bf9ad333` + companion test recalibrations `142c0671`, `7209110e`,
`d93495f5`, `fe756c9f`). Netflix only updated a subset of their golden
test assertions; **the following 9+ macOS-CI tests still reference the
pre-`bf9ad333` VMAF / ADM values and fail with sub-1% precision drift**:

- `local_explainer_test::test_explain_vmaf_results` — `VMAF_LE_score`
  76.68425574 → 76.66740228 (`places=4`)
- `vmafexec_test::test_run_vmafexec_runner_akiyo_multiply` — 132.732952
  → 132.732323 (`places=3`)
- `vmafexec_test::test_run_vmafexec_runner_akiyo_multiply_disable_enhn_gain` —
  88.030463 → 88.030322 (`places=4`)
- `vmafexec_test::test_run_vmafexec_runner_akiyo_multiply_no_enhn_gain_model` —
  same
- 5 × `vmafexec_feature_extractor::test_run_float_adm_fextractor_adm_*` —
  ADM scores shifted ~0.02–0.13 (the new upstream VIF feeds into ADM's
  `compute_dwt2_src_offset_adj`).

Netflix CI is Linux-only and the fork added macOS + Metal + MoltenVK as
fork-local CI surfaces. On Ubuntu, `tox -c python` skips entirely
because `envlist = py311` but the runner has Python 3.14 — so these
failures are only visible on the macOS lane, which uses a `brew`
Python 3.11 that actually runs the tests.

## Decision

Update the 9+ failing macOS-CI assertions to the post-`bf9ad333`
values observed in macOS CI output. Each updated line carries an
inline `# post-VIF-sync (#758) recal` comment so the next
`/sync-upstream` reviewer can spot fork-local divergence from
upstream test fixtures.

Add a `docs/rebase-notes.md` entry pointing at this ADR. The next
upstream sync that catches `bf9ad333`'s companion fixtures for these
test files should revert this PR's diff in favour of the upstream
values (the script naming the offending tests in rebase-notes makes
the revert mechanical).

## Consequences

Positive

- macOS clang (CPU), macOS clang (CPU) + DNN, macOS Metal jobs flip
  from red to green; the `local_explainer` + `vmafexec_*` assertions
  pass against the post-VIF-sync binary on macOS-libm precision.
- Ubuntu CI is unaffected (tox skips because of the py311 envlist
  mismatch; covered separately by ADR follow-up if/when that's
  fixed).
- The Netflix Golden D24 gate on Ubuntu (which DOES run those exact
  assertions through `make test-netflix-golden`) is unaffected
  because the `places=2-4` tolerance comfortably absorbs the ~1e-5
  drift on Ubuntu — the macOS failure was due to brew-Python's
  tox actually running the tests at the strict tolerance.

Negative / open

- Adopts upstream's eventual values **before** upstream has shipped
  them. If Netflix subsequently chooses different recalibrated
  values, our values will diverge until the next sync. The rebase
  note flags this for explicit review.
- Tightens fork-local divergence from upstream test fixtures by ~9
  assertions. Logged in `docs/rebase-notes.md` so the next
  `/sync-upstream` reverts them.

## Alternatives considered

- **Wait for upstream Netflix** to ship the companion fixtures.
  Rejected: master CI would stay red on macOS for the unknown
  number of weeks/months it takes upstream to ship. User-directed
  to fix now.
- **`@unittest.skipIf(Darwin)`** the failing tests.
  Rejected: violates `feedback_no_test_weakening` and the saved
  memory ("never lower thresholds, change baselines, or skip
  cases"). Recalibrating-with-rebase-note is the lesser evil
  because the values are still verified (just against the new
  binary).
- **Override `places` to `places=1`** on each test.
  Rejected: weakens precision contracts across the board; the new
  values are observable to `places=4` precision on macOS-libm so
  there's no need to widen the tolerance.

## References

- User direction (paraphrased, popup): "fork-recalibrate now, revert
  when upstream delivers; macOS CI is fork-added not upstream"
- ADR-0416 — VIF upstream sync (the source of the drift)
- `docs/rebase-notes.md` entry `fix/macos-test-recal-post-vif-sync`
- Upstream commits cherry-picked in #758 but missing the macOS
  companions: `142c0671`, `7209110e`, `d93495f5`, `fe756c9f`
