- **tests**: partial port of upstream Netflix/vmaf `python/test/`
  `MyTestCase` migration batch (2026-05-08, see ADR-0325). Cherry-picked
  four upstream commits — `38e905d1` (BD-rate `MyTestCase` + snake_case),
  `7df50f3a` (testutil full fixture set including 1/2/3/4-frame YUVs +
  4K/1080p cambi), `3a041a97` (`MyTestCase` adoption + sync of
  `feature_assembler` `vif`/`motion` assertions to fork-CPU output —
  values verified to match exactly to `places=4`), `e3827e4d`
  (asset / bootstrap / local_explainer `MyTestCase` adoption with rename
  of upstream's `test_fps_cmd` to `test_fps_cmd_via_filter_cmd` to avoid
  collision with the fork's pre-existing test). Eighteen further
  commits in the batch were deferred — they touch the four
  heavily-diverged Netflix-golden test files
  (`feature_extractor_test.py`, `vmafexec_feature_extractor_test.py`,
  `quality_runner_test.py`, `vmafexec_test.py`) and/or depend on YUV /
  dataset fixtures the fork does not ship. Netflix-golden gate output
  identical pre/post (9 failed, 162 passed, 2 skipped — pre-existing
  Python 3.14 / `numpy 2.x` repr + missing-`skimage` env failures
  only). CLAUDE.md §1 invariant preserved via per-hunk fork-CPU
  verification of every value-change.
