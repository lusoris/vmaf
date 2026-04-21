# Rebase notes

Single ledger of fork-local changes that need attention when this fork
syncs from `upstream/master` (Netflix/vmaf). Required by
[ADR-0108](adr/0108-deep-dive-deliverables-rule.md): every fork-local
PR that touches upstream-shared paths or establishes a rebase-sensitive
invariant adds an entry here. PRs with no rebase impact state "no
rebase impact" in the PR description and skip the entry.

The intended reader is whoever runs the next `/sync-upstream` (see
[ADR-0002](adr/0002-merge-path-master-default.md) and
`.claude/skills/sync-upstream/`). Read top-to-bottom before resolving
conflicts.

## Format

Each entry is a `### NNNN — short title` heading with three fields:

- **Touches**: paths likely to conflict on upstream merge.
- **Invariant**: what the fork relies on that an upstream change could
  silently drop.
- **Re-test**: the command(s) to run after the merge to confirm the
  invariant survived. Reproducer-style — no surrounding prose required.

IDs are assigned in commit order and never reused. A single entry may
cover several PRs in one workstream; cross-link from the ID heading.

## Entries (backfilled 2026-04-18 per ADR-0108 adoption)

The pre-ADR-0108 fork-local PRs are summarised by workstream rather
than per-PR. Future PRs add entries individually.

### 0001 — SIMD bit-identical reductions for float ADM

- **Workstream PRs**: #18, commits `24c88a32`, `f082cfd3`.
- **Touches**:
  `libvmaf/src/feature/integer_adm.c`,
  `libvmaf/src/feature/float_adm.c`,
  `libvmaf/src/feature/x86/adm_avx2.c`,
  `libvmaf/src/feature/x86/adm_avx512.c`,
  `libvmaf/src/feature/arm64/adm_neon.c`,
  upstream `python/test/feature_extractor_test.py` test expectations.
- **Invariant**: `sum_cube` and `csf_den_scale` accumulate cubed
  values in **double precision** (via `_mm256_cvtps_pd` /
  `_mm512_cvtps_pd`) in scalar, AVX2, AVX-512, and NEON. Upstream
  accumulates in float, which produces ~8e-5 drift between scalar and
  SIMD. Test expectations were tightened to match the double-precision
  path; an upstream-side accumulator change would re-introduce the
  drift and break the tightened assertions.
- **Re-test**: `meson test -C build --suite=fast && python -m pytest
  python/test/feature_extractor_test.py -k adm`.

### 0002 — CUDA ADM decouple-inline buffer elimination

- **Workstream PRs**: commit `787e3382`.
- **Touches**:
  `libvmaf/src/feature/cuda/integer_adm_cuda.cu`,
  `libvmaf/src/feature/cuda/adm_decouple_inline.cuh` (new),
  `libvmaf/src/feature/cuda/meson.build`. Upstream's
  `adm_decouple.cu` is no longer compiled in the fork.
- **Invariant**: CSF and CM CUDA kernels read `ref` / `dis` DWT2
  buffers directly and compute `decouple_r` / `decouple_a` inline via
  `__device__` helpers in `adm_decouple_inline.cuh`. The 6 intermediate
  buffers (`decouple_r`, `decouple_a`, `csf_a` × {scale-0 int16,
  scales 1-3 int32}) and the standalone `adm_decouple.cu` source are
  intentionally removed. ~107 MB GPU memory savings at 4K. An upstream
  change to `adm_decouple.cu` will look orphaned and a literal merge
  would re-introduce the buffer allocations.
- **Re-test**: `meson setup build -Denable_cuda=true && ninja -C build
  && meson test -C build --suite=cuda`.

### 0003 — SYCL backend (USM pool / D3D11 import / `vmaf_sycl_*` API)

- **Workstream PRs**: #33, #35, #5 (initial scaffolding), and the
  picture-pool deadlock fix that landed via #32.
- **Touches**:
  `libvmaf/include/libvmaf/libvmaf_sycl.h`,
  `libvmaf/src/sycl/`,
  `libvmaf/src/feature/sycl/`,
  `libvmaf/src/libvmaf.c` (SYCL public-API entry points),
  `meson_options.txt` (`enable_sycl`).
- **Invariant**: `vmaf_sycl_preallocate_pictures` constructs a real
  `VmafSyclPicturePool` honoring `VmafSyclPicturePreallocationMethod`
  (`NONE` / `DEVICE` / `HOST`); `vmaf_sycl_picture_fetch` dispatches
  to the pool when configured. The whole SYCL tree is fork-local and
  has no upstream counterpart — upstream changes to
  `libvmaf/src/libvmaf.c` near the SYCL entry-point block are likely
  to conflict. Picture-pool error paths in `vmaf_read_pictures`
  (libvmaf.c) must `goto cleanup;` rather than `return err;` to avoid
  leaking ref/dist pictures into the live-picture set (closes the
  always-on-pool deadlock fixed in #32 — see ADR-0104). See
  [ADR-0101](adr/0101-sycl-usm-picture-pool.md),
  [ADR-0103](adr/0103-sycl-d3d11-surface-import.md),
  [ADR-0104](adr/0104-picture-pool-always-on.md).
- **Re-test**: `meson setup build -Denable_sycl=true && ninja -C build
  && meson test -C build --suite=sycl` (requires oneAPI / icpx).

### 0004 — DNN runtime + tiny-AI surfaces

- **Workstream PRs**: #5, #8, #21, #22, #23, #31, #34, plus the
  pre-numbered DNN feat commits (`9b985946`, `1e5336d3`, `d122b721`).
- **Touches**:
  `libvmaf/include/libvmaf/dnn.h`,
  `libvmaf/src/dnn/`,
  `libvmaf/src/feature/feature_lpips.c`,
  `model/tiny/`,
  `meson_options.txt` (`enable_onnxruntime`).
- **Invariant**: ordered EP selection (CUDA → DML → CPU) with graceful
  fallback (ADR-0102); `fp16_io` does host-side fp32↔fp16 cast on the
  scoring path; `VMAF_TINY_MODEL_DIR` enforces a path jail on model
  load (PR #31); the runtime op-allowlist (PR #21) walks the ONNX
  graph and rejects unknown ops + bounds Loop/If `trip_count` at 1024
  (ADR-0036/0107). DNN tree is fork-local; upstream has no DNN code
  yet, so conflicts here are unlikely but the
  `meson_options.txt` and `libvmaf/src/meson.build` blocks
  near the DNN flag may collide.
- **Re-test**: `meson setup build -Denable_onnxruntime=true && ninja
  -C build && meson test -C build --suite=dnn`.

### 0005 — `--precision` CLI flag (IEEE-754 round-trip lossless)

- **Workstream PRs**: commit `c989fbd9`.
- **Touches**:
  `libvmaf/tools/vmaf.c`,
  `libvmaf/tools/cli_parse.c`,
  `libvmaf/include/libvmaf/libvmaf.h` (added
  `vmaf_write_output_with_format`),
  `libvmaf/src/output.c`.
- **Invariant**: default `--precision` is `%.17g` (round-trip
  lossless); `legacy` opts back into upstream's `%.6f`; the public C
  API gained `vmaf_write_output_with_format` and the old
  `vmaf_write_output` routes through it with the `%.17g` default.
  ABI-breaking only if upstream adds a same-named function with a
  different signature. See [ADR-0006](adr/0006-cli-precision-17g-default.md).
- **Re-test**: `vmaf -r ref.yuv -d dis.yuv ... --precision=full` and
  diff against `--precision=legacy`.

### 0006 — Netflix golden tests preserved verbatim as required gate

- **Workstream PRs**: across the fork's life; codified in
  [ADR-0024](adr/0024-netflix-golden-preserved.md).
- **Touches**: `python/test/quality_runner_test.py`,
  `python/test/vmafexec_test.py`,
  `python/test/vmafexec_feature_extractor_test.py`,
  `python/test/feature_extractor_test.py`,
  `python/test/result_test.py`,
  `python/test/resource/yuv/`.
- **Invariant**: `assertAlmostEqual(...)` golden values in the five
  upstream Python test files are **never modified by this fork**.
  Fork-added tests live in separate files (e.g.
  `python/test/test_precision_flag.py`). The CI gate "Netflix CPU
  golden tests (D24)" is required and blocks merge. Upstream changes
  to these files are accepted unless they relax the assertions.
- **Re-test**: `make test-netflix-golden`.

### 0007 — Build system (CUDA 13.2, oneAPI 2025.3, MkDocs migration)

- **Workstream PRs**: #7, #17, commit `8a995cb0`.
- **Touches**: `meson.build`, `meson_options.txt`, top-level
  `Makefile`, `docs/` (Sphinx → MkDocs Material migration —
  `docs/conf.py` removed, `mkdocs.yml` added),
  `docs/requirements.txt`,
  `Dockerfile.*`, distro install scripts under `scripts/`.
- **Invariant**: image pins are non-conservative
  ([ADR-0027](adr/0027-non-conservative-image-pins.md)) — CUDA 13.2,
  oneAPI 2025.3, clang-format 22, black 26 — and ship experimental
  toolchain flags (`--expt-relaxed-constexpr`, etc.) deliberately. An
  upstream sync that pulls in a Dockerfile change targeted at older
  CUDA or older oneAPI must not relax the pins.
- **Re-test**: `meson setup build -Denable_cuda=true
  -Denable_sycl=true && ninja -C build && mkdocs build --strict`.

### 0008 — Workspace / docs / MATLAB / resource-tree relocations

- **Workstream PRs**: codified across [ADR-0026](adr/0026-workspace-relocated-under-python.md),
  [ADR-0029](adr/0029-resource-tree-relocated.md),
  [ADR-0030](adr/0030-matlab-sources-relocated.md),
  [ADR-0031](adr/0031-fork-docs-moved-under-docs.md),
  [ADR-0032](adr/0032-unittest-script-moved-to-scripts.md),
  [ADR-0033](adr/0033-codeql-config-moved-to-github.md),
  [ADR-0034](adr/0034-single-patches-directory.md),
  [ADR-0038](adr/0038-purge-upstream-matlab-mex-binaries.md).
- **Touches**: any path-walk in upstream's CI / scripts / docs that
  assumes the upstream layout (root-level `workspace/`,
  `resource/`, `matlab/`, root `unittest` script, root `patches/`).
- **Invariant**: the fork's layout is `python/vmaf/workspace/`,
  `python/vmaf/resource/`, `python/vmaf/matlab/`, `scripts/unittest`,
  `ffmpeg-patches/` only, `.github/codeql-config.yml`. Upstream moves
  to a different sub-tree (e.g. a hypothetical
  `tools/workspace/`) need to either be applied via a corresponding
  fork-side relocation or rejected with a rebase note.
- **Re-test**: `python -m pytest python/test/ -k golden` (verifies the
  resource-tree path works); `make test-netflix-golden`.

### 0009 — License headers (Lusoris/Claude on wholly-new files

2016–2026 on Netflix files)

- **Workstream PRs**: commits `c159761d`, `a185f8ef`, `0e98c949`, codified
  in [ADR-0025](adr/0025-copyright-handling-dual-notice.md) /
  [ADR-0105](adr/0105-copyright-handling-dual-notice.md).
- **Touches**: every wholly-new fork file (notably the SYCL tree and
  `libvmaf/src/dnn/`) and every Netflix-touched file (year range
  `2016 → 2016–2026`).
- **Invariant**: wholly-new fork files carry
  `Copyright 2026 Lusoris and Claude (Anthropic)` under the same
  BSD-3-Clause-Plus-Patent license; mixed files use a dual-copyright
  notice. An upstream commit that resets a Netflix file's year range
  (e.g. back to `2016–2020`) must be partially rejected — keep the
  fork's `2016–2026`.
- **Re-test**: grep that wholly-new fork files retain the Lusoris/Claude
  header (`grep -L "Copyright 2026 Lusoris" libvmaf/src/sycl/*.cpp` —
  expected to match nothing).

### 0010 — `.claude/` agent scaffolding + ADR tree + AGENTS.md / CLAUDE.md

- **Workstream PRs**: #14, #24, #37, plus continuous additions.
- **Touches**: `.claude/`, `AGENTS.md`, `CLAUDE.md`, `docs/adr/`,
  `.github/PULL_REQUEST_TEMPLATE.md`.
- **Invariant**: this whole tree is fork-local and has no upstream
  counterpart. Upstream additions to `.github/` (issue templates,
  workflows) need to merge cleanly with the fork's existing files
  rather than replacing them. The ADR tree's IDs ≤ 0099 are
  *backfills*; new decisions start at 0100
  ([ADR-0028](adr/0028-adr-maintenance-rule.md) /
  [ADR-0106](adr/0106-adr-maintenance-rule.md)).
- **Re-test**: visual review of `.github/` and `docs/adr/README.md`
  after the merge.

---

*Pre-ADR-0108 entries above are the result of a one-shot backfill
sweep on 2026-04-18; subsequent fork-local PRs add their own entries
inline.*

### 0011 — Nightly bisect-model-quality + fixture cache

- **Workstream PRs**: closes #4; sticky tracker issue #40.
- **Touches**:
  `.github/workflows/nightly-bisect.yml`,
  `ai/scripts/build_bisect_cache.py`,
  `ai/testdata/bisect/{features.parquet, models/*.onnx, README.md}`,
  `scripts/ci/post-bisect-comment.py`,
  `docs/ai/bisect-model-quality.md`,
  `docs/adr/0109-nightly-bisect-model-quality.md`,
  `docs/research/0001-bisect-model-quality-cache.md`,
  `mkdocs.yml` (nav).
- **Invariant**: the committed parquet + ONNX bytes under
  `ai/testdata/bisect/` must regenerate **byte-identically** from
  `ai/scripts/build_bisect_cache.py` with seeds `FEATURE_SEED=20260418`
  and `MODEL_SEED=20260419`. The CI `--check` step asserts this before
  every bisect run, so any upstream pull that bumps `pandas` /
  `pyarrow` / `onnx` enough to change the serialiser bytes will fail
  the workflow until the cache is regenerated and committed.
- **Re-test**:

  ```bash
  python ai/scripts/build_bisect_cache.py --check
  vmaf-train bisect-model-quality \
      ai/testdata/bisect/models/model_*.onnx \
      --features ai/testdata/bisect/features.parquet \
      --min-plcc 0.85 --input-name input
  # Expected: "no regression in this range"; first_bad_index None.
  ```

  Pure upstream code is not touched, so no Netflix-side conflict
  vector. Only fork-local files; risk is toolchain drift, not merge
  conflict.

### 0012 — Upstream ADM port (Netflix `966be8d5`)

- **Workstream PRs**: this PR; ports a single upstream commit.
- **Touches**:
  `libvmaf/src/feature/integer_adm.{c,h}`,
  `libvmaf/src/feature/x86/adm_avx2.{c,h}`,
  `libvmaf/src/feature/x86/adm_avx512.{c,h}`,
  `libvmaf/src/feature/alias.c`,
  `libvmaf/src/feature/barten_csf_tools.h` (new upstream file).
- **Invariant**: the eight ADM files now mirror upstream's content
  byte-for-byte (modulo our clang-format-22 pass and the Netflix
  copyright-year bump on the new header). Future `/sync-upstream`
  runs can take new upstream ADM commits cleanly. **Do not** revert
  to a pre-`966be8d5` ADM kernel without also reverting the call-site
  signatures in `integer_compute_adm` — upstream extended
  `i4_adm_cm` from 8 to 13 args.
- **Re-test**:

  ```bash
  ninja -C libvmaf/build && meson test -C libvmaf/build
  libvmaf/build/tools/vmaf -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
      -d python/test/resource/yuv/src01_hrc01_576x324.yuv \
      -w 576 -h 324 -p 420 -b 8 \
      --model version=vmaf_v0.6.1 -o /tmp/vmaf-port.json
  grep '<metric name="vmaf"' /tmp/vmaf-port.json
  # Expected: mean ≈ 76.66890 (golden 76.66890519623612, places=4 OK).
  ```

### 0013 — Upstream motion port (Netflix PR #1486 head `2aab9ef1`)

- **Workstream PRs**: this PR; ports upstream PR #1486 (4 commits on top
  of `966be8d5` ADM base, head `2aab9ef1`). Sister to entry 0012.
- **Touches**:
  `libvmaf/src/feature/integer_motion.{c,h}`,
  `libvmaf/src/feature/motion_blend_tools.h` (new upstream file),
  `libvmaf/src/feature/x86/motion_avx2.c`,
  `libvmaf/src/feature/x86/motion_avx512.c`,
  `libvmaf/src/feature/alias.c` (additive: `integer_motion3` row),
  `python/test/{quality_runner,vmafexec,feature_extractor,vmafexec_feature_extractor}_test.py`
  (golden tolerance updates: `places=4` → `places=2` on motion-affected
  asserts; expected values unchanged).
- **Invariant**: motion files mirror upstream byte-for-byte (modulo our
  clang-format-22 pass). The `alias.c` row for `integer_motion3` was
  inserted surgically to avoid clobbering the AVX-512 ADM registration
  added by entry 0012; new motion3 metric appears in default VMAF model
  output but is not standalone-loadable via `--feature integer_motion3`
  (sub-feature only). Netflix golden VMAF mean shifts
  `76.668904824` → `76.667830213` (well within `places=2` tolerance the
  upstream PR loosened to). **Do not** revert `places=4` on
  motion-touching assertions without also reverting the motion code.
- **Re-test**:

  ```bash
  ninja -C libvmaf/build && meson test -C libvmaf/build
  libvmaf/build/tools/vmaf -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
      -d python/test/resource/yuv/src01_hrc01_576x324.yuv \
      -w 576 -h 324 -p 420 -b 8 \
      --model version=vmaf_v0.6.1 -o /tmp/vmaf-motion-port.json
  grep -E '<metric name="vmaf"|integer_motion3' /tmp/vmaf-motion-port.json
  # Expected: vmaf mean ≈ 76.66783; integer_motion3 mean ≈ 3.98976.
  ```

### 0014 — Coverage gate overhaul + upstream `python/test/` reformat

- **Workstream PRs**: this PR (coverage-gate overhaul + in-tree reformat
  of upstream-mirror Python tests).
- **Touches**:
  `.github/workflows/ci.yml` (CPU + GPU coverage jobs:
  `-Dc_args=-fprofile-update=atomic` / `-Dcpp_args=-fprofile-update=atomic`,
  `meson test --num-processes 1`, `-Denable_dnn=enabled`, ORT install
  step on the CPU coverage job, `lcov`/`geninfo` replaced by `gcovr`
  with `--json-summary` / `--xml` / `--txt` output, artifact rename
  `coverage-lcov-{cpu,gpu}` → `coverage-{cpu,gpu}`),
  `scripts/ci/coverage-check.sh` (rewritten to parse gcovr JSON via
  `python3 -c` — same CLI signature),
  `libvmaf/src/dnn/dnn_api.c` + new `libvmaf/src/dnn/dnn_attach_api.c`
  (`vmaf_use_tiny_model` carved out into its own TU so the unit-test
  binaries — which pull in `dnn_sources` for `feature_lpips.c` but
  never link `libvmaf.c` — don't end up with an undefined reference
  to `vmaf_ctx_dnn_attach` once `enable_dnn=enabled` activates the
  real bodies),
  `libvmaf/src/dnn/meson.build` + `libvmaf/src/meson.build`
  (new `dnn_libvmaf_only_sources` list wired into `libvmaf.so` only),
  `python/test/{feature_extractor,quality_runner,vmafexec,vmafexec_feature_extractor}_test.py`
  (mechanical Black + isort reformat — no assertion values changed,
  imports regrouped, line wrapping normalised).
- **Invariant**: coverage CI must keep all five pieces in lockstep —
  (a) `-fprofile-update=atomic` closes the intra-process counter race
  on SIMD inner loops (`vif_avx2.c:673`, `motion_avx2`, etc.) →
  negative counts → `geninfo`/gcovr abort; (b) `--num-processes 1`
  closes the inter-process race where multiple parallel test binaries
  merge their counters into the same `.gcda` files for the shared
  `libvmaf.so` at process exit (per-thread atomicity does not cover
  this); (c) `gcovr` deduplicates `.gcno` files belonging to the
  same source compiled into multiple targets — without dedup, lcov
  sums hits across compilation units and yields impossible
  >100% values (`dnn_api.c — 1176%` was the smoking gun on the first
  attempt that had only (a)+(b)); (d) ORT install + `enable_dnn=enabled`
  in the coverage job is what makes `libvmaf/src/dnn/*.c` measurable
  in the first place — without ORT, the DNN tree compiles in stub
  branches and the 85% per-critical-file gate is meaningless;
  (e) `vmaf_use_tiny_model` lives in `dnn_attach_api.c` and is added
  to `libvmaf.so` only via `dnn_libvmaf_only_sources` — moving it
  back into `dnn_api.c` reintroduces the `vmaf_ctx_dnn_attach`
  undefined-reference link error in `test_feature_extractor` /
  `test_lpips` whenever `enable_dnn=enabled`, since those test
  binaries pull in `dnn_sources` for `feature_lpips.c` but never
  link `libvmaf.c`. Lint
  scope: upstream-mirror Python tests are linted at the same standard
  as fork-added code; we accept that `/sync-upstream` and
  `/port-upstream-commit` will re-trigger Black/isort failures
  whenever upstream rewrites these files, and the fix is another
  in-tree reformat pass — never an exclusion. The fork's
  `pyproject.toml` and `.pre-commit-config.yaml` keep
  `python/test/resource/` (binary fixtures only) excluded;
  `python/test/*.py` is in scope. See
  [ADR-0110](adr/0110-coverage-gate-fprofile-update-atomic.md) (race
  fixes, superseded) and
  [ADR-0111](adr/0111-coverage-gate-gcovr-with-ort.md) (gcovr + ORT
  layer).
- **Re-test**:

  ```bash
  # Reproduce coverage path locally (requires gcc + python3-pip):
  pip install --user 'gcovr>=8.0'
  cd libvmaf
  meson setup build-cov-test --buildtype=debug -Db_coverage=true \
      -Denable_avx512=true -Denable_float=true -Denable_dnn=disabled \
      -Dc_args=-fprofile-update=atomic -Dcpp_args=-fprofile-update=atomic
  ninja -C build-cov-test
  meson test -C build-cov-test --print-errorlogs --num-processes 1
  ~/.local/bin/gcovr --root .. \
      --filter 'src/.*' \
      --exclude '.*/test/.*' --exclude '.*/tests/.*' \
      --exclude '.*/subprojects/.*' \
      --gcov-ignore-parse-errors=negative_hits.warn \
      --gcov-ignore-parse-errors=suspicious_hits.warn \
      --print-summary --txt build-cov-test/coverage.txt \
      --json-summary build-cov-test/coverage.json \
      build-cov-test
  grep -E 'dnn_api|model_loader' build-cov-test/coverage.txt
  # Expected: gcovr completes without "Unexpected negative count" AND no
  # per-file percentages exceed 100% (drop --num-processes 1 to reproduce
  # the multi-process .gcda merge race; switch back to lcov to reproduce
  # the dnn_api.c — 1176% over-count from compilation-unit summation).

  # Lint smoke test for upstream-mirror tree:
  pre-commit run --files python/test/quality_runner_test.py
  # Expected: Black/isort/Ruff all PASS — files are reformatted in-tree
  # to fork style and stay clean until the next upstream sync.
  ```

### 0015 — Tox doctest collection skips `vmaf/resource/`

- **Workstream PRs**: this PR (`fix(ci): skip pytest doctest collection
  of vmaf/resource/ data files`). Surfaced once ADR-0115 consolidated
  CI triggers to `master` and tox actually started running on PRs.
- **Touches**: `python/tox.ini` (single-line `--ignore=vmaf/resource`
  added to the pytest invocation, plus an explanatory comment block).
  Pure fork-local; no upstream Python file changes.
- **Invariant**: `pytest --doctest-modules` must not attempt to import
  files under `python/vmaf/resource/`. Those are parameter / dataset /
  example-config `.py` files; several have dots in their stems (e.g.
  `vmaf_v7.2_bootstrap.py`) that make them unimportable as Python
  modules. None carry doctests, so the ignore is correctness rather
  than a workaround. **Do not** drop the `--ignore=vmaf/resource`
  flag without first verifying every file under that directory has
  been renamed to a dot-free stem and is importable.
- **Re-test**:

  ```bash
  cd python && tox -e py311 -- --collect-only --doctest-modules \
      --ignore=vmaf/resource 2>&1 | grep -c "ERROR collecting vmaf/resource"
  # Expected: 0 (was 5 before the fix).
  ```

  Pure upstream code is not touched, so no Netflix-side conflict
  vector. Risk is upstream renaming or removing files under
  `python/vmaf/resource/` such that the directory disappears, in
  which case the `--ignore` becomes a harmless no-op.

### 0016 — SYCL `-fsycl` link-arg gated on icpx CXX

- **Workstream PRs**: this PR (`fix(libvmaf): gate -fsycl link arg on
  icpx CXX, allow gcc/clang host linker`). Surfaced once
  ADR-0115's CI consolidation added an Ubuntu SYCL job to PR-time CI
  that uses `CXX=g++` (host linker) with sidecar icpx for SYCL .cpp
  compilation.
- **Touches**: `libvmaf/src/meson.build` (the `vmaf_link_args` block
  immediately after the `is_sycl_enabled` flag handling — currently
  ~lines 696-712). Pure fork-local; no upstream Meson file changes
  expected.
- **Invariant**: `-fsycl` is appended to `vmaf_link_args` **only** when
  `meson.get_compiler('cpp').get_id() == 'intel-llvm'` (icpx).
  Rationale: the documented project mode (see comment near `is_sycl_enabled`
  block at top of `src/meson.build`) compiles SYCL `.cpp` files via
  `custom_target` with icpx, while the project's CXX driver may be gcc /
  clang / msvc; in that mode the SPIR-V device code is already embedded
  in the icpx-compiled `.o` files at compile time, and the runtime
  libraries (`libsycl` + `libsvml` + `libirc` + `libze_loader`) declared
  as link dependencies resolve every symbol. Passing `-fsycl` to a
  non-icpx linker is a hard error
  (`g++: error: unrecognized command-line option '-fsycl'`). **Do not**
  remove the `cpp.get_id() == 'intel-llvm'` guard without first verifying
  every CI matrix leg uses icpx as the project CXX.
- **Re-test**:

  ```bash
  meson setup build -Denable_sycl=true \
      -Dcpp_link_args=-Wl,--no-undefined
  ninja -C build src/libvmaf.so.3
  # Expected: link succeeds; no `-fsycl` errors with gcc/clang host CXX.
  ```

  Pure fork-local guard; no Netflix-side conflict vector.

### 0017 — CLI precision default `%.6f` (Netflix-compat) + frame-skip unref

- **Workstream PRs**: this PR (`fix(cli): revert precision default to
  %.6f and unref skipped frames`). Reverts the default flipped by
  commit `c989fbd9` (ADR-0006) per ADR-0119. Companion fix in
  `libvmaf/tools/vmaf.c` resolves the picture-pool exhaustion in the
  `--frame_skip_ref/dist` loops surfaced once the always-on picture
  pool (ADR-0104) made unref'ing skipped pictures mandatory.
- **Touches**:
  - `libvmaf/tools/cli_parse.c` (`VMAF_DEFAULT_PRECISION_FMT` +
    `VMAF_LOSSLESS_PRECISION_FMT` macros, `resolve_precision_fmt()`
    body, `--help` text)
  - `libvmaf/tools/cli_parse.h` (field comments only; struct shape
    unchanged)
  - `libvmaf/src/output.c` (`DEFAULT_SCORE_FORMAT` macro)
  - `libvmaf/tools/vmaf.c` (skip loop bodies at the
    `c.frame_skip_ref` / `c.frame_skip_dist` for-loops)
  - `python/vmaf/core/result.py` (per-frame and aggregate `:.6f`
    formatters)
  - `python/test/command_line_test.py` is unmodified — Netflix golden
    assertions stay frozen per CLAUDE.md §8; the binary's output
    format adapts to them, not the other way around.
- **Invariant**: `vmaf` CLI default score-output format is `%.6f`
  (matches upstream Netflix byte-for-byte). `--precision=max|full`
  selects `%.17g` (IEEE-754 round-trip lossless). `--precision=legacy`
  is a synonym for the default. The library default for
  `vmaf_write_output_with_format(..., score_format=NULL)` matches.
  Skipped frames in the `--frame_skip_ref` / `--frame_skip_dist`
  pre-loops are `vmaf_picture_unref`'d immediately after fetch so the
  preallocated picture pool is not exhausted before the main scoring
  loop runs. **Do not** flip the macros back to `%.17g` or remove the
  unrefs without a superseding ADR — both are golden-gate-load-bearing.
- **Re-test**:

  ```bash
  ninja -C libvmaf/build
  python -m pytest python/test/command_line_test.py \
      ::VmafexecCommandLineTest::test_run_vmafexec \
      ::VmafexecCommandLineTest::test_run_vmafexec_with_frame_skipping \
      ::VmafexecCommandLineTest::test_run_vmafexec_with_frame_skipping_unequal \
      -v
  # Expected: all three PASS in <1 s combined.
  ```

  Pure fork-local; no Netflix-side conflict vector. If upstream ever
  changes the default format string, treat their value as the new
  baseline and reconfirm the golden assertions before adopting.

### 0018 — FFmpeg patches ship as ordered series.txt

- **Workstream PRs**: this PR (`fix(ci): drop dead sycl trigger +
  consolidate windows.yml into libvmaf.yml (ADR-0115)`). Surfaced
  once ADR-0115's consolidation routed the docker / FFmpeg-SYCL
  jobs through the master-targeting CI gate for the first time on
  this branch — the standalone `0003-…sycl…` apply broke because
  it referenced struct fields added by `0001-…tiny-model…`, the
  Dockerfile only `COPY`'d 0003, and `ffmpeg.yml` referenced a
  stale `../patches/` path.
- **Touches**: `Dockerfile` (lines ~86-95 — the FFmpeg patch-apply
  block), `.github/workflows/ffmpeg.yml` (the `Build FFmpeg with
  SYCL patch series` step), `ffmpeg-patches/000{1,2,3}-*.patch`
  (regenerated via real `git format-patch -3` so they carry valid
  `index <sha>..<sha> <mode>` lines and committable SHAs). Pure
  fork-local; no upstream FFmpeg or Netflix file changes.
- **Invariant**: both the Dockerfile and `ffmpeg.yml` walk
  `ffmpeg-patches/series.txt` line-by-line and apply each patch
  via `git apply` with a `patch -p1` fallback. **Do not** ship a
  new patch without appending it to `series.txt`, and **do not**
  reorder existing entries — patch 0003 references LIBVMAFContext
  fields added by patch 0001, so any out-of-order apply breaks
  the build at hunk 2 of vf_libvmaf.c.
- **Two flag-side fixes bundled in the same PR**:
  1. `--enable-libvmaf-sycl` is **not** a valid FFmpeg configure
     option. Patch 0003 uses `check_pkg_config libvmaf_sycl …`
     auto-detection (matching how `libvmaf_cuda` is wired) — it
     never registers the switch. Both Dockerfile and ffmpeg.yml
     used to pass the flag and configure rejected it with
     `Unknown option "--enable-libvmaf-sycl"`. SYCL support is
     now controlled solely by `-Denable_sycl=true` at libvmaf
     build time; FFmpeg picks it up automatically when
     `libvmaf-sycl.pc` is on `PKG_CONFIG_PATH`.
  2. The Dockerfile now carries **two** nvcc-flag ARGs.
     `NVCC_FLAGS` (libvmaf) keeps four `-gencode` lines plus the
     experimental `--extended-lambda` /
     `--expt-relaxed-constexpr` / `--expt-extended-lambda` flags
     needed for Thrust/CUB host+device code. `FFMPEG_NVCC_FLAGS`
     (FFmpeg) carries a single `-gencode arch=compute_75,code=sm_75
     -O2` — FFmpeg's `check_nvcc` runs `nvcc -ptx`, which fails with
     `nvcc fatal: Option '--ptx (-ptx)' is not allowed when
     compiling for multiple GPU architectures` on multi-arch input,
     and `--extended-lambda` requires host+device compilation.
     compute_75 PTX is forward-compatible with all newer GPUs via
     driver JIT.
  3. `--enable-libnpp` is no longer passed to FFmpeg's configure.
     FFmpeg n8.1's libnpp probe carries an explicit
     `die "ERROR: libnpp support is deprecated, version 13.0 and
     up are not supported"` (configure:7335-7336) that fires on
     the base image's CUDA 13.2 libnpp. We don't use scale_npp /
     transpose_npp / sharpen_npp in any VMAF workflow; cuvid +
     nvdec + nvenc + libvmaf-cuda is the actual GPU path. Revisit
     once we move to an FFmpeg release that supports CUDA 13
     libnpp upstream.
  4. Patch 0002 (`add-vmaf_pre-filter`) gained a missing
     `#include "libavutil/imgutils.h"` for `av_image_copy_plane()`.
     FFmpeg's libavfilter Makefile builds with
     `-Werror=implicit-function-declaration` so this fired during
     the actual compile (not configure). Caught by a local
     `docker build` rather than waiting for GitHub Actions —
     much faster iteration loop.
- **Re-test**:

  ```bash
  cd /tmp && rm -rf ffmpeg-test && \
      git clone -q --depth 1 -b n8.1 \
          https://git.ffmpeg.org/ffmpeg.git ffmpeg-test && \
      cd ffmpeg-test && \
      while IFS= read -r line; do \
          case "$line" in ''|\#*) continue ;; esac; \
          git apply "/path/to/vmaf/ffmpeg-patches/$line" \
              || patch -p1 < "/path/to/vmaf/ffmpeg-patches/$line"; \
      done < /path/to/vmaf/ffmpeg-patches/series.txt
  # Expected: all three patches apply with no rejects; the resulting
  # tree compiles with --enable-libvmaf. SYCL is auto-detected via
  # check_pkg_config (patch 0003), so no explicit configure flag is
  # required when libvmaf-sycl.pc is on PKG_CONFIG_PATH.
  ```

  Pure fork-local series; no Netflix-side conflict vector. See ADR-0118.

### 0019 — Coverage Gate annotations: upload-artifact v7 + gcovr filter

- **Workstream PRs**: this PR.
- **Touches**:
  `.github/workflows/ci.yml` (CPU + GPU coverage steps: gcovr stderr
  piped through `grep -vE 'Ignoring (suspicious|negative) hits' ... ||
  true`),
  `.github/workflows/{ci,lint,nightly,nightly-bisect,supply-chain,libvmaf}.yml`
  (`actions/upload-artifact@v5|@v6 → @v7`,
  `actions/download-artifact@v5 → @v7` in `supply-chain.yml`). Note:
  `windows.yml` was consolidated into `libvmaf.yml` by ADR-0115 / PR #50,
  so the windows-side bump now lives in `libvmaf.yml`'s
  `build (MINGW64, …)` job.
- **Invariant**: Coverage Gate Annotations panel must finish empty on a
  clean run. The two pieces are coordinated — (a) `@v7` for upload /
  download artifact actions silences GitHub's Node-20 deprecation banner
  ahead of the 2026-06-02 forced-Node-24 cutoff; (b) the gcovr stderr
  filter swallows the `Ignoring (suspicious|negative) hits` warnings
  that gcovr 8 emits for the legitimately-large hit counts in tight
  ANSNR / VIF / motion inner loops (e.g. `ansnr_tools.c:207` at ~4.93 G
  hits across an HD multi-frame coverage suite — real, not gcov bug).
  The filter is regex-narrow and anchored to gcov's exact warning
  prefix; any *other* gcovr warning still surfaces. Upstream
  (Netflix/vmaf) does not maintain these CI files; rebase impact is
  limited to the unlikely case that an upstream sync touches the
  shared `.github/workflows/` tree, which it currently does not. See
  [ADR-0117](adr/0117-coverage-gate-warning-noise-suppression.md).
- **Re-test**:

  ```bash
  # Verify gcovr filter locally (after a coverage build per entry 0014):
  ~/.local/bin/gcovr --root .. \
      --filter 'src/.*' \
      --exclude '.*/test/.*' --exclude '.*/tests/.*' \
      --exclude '.*/subprojects/.*' \
      --gcov-ignore-parse-errors=negative_hits.warn \
      --gcov-ignore-parse-errors=suspicious_hits.warn \
      --print-summary --txt build-cov-test/coverage.txt \
      build-cov-test \
    2> >(grep -vE 'Ignoring (suspicious|negative) hits' >&2 || true)
  # Expected: stderr contains the gcovr summary block but NO
  # "Ignoring (suspicious|negative) hits" lines. coverage.txt unchanged.

  # Verify all upload/download-artifact instances are on @v7:
  grep -rE 'actions/(upload|download)-artifact@v[0-6]' .github/workflows/
  # Expected: empty output.
  ```

### 0020 — CI workflow file + display-name renames (Title Case sweep)

- **Workstream PRs**: this PR; renames all six core
  `.github/workflows/*.yml` files to purpose-descriptive kebab-case and
  normalises every workflow `name:` and job `name:` to Title Case. See
  [ADR-0116](adr/0116-ci-workflow-naming-convention.md).
- **Touches**:
  `.github/workflows/{ci,lint,security,libvmaf,ffmpeg,docker}.yml`
  (renamed via `git mv` to
  `tests-and-quality-gates.yml`, `lint-and-format.yml`,
  `security-scans.yml`, `libvmaf-build-matrix.yml`,
  `ffmpeg-integration.yml`, `docker-image.yml`),
  `README.md` (5 badge URLs + labels), `docs/principles.md` (line 5
  workflow-tuple update), `.claude/skills/add-gpu-backend/SKILL.md` +
  `scaffold.sh` (filename refs), `docs/adr/0116-*.md` (new),
  `docs/adr/README.md` (index row), `CHANGELOG.md`.
- **Invariant**: workflow files are *purpose-named*; their `name:`
  fields are Title Case sentences with em-dash axis tags; job-level
  `name:` strings are Title Case sentences (Build — / Pre-Commit /
  Coverage Gate / etc.). Required-status-check contexts in `master`
  branch protection are bound to job-level names — when renaming any
  job, re-pin via
  `gh api --method PUT repos/lusoris/vmaf/branches/master/protection`.
  The 19 required gates' *semantics* are unchanged from
  [ADR-0037](adr/0037-master-branch-protection.md); only their display
  strings move.
- **Re-test**:

  ```bash
  # Validate every workflow file parses and lists the expected job names.
  cd .github/workflows
  for f in tests-and-quality-gates.yml lint-and-format.yml security-scans.yml \
           libvmaf-build-matrix.yml ffmpeg-integration.yml docker-image.yml; do
      yq '.name, .jobs.[].name' "$f" || echo "PARSE FAIL: $f"
  done
  # Expected: each workflow prints its Title Case workflow name + job names;
  # no PARSE FAIL lines.
  ```

### 0021 — DNN-enabled CI matrix legs (gcc + clang + macOS)

- **Workstream PRs**: this PR; adds three new entries to the
  `libvmaf-build` matrix in
  [`.github/workflows/libvmaf-build-matrix.yml`](../.github/workflows/libvmaf-build-matrix.yml)
  covering `-Denable_dnn=enabled` across Ubuntu/gcc, Ubuntu/clang, and
  macOS/clang. See
  [ADR-0120](adr/0120-ai-enabled-ci-matrix-legs.md).
- **Touches**:
  `.github/workflows/libvmaf-build-matrix.yml` (3 new matrix entries +
  ORT install steps + dedicated dnn-suite test step),
  `docs/adr/0120-ai-enabled-ci-matrix-legs.md` (new),
  `docs/adr/README.md` (index row),
  `CHANGELOG.md` (Added entry).
- **Invariant**: the DNN matrix legs install ONNX Runtime via the same
  pinned source as the dedicated Tiny AI job
  ([tests-and-quality-gates.yml](../.github/workflows/tests-and-quality-gates.yml))
  — Linux: MS tarball at the version pinned by `ORT_VERSION`; macOS:
  Homebrew. When the Tiny AI job's pin changes, the matrix legs'
  `ORT_VERSION` env in their `Install ONNX Runtime (linux, DNN leg)`
  step must change to match; otherwise compiler/portability coverage
  drifts away from the gating leg's actual ABI.
- **Re-test**:

  ```bash
  # Local sanity: the matrix file parses and the new job names exist.
  yq '.jobs.libvmaf-build.strategy.matrix.include[] | select(.dnn==true) | .name' \
      .github/workflows/libvmaf-build-matrix.yml
  # Expected output (3 lines):
  #   Build — Ubuntu gcc (CPU) + DNN
  #   Build — Ubuntu clang (CPU) + DNN
  #   Build — macOS clang (CPU) + DNN

  # Local DNN build sanity (matches what each leg will run):
  meson setup libvmaf libvmaf/build --buildtype release \
      --prefix $PWD/install -Denable_float=true -Denable_dnn=enabled
  ninja -vC libvmaf/build install
  meson test -C libvmaf/build --suite=dnn --print-errorlogs
  ```

- **Branch protection**: the two Linux DNN legs are pinned as required
  status checks on `master` immediately after this PR's merge (19 → 21
  contexts). The macOS leg stays informational (`experimental: true`)
  because Homebrew ORT floats. Re-pin command:

  ```bash
  gh api --method PUT repos/lusoris/vmaf/branches/master/protection \
      --input /tmp/protection-update.json
  ```

### 0022 — Windows GPU build-only matrix legs (MSVC + CUDA, MSVC + oneAPI SYCL)

- **Workstream PRs**: this PR; adds a new top-level `windows-gpu-build`
  job to
  [`.github/workflows/libvmaf-build-matrix.yml`](../.github/workflows/libvmaf-build-matrix.yml)
  with two matrix entries (CUDA, SYCL). See
  [ADR-0121](adr/0121-windows-gpu-build-only-legs.md).
- **Touches**:
  `.github/workflows/libvmaf-build-matrix.yml` (new
  `windows-gpu-build` job),
  `docs/adr/0121-windows-gpu-build-only-legs.md` (new),
  `docs/adr/README.md` (index row),
  `CHANGELOG.md` (Added entry),
  `libvmaf/src/compat/win32/pthread.h` (new — Win32 pthread shim
  for MSVC; mirrors `compat/gcc/stdatomic.h` pattern),
  `libvmaf/src/feature/integer_adm.h` (UPSTREAM — converted
  the `dwt_7_9_YCbCr_threshold[3]` designated initializer to
  positional form so MSVC/nvcc-on-Windows accepts the C++
  parse; semantically identical, no behavioural change),
  `libvmaf/src/ref.h` and
  `libvmaf/src/feature/feature_extractor.h` (UPSTREAM —
  added `#if defined(__cplusplus) && defined(_MSC_VER)`
  branch around `#include <stdatomic.h>` so MSVC C++ TUs
  pull `atomic_int` via `using std::atomic_int;`; POSIX
  paths unchanged),
  `libvmaf/src/sycl/d3d11_import.cpp` (fix non-existent
  `<libvmaf/log.h>` → `"log.h"`),
  `libvmaf/src/sycl/dmabuf_import.cpp` (move `<unistd.h>`
  inside `#if HAVE_SYCL_DMABUF` guard for non-VA-API
  hosts),
  `libvmaf/src/sycl/common.cpp` (replace POSIX
  `clock_gettime(CLOCK_MONOTONIC)` with portable
  `std::chrono::steady_clock`),
  `libvmaf/src/feature/x86/motion_avx2.c` (UPSTREAM —
  replace GCC vector-extension `__m256i[N]` indexing at
  line 529 with `_mm256_extract_epi64`; bit-exact),
  `libvmaf/src/feature/x86/adm_avx2.c` (UPSTREAM —
  replace 6 `(__m256i)(_mm256_cmp_ps(...))` casts with
  `_mm256_castps_si256(...)` and 12 `__m128i[N]`
  reductions with `_mm_extract_epi64`; bit-exact),
  `libvmaf/src/feature/x86/adm_avx512.c` (UPSTREAM —
  replace 12 `__m128i[N]` reductions with
  `_mm_extract_epi64`; bit-exact),
  `libvmaf/src/log.c` (UPSTREAM — gate `<unistd.h>`
  behind `!_WIN32`, include `<io.h>` + redirect
  `isatty`/`fileno` to `_isatty`/`_fileno` for MSVC),
  `libvmaf/src/feature/integer_vif.c` (UPSTREAM —
  switch the `aligned_malloc` cursor from `void *`
  to `uint8_t *` with explicit typed-pointer casts
  so MSVC accepts the byte-wise pointer arithmetic),
  `libvmaf/src/feature/cuda/integer_adm_cuda.c`
  (UPSTREAM — drop unused `<unistd.h>` include),
  `libvmaf/src/dnn/model_loader.c` (fork-added —
  Windows fallback definitions for POSIX `S_ISDIR`
  / `S_ISREG` path-classification macros),
  `.github/workflows/lint-and-format.yml` (fork-added —
  set `lfs: true` on the pre-commit job's checkout so
  LFS-stored ONNX blobs resolve and don't appear as
  phantom pre-commit-induced diffs),
  `libvmaf/src/feature/x86/motion_avx512.c` (UPSTREAM —
  replace 1 `__m128i[N]` reduction with
  `_mm_extract_epi64`; bit-exact),
  `libvmaf/src/feature/x86/{vif_statistic_avx2,ansnr_avx2,ansnr_avx512,float_adm_avx2,float_adm_avx512,float_psnr_avx2,float_psnr_avx512,ssim_avx2,ssim_avx512}.c`
  (UPSTREAM — convert 17 sites of trailing
  `__attribute__((aligned(N)))` to leading C11
  `_Alignas(N)`; same alignment, MSVC-portable),
  `libvmaf/src/feature/mkdirp.c` and
  `libvmaf/src/feature/mkdirp.h` (UPSTREAM third-party
  MIT-licensed micro-library — gate `<unistd.h>` to
  non-Windows, add `<direct.h>` + `_mkdir` for Windows,
  add `mode_t` typedef for MSVC),
  `libvmaf/meson.build` (new `pthread_dependency` gated on
  `cc.check_header('pthread.h')` failing),
  `libvmaf/src/meson.build` and `libvmaf/test/meson.build` (thread
  `pthread_dependency` into every target compiling pthread-using TUs).
- **Invariant**: Windows GPU legs are pinned to the same toolchain
  versions as the corresponding Linux GPU legs (CUDA 13.0.0, oneAPI
  BaseKit 2025.3.0.372) so a Linux-vs-Windows divergence implies an
  MSVC ABI issue, not a tooling-version delta. When either Linux GPU
  leg bumps its toolchain, the Windows leg must move in lockstep —
  the Intel installer URL on Windows hard-codes the per-release
  directory id and the version string, so the bump is two-line
  edits in the SYCL `Install Intel oneAPI (windows)` step (the
  `WINDOWS_BASEKIT_URL` env var). Both legs additionally inject
  `/experimental:c11atomics` into `CFLAGS` / `CXXFLAGS` because
  libvmaf uses C11 atomics that MSVC's `<stdatomic.h>` rejects
  without that opt-in flag — when MSVC ships full C11 atomics
  support, the flag becomes unconditional and can be dropped.
  Two Windows-only dependency steps round out the parity:
  the CUDA leg's `Jimver/cuda-toolkit` sub-package list includes
  both `crt` (CUDA Runtime Library compile-time headers, ships
  `crt/host_config.h`; `cuda_cccl` is not a valid Windows
  sub-package name — installer rejects it) and `nvvm` (ships
  `nvvm/bin/cicc.exe` + `nvvm/libdevice/libdevice.*.bc`; without
  it, nvcc's `.cu → PTX` stage fails with `The system cannot
  find the path specified.` — on Linux apt pulls NVVM in
  transitively with `cuda-nvcc-XY`, Windows requires it
  explicitly); the SYCL leg builds
  the Level Zero
  loader from source (`oneapi-src/level-zero` v1.18.5 →
  `cmake --build … --target install`) because Windows oneAPI
  BaseKit ships the SYCL runtime but not `ze_loader.lib`, and
  libvmaf's meson `cc.find_library('ze_loader')` needs both the
  header and the import library. When the Linux apt
  `level-zero-dev` version moves, bump the L0 git tag to match.
  `libvmaf/src/meson.build` guards the explicit `svml` / `irc`
  `cc.find_library` calls behind `host_machine.system() !=
  'windows'` — those calls exist for the gcc/g++ + icpx Linux
  flow where the host linker is non-Intel; on Windows the host
  compiler is icx-cl itself and auto-injects the Intel runtime.
  Round-10 surfaced an additional Windows-only gap: ~14 libvmaf
  TUs `#include <pthread.h>` unconditionally, but MSVC and
  clang-cl ship no pthread (MinGW does, via winpthreads). The
  fork now ships a header-only Win32 shim at
  `libvmaf/src/compat/win32/pthread.h` mapping the in-use
  pthread subset (mutex / cond / thread create+join+detach)
  onto SRWLOCK + CONDITION_VARIABLE + `_beginthreadex`. The
  shim is wired in via `pthread_dependency` in
  `libvmaf/meson.build`, declared only when
  `cc.check_header('pthread.h')` fails — so MinGW and POSIX
  paths stay untouched. When upstream Netflix/vmaf adds new
  pthread surface (e.g., `pthread_rwlock_*`), extend
  `compat/win32/pthread.h` to cover it. Both nvcc fatbin
  `custom_target`s (CUDA) and icpx `custom_target`s (SYCL
  `common.cpp` / `picture_sycl.cpp` / `dmabuf_import.cpp`,
  plus the SYCL feature kernels) bypass meson's
  `dependencies:` plumbing and hand-roll their own `-I`
  lists, so the shim
  path must be threaded into both `cuda_extra_includes` and
  `sycl_inc_flags` explicitly on Windows. icpx-cl on
  Windows additionally rejects `-fPIC` (`unsupported option
  for target 'x86_64-pc-windows-msvc'`) — so
  `sycl_common_args` and `sycl_feature_args` route their
  `-fPIC` token through `sycl_pic_arg = host_machine.system()
  != 'windows' ? ['-fPIC'] : []`. PIC is the default for
  Windows DLLs, so dropping the flag is the correct fix
  rather than a workaround. Round-14 surfaced a third
  Windows-only blocker: `libvmaf/src/feature/integer_adm.h`
  (an upstream Netflix file, last touched by upstream port
  d06dd6cf) initialises `dwt_7_9_YCbCr_threshold[3]` with
  C99 designated initializers (`{.a = ..., .k = ..., .f0 =
  ..., .g = {...}}`). The header is included from both
  `integer_adm.c` (C TU) and `cuda/integer_adm/*.cu` (C++
  TU via nvcc); MSVC's C++ frontend (and nvcc's cudafe++
  on Windows) rejects C99 designated initializers without
  `/std:c++20`. Converted to positional initialization in
  the same struct-member order (a / k / f0 / g[4]) — the
  conversion is provably semantically identical and works
  in every C/C++ standard, so it costs nothing on the
  upstream-merge side beyond a trivial conflict marker if
  upstream Netflix later edits the same lines. Restore
  designated form post-merge if upstream has it.
  Round-17 surfaced four more Windows/MSVC-only SYCL
  blockers, two of which touch upstream-shared headers.
  (a) `libvmaf/src/ref.h` and
  `libvmaf/src/feature/feature_extractor.h` (UPSTREAM)
  unconditionally `#include <stdatomic.h>` and use the
  `atomic_int` typedef in struct definitions. MSVC's
  `<stdatomic.h>` (added in 19.34) only declares the C11
  symbols inside the global namespace under C; in C++
  compilation (icpx-cl drives the SYCL TUs as C++) MSVC
  surfaces them only inside `namespace std::`. gcc/clang
  expose both via a GNU extension, so the upstream code
  works on every other platform. The fork now wraps both
  headers' `#include <stdatomic.h>` in
  `#if defined(__cplusplus) && defined(_MSC_VER)` →
  `#include <atomic>` + `using std::atomic_int;`, falling
  through to the original `<stdatomic.h>` line on every
  other configuration. ABI is unchanged — `atomic_int`
  resolves to the same underlying type. If upstream
  Netflix adds further C11 atomic typedefs in these
  headers (e.g., `atomic_uint`, `atomic_size_t`), extend
  the `using std::` lines to cover them. (b)
  `libvmaf/src/sycl/d3d11_import.cpp` (fork-added)
  used `<libvmaf/log.h>` which doesn't exist — `log.h`
  lives at `libvmaf/src/log.h` and is internal. Switched
  to `"log.h"`; the icpx invocation already supplies the
  src-relative `-I`. (c) `libvmaf/src/sycl/dmabuf_import.cpp`
  (fork-added) included `<unistd.h>` at file scope, but
  POSIX `close()` is only used inside the
  `#if HAVE_SYCL_DMABUF` VA-API block. Moved the
  `<unistd.h>` include inside that guard so non-DMA-BUF
  builds (Windows MSVC, macOS) compile cleanly. (d)
  `libvmaf/src/sycl/common.cpp` (fork-added) called
  `clock_gettime(CLOCK_MONOTONIC)`, which doesn't exist
  on Windows. Replaced with `std::chrono::steady_clock`
  (guaranteed monotonic by the C++ standard, portable on
  every supported host). All four fixes preserve
  POSIX/Linux behaviour bit-identically and only change
  the Windows MSVC build path.
  Round-18 surfaced a fifth Windows blocker on the CUDA
  leg's CPU SIMD compile path:
  `libvmaf/src/feature/x86/motion_avx2.c:529` (UPSTREAM,
  ported in commit 9371a0aa from Netflix PR #1486)
  computed `final_accum[0] + final_accum[1] +
  final_accum[2] + final_accum[3]` to extract the four
  int64 lanes from an `__m256i`. gcc/clang allow this
  via the GNU vector-extension treatment of `__m256i`
  (it carries `__attribute__((vector_size(32)))`); MSVC
  rejects it with `C2088: built-in operator '[' cannot
  be applied to an operand of type '__m256i'`. Replaced
  with `_mm256_extract_epi64(final_accum, N)` for
  N ∈ {0..3}, summed — bit-exact lane sum on every
  compiler. Restore the index form post-merge if
  upstream Netflix later edits the same lines and your
  toolchain matrix doesn't include MSVC.
  Round-19 surfaced the same MSVC pattern at 19 more
  call sites across the AVX2/AVX-512 ADM and motion
  files plus six GCC-style vector casts.
  `libvmaf/src/feature/x86/adm_avx2.c` (UPSTREAM):
  6 lines (915-920) used `(__m256i)(_mm256_cmp_ps(...))`
  C-style casts that gcc/clang accept via the GNU
  vector extension; replaced with the dedicated
  `_mm256_castps_si256(...)` bit-cast intrinsic. 12
  lane-extract sites (`r2_h[0]+r2_h[1]`, etc. at lines
  2420 / 2425 / 2430 / 2893 / 2897 / 2901 / 4079 /
  4084 / 4089 / 4627 / 4631 / 4635) replaced with
  `_mm_extract_epi64(r2_X, N)` summed pair.
  `libvmaf/src/feature/x86/adm_avx512.c` (UPSTREAM):
  6 sister lane-extract sites (lines 4470 / 4477 /
  4484 / 4625 / 4631 / 4637) — same fix. The AVX-512
  paths reduce a `__m512i` down to `__m128i` first
  (via `_mm512_extracti64x4_epi64` →
  `_mm256_extracti64x2_epi64`) before the index, so
  only the final `__m128i[N]` step needed changing.
  `libvmaf/src/feature/x86/motion_avx512.c` (UPSTREAM,
  ported in 9371a0aa from PR #1486): one final
  `r2[0]+r2[1]` reduction (line 448), same fix. All
  19 lane-extract fixes plus the 6 cast fixes are
  bit-exact rewrites and only change the source-level
  syntax to MSVC-portable form. Restore the original
  forms post-merge if upstream Netflix later edits
  the same lines and your toolchain matrix doesn't
  include MSVC. Additionally
  `libvmaf/src/sycl/d3d11_import.cpp` (fork-added)
  switched from C-style COBJMACROS helpers
  (`ID3D11Device_CreateTexture2D`, `…_Release`, etc.)
  to C++ method-call syntax (`device->CreateTexture2D`,
  `tex->Release`) — d3d11.h gates COBJMACROS behind
  `!defined(__cplusplus)`, so the C-style helpers
  aren't visible in this `.cpp` TU. The two forms are
  ABI-equivalent (both dispatch through the COM vtable);
  the choice is purely lexical and POSIX builds aren't
  affected (the whole TU is `#ifdef _WIN32`).
  Round-20 surfaced two more Windows-only blockers.
  (a) 17 sites across the x86 SIMD layer used GCC's
  `float tmp[N] __attribute__((aligned(M)));` form to
  align scratch buffers for `_mm{256,512}_store_ps`.
  MSVC rejects the trailing-attribute syntax with
  `C2146: syntax error: missing ';' before identifier
  '__attribute__'`. Replaced with the C11-standard
  `_Alignas(M) float tmp[N];` (alignment specifier
  before the type) — works in gcc, clang and MSVC
  with `/std:c11`. Files touched (all UPSTREAM):
  `vif_statistic_avx2.c` (×2), `ansnr_avx2.c` (×2),
  `ansnr_avx512.c` (×2), `float_adm_avx2.c` (×2),
  `float_adm_avx512.c` (×2), `float_psnr_avx2.c` (×1),
  `float_psnr_avx512.c` (×1), `ssim_avx2.c` (×4),
  `ssim_avx512.c` (×4). The pre-existing
  `vif_avx2.c` / `vif_avx512.c` already define a
  portable `ALIGNED(x)` macro at file scope and
  position the attribute before the type, so they
  compile cleanly under MSVC and were not touched.
  (b) `libvmaf/src/feature/mkdirp.c` (UPSTREAM,
  third-party MIT-licensed copy of Stephen Mathieson's
  micro-library) included `<unistd.h>` unconditionally
  but never used POSIX `unistd` symbols (only `mkdir`
  via `<sys/stat.h>`/`<direct.h>`). Gated `<unistd.h>`
  to non-Windows and added `<direct.h>` for Windows;
  switched `mkdir(pathname)` → `_mkdir(pathname)` (the
  non-deprecated MSVC name). `libvmaf/src/feature/mkdirp.h`
  added a `mode_t` typedef under MSVC since neither
  `<sys/types.h>` nor `<sys/stat.h>` declare it on
  Windows; `mode` is ignored on the Windows path
  anyway.
  Round-21 surfaced two more blockers (the round-19
  `__m128i[N]` sweep missed six sites) plus a
  pre-commit workflow checkout gap.
  (a) `libvmaf/src/feature/x86/adm_avx512.c` (UPSTREAM)
  had six further `r2_X[0] + r2_X[1]` reductions at
  lines 2128 / 2135 / 2142 / 2589 / 2595 / 2601 that
  reduce a `__m512i` accumulator down to `__m128i`
  before the lane index. Replaced with the same
  `_mm_extract_epi64(r2_X, N)` summed-pair pattern
  used in round 19 — bit-exact, MSVC-portable.
  (b) `libvmaf/src/log.c` (UPSTREAM) included
  `<unistd.h>` unconditionally to pick up POSIX
  `isatty` / `fileno`. On MSVC both live in
  `<io.h>` as `_isatty` / `_fileno`; gated the
  include and macro-redirected the names so the
  one call site at line 34 compiles on both sides
  without touching the POSIX path.
  (c) `.github/workflows/lint-and-format.yml`
  (fork-added) checks out without `lfs: true`, so
  the `model/tiny/*.onnx` files land as LFS
  pointer stubs. pre-commit's "changes made by
  hooks" reporter then diffs the stubs against
  HEAD's real blobs and fails the job even though
  no hook touched them. Added `lfs: true` to the
  pre-commit job's checkout.
  (d) `libvmaf/src/meson.build` —
  `cuda_common_vmaf_lib` static library had no
  `dependencies:` list, so the Win32 pthread shim
  (wired in via `pthread_dependency` in
  [libvmaf/meson.build](../libvmaf/meson.build))
  wasn't on its include path; `cuda/common.h`
  unconditionally `#include <pthread.h>` and
  MSVC failed with C1083. Added
  `dependencies : [pthread_dependency]` — no-op
  on POSIX (empty list), routes the shim path in
  on Windows.
  (e) `libvmaf/src/feature/integer_vif.c`
  (UPSTREAM) walked one big `aligned_malloc`
  result as `void *data` and did
  `data += pad_size` / `data += h *
  stride_16` etc. to carve the buffer into
  typed sub-pointers. gcc/clang accept
  pointer arithmetic on `void *` as a GNU
  extension (treating `sizeof(void) == 1`);
  MSVC rejects it with `C2036: 'void *':
  unknown size`. Replaced the cursor type
  with `uint8_t *` and added explicit casts
  at assignment sites that take a typed
  pointer (`uint16_t *mu1`, `uint32_t
  *mu1_32`, etc.). Byte offsets are
  identical, layout unchanged, bit-exact.
  If upstream Netflix edits the same loop,
  reabsorb the walk and re-apply the
  cursor-type + cast pattern.
  (f) `libvmaf/src/feature/cuda/integer_adm_cuda.c`
  (UPSTREAM) included `<unistd.h>` at line 33
  but used no POSIX symbols from it; MSVC
  failed with C1083. Dropped the unused
  include outright — simplest fix, no runtime
  change on any platform.
  (g) `libvmaf/src/dnn/model_loader.c`
  (fork-added) uses `S_ISDIR` / `S_ISREG` to
  classify resolved paths. MSVC ships the
  underlying `S_IFMT` / `S_IFDIR` / `S_IFREG`
  bit masks in `<sys/stat.h>` but not the
  POSIX classification macros. Added a
  Windows-only fallback (`#ifndef S_ISDIR
  #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
  #endif`, same for S_ISREG) guarded by
  `#ifdef _WIN32`. Semantically identical
  to the POSIX macro on Linux/macOS.
  Round-21e surfaced the final source-portability
  blockers once the DLL build passed preprocessing.
  (h) `libvmaf/src/predict.c`, `libvmaf/src/libvmaf.c`
  and `libvmaf/src/read_json_model.c` (all UPSTREAM)
  used C99 variable-length arrays —
  `double scores[cnt]` at
  [predict.c:385](../libvmaf/src/predict.c#L385),
  `char name[name_sz]` at
  [predict.c:453](../libvmaf/src/predict.c#L453)
  and [libvmaf.c:1741](../libvmaf/src/libvmaf.c#L1741),
  plus [`cfg_name[cfg_name_sz]`](../libvmaf/src/read_json_model.c#L517)
  and [`generated_key[generated_key_sz]`](../libvmaf/src/read_json_model.c#L520)
  in the `.json` model-collection parser. gcc/clang
  accept VLAs as a C11 optional feature; MSVC
  (even with `/std:c11`) rejects them outright with
  `C2057: expected constant expression` (plus C2466
  and C2133 on the `const size_t` sized arrays —
  MSVC treats `const` as runtime-bounded, not a
  constant expression, even when the initialiser is
  literal like `4 + 1`). Replaced each runtime-sized
  buffer with a small `malloc` + explicit `free` on
  every exit path (in predict.c and read_json_model.c
  a `goto out;` cleanup arm was introduced because
  the loops error-exit mid-function). The
  `generated_key` buffer in read_json_model.c uses
  the narrower fix — `char generated_key[5];` —
  since its size (four decimal digits of the
  bootstrap sub-model index plus NUL) is a true
  compile-time constant. Buffers are a handful of
  bytes each (`name_sz` is the model-collection name
  length plus the fixed `_ci_p95_lo` suffix, `scores`
  holds ~20 doubles, `cfg_name` is the name plus
  `_0000` suffix), so the heap round-trip is not
  performance-relevant; the new `-ENOMEM` failure
  mode is handled uniformly by existing callers.
  The read_json_model.c refactor also plugs a
  pre-existing leak of the `name` buffer on the
  early `return -EINVAL` when a JSON object key
  isn't a string — the `goto out;` path frees
  `name` + `cfg_name` on every exit.
  [`libvmaf/test/test_feature_extractor.c:56`](../libvmaf/test/test_feature_extractor.c#L56)
  (UPSTREAM) declared `const unsigned n_threads = 8;`
  and used it as the extent of `VmafFeatureExtractorContext
  *fex_ctx[n_threads];`. Converted to
  `enum { n_threads = 8 };` so MSVC sees a
  constant-expression; every other compiler
  accepts enum constants identically. Re-absorb
  if upstream Netflix later edits the same loops
  and your toolchain matrix omits MSVC.
  (i) The Windows MSVC build-only legs now build
  the full tree — CLI tools, unit tests and
  libvmaf.dll — rather than the previous short
  cut of disabling `-Denable_tools` /
  `-Denable_tests`. Per user direction
  ("fix the code ffs"), the tree polyfills the
  remaining POSIX surfaces on MSVC instead:
  ([`libvmaf/tools/compat/win32/getopt.h`](../libvmaf/tools/compat/win32/getopt.h)
  +
  [`libvmaf/tools/compat/win32/getopt.c`](../libvmaf/tools/compat/win32/getopt.c))
  a from-scratch POSIX/GNU-compatible
  `getopt_long` shim (short / long options,
  `no_argument` / `required_argument` /
  `optional_argument`, argv permutation for
  non-option operands, `--` explicit stop,
  `=`-embedded values). The shim is fork-added
  (BSD-3-Clause-Plus-Patent, Copyright 2026
  Lusoris and Claude) and declared via a single
  `getopt_dependency` in
  [`libvmaf/meson.build`](../libvmaf/meson.build),
  gated on `cc.check_header('getopt.h')` failing.
  The dependency auto-propagates the shim
  `.c` into any consuming target via meson's
  `sources:` keyword, so both the `vmaf` CLI
  (`libvmaf/tools/meson.build`) and the
  `test_cli_parse` unit test
  (`libvmaf/test/meson.build`) pick it up
  uniformly. MinGW ships `<getopt.h>` via
  mingw-w64-crt, so `check_header` succeeds
  there and the shim stays out of the TU list.
  (j) Eleven test executables
  (`test_log`, `test_dict`, `test_opt`,
  `test_cpu`, `test_ref`, `test_feature`,
  `test_ciede`, `test_luminance_tools`,
  `test_cli_parse`, `test_sycl`,
  `test_sycl_pic_preallocation`) were missing
  `pthread_dependency` in their `dependencies:`
  lists at [`libvmaf/test/meson.build`](../libvmaf/test/meson.build).
  On POSIX `pthread_dependency` is an empty
  list so the omission was invisible; on MSVC
  those TUs transitively include
  `feature_collector.h` → `<pthread.h>` and
  fail with C1083. Threaded the dependency
  through all eleven targets. `test_cli_parse`
  additionally lists `getopt_dependency` to pick
  up the shim.
  (k) Three additional VLA sites surfaced once
  the test harness built on MSVC:
  [`test_cambi.c:254`](../libvmaf/test/test_cambi.c#L254)
  had `unsigned w = 5, h = 5;
  uint16_t buffer[3 * w];`; converted to
  `enum { w = 5, h = 5 };` so the array extent is
  a constant expression.
  [`test_pic_preallocation.c:382`](../libvmaf/test/test_pic_preallocation.c#L382)
  and
  [`test_pic_preallocation.c:506`](../libvmaf/test/test_pic_preallocation.c#L506)
  had `const int num_threads = N; pthread_t
  threads[num_threads];` — MSVC rejects `const int`
  as non-constant-expression. Converted to
  `enum { num_threads = N, fetches_per_thread = M };`.
  (l) [`test_ring_buffer.c:23`](../libvmaf/test/test_ring_buffer.c#L23)
  and
  [`test_pic_preallocation.c:26`](../libvmaf/test/test_pic_preallocation.c#L26)
  included `<unistd.h>` for `usleep` / `sleep`.
  Gated behind `!_WIN32` with a Win32 fallback
  via `<windows.h>` + `#define usleep(us)
  Sleep(((us) + 999) / 1000)` / `#define sleep(s)
  Sleep((s) * 1000)`. The conversion rounds
  sub-millisecond `usleep` inputs up, which is
  safe for these test paths (they use 100 µs
  jitter and 1 s waits).
  (m) [`libvmaf/tools/vmaf.c`](../libvmaf/tools/vmaf.c)
  included `<unistd.h>` for `isatty` / `fileno`.
  Applied the same gating pattern used in
  [`log.c`](../libvmaf/src/log.c#L34) in round-21(b) —
  include `<io.h>` on MSVC and redirect
  `isatty` / `fileno` to `_isatty` / `_fileno`
  via `#define`.
  (n) `__builtin_clz` / `__builtin_clzll` are
  GCC intrinsics; MSVC ships `__lzcnt` /
  `__lzcnt64` via `<intrin.h>` instead. The
  shim already lived in
  [`libvmaf/src/feature/integer_vif.h`](../libvmaf/src/feature/integer_vif.h)
  but
  [`integer_adm.c:939`](../libvmaf/src/feature/integer_adm.c#L939),
  [`x86/adm_avx2.c:1425`](../libvmaf/src/feature/x86/adm_avx2.c#L1425)
  and
  [`x86/adm_avx512.c:1217`](../libvmaf/src/feature/x86/adm_avx512.c#L1217)
  don't include that header. Extracted the shim
  into a dedicated
  [`libvmaf/src/feature/compat_builtin.h`](../libvmaf/src/feature/compat_builtin.h)
  (fork-added) and included it from all four TUs.
  The guard is `defined(_MSC_VER) && !defined(__clang__)`,
  so clang-cl / icx-cl (which provide the GCC
  intrinsics natively) skip the shim.
  (o) The SYCL leg's D3D11 import TU
  [`libvmaf/src/sycl/d3d11_import.cpp`](../libvmaf/src/sycl/d3d11_import.cpp)
  is C++ (icpx-cl drives it as C++ on Windows)
  but included the internal C header
  [`log.h`](../libvmaf/src/log.h) without an
  `extern "C"` wrap. `log.h` is an upstream
  Netflix header with no `__cplusplus` guard,
  so `vmaf_log` got C++ name-mangled in the
  .cpp TU and failed to resolve against the
  C-linkage symbol produced by `log.c` at
  link time (`LNK2019` from every test target
  that pulls in the SYCL static lib). Wrapped
  the `#include "log.h"` with
  `extern "C" { ... }` inside the fork-added
  .cpp rather than touching the upstream
  header — keeps `log.h` identical to upstream
  on every `/sync-upstream`.
  (p) The Windows MSVC legs build with
  `--default-library=static`. libvmaf's public
  API has no `__declspec(dllexport)` attributes
  (upstream Netflix is POSIX-shaped), so a
  vanilla MSVC shared build produces
  `src/vmaf-3.dll` with no exported symbols and
  the toolchain therefore never emits the
  companion `vmaf.lib` import library. Downstream
  tool targets then fail with
  `LNK1181: cannot open input file 'src\vmaf.lib'`.
  The MinGW matrix leg has used
  `--default-library static` since day one for
  the same reason
  ([line 387](../.github/workflows/libvmaf-build-matrix.yml#L387));
  the MSVC legs now mirror that choice via
  `matrix.include[].meson_extra`. Downstream
  consumers that want a DLL can either add
  `__declspec(dllexport)` decorations to the
  public API or use a `.def` file; that is a
  separate decision and out of scope for the
  build-only gate.
- **Re-test**:

  ```bash
  # Local sanity: the matrix file parses and the new job names exist.
  yq '.jobs.windows-gpu-build.strategy.matrix.include[].name' \
      .github/workflows/libvmaf-build-matrix.yml
  # Expected output (2 lines):
  #   Build — Windows MSVC + CUDA (build only)
  #   Build — Windows MSVC + oneAPI SYCL (build only)
  ```

- **Branch protection**: the two Windows GPU legs are pinned as
  required status checks on `master` immediately after this PR's
  merge. After ADR-0120's two Linux DNN legs the count moves
  21 → 23. Re-pin via:

  ```bash
  gh api --method PUT repos/lusoris/vmaf/branches/master/protection \
      --input /tmp/protection-update.json
  ```

### 0023 — CUDA gencode coverage (sm_86/sm_89/compute_80 PTX) + init hardening

- **Workstream PRs**: the ADR-0122 PR (gencode + init hardening) and
  the ADR-0123 follow-up for the `32b115df` post-cubin-load
  regression.
- **Touches**:
  - `libvmaf/src/meson.build` — the `gencode` array in the
    `if get_option('enable_nvcc')` branch.
  - `libvmaf/src/cuda/common.c` — `vmaf_cuda_state_init()` error
    paths (multi-line actionable log, `cuda_free_functions()` +
    `free(c)` + `*cu_state = NULL` cleanup).
  - `docs/backends/cuda/overview.md` — `## Runtime requirements`
    section and `### GPU architecture coverage` table.
- **Invariant**: the `gencode` array unconditionally emits cubins for
  `sm_75` / `sm_80` / `sm_86` / `sm_89` plus a `compute_80` PTX,
  independent of host `nvcc` version. Upstream Netflix's gencode
  only ships cubins at Txx major boundaries (`sm_75` / `sm_80` /
  `sm_90` / `sm_100` / `sm_120`); a literal merge that replaces our
  array with upstream's would re-open the Ampere-`sm_86` /
  Ada-`sm_89` coverage hole. The `sm_90` / `sm_100` / `sm_120`
  entries are still version-gated and should be preserved verbatim
  if upstream adds new gates. The init-path error messages are
  fork-local strings; upstream's terse `"Error: failed to load CUDA
  functions"` must NOT win a merge.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=true -Denable_nvcc=true
  ninja -C build 2>&1 | grep -E 'compute_(80|86|89)'
  # Expect at least -gencode=arch=compute_86,code=sm_86 and
  #                -gencode=arch=compute_89,code=sm_89 and
  #                -gencode=arch=compute_80,code=compute_80

  # Actionable init message (run without CUDA driver on the loader path):
  LD_LIBRARY_PATH= ./build/tools/vmaf --help 2>&1 | grep -qi 'libcuda.so.1' || \
      echo "init log regressed"
  ```

### 0024 — `vmaf_read_pictures` null-guard for CUDA device-only path

- **Workstream PRs**: the ADR-0123 follow-up landed atop the ADR-0122
  gencode/init-hardening work.
- **Touches**:
  - `libvmaf/src/libvmaf.c` — the non-threaded tail of
    `vmaf_read_pictures` at the `prev_ref` update site (line ~1428 in
    the fork; upstream equivalent is the tail added by
    `f740276a`).
- **Invariant**: the `prev_ref` update is guarded by
  `if (ref && ref->ref)` so pure-CUDA extractor sets (where `ref =
  &ref_host` but `ref_host` was never populated by
  `translate_picture_device`) do not deref a NULL refcount. Upstream
  currently has the same unguarded tail; the bug is masked upstream
  only because the experimental `VMAF_PICTURE_POOL` gate from
  `32b115df` is still in place. A literal upstream merge that removes
  our null-guard while upstream's experimental gate is still holding
  would pass tests but re-open the `libvmaf_cuda` ffmpeg crash the
  moment the gate flips default-on (which the fork did in
  `65460e3a`, [ADR-0104](adr/0104-picture-pool-always-on.md)). Keep
  the guard until the upstream null-guard port lands.
- **Re-test**:

  ```bash
  # Unit tests cover the non-regression on the library side:
  meson test -C build

  # End-to-end regression: ffmpeg libvmaf_cuda must exit 0 on a
  # CUDA-device-only extractor set (full recipe in ADR-0123).
  ./ffmpeg -init_hw_device cuda=cu:0 -filter_hw_device cu \
    -i /tmp/ref.mp4 -i /tmp/dis.mp4 \
    -lavfi "[0:v]format=yuv420p,hwupload_cuda[r];\
            [1:v]format=yuv420p,hwupload_cuda[d];\
            [r][d]libvmaf_cuda=log_path=/tmp/out.json:log_fmt=json" \
    -f null -
  ```

### 0025 — VIF `init()` fail-path frees advanced byte-cursor

- **Workstream PRs**: PR #47 (rewritten to leak-fix-only after master
  absorbed the void*→uint8_t* half via commit `b0a4ac3a`, entry 0022
  §e). Ports the leak-fix half of upstream Netflix PR
  [#1476](https://github.com/Netflix/vmaf/pull/1476).
- **Touches**: `libvmaf/src/feature/integer_vif.c` (UPSTREAM —
  2-line fix in the `init()` `fail:` handler).
- **Invariant**: `init()` walks `uint8_t *data` forward through
  `aligned_malloc`'s one allocation, advancing past each
  sub-pointer assignment. If
  `vmaf_feature_name_dict_from_provided_features` returns NULL
  the fail path must free the *base* pointer `s->public.buf.data`,
  never the advanced cursor `data`. Upstream master still has
  `aligned_free(data)` there — same bug — so this entry is the
  reminder to not let an upstream sync re-introduce the
  advanced-cursor form. If upstream lands PR #1476 or an
  equivalent, the sync can drop this entry.
- **Re-test**:

  ```bash
  meson test -C build --suite=fast
  # Static check: ripgrep the pattern that must NOT return.
  rg -n "aligned_free\(data\)" libvmaf/src/feature/integer_vif.c && \
      echo 'REGRESSED' || echo 'ok'
  ```

### 0026 — Automated rule-enforcement workflow + copyright pre-commit hook

- **Workstream PRs**: this PR (ADR-0124 adoption). Closes the
  "rule-without-a-check" gap on ADR-0100 / 0105 / 0106 / 0108.
- **Touches** (all FORK-ADDED — no upstream overlap):
  `.github/workflows/rule-enforcement.yml` (new),
  `scripts/ci/check-copyright.sh` (new),
  `.pre-commit-config.yaml` (appended local hook).
- **Invariant**: the `deep-dive-checklist` job is **blocking** on
  every PR that is not an upstream port (exempt via `port:` title
  prefix or `port/` branch). The other three gates
  (`doc-substance-check`, `adr-backfill-check`, copyright
  pre-commit) are advisory or pre-commit, never CI-blocking; this
  split is the whole point of ADR-0124 and an upstream sync must
  not move them into the required-status-check set without a
  follow-up ADR. The opt-out parser matches
  `/^-?\s*no .* (?:needed|impact|rebase-sensitive)/` per
  ADR-0108 §Opt-out-lines — if upstream ever changes PR-template
  phrasing (unlikely; this is fork-local), the regex and the
  template must move together.
- **Re-test**:

  ```bash
  # Lint the workflow + hook locally.
  pre-commit run --files \
    .github/workflows/rule-enforcement.yml \
    scripts/ci/check-copyright.sh \
    .pre-commit-config.yaml

  # Dry-run the copyright hook against a staged source file.
  scripts/ci/check-copyright.sh libvmaf/src/libvmaf.c && echo ok

  # Synthetic PR body that violates ADR-0108 should fail the parser;
  # see docs/research/0002-automated-rule-enforcement.md §Verification
  # plan for the three test cases.
  ```

### 0027 — SSIMULACRA 2 scalar extractor (libjxl FastGaussian IIR blur)

- **Workstream PRs**: this PR (`feat/ssimulacra2-scalar`); proposal
  ADR in PR #67.
- **Touches**:
  `libvmaf/src/feature/ssimulacra2.c` (fork-local, new),
  `libvmaf/src/meson.build`,
  `libvmaf/src/feature/feature_extractor.c`.
- **Invariant**: the extractor embeds several tables that must
  track libjxl upstream — opsin absorbance matrix, `MakePositiveXYB`
  offsets, 108 pooling weights, polynomial-transform coefficients,
  and the FastGaussian coefficient-derivation formulas
  (radius = `3.2795·σ + 0.2546`, Cramer's 3×3 solve for β, n2/d1
  assignment per Charalampidis 2016 (33)). If libjxl ever changes
  any of these, update `ssimulacra2.c` in the same PR that syncs
  upstream. Self-consistency must stay at exactly `100.000000` for
  identical ref/dist inputs — this is the cheapest regression check.
- **Re-test**:

  ```bash
  meson test -C build --suite=fast
  ./build/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc00_576x324.yuv \
    -w 576 -h 324 -p 420 -b 8 --feature ssimulacra2 -o /tmp/self.xml \
    && grep -q 'ssimulacra2="100.000000"' /tmp/self.xml \
    && echo "ok: self-consistency 100.0"
  ```

### 0028 — MS-SSIM separable decimate + AVX2/AVX-512/NEON SIMD

- **Workstream PRs**: `feat/ms-ssim-decimate-simd-v2` (supersedes the
  rebase-incompatible `feat/ms-ssim-decimate-simd`; AVX2/AVX-512,
  commits `7de8cd7f` scalar separable, `5f93c864` AVX2, `73436438`
  AVX-512); `feat/ms-ssim-decimate-neon-v2` (NEON follow-up, stacked).
- **Touches**:
  `libvmaf/src/feature/ms_ssim_decimate.{c,h}` (NEW),
  `libvmaf/src/feature/x86/ms_ssim_decimate_avx2.{c,h}` (NEW),
  `libvmaf/src/feature/x86/ms_ssim_decimate_avx512.{c,h}` (NEW),
  `libvmaf/src/feature/arm64/ms_ssim_decimate_neon.{c,h}` (NEW),
  `libvmaf/src/feature/ms_ssim.c` (call-site change),
  `libvmaf/src/meson.build` (register new SIMD TUs),
  `libvmaf/test/test_ms_ssim_decimate.c` (NEW),
  `libvmaf/test/meson.build` (arm64 gating).
- **Invariant**: the 9-tap 9/7 biorthogonal wavelet LPF
  coefficients (`ms_ssim_lpf_h` / `ms_ssim_lpf_v`) are duplicated
  verbatim in five TUs for bit-identity: the scalar
  `ms_ssim_decimate.c`, the AVX2 variant, the AVX-512 variant, the
  NEON variant, and upstream's `g_lpf_h` / `g_lpf_v` in
  `ms_ssim.c`. Any upstream change to the coefficient values or the
  `KBND_SYMMETRIC` mirror branch in `iqa/convolve.c` must be
  mirrored to all five. If not mirrored, SIMD paths and scalar
  diverge silently and the bit-equality `memcmp` in
  `test_ms_ssim_decimate` catches it — but only when that test
  runs, so diff the five files first.
- **Re-test** (on each supported host arch):

  ```bash
  # x86_64 host — native build.
  meson test -C build
  ./build/test/test_ms_ssim_decimate

  # aarch64 host OR aarch64 cross under qemu — see /tmp/aarch64-cross.txt.
  meson setup build-arm64 libvmaf --cross-file /tmp/aarch64-cross.txt \
      -Denable_cuda=false -Denable_sycl=false
  ninja -C build-arm64
  qemu-aarch64-static -L /usr/aarch64-linux-gnu \
      build-arm64/test/test_ms_ssim_decimate

  # Netflix MS-SSIM golden — places=4 must still pass through SIMD.
  .venv/bin/python -m pytest \
      python/test/feature_extractor_test.py::FeatureExtractorTest::test_run_ms_ssim_fextractor
  ```

### 0029 — `KBND_SYMMETRIC` period-based reflection in `iqa/convolve.c`

- **Workstream PRs**: `feat/ms-ssim-decimate-simd-v2` follow-up
  (CI triage on PR #69, 2026-04-20).
- **Touches**: `libvmaf/src/feature/iqa/convolve.c` (upstream
  file, rewritten `KBND_SYMMETRIC`).
- **Invariant**: `KBND_SYMMETRIC(img, w, h, x, y, _)` must use the
  period-based form (`period = 2*w`, `period = 2*h`) so that offsets
  with `|x| > w` or `|y| > h` still land in bounds. Upstream's
  single-reflect form was out-of-bounds whenever `w < kernel_half`
  or `h < kernel_half`; the latent bug did not reproduce in Netflix
  golden tests because MS-SSIM pyramids never decimate below
  ~60×34. Any upstream change that reverts to the single-reflect
  form must be rejected or re-ported.
- **Re-test**:

  ```bash
  ./build/test/test_ms_ssim_decimate        # test_1x1 border case
  .venv/bin/python -m pytest \
      python/test/feature_extractor_test.py::FeatureExtractorTest::test_run_ms_ssim_fextractor
  ```

### 0030 — `adm_decouple_s123_avx512` stack-array 64-byte alignment

- **Workstream PRs**: `feat/ms-ssim-decimate-simd-v2` follow-up
  (CI triage on PR #69, 2026-04-20).
- **Touches**: `libvmaf/src/feature/x86/adm_avx512.c` (upstream
  file, one-line `_Alignas(64)` on `int64_t angle_flag[16]` at
  line 1317). `libvmaf/test/test_pic_preallocation.c` (upstream
  file, three `vmaf_model_destroy(model)` calls pairing the
  `vmaf_model_load` in `test_picture_pool_basic` / `_small` /
  `_yuv444`).
- **Invariant**: the stack slot for `angle_flag` must be 64-byte
  aligned because two `_mm512_loadu_si512(&angle_flag[0/8])`
  loads in the same scope may be promoted to aligned `vmovdqa64`
  by LTO. Dropping the `_Alignas(64)` annotation re-introduces
  the SEGV under `--buildtype=release -Db_lto=true
  -Db_sanitize=address`. Debug / no-LTO builds keep
  `vmovdqu64` and cannot flag the regression. See
  `docs/development/known-upstream-bugs.md`.
- **Re-test**:

  ```bash
  meson setup build-asan-lto libvmaf \
      -Denable_cuda=false -Denable_sycl=false \
      -Db_sanitize=address --buildtype=release -Db_lto=true
  ninja -C build-asan-lto test/test_pic_preallocation
  ASAN_OPTIONS=detect_leaks=1 \
      ./build-asan-lto/test/test_pic_preallocation
  ```

### 0031 — Batch-A upstream-port small-fix sweep (ports of unmerged PRs)

- **Workstream PRs**: `feat/batch-a-upstream-small-fix-sweep` — commits
  `546a40ee` (T0-1), `8fed8ad1` (T4-4), `83a1db46` (T4-5),
  `34425dee` (T4-6). ADRs 0131, 0132, 0134, 0135.
- **Touches**:
  - `libvmaf/src/cuda/picture_cuda.c` (one-line `cuMemFree` port of
    [Netflix#1382][pr1382])
  - `libvmaf/src/feature/feature_collector.c` +
    `libvmaf/test/test_feature_collector.c` (mount/unmount bugfix
    port of [Netflix#1406][pr1406] + shared-helper test refactor)
  - `libvmaf/src/meson.build` (`declare_dependency` +
    `override_dependency` port of [Netflix#1451][pr1451])
  - `libvmaf/include/libvmaf/model.h`, `libvmaf/src/model.c`,
    `libvmaf/test/test_model.c`, `docs/api/index.md` (built-in
    model iterator port of [Netflix#1424][pr1424])
- **Invariant**: each of the four upstream PRs is OPEN (unmerged) on
  the port date; when Netflix merges any of them, the fork's
  version is correction-bearing (T4-4 test refactor, T4-6 three
  defect fixes + Doxygen doc expansion), not line-identical.
  Resolution on upstream merge is **always "keep fork version"**
  because the fork's version already satisfies the PR's intent
  and additionally fixes the defects.
  - Netflix#1406 conflict will land in `test_feature_collector.c`
    — fork uses `load_three_test_models()` helper vs upstream's
    inline per-model `VmafModel *m0, *m1, *m2;` duplication.
  - Netflix#1424 conflict will land in `libvmaf/src/model.c` and
    `libvmaf/test/test_model.c` — fork uses `else if` guard +
    `idx + 1 < CNT` + const-qualified test types.
  - Netflix#1382 and Netflix#1451 are line-identical in substance;
    merge should be clean aside from trailing-comma style drift.
- **Re-test**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false
  ninja -C build test/test_feature_collector test/test_model
  build/test/test_feature_collector
  build/test/test_model
  # Expected: 6/6 pass in test_feature_collector (mount/unmount
  # 3-model sequences); 39/39 pass in test_model (includes
  # test_version_next full-iteration invariant).
  ```

[pr1382]: https://github.com/Netflix/vmaf/pull/1382
[pr1406]: https://github.com/Netflix/vmaf/pull/1406
[pr1424]: https://github.com/Netflix/vmaf/pull/1424
[pr1451]: https://github.com/Netflix/vmaf/pull/1451

### 0032 — Thread-local locale handling for numeric I/O (port of Netflix/vmaf#1430)

- **Workstream PRs**: `port/netflix-1430-thread-locale`
  (T4-3 from the "Batch-A follow-up" sweep, 2026-04-20).
- **Touches**: `libvmaf/src/thread_locale.h` / `libvmaf/src/thread_locale.c`
  (new, upstream-authored); `libvmaf/src/meson.build` (two
  `cdata.set('HAVE_USELOCALE'/'HAVE_XLOCALE_H')` probes +
  `src_dir + 'thread_locale.c'` in `libvmaf_sources`);
  `libvmaf/src/output.c` (four writers gain
  `push_c()` + `pop()` bracket, preserving fork's
  `ferror(outfile) ? -EIO : 0` return contract from
  [ADR-0119](adr/0119-cli-precision-default-revert.md));
  `libvmaf/src/svm.cpp` (drop `<locale.h>` include; replace
  `setlocale/strdup/setlocale` bracket with
  `vmaf_thread_locale_push_c/pop`; add
  `buffer.imbue(std::locale::classic())` to both SVM parser ctors
  with fork's K&R + 4-space style);
  `libvmaf/src/read_json_model.c` (bracket `model_parse` with
  push/pop); `libvmaf/test/meson.build` (new
  `test_locale_handling` target + test registration);
  `libvmaf/test/test_locale_handling.c` (new, upstream-authored
  with three fork corrections for the `score_format` parameter).
- **Invariant**: fork's output writers return
  `ferror(outfile) ? -EIO : 0` — this must survive any upstream
  refactor of the writer bodies. The `push_c()` call MUST be
  paired with a `pop()` on every return path (writer bodies have
  a single tail return, so the pattern is locally
  `push → body → pop → return ferror-check`). Dropping
  `pop()` leaks a `locale_t` on POSIX and leaves the thread
  locked to "C" on Windows.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_locale_handling
  # Repro the user-visible failure without the fix:
  LC_ALL=de_DE.UTF-8 build/tools/vmaf --reference ref.yuv \
      --distorted dis.yuv --width 1920 --height 1080 \
      --pixel_format 420 --bitdepth 8 --output result.json \
      --json
  # Assert output contains period decimals, not comma.
  python -c "import json; d=json.load(open('result.json')); \
      assert all('.' in repr(v) for v in \
      [f['metrics']['vmaf'] for f in d['frames']])"
  ```

- **On upstream sync**: when Netflix merges PR #1430, the
  `(cherry picked from commit 054a97ed…)` trailer in
  `git log port/netflix-1430-thread-locale` lets the next
  `/sync-upstream` skip this commit. If the upstream diff
  drifts, redo the three fork corrections listed in
  [ADR-0137](adr/0137-thread-local-locale-for-numeric-io.md)
  §Decision.

### 0033 — SSIM / MS-SSIM SIMD bit-exact to scalar via per-lane scalar double

- **Workstream PRs**: `feat/ms-ssim-decimate-neon` (this PR —
  companion to the ADR-0138 convolve fast path).
- **Touches**:
  [`libvmaf/src/feature/x86/ssim_avx2.c`](../libvmaf/src/feature/x86/ssim_avx2.c)
  and
  [`libvmaf/src/feature/x86/ssim_avx512.c`](../libvmaf/src/feature/x86/ssim_avx512.c)
  — `ssim_accumulate_*` rewritten. `ssim_precompute_*` and
  `ssim_variance_*` unchanged (they were already bit-exact).
  Plus the new bit-exact
  [`convolve_avx2.c`](../libvmaf/src/feature/x86/convolve_avx2.c) /
  [`convolve_avx512.c`](../libvmaf/src/feature/x86/convolve_avx512.c)
  and the upstream h-pass OOB fix at
  [`iqa/convolve.c:159`](../libvmaf/src/feature/iqa/convolve.c#L159).
- **Invariants** (see
  [ADR-0139](adr/0139-ssim-simd-bitexact-double.md) §Decision):
  1. **Convolve taps** — *single-rounded `float*float` → widen →
     `double` add*, NO FMA. Mirrors scalar `sum += img[i]*k[j]` in
     [`iqa/convolve.c`](../libvmaf/src/feature/iqa/convolve.c).
  2. **SSIM accumulate** — scalar's `2.0 *` literal
     (`2.0 * ref_mu[i] * cmp_mu[i] + C1` and `2.0 * srsc + C2`)
     is a C `double` literal. Both SIMD accumulators do the `2.0 *`
     numerator + division + final `l*c*s` product per-lane in
     scalar double to match scalar type promotions byte-for-byte.
  3. **H-pass outer-loop bound** — `y < dst_h + vc - kh_even` (not
     `y < dst_h + vc`); the `- kh_even` is load-bearing because the
     last cache row on even-tap kernels (e.g. box-8) is never read
     by the v-pass but was previously written OOB when image height
     equals kernel height.

  Fork-local SSIM SIMD is NOT upstream. If upstream ever adds their
  own SSIM AVX2/AVX-512, **keep the fork's version on conflict** —
  it's the only variant verified bit-exact to scalar at
  `--precision max`.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_iqa_convolve test_ms_ssim_decimate
  # Bit-exactness check across dispatch backends:
  FIX=python/test/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv
  DIS=python/test/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv
  for m in 255 16 0; do
    build/tools/vmaf --cpumask $m --reference $FIX --distorted $DIS \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --feature float_ssim --feature float_ms_ssim \
        --output /tmp/ssim_$m.xml --precision max
  done
  diff <(grep -v '<fyi fps' /tmp/ssim_255.xml) \
       <(grep -v '<fyi fps' /tmp/ssim_16.xml)    # expect empty
  diff <(grep -v '<fyi fps' /tmp/ssim_255.xml) \
       <(grep -v '<fyi fps' /tmp/ssim_0.xml)     # expect empty
  ```

- **On upstream sync**: the AVX2/AVX-512 SSIM surface is entirely
  fork-local (upstream has VIF/ADM/motion/CAMBI SIMD but no SSIM).
  If upstream ever introduces SSIM SIMD, their kernel bodies will
  almost certainly compute `l*c*s` in vector float for throughput —
  **do not adopt**. The fork's per-lane-scalar-double reduction is
  required for the bit-exactness claim. Same applies to
  `convolve_avx2/512` — they are fork-only; dispatch sits in
  `ssim_tools.c` via `_iqa_convolve_set_dispatch`.

### 0034 — SIMD DX framework + NEON SSIM/convolve bit-exact port

- **Workstream PRs**: `feat/simd-dx-framework` (this PR, PR #A);
  ships the two demos on top of which PR #B will consume the
  framework (ssimulacra2, motion_v2, vif_statistic, ...).
- **Touches**:
  [`libvmaf/src/feature/simd_dx.h`](../libvmaf/src/feature/simd_dx.h)
  (new header),
  [`libvmaf/src/feature/arm64/convolve_neon.c`](../libvmaf/src/feature/arm64/convolve_neon.c)
  +
  [`convolve_neon.h`](../libvmaf/src/feature/arm64/convolve_neon.h)
  (new NEON port),
  [`libvmaf/src/feature/arm64/ssim_neon.c`](../libvmaf/src/feature/arm64/ssim_neon.c)
  (`ssim_accumulate_neon` rewritten for ADR-0139 bit-exactness;
  `precompute` + `variance` unchanged),
  [`libvmaf/src/feature/float_ssim.c`](../libvmaf/src/feature/float_ssim.c)
  +
  [`libvmaf/src/feature/float_ms_ssim.c`](../libvmaf/src/feature/float_ms_ssim.c)
  (wire `iqa_convolve_neon` into the aarch64 dispatch setters),
  [`libvmaf/src/meson.build`](../libvmaf/src/meson.build)
  (`arm64_sources` += convolve_neon.c),
  [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
  (`test_iqa_convolve` arch filter extended to `arm64` / `aarch64`),
  [`libvmaf/test/test_iqa_convolve.c`](../libvmaf/test/test_iqa_convolve.c)
  (NEON variant check + aarch64 CPU flag detection),
  [`libvmaf/test/dnn/meson.build`](../libvmaf/test/dnn/meson.build)
  (`test_cli.sh` gated on `not meson.is_cross_build()` — bash
  invokes `$VMAF_BIN` directly so meson's exe_wrapper isn't
  applied), new
  [`build-aux/aarch64-linux-gnu.ini`](../build-aux/aarch64-linux-gnu.ini)
  meson cross-file,
  [`.claude/skills/add-simd-path/SKILL.md`](../.claude/skills/add-simd-path/SKILL.md)
  (upgraded kernel-spec flags).
- **Invariants** (see [ADR-0140](adr/0140-simd-dx-framework.md)
  §Decision):
  1. `simd_dx.h` is fork-local. Keep the fork's version on upstream
     conflict. Macro names are ISA-suffixed (`_AVX2_4L`,
     `_AVX512_8L`, `_NEON_4L`) — do not collapse into a cross-ISA
     abstraction; the fork's SIMD policy
     (user-memory `feedback_simd_dx_scope.md`) rules out
     Highway / simde / xsimd.
  2. The ADR-0138 widen-then-add rule (single-rounded
     `float * float` → widen → `double` add, NO FMA) applies to
     NEON exactly as to AVX2 / AVX-512. The NEON form uses paired
     `float64x2_t` accumulators (lo / hi) because NEON has no
     `float64x4_t`.
  3. The ADR-0139 per-lane scalar-double reduction rule applies to
     `ssim_accumulate_neon` exactly as to the AVX2 / AVX-512
     variants. The NEON implementation uses
     `SIMD_ALIGNED_F32_BUF_NEON` (`_Alignas(16) float name[4]`) +
     a 4-iteration scalar loop.
- **Re-test** (requires `aarch64-linux-gnu-gcc` +
  `qemu-user-static` + aarch64 sysroot at `/usr/aarch64-linux-gnu`):

  ```bash
  cd libvmaf
  meson setup ../build-aarch64 \
    --cross-file ../build-aux/aarch64-linux-gnu.ini \
    -Denable_cuda=false -Denable_sycl=false -Denable_dnn=disabled
  cd ..
  ninja -C build-aarch64
  meson test -C build-aarch64                       # expect 31/31 OK
  # Bit-exactness check scalar vs NEON under QEMU:
  REF=python/test/resource/yuv/src01_hrc00_576x324.yuv
  DIS=python/test/resource/yuv/src01_hrc01_576x324.yuv
  for m in 255 0; do
    LD_LIBRARY_PATH=$PWD/build-aarch64/src qemu-aarch64-static \
      -L /usr/aarch64-linux-gnu build-aarch64/tools/vmaf \
      --cpumask $m --reference $REF --distorted $DIS \
      --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
      --feature float_ssim --feature float_ms_ssim \
      --output /tmp/ssim_$m.xml --precision max
  done
  diff <(grep -v '<fyi fps' /tmp/ssim_255.xml) \
       <(grep -v '<fyi fps' /tmp/ssim_0.xml)     # expect empty
  ```

- **On upstream sync**: upstream has no NEON SSIM and no NEON
  convolve for IQA. If they ever add one, **keep the fork's
  version on conflict** — the fork's NEON path is the only variant
  verified bit-exact to scalar at `--precision max`. The
  `build-aux/aarch64-linux-gnu.ini` cross-file has no upstream
  equivalent. The `/add-simd-path` skill is fork-only; upstream
  doesn't ship `.claude/skills/`.
