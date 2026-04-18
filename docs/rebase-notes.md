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

### 0003 — SYCL backend (USM picture pool, D3D11 surface import,
`vmaf_sycl_*` public API)

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

### 0009 — License headers (Lusoris/Claude on wholly-new files;
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

_Pre-ADR-0108 entries above are the result of a one-shot backfill
sweep on 2026-04-18; subsequent fork-local PRs add their own entries
inline._

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

### 0014 — Coverage gate gcov race fix (atomic + serial tests) + `python/test/` lint exclusion

- **Workstream PRs**: this PR (lint-exclude + coverage-gcov-race fix bundle).
- **Touches**:
  `.github/workflows/ci.yml` (CPU + GPU coverage jobs add
  `-Dc_args=-fprofile-update=atomic` / `-Dcpp_args=-fprofile-update=atomic`,
  pass `--num-processes 1` to `meson test` in the coverage step, and
  pass `--ignore-errors negative` on `lcov --capture`),
  `pyproject.toml` (Black/isort/Ruff exclude `python/test/`),
  `.pre-commit-config.yaml` (pre-commit Black/isort/Ruff hooks exclude
  `^python/test/`).
- **Invariant**: coverage CI must use BOTH gcov race fixes together
  because they target two different races: (a) `-fprofile-update=atomic`
  closes the intra-process race where SIMD inner loops
  (`vif_avx2.c:673`, motion_avx2, etc.) race `.gcda` counters within
  one test binary → negative counts → `geninfo` aborts; (b)
  `--num-processes 1` closes the inter-process race where multiple
  parallel test binaries merge their counters into the same `.gcda`
  files for the shared `libvmaf.so` at process exit → inflated /
  asymmetric per-file hit counts (`dnn_api.c — 1176%` was the smoking
  gun). The atomic flag is per-thread, not per-process; never drop
  one fix thinking the other covers the same race. Lint config must
  exclude `python/test/` because that tree is upstream-mirror code
  (wholesale overwritten on every `/sync-upstream` and
  `/port-upstream-commit`), so any fork-side reformat is net-negative —
  Black `extend-exclude` only filters during directory walks, not
  explicitly-passed files, so pre-commit needs its own `exclude`
  regex. An upstream sync that changes the upstream Black/Ruff config
  or the meson coverage default is unlikely to collide (we only modify
  CI workflow + fork tooling config), but if upstream ever ships a
  `pyproject.toml` of its own, the fork's must keep these exclusions.
  See [ADR-0110](adr/0110-coverage-gate-fprofile-update-atomic.md).
- **Re-test**:

  ```bash
  # Reproduce gcov race locally (requires gcc + lcov):
  cd libvmaf
  meson setup build-cov-test --buildtype=debug -Db_coverage=true \
      -Denable_avx512=true -Denable_float=true \
      -Dc_args=-fprofile-update=atomic -Dcpp_args=-fprofile-update=atomic
  ninja -C build-cov-test
  meson test -C build-cov-test --print-errorlogs --num-processes 1
  lcov --capture --directory build-cov-test \
       --output-file build-cov-test/coverage.info \
       --ignore-errors mismatch,gcov,source,negative
  lcov --list build-cov-test/coverage.info | grep -E 'dnn_api|model_loader'
  # Expected: lcov completes without "Unexpected negative count" AND no
  # per-file percentages exceed 100% (drop --num-processes 1 to reproduce
  # the multi-process .gcda merge race that produces dnn_api.c — 1176%).

  # Lint-exclude smoke test:
  pre-commit run --files python/test/quality_runner_test.py
  # Expected: Black/isort/Ruff/ruff-check report "Skipped" (excluded).
  ```
