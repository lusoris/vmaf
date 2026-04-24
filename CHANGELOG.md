# Change Log

> The "Unreleased / lusoris fork" section below tracks fork-specific changes
> on top of upstream Netflix/vmaf. From here on, release-please generates
> entries automatically from Conventional Commits.

## [Unreleased] — lusoris fork (3.0.0-lusoris.0)

### Added

- **`motion_v2` NEON SIMD** (fork-local): aarch64 users now get a
  NEON fast path for the `motion_v2` feature. Scalar + AVX2 + AVX-512
  variants already existed; this closes the ISA-parity gap (backlog
  T3-4). The NEON impl uses arithmetic right-shift throughout
  (`vshrq_n_s64`, `vshlq_s64(v, -bpc)`) to match the scalar C `>>`
  semantics byte-for-byte — deliberately diverging from the fork's
  AVX2 variant, which uses logical `_mm256_srlv_epi64` and can
  diverge on negative-diff pixels; an AVX2 re-audit is queued as
  follow-up. Five small `static inline` helpers keep every function
  under ADR-0141's 60-line budget; zero clang-tidy warnings, no
  NOLINT. Verified bit-exact under QEMU user-mode on the Netflix
  `src01_hrc00/01_576x324` pair. See
  [ADR-0145](docs/adr/0145-motion-v2-neon-bitexact.md).

### Fixed

- **`float_ms_ssim` rejects input below 176×176 at init**
  (Netflix upstream issue
  [#1414](https://github.com/Netflix/vmaf/issues/1414)). The
  5-level 11-tap MS-SSIM pyramid walks off the kernel footprint
  at a mid-level scale for inputs below 176×176 (QCIF and
  smaller), previously producing a confusing mid-run `error:
  scale below 1x1!` + cascading `problem reading pictures` /
  `problem flushing context`. The fix checks `w < GAUSSIAN_LEN
  << (SCALES - 1)` at init and returns `-EINVAL` with a helpful
  error that names the input resolution, the required minimum
  (176×176), and the upstream issue. Minimum is derived from
  the existing filter constants so it stays in sync if those
  ever change. Visible behaviour: init now fails immediately
  instead of mid-stream; zero impact on inputs ≥176×176. New
  3-subtest reducer in `test_float_ms_ssim_min_dim.c` verified
  to fail pre-fix and pass post-fix. Closes backlog item T1-4.
  See [ADR-0153](docs/adr/0153-float-ms-ssim-min-dim-netflix-1414.md).

### Changed

- **`vmaf_read_pictures` now rejects non-monotonic indices with
  `-EINVAL`** (Netflix upstream issue
  [#910](https://github.com/Netflix/vmaf/issues/910)). The
  `integer_motion` / motion2 / motion3 extractors keep sliding-
  window state keyed by `index % N`, so submitting frames out of
  order or with duplicate indices silently corrupts their
  ring-buffers. The reported symptom was a missing
  `integer_motion2_score` on the last frame whenever submission
  order didn't match frame order. The fix is a monotonic-index
  guard at the API boundary (new `last_index` + `have_last_index`
  fields on `VmafContext`, checked inside the existing
  `read_pictures_validate_and_prep` helper from ADR-0146): strictly
  increasing indices are accepted (gaps fine); duplicates and
  regressions return `-EINVAL`. **Visible behaviour change**:
  duplicates / out-of-order submissions that previously produced
  silent-wrong-answer now fail with `-EINVAL` — well-defined
  rejection replaces ill-defined corruption. Zero impact on
  in-tree callers (vmaf CLI + test suite already iterate strictly
  increasing); downstream integrations that deliberately submit
  non-monotonic indices need to either track the next-index
  themselves or reset the context. 3-subtest reducer in
  `test_read_pictures_monotonic.c` verified to fail pre-fix and
  pass post-fix. Closes backlog item T1-2. See
  [ADR-0152](docs/adr/0152-vmaf-read-pictures-monotonic-index.md).

### Added

- **i686 (32-bit x86) build-only CI job** (reproduces Netflix
  upstream issue [#1481](https://github.com/Netflix/vmaf/issues/1481)).
  New matrix row in `.github/workflows/libvmaf-build-matrix.yml`
  (`Build — Ubuntu i686 gcc (CPU, no-asm)`) invokes
  `meson setup libvmaf libvmaf/build --cross-file=build-aux/i686-linux-gnu.ini -Denable_asm=false`,
  pinning the workaround documented in upstream's bug report.
  New cross-file `build-aux/i686-linux-gnu.ini` (gcc + `-m32`,
  `cpu_family = 'x86'`, `cpu = 'i686'`) + new install-deps step
  installing `gcc-multilib` + `g++-multilib`. Test + tox steps
  skipped for the i686 leg because meson marks cross-built tests
  as `SKIP 77` (the host can run i686 binaries natively but meson
  doesn't know that). Fixing the underlying AVX2
  `_mm256_extract_epi64` compile failure (24 call sites in
  `adm_avx2.c`) is **explicitly out of scope** — this entry adds
  the CI gate only. Closes backlog item T4-8. See
  [ADR-0151](docs/adr/0151-i686-ci-netflix-1481.md).

- **Windows MSYS2/MinGW CUDA build support** (port of Netflix
  upstream PR [#1472](https://github.com/Netflix/vmaf/pull/1472),
  birkdev, 2026-03-16, OPEN). Enables
  `-Denable_cuda=true -Denable_nvcc=true` on Windows with MSYS2 +
  MinGW-GCC host compiler + MSVC Build Tools + CUDA toolkit.
  Source-portability guards in CUDA headers + `.cu` files: drop
  `<pthread.h>` from `cuda/common.h`; DEVICE_CODE guards on
  `<ffnvcodec/*>` vs `<cuda.h>` in `cuda_helper.cuh` +
  `picture.h`; `#ifndef DEVICE_CODE` around `feature_collector.h`
  in 5 ADM `.cu` files. Meson build plumbing: `vswhere`-based
  `cl.exe` discovery (without adding it to PATH, which would
  break MinGW-GCC CPU build), Windows SDK + MSVC include path
  injection via `-I` flags to nvcc, CUDA version detection via
  `nvcc --version` (replaces `meson.get_compiler('cuda')` which
  needs MSVC as default C compiler). Fork carve-outs: keep
  positional (not `#ifndef __CUDACC__`) initializers in
  `integer_adm.h`; keep `pthread_dependency` on `cuda_static_lib`
  because `ring_buffer.c` still uses pthread directly; merge
  fork's ADR-0122 gencode coverage block with upstream's new
  nvcc-detect block. Drive-by: rename reserved `__VMAF_SRC_*_H__`
  header guards to `VMAF_SRC_*_INCLUDED` per ADR-0141. Linux
  CPU build 32/32 + Linux CUDA build 35/35 pass; Windows CUDA
  build not yet CI-validated (tracked as T7-3 — self-hosted
  Windows+CUDA runner enrollment). Closes backlog item T4-2.
  See [ADR-0150](docs/adr/0150-port-netflix-1472-cuda-windows.md).

### Fixed

- **FIFO-mode workfile/procfile opens no longer race-hang on slow
  systems** (port of Netflix upstream PR
  [#1376](https://github.com/Netflix/vmaf/pull/1376)). The Python
  harness under `python/vmaf/core/executor.py` +
  `python/vmaf/core/raw_extractor.py` previously waited for child
  processes to create named pipes via a 1-second `os.path.exists()`
  polling loop, which could time out on loaded CI / virtualised
  hosts. Replaced with `multiprocessing.Semaphore(0)` signalled
  by the child processes after `os.mkfifo(...)`; parent acquires
  with 5-second soft-timeout warning then blocks indefinitely.
  Applied to both the base `Executor` class and the
  `ExternalVmafExecutor`-style subclass. Fork carve-outs:
  upstream's `__version__ = "3.0.0" → "4.0.0"` bump is **not**
  applied (fork tracks its own versioning per ADR-0025); unused
  `from time import sleep` imports removed per ADR-0141.
  Closes backlog item T4-7. See
  [ADR-0149](docs/adr/0149-port-netflix-1376-fifo-semaphore.md).

### Changed

- **IQA reserved-identifier rename + touched-file lint cascade
  cleanup** (refactor, fork-local). Rename every `_iqa_*` /
  `struct _kernel` / `_ssim_int` / `_map_reduce` / `_map` /
  `_reduce` / `_context` / `_ms_ssim_map` / `_ssim_map` /
  `_ms_ssim_reduce` / `_ssim_reduce` / `_alloc_buffers` /
  `_free_buffers` symbol and the underscore-prefixed header
  guards (`_CONVOLVE_H_`, `_DECIMATE_H_`, `_SSIM_TOOLS_H_`,
  `__VMAF_MS_SSIM_DECIMATE_H__`) to their non-reserved
  spellings across the IQA tree (21 files). Sweeps the
  ADR-0141 touched-file lint cascade that surfaced
  (~40 pre-existing warnings across `ssim.c`, `ms_ssim.c`,
  `integer_ssim.c`, `iqa/*.{c,h}`, `convolve_*.{c,h}`,
  `test_iqa_convolve.c`): `static` / cross-TU NOLINT for
  `misc-use-internal-linkage`, `size_t` casts for
  `bugprone-implicit-widening-of-multiplication-result`,
  multi-decl splits, function-size refactors of `calc_ssim` /
  `compute_ssim` / `compute_ms_ssim` / `run_gauss_tests` via
  small named `static` helpers, `(void)` casts for unused
  feature-extractor lifecycle parameters, and scoped
  NOLINTBEGIN/END for `clang-analyzer-security.ArrayBound`
  false positives on the kernel-offset clamps and for
  `clang-analyzer-unix.Malloc` on test-helper allocations
  that leak by design at process exit. Bit-identical VMAF
  score on Netflix golden pair `src01_hrc00/01_576x324`
  (scalar vs SIMD, with `--feature float_ssim --feature
  float_ms_ssim` and the full `vmaf_v0.6.1` model). Closes
  backlog item T7-6. See
  [ADR-0148](docs/adr/0148-iqa-rename-and-cleanup.md).

- **Thread-pool job-object recycling** (perf, fork-local port of
  Netflix upstream PR [#1464](https://github.com/Netflix/vmaf/pull/1464),
  thread-pool portion only). `libvmaf/src/thread_pool.c` now recycles
  `VmafThreadPoolJob` slots via a mutex-protected free list rather
  than `malloc`/`free` on every enqueue, and stores payloads ≤ 64
  bytes inline in the job struct (`char inline_data[64]`) so the
  common-case enqueue path avoids a second allocation entirely.
  Adapted to the fork's `void (*func)(void *data, void **thread_data)`
  signature and per-worker `VmafThreadPoolWorker` data path (which
  upstream lacks). **~1.8–2.6× enqueue throughput** on a 500 000-job
  4-thread micro-benchmark; bit-identical VMAF scores between
  `--threads 4` and serial, and between `VMAF_CPU_MASK=0` and `=255`
  under `--threads 4`. Closes the thread-pool half of backlog T3-6
  (the AVX2 PSNR half was already covered by fork commit `81fcd42e`).
  See [ADR-0147](docs/adr/0147-thread-pool-job-pool.md).

- **Function-size NOLINT sweep** — refactored every
  `readability-function-size` NOLINT suppression in `libvmaf/src/` (20
  sites across 12 files: `dict.c`, `picture.c`, `picture_pool.c`,
  `predict.c`, `libvmaf.c`, `output.c`, `read_json_model.c`,
  `feature/feature_extractor.c`, `feature/feature_collector.c`,
  `feature/iqa/convolve.c`, `feature/iqa/ssim_tools.c`,
  `feature/x86/vif_statistic_avx2.c`) into small named `static`
  helpers. IQA / SIMD files use `static inline` helpers threaded
  through an explicit `struct vif_simd8_lane` to preserve the
  ADR-0138 / ADR-0139 bit-exactness invariants (per-lane scalar-float
  reduction, single-rounded float-mul → widen → double-add).
  Netflix-golden-pair VMAF score remains bit-identical between
  `VMAF_CPU_MASK=0` and `VMAF_CPU_MASK=255`. Zero new NOLINTs
  introduced. Drive-by fixes: TU-static `_calc_scale` →
  `iqa_calc_scale` for `bugprone-reserved-identifier`; tightened
  `calloc(w * h, ...)` widening; separated multi-declaration forms;
  `model_collection_parse_loop` now writes directly through
  `cfg_name` instead of the aliased `c->name` (drops the last
  `readability-non-const-parameter` NOLINT). See
  [ADR-0146](docs/adr/0146-nolint-sweep-function-size.md).

- **VIF AVX2 convolve: generalised for arbitrary filter widths** (port of
  Netflix upstream [`f3a628b4`](https://github.com/Netflix/vmaf/commit/f3a628b4),
  Kyle Swanson, 2026-04-21). `libvmaf/src/feature/common/convolution_avx.c`
  drops from 2,747 LoC of branch-unrolled kernels specialised to
  `fwidth ∈ {3, 5, 9, 17}` down to 247 LoC of a single parametric 1-D
  scanline pair. New `MAX_FWIDTH_AVX_CONV` ceiling in `convolution.h`
  lets the VIF AVX2 dispatch in `vif_tools.c` drop its hard-coded
  fwidth whitelist. Fork cleanup per ADR-0141: four helpers now
  `static`, strides widened to `ptrdiff_t` to eliminate
  `bugprone-implicit-widening-of-multiplication-result` on every
  pointer-offset site. Paired with a 10× loosening of the Netflix
  golden tolerance on two full-VMAF assertions
  (`VMAF_score`, `VMAFEXEC_score`: `places=2` → `places=1`),
  matching Netflix's own upstream test update. The generalised
  kernel's accumulation order differs at ULP scale vs the
  specialised ones; drift is orders of magnitude below perceptual
  discriminability. See
  [ADR-0143](docs/adr/0143-port-netflix-f3a628b4-generalized-avx-convolve.md).

### Added

- **VIF: configurable `vif_sigma_nsq` feature parameter**: port of
  Netflix upstream [`18e8f1c5`](https://github.com/Netflix/vmaf/commit/18e8f1c5)
  (Kyle Swanson, 2026-04-20) promoting VIF's hard-coded neural-noise
  variance `static const float sigma_nsq = 2` into a runtime-configurable
  double parameter `vif_sigma_nsq` (range `[0.0, 5.0]`, alias `snsq`,
  default `2.0`). Threaded through `compute_vif` → `vif_statistic_s` and
  the fork-local `vif_statistic_s_avx2` AVX2 variant (which upstream does
  not ship; its signature was extended in lockstep so both paths agree on
  the new 14-argument contract). Default-path scores bit-identical to
  pre-port master. Use via CLI:
  `vmaf --feature float_vif:snsq=2.5 ...` or per-model. See
  [ADR-0142](docs/adr/0142-port-netflix-18e8f1c5-vif-sigma-nsq.md).
- **Governance — Q2 2026 modernization ADRs**: four Proposed ADRs +
  four research digests scoping the next modernization workstreams
  (no implementation yet):
  - [ADR-0126](docs/adr/0126-ssimulacra2-extractor.md) /
    [Research-0003](docs/research/0003-ssimulacra2-port-sourcing.md):
    SSIMULACRA 2 feature extractor (port libjxl C++ reference).
  - [ADR-0127](docs/adr/0127-vulkan-compute-backend.md) /
    [Research-0004](docs/research/0004-vulkan-backend-design.md):
    Vulkan compute backend (volk + GLSL→SPIR-V + VMA, VIF
    pathfinder).
  - [ADR-0128](docs/adr/0128-embedded-mcp-in-libvmaf.md) /
    [Research-0005](docs/research/0005-embedded-mcp-transport.md):
    Embedded MCP server in libvmaf (SSE + UDS + stdio, flag-gated).
  - [ADR-0129](docs/adr/0129-tinyai-ptq-quantization.md) /
    [Research-0006](docs/research/0006-tinyai-ptq-accuracy-targets.md):
    Tiny-AI PTQ int8 (static + dynamic + QAT per-model via
    `model/registry.json`).
- **SIMD DX framework — `simd_dx.h` + upgraded `/add-simd-path` skill**:
  fork-internal header
  ([`libvmaf/src/feature/simd_dx.h`](libvmaf/src/feature/simd_dx.h))
  that codifies the ADR-0138 (widen-then-add) and ADR-0139 (per-lane
  scalar-double reduce) patterns as ISA-suffixed macros
  (`SIMD_WIDEN_ADD_F32_F64_AVX2_4L` / `_AVX512_8L` / `_NEON_4L`,
  `SIMD_ALIGNED_F32_BUF_*`, `SIMD_LANES_*`). Zero runtime overhead —
  each macro documents its scalar C equivalent and is guarded by the
  matching `__AVX2__` / `__AVX512F__` / `__ARM_NEON` ifdef. The
  `/add-simd-path` skill
  ([`.claude/skills/add-simd-path/SKILL.md`](.claude/skills/add-simd-path/SKILL.md))
  gained `--kernel-spec=widen-add-f32-f64|per-lane-scalar-double|none`,
  `--lanes=N`, and `--tail=scalar|masked` flags so new SIMD TUs
  scaffold from a short declaration instead of a cold copy-paste.
  Demonstrated on two real kernels in the same PR: a new bit-exact
  `iqa_convolve_neon`
  ([`libvmaf/src/feature/arm64/convolve_neon.c`](libvmaf/src/feature/arm64/convolve_neon.c))
  and a bit-exactness fix for `ssim_accumulate_neon` that mirrors the
  ADR-0139 x86 fix. Together they complete the SSIM / MS-SSIM SIMD
  coverage on aarch64. See
  [ADR-0140](docs/adr/0140-simd-dx-framework.md) +
  [research digest 0013](docs/research/0013-simd-dx-framework.md).
- **aarch64 cross-compile lane**:
  [`build-aux/aarch64-linux-gnu.ini`](build-aux/aarch64-linux-gnu.ini)
  meson cross-file for `aarch64-linux-gnu-gcc` +
  `qemu-aarch64-static`. The `test_iqa_convolve` meson target now
  covers `arm64` / `aarch64` alongside `x86_64` / `x86` so future NEON
  convolve changes gate on the same bit-exactness contract as the x86
  variants.
- **I18N / thread-safety**: `thread_locale.h/.c` cross-platform thread-local
  locale abstraction ported from upstream PR
  [Netflix/vmaf#1430](https://github.com/Netflix/vmaf/pull/1430) (Diego Nieto,
  Fluendo). `vmaf_write_output_{xml,json,csv,sub}`, `svm_save_model`,
  `vmaf_read_json_model`, and both SVM model parsers now switch the calling
  thread's locale to `"C"` for numeric I/O instead of using the
  process-global `setlocale` bracket. POSIX.1-2008 `uselocale` +
  `newlocale(LC_ALL_MASK)` on Linux/macOS/BSD; `_configthreadlocale` +
  per-thread `setlocale` on Windows; graceful no-op fallback elsewhere.
  Fixes a latent data-race under multi-threaded hosts (ffmpeg filter graphs
  with multiple VMAF instances, MCP server worker pools) where one thread's
  `setlocale(LC_ALL, "C")` bracket would clobber another thread's active
  locale mid-call. See
  [ADR-0137](docs/adr/0137-thread-local-locale-for-numeric-io.md).
- **Public API**: `vmaf_model_version_next(prev, &version)` iterator for
  enumerating the built-in VMAF model versions compiled into the
  library. Opaque-handle cursor — NULL to start, NULL-return to stop.
  Ports [Netflix#1424](https://github.com/Netflix/vmaf/pull/1424) with
  three correctness corrections (NULL-pointer arithmetic UB,
  off-by-one returning the sentinel, const-qualifier mismatches in the
  test); see [ADR-0135](docs/adr/0135-port-netflix-1424-expose-builtin-model-versions.md).
- **Build**: libvmaf now exports `libvmaf_dep` via `declare_dependency`
  and registers an `override_dependency('libvmaf', ...)` in
  `libvmaf/src/meson.build`, so the fork is consumable as a meson
  subproject with the standard `dependency('libvmaf')` idiom. Ports
  [Netflix#1451](https://github.com/Netflix/vmaf/pull/1451); see
  [ADR-0134](docs/adr/0134-port-netflix-1451-meson-declare-dependency.md).
- **Metric**: SSIMULACRA 2 scalar feature extractor
  ([`libvmaf/src/feature/ssimulacra2.c`](libvmaf/src/feature/ssimulacra2.c))
  — port of libjxl's perceptual similarity metric on top of the fork's
  YUV pipeline. Ingests YUV 4:2:0/4:2:2/4:4:4 at 8/10/12 bpc with a
  configurable YUV→RGB matrix (`yuv_matrix` option, BT.709 limited
  default), converts through linear RGB → XYB → 6-scale pyramid with
  SSIMMap + EdgeDiffMap + canonical 108-weight polynomial pool.
  Pyramid blur is a bit-close C port of libjxl's `FastGaussian`
  3-pole recursive IIR (`lib/jxl/gauss_blur.cc`,
  Charalampidis 2016 truncated-cosine approximation, k={1,3,5},
  zero-pad boundaries). Registered as feature `ssimulacra2` — one
  scalar per frame in `[0, 100]`, identity inputs return exactly
  `100.000000`. Scalar only; AVX2/AVX-512/NEON follow-ups are
  separate PRs. See
  [ADR-0130](docs/adr/0130-ssimulacra2-scalar-implementation.md) +
  [Research-0007](docs/research/0007-ssimulacra2-scalar-port.md).
- **CLI**: `--precision $spec` flag for score output formatting.
  - `N` (1..17) → `printf "%.<N>g"`
  - `max` / `full` → `"%.17g"` (round-trip lossless, opt-in)
  - `legacy` → `"%.6f"` (synonym for the default)
  - default (no flag) → `"%.6f"` (Netflix-compatible per ADR-0119;
    supersedes ADR-0006's original `%.17g` default)
- **Public API**: `vmaf_write_output_with_format()` accepts a `score_format`
  string; old `vmaf_write_output()` routes through the new function with
  `"%.6f"` default.
- **GPU backends**: SYCL/oneAPI backend (Lusoris + Claude); CUDA backend
  optimizations (decoupled buffer elimination, VIF rd_stride, ADM inline
  decouple).
- **Numerical correctness**: float ADM `sum_cube` and `csf_den_scale` use
  double-precision accumulation in scalar/AVX2/AVX512 paths to eliminate
  ~8e-5 drift between scalar and SIMD reductions.
- **MS-SSIM SIMD**: separable scalar-FMA decimate with AVX2 (8-wide),
  AVX-512 (16-wide), and NEON (4-wide) variants for the 9-tap 9/7
  biorthogonal wavelet LPF used by the MS-SSIM scale pyramid. Per-lane
  `_mm{256,512}_fmadd_ps` (x86) / `vfmaq_n_f32` (aarch64) with
  broadcast coefficients produces output byte-identical to the scalar
  reference; stride-2 horizontal deinterleave via
  `_mm256_shuffle_ps`+`_mm256_permute4x64_pd` (AVX2),
  `_mm512_permutex2var_ps` (AVX-512), and `vld2q_f32` (NEON). Runtime
  dispatch prefers AVX-512 > AVX2 > scalar on x86 and NEON > scalar on
  arm64. Netflix MS-SSIM golden passes at places=4 through every
  dispatched path; 10 synthetic `memcmp` cases (1x1 border, odd
  dimensions, 1920x1080) verify strict byte-equality in
  [`libvmaf/test/test_ms_ssim_decimate.c`](libvmaf/test/test_ms_ssim_decimate.c).
  See [ADR-0125](docs/adr/0125-ms-ssim-decimate-simd.md).
- **AI-agent scaffolding**: `.claude/` directory with 7 specialized review
  agents (c-, cuda-, sycl-, vulkan-, simd-, meson-reviewer, perf-profiler),
  18 task skills, hooks for unsafe-bash blocking and auto-format,
  `CLAUDE.md` + `AGENTS.md` onboarding, `docs/principles.md` (Power-of-10 +
  JPL-C-STD + CERT + MISRA).
- **Quality gates**: GitHub Actions workflows for CI (Netflix golden gate
  D24, sanitizers, cross-backend ULP), lint (clang-tidy, cppcheck,
  pre-commit), security (semgrep, CodeQL, gitleaks, dependency-review),
  supply-chain (SBOM, Sigstore keyless signing, SLSA L3 provenance).
- **Tiny AI**: nightly `bisect-model-quality` workflow
  ([`.github/workflows/nightly-bisect.yml`](.github/workflows/nightly-bisect.yml))
  runs `vmaf-train bisect-model-quality` against a deterministic
  synthetic placeholder cache
  ([`ai/testdata/bisect/`](ai/testdata/bisect/),
  reproducible from
  [`ai/scripts/build_bisect_cache.py`](ai/scripts/build_bisect_cache.py))
  and posts the verdict + per-step PLCC/SROCC/RMSE table to sticky
  tracker issue #40. Real DMOS-aligned cache swaps in via a follow-up;
  see [ADR-0109](docs/adr/0109-nightly-bisect-model-quality.md) +
  [Research-0001](docs/research/0001-bisect-model-quality-cache.md).
  Closes #4.
- **CI**: three DNN-enabled matrix legs in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  — `Build — Ubuntu gcc (CPU) + DNN`, `Build — Ubuntu clang (CPU) + DNN`,
  `Build — macOS clang (CPU) + DNN`. Each leg installs ONNX Runtime
  (Linux: MS tarball pinned to 1.22.0; macOS: Homebrew) and runs the
  meson `dnn` test suite plus full `ninja test`. The two Linux legs
  are pinned to required status checks on `master`; the macOS leg
  stays `experimental: true` because Homebrew ORT floats. See
  [ADR-0120](docs/adr/0120-ai-enabled-ci-matrix-legs.md) +
  [`docs/rebase-notes.md` entry 0021](docs/rebase-notes.md).
- **CI**: two Windows GPU build-only matrix legs in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  — `Build — Windows MSVC + CUDA (build only)` and
  `Build — Windows MSVC + oneAPI SYCL (build only)`. Both gate the
  MSVC build-portability of the CUDA host code and SYCL `vmaf_sycl_*`
  C-API entry points, respectively. No test step (windows-latest has
  no GPU). Both legs are pinned to required status checks on `master`.
  See [ADR-0121](docs/adr/0121-windows-gpu-build-only-legs.md) +
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: Win32 `pthread.h` compat shim at
  [`libvmaf/src/compat/win32/pthread.h`](libvmaf/src/compat/win32/pthread.h)
  — header-only, maps the in-use pthread subset (mutex / cond / thread
  create+join+detach + `PTHREAD_MUTEX_INITIALIZER` /
  `PTHREAD_COND_INITIALIZER`) onto Win32 SRWLOCK + CONDITION_VARIABLE +
  `_beginthreadex`. Wired in via a new `pthread_dependency` in
  `libvmaf/meson.build`, gated on `cc.check_header('pthread.h')`
  failing — POSIX and MinGW (winpthreads) builds are untouched. Lets
  the Windows MSVC GPU legs from ADR-0121 actually compile the libvmaf
  core (~14 TUs `#include <pthread.h>` unconditionally). Pattern
  mirrors the long-standing `compat/gcc/stdatomic.h` shim. nvcc fatbin
  and icpx SYCL `custom_target`s additionally thread the shim include
  path through `cuda_extra_includes` / `sycl_inc_flags` on Windows
  (custom targets bypass meson's `dependencies:` plumbing).
- **Build**: SYCL Windows host-arg handling in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) — `icpx-cl`
  on Windows targets `x86_64-pc-windows-msvc` and rejects `-fPIC`.
  `sycl_common_args` / `sycl_feature_args` now route the flag through
  `sycl_pic_arg = host_machine.system() != 'windows' ? ['-fPIC'] : []`
  instead of hard-coding it. PIC is the default for Windows DLLs, so
  dropping the flag is the correct build-system fix, not a workaround.
- **Build**: SYCL Windows source portability — four MSVC C++
  blockers fixed so `icpx-cl` compiles the SYCL TUs.
  (1) [`libvmaf/src/ref.h`](libvmaf/src/ref.h) +
  [`libvmaf/src/feature/feature_extractor.h`](libvmaf/src/feature/feature_extractor.h)
  (UPSTREAM) gained an `#if defined(__cplusplus) && defined(_MSC_VER)`
  branch that pulls `atomic_int` via `using std::atomic_int;` —
  MSVC's `<stdatomic.h>` only surfaces the C11 typedefs in
  `namespace std::` under C++, while gcc/clang expose them globally
  via a GNU extension. POSIX paths fall through to the original
  `<stdatomic.h>` line; ABI unchanged. (2)
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  switched `<libvmaf/log.h>` (non-existent) to `"log.h"` (the actual
  internal header). (3)
  [`libvmaf/src/sycl/dmabuf_import.cpp`](libvmaf/src/sycl/dmabuf_import.cpp)
  moved `<unistd.h>` inside `#if HAVE_SYCL_DMABUF` — POSIX `close()`
  is only used in the VA-API path, so non-DMA-BUF hosts (Windows
  MSVC, macOS) no longer fail with `'unistd.h' file not found`. (4)
  [`libvmaf/src/sycl/common.cpp`](libvmaf/src/sycl/common.cpp)
  replaced POSIX `clock_gettime(CLOCK_MONOTONIC)` with
  `std::chrono::steady_clock` — guaranteed monotonic by the C++
  standard and portable on every supported host. All four preserve
  POSIX/Linux behaviour bit-identically. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: CUDA Windows source portability — fifth MSVC blocker
  fixed on the CUDA leg's CPU SIMD compile path.
  [`libvmaf/src/feature/x86/motion_avx2.c`](libvmaf/src/feature/x86/motion_avx2.c)
  (UPSTREAM) line 529 indexed an `__m256i` directly
  (`final_accum[0] + ... + final_accum[3]`) — gcc/clang allow this
  via the GNU vector extension, MSVC rejects it with `C2088:
  built-in operator '[' cannot be applied to an operand of type
  '__m256i'`. Replaced with four `_mm256_extract_epi64` calls,
  summed — bit-exact lane sum on every compiler. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: x86 SIMD Windows source portability — sweep that
  finishes the MSVC compile of the libvmaf CPU SIMD layer.
  Round-19 surfaced the same vector-extension pattern at 19 more
  call sites plus 6 GCC-style `(__m256i)x` casts.
  [`libvmaf/src/feature/x86/adm_avx2.c`](libvmaf/src/feature/x86/adm_avx2.c)
  (UPSTREAM) had 6 lines using
  `(__m256i)(_mm256_cmp_ps(...))` casts (replaced with
  `_mm256_castps_si256(...)`) and 12 sites of `__m128i[N]`
  lane-extract reductions (replaced with `_mm_extract_epi64`).
  [`libvmaf/src/feature/x86/adm_avx512.c`](libvmaf/src/feature/x86/adm_avx512.c)
  (UPSTREAM) had 6 sister lane-extract reductions on the
  AVX-512 paths.
  [`libvmaf/src/feature/x86/motion_avx512.c`](libvmaf/src/feature/x86/motion_avx512.c)
  (UPSTREAM, ported from PR #1486) had one final lane-extract
  reduction. All 19 + 6 fixes are bit-exact rewrites — gcc/clang
  emit identical vextract+padd sequences either way.
  Additionally
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  switched from C-style COBJMACROS helpers
  (`ID3D11Device_CreateTexture2D`, etc.) to C++ method-call syntax
  (`device->CreateTexture2D`) because d3d11.h gates COBJMACROS
  behind `!defined(__cplusplus)` and the TU compiles as C++
  under icpx-cl. ABI-equivalent. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: x86 SIMD alignment specifier — round-20 swap from
  GCC trailing `__attribute__((aligned(N)))` to C11-standard
  leading `_Alignas(N)` across 17 scratch-buffer sites in
  `vif_statistic_avx2.c` (UPSTREAM), `ansnr_avx{2,512}.c`
  (UPSTREAM), `float_adm_avx{2,512}.c` (UPSTREAM),
  `float_psnr_avx{2,512}.c` (UPSTREAM) and `ssim_avx{2,512}.c`
  (UPSTREAM). Same alignment guarantee, MSVC-portable
  (`/std:c11`). The pre-existing portable `ALIGNED(x)` macro in
  `vif_avx{2,512}.c` was already MSVC-clean and remains untouched.
- **Build**: `mkdirp` Windows portability —
  [`libvmaf/src/feature/mkdirp.c`](libvmaf/src/feature/mkdirp.c)
  and
  [`libvmaf/src/feature/mkdirp.h`](libvmaf/src/feature/mkdirp.h)
  (third-party MIT-licensed micro-library) gate `<unistd.h>` to
  non-Windows, add `<direct.h>` + `_mkdir` on MSVC, and provide a
  local `mode_t` typedef (MSVC's `<sys/types.h>` doesn't declare
  it). The `mode` argument is silently ignored on the Windows
  path — same behaviour as before for POSIX callers. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: round-21 MSVC mop-up —
  [`libvmaf/src/feature/x86/adm_avx512.c`](libvmaf/src/feature/x86/adm_avx512.c)
  (UPSTREAM) adds six more `_mm_extract_epi64` rewrites at lines
  2128 / 2135 / 2142 / 2589 / 2595 / 2601 that the round-19 sweep
  missed (bit-exact).
  [`libvmaf/src/log.c`](libvmaf/src/log.c) (UPSTREAM) gates
  `<unistd.h>` to non-Windows and pulls `_isatty` / `_fileno` from
  `<io.h>` on MSVC via macro redirection; the single `isatty(fileno
  (stderr))` call site compiles unchanged on every platform.
  See [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **CI**: `.github/workflows/lint-and-format.yml` pre-commit job
  now checks out with `lfs: true`. Without it `model/tiny/*.onnx`
  lands as LFS pointer stubs and pre-commit's "changes made by
  hooks" reporter flags the stubs as pre-commit-induced
  modifications against HEAD's resolved blobs, failing the job
  even though no hook touched them. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: round-21e MSVC mop-up — the Windows MSVC legs now
  build the full tree (CLI tools, unit tests, `libvmaf.dll`)
  instead of the earlier short cut of skipping tools / tests.
  Source changes:
  (i) eight C99 variable-length arrays converted to compile-time
  constants or heap allocations —
  [`libvmaf/src/predict.c:385,453`](libvmaf/src/predict.c),
  [`libvmaf/src/libvmaf.c:1741`](libvmaf/src/libvmaf.c),
  [`libvmaf/src/read_json_model.c:517,520`](libvmaf/src/read_json_model.c),
  [`libvmaf/test/test_feature_extractor.c:56`](libvmaf/test/test_feature_extractor.c),
  [`libvmaf/test/test_cambi.c:254`](libvmaf/test/test_cambi.c),
  [`libvmaf/test/test_pic_preallocation.c:382,506`](libvmaf/test/test_pic_preallocation.c);
  (ii) fork-added POSIX/GNU `getopt_long` shim at
  [`libvmaf/tools/compat/win32/`](libvmaf/tools/compat/win32/)
  (header + ~260-line companion source) declared via a single
  `getopt_dependency` in
  [`libvmaf/meson.build`](libvmaf/meson.build) that
  auto-propagates the .c into the `vmaf` CLI and
  `test_cli_parse`;
  (iii) `pthread_dependency` threaded through the eleven test
  targets in
  [`libvmaf/test/meson.build`](libvmaf/test/meson.build)
  that transitively include `<pthread.h>` via
  `feature_collector.h`;
  (iv) `<unistd.h>` → `<io.h>` redirection
  (`isatty`/`fileno` → `_isatty`/`_fileno`) added to
  [`libvmaf/tools/vmaf.c`](libvmaf/tools/vmaf.c);
  (v) `<unistd.h>` → `<windows.h>` + `Sleep` macros
  added to
  [`libvmaf/test/test_ring_buffer.c`](libvmaf/test/test_ring_buffer.c)
  and
  [`libvmaf/test/test_pic_preallocation.c`](libvmaf/test/test_pic_preallocation.c)
  for `usleep` / `sleep`;
  (vi) `__builtin_clz` / `__builtin_clzll` MSVC fallback via
  `__lzcnt` / `__lzcnt64` extracted into
  [`libvmaf/src/feature/compat_builtin.h`](libvmaf/src/feature/compat_builtin.h)
  and included from the three TUs that use the builtin
  (`integer_adm.c`, `x86/adm_avx2.c`, `x86/adm_avx512.c`);
  (vii) `extern "C"` wrap added around
  `#include "log.h"` in
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  so `vmaf_log` resolves against the C-linkage symbol
  produced by `log.c` when this .cpp TU gets pulled into
  a SYCL-enabled test executable by icpx-cl. Upstream
  `log.h` has no `__cplusplus` guard; the wrap keeps the
  fork-local fix inside the fork-added .cpp instead of
  touching the shared header.
  Workflow change: both Windows MSVC matrix legs now pass
  `--default-library=static` in `meson_extra` because libvmaf's
  public API carries no `__declspec(dllexport)` — a vanilla
  MSVC shared build produces an empty import lib and tools
  fail with `LNK1181`. Mirrors the MinGW leg's static-link
  choice. Both MSVC CUDA and MSVC SYCL legs validated
  locally end-to-end on a Windows Server 2022 VM with
  CUDA 13.0, oneAPI 2025.3, and Level Zero loader v1.18.5
  prior to push.
  See [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md)
  paragraphs (h)–(p).
- **CUDA**: out-of-the-box GPU coverage for Ampere `sm_86` (RTX 30xx)
  and Ada `sm_89` (RTX 40xx). The gencode array in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) now
  unconditionally emits cubins for `sm_75` / `sm_80` / `sm_86` /
  `sm_89` plus a `compute_80` PTX backward-JIT fallback, independent
  of host `nvcc` version. Upstream Netflix only shipped cubins at Txx
  major boundaries, so Ampere-`sm_86` / Ada-`sm_89` ran on a
  `compute_90` PTX that cannot JIT backward and fell over at
  kernel-load time on every consumer 3080/3090/4070/4090. See
  [ADR-0122](docs/adr/0122-cuda-gencode-coverage-and-init-hardening.md)
  and [`docs/rebase-notes.md` entry 0023](docs/rebase-notes.md).
- **CUDA**: actionable init-failure logging in
  [`libvmaf/src/cuda/common.c`](libvmaf/src/cuda/common.c). When
  `cuda_load_functions()` (the `nv-codec-headers` dlopen wrapper
  around `libcuda.so.1`) fails, `vmaf_cuda_state_init()` now emits a
  multi-line message naming the missing library, the loader-path
  check command (`ldconfig -p | grep libcuda`), and the docs section
  at
  [`docs/backends/cuda/overview.md#runtime-requirements`](docs/backends/cuda/overview.md#runtime-requirements).
  A parallel message on `cuInit(0)` failure distinguishes
  driver-load failure from userspace/kernel version skew. Also fixes
  a pre-existing leak on both error paths (`cuda_free_functions()` +
  `free(c)` + `*cu_state = NULL`). See
  [ADR-0122](docs/adr/0122-cuda-gencode-coverage-and-init-hardening.md).
- **Automated rule-enforcement for four process ADRs**: new workflow
  [`.github/workflows/rule-enforcement.yml`](.github/workflows/rule-enforcement.yml)
  plus a pre-commit `check-copyright` hook
  ([`scripts/ci/check-copyright.sh`](scripts/ci/check-copyright.sh)) close
  the "rule-without-a-check" gap on
  [ADR-0100](docs/adr/0100-project-wide-doc-substance-rule.md),
  [ADR-0105](docs/adr/0105-copyright-handling-dual-notice.md),
  [ADR-0106](docs/adr/0106-adr-maintenance-rule.md), and
  [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md). The
  ADR-0108 six-deliverable checklist is **blocking**; the other
  three are advisory comments because their predicates require
  human judgement (pure-refactor exemption, ADR-triviality call,
  copyright-template choice). Upstream-port PRs (`port:` title /
  `port/` branch) are exempt. Reviewer documentation at
  [`docs/development/automated-rule-enforcement.md`](docs/development/automated-rule-enforcement.md).
  First `--all-files` pass also backfilled 18 pre-existing missing
  headers (13 upstream C files Netflix 2016–2026, 4 fork-authored
  NEON sources + `python/compat/config.h` Lusoris+Claude 2026);
  `libvmaf/src/pdjson.{c,h}` (vendored JSON parser) and
  `python/vmaf/matlab/` (upstream MATLAB MEX) are excluded from
  the hook rather than receiving synthetic headers. See
  [ADR-0124](docs/adr/0124-automated-rule-enforcement.md) and
  [Research-0002](docs/research/0002-automated-rule-enforcement.md).

### Changed

- **Upstream port — ADM** (Netflix `966be8d5`, fork PR #44, merged
  `d06dd6cf`): integer ADM kernels + AVX2/AVX-512 SIMD paths +
  `barten_csf_tools.h` ported wholesale; `i4_adm_cm` signature extended
  from 8 to 13 args. Netflix golden VMAF mean unchanged at
  `76.66890` (places=4 OK). See
  [`docs/rebase-notes.md` entry 0012](docs/rebase-notes.md).
- **Upstream port — motion** (Netflix PR #1486 head `2aab9ef1`, sister
  to ADM port): integer motion + AVX2/AVX-512 paths +
  `motion_blend_tools.h` ported wholesale; new `integer_motion3`
  sub-feature appears in the default VMAF model output. Golden mean
  shifts `76.66890` → `76.66783` (within `places=2` tolerance the
  upstream PR loosened to). See
  [`docs/rebase-notes.md` entry 0013](docs/rebase-notes.md).
- Python diagnostic output (`Result._get_perframe_score_str`) now emits
  scores at `%.17g` instead of `%.6f` for round-trip reproducibility.
- Copyright headers across Netflix-authored sources updated `2016-2020` →
  `2016-2026`.
- **CI hygiene — Node 24 stragglers**: finish the `@v7` bump left over
  after the rename sweep (rebase-notes 0019/0020). `scorecard.yml`
  SHA-pinned `actions/upload-artifact@330a01c4... # v5` →
  `@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1`, and
  `nightly-bisect.yml` `actions/setup-python@v5` → `@v6`. Resolves the
  last `Node.js 20 actions are deprecated` runner warnings ahead of
  the **2026-06-02** forced-Node-24 cutover (full Node-20 removal
  2026-09-16). Every other workflow was already bumped on master.
- **Engineering process**: every fork-local PR now ships the six
  deep-dive deliverables (research digest, decision matrix in the
  ADR, `AGENTS.md` invariant note, reproducer command, fork-changelog
  entry, rebase note) in the same PR. New scaffolding:
  [`docs/research/`](docs/research/),
  [`docs/rebase-notes.md`](docs/rebase-notes.md), updated
  [`PR template`](.github/PULL_REQUEST_TEMPLATE.md). See
  [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md). Existing
  fork-local PRs have a one-shot rebase-notes backfill (10 grouped
  workstream entries) so the next upstream sync starts from a
  populated ledger. Closes #38.
- **Coverage gate**: end-to-end overhaul. (1) Build with
  `-fprofile-update=atomic` (CPU + GPU jobs) so parallel meson tests
  stop racing the `.gcda` counters on instrumented SIMD inner loops —
  eliminates the "Unexpected negative count for vif_avx2.c:673"
  geninfo hard-fail. (2) Run `meson test --num-processes 1` in the
  coverage steps so multiple test binaries don't concurrently merge
  their counters into the same `.gcda` files for the shared
  `libvmaf.so` (the on-exit merge is a multi-PROCESS race the atomic
  flag does not cover). (3) Switch `lcov` → `gcovr`: `gcovr`
  deduplicates `.gcno` files belonging to the same source compiled
  into multiple targets, fixing the `dnn_api.c — 1176%` over-count
  that surfaced after (1)+(2) on the first attempt. (4) Install
  ONNX Runtime in the coverage job and build with
  `-Denable_dnn=enabled` so `libvmaf/src/dnn/*.c` contribute real
  coverage instead of stubs (the 85% per-critical-file gate was
  previously unmeasurable). Coverage artifact is now
  `coverage.{xml,json,txt}` (Cobertura + gcovr JSON summary +
  human-readable text). (5) Carve `vmaf_use_tiny_model` out of
  `libvmaf/src/dnn/dnn_api.c` into a new
  `libvmaf/src/dnn/dnn_attach_api.c` so the unit-test binaries —
  which pull in `dnn_sources` for `feature_lpips.c` but never link
  `libvmaf.c` — don't end up with an undefined reference to
  `vmaf_ctx_dnn_attach` once `enable_dnn=enabled` activates the real
  bodies. The new TU is wired into `libvmaf.so` only via a separate
  `dnn_libvmaf_only_sources` list.
  See [ADR-0110](docs/adr/0110-coverage-gate-fprofile-update-atomic.md)
  (race fixes) and
  [ADR-0111](docs/adr/0111-coverage-gate-gcovr-with-ort.md)
  (gcovr + ORT) and
  [`docs/rebase-notes.md` entry 0014](docs/rebase-notes.md).
- **Lint scope**: upstream-mirror Python tests under `python/test/*.py`
  are now linted at the same standard as fork-added code. Mechanical
  Black + isort reformat applied to the four Netflix golden test
  files (`feature_extractor_test.py`, `quality_runner_test.py`,
  `vmafexec_test.py`, `vmafexec_feature_extractor_test.py`) — no
  assertion values changed; imports regrouped, line wrapping
  normalised. `python/test/resource/` (binary fixtures) remains
  excluded. Per user direction "don't skip linting on upstream
  things": `/sync-upstream` and `/port-upstream-commit` will
  re-trigger lint failures whenever upstream rewrites these files,
  and the fix is another in-tree reformat pass — never an exclusion.
  See [`docs/rebase-notes.md` entry 0014](docs/rebase-notes.md).
- **Coverage Gate annotations cleanup**: `actions/upload-artifact@v5|@v6
  → @v7` (and `actions/download-artifact@v5 → @v7` on supply-chain.yml)
  across every workflow under `.github/workflows/`, ahead of GitHub's
  2026-06-02 forced-Node-24 cutoff that turns the current Node 20
  deprecation banner into a hard error. Coverage Gate gcovr
  invocations also pipe stderr through `grep -vE 'Ignoring
  (suspicious|negative) hits' ... || true` so the chatty annotation
  for legitimately-large hit counts on tight inner loops (e.g.
  `ansnr_tools.c:207` at ~4.93 G hits across an HD multi-frame
  coverage suite) is dropped without losing the underlying data —
  `--gcov-ignore-parse-errors=suspicious_hits.warn` still tells
  gcovr to accept the count, only the annotation is filtered. The
  filter regex is anchored to gcov's exact warning prefix, so any
  *other* gcovr warning still surfaces. See
  [ADR-0117](docs/adr/0117-coverage-gate-warning-noise-suppression.md)
  and [`docs/rebase-notes.md` entry 0015](docs/rebase-notes.md).

### Fixed

- **SSIM / MS-SSIM NEON bit-exactness to scalar**: fork-local
  `ssim_accumulate_neon`
  ([`libvmaf/src/feature/arm64/ssim_neon.c`](libvmaf/src/feature/arm64/ssim_neon.c))
  previously carried the same ~0.13 float-ULP drift on
  `float_ms_ssim` / ~6 × 10⁻⁸ drift on `float_ssim` that ADR-0139
  fixed for AVX2 / AVX-512 — it was never surfaced because CI has no
  aarch64 runner. The NEON accumulator now computes the float-valued
  intermediates in vector float (`float32x4_t`) and spills to
  `SIMD_ALIGNED_F32_BUF_NEON(4)` buffers so the
  `2.0 * mu1 * mu2 + C1` numerator + division + `l*c*s` triple
  product run per-lane in scalar double, matching the x86 fix. Also
  plugged the aarch64 `iqa_convolve` gap — there was no NEON convolve
  at all before this PR; the VIF / ADM features used the scalar path
  on aarch64 while x86 ran AVX2 / AVX-512. Verified bit-identical
  under `qemu-aarch64-static` on both Netflix `src01_hrc00/01_576x324`
  and `checkerboard_1920_1080_10_3` pairs at `--precision max`. See
  [ADR-0140](docs/adr/0140-simd-dx-framework.md) +
  [research digest 0013](docs/research/0013-simd-dx-framework.md).
- **SSIM / MS-SSIM AVX2 + AVX-512 bit-exactness to scalar**: fork-local
  `ssim_accumulate_avx2` / `ssim_accumulate_avx512`
  ([`libvmaf/src/feature/x86/ssim_avx2.c`](libvmaf/src/feature/x86/ssim_avx2.c),
  [`libvmaf/src/feature/x86/ssim_avx512.c`](libvmaf/src/feature/x86/ssim_avx512.c))
  previously computed the `l`, `c`, `s` factors as vector float and
  produced the `l * c * s` triple product in float before accumulating
  to double — that diverged from the scalar reference by ~0.13 float
  ULPs (8th decimal) on `float_ms_ssim`, because scalar evaluates
  `2.0 * mu1 * mu2 + C1` and `2.0 * srsc + C2` in double (the literal
  `2.0` is a C `double`) and runs `lv * cv * sv` as three double
  multiplies. The SIMD accumulators now compute the float-valued
  intermediates (`srsc`, denominators, `sv`) in vector float and do
  the double-precision numerator + division + triple product per-lane
  in scalar double inside an 8/16-wide inner loop, matching scalar
  byte-for-byte. Verified: scalar = AVX2 = AVX-512 bit-identical at
  `--precision max` on both Netflix `src01_hrc00/01_576x324` and
  `checkerboard_1920_1080_10_3` pairs. `ssim_precompute_*` and
  `ssim_variance_*` were already bit-exact (pure elementwise float
  ops). Companion fix to the new bit-exact `_iqa_convolve_avx2/512`
  dispatch. See
  [ADR-0139](docs/adr/0139-ssim-simd-bitexact-double.md) +
  [ADR-0138](docs/adr/0138-iqa-convolve-avx2-bitexact-double.md).
- **CUDA multi-session `vmaf_cuda_picture_free` assertion-0 crash**:
  two or more concurrent CUDA sessions freeing pictures tripped
  `Assertion 0 failed` inside the driver because
  `cuMemFreeAsync(ptr, stream)` enqueued the free on a stream that
  was destroyed two statements later. The fork swaps the async call
  for synchronous `cuMemFree` at
  [`libvmaf/src/cuda/picture_cuda.c:247`](libvmaf/src/cuda/picture_cuda.c#L247);
  the preceding `cuStreamSynchronize` already removed any async
  overlap so perf is unchanged. Ports
  [Netflix#1382](https://github.com/Netflix/vmaf/pull/1382)
  (tracking [Netflix#1381](https://github.com/Netflix/vmaf/issues/1381));
  see [ADR-0131](docs/adr/0131-port-netflix-1382-cumemfree.md).
- **`vmaf_feature_collector_mount_model` list-corruption on ≥3
  models**: the upstream body advanced the `*head` pointer-to-pointer
  instead of walking a local cursor, overwriting the head element
  with its own successor and losing every entry past the second.
  Fork rewrites mount/unmount in
  [`libvmaf/src/feature/feature_collector.c`](libvmaf/src/feature/feature_collector.c)
  with a correct traversal, and `unmount_model` now returns
  `-ENOENT` (not `-EINVAL`) when the requested model isn't mounted
  so callers can distinguish misuse from not-found. Test coverage
  extended to a 3-element mount / unmount sequence. Ports
  [Netflix#1406](https://github.com/Netflix/vmaf/pull/1406);
  see [ADR-0132](docs/adr/0132-port-netflix-1406-feature-collector-model-list.md).
- **`KBND_SYMMETRIC` sub-kernel-radius out-of-bounds reflection**:
  upstream's 2-D symmetric boundary extension reflected the index a
  single time, which leaves out-of-bounds values whenever the input
  dimension is smaller than the kernel half-width (for the 9-tap
  MS-SSIM LPF, `n ≤ 3`). The fork rewrites `KBND_SYMMETRIC` in
  [`libvmaf/src/feature/iqa/convolve.c`](libvmaf/src/feature/iqa/convolve.c)
  and the scalar / AVX2 / AVX-512 / NEON `ms_ssim_decimate_mirror`
  helpers into the period-based form (`period = 2*n`) that bounces
  correctly for any offset. Netflix golden outputs are unchanged
  because 576×324 and 1920×1080 inputs never exercise the
  sub-kernel-radius regime. See
  [`docs/development/known-upstream-bugs.md`](docs/development/known-upstream-bugs.md)
  and [ADR-0125](docs/adr/0125-ms-ssim-decimate-simd.md).
- **`adm_decouple_s123_avx512` LTO+release SEGV**: the stack array
  `int64_t angle_flag[16]` is read via two `_mm512_loadu_si512`
  calls. Under `--buildtype=release -Db_lto=true`, link-time
  alignment inference promotes them to `vmovdqa64`, which faults
  because the C default stack alignment for `int64_t[16]` is 8
  bytes. Annotating the array with `_Alignas(64)` at
  [`libvmaf/src/feature/x86/adm_avx512.c:1317`](libvmaf/src/feature/x86/adm_avx512.c#L1317)
  keeps both the unaligned source form and the LTO-promoted aligned
  form correct. Debug / no-LTO builds, and every CI sanitizer job,
  are unaffected.
- **`test_pic_preallocation` VmafModel leaks**:
  `test_picture_pool_basic` / `_small` / `_yuv444` loaded a
  `VmafModel` via `vmaf_model_load` and never freed it, so
  LeakSanitizer reported 208 B direct + 23 KiB indirect per test.
  Paired each load with `vmaf_model_destroy(model)` in
  [`libvmaf/test/test_pic_preallocation.c`](libvmaf/test/test_pic_preallocation.c).
- **`libvmaf_cuda` ffmpeg filter segfault on first frame**: external
  reporter (2026-04-19) hit a SIGSEGV in `vmaf_ref_fetch_increment` on
  every invocation of ffmpeg's `libvmaf_cuda` filter against the fork's
  master build. Root cause is a three-commit composition: upstream
  `32b115df` (2026-04-07) added the experimental `VMAF_PICTURE_POOL`
  with an always-live `vmaf->prev_ref` slot; upstream `f740276a`
  (2026-04-09) moved the `vmaf_picture_ref(&vmaf->prev_ref, ref)` tail
  onto the non-threaded path without guarding against `ref->ref ==
  NULL`; fork commit `65460e3a` ([ADR-0104](docs/adr/0104-picture-pool-always-on.md))
  dropped the `VMAF_PICTURE_POOL` meson gate for ABI stability
  (+10 fps CPU gain), exposing the unguarded deref to every default
  build. On the CUDA-device-only extractor set that the ffmpeg filter
  registers, `rfe_hw_flags` returns `HW_FLAG_DEVICE` only,
  `translate_picture_device` early-returns without downloading, and
  `ref_host` stays zero-initialised — the subsequent
  `vmaf_picture_ref(&prev_ref, &ref_host)` deref'd `NULL`. Fix is a
  narrow null-guard at `libvmaf/src/libvmaf.c:1428`
  (`if (ref && ref->ref) vmaf_picture_ref(...)`). Semantically correct,
  not merely defensive: the only `VMAF_FEATURE_EXTRACTOR_PREV_REF`
  consumer is CPU `integer_motion_v2`, which is never registered
  alongside a pure-CUDA set. SYCL is unaffected (`vmaf_read_pictures_sycl`
  does not touch `prev_ref`). Always-on picture pool stays. See
  [ADR-0123](docs/adr/0123-cuda-post-cubin-load-regression-32b115df.md);
  follow-up item to port the null-guard upstream to Netflix/vmaf.
- **VIF `init()` fail-path leak**: `libvmaf/src/feature/integer_vif.c`'s
  `init()` carves one `aligned_malloc` into the VifBuffer sub-pointers by
  walking a `uint8_t *data` cursor forward through the allocation. When
  `vmaf_feature_name_dict_from_provided_features` returned NULL, the
  fail-path called `aligned_free(data)` on the *advanced* cursor — not a
  valid `aligned_malloc` return — leaking the whole block and passing a
  garbage pointer to `free`. Fail path now frees `s->public.buf.data`,
  the saved base pointer. Ported from Netflix upstream PR
  [#1476](https://github.com/Netflix/vmaf/pull/1476); the companion
  void*→uint8_t* UB portability fix from that PR is already on master
  (commit `b0a4ac3a`, rebase-notes 0022 §e).
- **CLI precision default reverted to `%.6f` (Netflix-compat)**: ADR-0006
  shipped `%.17g` as the default for round-trip-lossless output, but
  several Netflix golden tests in `python/test/command_line_test.py`,
  `vmafexec_test.py` etc. do *exact-string* matches against XML output
  (not `assertAlmostEqual`), so the wider default broke the gate. Default
  now matches upstream Netflix byte-for-byte; `--precision=max` (alias
  `full`) is the explicit opt-in for `%.17g`. `--precision=legacy` is
  preserved as a synonym for the (new) default. Library
  `vmaf_write_output_with_format(..., NULL)` and `python/vmaf/core/result.py`
  formatters revert in lockstep. See
  [ADR-0119](docs/adr/0119-cli-precision-default-revert.md) (supersedes
  [ADR-0006](docs/adr/0006-cli-precision-17g-default.md)). Latent on
  master 2026-04-15 → 2026-04-19; surfaced by ADR-0115's CI consolidation
  routing tox through master-targeting PRs.
- **`--frame_skip_ref` / `--frame_skip_dist` hang**: the skip loops in
  `libvmaf/tools/vmaf.c` fetched pictures from the preallocated picture
  pool (now always-on per ADR-0104) but never `vmaf_picture_unref`'d
  them, exhausting the pool after N skips and blocking the next fetch
  indefinitely. Each skipped picture is now unref'd immediately after
  fetch. Surfaced by `test_run_vmafexec_with_frame_skipping{,_unequal}`
  hanging locally (timeout 60 s, no output written) once tox started
  exercising both flags on master-targeting PRs.
- **CI tox doctest collection**: `pytest --doctest-modules` errored on five
  upstream files under `python/vmaf/resource/` (parameter / dataset / example
  config files; `vmaf_v7.2_bootstrap.py` and friends — dots in the stem make
  them unimportable as Python modules). Tox commands now pass
  `--ignore=vmaf/resource` so doctest collection skips that subtree. The
  files carry no doctests to begin with, so this is correctness, not a
  workaround. Surfaced by ADR-0115's CI trigger consolidation, which finally
  ran tox on PRs to master.

- **SYCL build with non-icpx host CXX**: `libvmaf/src/meson.build`
  unconditionally added `-fsycl` to the libvmaf shared-library link args
  whenever SYCL was enabled, even when the project's C++ compiler was
  gcc / clang / msvc. The host link driver does not understand `-fsycl`
  and failed with `g++: error: unrecognized command-line option '-fsycl'`
  at the `libvmaf.so` link step. The arg is now gated on
  `meson.get_compiler('cpp').get_id() == 'intel-llvm'`. The runtime
  libraries (libsycl + libsvml + libirc + libze_loader) declared as link
  dependencies already cover the gcc/clang link path, matching the
  documented "host C++ + sidecar icpx" project mode. Surfaced by
  ADR-0115's CI consolidation, which added an Ubuntu SYCL job that
  exercises this configuration on PRs to master.

- **FFmpeg patch series application**: `Dockerfile` and
  `.github/workflows/ffmpeg.yml` now walk `ffmpeg-patches/series.txt`
  and apply each patch in order via `git apply` with a `patch -p1`
  fallback. The Dockerfile previously `COPY`'d only patch 0003 (which
  fails to apply standalone because it references `LIBVMAFContext`
  fields added by patch 0001), and `ffmpeg.yml` referenced a stale
  `../patches/ffmpeg-libvmaf-sycl.patch` that no longer existed.
  Patches `0001-libvmaf-add-tiny-model-option.patch`,
  `0002-add-vmaf_pre-filter.patch`, and
  `0003-libvmaf-wire-sycl-backend-selector.patch` were also
  regenerated via real `git format-patch -3` so they carry valid
  `index <sha>..<sha> <mode>` header lines (the originals were
  hand-stubbed with placeholder SHAs and `git apply` choked on them).
  Docker images and CI FFmpeg-SYCL builds now exercise the full
  fork-added FFmpeg surface (tiny-AI + `vmaf_pre` + SYCL selector),
  not just SYCL. Also drops the bogus `--enable-libvmaf-sycl`
  configure flag (patch 0003 wires SYCL via `check_pkg_config`
  auto-detection — there is no such configure switch) and splits
  the Dockerfile's nvcc flags into a libvmaf set
  (`NVCC_FLAGS`, retains the four `-gencode` lines plus
  `--extended-lambda` and the `--expt-*` flags for Thrust/CUB) and
  an FFmpeg set (`FFMPEG_NVCC_FLAGS`, single-arch
  `compute_75,sm_75` matching FFmpeg's own modern-nvcc default —
  PTX is forward-compatible via driver JIT) so FFmpeg's
  `check_nvcc -ptx` probe stops failing with `nvcc fatal: Option
  '--ptx (-ptx)' is not allowed when compiling for multiple GPU
  architectures`. Also drops `--enable-libnpp` from FFmpeg
  configure — FFmpeg n8.1 explicitly `die`s if libnpp >= 13.0
  (configure:7335-7336 `"libnpp support is deprecated, version
  13.0 and up are not supported"`), and we don't actually use
  scale_npp / transpose_npp filters in VMAF workflows; cuvid +
  nvdec + nvenc + libvmaf-cuda are what we exercise. Patch 0002
  also gained a missing `#include "libavutil/imgutils.h"` for
  `av_image_copy_plane` (caught by the local docker build —
  upstream FFmpeg builds with `-Werror=implicit-function-declaration`).
  See ADR-0118 and entry 0018.

- **CI workflow naming**: renamed all six core `.github/workflows/*.yml`
  files to purpose-descriptive kebab-case (e.g. `ci.yml` →
  `tests-and-quality-gates.yml`, `libvmaf.yml` →
  `libvmaf-build-matrix.yml`) and normalised every workflow `name:` and
  job `name:` to Title Case. Required-status-check contexts in
  `master` branch protection re-pinned in the same merge window. See
  [ADR-0116](docs/adr/0116-ci-workflow-naming-convention.md) +
  [`docs/rebase-notes.md` entry 0020](docs/rebase-notes.md).

### Re-attributed

- 11 SYCL files in `libvmaf/{include,src,test}/.../sycl/` from
  `Netflix, Inc.` to `Lusoris and Claude (Anthropic)` — these files were
  authored entirely by the fork.

## (2022-04-11) [v2.3.1]

This is a minor release with some CAMBI extensions and speed-ups and adding it
to AOM CTC v3, as well as a few minor fixes/cleanups.

- CAMBI extensions: full reference, PQ eotf, up to 16 bit-depth support,
  max_log_contrast parameter.
- CAMBI: option to output heatmaps.

## (2021-10-16) [v2.3.0]

New release to add CAMBI (Contrast Aware Multiscale Banding Index).

- Python library: add encode width and height to Asset.
- libvmaf: add pixel format VMAF_PIX_FMT_YUV400P.
- Add cambi; add tests.
- Improve documentation. (#912)

## (2021-09-20) [v2.2.1]

This is another minor release to address a few last minute items for the AOM CTC
v2, as well as a few minor fixes/cleanups.

- Fix a race condition in vmaf_thread_pool_wait(). (#894)
- Avoid chroma resampling for 420mpeg2 y4m input (#906)

## (2021-07-02) [v2.2.0]

This is a minor release to address a few items for the AOM CTC v2, as well as a
few minor fixes/cleanups.

- Fixes a CIEDE-2000 precision issue, where cross-platform mismatches were seen.
  (#878)
- Adds libvmaf API function vmaf_feature_dictionary_free(). (#879)

## (2021-01-13) [v2.1.1]

This is a minor release to address a few last minute items for the initial AOM CTC.

**New features:**

- Fixes a SSIM/MS-SSIM precision bug where a lossless comparison did not always
  result in a perfect 1.0 score. (#796).
- Adds feature extractor options to clip the dB scores for both PSNR/SSIM.
  --aom_ctc v1.0 has been updated to use these clipping options according to the
  AOM CTC. (#802).

## (2020-12-30) [v2.1.0]

This is a minor release for the initial AOM CTC. Support has been added for
templated feature names. While this is a general purpose software feature,
templated feature names are immediately useful for simultaneous computation of
VMAF and VMAF NEG since the two metrics rely on slightly different VIF/ADM
variations. Global feature overrides via the `--feature` flag are no longer
supported, instead individual models can have their features overloaded
individually, the syntax for which is as follows:

 ```sh
--model version=vmaf_v0.6.1:vif.vif_enhn_gain_limit=1.0:adm.adm_enhn_gain_limit=1.0
```

**New features:**

- Per-model feature overloading via new API `vmaf_model_feature_overload()`.
- Multiple unique configurations of the same feature extractor may be registered
  run at the same time.
- `--aom_ctc v1.0` preset, encompassing all metrics specified by the AOM CTC.

## (2020-12-4) [2.0.0]

**New features:**

- Add PSNR-HVS and CIEDE2000 metrics.
- ci/actions: upload linux/macos artifacts (#738)
- libvmaf/feature: deprecate daala_ssim (#735)
- libvmaf: remove support for pkl models
- libvmaf/psnr: rewrite using integer types, 2x speedup
- vmaf: if no model is specified, enable v0.6.1 by default (#730)
- libvmaf/x86: add AVX2/AVX-512 optimizations for adm, vif and motion
- ci/actions: add xxd to build dependencies for Windows
- libvmaf: add support for built-in models
- libvmaf/integer_vif: use symmetrical mirroring on edges
- Fix log2 by replacing log2f_approx with log2f
- libvmaf_rc: provide a backwards compatible compute_vmaf(), link vmafossexec with
  libvmaf
- libvmaf: add framework support for json models
- libvmaf/libsvm: update libsvm to version 324
- libvmaf/motion: add motion_force_zero to motion fex
- return sha1 if Asset string is longer than 255
- Add CID/iCID Matlab source code
- build: unbreak x86 builds (Fixes: #374)
- Add 12bit and 16bit support for python YUV reader; add tests.
- Add PypsnrFeatureExtractor
- Add processes to FeatureAssembler. (#662)

**Fixed bugs:**

- fix motion flush for single frame input
- Fixing the perf_metric for a single entry list input

## (2020-8-24) [1.5.3]

(Updates since 1.5.1)

**Fixed bugs:**

- Fix inverted height and width in integer_motion in vmaf_rc (#650).

**New features:**

- libvmaf: add support for CSV and JSON logging
- Python: Add an (optional) step in Executor class to do python-based processing
  to ref/dis files (#523).
- Restructure python project and documentation (#544).
- Move test resource to Netflix/vmaf_resource repo (#552).
- Add Github CI (#558).
- Add vmaf_float_v0.6.1neg model; add vif_enhn_gain_limit and adm_enhn_gain_limit
  options to vmaf_rc.
- Update documentation for FFmpeg+libvmaf.
- Improvements to AucPerfMetric (#643).
- Add motion_force_zero option to vmaf_rc.

## (2020-6-30) [1.5.2]

**Fixed bugs:**

- Fix pkgconfig version sync issue (#572)

**New features:**

- libvmaf_rc general improvements

## (2020-2-27) [1.5.1]

**New features:**

- `libvmaf` has been relocated, and now has its own self-enclosed source tree
  (`./libvmaf/`) and build system (`meson`).
- Update license to BSD+Patent.
- Migrate the build system from makefile to meson.
- Introduce a new release candidate API with the associated library `libvmaf_rc`
  and executable `vmaf_rc` under `./libvmaf/build`.
- Add SI and TI feature extractor python classes.
- Add fixed-point SSIM implementation.
- Migrate to python3.

## (2019-9-8) [1.3.15]

**Fixed bugs:**

- Fix a case when CPU cores > 128(MAX_NUM_THREADS) / 3 (#319).
- Avoid dis-filtering ref when not needed, fix return type (#325).
- Update name of file for failed dis_path fopen (#334).
- A few compilation fixes (warnings and errors) (#326).
- Bump up g++ version to 9 for travis (#352).
- Use stat struct instead of ftell to retrieve the file size (#350).

**New features:**

- Write aggregate scores, exec FPS to json output.
- Add support for python3 (#332).
- Print progress in vmafossexec (#337).
- Add VMAF logo.
- Add link to report VMAF bad cases.

## (2019-3-1) [1.3.14]

**Fixed bugs:**

- Fix VMAF value mismatch on 160x90 videos after optimization (#315).
- Fix w10 error with using uninitialized offset_flag variable (#302).

**New features:**

- Add automated Windows builds with AddVeyor (#313).
- Report aggregate CI scores and fix empty model name in log (#304).

## (2019-1-31) [1.3.13]

**New features:**

- Optimized C code for speed. Running in multithreading mode, `vmafossexec`
  achieves ~40% run time reduction compared to the previous version.
- Printed out individual vmaf bootstrap scores in text file from `vmafossexec`.
- refactored windows solution (#283) (#284) (#285) (#291) (#298).

## (2018-12-17) [1.3.11]

**New features:**

- Revise number of bootstrap models definition:
  model/vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.pkl has 21 models (20 bootstrap models and
  one using the full data). From these 21 models, the 20 of them are same as
  v0.6.2, only added an additional bootstrap model.
- Output the per bootstrap model predictions from wrapper/vmafossexec.
- Print bootstrap individual scores in xml and json.
- Add BD-rate calculator and update documentation.
- Report aggregate PSNR, SSIM, and MS-SSIM scores.
- Add sklearn linear regression class to TrainTestModel.
- Enable BRISQUE feature in VMAF training with bootstrapping.
- Add --save-plot option to command line tools.
- Add ST-RREDOpt (time optimized), ST-MAD feature extractors, quality runners and
  unittestts. Refactor ST-RRED feature extractor. (#216)

**Fixed bugs:**

- Bug fixed. When start vmaf in multi-thread at the same time. (#239)
- Fix name of min function in vmaf.h and vmaf.cpp. (#227)
- Fix implicit declaration of functions (#225)

## (2018-9-13) [1.3.10]

**New features:**

- Remove sureal as a submodule to vmaf. sureal is now available through pip install.

## (2018-8-7) [1.3.9]

**Fixed bugs:**

- libvmaf: fix case where user defined read_frame() callback was being ignored.

## (2018-6-21) [1.3.8]

**Fixed bugs:**

- Fix compute_vmaf boolean type issue (#178).

## (2018-6-12) [1.3.7]

**New features:**

- Add the --ci option to calculate confidence intervals to predicted VMAF scores
  (run_vmaf, run_vmaf_in_batch, ffmpeg2vmaf, vmafossexec).
- Update libvmaf version to 1.3.7 after compute_vmaf() interface change (added
  enable_conf_interval option).
- Add new models: 1) model/vmaf_4k_v0.6.1.pkl for 4KTV viewing at distance 1.5H,
  2) model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl for VMAF prediction with a confidence
  interval, 3) model/vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.pkl for 4KTV viewing at
  distance 1.5H, with a confidence interval.

## (2018-6-4) [1.3.6]

**New features:**

- Update libvmaf version to 1.3.6 (to make consistent with VDK version from now
  on) after compute_vmaf() interface change (added thread and subsample options).
- Add the option to set the number of threads to use in vmafossexec.
- Add the option to subsample frames to save computation in vmafossexec.

## (2018-5-23) [1.3.5]

**New features:**

- Add multi-threading to vmafossexec.

## (2018-5-8) [1.3.4]

**Refactoring:**

- Refactor mos out of vmaf repo; rename to sureal as submodule.
- Refactor TrainTestModel to make predict() to output dictionary.
- Refactor TrainTestModel.
- Rename KFLK metric to AUC (Area Under the Curve) for better interpretability.

**New features:**

- Add bootstrapping to VMAF. Add two new classes BootstrapVmafQualityRunner and
  BaggingVmafQualityRunner
- Add Resolving Power Performance Metric.
- Add BRISQUE and NIQE feature extractors. Added two new classes
  BrisqueNorefFeatureExtractor and NiqeNorefFeatureExtractor. Add
  NiqeQualityRunner.

**Fixed bugs:**

- Add .gitattributes (#127). Force .pkl and .model files to retain LF line-ending.
  Required for use on Windows where model files would otherwise be checked out as
  CRLF which VMAF's parser doesn't handle.
- Allow MinGW compilation of ptools (#133). ptools doesn't build on MinGW as *nix
  socket headers are included. This patch selects Windows headers for MinGW
  builds.
- Update compute vmaf interface (#138). Update VMAF version in libvmaf.pc and etc.
  Catch logic error (resulted from wrong model file format) in compute_vmaf(). Use
  custom error code.

## (2017-12-3) [1.3.3]

**Fixed bugs:**

- Update VMAF version to 0.6.2 after compute_vmaf() interface change (#124).

## (2017-12-3) [1.3.2]

**Refactoring:**

- Lift check for exec existence during program load.
- Refactor psnr, ssim, ms_ssim and vmaf_feature to call ExternalProgramCaller.
- Refactor feature/Makefile to make executables depend on libvmaf.a.
- Refactor wrapper/Makefile to include additional objs in libvmaf.a but exclude
  main.o.
- Remove ar -d command after removing main.o from libvmaf.a.

**New features:**

- Generalize read_dataset.
- Update default Asset resampling method to bicubic (#116).
- Extend ffmpeg2vmaf script to allow ref/dis input to be YUV (#118).
- Improve README.md (#121).

**Fixed bugs:**

- Temporary fix Visual Studio builds (#112).
- Avoid unnecessary dependency on matplotlib in run_vmaf (#114).
- Remove unneeded dependencies in Dockerfile, fixes #115 (#117).
- MinGW support (#123).
- Change compute_vmaf() interface to return an error code instead of throw an
  error #124 (#126).

## (2017-8-12) [1.3.1]

**Refactoring:**

- Refactor NorefExecutorMixin to eliminate repeated codes.
- Refactor C code: get rid of unused double functions; uniformly use read_frame
  callback function to void repeated code;
- Add strip option to Makefile.

**New features:**

- Update Asset class: add copy functions to Asset; add ref/dis_yuv_type; deprecate
  yuv_type; add ref/dis_start_sec;
- Update subjective models: add confidence interval to subjective model
  parameters; refactor MLE model and make subclasses; add run_subj command line.
- Recommend pip, add ffmpeg2vmaf info and reorganize prerequisite installation (#88).
- Reduce sleep time in parallel_map.
- Add library interface for VMAF (#90).
- Add VisualStudio2015 support (#92).
- Add example of image dataset notyuv.
- Add pkgconfig file and changed Makefile.
- Add VmafPhoneQualityRunner class.
- Add DMOS_MLE_CO subjective model.

**Fixed bugs:**

- Update RegressionMixin to handle AUC exception for dicitonary-style dataset.
- Fix Makefile fedora libptools issue. (#98)

## (2017-4-13) [1.2.4]

**Refactoring:**

- Deprecate run_executors_in_parallel.
- Refactor NorefFeatureExtractor into NorefExecutorMixin so that it can be used
  for all executors.
- Add abstract methods to some base classes.

**New features:**

- Add ST-RRED runner (StrredQualityRunner), based on "Video Quality Assessment by
  Reduced Reference Spatio-Temporal Entropic Differencing", by R. Soundararaajan,
  A. Bovik.
- Add start/end frame support for Executor.

## (2017-3-8) [1.2.3]

**New features:**

- Refactor to replace config.ROOT with config.VmafConfig.

## (2017-3-1) [1.2.2]

**New features:**

- Generalize Result and FileSystemResultStore to allow None values.

## (2017-2-27) [1.2.1]

**Tasks:**

- Refactor to prepare for pypi packaging.

## (2017-2-20) [1.2.0]

**New features:**

- Updated VMAF model to version v0.6.1. Changes include: 1) added a custom model
  for cellular phone screen viewing; 2) trained using new dataset, covering more
  difficult content; 3) elementary metric fixes: ADM behavior at near-black
  frames, motion behavior at scene boundaries; 4) compressed quality score range
  by 20% to accommodate higher dynamic range; 5) Use MLE instead of DMOS as
  subjective model.

## (2017-1-24) [1.1.23]

**Fixed bugs:**

- Replace subprocess.call with run_process (checking return value).

## (2017-1-22) [1.1.22]

**New features:**

- Add command line ffmpeg2vmaf, which takes encoded videos as input.

## (2017-1-18) [1.1.21]

**New features:**

- Allow processing non-YUV input videos.

## (2016-12-20) [1.1.20]

**New features:**

- Add STRRED runner.

## (2016-12-19) [1.1.19]

**New features:**

- Allow specifying crop and pad parameter in dataset files.

## (2016-12-8) [1.1.18]

**Fixed bugs:**

- Replace pathos with custom function for parallel executor running.

## (2016-12-8) [1.1.17]

**Fixed bugs:**

- Fix command line run_testing issue. Add command line test cases.

## (2016-12-5) [1.1.16]

**New features:**

- Speed up VMAF convolution operation by AVX.

## (2016-11-30) [1.1.15]

**Fixed bugs:**

- Fix vmafossexec memory leakage.

## (2016-11-28) [1.1.14]

**New features:**

- Add enable_transform_score option to VmafQualityRunner, VmafossExecQualityRunner.

## (2016-11-18) [1.1.13]

**Fixed bugs:**

- Fix a bug in DatasetReader.to_aggregated_dataset_file.

## (2016-11-15) [1.1.12]

**New features:**

- Add Travis continuous integration.

## (2016-11-11) [1.1.11]

**New features:**

- Add implementation of AUC (Area Under the Curve) - quality metric evaluation
  method based on AUC. Refer to: L. Krasula, K. Fliegel, P. Le Callet, M.Klima,
  "On the accuracy of objective image and video quality models: New methodology
  for performance evaluation", QoMEX 2016.

## (2016-11-07) [1.1.10]

**New features:**

- Add options to use custom subjective models in run_vmaf_training and run_testing
  commands.

## (2016-11-02) [1.1.9]

**New features:**

- Add DatasetReader and subclasses; add SubjectiveModel and subclasses.

## (2016-10-19) [1.1.8]

**New features:**

- Add quality runners for each individual VMAF elementary metrics.

## (2016-10-14) [1.1.7]

**Fixed bugs:**

- Issue #36: SSIM and MS-SSIM sometimes get negative values.

## (2016-10-10) [1.1.6]

**New features:**

- Add Xcode project support.
- Add more pooling options (median, percx) to CLIs.

## (2016-10-8) [1.1.5]

**New features:**

- Add support for docker usage (#30).

## (2016-10-7) [1.1.4]

**Fixed bugs:**

- Issue #29: Make ptools build under Fedora.

## (2016-10-6) [1.1.3]

**New features:**

- Generalize dataset format to allow per-content YUV format.

## (2016-10-5) [1.1.2]

**Fixed bugs:**

- Make ptools work under Mac OS.
- Update SklearnRandomForestTrainTestModel test with sklearn 0.18.

## (2016-09-29) [1.1.1]

**New features:**

- Update command lines run_vmaf, run_psnr, run_vmaf_in_batch, run_cleaning_cache,
  run_vmaf_training and run_testing.

## (2016-09-28) [1.1.0]

**New features:**

- Update wrapper/vmafossexec: 1) it now takes pkl model file as input, so that
  slopes/intercepts are no longer hard-coded; 2) it now takes multiple YUV input
  formats; 3) add flag to enable/disable VMAF score clipping at 0/100; 4) allow
  customly running PSNR/SSIM/MS-SSIM; 5) allow customly outputing XML/JSON
- Add SSIM/MS-SSIM option in run_testing.

## (2016-09-09) [1.0.9]

**Fixed bugs:**

- Move VmafQualityRunnerWithLocalExplainer to quality_runner_adhoc to resolve
  multiple instances of VMAF found when calling QualityRunner.find_subclass.

**New features:**

- Add custom_clip_0to1 to TrainTestModel.

## (2016-09-07) [1.0.8]

**New features:**

- Generalize read_dataset to allow specifying width, height and resampling method
  on which to calculate quality.
- Add bicubic to SUPPORTED_RESAMPLING_TYPES for Asset.
- Update Asset rule with resampling_type in **str** to avoid duplicates in data
  store.

## (2016-08-20) [1.0.7]

**New features:**

- Update VmafFeatureExtractor to 0.2.2b with scaled ADM features exposed (adm_scale0-3).

## (2016-08-20) [1.0.6]

**New features:**

- Add DisYUVRawVideoExtractor and related classes.
- Add NeuralNetworkTrainTestModel base class that integrates TensorFlow.
- Add example class ToddNoiseClassifierTrainTestModel.

## (2016-08-20) [1.0.5]

**New features:**

- Add LocalExplainer class.
- Add show_local_explanation option to run_vmaf script.

## (2016-07-21) [1.0.4]

**Fixed bugs:**

- Fix a series of numerical issues in VMAF features, increment
  VmafFeatureExtractor version number.
- Retrain VmafQualityRunner after feature update, increment version number.

## (2016-07-20) [1.0.3]

**New features:**

- Add base class NorefFeatureExtractor for any feature extractor that do not
  use a reference video.
- Add MomentNorefFeatureExtractor subclassing NorefFeatureExtractor as an example
  implementation.

## (2016-06-16) [1.0.2]

**New features:**

- Refactor feature code to expose ssim/ms-ssim, speed up ssim/ms-ssim.

## (2016-06-10) [1.0.1]

**Fixed bugs:**

- Fix feature while looping by moving feof to after read_image.
- Fix issue #2 use hashed string for log filename and result filename to avoid
  file names getting too long.

**New features:**

- Add SsimFeatureExtractor and MsSsimFeatureExtractor with intermediate features
  (luminence, contrast, structure).
