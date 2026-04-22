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
| [ADR-0006](0006-cli-precision-17g-default.md) | Set CLI precision default to `%.17g` with `--precision` flag | Superseded by [ADR-0119](0119-cli-precision-default-revert.md) | cli, testing, python |
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
| [ADR-0025](0025-copyright-handling-dual-notice.md) | Copyright handling preserves Netflix, adds Lusoris/Claude | Superseded by [ADR-0105](0105-copyright-handling-dual-notice.md) | license, docs |
| [ADR-0026](0026-workspace-relocated-under-python.md) | Relocate Python harness workspace under `python/vmaf/` | Accepted | workspace, python, docs |
| [ADR-0027](0027-non-conservative-image-pins.md) | Non-conservative image pins + experimental toolchain flags | Accepted | ci, cuda, sycl, build, supply-chain |
| [ADR-0028](0028-adr-maintenance-rule.md) | Every non-trivial decision gets an ADR before the commit | Superseded by [ADR-0106](0106-adr-maintenance-rule.md) | docs, planning, agents |
| [ADR-0029](0029-resource-tree-relocated.md) | Relocate resource tree under `python/vmaf/` | Accepted | workspace, python, docs |
| [ADR-0030](0030-matlab-sources-relocated.md) | Relocate MATLAB sources under `python/vmaf/` | Accepted | workspace, matlab, python |
| [ADR-0031](0031-fork-docs-moved-under-docs.md) | Fork-added docs live under `docs/` | Accepted | docs, workspace |
| [ADR-0032](0032-unittest-script-moved-to-scripts.md) | Relocate root `unittest` script to `scripts/` | Accepted | testing, workspace |
| [ADR-0033](0033-codeql-config-moved-to-github.md) | Relocate CodeQL config to `.github/` | Accepted | security, ci, github |
| [ADR-0034](0034-single-patches-directory.md) | Delete `patches/` leftover; keep only `ffmpeg-patches/` | Accepted | workspace, build |
| [ADR-0035](0035-claude-hooks-schema-fix.md) | Migrate `.claude/settings.json` hooks to current schema | Accepted | claude, agents |
| [ADR-0036](0036-tinyai-wave1-scope-expansion.md) | Tiny-AI Wave 1 scope expanded beyond D20–D23 | Superseded by [ADR-0107](0107-tinyai-wave1-scope-expansion.md) | ai, dnn, cli, framework, mcp |
| [ADR-0037](0037-master-branch-protection.md) | Protect `master` branch on GitHub with required checks | Accepted | github, ci, security, release |
| [ADR-0038](0038-purge-upstream-matlab-mex-binaries.md) | Purge upstream MATLAB MEX compiled binaries from tree | Accepted | security, matlab, supply-chain |
| [ADR-0039](0039-onnx-runtime-op-walk-registry.md) | Pull forward runtime op-allowlist walk + model registry | Accepted | ai, dnn, security, supply-chain |
| [ADR-0040](0040-dnn-session-multi-input-api.md) | Extend DNN session API to multi-input/output with named bindings | Accepted | ai, dnn, cli |
| [ADR-0041](0041-lpips-sq-extractor.md) | Ship LPIPS-SqueezeNet FR extractor with inverse-ImageNet in graph | Accepted | ai, dnn, cli |
| [ADR-0042](0042-tinyai-docs-required-per-pr.md) | Tiny-AI PRs must ship human-readable docs in the same PR | Accepted | ai, dnn, docs |
| [ADR-0100](0100-project-wide-doc-substance-rule.md) | Every user-discoverable change ships docs in the same PR (project-wide) | Accepted | docs, agents, framework |
| [ADR-0101](0101-sycl-usm-picture-pool.md) | Implement USM-backed picture pre-allocation pool for SYCL | Accepted | sycl, gpu, picture-api, memory |
| [ADR-0102](0102-dnn-ep-selection-and-fp16-io.md) | DNN EP selection is ordered + graceful; `fp16_io` does a host-side fp32↔fp16 cast | Accepted | ai, dnn, cli |
| [ADR-0103](0103-sycl-d3d11-surface-import.md) | Implement `vmaf_sycl_import_d3d11_surface` as staging-texture H2D path | Accepted | sycl, windows, api |
| [ADR-0104](0104-picture-pool-always-on.md) | Compile `picture_pool` unconditionally and size it for the live-picture set | Accepted | api, build, cli |
| [ADR-0105](0105-copyright-handling-dual-notice.md) | Copyright handling preserves Netflix and adds Lusoris/Claude (paraphrased re-statement) | Supersedes [ADR-0025](0025-copyright-handling-dual-notice.md) | license, docs |
| [ADR-0106](0106-adr-maintenance-rule.md) | Every non-trivial decision gets its own ADR file before the commit (paraphrased re-statement) | Supersedes [ADR-0028](0028-adr-maintenance-rule.md) | docs, planning, agents |
| [ADR-0107](0107-tinyai-wave1-scope-expansion.md) | Tiny-AI Wave 1 scope expanded beyond ADR-0020 through ADR-0023 (paraphrased re-statement) | Supersedes [ADR-0036](0036-tinyai-wave1-scope-expansion.md) | ai, dnn, cli, framework, mcp |
| [ADR-0108](0108-deep-dive-deliverables-rule.md) | Every fork-local PR ships the six deep-dive deliverables (research digest, decision matrix, AGENTS.md invariant, reproducer, changelog entry, rebase note) | Accepted | docs, agents, framework, planning |
| [ADR-0109](0109-nightly-bisect-model-quality.md) | Nightly `bisect-model-quality` runs against a synthetic placeholder cache (real DMOS-aligned cache swaps in via follow-up) | Accepted | ai, ci, tiny-ai, framework |
| [ADR-0110](0110-coverage-gate-fprofile-update-atomic.md) | Coverage gate uses `-fprofile-update=atomic` to survive parallel meson tests on instrumented SIMD code | Superseded by ADR-0111 | ci, build, simd, testing |
| [ADR-0111](0111-coverage-gate-gcovr-with-ort.md) | Coverage gate switches `lcov` → `gcovr` and installs ORT in the coverage job (fixes 1176% over-count + DNN-stub coverage gap; layers on ADR-0110 race fixes) | Accepted | ci, build, dnn, testing |
| [ADR-0112](0112-ort-backend-testability-surface.md) | Expose `ort_backend.c` static helpers (fp16 conversion, resolve_name) via private internal header so `test_ort_internals` can unit-test edge branches the public API can't reach on a CPU-only ORT build | Accepted | dnn, testing, coverage |
| [ADR-0113](0113-ort-create-session-fallback-multi-ep-ci.md) | `vmaf_ort_open` falls back to CPU when `CreateSession` fails after a non-CPU EP attached; coverage CI installs `onnxruntime-gpu` + `libcudart12` to exercise EP-attach success arms | Accepted | dnn, ci, coverage, ort |
| [ADR-0114](0114-coverage-gate-per-file-overrides.md) | `coverage-check.sh` gains a per-file critical-coverage override map; `dnn/ort_backend.c` + `dnn/dnn_api.c` floor at 78% (structural EP-availability ceiling per ADR-0112) | Accepted | ci, coverage, dnn, ort, gate |
| [ADR-0115](0115-ci-trigger-master-only-and-matrix-consolidation.md) | All CI workflows trigger on `[master]` only (drop dead `sycl` branch); delete `windows.yml` and merge into `libvmaf.yml` preserving the `build (MINGW64, …)` required-status-check name | Accepted | ci, github, build, framework |
| [ADR-0116](0116-ci-workflow-naming-convention.md) | CI workflow naming convention — purpose-named files + Title Case display names | Accepted | ci, github, docs |
| [ADR-0117](0117-coverage-gate-warning-noise-suppression.md) | Bump `actions/upload-artifact@v5`→`@v7` (Node 24) repo-wide; filter gcovr `Ignoring suspicious hits` stderr noise so the Coverage Gate Annotations panel finishes empty | Accepted | ci, coverage, gcovr, github-actions |
| [ADR-0118](0118-ffmpeg-patch-series-application.md) | `ffmpeg-patches/` is a quilt-style series applied via `series.txt` ordering by both Dockerfile and `ffmpeg.yml`; patches regenerated via real `git format-patch -3` carrying valid index lines + signed-off-by trail | Accepted | ci, build, ffmpeg, docker, sycl, ai |
| [ADR-0119](0119-cli-precision-default-revert.md) | Revert CLI precision default from `%.17g` to `%.6f` so the Netflix CPU golden gate (CLAUDE.md §8) passes without per-call-site flags; `--precision=max` keeps the round-trip-lossless opt-in | Supersedes [ADR-0006](0006-cli-precision-17g-default.md) | cli, testing, python, golden-gate |
| [ADR-0120](0120-ai-enabled-ci-matrix-legs.md) | Add three DNN-enabled matrix legs (Ubuntu gcc, Ubuntu clang, macOS clang) to `libvmaf-build-matrix.yml` so the ORT C-API surface and `dnn` meson suite are exercised across compilers/OSes; macOS leg `experimental: true` (Homebrew ORT floats) | Accepted | ci, ai, dnn, ort, build, github-actions |
| [ADR-0121](0121-windows-gpu-build-only-legs.md) | Add Windows GPU build-only matrix legs (`Build — Windows MSVC + CUDA (build only)` + `Build — Windows MSVC + oneAPI SYCL (build only)`) so MSVC build-portability of the CUDA / SYCL backends is gated on PR, not from downstream user reports | Accepted | ci, build, cuda, sycl, github-actions |
| [ADR-0122](0122-cuda-gencode-coverage-and-init-hardening.md) | Unconditional CUDA cubin coverage for `sm_86` / `sm_89` + `compute_80` PTX fallback in `libvmaf/src/meson.build`; actionable multi-line `libcuda.so.1` dlopen-failure + `cuInit` messages in `vmaf_cuda_state_init()` (with pre-existing leak fix on error paths) | Accepted | cuda, build, docs |
| [ADR-0123](0123-cuda-post-cubin-load-regression-32b115df.md) | Fix ffmpeg `libvmaf_cuda` null-deref at `vmaf_read_pictures` tail: `prev_ref` update on CUDA-device-only extractor set dereferenced zero-initialised `ref_host`. Null-guard the `vmaf_picture_ref(&vmaf->prev_ref, ref)` call. Upstream `f740276a` + `32b115df` + fork `65460e3a` combined to reach default builds. | Accepted | cuda, regression, upstream-sync |
| [ADR-0124](0124-automated-rule-enforcement.md) | Automate the four rule-bearing process ADRs (0100 doc-substance, 0105 copyright, 0106 ADR-per-decision, 0108 deep-dive deliverables). New `rule-enforcement.yml` workflow with one blocking job (`deep-dive-checklist`) + two advisory PR-comment jobs (`doc-substance-check`, `adr-backfill-check`); pre-commit hook for the copyright template. | Accepted | ci, agents, framework, docs, license |
| [ADR-0125](0125-ms-ssim-decimate-simd.md) | MS-SSIM decimate fast paths: AVX2 + AVX-512 specialised 9×9 separable LPF factor-2 kernels under `libvmaf/src/feature/x86/`; vendored `iqa/decimate.c` stays untouched; bit-exactness enforced via a scalar-separable reference; NEON deferred to follow-up | Proposed | simd, testing, agents |
| [ADR-0126](0126-ssimulacra2-extractor.md) | SSIMULACRA 2 perceptual metric as a fork-local feature extractor — port libjxl C++ reference to `libvmaf/src/feature/ssimulacra2.c`, scalar-first, float-tolerance bit-closeness | Proposed | metrics, feature-extractor, docs, agents |
| [ADR-0127](0127-vulkan-compute-backend.md) | Vulkan compute backend — vendor-neutral GPU path alongside CUDA/SYCL/HIP; volk loader, GLSL→SPIR-V via glslc, VMA allocator, VIF as pathfinder | Proposed | gpu, vulkan, backend, build, agents |
| [ADR-0128](0128-embedded-mcp-in-libvmaf.md) | Embedded MCP server inside libvmaf — SSE + UDS + stdio transports, build-flag-gated, new `libvmaf_mcp.h` header, Power-of-10 compliant via dedicated MCP thread + SPSC queue | Proposed | mcp, agents, api, build, docs |
| [ADR-0129](0129-tinyai-ptq-quantization.md) | Tiny-AI post-training int8 quantisation — static + dynamic + QAT per model via `quant_mode` field in `model/registry.json`; three scripts under `ai/scripts/`; CI accuracy-budget gate | Proposed | ai, onnx, quantization, model, docs |
| [ADR-0130](0130-ssimulacra2-scalar-implementation.md) | Ship scalar C port of the SSIMULACRA 2 metric (libjxl tools/ssimulacra2.cc). BT.709 limited-range YUV→sRGB→linear→XYB pipeline, 6-scale pyramid with separable Gaussian blur (σ=1.5, reflect padding) replacing libjxl's FastGaussian IIR, 108-weight polynomial pool. Snapshot JSON deferred to follow-up PR. Implementation closeout for ADR-0126 (Proposed, PR #67). | Accepted | metrics, feature-extractor, ssimulacra2 |
| [ADR-0131](0131-port-netflix-1382-cumemfree.md) | Port Netflix#1382 — swap `cuMemFreeAsync(ptr, priv->cuda.str)` for synchronous `cuMemFree(ptr)` at `libvmaf/src/cuda/picture_cuda.c:247` to fix the multi-session assert-0 crash reported in Netflix#1381. Preceding `cuStreamSynchronize` already removed the async-overlap benefit, so no perf regression. | Accepted | cuda, upstream-port, correctness |
| [ADR-0132](0132-port-netflix-1406-feature-collector-model-list.md) | Port Netflix#1406 — fix `vmaf_feature_collector_mount_model` list traversal bug (dereferenced-pointer mutation corrupted the list on ≥3 mounts) and `unmount_model` error-code semantics (`-EINVAL` → `-ENOENT` on not-found). Refactored the upstream test extension into a shared `load_three_test_models` helper to keep per-test bodies under JPL-Power-of-10 rule-4 size thresholds. | Accepted | upstream-port, correctness, testing |
| [ADR-0133](0133-ci-clang-tidy-push-delta.md) | Unify push-event and pull-request-event clang-tidy scoping: both now scan only the event's delta (`<before>..HEAD` on push, `origin/<base>...HEAD` on PR). Master-post-merge no longer re-scans every vendored file (`libvmaf/src/svm.cpp`, `libvmaf/src/cuda/*.c` without CUDA headers) and surfaces long-latent warnings unrelated to the push. `actions/checkout@v6` bumped to `fetch-depth: 0`. | Accepted | ci, lint, clang-tidy |
| [ADR-0134](0134-port-netflix-1451-meson-declare-dependency.md) | Port Netflix#1451 — append `declare_dependency(link_with: libvmaf, include_directories: [libvmaf_inc])` + `meson.override_dependency('libvmaf', ...)` at the end of `libvmaf/src/meson.build` so consumers can use the fork as a meson subproject with the standard `dependency('libvmaf')` idiom. Fork deviates from upstream by using trailing-comma style to match fork build-file conventions. | Accepted | build, upstream-port |
| [ADR-0135](0135-port-netflix-1424-expose-builtin-model-versions.md) | Port Netflix#1424 — expose `vmaf_model_version_next(prev, &version)` public iterator for built-in VMAF model versions. Corrects three upstream defects during port: NULL-pointer arithmetic UB (missing `else`), off-by-one returning the `{0}` sentinel, and const-qualifier mismatches in the test. Adds `BUILT_IN_MODEL_CNT == 0` early-return for zero-models build configurations. Doxygen-style header doc replaces upstream's one-liner. | Accepted | api, upstream-port, correctness |
| [ADR-0136](0136-ci-deliverables-checker-strip-markdown.md) | Strip markdown emphasis/code characters (`` ` ``, `*`, `_`) from the PR body before the `Deep-Dive Deliverables Checklist` grep. The template ships label bullets like ``- [ ] **`AGENTS.md` invariant note**`` — backticks inserted characters between tokens and broke the literal-item regex, rejecting conforming PRs. One-line `tr -d` pass applied to both parse and diff-verification steps. | Accepted | ci, rule-enforcement, adr-0108 |
| [ADR-0137](0137-thread-local-locale-for-numeric-io.md) | Port upstream Netflix/vmaf PR #1430 (thread-local locale handling via `thread_locale.h/.c`) into the fork. Replaces process-global `setlocale(LC_ALL, "C")` bracket in `svm_save_model` + output writers with POSIX.1-2008 `uselocale` (Linux/macOS/BSD), Windows `_configthreadlocale`, graceful fallback elsewhere. Fork corrections: `NULL` score_format in test, merge ADR-0119 `ferror` return contract with upstream pop, drop dead `<locale.h>` include in svm.cpp. | Accepted | port, libvmaf, i18n, thread-safety, upstream-port |
| [ADR-0138](0138-iqa-convolve-avx2-bitexact-double.md) | Add AVX2 + AVX-512 bit-exact fast path for `_iqa_convolve` using the three-stage pattern *single-rounded float mul → widen to double → double add* (no FMA) to mirror scalar `sum += img[i]*k[j]` under FLT_EVAL_METHOD == 0. Specialised for MS-SSIM / SSIM invariants (11-tap Gaussian or 8-tap square, normalised, separable). Dispatch in `ssim_tools.c` via new `_iqa_convolve_set_dispatch`; vendored `iqa/convolve.c` untouched. Profile-driven after the 51% self-time hot spot revealed post-decimate-SIMD. Bit-identical to scalar; Netflix golden gate unchanged. | Proposed | simd, performance |
| [ADR-0139](0139-ssim-simd-bitexact-double.md) | Rewrite fork-local `ssim_accumulate_avx2/avx512` to match scalar `ssim_accumulate_default_scalar` byte-for-byte by doing the `2.0 * mu1*mu2 + C1` numerator and the final `l*c*s` product per-lane in scalar **double** rather than vector float. Prior SIMD computed l, c in float and multiplied as float → 8th-decimal (~0.13 float-ULP) drift on MS-SSIM. `precompute` / `variance` are pure elementwise float ops and already bit-exact. Verified on Netflix + checkerboard pairs: scalar = AVX2 = AVX-512 bit-identical at `--precision max`. | Proposed | simd, performance, bit-exact |
| [ADR-0140](0140-simd-dx-framework.md) | Ship a two-part SIMD DX framework — `libvmaf/src/feature/simd_dx.h` with ISA-specific macros for the ADR-0138 widen-then-add pattern and the ADR-0139 per-lane-scalar-double reduction pattern, plus an upgraded `/add-simd-path` skill that scaffolds TU + header + unit test + meson rows from a kernel spec. Demonstrated on convolve NEON (ADR-0138 port to aarch64) and ssim NEON bit-exactness audit (research-0012 follow-up). Cross-compile + QEMU audit gate verifies NEON = scalar at `--precision max`. Enables PR #B (ssimulacra2 AVX2+NEON, motion_v2 NEON, vif_statistic AVX-512+NEON, etc.) at higher throughput. | Proposed | simd, dx, build, agents |
| [ADR-0141](0141-touched-file-cleanup-rule.md) | Every PR leaves every file it touches lint-clean to the fork's strictest profile — fork-local AND upstream-mirror. Refactor first; `// NOLINT` reserved for cases where refactoring would break a load-bearing invariant (ADR-0138 / ADR-0139 bit-exactness, upstream-parity identifier the rebase story depends on). Every NOLINT cites inline the ADR / research digest / rebase invariant that forces it. Historical debt (18 pre-existing `readability-function-size` NOLINTs + upstream `_iqa_*` suppressions) stays scoped to backlog item T7-5; this ADR does not backdate the rule. | Accepted | ci, process, code-quality, agents |
