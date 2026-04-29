# AGENTS.md — libvmaf

Orientation for any coding agent working inside `libvmaf/`. Root orientation
lives in [../AGENTS.md](../AGENTS.md); this file is the scoped hand-off for
this subtree. Claude Code equivalents in [../CLAUDE.md](../CLAUDE.md).

## Scope

The C engine — VMAF metric, feature extractors, backends, public API,
CLI (`tools/vmaf`, `tools/vmaf_bench`), and C unit tests.

```
libvmaf/
  include/libvmaf/   # public C API (libvmaf.h, dnn.h, model.h, picture.h, ...)
  src/               # engine + feature extractors + backends
    cuda/            # CUDA backend runtime (picture, dispatch, ring buffer)
    sycl/            # SYCL backend runtime (queue, USM, dmabuf import)
    dnn/             # ONNX Runtime integration (tiny AI)
    feature/         # per-feature CPU implementations
      x86/           # AVX2 / AVX-512 SIMD paths
      arm64/         # NEON SIMD paths
      cuda/          # CUDA feature kernels
      sycl/          # SYCL feature kernels
  test/              # C unit tests (µnit-style: test.h + mu_run_test)
  tools/             # vmaf CLI, vmaf_bench, cli_parse
  meson.build
  meson_options.txt
```

## Ground rules for this subtree

- **Coding standards**: NASA/JPL Power of 10 + JPL-C-STD + SEI CERT C (see
  [../docs/principles.md](../docs/principles.md)). `.clang-tidy` enforces.
- **License headers**: Netflix-header-preserving for upstream-touched files;
  `Copyright 2026 Lusoris and Claude (Anthropic)` for wholly-new files.
  See [ADR-0025](../docs/adr/0025-copyright-handling-dual-notice.md).
- **Style**: K&R, 4-space, 100-char columns, `.clang-format` authoritative.
- **Banned functions** (see `docs/principles.md §1.2 rule 30`): `gets`,
  `strcpy`, `strcat`, `sprintf`, `strtok`, `atoi`, `atof`, `rand`, `system`.
- **Every non-void return value is checked or explicitly `(void)`-discarded.**
- **Every new file starts with the license header** (Netflix preserved on
  upstream-touched; Lusoris/Claude on wholly-new — see ADR-0025).

## Workflows routed here

| Task | Skill |
| --- | --- |
| Add a feature extractor | [../.claude/skills/add-feature-extractor/SKILL.md](../.claude/skills/add-feature-extractor/SKILL.md) |
| Add a SIMD path (AVX2 / AVX-512 / NEON) | [../.claude/skills/add-simd-path/SKILL.md](../.claude/skills/add-simd-path/SKILL.md) |
| Add a GPU backend (CUDA / SYCL / HIP / Vulkan) | [../.claude/skills/add-gpu-backend/SKILL.md](../.claude/skills/add-gpu-backend/SKILL.md) |
| Register a model JSON | [../.claude/skills/add-model/SKILL.md](../.claude/skills/add-model/SKILL.md) |
| Cross-backend numeric diff | [../.claude/skills/cross-backend-diff/SKILL.md](../.claude/skills/cross-backend-diff/SKILL.md) |
| Profile a hot path | [../.claude/skills/profile-hotpath/SKILL.md](../.claude/skills/profile-hotpath/SKILL.md) |

## Governing ADRs

- [ADR-0006](../docs/adr/0006-cli-precision-17g-default.md) — CLI precision default `%.17g`, propagates to `output.c` and Python.
- [ADR-0012](../docs/adr/0012-coding-standards-jpl-cert-misra.md) — the coding-standards stack.
- [ADR-0022](../docs/adr/0022-inference-runtime-onnx.md) — execution-provider mapping ORT↔backends.
- [ADR-0024](../docs/adr/0024-netflix-golden-preserved.md) — golden-data gate (three CPU reference pairs, never modified).
- [ADR-0025](../docs/adr/0025-copyright-handling-dual-notice.md) — dual-copyright policy.
- [ADR-0137](../docs/adr/0137-thread-local-locale-for-numeric-io.md) —
  thread-local locale abstraction (`thread_locale.h`) for all numeric I/O.

## Rebase-sensitive invariants

- **Output writers return `ferror(outfile) ? -EIO : 0`.**
  `vmaf_write_output_{xml,json,csv,sub}` in
  [src/output.c](src/output.c) use a single tail `return` that
  checks `ferror(outfile)` — per [ADR-0119](../docs/adr/0119-cli-precision-default-revert.md).
  Any upstream patch that changes the tail to bare `return 0`
  must be merged so the fork's `ferror` check survives. The
  thread-locale bracket from [ADR-0137](../docs/adr/0137-thread-local-locale-for-numeric-io.md)
  is `push_c()` at entry → body → `pop()` before the `ferror`
  check; dropping the `pop()` leaks a `locale_t` on POSIX and
  leaves the calling thread locked to `"C"` on Windows.
- **Thread-pool job recycling + inline data buffer** (fork-local,
  ADR-0147): [`src/thread_pool.c`](src/thread_pool.c) recycles
  `VmafThreadPoolJob` slots via a `pool->free_jobs` free list
  (mutex-protected by `queue.lock`) and stores payloads ≤
  `JOB_INLINE_DATA_SIZE` (64 bytes) inside `job->inline_data`
  instead of a second `malloc`. The cleanup path distinguishes
  inline from heap payloads via the
  `job->data != job->inline_data` guard in
  `vmaf_thread_pool_job_clear_data`; do not collapse this
  check during a rebase — freeing `inline_data` would corrupt
  the slot. The fork's `func(void *data, void **thread_data)`
  signature and `VmafThreadPoolWorker` per-worker-data path must
  survive any upstream sync; Netflix upstream PR #1464 (closed)
  has a similar job-pool but uses the bare
  `func(void *data)` signature — on conflict keep the fork's
  two-arg signature and merge only the pool-mechanics changes.
  See [ADR-0147](../docs/adr/0147-thread-pool-job-pool.md) and
  [rebase-notes 0040](../docs/rebase-notes.md).

- **Embedded MCP scaffold contract** (fork-local, [ADR-0209](../docs/adr/0209-mcp-embedded-scaffold.md)).
  [`src/mcp/mcp.c`](src/mcp/mcp.c) is the audit-first stub TU
  for the in-process MCP server declared in
  [`include/libvmaf/libvmaf_mcp.h`](include/libvmaf/libvmaf_mcp.h).
  Every public entry point validates its arguments first
  (`-EINVAL` on NULLs / negative fds / NULL paths) **then** returns
  `-ENOSYS`. The 12-sub-test smoke at
  [`test/test_mcp_smoke.c`](test/test_mcp_smoke.c) pins the
  contract; the T5-2b runtime PR flips bodies in place and
  updates the smoke expectations in the same commit. Do not
  drop the NULL-argument validation when wiring real bodies —
  the smoke tests for `_init`, `_start_uds`, `_start_stdio`
  rely on early `-EINVAL` even after the runtime arrives. The
  `enable_mcp` umbrella flag must default `false` until all
  three transport bodies are stable; the silent-flip risk is
  the same as ADR-0175's Vulkan precedent.

- **GPU-parity matrix gate contract** (fork-local,
  [ADR-0214](../docs/adr/0214-gpu-parity-ci-gate.md)).
  [`scripts/ci/cross_backend_parity_gate.py`](../scripts/ci/cross_backend_parity_gate.py)
  is the single source of truth for the per-feature absolute
  tolerance every (CPU↔GPU, GPU↔GPU) cell must respect. The CI
  job `vulkan-parity-matrix-gate` in
  [tests-and-quality-gates.yml](../.github/workflows/tests-and-quality-gates.yml)
  runs it on every PR over CPU↔Vulkan/lavapipe; CUDA/SYCL/hardware-
  Vulkan are advisory until a self-hosted runner exists. Do not
  tighten a `FEATURE_TOLERANCE` entry without a measurement-driven
  follow-up ADR (per CLAUDE.md §12 r1). Adding a new feature with
  a GPU twin requires (1) a `FEATURE_METRICS` entry, (2) a
  `FEATURE_TOLERANCE` entry if the feature relaxes places=4, and
  (3) a row in
  [`docs/development/cross-backend-gate.md`](../docs/development/cross-backend-gate.md).

Backend-specific orientation:

- [src/cuda/AGENTS.md](src/cuda/AGENTS.md) — CUDA backend runtime
- [src/sycl/AGENTS.md](src/sycl/AGENTS.md) — SYCL backend runtime
- [src/dnn/AGENTS.md](src/dnn/AGENTS.md) — ONNX Runtime integration (tiny AI)
- [src/feature/AGENTS.md](src/feature/AGENTS.md) — feature extractors + SIMD
- [test/AGENTS.md](test/AGENTS.md) — C unit tests

## Build

```bash
meson setup build [-Denable_cuda=true|false] [-Denable_sycl=true|false] [-Denable_dnn=auto]
ninja -C build
meson test -C build
```

Shortcut: `/build-vmaf --backend=cpu|cuda|sycl|all`.

## Backend-engagement foot-guns (read before benching)

Two CLI flags govern backend selection at runtime; the relationship is
**not** "set the flag for the backend you want". A run that looks like
it's exercising CUDA can silently fall through to CPU and still produce
the expected score (because CUDA extractors emit the same logical
features). Symptoms reviewers see: bit-exact CPU/CUDA/SYCL pools,
identical fps across backends — **always wrong on a non-trivial fixture
size unless the flags are right.**

- **`--gpumask` is a CUDA *disable* bitmask, not a device pin.**
  `compute_fex_flags` ([`src/libvmaf.c::compute_fex_flags`](src/libvmaf.c))
  enables the CUDA dispatch slot only when `gpumask == 0`. Any
  nonzero value disables CUDA. Public-header semantics:
  `if gpumask: disable CUDA` (see
  [`include/libvmaf/libvmaf.h`](include/libvmaf/libvmaf.h) `VmafConfiguration::gpumask`).
- **`--backend cuda` currently *initialises* CUDA but disables the
  dispatch slot.** The CLI sets `gpumask = 1` ([`tools/cli_parse.c::parse_cli_args`](tools/cli_parse.c))
  intending it as a device pin, but the runtime treats `gpumask = 1`
  as "disable CUDA". So `--backend cuda` runs CUDA init and then
  routes the actual feature extractors through the CPU path. The
  vmaf_v0.6.1-style models then emit identical scores because the
  CPU code is doing the work. **This is a bug** (tracked separately;
  do not assume `--backend cuda` works for benching until it lands).
- **`--no_cuda` / `--no_sycl` are *disable*-only.** Pairing
  `--no_sycl` alone (without `--gpumask`) does NOT enable CUDA — it
  just disables SYCL while leaving CUDA unrequested. The CLI inits
  CUDA only when `c.use_gpumask && !c.no_cuda` (see
  [`tools/vmaf.c`](tools/vmaf.c) device-init block).

**Correct invocations for backend bench / cross-backend diff:**

| intent | flags |
|---|---|
| CPU only | `--no_cuda --no_sycl` |
| CUDA | `--gpumask=0 --no_sycl` |
| SYCL | `--sycl_device=0 --no_cuda` |
| Vulkan | `--vulkan_device=N` (no `--no_cuda`/`--no_sycl` interaction) |

Verify CUDA actually engaged by inspecting the JSON `frames[0].metrics`
key set: CPU emits 14–15 keys (`integer_aim`, `integer_motion3`,
`integer_adm3` are CPU-only); CUDA emits 11–12 keys (the CPU-only
extras absent). Same-key-count + identical pool across two backends =
both ran the same code path.

The bench script `testdata/bench_all.sh` historically used the wrong
flag pattern (`--no_sycl` for "CUDA"). Numbers from runs older than
2026-04-28 in `docs/benchmarks.md` were CPU-on-CPU comparisons. See
[ADR-0064 in rebase-notes](../docs/rebase-notes.md) and PR #169 for
the corrected methodology.
