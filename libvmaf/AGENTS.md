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
