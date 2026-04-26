# AGENTS.md — VMAF Fork (Lusoris)

## 🌟 GLOBAL PROJECT RULES (TOP PRIORITY)

These 2 rules apply to ALL agents, ALL tools, and ALL commits — without exception.
They override everything else in this file.

1. **NEVER modify Netflix golden-data assertions** (`python/test/` `assertAlmostEqual` values).
   They are the numerical-correctness ground truth. If scores drift, fix the code — not the assertions.
2. **EVERY user-discoverable surface gets human-readable documentation in the same PR** as the code.
   No docs = unmergeable PR. ADRs and code comments are not substitutes.

Orientation for any coding agent (Cursor, Copilot, Aider, Continue, Cody, Codeium, etc.)
opened in this repo. For Claude Code–specific tooling (skills, hooks), see
[CLAUDE.md](CLAUDE.md) — the same operational content with Claude-specific commands.

## 1. What this repo is

- Fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) — perceptual video quality
  assessment.
- Additions over upstream: SYCL / CUDA / HIP GPU backends, AVX2/AVX-512/NEON SIMD,
  a full-precision CLI flag (default `%.17g`), tiny-AI surface (ONNX Runtime), MCP server.
- License: BSD-3-Clause-Plus-Patent (upstream license preserved — see [LICENSE](LICENSE)).
- Default branch: `master`. Upstream tracked as remote `upstream`.

## 2. Build

Meson + Ninja.

```
meson setup build [-Denable_cuda=true|false] [-Denable_sycl=true|false]
ninja -C build
```

## 3. Test

```
meson test -C build                     # unit tests
meson test -C build --suite=fast        # fast subset
make test                               # full + ASan + UBSan
make test-netflix-golden                # Netflix CPU golden-data gate (see §8)
```

## 4. Lint / format

```
make lint          # clang-tidy + cppcheck + iwyu + ruff + semgrep
make format        # clang-format + black + isort (writes)
make format-check  # dry-run (CI / pre-commit)
```

## 5. Repository layout

See [CLAUDE.md §5](CLAUDE.md) — identical. Briefly:

- `libvmaf/src/` — C engine
- `libvmaf/src/{cuda,sycl,dnn}/` — GPU / DNN backends
- `libvmaf/src/feature/{x86,arm64,cuda,sycl}/` — per-platform feature implementations
- `libvmaf/tools/` — `vmaf` CLI + `vmaf_bench`
- `python/vmaf/` + `python/test/` — Python bindings + tests (golden-data here)
- `ai/` — PyTorch tiny-model training
- `mcp-server/` — MCP JSON-RPC server
- `model/` — VMAF models (.json / .pkl / .onnx)
- `testdata/` — fork-added YUV + snapshot JSONs
- `docs/principles.md` — canonical engineering standards

## 6. Coding standards

All C code must conform to:

- NASA/JPL Power of 10 (enforced by `.clang-tidy`)
- JPL Institutional Coding Standard for C (applicable subset)
- SEI CERT C & CERT C++ (mandatory)
- MISRA C:2012 (informative subset)

Banned functions, pointer/loop/alloc restrictions, and the exact list of `.clang-tidy`
checks that codify them are in [docs/principles.md](docs/principles.md).

Style: K&R, 4-space indent, 100-char line budget (see `.clang-format`).

## 7. Conventional entry points for common tasks

These operational workflows are each codified as a Claude skill under `.claude/skills/`.
Agents without slash-command routing should read the corresponding `SKILL.md` and follow
the steps manually.

| Task                         | See                                               |
|------------------------------|---------------------------------------------------|
| Build against a backend      | `.claude/skills/build-vmaf/SKILL.md`              |
| Add new GPU backend          | `.claude/skills/add-gpu-backend/SKILL.md`         |
| Add SIMD path                | `.claude/skills/add-simd-path/SKILL.md`           |
| Add feature extractor        | `.claude/skills/add-feature-extractor/SKILL.md`   |
| Add a model                  | `.claude/skills/add-model/SKILL.md`               |
| Cross-backend numeric diff   | `.claude/skills/cross-backend-diff/SKILL.md`      |
| Profile a hot path           | `.claude/skills/profile-hotpath/SKILL.md`         |
| Bisect a regression          | `.claude/skills/bisect-regression/SKILL.md`       |
| Port upstream commit         | `.claude/skills/port-upstream-commit/SKILL.md`    |
| Sync with upstream master    | `.claude/skills/sync-upstream/SKILL.md`           |
| Regenerate test snapshots    | `.claude/skills/regen-snapshots/SKILL.md`         |
| Release dry-run              | `.claude/skills/prep-release/SKILL.md`            |

## 8. Netflix golden-data gate — never modify

The fork preserves three Netflix-authored CPU reference test pairs as the
numerical-correctness ground truth:

1. **Normal** — `src01_hrc00_576x324.yuv` ↔ `src01_hrc01_576x324.yuv`
2. **Checkerboard 1-px** — `checkerboard_1920_1080_10_3_0_0.yuv` ↔ `..._1_0.yuv`
3. **Checkerboard 10-px** — `checkerboard_1920_1080_10_3_0_0.yuv` ↔ `..._10_0.yuv`

YUV files: `python/test/resource/yuv/`. Golden-score assertions are hardcoded as
`assertAlmostEqual(...)` calls in `python/test/`. **These assertions are never modified
by any PR.** They run in CI as a required status check. Fork-added tests live in separate
files and directories.

## 9. Snapshot regeneration

`testdata/scores_cpu_*.json` and `testdata/netflix_benchmark_results.json` are
fork-added GPU/SIMD snapshots — NOT Netflix golden data. If an intentional numerical
change is needed, regenerate them and include the justification in the commit message.
Non-justified changes in these files will be rejected in review.

## 10. Upstream sync

`git remote add upstream https://github.com/Netflix/vmaf.git` (once), then follow
`.claude/skills/sync-upstream/SKILL.md` to open a sync PR. Individual commits go
through `port-upstream-commit`.

## 11. Release

Release-please is triggered by pushes to `master`. Version scheme `v3.x.y-lusoris.N`,
tracking upstream version + a fork suffix. Signing is keyless via Sigstore / GitHub OIDC.

## 12. Hard rules

1. Never modify Netflix golden-score assertions (§8).
2. Never `git push --force` to `master`.
3. Never commit directly to `master` — PR with squash or fast-forward only.
4. Never merge without `make lint` + `make test` green locally.
5. Every commit message is Conventional Commits (`type(scope): subject`) — enforced by
   the `commit-msg` git hook.
6. Every new `.c` / `.h` / `.cpp` / `.cu` file starts with the applicable license
   header (wholly-new fork files: `Copyright 2026 Lusoris and Claude (Anthropic)`;
   files touching Netflix code: Netflix header preserved).
7. Every PR that adds or changes a user-discoverable surface ships
   **human-readable documentation** under `docs/` in the same PR as the code.
   User-discoverable means: CLI flags or binaries, public C API under
   `libvmaf/include/`, feature extractors, GPU backends / SIMD paths,
   `meson_options.txt` build flags, ffmpeg filter options, MCP tools, tiny-AI
   surfaces, and user-visible log / error / output-schema changes. Docs land
   in the existing topic tree (CLI → `docs/usage/`, C API → `docs/api/`,
   extractors → `docs/metrics/`, backends → `docs/backends/`, build/release
   → `docs/development/`, tiny-AI → `docs/ai/`, MCP → `docs/mcp/`,
   architecture → `docs/architecture/`). The minimum bar is *per-surface*
   (see [ADR-0100](docs/adr/0100-project-wide-doc-substance-rule.md)
   §Per-surface minimum bars). Tiny-AI keeps the tighter 5-point bar in
   [ADR-0042](docs/adr/0042-tinyai-docs-required-per-pr.md) as the
   specialisation. Code comments and ADRs are *not substitutes* — they
   explain decisions to maintainers, not usage to humans. Internal refactors,
   bug fixes with no user-visible delta, and test-only changes are excluded.
8. Every non-trivial architectural, policy, or scope decision ships as its own
   ADR file `docs/adr/NNNN-kebab-case.md` following
   [docs/adr/0000-template.md](docs/adr/0000-template.md) **before** the
   commit that implements it lands, and adds an index row in
   [docs/adr/README.md](docs/adr/README.md). Non-trivial = another engineer
   could reasonably have chosen differently. Bug fixes and implementation
   details do not need an ADR. Cite `req` (direct user quote) or
   `Q<round>.<q>` (popup answer) in the ADR's `## References` section. See
   [ADR-0028](docs/adr/0028-adr-maintenance-rule.md).
9. Every fork-local PR ships the **six deep-dive deliverables** in the same
   PR (per [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md)):
   (a) research digest under [`docs/research/`](docs/research/) (or
   "no digest needed: trivial"); (b) decision matrix in the accompanying
   ADR's `## Alternatives considered` (or "no alternatives: only-one-way
   fix"); (c) `AGENTS.md` invariant note in the relevant package (or
   "no rebase-sensitive invariants"); (d) reproducer / smoke-test command
   in the PR description; (e) `CHANGELOG.md` "lusoris fork" entry;
   (f) entry in [`docs/rebase-notes.md`](docs/rebase-notes.md) (or
   `no rebase impact: REASON`). *Fork-local* means anything not a verbatim
   port of upstream Netflix/vmaf code; pure upstream syncs and
   `port-upstream-commit` PRs are exempt. The PR template
   ([.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md))
   carries the checklist.
10. Every PR leaves every file it touches **lint-clean** to the fork's
    strictest profile (clang-tidy + cppcheck + `make lint`), whether the
    file is fork-local or upstream-mirror. "Touches" = any hunk in the
    PR's diff against its merge base. Refactor first; `// NOLINT` is
    reserved for cases where refactoring would break a load-bearing
    invariant (ADR-0138 / ADR-0139 bit-exactness pattern,
    upstream-parity identifier the rebase story depends on). Every
    NOLINT cites inline the ADR / research digest / rebase invariant
    that forces it. Historical pre-2026-04-21 NOLINTs are scoped to
    backlog item T7-5 (one sweep-PR). See
    [ADR-0141](docs/adr/0141-touched-file-cleanup-rule.md).
11. Every PR that touches a libvmaf public surface (C-API entry
    points, public headers, CLI flags, `meson_options.txt`
    entries, or any symbol probed by the `enabled libvmaf*`
    `check_pkg_config` lines) updates the relevant
    `ffmpeg-patches/000*-*.patch` file in the **same PR**. The
    fork ships FFmpeg integration as a patch stack against
    `n8.1`; libvmaf-side surface drift breaks the patches
    silently for the next rebase. Verify with
    `for p in ffmpeg-patches/000*-*.patch; do git -C ffmpeg-8
    apply --check "$p"; done` before pushing. Pure libvmaf
    internals (kernel impls, refactors that don't change
    headers), doc-only changes, and test-only changes are
    exempt. See
    [ADR-0186](docs/adr/0186-vulkan-image-import-impl.md).

## 13. Interaction style — prefer structured popup questions

When your host agent exposes a structured-question UI (Claude Code's `AskUserQuestion`,
Cursor's choice prompt, Aider's multi-choice, etc.), **use it instead of posting a
wall of numbered questions in prose**. The user clicks through options in seconds;
prose questionnaires force them to scroll, parse, and type structured replies.

Rules of thumb:

- 2–4 focused questions per round, 2–4 concrete options each.
- Mark the recommended option `(Recommended)` when one clearly wins.
- Reserve prose for setting up the question, not for the question itself.
- Still fine to answer in prose — this rule applies to *asking*, not to reporting.
