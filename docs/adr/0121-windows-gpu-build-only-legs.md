# ADR-0121: Windows GPU build-only matrix legs (MSVC + CUDA, MSVC + oneAPI SYCL)

- **Status**: Accepted
- **Date**: 2026-04-19
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: ci, build, cuda, sycl, github-actions

## Context

The fork's GPU backends (CUDA + SYCL) are exercised in CI **only on
Linux**. The existing GPU matrix entries in
[libvmaf-build-matrix.yml](../../.github/workflows/libvmaf-build-matrix.yml)
all run on `ubuntu-latest` with gcc as the C/C++ host compiler:

- `Build — Ubuntu CUDA` (gcc + nvcc)
- `Build — Ubuntu SYCL` (gcc host + icpx for SYCL .cpp)
- `Build — Ubuntu SYCL + CUDA`
- `Build — Ubuntu CUDA Static`, `Build — Ubuntu SYCL Static`

The single existing Windows job is `Build — Windows MinGW64 (CPU)` —
MSYS2 / MinGW-w64 with `-Denable_cuda=false -Denable_sycl=false`. So
**no** CI leg currently verifies that the CUDA host code or the SYCL
.cpp sources even *compile* against the MSVC ABI.

This is a real coverage hole:

1. Windows is a tier-1 platform for the libvmaf binary distribution
   (vmaf.exe artifact is uploaded by the MinGW job and consumed by
   downstream users).
2. The CUDA backend's host code (`libvmaf/src/cuda/*.c`) uses POSIX
   patterns (e.g. `pthread_*` aliases, file descriptors for dma-buf
   import paths) that need conditional `#ifdef _WIN32` guards. A
   Linux-only CI cannot catch a regression that breaks the MSVC build.
3. The SYCL backend's `vmaf_sycl_*` C-API entry points
   (`libvmaf/src/sycl/`) similarly need to compile cleanly under
   `icx-cl` (the Windows DPC++ driver), which has subtly different
   `__declspec(dllexport)` and CRT-linkage requirements than the
   Linux `icpx` driver.
4. Downstream Windows users who try `meson setup -Denable_cuda=true`
   on Windows currently hit unguarded build breakage that we discover
   only via downstream issue reports.

The user explicitly scoped this PR as **"Add both CUDA + SYCL Windows
build-only legs"** (see References) — the most ambitious of the
three options I offered. "Build-only" because `windows-latest` GitHub
runners have no GPU hardware, so executing GPU kernels is not
possible; what we *can* do is verify compile + link.

## Decision

Add a new top-level `windows-gpu-build` job in
[libvmaf-build-matrix.yml](../../.github/workflows/libvmaf-build-matrix.yml)
running on `windows-latest`, with two matrix entries:

| Display name | Backend | Toolchain | Build-only? |
| --- | --- | --- | --- |
| `Build — Windows MSVC + CUDA (build only)` | CUDA | MSVC + nvcc 13.0.0 | yes |
| `Build — Windows MSVC + oneAPI SYCL (build only)` | SYCL | MSVC + icx-cl 2025.3 | yes |

Both legs:

- Use [`ilammy/msvc-dev-cmd@v1`](https://github.com/ilammy/msvc-dev-cmd)
  to set up the MSVC dev environment (community-standard action that
  runs `vcvars64.bat` and exports the env vars). Required because
  nvcc shells out to `cl.exe` and `icx-cl` is a `cl.exe`-compatible
  driver.
- Pin CUDA tooling to the exact same version as the Linux CUDA leg:
  CUDA 13.0.0 (via [`Jimver/cuda-toolkit@v0.2.35`](https://github.com/Jimver/cuda-toolkit)).
  Identical pins keep the MSVC-vs-Linux comparison meaningful — if
  a build breaks here but not on Linux, it's an MSVC ABI issue, not
  a tooling-version delta.
- Install oneAPI on Windows via the official **`oneapi-src/oneapi-ci`
  offline-installer pattern**: `curl` the Intel Base Toolkit
  Windows installer, extract the bootstrapper, run with
  `--components=intel.oneapi.win.cpp-dpcpp-common --eula=accept`
  plus `NEED_VS{2017,2019,2022}_INTEGRATION=0` to skip the (slow)
  Visual Studio plug-in steps. Pinned to BaseKit 2025.3.0.372 to
  match the Linux SYCL leg's `intel-oneapi-compiler-dpcpp-cpp-2025.3`.
  Two earlier candidates failed: (1)
  [`rscohn2/setup-oneapi@v0`](https://github.com/rscohn2/setup-oneapi)
  is **Linux-only** — every installer URL in its `src/main.js`
  ends in `.sh`; (2) Chocolatey's `intel-oneapi-basekit` package
  is **not on the community feed** (verified 2026-04-19: lookup
  fast-failed in 3 s with `package was not found with the
  source(s) listed`). The Intel offline installer is what Intel
  themselves use in CI
  (`oneapi-src/oneapi-ci/scripts/install_windows.bat`), so it
  gives us the most stable, owner-blessed install path on Windows.
- Inject `/experimental:c11atomics` into `CFLAGS` and `CXXFLAGS`
  before `meson setup` on Windows. libvmaf uses C11 atomics
  (`stdatomic.h` + `__atomic_*`); MSVC's `<stdatomic.h>` errors
  with `"C atomic support is not enabled"` unless the
  `/experimental:c11atomics` compiler flag is set — the flag is
  opt-in until MSVC ships full C11/C17 atomics support. Setting
  it via env var is preferable to a meson native file because
  the flag is purely build-system-conditional, not source-side:
  gcc / clang / icpx / nvcc don't need it.
- Skip the test step entirely. `windows-latest` has no GPU; running
  even CPU-only tests would consume runner minutes for no signal
  beyond what the Linux legs already provide.

The two job names are pinned to **required** status checks on
`master` immediately after this PR's merge (21 → 23 contexts;
counting the two Linux DNN legs from ADR-0120 if that PR landed
first).

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| **Status quo (Linux GPU only)** | Zero CI cost. | MSVC build-portability regressions only surface from downstream user reports. | The whole motivation for this ADR is closing exactly that hole. |
| **Add only the CUDA Windows leg** | Half the CI minutes; CUDA is the more popular GPU backend. | SYCL on Windows is the less-tested backend → most likely to bit-rot → most valuable to gate. | Half coverage of the hole isn't satisfying; user scope was both. |
| **Add Windows GPU legs but as `experimental: true` (informational)** | Avoids gating master on Windows GPU runner flakiness (Jimver/cuda-toolkit network occasionally times out). | Defeats the purpose: an informational green-vs-yellow distinction won't surface PR regressions to authors who don't notice yellow. | Pin them as required; flakiness is rare enough to absorb the occasional re-run. |
| **Run actual tests via Windows GPU self-hosted runner** | Real GPU-on-CI coverage, not just compile/link. | Requires self-hosted runner infrastructure with a discrete GPU; security/maintenance overhead disproportionate to the benefit; out of scope for this PR. | Out of scope. Could be a separate ADR if the fork ever justifies a self-hosted Windows GPU host. |
| **Use cmake instead of meson on Windows** | cmake has slightly better MSVC integration historically. | The fork has already standardised on meson everywhere; introducing cmake just for Windows GPU legs would create an unmaintained second build path. | Meson works fine on Windows + MSVC; just need correct env vars. |

## Consequences

**Positive:**

- MSVC + CUDA build regressions and MSVC + oneAPI SYCL build
  regressions surface in CI on the PR that introduces them, not from
  downstream user reports months later.
- Windows tier-1 status is upheld for the GPU backends, not just for
  CPU.
- The `vmaf.exe` artifact uploaded by the MinGW CPU job and the
  (eventual, future) MSVC GPU `vmaf.exe` artifact share the same
  test-portability story going forward.

**Negative:**

- Two additional `windows-latest` matrix runs per PR. Each Windows
  runner is ~2× the cost of a Linux runner in GHA minutes. Estimated
  cost: ~25 min wall-clock added per CI run (parallel across the
  matrix), ~50 GHA minutes added per run. Acceptable on the public
  fork's free tier.
- Build-only ≠ runtime-tested. A regression that compiles cleanly
  but produces wrong output on Windows GPUs would still slip through.
  Mitigated by Linux GPU legs catching most behavioural regressions
  via the Ubuntu CUDA / SYCL legs.
- One new third-party action in the workflow:
  `ilammy/msvc-dev-cmd@v1`. Widely used with a good security
  record, but additional supply-chain surface beyond the existing
  `Jimver/cuda-toolkit`. The SYCL leg additionally pulls a
  signed installer from
  `registrationcenter-download.intel.com` over HTTPS — Intel-owned
  infrastructure, the same source `oneapi-src/oneapi-ci` uses.
- The Intel offline-installer URL hard-codes the BaseKit version
  (`2025.3.0.372`) and a per-release directory id. When the Linux
  SYCL leg bumps oneAPI, this URL must be updated in lockstep —
  drift defeats the parity invariant. See `docs/rebase-notes.md`
  entry 0022 for the touch-list.

**Neutral / follow-ups:**

- Branch protection re-pinned atomically with this ADR's merge to add
  `Build — Windows MSVC + CUDA (build only)` and
  `Build — Windows MSVC + oneAPI SYCL (build only)` as required
  contexts.
- A future ADR may add a Windows GPU runtime-test job once a
  self-hosted Windows GPU runner is justified.
- A future ADR may pin the third-party actions to commit SHAs
  (consistent with whatever SHA-pinning policy the rest of the repo
  adopts).

## References

- [ADR-0115](0115-ci-trigger-master-only-and-matrix-consolidation.md) —
  matrix consolidation; this ADR adds a new job to the consolidated
  workflow.
- [ADR-0116](0116-ci-workflow-naming-convention.md) — Title Case
  display names; the two new legs follow that convention.
- [ADR-0120](0120-ai-enabled-ci-matrix-legs.md) — sister ADR (DNN-on
  matrix legs) landed immediately before this one.
- [ADR-0037](0037-master-branch-protection.md) — branch-protection
  policy that the post-merge re-pin updates.
- [ilammy/msvc-dev-cmd@v1](https://github.com/ilammy/msvc-dev-cmd)
- [Jimver/cuda-toolkit@v0.2.35](https://github.com/Jimver/cuda-toolkit)
- [Intel `oneapi-src/oneapi-ci` — official Windows install pattern](https://github.com/oneapi-src/oneapi-ci/blob/master/scripts/install_windows.bat)
- [NVIDIA Windows CUDA sub-package list](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software)
- [MSVC `/experimental:c11atomics` opt-in flag](https://learn.microsoft.com/en-us/cpp/build/reference/std-specify-language-standard-version)
- `req` (paraphrased): user picked "Add both CUDA + SYCL Windows
  build-only legs" via the post-cascade scope popup, after I offered
  it as the most-ambitious of three scope choices for Windows GPU
  coverage.
- Per-surface doc impact: this ADR documents the workflow-file change
  and the branch-protection delta. The CUDA / SYCL backends
  themselves are unchanged; no `docs/backends/` edit needed beyond
  the ADR.
