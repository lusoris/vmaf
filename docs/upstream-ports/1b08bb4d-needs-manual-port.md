# Upstream port — Netflix/vmaf `1b08bb4d` (needs manual port)

**Upstream commit:** [`1b08bb4d`](https://github.com/Netflix/vmaf/commit/1b08bb4da533fe3bc08700e4c1adb45f5adccc0b)
*"runtime load CUDA driver libraries using nv-codec-headers and support clang compilation"*
— Maximilian Müller (NVIDIA), 2025-09-01.

**Status:** DRAFT — irreconcilable cherry-pick, needs manual rewrite.

## What the upstream commit does

Restructures the fork's CUDA backend along three axes:

1. **Runtime-load CUDA driver libraries** via `nv-codec-headers` (`<ffnvcodec/dynlink_loader.h>`) instead of static-linking
   `libcuda.so` at build time. Removes hard build dependency on the
   CUDA Driver API stub library.
2. **Clang+CUDA compilation support** — drops the `nvcc`-only
   assumption in the meson recipe so `clang` can compile `.cu` files
   directly (CUDA Toolkit headers + libs still required).
3. **Dockerfile rewrite** — collapses the separate `Dockerfile.cuda`
   image into the main `Dockerfile`; adds a new `Dockerfile.ffmpeg`;
   removes redundant CUDA event recreation in extractors.

## Why the cherry-pick failed

`git cherry-pick 1b08bb4d` against fork `master` produces 7 conflicting
files plus one delete/modify conflict:

| file | conflict class |
|---|---|
| `libvmaf/src/feature/cuda/integer_motion/motion_score.cu` | content |
| `libvmaf/src/feature/cuda/integer_motion_cuda.c` | content |
| `libvmaf/src/feature/cuda/integer_vif_cuda.c` | content |
| `libvmaf/src/libvmaf.c` | content |
| `libvmaf/src/meson.build` | content |
| `libvmaf/src/picture.h` | content |
| `libvmaf/tools/vmaf.c` | content |
| `resource/doc/docker.md` | delete-vs-modify (fork removed `resource/doc/`) |

The fork's CUDA backend has progressed past the upstream baseline this
patch was authored against — picture-pool refactors (T-GPU-DEDUP-1+),
async pending-fence rings (T7-29), and several extractor-level
refactors all touch the same lines. The patch shape no longer maps
cleanly onto the fork.

## Required manual-port plan

A future PR should re-implement each of the three axes independently:

1. **`nv-codec-headers` runtime load** — adopt `<ffnvcodec/dynlink_loader.h>`
   in `libvmaf/src/cuda/common.c`/`common.h`, replace the direct
   `libcuda.so` linkage in `libvmaf/src/meson.build`, and add a fallback
   path for environments without `nv-codec-headers` so existing CI rows
   keep working. Likely needs an ADR (build-system delta).
2. **Clang+CUDA compilation** — extend the existing `enable_cuda` meson
   option to detect `nvcc` vs `clang` host compiler and dispatch the
   right `cuda_args`. Pairs with the fork's existing CI matrix
   (`gcc / clang / nvcc`).
3. **Docker reorg** — separate PR; fork already maintains its own
   `Dockerfile` topology (`Dockerfile.ffmpeg`, GPU images), so this is
   a docs/build-only refactor.

Each axis should land as its own `port/` PR with the relevant ADR.
This umbrella note exists so future readers see why the trivial
`/port-upstream-commit 1b08bb4d` path is closed.

## References

- Upstream commit: <https://github.com/Netflix/vmaf/commit/1b08bb4da533fe3bc08700e4c1adb45f5adccc0b>
- Sync-upstream report 2026-05-02: `docs/sync-upstream/2026-05-02-sync-report.md`
  on branch `chore/sync-upstream-2026-05` (PR #295), table "PORT —
  small fix worth picking", row `1b08bb4d`.
