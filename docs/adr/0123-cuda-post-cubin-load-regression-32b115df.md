# ADR-0123: CUDA post-cubin-load regression — upstream 32b115df

- **Status**: Proposed
- **Date**: 2026-04-19
- **Deciders**: lusoris
- **Tags**: `cuda`, `investigation`, `upstream-sync`

## Context

ADR-0122 (CUDA gencode coverage + init hardening) ships defensive
improvements — they widen GPU coverage and turn driver-load failures
into actionable log messages — but are explicitly **not** the fix for
the runtime crash reported by external user lawrence on 2026-04-19.

Lawrence's setup:

- Ampere `sm_86` (RTX 30xx-class GPU).
- Static ffmpeg build (`ldd` shows only libc-family; no `libvmaf.so`,
  no `libcuda.so` — the whole libvmaf is baked in at link time).
- Has his downstream `vmaf-nvcc.patch` that adds `sm_86` to the
  gencode array. So cubin coverage is not the issue for *his*
  repro — ADR-0122's gencode change is what upstream Netflix
  needed, not what lawrence's patched build needed.
- Nonetheless his build still crashes on the first frame after CUDA
  initialisation completes.

Lawrence's own diagnosis (verbatim, 2026-04-19):

> That commit is from a couple weeks ago. However I'm almost 99%
> certain that `32b115df92f04e715ad3efa1a66ae925dc69844d` from
> upstream is what introduced the issue.
>
> It wasn't until it rebased that the issues started happening in
> your fork too.

Upstream commit `32b115df92f04e715ad3efa1a66ae925dc69844d`
("libvmaf: add experimental `VMAF_BATCH_THREADING` and
`VMAF_PICTURE_POOL` threading modes", Kyle Swanson, 2026-04-07)
touches:

- `libvmaf/include/libvmaf/libvmaf.h` (+41 public API)
- `libvmaf/src/libvmaf.c` (+227 lines in the submit/collect core)
- `libvmaf/src/picture_pool.{c,h}` (+282 lines, new file)
- `libvmaf/src/thread_pool.{c,h}` (+62 lines)
- `libvmaf/test/test_pic_preallocation.c` (+540 lines, new test)

The prior rebase of the lusoris fork onto upstream `master` picked
this commit up; it is present on every fork branch that contains
the most recent upstream sync (including `master`).

This is a **regression introduced upstream** that happens to be
triggered by our fork's use of the CUDA backend — the new threading
modes change how pictures are submitted to feature extractors, and
the CUDA runtime path (ring-buffered double-buffer submit with
per-extractor streams) is more sensitive to ordering than the CPU
path. A confirmed bisect and reproducer on a GPU-capable runner is
required before a fix commits.

## Decision

*This ADR tracks the investigation; the concrete decision lands in
a follow-up Supersedes-or-extends ADR once the bisect is done.*

The investigation is scoped as follows:

1. **Bisect 32b115df on a GPU-capable host.** Confirm that
   checkouts of the lusoris fork with `32b115df` reverted no longer
   crash on `sm_86` / `sm_89` / any other consumer NVIDIA card with
   a CUDA-enabled static ffmpeg. This also validates that lawrence's
   identification is correct and not a coincidence.

2. **Reproduce on a minimal input.** Confirm the crash happens on
   a tiny YUV pair (a few frames) under plain
   `./build/tools/vmaf --feature integer_vif --reference ... --distorted ...`
   with CUDA auto-selected, not only through ffmpeg. Narrows the
   failure to libvmaf's own submit/collect path vs ffmpeg
   integration.

3. **Symbolise the crash.** Run under `cuda-gdb` or CUDA compute
   sanitizer on the bisected-bad build to confirm whether it is:
   - A null-pointer deref in the new picture-pool path (most
     likely, given symptoms).
   - A cross-stream synchronisation bug where the new picture-pool
     releases a picture back to the pool before the GPU has
     finished reading it.
   - A thread-pool lifecycle bug that affects the CUDA
     finished-event callback specifically.

4. **Choose remediation** from the options in the Alternatives
   table. Document the choice in a follow-up ADR that supersedes
   the "Proposed" status of this one and lands the fix.

## Alternatives considered

*Bisect / reproduction / fix-option decisions are the focus of the
follow-up ADR. Captured here as the current working shortlist so the
open options remain visible:*

| Option | Pros | Cons |
| --- | --- | --- |
| Revert `32b115df` on `master` until upstream fixes it | Fastest path to green CUDA for fork users. | Re-opens sync-with-upstream debt; every subsequent `/sync-upstream` must skip or re-revert. |
| Gate `32b115df` behind `-Dexperimental_threading=false` (off by default) | Code stays in-tree; disarmed at runtime; keeps upstream sync clean. | Requires a meson-option shim plus a runtime guard in every new entry point that enters the batch/pool paths. |
| Fix the regression in-place + send the fix upstream | Upstream gets the benefit too; smallest long-term diff. | Requires reproducing the bug locally; may take longer than users can wait. |
| Combined: off-by-default gate now, fix upstream in parallel | Immediate user safety + long-term correctness. | Largest PR surface in one go; more review load. |

## Consequences

*Defined once the follow-up ADR lands. This stub records only that
the investigation is open and the option set above is the starting
point.*

## References

- ADR-0122 — defensive hardening (gencode + init) landed in the PR
  that preceded this investigation.
- Upstream commit `32b115df92f04e715ad3efa1a66ae925dc69844d`
  ("libvmaf: add experimental VMAF_BATCH_THREADING and
  VMAF_PICTURE_POOL threading modes", Kyle Swanson, 2026-04-07).
- External-user repro thread (2026-04-19, "lawrence"): *"I'm almost
  99% certain that 32b115df… is what introduced the issue. It
  wasn't until it rebased that the issues started happening in
  your fork too."*
- Memory: `project_cuda_framesync_segfault.md` — running log of the
  investigation.
- Source: `req` — user confirmed scope: *paraphrased:* "Land
  defensive PR first, then open focused ADR-0123 vs 32b115df."
