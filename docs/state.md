# Fork bug-status — `docs/state.md`

_Updated: 2026-04-29._

The tracked, in-tree register of **bug status** for this fork. Per
[ADR-0165](adr/0165-state-md-bug-tracking.md) and
[CLAUDE.md §12 rule 13](../CLAUDE.md), every PR that closes, opens,
or rules out a bug updates this file in the same PR. The goal is to
prevent re-investigation of already-closed bugs across session resets.

**Scope split:**

- **This file** — bug status only (Open / Recently closed /
  Confirmed not-affected / Deferred).
- [`docs/adr/`](adr/) — architectural and policy decisions
  (one file per non-trivial choice; immutable once Accepted).
- [`.workingdir2/BACKLOG.md`](../.workingdir2/BACKLOG.md) — local
  planning dossier (T-numbered backlog items, gitignored).

`Netflix#N` = upstream issue / PR; `#N` = fork PR.

## Open bugs

_Bugs known to affect the fork or the user-visible surface, with no
landed fix yet._

| Bug | Summary | Reproducer | Owner | Target |
|---|---|---|---|---|
| _(none)_ | All known correctness bugs that affect the fork are either landed or explicitly deferred (see below). New entries go here as discovered. | — | — | — |

## Deferred (waiting on external dataset access)

| Item | Defer rationale | Reopen trigger |
|---|---|---|
| Tiny-AI C1 baseline `fr_regressor_v1.onnx` (BACKLOG T6-1 sub-item) | Netflix Public Dataset was access-gated (Google Drive folder requiring manual request to Netflix); cannot be downloaded programmatically. C1's defining target is "match `vmaf_v0.6.1` PLCC on Netflix Public", so substituting another dataset would ship a non-comparable number. | **TRIGGERED 2026-04-29** — dataset is locally available at `.workingdir2/netflix/` (9 ref + 70 dis YUVs, ~37 GB, gitignored; provided by lawrence 2026-04-27). T6-1a unblocked; training pipeline ready when scheduled. ADR-0168 § "Defer C1" carries the original audit trail. |

## Recently closed (last ~3 months)

_Bugs closed in the last ~90 days. Older entries roll off into
`git log` and the per-PR ADRs._

| Bug | Closed by | ADR | Verification |
|---|---|---|---|
| **T7-16** — NVIDIA-Vulkan + SYCL `adm_scale2` boundary drift (2.4e-4, 1/48 frames) baseline at PR #120 | this PR (empirical close, sister of T7-15) | — (verification-only close, no ADR) | `python3 scripts/ci/cross_backend_vif_diff.py --feature adm --backend vulkan --device 0` (NVIDIA proprietary 595.58.3.0) reports `adm_scale2` max_abs_diff = 1e-6 (JSON `%f` print floor; ULP=0) at `places=4`, 0/48 mismatches. Same bit-exact result on Vulkan device 1 (Arc Mesa anv 26.0.5) and SYCL device 0 (Arc A380). 2.4e-4 baseline at PR #120 is gone. No `adm_vulkan.c` / `adm_sycl.cpp` commits since PR #120 — same NVCC / driver / SYCL-runtime upgrade hypothesis as T7-15 |
| **T7-15** — `motion_cuda` + `motion_sycl` 2.6e-3 SAD drift vs CPU `integer_motion` on 47/48 frames (surfaced by PR #120's corrected cross-backend gate) | #172 (empirical close — no motion-kernel commits between PR #120 and master; the NVCC 13.x / NVIDIA-driver upgrade since PR #120 is the most likely cause of the bit-exact restoration) | — (no ADR — verification-only close; reopened as a new T-row if the gate ever re-fails) | `python3 scripts/ci/cross_backend_vif_diff.py --feature motion --backend cuda` reports `max_abs_diff=0.0` at `places=8` over 48 frames (was 2.6e-3 47/48 mismatches). SYCL on Arc and Vulkan on Mesa anv each show 1e-6 (JSON `%f` print-rounding floor; ULP=0). All three backends pass the existing `places=4` contract; the gate locks the contract going forward |
| `libvmaf_vulkan.h` not installed under prefix → FFmpeg `--enable-libvmaf-vulkan` silently drops the filter (lawrence repro 2026-04-28 19:27) | #175 (`4b43ad2f`, 2026-04-28) | — | `meson install --destdir /tmp/x` produces `/tmp/x/usr/local/include/libvmaf/libvmaf_vulkan.h` post-fix (was missing); FFmpeg `configure --enable-libvmaf-vulkan` then passes the `check_pkg_config libvmaf_vulkan ... libvmaf/libvmaf_vulkan.h vmaf_vulkan_state_init_external` probe and the `libvmaf_vulkan` filter actually builds |
| `libvmaf.pc` Cflags leak (build-dir `-include` path) on static builds — broke lawrence's BtbN FFmpeg `configure` 2026-04-27 22:19 | this PR | [ADR-0200](adr/0200-volk-priv-remap-pkgconfig-leak-fix.md) | `pkg-config --cflags libvmaf` post-fix returns `-I${includedir} -I${includedir}/libvmaf -DVK_NO_PROTOTYPES -pthread` (no leaked path); rename behaviour byte-for-byte identical (0 GLOBAL `vk*`, 719 `vmaf_priv_vk*` in static `libvmaf.a`); shared `libvmaf.so` Cflags unchanged |
| volk / `vk*` symbol clash in BtbN-style fully-static FFmpeg builds (lawrence repro 2026-04-27) | #152 (`73620ff5`, 2026-04-27) | [ADR-0198](adr/0198-volk-priv-remap-static-archive.md) | Static `nm libvmaf.a` reports 0 GLOBAL `vk*` (was ~700); BtbN-style `gcc -static main.c libvmaf.a libvulkan-stub.a` link succeeds; `test_vulkan_smoke` 10/10 pass |
| Netflix#755 — `vmaf_score_pooled` interleaves with `vmaf_read_pictures` | #91 `9b983e0a` (2026-04-24) | [ADR-0154](adr/0154-score-pooled-eagain.md) | API contract test + Netflix golden gate (CPU bit-identical) |
| Netflix#910 — out-of-order flush misses last frame | #88 `f478c65d` (2026-04-24) | [ADR-0152](adr/0152-monotonic-index-rejection.md) | Regression test rejects non-monotonic indices with `-EINVAL` |
| Netflix#1414 — `float_ms_ssim` broken at <176×176 | #90 `7905ac78` (2026-04-24) | [ADR-0153](adr/0153-float-ms-ssim-min-size.md) | Init-time rejection with `-EINVAL` + regression test |
| Netflix#1420 — CUDA concurrency assert at `cuda/common.c:166` | #93 `49a64088` (2026-04-24) | [ADR-0156](adr/0156-cuda-graceful-error-propagation.md) | 178 `CHECK_CUDA` sites replaced with `-errno` propagation; OOM reducer hits `-ENOMEM` (was: `assert(0)`) |
| Netflix#1300 — CUDA preallocation memory leak | #94 `fd1b22c2` (2026-04-24) | [ADR-0157](adr/0157-cuda-state-free-api.md) | New `vmaf_cuda_state_free()` API + ASan reducer confirms 0 framework-side leaked bytes across 10 init/preallocate/fetch/close cycles |
| Netflix#1486 — motion edge-mirror + `motion_max_val` + `motion3` output | #95 `383190a4` (2026-04-24) | [ADR-0158](adr/0158-motion-updates-verified.md) | Doc-only verify-PR; substance already on master via earlier incremental commits |
| Netflix#1376 — Python FIFO hang on slow IO | #85 `e5a52e74` (2026-04-24) | [ADR-0149](adr/0149-fifo-semaphore.md) | Replaces 1-second polling with `multiprocessing.Semaphore` |
| Netflix#1472 — CUDA feature extraction broken on Windows MSYS2/MinGW | #86 `f9d1cae2` (2026-04-24) | [ADR-0150](adr/0150-cuda-windows-msys2.md) | Linux CPU 32/32 + CUDA 35/35 + Windows MSVC+CUDA CI build-only green |
| Netflix#1430 — locale-unsafe parsing (comma decimal) | #74 `e0e78db3` (earlier) | [ADR-0137](adr/0137-thread-locale-handling.md) | New `thread_locale.{c,h}` subsystem; round-trip parse tests |
| Netflix#1382 / #1381 — `cuMemFreeAsync` use-after-free on concurrent free | #72 (Batch-A) | [ADR-0131](adr/0131-batch-a-cumemfree.md) | `cuMemFree` port; assertion-0 crash no longer reproduces |
| Netflix#1476 — UB in `void*` pointer arithmetic + VIF-init memory leak | leak: #47; UB: master `b0a4ac3a` | — | ASan repro green before/after |
| CUDA framesync segfault on null cubin | #62 `661a8ac9`; #60 `d3b6fad6` | [ADR-0123](adr/0123-cuda-framesync-null-guard.md) / [ADR-0122](adr/0122-cuda-post-cubin-load-hardening.md) | Null-guard + post-cubin-load hardening; segfault no longer reproduces |

## Confirmed not-affected (or already-fixed upstream of the fork's master)

_Netflix bugs that surfaced during triage but **don't** apply to the
fork's code paths. Recording them here protects future sessions from
re-investigating dead ends._

| Netflix issue | Status on this fork | Evidence |
|---|---|---|
| Netflix#1032 — PSNR-HVS NaN on 16-bit | **Already-fixed upstream `b1e3f3bd`** is in fork master; CLI rejects bpc>12 with `-EINVAL` and clear error, no NaN produced | Verified by reading `libvmaf/tools/cli_parse.c` bpc validation + `libvmaf/src/feature/psnr_hvs.c` |
| Netflix#1449 — SSIM incorrect when smaller dimension > 384 px | **Already-fixed upstream `7e16db0a`** (scale option). Fork default is `auto` (Wang-Bovik paper); `float_ssim=scale=1` gives full-res SSIM | Verified via `/cross-backend-diff` on test fixtures |
| Netflix#1481 — i686 (32-bit x86) build regression | **Build-only matrix row exists** ([`libvmaf-build-matrix.yml`](../.github/workflows/libvmaf-build-matrix.yml) i686 cross-file with `-Denable_asm=false`); reproduces the regression for any future drift | [ADR-0151](adr/0151-i686-build-row.md) |

## Deferred (waiting on external trigger)

_Bugs known to affect the fork where the fix is gated on an external
event — typically Netflix merging an upstream fix that the fork
preserves bit-exactness against._

| Bug | Defer rationale | Reopen trigger | Watching |
|---|---|---|---|
| Netflix#955 — `i4_adm_cm` rounding overflow (`1u << 31` overflows `int32_t add_bef_shift_flt[]`) | Bit-exactness against Netflix golden requires preserving the overflow until Netflix merges their own fix and updates the goldens | Netflix merges PR #1494 (`feature/adm: fix integer precision issue`) to master | Scheduled remote agent fires 2026-05-01 to check PR #1494 status (re-runs weekly until merged). [ADR-0155](adr/0155-i4-adm-cm-defer.md) |

## Update protocol

When a PR closes / opens / rules out a bug:

1. Add or move the row in the appropriate section above.
2. Cross-link the ADR (if any), the PR number + commit, and the
   Netflix issue (if applicable).
3. For "Recently closed" entries, include enough verification detail
   that a future session can confirm the fix without re-running the
   reducer.
4. For "Confirmed not-affected" rows, cite the file path + reasoning
   that proves the fork is not in scope.

Older "Recently closed" rows roll off after ~90 days; the audit
trail then lives in `git log` and the closing ADR.
