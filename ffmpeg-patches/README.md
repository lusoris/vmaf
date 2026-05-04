# ffmpeg-patches/

Local patches against FFmpeg **n8.1** for integrating this VMAF fork into
`libavfilter/vf_libvmaf*` plus a new `vf_vmaf_pre` filter.

## Contents

- **`0001-libvmaf-add-tiny-model-option.patch`** — adds `tiny_model` /
  `tiny_device` / `tiny_threads` / `tiny_fp16` options on the existing
  `libvmaf` filter; calls `vmaf_use_tiny_model()` when set.
- **`0002-add-vmaf_pre-filter.patch`** — new `vmaf_pre` filter: luma-in /
  luma-out learned pre-processing via `vmaf_dnn_session_*`. Chroma
  planes pass through.
- **`0003-libvmaf-wire-sycl-backend-selector.patch`** — adds `sycl_device`
  / `sycl_profile` options on the `libvmaf` filter; invokes
  `vmaf_sycl_state_init()` + `vmaf_sycl_import_state()` when
  `sycl_device >= 0`.
- **`0004-libvmaf-wire-vulkan-backend-selector.patch`** — adds
  `vulkan_device` option on the `libvmaf` filter; wires
  `vmaf_vulkan_state_init` + the deferred-pool path from ADR-0238.
- **`0005-libvmaf-add-libvmaf-sycl-filter.patch`** — registers a
  dedicated `libvmaf_sycl` filter for zero-copy VAAPI/QSV import
  (consumes `AVFrame->data[3] -> mfxFrameSurface1*`).
- **`0006-libvmaf-add-libvmaf-vulkan-filter.patch`** — registers a
  dedicated `libvmaf_vulkan` filter for zero-copy VkImage import per
  [ADR-0186](../docs/adr/0186-vulkan-image-import-impl.md).

Every patch is guarded by `check_pkg_config` so it degrades gracefully when
libvmaf was built without the relevant feature (`-Denable_dnn`, `-Denable_sycl`,
`-Denable_vulkan`).

## What works without a patch

Any feature extractor registered in the fork's libvmaf — including
**SSIMULACRA 2**, PSNR-HVS, VIF, ADM, MS-SSIM, motion, CAMBI, etc. — is
already reachable through upstream FFmpeg's stock `libvmaf` filter via
its `feature` option. No patch needed:

```bash
ffmpeg -i ref.mp4 -i dist.mp4 \
  -lavfi "[0:v][1:v]libvmaf=feature='name=ssimulacra2'" \
  -f null -
```

The same `feature=` syntax accepts per-extractor options:

```bash
# BT.709 full-range matrix
feature='name=ssimulacra2:yuv_matrix=2'
```

Works with both the stock `libvmaf` filter and, via CUDA, the
`libvmaf_cuda` filter (SSIMULACRA 2 currently has no CUDA path; it falls
back to CPU SIMD — AVX-512 / AVX2 / NEON per ADR-0161 / 0162 / 0163).

The patches in this directory only cover fork-added surfaces that
DO NOT fit the generic `feature=` plumbing: the DNN session API
(`tiny_model`), the learned pre-processing filter (`vmaf_pre`), the
SYCL / Vulkan backend selectors on the `libvmaf` filter, and the
dedicated `libvmaf_sycl` / `libvmaf_vulkan` filters for zero-copy
hardware-frame import.

## How to apply

```bash
cd /path/to/ffmpeg    # must be at tag n8.1
for p in /path/to/vmaf/ffmpeg-patches/000*-*.patch; do
    git am --3way "$p" || break
done
```

Or via the helper skill: `/ffmpeg-apply-patches /path/to/ffmpeg`.

> **Verification gate (CLAUDE.md §12 r14)**: patches `0002` through
> `0006` build on each other (each assumes the cumulative state of
> every earlier patch). Per
> [ADR-0118](../docs/adr/0118-ffmpeg-patch-series-application.md) and
> [ADR-0186](../docs/adr/0186-vulkan-image-import-impl.md), the
> verification gate for any change touching a libvmaf C-API surface
> the patches consume is a **series replay** against a pristine
> `n8.1` checkout (cumulative `git am --3way`), NOT a per-patch
> `git apply --check`. The latter rejects `0002+` because they
> reference cumulative-state hunks that don't exist in pristine
> `n8.1`. The most recent no-drift verification is
> [ADR-0277 (2026-05-04)](../docs/adr/0277-ffmpeg-patches-refresh-2026-05-04.md);
> the procedure is captured in
> [`docs/rebase-notes.md`](../docs/rebase-notes.md) under the same
> ADR heading.

## How to smoke-test

```bash
bash ffmpeg-patches/test/build-and-run.sh
```

Requires `libvmaf` to be installed (`pkg-config --cflags libvmaf` must
resolve). Set `VMAF_PREFIX` to point at a non-standard install prefix.
Pins `FFMPEG_SHA=n8.1`; override to test against a newer tag.

## How to regenerate

After editing FFmpeg locally, run `/ffmpeg-build-patches` to diff against
the tracked upstream base and rewrite the numbered patches in place.

## License

BSD-3-Clause-Plus-Patent for patches authored in this repo; patches are
applied against FFmpeg (LGPL / GPL) — the resulting linked binary's
distribution terms are governed by FFmpeg's license, not this one.
