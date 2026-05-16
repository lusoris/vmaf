# HIP Feature Extractors — Invariant Notes

## Memory copy direction enum discipline

Every `hipMemcpy*` call's direction enum **must match the actual memory placement** of source and destination pointers:

- `hipMemcpyHostToDevice`: source is host-accessible (CPU pointer), destination is device-side
- `hipMemcpyDeviceToHost`: source is device-side, destination is host-accessible (CPU or pinned)
- `hipMemcpyDeviceToDevice`: source and destination are both device-side

Mismatches are undefined behavior on some HIP runtimes and may silently corrupt results or trigger runtime faults.

**Established patterns:**
- Picture planes arrive from the VMAF pipeline as CPU-side `VmafPicture` structs with `data[0..2]` pointers (host memory). Copying these into device-allocated staging buffers requires `hipMemcpyHostToDevice`.
- Readback buffers allocated via `hipHostMalloc` in `src/hip/kernel_template.c` are host-pinned memory, safe to use with `hipMemcpyDeviceToHost` for kernel output collection.

See PR #[TBD] / ADR-[TBD] for the discovery and fix of `integer_psnr_hip.c` lines 316/322 (2026-05-16 GPU audit).

## Registration coverage invariant

Every new HIP `VmafFeatureExtractor` added to `feature_extractor.c`'s
`feature_extractor_list[]` MUST appear in `libvmaf/test/test_hip_smoke.c`'s
registration table with a `vmaf_get_feature_extractor_by_name` assertion
in the same PR. Raw kernel-stub helpers (files exporting only
`vmaf_hip_<name>_init` / `_run` / `_destroy` without a
`VmafFeatureExtractor` struct) are exempt until they are promoted to
registered extractors.
