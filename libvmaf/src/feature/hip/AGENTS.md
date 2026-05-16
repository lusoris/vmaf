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

## `integer_psnr_hip.c` honours `enable_chroma` option parity (ADR-0453)

`integer_psnr_hip.c` mirrors the CUDA / Vulkan / SYCL twins: the
`enable_chroma` boolean option (default `true`) controls whether chroma
planes (Cb, Cr) are dispatched alongside luma. When `false`, or when the
source is YUV4:0:0, `n_planes` is clamped to 1 in `init` and only
`psnr_y` is emitted. The per-plane readback array `rb[PSNR_HIP_NUM_PLANES]`
and staging buffer arrays `ref_in[3]` / `dis_in[3]` are sized statically;
only the first `n_planes` slots are allocated and used.
