# GPU backends C API — `libvmaf_cuda.h` / `libvmaf_sycl.h`

The CUDA and SYCL backends each add their own small API on top of the core
`libvmaf.h` surface — a state object, picture preallocation helpers, and
(SYCL only) zero-copy import paths. This page is the reference for both.

Core API primer: [index.md](index.md). CLI equivalents:
[../usage/cli.md#backend-selection](../usage/cli.md#backend-selection).
Backend dispatch rules + runtime precedence:
[../backends/index.md](../backends/index.md).

## When these headers apply

- The CUDA header is only useful in a build with `-Denable_cuda=true`
  (linking CUDA runtime + nvcc-compiled kernels). When disabled, the symbols
  are absent and calls won't link.
- The SYCL header is only useful in a build with `-Denable_sycl=true`
  (linking oneAPI / Level Zero). Same rule.
- To write portable code that compiles against any libvmaf build, wrap
  GPU-specific sections in `#ifdef HAVE_CUDA` / `#ifdef HAVE_SYCL`, which
  `pkg-config --cflags libvmaf` surfaces automatically.

## CUDA

### Header

[`libvmaf/include/libvmaf/libvmaf_cuda.h`](../../libvmaf/include/libvmaf/libvmaf_cuda.h)

### Lifecycle addition

```
  vmaf_init()
  vmaf_cuda_state_init()         ← new
  vmaf_cuda_import_state()       ← new; hands state to ctx
  vmaf_cuda_preallocate_pictures()
  loop:
    vmaf_cuda_fetch_preallocated_picture()
    ...                                     write into .data[i]
    vmaf_read_pictures()
  vmaf_score_pooled()
  vmaf_close()
  /* state is owned by VmafContext after import; freed with vmaf_close().
   * Use vmaf_cuda_state_free() only when the import never happened —
   * see "Explicit free" below. */
```

### State

```c
typedef struct VmafCudaState VmafCudaState;

typedef struct VmafCudaConfiguration {
    void *cu_ctx;   /* CUcontext; NULL → libvmaf creates one on device 0 */
} VmafCudaConfiguration;

int vmaf_cuda_state_init(VmafCudaState **out, VmafCudaConfiguration cfg);
int vmaf_cuda_import_state(VmafContext *ctx, VmafCudaState *state);
void vmaf_cuda_state_free(VmafCudaState **state);
```

- `cu_ctx = NULL` — libvmaf creates a fresh CUDA context on CUDA device 0.
  This is the common case for standalone tooling.
- `cu_ctx != NULL` — must be a `CUcontext` from the driver API; libvmaf
  adopts it for all allocations. Use this to interop with an application
  that already owns a context (e.g. NVENC / NVDEC).

### Ownership and explicit free

After `vmaf_cuda_import_state(ctx, state)`, the **context owns the
state** — `vmaf_close(ctx)` frees it. Do not import the same state into
two contexts.

`vmaf_cuda_state_free(VmafCudaState **state)` (added in [ADR-0157](../adr/0157-cuda-state-free-api.md))
is the **escape hatch for the pre-import path**: the caller built a
`VmafCudaState` via `vmaf_cuda_state_init()` but never handed it to a
context (e.g. early `vmaf_init()` failure, or a benchmark harness that
constructs and tears down a state without scoring). It tears down the
ring buffer + mutex and releases the cold-start primary context. After
the call the pointer is set to `NULL`.

```c
VmafCudaState *cuda = NULL;
int err = vmaf_cuda_state_init(&cuda, (VmafCudaConfiguration){ .cu_ctx = NULL });
if (err) { return err; }

if (some_unrelated_setup_failed()) {
    vmaf_cuda_state_free(&cuda);   /* not yet imported — caller frees */
    return -1;
}

err = vmaf_cuda_import_state(ctx, cuda);
/* From here on, ctx owns cuda; vmaf_close(ctx) handles the free.
 * Calling vmaf_cuda_state_free(&cuda) AFTER a successful import is
 * undefined behaviour — the context will double-free at vmaf_close. */
```

The asymmetry with the SYCL flavour (`vmaf_sycl_state_free` — see
below — is **always** required) is deliberate: CUDA state is
context-owned post-import; SYCL state outlives a single scoring session
because the queue is queue-scoped. Match the API to the lifetime model
of the underlying runtime.

### Picture preallocation

```c
enum VmafCudaPicturePreallocationMethod {
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
};

typedef struct VmafCudaPictureConfiguration {
    struct { unsigned w, h; unsigned bpc; enum VmafPixelFormat pix_fmt; } pic_params;
    enum VmafCudaPicturePreallocationMethod pic_prealloc_method;
} VmafCudaPictureConfiguration;

int vmaf_cuda_preallocate_pictures(VmafContext *ctx, VmafCudaPictureConfiguration cfg);
int vmaf_cuda_fetch_preallocated_picture(VmafContext *ctx, VmafPicture *pic);
```

Preallocation methods:

| Method | `.data[i]` memory | Use when |
| --- | --- | --- |
| `NONE` | caller-provided | You already own device buffers and set `.data[i]` yourself before `vmaf_read_pictures()`. |
| `DEVICE` | `cudaMalloc` | Your source data already lives on GPU (decoder output, encoder input). No H2D copy on `vmaf_read_pictures`. |
| `HOST` | `malloc` | Your source is on host; libvmaf inserts the H2D copy. |
| `HOST_PINNED` | `cudaMallocHost` | Host source but you want overlap with async compute — pinned memory allows concurrent DMA. |

`HOST_PINNED` is almost always the right choice for CPU-decoded feeds; the
peak throughput difference vs `HOST` on a PCIe-4 x16 link is 15–25% for
1080p. See [../backends/cuda/overview.md](../backends/cuda/overview.md).

### Complete CUDA example

```c
#include <cuda.h>
#include <libvmaf/libvmaf.h>
#include <libvmaf/libvmaf_cuda.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>

int main(void) {
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_WARNING, .n_threads = 4 };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, cfg);

    VmafCudaState *cuda = NULL;
    VmafCudaConfiguration cu_cfg = { .cu_ctx = NULL };  /* libvmaf picks device 0 */
    err = vmaf_cuda_state_init(&cuda, cu_cfg);
    err = vmaf_cuda_import_state(vmaf, cuda);

    VmafModel *model = NULL;
    VmafModelConfig mcfg = { .name = "vmaf" };
    err = vmaf_model_load(&model, &mcfg, "vmaf_v0.6.1");
    err = vmaf_use_features_from_model(vmaf, model);

    VmafCudaPictureConfiguration pcfg = {
        .pic_params = { .w = 1920, .h = 1080, .bpc = 8, .pix_fmt = VMAF_PIX_FMT_YUV420P },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
    };
    err = vmaf_cuda_preallocate_pictures(vmaf, pcfg);

    for (unsigned i = 0; i < nframes; i++) {
        VmafPicture ref = {0}, dist = {0};
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        /* fill ref.data[i] / dist.data[i] — host-pinned, write normally */
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);

    double score;
    err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN, &score, 0, UINT_MAX);
    printf("VMAF: %.17g\n", score);

    vmaf_model_destroy(model);
    vmaf_close(vmaf);  /* also frees the CUDA state */
    return 0;
}
```

### Limitations

- Single device. `VmafCudaConfiguration` does not expose a device index;
  launch libvmaf on device N by setting the current context to N before
  `vmaf_cuda_state_init()` (via `cuCtxSetCurrent` or `cudaSetDevice`).
- No stream parameter. libvmaf runs its own streams internally; interop with
  an external stream is not exposed in v1.
- No HIP path in this header. A HIP backend is planned but not yet
  scaffolded — the `enable_hip` meson option does not exist. See
  [backends/index.md](../backends/index.md).

## SYCL

### Header

[`libvmaf/include/libvmaf/libvmaf_sycl.h`](../../libvmaf/include/libvmaf/libvmaf_sycl.h)

### State

```c
typedef struct VmafSyclState VmafSyclState;

typedef struct VmafSyclConfiguration {
    int device_index;        /* -1 = SYCL default device; 0+ = specific ordinal */
    int enable_profiling;    /* non-zero: queue w/ enable_profiling property */
} VmafSyclConfiguration;

int  vmaf_sycl_state_init(VmafSyclState **out, VmafSyclConfiguration cfg);
int  vmaf_sycl_import_state(VmafContext *ctx, VmafSyclState *state);
void vmaf_sycl_state_free(VmafSyclState **state);
int  vmaf_sycl_list_devices(void);
```

`vmaf_sycl_state_free` is unusual — SYCL state is *not* owned by the context
after import. You must call it explicitly after `vmaf_close(ctx)`. This
asymmetry exists because SYCL USM allocations are queue-scoped and the
queue outlives one scoring session.

`vmaf_sycl_list_devices` enumerates `device_type::gpu` only (CPU / FPGA /
accelerator devices are skipped) and prints one line per device with its
ordinal, platform, vendor, driver version, and fp64 support flag. Returns
the count, or `-EIO` on a SYCL exception. Used by `vmaf_bench --list-devices`
([../usage/bench.md](../usage/bench.md)).

### Picture preallocation (simple path)

```c
enum VmafSyclPicturePreallocationMethod {
    VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_NONE,
    VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_DEVICE,
    VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_HOST,
};

typedef struct VmafSyclPictureConfiguration {
    struct { unsigned w, h; unsigned bpc; enum VmafPixelFormat pix_fmt; } pic_params;
    enum VmafSyclPicturePreallocationMethod pic_prealloc_method;
} VmafSyclPictureConfiguration;

int vmaf_sycl_preallocate_pictures(VmafContext *ctx, VmafSyclPictureConfiguration cfg);
int vmaf_sycl_picture_fetch(VmafContext *ctx, VmafPicture *pic);
```

**Known bug — do not rely on this API.** Unlike the CUDA flavour, the SYCL
simple path does **not** currently honor its preallocation enum.
`vmaf_sycl_preallocate_pictures` is a no-op stub and
`vmaf_sycl_picture_fetch` allocates via the regular host
`vmaf_picture_alloc()` — the `DEVICE` / `HOST` enum values are declared for
symmetry with CUDA but are silently ignored
([`libvmaf/src/libvmaf.c`](../../libvmaf/src/libvmaf.c)). Tracked as
[issue #26](https://github.com/lusoris/vmaf/issues/26). Use the frame-buffer
API below — that is the real GPU-resident path on SYCL today.

### Zero-copy frame-buffer path

For callers that own a GPU-resident decode pipeline (Intel VPL, VA-API,
D3D11), the simple preallocation path forces an unnecessary copy. The
zero-copy API exposes two shared Y-plane buffers (ref + dis) and
alternative ingest entry points that skip `vmaf_picture_alloc` entirely.

```c
int vmaf_sycl_init_frame_buffers (VmafContext *ctx, unsigned w, unsigned h, unsigned bpc);
int vmaf_sycl_get_frame_buffers  (VmafContext *ctx, void **ref, void **dis);
int vmaf_read_pictures_sycl      (VmafContext *ctx, unsigned index);  /* replaces vmaf_read_pictures */
int vmaf_sycl_wait_compute       (VmafContext *ctx);
int vmaf_flush_sycl              (VmafContext *ctx);                  /* replaces (NULL,NULL,0) flush */
```

Typical zero-copy loop:

```c
vmaf_sycl_init_frame_buffers(vmaf, W, H, 8);
void *ref_buf = NULL, *dis_buf = NULL;
vmaf_sycl_get_frame_buffers(vmaf, &ref_buf, &dis_buf);

for (unsigned i = 0; i < nframes; i++) {
    /* write Y-plane luma directly into ref_buf / dis_buf
     * (kernels, SYCL events, dmabuf imports — whatever the decoder exposes) */
    vmaf_read_pictures_sycl(vmaf, i);
    /* vmaf_sycl_wait_compute() is only needed if you must reuse ref_buf/dis_buf
     * for a later frame while compute is still in flight */
}
vmaf_flush_sycl(vmaf);
```

### GPU-resident import paths

```c
/* Linux / Level Zero */
int  vmaf_sycl_dmabuf_import (VmafSyclState *state, int fd, size_t size, void **ptr);
void vmaf_sycl_dmabuf_free   (VmafSyclState *state, void *ptr);

int  vmaf_sycl_import_va_surface (VmafSyclState *state, void *va_display,
                                  unsigned int va_surface, int is_ref,
                                  unsigned w, unsigned h, unsigned bpc);

int  vmaf_sycl_upload_plane (VmafSyclState *state, const void *src, unsigned pitch,
                             int is_ref, unsigned w, unsigned h, unsigned bpc);

/* Windows (conditional) */
#ifdef _WIN32
int  vmaf_sycl_import_d3d11_surface (VmafSyclState *state, void *d3d11_device,
                                     void *d3d11_texture, unsigned subresource,
                                     int is_ref, unsigned w, unsigned h, unsigned bpc);
#endif
```

- `vmaf_sycl_dmabuf_import` is the **primitive** — turns a DMA-BUF fd into a
  SYCL device pointer via Level Zero external memory import. Stable.
- `vmaf_sycl_import_va_surface` is the **convenience wrapper** on top of
  dmabuf — preferred path for a VA-API decode feed. Falls back to
  `vaGetImage + memcpy` when the DRM-PRIME export fails (older Mesa /
  proprietary drivers).
- `vmaf_sycl_upload_plane` is the **platform-agnostic escape hatch** —
  `memcpy` from a host pointer. Use when nothing better works or when you
  need a baseline for benchmarking.
- `vmaf_sycl_import_d3d11_surface` (Windows only) is **declared in the
  public header but not implemented in-tree** (`rg` finds zero
  definitions). The Doxygen block describes an intended host-roundtrip
  design — staging texture → CPU map → H2D memcpy — but no translation
  unit provides the symbol today. Tracked as
  [issue #27](https://github.com/lusoris/vmaf/issues/27). On Windows,
  use `vmaf_sycl_upload_plane` for a host → USM fallback.

See [ADR-0016](../adr/0016-sycl-to-master-merge-conflict-policy.md) for how
these APIs landed and [../backends/sycl/overview.md](../backends/sycl/overview.md)
for the ingestion-path decision tree.

### Profiling helpers

```c
int  vmaf_sycl_profiling_enable     (VmafSyclState *state);
void vmaf_sycl_profiling_disable    (VmafSyclState *state);
void vmaf_sycl_profiling_print      (VmafSyclState *state);
int  vmaf_sycl_profiling_get_string (VmafSyclState *state, char **out);
```

Profiling must be enabled *at init time* — the SYCL queue is created with
the `enable_profiling` property inside `vmaf_sycl_state_init()` only when
`VmafSyclConfiguration.enable_profiling = 1` is passed. `vmaf_sycl_profiling_enable`
does **not** re-create the queue; it only flips a `bool` on the state
(`libvmaf/src/sycl/common.cpp:1053`). If the queue was not built with
`enable_profiling`, calling `vmaf_sycl_profiling_enable` succeeds but
subsequent `get_profiling_info` calls on kernel events will throw a
`sycl::exception`. In practice: set `enable_profiling=1` at init, then use
the enable/disable pair to gate which frame ranges get timed.

`vmaf_sycl_profiling_get_string` yields a caller-owned buffer — free with
`free()`. Equivalent to `vmaf_bench --gpu-profile`
([../usage/bench.md](../usage/bench.md#performance-benchmark-default)).

### Limitations

- Zero-copy ingest paths (`dmabuf_import`, `import_va_surface`) require
  the SYCL queue to use the Level Zero backend — they call
  `sycl::get_native<ext_oneapi_level_zero>` directly. On an OpenCL-backend
  SYCL build these throw `sycl::exception`, which the wrapper catches and
  converts to `-EIO`. The error log is generic (`"SYCL DMA-BUF import
  exception: <what()>"`) rather than a specific "not on Level Zero"
  diagnostic, so callers that want a graceful degradation should detect
  their own backend via `sycl::queue::get_backend()` up front and fall
  back to `vmaf_sycl_upload_plane` without relying on the log text.
- `vmaf_sycl_import_d3d11_surface` is **declared but unimplemented**
  (ghost symbol — see [issue #27](https://github.com/lusoris/vmaf/issues/27)).
  Windows callers must use `vmaf_sycl_upload_plane` today.
- `vmaf_sycl_init_frame_buffers` is single-resolution. Changing `w`/`h`/`bpc`
  mid-stream requires `vmaf_close` + re-init.

## Vulkan

> **Status: scaffold only as of v3.0.** Every entry point in
> [`libvmaf_vulkan.h`](../../libvmaf/include/libvmaf/libvmaf_vulkan.h)
> currently returns `-ENOSYS`. The header lands so downstream
> consumers can compile against the API surface; the runtime + first
> kernel arrive in follow-up PRs per
> [ADR-0127](../adr/0127-vulkan-backend-decision.md) and
> [ADR-0175](../adr/0175-vulkan-backend-scaffold.md). Build with
> `-Denable_vulkan=enabled` to compile the scaffold; it has no
> Vulkan SDK requirement until the runtime PR (T5-1b).

### Header

[`libvmaf/include/libvmaf/libvmaf_vulkan.h`](../../libvmaf/include/libvmaf/libvmaf_vulkan.h)

### State

```c
typedef struct VmafVulkanState VmafVulkanState;

typedef struct VmafVulkanConfiguration {
    int device_index;       /* -1 = first device with compute queue */
    int enable_validation;  /* non-zero: load VK_LAYER_KHRONOS_validation */
} VmafVulkanConfiguration;

int  vmaf_vulkan_available(void);
int  vmaf_vulkan_state_init(VmafVulkanState **out, VmafVulkanConfiguration cfg);
int  vmaf_vulkan_import_state(VmafContext *ctx, VmafVulkanState *state);
void vmaf_vulkan_state_free(VmafVulkanState **state);
int  vmaf_vulkan_list_devices(void);
```

The lifetime model mirrors CUDA's: after
`vmaf_vulkan_import_state(ctx, state)` the context owns the state
and `vmaf_close(ctx)` frees it. The `_state_free()` helper exists
for the pre-import escape hatch (caller built a state but never
imported it — e.g. early `vmaf_init()` failure or a benchmark
harness that constructs and tears down a state without scoring).

`vmaf_vulkan_available()` returns `1` when libvmaf was built with
`-Denable_vulkan=enabled` and `0` otherwise. Until the runtime PR
lands, the helper reports "the build was opted in" rather than "a
working runtime is available" — operators read the docs for status.

### Lifecycle (planned, post-runtime PR)

```text
  vmaf_init()
  vmaf_vulkan_state_init()         ← scaffold returns -ENOSYS today
  vmaf_vulkan_import_state()       ← state ownership transfers
  ...
  vmaf_score_pooled()
  vmaf_close()                     ← frees the imported state
```

### Limitations (scaffold-only)

- `vmaf_vulkan_state_init` returns `-ENOSYS` until the runtime PR
  (T5-1b) wires `volk` + `dependency('vulkan')` + VkInstance /
  VkDevice / compute queue selection.
- `vmaf_vulkan_import_state` returns `-ENOSYS` for the same reason.
- `vmaf_vulkan_list_devices` returns `0` (no devices probed yet).
- No picture preallocation API surface yet — the runtime PR adds
  the equivalent of `VmafCudaPicturePreallocationMethod`.
- The ffmpeg `libvmaf` filter declares a `vulkan_device=N` option
  (added by `ffmpeg-patches/0004-libvmaf-wire-vulkan-backend-selector.patch`)
  but flipping it to `>= 0` is currently a no-op until the runtime
  patch series lands. Documented in
  [`docs/usage/ffmpeg.md`](../usage/ffmpeg.md).

## Related

- [index.md](index.md) — core API (everything on this page sits on top of it)
- [dnn.md](dnn.md) — tiny-AI session API (separate from classic GPU dispatch)
- [../usage/cli.md#backend-selection](../usage/cli.md#backend-selection) —
  `--no_cuda` / `--no_sycl` / `--sycl_device` flags
- [../backends/cuda/overview.md](../backends/cuda/overview.md) and
  [../backends/sycl/overview.md](../backends/sycl/overview.md) —
  user-facing backend pages
- [../usage/bench.md](../usage/bench.md) — `vmaf_bench`, which consumes
  these APIs to produce the perf + validation tables
- [ADR-0016](../adr/0016-sycl-to-master-merge-conflict-policy.md),
  [ADR-0022](../adr/0022-inference-runtime-onnx.md),
  [ADR-0027](../adr/0027-non-conservative-image-pins.md) —
  governing decisions
