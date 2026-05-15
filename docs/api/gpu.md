# GPU backends C API — `libvmaf_cuda.h` / `libvmaf_sycl.h` / `libvmaf_vulkan.h` / `libvmaf_hip.h` / `libvmaf_metal.h`

Each GPU backend adds its own small API on top of the core
`libvmaf.h` surface — a state object, picture preallocation helpers, and
(SYCL / Vulkan / Metal) zero-copy import paths. This page is the reference for
all five backends; HIP still has three unported feature kernels, while Metal
has a live Apple-Silicon runtime and first kernel batch.

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
- The Vulkan header requires `-Denable_vulkan=true` (linking volk + the
  compute-shader feature kernels). Same rule.
- The HIP header requires `-Denable_hip=true -Denable_hipcc=true` (linking
  ROCm). 8 of 11 feature kernels are real; 3 stubs (`adm`, `vif`,
  `integer_motion`) return `-ENOSYS`.
- The Metal header requires `-Denable_metal=auto/enabled` on macOS.
  Runtime entry points are live on Apple Silicon; unsupported devices
  return `-ENODEV`. Eight feature kernels are currently wired.
- To write portable code that compiles against any libvmaf build, wrap
  GPU-specific sections in `#ifdef HAVE_CUDA` / `#ifdef HAVE_SYCL` /
  `#ifdef HAVE_VULKAN` / `#ifdef HAVE_HIP` / `#ifdef HAVE_METAL`, which
  `pkg-config --cflags libvmaf` surfaces automatically.

## CUDA

### Header

[`libvmaf/include/libvmaf/libvmaf_cuda.h`](../../libvmaf/include/libvmaf/libvmaf_cuda.h)

### Lifecycle addition

```text
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

`vmaf_cuda_state_free(VmafCudaState **state)` (added in [ADR-0157](../adr/0157-cuda-preallocation-leak-netflix-1300.md))
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
- No HIP path in this header. The HIP / AMD-ROCm backend is being
  scaffolded under T7-10 (PR #200, in flight) — a future
  `libvmaf_hip.h` will mirror this surface. See
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

> **Status: T5-1c closed — full default-model coverage on Vulkan.**
> The state-level entry points (`vmaf_vulkan_state_init` /
> `_import_state` / `_state_free` / `_list_devices` /
> `_available`) plumb a real volk-backed VkInstance + VkDevice +
> compute queue. Live extractors: `vif_vulkan`, `motion_vulkan`,
> `motion_v2_vulkan`, `adm_vulkan`, `float_adm_vulkan`,
> `float_vif_vulkan`, `float_motion_vulkan`, `psnr_vulkan` (luma;
> chroma via T3-15(b) PR #204), `psnr_hvs_vulkan`, `ciede_vulkan`,
> `float_ssim_vulkan`, `float_ms_ssim_vulkan`, `float_ansnr_vulkan`,
> `float_moment_vulkan`, `ssimulacra2_vulkan`, `cambi_vulkan`. The
> default `vmaf_v0.6.1` model runs end-to-end on Vulkan against a
> real ICD. Image-import zero-copy entry points
> (`vmaf_vulkan_state_init_external` / `_import_image` /
> `_wait_compute` / `_read_imported_pictures`) landed via T7-29 /
> [ADR-0186](../adr/0186-vulkan-image-import-impl.md) for FFmpeg
> AVVulkanDeviceContext interop. Build with
> `-Denable_vulkan=enabled`; the build picks up `volk` and
> `dependency('vulkan')` automatically. See
> [ADR-0127](../adr/0127-vulkan-compute-backend.md),
> [ADR-0175](../adr/0175-vulkan-backend-scaffold.md),
> [ADR-0176](../adr/0176-vulkan-vif-cross-backend-gate.md),
> [ADR-0177](../adr/0177-vulkan-motion-kernel.md),
> [ADR-0178](../adr/0178-vulkan-adm-kernel.md),
> [ADR-0186](../adr/0186-vulkan-image-import-impl.md),
> [ADR-0210](../adr/0210-cambi-vulkan-integration.md).

### Header

[`libvmaf/include/libvmaf/libvmaf_vulkan.h`](../../libvmaf/include/libvmaf/libvmaf_vulkan.h)

### State

```c
typedef struct VmafVulkanState VmafVulkanState;

typedef struct VmafVulkanConfiguration {
    int device_index;                /* -1 = first device with compute queue */
    int enable_validation;           /* non-zero: load VK_LAYER_KHRONOS_validation */
    unsigned max_outstanding_frames; /* 0 = default (4); clamped to [1, 8] */
} VmafVulkanConfiguration;

int      vmaf_vulkan_available(void);
int      vmaf_vulkan_state_init(VmafVulkanState **out, VmafVulkanConfiguration cfg);
unsigned vmaf_vulkan_state_max_outstanding_frames(const VmafVulkanState *state);
int      vmaf_vulkan_import_state(VmafContext *ctx, VmafVulkanState *state);
void     vmaf_vulkan_state_free(VmafVulkanState **state);
int      vmaf_vulkan_list_devices(void);
```

The lifetime model mirrors CUDA's: after
`vmaf_vulkan_import_state(ctx, state)` the context owns the state
and `vmaf_close(ctx)` frees it. The `_state_free()` helper exists
for the pre-import escape hatch (caller built a state but never
imported it — e.g. early `vmaf_init()` failure or a benchmark
harness that constructs and tears down a state without scoring).

`vmaf_vulkan_available()` returns `1` when libvmaf was built with
`-Denable_vulkan=enabled` and `0` otherwise.

### Lifecycle

```text
  vmaf_init()
  vmaf_vulkan_state_init()         ← creates VkInstance + VkDevice + compute queue
  vmaf_vulkan_import_state()       ← state ownership transfers to ctx
  ...
  vmaf_score_pooled()
  vmaf_close()                     ← frees the imported state
```

For zero-copy interop with caller-owned VkInstance / VkDevice handles
(typically from FFmpeg's `AVVulkanDeviceContext`), use
`vmaf_vulkan_state_init_external` together with
`vmaf_vulkan_import_image` / `vmaf_vulkan_wait_compute` /
`vmaf_vulkan_read_imported_pictures`. See
[ADR-0186](../adr/0186-vulkan-image-import-impl.md) and
[`backends/vulkan/overview.md`](../backends/vulkan/overview.md).

#### Async pending-fence pipelining (v2 — ADR-0251)

`vmaf_vulkan_import_image` is **non-blocking** as of T7-29
part 4. It records the GPU copy, submits to the compute
queue, and returns immediately — the caller's decoder
thread can run ahead while libvmaf's transfer queue drains
in the background. Up to `max_outstanding_frames` (default
`4`) frames may be in flight before the next
`import_image` call back-pressures on the oldest fence.

The drain happens automatically inside
`vmaf_vulkan_state_build_pictures` (called by
`vmaf_vulkan_read_imported_pictures`); callers who need an
explicit drain — e.g. before reusing the imported VkImage
on the decoder side — call `vmaf_vulkan_wait_compute()`,
which now blocks on every outstanding fence in the ring.

Memory cost: the staging arena scales with
`max_outstanding_frames`. At the default depth and 1080p
8-bit Y, the arena is roughly **16 MiB** of host-visible
buffers per `VmafVulkanState`. Higher resolutions or
multi-state setups should size accordingly.

The ring depth is configurable via
`VmafVulkanConfiguration.max_outstanding_frames` (0 selects
the canonical default of 4; values are clamped to [1, 8]
internally). The clamped value is observable via
`vmaf_vulkan_state_max_outstanding_frames()`. ADR-0235
follow-up #3, T7-29 part 4 (this knob currently affects only
`vmaf_vulkan_state_init`; external-handles callers receive
the default until a separate ABI bump extends
`VmafVulkanExternalHandles`).

#### Picture preallocation (ADR-0238)

Mirrors the CUDA / SYCL preallocation surface:

```c
enum VmafVulkanPicturePreallocationMethod {
    VMAF_VULKAN_PICTURE_PREALLOCATION_METHOD_NONE = 0,
    VMAF_VULKAN_PICTURE_PREALLOCATION_METHOD_HOST,
    VMAF_VULKAN_PICTURE_PREALLOCATION_METHOD_DEVICE,
};

typedef struct VmafVulkanPictureConfiguration {
    struct {
        unsigned w, h;
        unsigned bpc;
        enum VmafPixelFormat pix_fmt;
    } pic_params;
    enum VmafVulkanPicturePreallocationMethod pic_prealloc_method;
} VmafVulkanPictureConfiguration;

int vmaf_vulkan_preallocate_pictures(VmafContext *vmaf, VmafVulkanPictureConfiguration cfg);
int vmaf_vulkan_picture_fetch(VmafContext *vmaf, VmafPicture *pic);
```

`HOST` allocates pictures via the regular `vmaf_picture_alloc`;
`DEVICE` backs each picture's luma plane with a host-visible Vulkan
buffer (VMA `AUTO_PREFER_HOST`) — the persistent mapped pointer is
exposed as `pic->data[0]`, so the caller writes once and the kernel
descriptor sets read the same memory. Pool depth is fixed at the
canonical `frames-in-flight = 2` (matches SYCL); pictures are
dispensed round-robin via `vmaf_vulkan_picture_fetch`. Fetch falls
back to a host-backed picture if the caller skipped
`preallocate_pictures` entirely.

### Limitations

- Pool depth is currently compile-time `pic_cnt = 2` (matches SYCL).
  Growing the depth is an additive
  `VmafVulkanPictureConfiguration` field — gated on a real workload
  needing more.
- Pool currently allocates the Y plane only (matches SYCL). Chroma-aware
  extractors that want preallocated U/V planes need a follow-up.
- The ffmpeg `libvmaf` filter exposes `vulkan_device=N` (set to
  `>= 0` to enable the Vulkan backend; see
  [`docs/usage/ffmpeg.md`](../usage/ffmpeg.md)). Image-import
  zero-copy through `AVVulkanDeviceContext` is wired by
  `ffmpeg-patches/0004-libvmaf-wire-vulkan-backend-selector.patch`
  on top of T7-29's `_state_init_external` API.
- HIP / AMD-ROCm support: `libvmaf_hip.h` is shipping (T7-10 scaffold,
  ADR-0212; runtime + 8/11 real feature kernels via PRs #686/#695/#696/#710/#712).
  3 kernels remain `-ENOSYS` stubs (adm/vif/integer_motion). FFmpeg
  integration is wired by `ffmpeg-patches/0011-libvmaf-wire-hip-backend-selector.patch`
  (`--enable-libvmaf-hip` + `hip_device=N`, ADR-0380).

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
