# DNN session API — `libvmaf/dnn.h`

The DNN surface in [`libvmaf/include/libvmaf/dnn.h`](../../libvmaf/include/libvmaf/dnn.h)
lets callers load and run tiny ONNX models from C, either attached to a
`VmafContext` (so DNN scores show up next to SVM scores in the normal VMAF
report) or as a standalone session (luma-in / luma-out filter-style
inference, no VmafContext required).

This is the runtime half of the tiny-AI surface; training lives in `ai/`.
See [ADR-0022](../adr/0022-inference-runtime-onnx.md) (ORT as the inference
runtime) and [ADR-0023](../adr/0023-tinyai-user-surfaces.md) (the four user
surfaces: CLI, C API, ffmpeg, training).

## Availability check

```c
int vmaf_dnn_available(void);
```

Returns `1` if libvmaf was built with `-Denable_dnn=true` and ONNX Runtime is
linked, `0` otherwise. When `0`, every other entry point in `dnn.h` returns
`-ENOSYS`. This is the cheap way to branch between DNN and classic-only
build configs at runtime without wrapping every call in its own check.

## Device config — `VmafDnnConfig`

```c
typedef enum VmafDnnDevice {
    VMAF_DNN_DEVICE_AUTO     = 0,
    VMAF_DNN_DEVICE_CPU      = 1,
    VMAF_DNN_DEVICE_CUDA     = 2,
    VMAF_DNN_DEVICE_OPENVINO = 3,  /* covers SYCL / oneAPI / Intel GPU */
    VMAF_DNN_DEVICE_ROCM     = 4,
} VmafDnnDevice;

typedef struct VmafDnnConfig {
    VmafDnnDevice device;
    int  device_index;  /* multi-GPU index; 0 for single-GPU/CPU */
    int  threads;       /* CPU EP intra-op threads; 0 = ORT default */
    bool fp16_io;       /* request fp16 tensors when supported */
} VmafDnnConfig;
```

- `device = AUTO` — today equivalent to `CPU`. The backend
  ([`ort_backend.c:85-98`](../../libvmaf/src/dnn/ort_backend.c)) only
  special-cases `VMAF_DNN_DEVICE_CUDA` (guarded by `ORT_API_HAS_CUDA`);
  every other value — `AUTO`, `CPU`, `OPENVINO`, `ROCM`, unknown — falls
  through to the default CPU EP. The "CUDA → OpenVINO → ROCm → CPU"
  preference order is **not** implemented yet; tracked as
  [issue #30](https://github.com/lusoris/vmaf/issues/30).
- `device = CUDA` — only functional when ORT itself was built with the CUDA
  EP (`ORT_API_HAS_CUDA`). If not, the call silently falls back to CPU and
  succeeds — there is no error.
- `device = OPENVINO` / `device = ROCM` — **accepted but ignored today**.
  The enum values exist so client code can stop using them when EPs are
  added without an API break, but right now they produce a CPU EP session
  (same issue #30).
- `fp16_io` — **currently a ghost field**. Declared in the header but not
  read anywhere in `libvmaf/src/dnn/`. Kept in the config struct so the ABI
  doesn't break when honored; file a follow-up if you need it wired up.
- `threads = 0` — lets ORT pick. Set explicitly when pinning affinity or
  benchmarking.

Pass `NULL` for `cfg` in any function that accepts one to use
`VMAF_DNN_DEVICE_AUTO` with zero device index, zero threads, no fp16.

## Attached mode — `vmaf_use_tiny_model`

```c
int vmaf_use_tiny_model(VmafContext *ctx,
                        const char *onnx_path,
                        const VmafDnnConfig *cfg);
```

Register a tiny ONNX model on a live `VmafContext`. The model participates in
the per-frame pipeline; its outputs appear in the report alongside SVM
scores. Use this when you want "VMAF + tiny AI score" in the same run.

Returns:

- `0` — success.
- `-ENOSYS` — built without DNN support.
- `-EINVAL` — bad args (null `ctx` or `onnx_path`).
- `-ENOENT` — `onnx_path` does not exist or is not a regular file.
- `-E2BIG` — file exceeds `VMAF_MAX_MODEL_BYTES` (default 50 MB — defence
  against adversarial bloat; see
  [ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md)).
- `-ENOMEM` — session allocation failed (ORT env, session options, or
  internal buffer allocation).
- Negative `errno` from the operator-allowlist walk if the model contains a
  banned op.

Equivalent CLI flag: `--tiny-model <path>`
([usage/cli.md](../usage/cli.md#tiny-ai-flags-fork-added)).

## Standalone sessions — `VmafDnnSession`

Standalone mode is for filter-style inference that does not need a
`VmafContext` — e.g. a learned de-banding preprocessor that mutates a luma
plane before downstream processing.

```c
typedef struct VmafDnnSession VmafDnnSession;

int  vmaf_dnn_session_open (VmafDnnSession **out, const char *onnx_path, const VmafDnnConfig *cfg);
void vmaf_dnn_session_close(VmafDnnSession *sess);
```

Both `*_session_open` and `vmaf_use_tiny_model` apply the same
size-cap + operator-allowlist walk. See
[ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md) for the allowlist
and [ADR-0041](../adr/0041-lpips-sq-extractor.md) for an example of an
extractor that uses a session under the hood.

### Luma-only convenience call

```c
int vmaf_dnn_session_run_luma8(VmafDnnSession *sess,
                               const uint8_t *in,  size_t in_stride,
                               int w, int h,
                               uint8_t *out,       size_t out_stride);
```

Runs one luma-in / luma-out pass. Only works when the ONNX graph has:

- exactly one float32 input of static shape `[1, 1, H, W]`,
- exactly one output of the same shape.

The implementation:

1. Reads `in` (uint8 luma), normalises to `[0, 1]` (applies mean/std from the
   sidecar JSON if present).
2. Runs ORT.
3. De-normalises, rounds, clamps to `[0, 255]`, writes `out`.

Errors:

- `-EINVAL` — `sess`, `in`, or `out` is NULL.
- `-ENOTSUP` — graph shape isn't the supported NCHW `[1,1,H,W]` luma layout,
  or ORT returned fewer output elements than `w*h`.
- `-ERANGE` — `w`/`h` don't match the graph's static input shape. Use
  `vmaf_dnn_session_run()` for dynamic shapes.
- `-ENOSYS` — libvmaf was built without ONNX Runtime support
  (`enable_onnx=false`); every DNN entry point returns `-ENOSYS` in that
  configuration.

### General named-binding call

For models with multiple inputs / outputs or non-luma shapes, use the
general call:

```c
typedef struct VmafDnnInput {
    const char    *name;   /* bind by graph name; NULL = positional */
    const float   *data;   /* row-major float32 */
    const int64_t *shape;  /* rank dims */
    size_t         rank;
} VmafDnnInput;

typedef struct VmafDnnOutput {
    const char *name;      /* bind by graph name; NULL = positional */
    float      *data;      /* caller-owned */
    size_t      capacity;  /* element count allocated */
    size_t      written;   /* OUT: element count produced */
} VmafDnnOutput;

int vmaf_dnn_session_run(VmafDnnSession *sess,
                         const VmafDnnInput *inputs,   size_t n_inputs,
                         VmafDnnOutput     *outputs,   size_t n_outputs);
```

Name-binding (`name != NULL`) resolves by the ONNX graph's declared input /
output names. Positional binding (`name == NULL`) uses the tensor's array
index. Mix is allowed but discouraged — pick one style per session.

Errors:

- `-ENOSYS` — built without DNN support.
- `-EINVAL` — mismatched arity, null pointers, rank zero.
- `-ENOMEM` — allocation failure (per-input staging buffer, or tensor
  creation).
- `-ENOSPC` — some `outputs[i].capacity` is smaller than the produced tensor.
  On this return, `outputs[i].written` is populated with the required
  element count (the code sets `written = produced` *before* the capacity
  check), so the caller can resize and retry with the same bindings.
- `-EIO` — ORT failure (bad graph, EP crash, OOM on device). The diagnostic
  is logged via the `VmafContext` log callback if one is configured (for
  sessions opened without a `VmafContext`, logging goes through the
  library's global log sink).

See [ADR-0040](../adr/0040-dnn-session-multi-input-api.md) for the rationale
behind multi-input/output + named binding.

## Thread-safety

- A single `VmafDnnSession` is **not** re-entrant. Driving inference from two
  threads requires either per-thread sessions or external locking.
- Opening multiple sessions concurrently is safe; they do not share state
  beyond process-global ORT singletons.
- Attaching a tiny model via `vmaf_use_tiny_model()` is subject to the same
  single-driver rule as the rest of the `VmafContext` API — see
  [index.md](index.md#thread-safety).

## Runnable example — standalone luma filter

```c
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <libvmaf/dnn.h>

int main(int argc, char **argv)
{
    if (!vmaf_dnn_available()) {
        fprintf(stderr, "libvmaf was built without DNN support\n");
        return 2;
    }

    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = { .device = VMAF_DNN_DEVICE_AUTO };

    int err = vmaf_dnn_session_open(&sess, argv[1], &cfg);
    if (err < 0) {
        fprintf(stderr, "open failed: %d (%s)\n", err, strerror(-err));
        return 1;
    }

    const int W = 1920, H = 1080;
    uint8_t *in  = malloc((size_t)W * H);
    uint8_t *out = malloc((size_t)W * H);
    /* ...fill `in` with luma from your pipeline... */

    err = vmaf_dnn_session_run_luma8(sess, in, W, W, H, out, W);
    if (err < 0) fprintf(stderr, "run failed: %d\n", err);

    free(in); free(out);
    vmaf_dnn_session_close(sess);
    return err < 0 ? 1 : 0;
}
```

Build:

```
cc filter.c -o filter $(pkg-config --cflags --libs libvmaf)
```

Only works when libvmaf was built with `-Denable_dnn=true`.

## Known limitations

- Operator allowlist covers the set required by tiny FR / NR / filter models
  shipped in `model/tiny/`; untrusted models with new op types will be
  rejected at `_open`. Extend the allowlist via the registry — see
  [ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md).
- `VMAF_DNN_DEVICE_OPENVINO` and `VMAF_DNN_DEVICE_ROCM` are **accepted but
  ignored** — the backend only wires up CPU (default) and CUDA (when
  `ORT_API_HAS_CUDA`). Adding the OpenVINO + ROCm EP append calls is
  tracked as [issue #30](https://github.com/lusoris/vmaf/issues/30).
- `VmafDnnConfig.fp16_io` is **accepted but ignored** — currently a ghost
  field. Same tracking issue.
- There is no callback / progress hook; inference is synchronous per call.
- Sessions are heap-only; no stack-allocated variant.

## Related

- [ADR-0022](../adr/0022-inference-runtime-onnx.md) — choice of ONNX Runtime.
- [ADR-0023](../adr/0023-tinyai-user-surfaces.md) — where the CLI / C API / ffmpeg / training surfaces intersect.
- [ADR-0036](../adr/0036-tinyai-wave1-scope-expansion.md) — Wave 1 scope
  (LPIPS, saliency, per-shot CRF, `vmaf_post`, allowlist `Loop`/`If`, MCP VLM).
- [ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md) — operator
  allowlist + model registry.
- [ADR-0040](../adr/0040-dnn-session-multi-input-api.md) — multi-input/output
  named-binding API.
- [ADR-0041](../adr/0041-lpips-sq-extractor.md) — LPIPS-SqueezeNet extractor
  (consumer of this API).
- [ADR-0042](../adr/0042-tinyai-docs-required-per-pr.md) — tiny-AI doc
  specialisation.
- [../ai/inference.md](../ai/inference.md) — CLI-side tiny-AI walkthrough.
